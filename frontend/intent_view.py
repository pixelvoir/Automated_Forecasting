"""Pipeline Setup tab — the intent form (pure rendering; callbacks live in callbacks.py).

This is the pipeline's single user checkpoint: timestamp, target (+aggregation including
event counts), scope, series key, exogenous drivers, forecast frequency and horizon —
pre-filled from ``_intent.suggestions`` (rule detection + LLM refinement, Stage 2.5),
overridden by ``_intent.selections`` when the user already confirmed once. Confirming
runs cleaning → validation → forecast EDA in one chain.
"""
import dash_bootstrap_components as dbc
from dash import dcc, html

from frontend.ui import horizon_datepicker

_CONF_BADGE = {
    "high": ("auto", "success"),
    "medium": ("suggested", "info"),
    "low": ("confirm", "warning"),
}

_HIDDEN = {"display": "none"}


def _cicon(name, **style):
    return html.I(className=f"bi {name}", style={"marginRight": "5px", **style})


def _lbl(text, confidence=None):
    kids = [html.Span(text)]
    if confidence in _CONF_BADGE:
        word, color = _CONF_BADGE[confidence]
        kids.append(dbc.Badge(word, color=color, className="ms-2",
                              style={"fontSize": "0.6rem", "verticalAlign": "middle"}))
    return html.Label(kids, className="form-label",
                      style={"fontSize": "0.8rem", "color": "var(--ink-muted)"})


def llm_intent_banner(suggestions: dict):
    """How the pre-filled suggestions were produced: LLM (with its reasoning) or rules."""
    llm = (suggestions or {}).get("llm") or {}
    if llm.get("suggested_by") == "llm":
        return dbc.Alert(
            [_cicon("bi-stars", color="#10b981"),
             html.Strong("LLM-suggested setup"),
             html.Span(f" ({llm.get('model', '')})",
                       style={"fontSize": "0.75rem", "color": "#6ee7b7"}),
             html.Div(llm.get("rationale") or "",
                      style={"fontSize": "0.78rem", "marginTop": "2px"})],
            color="success", className="py-2 mb-3",
        )
    if llm.get("error"):
        return dbc.Alert(
            [_cicon("bi-cpu", color="#f59e0b"),
             html.Strong("Rule-based suggestions"),
             html.Span(" — LLM refinement failed: ", style={"fontSize": "0.8rem"}),
             html.Code(str(llm["error"])[:160], style={"fontSize": "0.72rem"})],
            color="warning", className="py-2 mb-3",
        )
    return dbc.Alert(
        [_cicon("bi-cpu"), "Rule-based suggestions — review before running."],
        color="secondary", className="py-2 mb-3",
    )


def build_intent_form(suggestions: dict, selections: dict | None, already_run: bool):
    """The confirm form. ``suggestions`` = _intent.suggestions (forecast_intent.json);
    ``selections`` = what the user last confirmed (wins over suggestions)."""
    selections = selections or {}
    ts_info = suggestions.get("timestamp", {})
    target_info = suggestions.get("target", {})
    group_info = suggestions.get("group", {})
    freq_info = suggestions.get("frequency", {})
    evidence = suggestions.get("evidence", {})

    ts_default = selections.get("timestamp_col") or ts_info.get("suggested")
    ts_options = [
        {"label": f"{c['col']}  ({c.get('frequency', '?')})"
                  + ("  — ETL/audit" if c.get("is_audit") else ""),
         "value": c["col"]}
        for c in ts_info.get("candidates", [])
    ]

    cands = target_info.get("candidates", [])
    cand_map = {c["col"]: c for c in cands}
    target_default = selections.get("target_col") or target_info.get("suggested")
    target_options = [
        {"label": (f"Count of {c['col']} per period" if c.get("kind") == "count"
                   else c["col"]),
         "value": c["col"]}
        for c in cands
    ]
    if target_default and target_default not in cand_map:
        target_options.append({"label": target_default, "value": target_default})

    agg_default = (selections.get("agg") or target_info.get("suggested_agg")
                   or cand_map.get(target_default, {}).get("agg_hint", "sum"))

    scope_default = selections.get("scope") or suggestions.get("scope", {}).get("suggested", "aggregate")
    group_default = selections.get("group_cols") or group_info.get("suggested", [])
    col_nunique = group_info.get("column_nunique", {})
    group_options = [{"label": f"{c}  ({n:,} values)", "value": c}
                     for c, n in sorted(col_nunique.items(), key=lambda kv: kv[1])]

    exog_cands = suggestions.get("exogenous", {}).get("candidates", [])
    exog_default = (selections.get("exog_cols") if selections.get("exog_cols") is not None
                    else [c for c in suggestions.get("exogenous", {}).get("suggested", exog_cands)
                          if c != target_default])

    freq_default = selections.get("forecast_frequency") or freq_info.get("suggested")
    freq_options = [{"label": f.capitalize(), "value": f} for f in freq_info.get("options", [])]
    horizon_default = selections.get("horizon") or suggestions.get("horizon", {}).get("suggested", 12)

    if evidence.get("event_log_mode"):
        note = ("Event-log data detected (each row is one event) — the natural forecast "
                "is events per period, i.e. a count of the ID column.")
    elif evidence.get("is_panel"):
        note = (f"Panel data detected: ~{evidence.get('rows_per_timestamp', 0):,.0f} rows "
                f"per timestamp. 'Overall' forecasts the aggregate; 'Per-series' also "
                "profiles each series.")
    else:
        note = "Single continuous series detected."

    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.P([_cicon("bi-sliders"), "Confirm Forecast Setup"],
                       className="fw-semibold mb-0 section-title"),
                dbc.Button([_cicon("bi-arrow-clockwise"), "Re-detect"],
                           id="btn-intent-redetect", color="link", size="sm",
                           className="p-0 text-decoration-none"),
            ], className="d-flex justify-content-between align-items-center mb-2"),
            html.P(note, style={"fontSize": "0.75rem", "color": "var(--ink-faint)"}, className="mb-3"),

            dbc.Row([
                dbc.Col([
                    _lbl("Timestamp column", ts_info.get("confidence")),
                    dcc.Dropdown(id="dropdown-intent-ts", options=ts_options,
                                 value=ts_default, clearable=False,
                                 style={"fontSize": "0.85rem"}),
                ], md=4),
                dbc.Col([
                    _lbl("Target (what to forecast)", target_info.get("confidence")),
                    dcc.Dropdown(id="dropdown-intent-target", options=target_options,
                                 value=target_default, clearable=False,
                                 style={"fontSize": "0.85rem"}),
                ], md=4),
                dbc.Col([
                    _lbl("Aggregation per period"),
                    dcc.Dropdown(
                        id="dropdown-intent-agg",
                        options=[
                            {"label": "Sum (totals, demand)", "value": "sum"},
                            {"label": "Mean (rates, percentages)", "value": "mean"},
                            {"label": "Count of distinct values (events)", "value": "nunique"},
                        ],
                        value=agg_default, clearable=False, style={"fontSize": "0.85rem"},
                    ),
                ], md=4),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    _lbl("Scope", suggestions.get("scope", {}).get("confidence")),
                    dcc.RadioItems(
                        id="radio-intent-scope",
                        options=[
                            {"label": "  Overall (aggregate)", "value": "aggregate"},
                            {"label": "  Per-series (panel)", "value": "per_series"},
                        ],
                        value=scope_default,
                        inputStyle={"marginRight": "6px", "accentColor": "#6366f1"},
                        labelStyle={"display": "block", "marginBottom": "4px",
                                    "cursor": "pointer", "color": "var(--ink-muted)",
                                    "fontSize": "0.82rem"},
                    ),
                ], md=4),
                dbc.Col([
                    html.Div(id="intent-group-wrap", children=[
                        _lbl("Series key (what identifies one series)",
                             group_info.get("confidence")),
                        dcc.Dropdown(id="dropdown-intent-group", options=group_options,
                                     value=group_default, multi=True,
                                     placeholder="Select grouping column(s)…",
                                     style={"fontSize": "0.85rem"}),
                        html.Small(
                            "Also used to clean each series within its own boundary.",
                            style={"color": "var(--ink-faint)", "fontSize": "0.7rem"}),
                    ]),
                ], md=8),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    _lbl("Exogenous drivers (optional)"),
                    dcc.Dropdown(id="dropdown-intent-exog",
                                 options=[{"label": c, "value": c} for c in exog_cands],
                                 value=exog_default, multi=True,
                                 placeholder="No exogenous columns",
                                 style={"fontSize": "0.85rem"}),
                ], md=6),
                dbc.Col([
                    _lbl("Forecast frequency", freq_info.get("confidence")),
                    dcc.Dropdown(id="dropdown-intent-freq", options=freq_options,
                                 value=freq_default, clearable=False,
                                 style={"fontSize": "0.85rem"}),
                ], md=3),
                dbc.Col([
                    _lbl("Horizon (periods)"),
                    dbc.Input(id="input-intent-horizon", type="number", min=1, step=1,
                              value=horizon_default, size="sm"),
                    *horizon_datepicker(suggestions.get("data_end"),
                                        "date-intent-horizon-end", "intent-horizon-note"),
                ], md=3),
            ], className="mb-2"),

            html.Small(
                [_cicon("bi-info-circle", fontSize="0.7rem"),
                 "Scope, frequency and horizon can be changed later on the Forecast EDA tab "
                 "without re-cleaning. The series key, target and aggregation affect cleaning, "
                 "so changing those re-runs the pipeline from here."],
                className="d-block mb-3", style={"color": "var(--ink-faint)", "fontSize": "0.72rem"}),

            dbc.Row([
                dbc.Col([
                    dbc.Switch(id="switch-use-llm", value=True,
                               label="Use LLM for pipeline (cleaning recipe + model selection)",
                               className="mt-1",
                               style={"fontSize": "0.82rem", "color": "var(--ink-muted)"}),
                ], md=6),
            ], className="mb-2"),

            dcc.Loading(
                dbc.Button(
                    [_cicon("bi-play-fill"),
                     "Re-Run Pipeline" if already_run else "Run Pipeline"],
                    id="btn-confirm-run", color="primary", className="w-100",
                    disabled=not (ts_default and target_default),
                    style={"fontWeight": "600", "letterSpacing": "0.02em"},
                ),
                type="circle", color="#6366f1", delay_show=100,
                target_components={"cleaning-status": "children"},
            ),
            html.Small("Runs the full pipeline: clean → validate → forecast EDA → model selection.",
                       className="d-block text-center mt-1",
                       style={"color": "var(--ink-faint)", "fontSize": "0.7rem"}),
        ]),
        className="mb-3",
        style={"background": "rgba(30, 41, 59, 0.7)",
               "border": "1px solid rgba(99, 102, 241, 0.3)"},
    )


def detect_prompt_card():
    """Shown when suggestions haven't been produced yet (auto-detect failed or old run)."""
    return dbc.Card(dbc.CardBody([
        html.P([_cicon("bi-sliders"), "Detect Forecast Setup"],
               className="fw-semibold mb-2 section-title"),
        html.P("Scans the data and suggests the forecast target, timestamp, series "
               "grouping and horizon (LLM-assisted) — you confirm everything before "
               "cleaning runs.",
               style={"fontSize": "0.8rem", "color": "var(--ink-muted)"}),
        dcc.Loading(
            dbc.Button([_cicon("bi-search"), "Detect Forecast Setup"],
                       id="btn-intent-redetect", color="primary",
                       style={"fontWeight": "600"}),
            type="circle", color="#6366f1", delay_show=100,
            target_components={"cleaning-status": "children"},
        ),
    ]), className="mb-3",
        style={"background": "rgba(30, 41, 59, 0.7)",
               "border": "1px solid rgba(99, 102, 241, 0.3)"})


