"""Model Select tab — pure rendering (no callbacks; those live in callbacks.py).

Read-only view of the Stage 5 decision in ``_stage5`` (= runs/{id}/model_selection.json):
who picked what (LLM vs rule engine), why, the training hints Stages 6/7 will consume,
the full eligibility table and the rule-by-rule trace. Selection is auto-chained after
forecast EDA by the Pipeline Setup confirm button; the only control here is a re-run
row with its own use-LLM toggle (selection is cheap — one JSON read + one LLM call).
"""
import json

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

_TEXT = "var(--bs-body-color)"

_SH = {"color": "#94a3b8", "fontSize": "0.78rem", "textTransform": "uppercase",
       "letterSpacing": "0.06em"}

_CONF_COLOR = {"high": "success", "medium": "info", "low": "warning"}
_CAT_COLOR = {"baseline": "secondary", "statistical": "primary",
              "intermittent": "warning", "ml": "success", "deep_learning": "dark"}
_CAT_LABEL = {"baseline": "Baseline", "statistical": "Statistical",
              "intermittent": "Intermittent demand", "ml": "Machine learning",
              "deep_learning": "Deep learning"}
_POLICY_LABEL = {"aggregate": "One aggregate series",
                 "per_series_local": "Per-series (one model each)",
                 "global": "Global (one model, all series)"}


def _cicon(name, **style):
    return html.I(className=f"bi {name}", style={"marginRight": "5px", **style})


def _tile(icon, label, value, sub=None, width=3):
    kids = [
        html.Div(
            [html.I(className=f"bi {icon}",
                    style={"color": "#64748b", "marginRight": "4px", "fontSize": "0.7rem"}),
             html.Span(label, style={"fontSize": "0.65rem", "color": "#94a3b8",
                                     "textTransform": "uppercase", "letterSpacing": "0.06em"})],
            className="d-flex align-items-center mb-1",
        ),
        html.Div(value, style={"fontSize": "1rem", "fontWeight": "700",
                               "letterSpacing": "-0.02em", "color": _TEXT}),
    ]
    if sub:
        kids.append(html.Div(sub, style={"fontSize": "0.68rem", "color": "#64748b"}))
    return dbc.Col(kids, width=width, className="mb-2")


def _datatable(rows, columns):
    # Mirrors fcst_eda_view._datatable (kept local — cross-view imports stay zero).
    return dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in columns],
        style_table={"overflowX": "auto", "borderRadius": "10px", "overflow": "hidden"},
        style_cell={
            "fontSize": "12px", "padding": "8px 12px", "textAlign": "left",
            "backgroundColor": "#161b2e", "color": "#e2e8f0", "border": "none",
            "borderBottom": "1px solid rgba(255,255,255,0.05)",
            "fontFamily": "Inter, sans-serif", "whiteSpace": "normal", "height": "auto",
        },
        style_header={
            "fontWeight": "600", "backgroundColor": "#1e2235", "color": "#64748b",
            "fontSize": "10px", "textTransform": "uppercase", "letterSpacing": "0.06em",
            "border": "none", "borderBottom": "1px solid rgba(255,255,255,0.10)",
            "fontFamily": "Inter, sans-serif",
        },
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#1a1f30"}],
        page_size=15,
    )


def _label_map(sel) -> dict:
    return {m["id"]: m["label"] for m in sel.get("eligible_models") or []}


# ── Decision hero ────────────────────────────────────────────────────────────

def _decision_hero(sel):
    labels = _label_map(sel)
    model_label = labels.get(sel["model"], sel["model"])
    cat = sel.get("category")
    rule = sel.get("rule") or {}
    if sel.get("source") == "llm":
        picked_by = f"decided by LLM ({(sel.get('llm') or {}).get('model') or '?'})"
    else:
        picked_by = f"decided by rule engine ({rule.get('rule_id', '?')})"

    runner = sel.get("runner_up")
    runner_line = []
    if runner:
        runner_line = [html.Div([
            html.Span("Runner-up: ", style={"color": "#64748b", "fontSize": "0.75rem"}),
            html.Span(labels.get(runner, runner),
                      style={"color": _TEXT, "fontSize": "0.78rem", "fontWeight": "600"}),
        ], className="mt-1")]

    return dbc.Card(dbc.CardBody([
        html.Div([
            html.Span([_cicon("bi-cpu"), "Selected Model"], style=_SH),
            html.Small(picked_by, style={"color": "#64748b", "fontSize": "0.68rem"}),
        ], className="d-flex justify-content-between align-items-center mb-2"),
        html.Div([
            html.Span(model_label,
                      style={"fontSize": "1.5rem", "fontWeight": "700",
                             "letterSpacing": "-0.02em", "color": _TEXT,
                             "marginRight": "12px"}),
            dbc.Badge(_CAT_LABEL.get(cat, cat), color=_CAT_COLOR.get(cat, "secondary"),
                      className="me-1", style={"verticalAlign": "middle"}),
            dbc.Badge(f"{sel.get('confidence', '?')} confidence",
                      color=_CONF_COLOR.get(sel.get("confidence"), "secondary"),
                      style={"verticalAlign": "middle"}),
        ], className="d-flex align-items-center flex-wrap mb-2"),
        html.P(sel.get("reason") or "", className="mb-0",
               style={"fontSize": "0.85rem", "color": "#94a3b8"}),
        *runner_line,
    ]), className="mb-3", style={"borderLeft": "3px solid #6366f1"})


# ── LLM provenance banner ────────────────────────────────────────────────────

def _provenance(sel):
    llm = sel.get("llm") or {}
    rule = sel.get("rule") or {}
    labels = _label_map(sel)

    if llm.get("error"):
        return [dbc.Alert(
            [_cicon("bi-exclamation-triangle"),
             f"LLM unavailable ({llm['error']}) — the rule-engine pick stands."],
            color="warning", className="py-2 mb-3")]
    if llm.get("model") is None:
        return [dbc.Alert(
            [_cicon("bi-sliders"),
             "LLM disabled for this selection — the deterministic rule engine decided."],
            color="secondary", className="py-2 mb-3")]

    bits = [f"LLM ({llm['model']}) reviewed the evidence"]
    if sel.get("model") != rule.get("model"):
        bits.append(f"and overrode the rule engine's pick "
                    f"({labels.get(rule.get('model'), rule.get('model'))} → "
                    f"{labels.get(sel['model'], sel['model'])})")
    else:
        bits.append("and agreed with the rule engine")
    overrides = llm.get("sanitizer_overrides") or []
    if overrides:
        bits.append(f"(sanitizer reverted: {', '.join(overrides)})")
    return [dbc.Alert([_cicon("bi-stars"), " ".join(bits) + "."],
                      color="info", className="py-2 mb-3")]


# ── Ranked top-5 ─────────────────────────────────────────────────────────────

def _ranking_card(sel):
    """The ranked top-5 (rule suitability scores, optionally re-ranked by the LLM) —
    the list the user picks training candidates from. Absent on pre-ranking runs."""
    ranking = sel.get("ranking") or []
    if not ranking:
        return None
    rows = []
    for e in ranking:
        cat = e.get("category")
        rows.append(html.Div([
            html.Span(f"#{e['rank']}",
                      style={"color": "#6366f1", "fontWeight": "700",
                             "fontSize": "0.85rem", "minWidth": "32px",
                             "display": "inline-block"}),
            html.Span(e.get("label") or e["model"],
                      style={"color": _TEXT, "fontWeight": "600", "fontSize": "0.88rem",
                             "marginRight": "8px"}),
            dbc.Badge(_CAT_LABEL.get(cat, cat), color=_CAT_COLOR.get(cat, "secondary"),
                      className="me-1", style={"fontSize": "0.6rem"}),
            dbc.Badge("LLM" if e.get("source") == "llm" else "rules",
                      color="info" if e.get("source") == "llm" else "secondary",
                      style={"fontSize": "0.6rem"}),
            html.Div(e.get("reason") or "", style={"fontSize": "0.75rem",
                                                   "color": "#94a3b8",
                                                   "marginLeft": "32px"}),
        ], className="mb-2"))
    return dbc.Card(dbc.CardBody([
        html.Div(html.Span([_cicon("bi-trophy"), "Ranked candidates (best first)"],
                           style=_SH), className="mb-2"),
        html.P("Ranked by expected accuracy on this series' statistics. The top two are "
               "pre-selected on the Training tab — train up to five and compare.",
               style={"fontSize": "0.75rem", "color": "#64748b"}, className="mb-2"),
        *rows,
    ]), className="mb-3")


# ── Training hints ───────────────────────────────────────────────────────────

def _hints_card(sel):
    hints = sel.get("training_hints") or {}
    derived = sel.get("derived") or {}
    periods = hints.get("seasonal_periods") or []
    exog = hints.get("exog_usable") or []

    exog_kids = []
    if exog:
        badges = []
        for e in exog:
            txt = f"{e['col']} @ lag {e['best_lag']} (corr {e['best_corr']})"
            if e.get("seasonal_echo"):
                txt += " ⚠ seasonal echo?"
            badges.append(dbc.Badge(txt, color="secondary", className="me-1 mb-1"))
        exog_kids = [html.Div([
            html.Div([html.I(className="bi bi-diagram-3",
                             style={"color": "#64748b", "marginRight": "4px",
                                    "fontSize": "0.7rem"}),
                      html.Span("Usable exogenous drivers (leading)",
                                style={"fontSize": "0.65rem", "color": "#94a3b8",
                                       "textTransform": "uppercase",
                                       "letterSpacing": "0.06em"})],
                     className="d-flex align-items-center mb-1"),
            html.Div(badges),
        ], className="mt-1")]
    if derived.get("vif_collinear"):
        exog_kids.append(html.Div(
            [_cicon("bi-exclamation-triangle", color="#d97706", fontSize="0.7rem"),
             html.Span("High VIF — the exogenous drivers are collinear; at most 1–2 add signal.",
                       style={"fontSize": "0.72rem", "color": "#d97706"})],
            className="d-flex align-items-center mt-1"))

    return dbc.Card(dbc.CardBody([
        html.Div(html.Span([_cicon("bi-wrench-adjustable"), "Training Hints"], style=_SH),
                 className="mb-2"),
        dbc.Row([
            _tile("bi-collection", "Policy",
                  _POLICY_LABEL.get(hints.get("policy"), hints.get("policy") or "—")),
            _tile("bi-arrow-repeat", "Seasonal periods",
                  " · ".join(str(p) for p in periods) if periods else "—"),
            _tile("bi-magic", "Transform", hints.get("transform") or "none"),
            _tile("bi-rulers", "Eval metric", hints.get("metric") or "—"),
        ]),
        *exog_kids,
    ]), className="mb-3")


# ── Eligibility table + rule trace ───────────────────────────────────────────

def _eligible_table(sel):
    rule = sel.get("rule") or {}
    rows = []
    for m in sel.get("eligible_models") or []:
        marker = ""
        if m["id"] == sel.get("model"):
            marker = "◀ selected"
        elif m["id"] == sel.get("runner_up"):
            marker = "runner-up"
        elif m["id"] == rule.get("model") and rule.get("model") != sel.get("model"):
            marker = "rule pick"
        rows.append({
            "Model": m["label"],
            "Category": _CAT_LABEL.get(m["category"], m["category"]),
            "Available": "yes" if m["available"] else "no",
            "Eligible": m["excluded_reason"] or "yes",
            "": marker,
        })
    return html.Div([
        html.Div(html.Span([_cicon("bi-list-check"), "Candidate Models"], style=_SH),
                 className="mb-2 mt-1"),
        _datatable(rows, ["Model", "Category", "Available", "Eligible", ""]),
    ], className="mb-3")


def _rule_trace(sel):
    rule = sel.get("rule") or {}
    llm = sel.get("llm") or {}
    rows = []
    for t in rule.get("trace") or []:
        mark = "✓" if t.get("fired") else "✗"
        color = "#059669" if t.get("fired") else "#64748b"
        rows.append(html.Div([
            html.Span(f"{mark} {t.get('rule')}",
                      style={"color": color, "fontWeight": "600", "fontSize": "0.78rem",
                             "minWidth": "42px", "display": "inline-block"}),
            html.Span(t.get("detail") or "", style={"color": "#94a3b8", "fontSize": "0.78rem"}),
        ], className="mb-1"))

    kids = [html.Div(rows, className="mt-2")]
    if llm.get("response"):
        kids.append(html.Div("Raw LLM response", className="mt-2 mb-1",
                             style={"fontSize": "0.7rem", "color": "#64748b",
                                    "textTransform": "uppercase", "letterSpacing": "0.06em"}))
        kids.append(html.Pre(json.dumps(llm["response"], indent=2),
                             style={"fontSize": "11px", "background": "#161b2e",
                                    "color": "#94a3b8", "padding": "10px",
                                    "borderRadius": "10px", "maxHeight": "260px",
                                    "overflowY": "auto"}))
    return html.Details([
        html.Summary("Rule-engine trace", style={"cursor": "pointer", "color": "#64748b",
                                                 "fontSize": "0.8rem"}),
        *kids,
    ], className="mb-3")


# ── Re-run row ───────────────────────────────────────────────────────────────

def _rerun_row(has_selection):
    return dbc.Card(dbc.CardBody(dbc.Row([
        dbc.Col(
            dbc.Switch(id="switch-model-llm", value=True,
                       label="Use LLM for model selection",
                       style={"fontSize": "0.82rem", "color": "#94a3b8"},
                       className="mt-2"),
            md=7),
        dbc.Col(
            dcc.Loading(
                dbc.Button(
                    [_cicon("bi-cpu"),
                     "Re-run Selection" if has_selection else "Run Selection"],
                    id="btn-model-rerun", color="primary", className="w-100",
                    style={"fontWeight": "600"},
                ),
                type="circle", color="#6366f1", delay_show=100,
                target_components={"cleaning-status": "children"},
            ),
            md=5),
    ])), className="mb-3")


# ── Tab entry point ──────────────────────────────────────────────────────────

def render_model_tab(data):
    if not data:
        return html.Div(
            [_cicon("bi-info-circle", fontSize="2rem", color="#334155",
                    display="block", marginBottom="12px", marginRight="0"),
             html.P("Load a dataset on the Data tab first.",
                    style={"color": "#64748b", "fontSize": "0.9rem"})],
            className="text-center mt-5 pt-4",
        )

    stage4 = data.get("_stage4") or {}
    eda_exists = bool((stage4.get("eda") or {}).get("payload"))
    sel = data.get("_stage5")

    if not eda_exists and not sel:
        return html.Div(
            [_cicon("bi-info-circle", fontSize="2rem", color="#334155",
                    display="block", marginBottom="12px", marginRight="0"),
             html.P("Run the pipeline on the Pipeline Setup tab first — model "
                    "selection needs the forecast-EDA evidence.",
                    style={"color": "#64748b", "fontSize": "0.9rem"})],
            className="text-center mt-5 pt-4",
        )

    header = html.Div([
        html.Span([_cicon("bi-cpu"), "Model Select"],
                  className="fw-semibold section-title"),
    ], className="mb-3")

    if not sel:
        return html.Div([
            header,
            dbc.Alert(
                [_cicon("bi-info-circle"),
                 "Forecast EDA is done but no model has been selected yet — run it below."],
                color="secondary", className="py-2 mb-3"),
            _rerun_row(has_selection=False),
        ])

    ranking_card = _ranking_card(sel)
    return html.Div([
        header,
        *_provenance(sel),
        _decision_hero(sel),
        *([ranking_card] if ranking_card else []),
        _hints_card(sel),
        _rerun_row(has_selection=True),
        _eligible_table(sel),
        _rule_trace(sel),
    ])
