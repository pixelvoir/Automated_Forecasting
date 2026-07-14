"""Training tab — pure rendering (callbacks live in callbacks.py).

The Stage 5 pick is a *default*, not a mandate: the user chooses one or more ELIGIBLE
models to actually train (seasonal-naive baseline is always added for the MASE floor). A
live cost-estimate banner tiers the request fast/moderate/heavy from
``policy × n_series × models``; a "heavy" per-series panel needs an explicit confirm before
the (job-managed, cancellable) training launches.

Reads ``_stage5`` (recommendation + eligibility), ``_stage4`` (history for the chart) and,
after a run, ``_stage7`` (the training report) + ``_stage6`` (the feature recipe).

Chart palette matches the Forecast EDA tab: history #6366f1, forecast #059669, backtest
prediction #d97706, interval band a muted emerald.
"""
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dash_table, dcc, html

_TEXT = "var(--bs-body-color)"
_CHART_BG = "#161b2e"
_GRID = "rgba(255,255,255,0.06)"
_INK = "#94a3b8"
_C_HIST = "#6366f1"
_C_FC = "#059669"
_C_BT = "#d97706"

_SH = {"color": "var(--ink-muted)", "fontSize": "0.78rem", "textTransform": "uppercase",
       "letterSpacing": "0.06em"}
_CAT_LABEL = {"baseline": "Baseline", "statistical": "Statistical",
              "intermittent": "Intermittent", "ml": "Machine learning",
              "deep_learning": "Deep learning"}
_POLICY_LABEL = {"aggregate": "One aggregate series",
                 "per_series_local": "Per-series (one model each)",
                 "global": "Global (one model, all series)"}
_TIER = {"fast": ("success", "bi-lightning-charge", "Fast"),
         "moderate": ("info", "bi-hourglass-split", "Moderate"),
         "heavy": ("warning", "bi-exclamation-triangle", "Heavy")}
_MAX_MODELS = 5


def _cicon(name, **style):
    return html.I(className=f"bi {name}", style={"marginRight": "5px", **style})


def _fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.{nd}f}" if abs(v) < 1e6 else f"{v:,.1f}"
    return str(v)


# ── Cost estimate banner ───────────────────────────────────────────────────────
# The numbers come from GET /runs/{id}/train-estimate (trainer.estimate_for_run — the
# same authoritative math as the /train heavy gate, wall-clock seconds calibrated from
# past runs on this machine). This module only renders; callbacks.py does the fetch.

_ROUTE_LABEL = {"global": "global — one pass over all series",
                "per_series": "per series",
                "single": "single series"}


def _fmt_duration(seconds) -> str:
    s = float(seconds or 0)
    if s < 60:
        return f"{max(int(round(s)), 1)}s"
    if s < 3600:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


def default_selection(data) -> list:
    """The model list the picker pre-selects — exported so callbacks.py can fetch the
    initial estimate for exactly what the user will see selected."""
    sel = (data or {}).get("_stage5") or {}
    option_ids = [o["value"] for o in _eligible_options(sel)]
    return _default_models(sel, option_ids)


def _hidden_confirm():
    # keep the id in the tree so btn-train's State always resolves
    return html.Div(dbc.Switch(id="switch-train-confirm", value=False),
                    style={"display": "none"})


def build_estimate_banner(data, selected, profile=None, est=None):
    """The live cost banner + (when heavy) the confirm switch. `est` is the server
    estimate fetched by callbacks.py; None means the API couldn't be reached."""
    if est is None:
        return dbc.Alert([
            html.Div([_cicon("bi-question-circle"),
                      html.Strong("Cost estimate unavailable"),
                      html.Span("  (could not reach the API — is run_dev.bat running?)",
                                style={"fontSize": "0.8rem", "marginLeft": "4px"})],
                     className="d-flex align-items-center"),
            _hidden_confirm(),
        ], color="secondary", className="py-2 mb-3")

    color, icon, label = _TIER.get(est.get("tier"), _TIER["fast"])
    detail = (f"{est.get('n_series', 1):,} series · {est.get('n_models', 0)} model(s) · "
              f"{_POLICY_LABEL.get(est.get('policy'), est.get('policy'))} · "
              f"profile {est.get('profile')}")
    kids = [html.Div([
        _cicon(icon),
        html.Strong(f"Estimated training time: ~{_fmt_duration(est.get('est_seconds'))} "),
        html.Span(f"(range {_fmt_duration(est.get('est_low_seconds'))}–"
                  f"{_fmt_duration(est.get('est_high_seconds'))} · {label})",
                  style={"fontSize": "0.8rem", "marginLeft": "4px"}),
    ], className="d-flex align-items-center"),
        html.Div(detail, style={"fontSize": "0.74rem", "marginTop": "4px", "opacity": 0.85})]

    rows = []
    for p in est.get("per_model") or []:
        name = p["model"] + (" (baseline, always trained)" if p.get("is_baseline") else "")
        cal = "" if p.get("calibrated") else " — first-run guess"
        rows.append(html.Div(
            f"• {name}: ~{_fmt_duration(p.get('seconds'))} "
            f"({_ROUTE_LABEL.get(p.get('route'), p.get('route'))}{cal})",
            style={"fontSize": "0.74rem", "marginTop": "2px"}))
    if rows:
        kids.append(html.Div(rows, className="mt-1"))

    if est.get("recommendation"):
        kids.append(html.Div([_cicon("bi-lightbulb", fontSize="0.72rem"),
                              est["recommendation"]],
                             style={"fontSize": "0.76rem", "marginTop": "6px"}))

    if est.get("tier") == "heavy":
        kids.append(dbc.Switch(id="switch-train-confirm", value=False,
                               label="Confirm — train anyway",
                               style={"fontSize": "0.82rem", "marginTop": "6px"}))
    else:
        kids.append(_hidden_confirm())
    return dbc.Alert(kids, color=color, className="py-2 mb-3")


# ── Model picker ──────────────────────────────────────────────────────────────

def _eligible_options(sel):
    opts = []
    for m in sel.get("eligible_models") or []:
        if m.get("available", True) and not m.get("excluded_reason"):
            opts.append({"label": f"{m['label']} ({_CAT_LABEL.get(m['category'], m['category'])})",
                         "value": m["id"]})
    return opts


def _default_models(sel, option_ids):
    # Top-2 of the Stage 5 ranking; pre-ranking runs fall back to pick + runner-up.
    picks = [e.get("model") for e in (sel.get("ranking") or [])[:2]] \
        or [sel.get("model"), sel.get("runner_up")]
    return [m for m in dict.fromkeys(picks) if m and m in option_ids] or option_ids[:1]


def _excluded_note(sel):
    """Why some catalog models aren't offered (e.g. NHITS needs 200 points — a short
    aggregate can't select it, by design). Content-level guidance, not locking."""
    excluded = [(m.get("label", m["id"]), m.get("excluded_reason"))
                for m in sel.get("eligible_models") or [] if m.get("excluded_reason")]
    if not excluded:
        return None
    return html.Div(
        [html.Div(f"· {label}: {reason}", style={"fontSize": "0.72rem", "color": "var(--ink-faint)"})
         for label, reason in excluded],
        className="mb-2",
    )


_PROFILE_OPTS = [
    {"label": "Fast — smallest budgets, no tuning refinement", "value": "fast"},
    {"label": "Balanced — 30 Optuna trials (default)", "value": "balanced"},
    {"label": "Max — 100 trials + global-panel tuning (1-3h class)", "value": "max"},
]


def _picker_card(data, sel, default_profile="balanced", initial_estimate=None):
    options = _eligible_options(sel)
    option_ids = [o["value"] for o in options]
    default = _default_models(sel, option_ids)
    excluded_note = _excluded_note(sel)
    return dbc.Card(dbc.CardBody([
        html.Div(html.Span([_cicon("bi-ui-checks"), "Models to train"], style=_SH),
                 className="mb-2"),
        html.P([f"Stage 5 ranks ", html.Strong(_label_of(sel, sel.get("model"))),
                " first — the top two are pre-selected. Pick up to ",
                html.Strong(str(_MAX_MODELS)),
                " models. Your selection is final: the best of YOUR models is crowned; "
                "the always-trained seasonal-naive baseline is only the MASE reference."],
               style={"fontSize": "0.82rem", "color": "var(--ink-muted)"}),
        dcc.Dropdown(id="dropdown-train-models", multi=True, options=options, value=default,
                     placeholder="Choose one or more models…", className="mb-3",
                     style={"fontSize": "0.85rem"}),
        *([excluded_note] if excluded_note is not None else []),
        html.Div([
            html.Span("Accuracy profile", style={"fontSize": "0.74rem", "color": "var(--ink-faint)",
                                                 "textTransform": "uppercase",
                                                 "letterSpacing": "0.05em"}),
            dcc.Dropdown(id="dropdown-train-profile", options=_PROFILE_OPTS,
                         value=default_profile, clearable=False,
                         style={"fontSize": "0.82rem"}),
        ], className="mb-3"),
        html.Div(id="train-estimate-banner",
                 children=build_estimate_banner(data, default, default_profile,
                                                est=initial_estimate)),
        dcc.Loading(
            dbc.Button([_cicon("bi-play-fill"), "Train Models"], id="btn-train",
                       color="primary", className="w-100", style={"fontWeight": "600"}),
            type="circle", color="#6366f1", delay_show=100,
            target_components={"cleaning-status": "children"}),
    ]), className="mb-3")


def _label_of(sel, mid):
    for m in sel.get("eligible_models") or []:
        if m["id"] == mid:
            return m["label"]
    return mid or "—"


# ── Results: leaderboard + chart + details ────────────────────────────────────

def _leaderboard(report):
    rows = []
    for e in report.get("results") or []:
        m = e.get("metrics") or {}
        notes = []
        if e.get("is_best"):
            notes.append("◀ best")
        if e.get("is_primary"):
            notes.append("your pick")
        if e.get("is_baseline"):
            notes.append("baseline (reference)")
        if e.get("error"):
            notes = ["error: " + e["error"][:40]]
        rows.append({
            "Model": e.get("label", e["model"]),
            "MASE": _fmt(m.get("mase")), "RMSE": _fmt(m.get("rmse"), 1),
            "sMAPE": _fmt(m.get("smape")), "MAPE": _fmt(m.get("mape")),
            "R²": _fmt(m.get("r2")), "n": m.get("n") or 0,
            "Time": (f"{e['train_seconds']:.1f}s" if e.get("train_seconds") is not None
                     else "—"),
            "Notes": " · ".join(notes),
        })
    cols = ["Model", "MASE", "RMSE", "sMAPE", "MAPE", "R²", "n", "Time", "Notes"]
    return dash_table.DataTable(
        data=rows, columns=[{"name": c, "id": c} for c in cols],
        style_table={"overflowX": "auto", "borderRadius": "10px", "overflow": "hidden"},
        style_cell={"fontSize": "12px", "padding": "8px 12px", "textAlign": "left",
                    "backgroundColor": "#161b2e", "color": "#e2e8f0", "border": "none",
                    "borderBottom": "1px solid rgba(255,255,255,0.05)",
                    "fontFamily": "Inter, sans-serif", "whiteSpace": "normal", "height": "auto"},
        style_header={"fontWeight": "600", "backgroundColor": "#1e2235", "color": "var(--ink-faint)",
                      "fontSize": "10px", "textTransform": "uppercase", "letterSpacing": "0.06em",
                      "border": "none", "borderBottom": "1px solid rgba(255,255,255,0.10)"},
        style_data_conditional=[
            {"if": {"filter_query": "{Notes} contains 'best'"},
             "backgroundColor": "rgba(5,150,105,0.14)"},
            {"if": {"filter_query": "{Notes} contains 'baseline'"}, "color": "var(--ink-muted)"},
        ],
        page_size=12)


# Interactive but unobtrusive: toolbar on hover only, wheel-zoom on, double-click resets.
_GRAPH_CONFIG = {
    "displayModeBar": "hover",
    "scrollZoom": True,
    "doubleClick": "reset",
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}


def _style_fig(fig, height=320):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=_CHART_BG,
        font=dict(family="Inter, sans-serif", size=11, color=_INK),
        margin=dict(l=55, r=15, t=20, b=30), height=height, hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10)),
        dragmode="zoom",
        modebar=dict(bgcolor="rgba(0,0,0,0)", color="#64748b",
                     activecolor="#e2e8f0", orientation="v"),
        hoverlabel=dict(bgcolor="#1e2235", font=dict(color="#e2e8f0", size=11)))
    fig.update_xaxes(gridcolor=_GRID, zeroline=False, showline=False)
    fig.update_yaxes(gridcolor=_GRID, zeroline=False, showline=False)
    return fig


def forecast_figure(entry, history):
    """History + backtest + forecast chart for one model entry (or one series' arrays
    from /series-forecast). Public: the update_train_chart callback reuses it."""
    fig = go.Figure()
    bt = entry.get("backtest") or {}
    if history and history.get("x"):
        fig.add_scatter(x=history["x"], y=history["y"], mode="lines", name="History",
                        line=dict(width=1.8, color=_C_HIST))
    elif bt.get("actual"):
        # single-series view: no downsampled history in the store — the backtest
        # window's actuals are the history proxy
        fig.add_scatter(x=bt["ds"], y=bt["actual"], mode="lines", name="Actual",
                        line=dict(width=1.8, color=_C_HIST))
    if bt.get("ds"):
        fig.add_scatter(x=bt["ds"], y=bt["predicted"], mode="lines", name="Backtest fit",
                        line=dict(width=1.6, color=_C_BT, dash="dot"))
    fc = entry.get("forecast") or {}
    if fc.get("ds"):
        if fc.get("lo95") and fc.get("hi95"):
            fig.add_scatter(x=list(fc["ds"]) + list(fc["ds"])[::-1],
                            y=list(fc["hi95"]) + list(fc["lo95"])[::-1],
                            fill="toself", fillcolor="rgba(5,150,105,0.15)",
                            line=dict(width=0), name="95% interval", hoverinfo="skip")
        fig.add_scatter(x=fc["ds"], y=fc["yhat"], mode="lines+markers", name="Forecast",
                        line=dict(width=2.2, color=_C_FC), marker=dict(size=4))
    fig = _style_fig(fig, height=380)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.07, bgcolor="#1e2235",
                                      bordercolor="rgba(255,255,255,0.08)", borderwidth=1))
    return fig


def _details(report, data):
    stage6 = data.get("_stage6") or {}
    recipe = stage6.get("recipe") or {}
    rat = recipe.get("rationale") or {}
    feat_kids = [html.Div([html.Strong(f"{k}: "), html.Span(str(v))],
                          style={"fontSize": "0.76rem", "color": "var(--ink-muted)", "marginBottom": "3px"})
                 for k, v in rat.items()]
    n_feat = stage6.get("n_features", 0)
    feat_kids.append(html.Div(
        f"{n_feat} features materialized" if n_feat else
        "Feature matrix is materialized only when an ML model (LightGBM/XGBoost) is trained.",
        style={"fontSize": "0.72rem", "color": "var(--ink-faint)", "marginTop": "4px"}))

    params_rows = []
    for e in report.get("results") or []:
        if e.get("error") or not e.get("params"):
            continue
        p = ", ".join(f"{k}={v}" for k, v in list(e["params"].items())[:6])
        tuning = e.get("tuning")
        if tuning and tuning.get("trials"):
            p += f"  ·  optuna: {tuning['trials']} trials, best_rmse={tuning.get('best_rmse')}"
        params_rows.append(html.Div([html.Strong(e.get("label", e["model"]) + ": "),
                                     html.Code(p, style={"fontSize": "0.72rem", "color": "#cbd5e1"})],
                                    style={"marginBottom": "4px"}))

    cv = report.get("cv") or {}
    meta = (f"CV: {cv.get('strategy')} · {cv.get('n_windows')} window(s) × h={cv.get('h')}  |  "
            f"season={report.get('season_length')}  |  transform={report.get('transform')}  |  "
            f"trained {report.get('series_trained')}/{report.get('n_series')} series")
    if report.get("total_seconds") is not None:
        meta += (f"  |  total {report['total_seconds']}s "
                 f"({report.get('accuracy_profile') or 'balanced'} profile)")
    return html.Details([
        html.Summary("Feature recipe, hyperparameters & CV detail",
                     style={"cursor": "pointer", "color": "var(--ink-faint)", "fontSize": "0.8rem"}),
        html.Div(meta, style={"fontSize": "0.74rem", "color": "var(--ink-faint)", "margin": "8px 0"}),
        html.Div(html.Span([_cicon("bi-diagram-3"), "EDA-seeded features"], style=_SH),
                 className="mt-2 mb-1"),
        html.Div(feat_kids),
        html.Div(html.Span([_cicon("bi-sliders"), "Model parameters"], style=_SH),
                 className="mt-3 mb-1"),
        html.Div(params_rows or [html.Span("—", style={"color": "var(--ink-faint)"})]),
    ], className="mb-2")


def history_for(data, report):
    """Aggregate history arrays from Stage 4 (aggregate scope only — a panel's summed
    history isn't stored downsampled per series)."""
    if report.get("scope") == "per_series":
        return None
    return (((data.get("_stage4") or {}).get("eda") or {}).get("full") or {}) \
        .get("aggregate", {}).get("plot_data", {}).get("series")


def _tile(icon, label, value, sub=None, width=3):
    kids = [
        html.Div([html.I(className=f"bi {icon}",
                         style={"color": "var(--ink-faint)", "marginRight": "4px",
                                "fontSize": "0.7rem"}),
                  html.Span(label, style={"fontSize": "0.65rem", "color": "var(--ink-muted)",
                                          "textTransform": "uppercase",
                                          "letterSpacing": "0.06em"})],
                 className="d-flex align-items-center mb-1"),
        html.Div(value, style={"fontSize": "1rem", "fontWeight": "700",
                               "letterSpacing": "-0.02em", "color": _TEXT}),
    ]
    if sub:
        kids.append(html.Div(sub, style={"fontSize": "0.68rem", "color": "var(--ink-faint)"}))
    return dbc.Col(kids, width=width, className="mb-2")


def _insights_strip(entry):
    """Target-level takeaways for one model's forecast (trainer's target_insights).
    Returns an empty div for old reports / error entries — never breaks a reload."""
    ins = (entry or {}).get("target_insights")
    if not ins:
        return html.Div()
    pct = ins.get("pct_change")
    pct_str = (f"{pct:+.1f}%" if pct is not None else "—")
    pct_color = "#059669" if (pct or 0) >= 0 else "#e66767"
    next_ds = str(ins.get("next_ds") or "")[:10]
    tiles = dbc.Row([
        _tile("bi-calendar-event", "Next period",
              _fmt(ins.get("next_value"), 2), sub=next_ds),
        _tile("bi-plus-slash-minus", "Horizon total",
              _fmt(ins.get("horizon_total"), 2), sub="sum of all forecast periods"),
        _tile("bi-graph-up-arrow", "vs trailing history",
              html.Span(pct_str, style={"color": pct_color}),
              sub=(f"history total {_fmt(ins.get('trailing_total'), 2)}"
                   if ins.get("trailing_total") is not None
                   else "history shorter than horizon")),
        _tile("bi-arrows-expand", "Widest 95% band",
              _fmt(ins.get("widest_interval_95"), 2),
              sub="max hi95 − lo95 across the horizon"),
    ])
    return dbc.Card(dbc.CardBody(tiles), className="mb-3",
                    style={"borderLeft": "3px solid #d97706"})


def results_block(data, report):
    """The post-training results (hero card, insights strip, model/series-switchable
    chart, leaderboard, details). Rendered on the RESULTS section — exported for
    render_results_body in callbacks.py."""
    return _results_block(data, report)


def _results_block(data, report):
    results = report.get("results") or []
    trained = [e for e in results if not e.get("error")]
    hero = next((e for e in results if e.get("is_best")), trained[0] if trained else None)

    kids = []
    if hero:
        m = hero.get("metrics") or {}
        kids.append(dbc.Card(dbc.CardBody([
            html.Div(html.Span([_cicon("bi-trophy"), "Best of your selected models"],
                               style=_SH), className="mb-1"),
            html.Div([
                html.Span(hero.get("label", hero["model"]),
                          style={"fontSize": "1.4rem", "fontWeight": "700", "color": _TEXT,
                                 "marginRight": "12px"}),
                dbc.Badge(f"MASE {_fmt(m.get('mase'))}", color="success", className="me-1"),
                dbc.Badge(f"sMAPE {_fmt(m.get('smape'))}%", color="secondary"),
            ], className="d-flex align-items-center flex-wrap"),
            html.P("MASE < 1 means the model beats the seasonal-naive reference. The crown "
                   "goes to the best of the models you selected — the baseline can rank "
                   "higher in the table without winning.",
                   className="mb-0 mt-1", style={"fontSize": "0.78rem", "color": "var(--ink-faint)"}),
        ]), className="mb-3", style={"borderLeft": "3px solid #059669"}))

        # Target insights follow the chart-model dropdown (update_train_chart re-renders
        # this div's children) — initialized from the hero entry.
        kids.append(html.Div(id="train-insights-strip", children=_insights_strip(hero)))

        history = history_for(data, report)

        # Chart card: EVERY trained model is viewable (dropdown), not just the best —
        # and for per-series runs a series picker charts any single series via
        # /runs/{id}/series-forecast (update_train_chart callback).
        model_options = [{"label": e.get("label", e["model"]), "value": e["model"]}
                         for e in trained]
        series_ids = report.get("series_ids") or []
        series_row_style = {} if (report.get("has_series_forecasts") and series_ids) \
            else {"display": "none"}
        kids.append(dbc.Card(dbc.CardBody([
            html.Div(html.Span([_cicon("bi-graph-up-arrow"),
                                "History, backtest & forecast"], style=_SH),
                     className="mb-2"),
            dbc.Row([
                dbc.Col([
                    html.Div("Model", style={"fontSize": "0.68rem", "color": "var(--ink-muted)",
                                             "textTransform": "uppercase"}),
                    dcc.Dropdown(id="dropdown-train-chart-model", options=model_options,
                                 value=hero["model"], clearable=False,
                                 style={"fontSize": "0.82rem"}),
                ], md=5),
                dbc.Col([
                    html.Div("Series (per-series runs)",
                             style={"fontSize": "0.68rem", "color": "var(--ink-muted)",
                                    "textTransform": "uppercase"}),
                    dcc.Dropdown(id="dropdown-train-chart-series",
                                 options=[{"label": s, "value": s} for s in series_ids],
                                 value=None, placeholder="All series (sum)",
                                 style={"fontSize": "0.82rem"}),
                ], md=7, style=series_row_style),
            ], className="mb-2"),
            dcc.Loading(
                dcc.Graph(id="graph-train-chart",
                          figure=forecast_figure(hero, history),
                          config=_GRAPH_CONFIG,
                          style={"borderRadius": "10px", "overflow": "hidden"}),
                type="circle", color="#6366f1", delay_show=150),
        ]), className="mb-3"))

    kids.append(dbc.Card(dbc.CardBody([
        html.Div(html.Span([_cicon("bi-list-ol"), "Leaderboard (ranked by MASE)"], style=_SH),
                 className="mb-2"),
        _leaderboard(report),
    ]), className="mb-3"))
    kids.append(_details(report, data))
    return kids


# ── Tab entry point ───────────────────────────────────────────────────────────

def render_training_controls(data, default_profile="balanced", initial_estimate=None):
    """The training CONTROLS only (model picker + estimate banner + Train button) —
    lives inside the Pipeline section's Train card. The results (leaderboard, chart,
    hero) render on the Results section via results_block()."""
    if not data:
        return html.P("Load a dataset on the Setup section first.",
                      style={"color": "var(--ink-faint)", "fontSize": "0.85rem"}, className="mb-0")

    sel = data.get("_stage5")
    if not sel:
        return dbc.Alert([_cicon("bi-info-circle"),
                          "Run model selection first (confirm the forecast setup on Setup, "
                          "or re-run it from the Model Selection card above)."],
                         color="secondary", className="py-2 mb-0")

    report = data.get("_stage7")
    blocks = [_picker_card(data, sel, default_profile, initial_estimate)]
    if report:
        blocks.append(dbc.Alert(
            [_cicon("bi-check-circle"),
             f"Trained {len([e for e in report.get('results') or [] if not e.get('error')])} "
             "model(s) — see the Results section for the leaderboard, charts and report."],
            color="success", className="py-2 mb-0"))
    else:
        blocks.append(dbc.Alert(
            [_cicon("bi-arrow-up-circle"),
             "Choose your models above and click Train — walk-forward backtest, a "
             "seasonal-naive baseline and a horizon forecast will appear on Results."],
            color="secondary", className="py-2 mb-0"))
    return html.Div(blocks)
