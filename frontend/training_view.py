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

_SH = {"color": "#94a3b8", "fontSize": "0.78rem", "textTransform": "uppercase",
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
_ML_IDS = {"lightgbm", "xgboost"}
_MAX_MODELS = 4


def _cicon(name, **style):
    return html.I(className=f"bi {name}", style={"marginRight": "5px", **style})


def _fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.{nd}f}" if abs(v) < 1e6 else f"{v:,.1f}"
    return str(v)


# ── Cost estimate (client-side mirror of trainer.estimate_cost) ───────────────

def estimate_cost(n_series: int, policy: str, selected: list) -> dict:
    """Advisory only — the /train route enforces the authoritative gate. Kept in sync with
    pipeline/trainer.estimate_cost and the settings.yaml tier thresholds."""
    units = 0.0
    for m in selected or []:
        if m in _ML_IDS:
            units += 1.5 if policy == "global" else (n_series * 1.5 if policy == "per_series_local" else 1.5)
        else:
            units += n_series if policy == "per_series_local" else 1
    tier = "heavy" if units > 2000 else ("moderate" if units > 200 else "fast")
    return {"tier": tier, "units": round(units, 1), "n_series": int(n_series), "policy": policy}


def _panel_n_series(data) -> int:
    payload = ((data.get("_stage4") or {}).get("eda") or {}).get("payload") or {}
    return int((payload.get("panel") or {}).get("n_series") or 1)


def build_estimate_banner(data, selected):
    """The live cost banner + (when heavy) the confirm switch. Rendered on load and by the
    update_train_estimate callback when the model selection changes."""
    hints = (data.get("_stage5") or {}).get("training_hints") or {}
    policy = hints.get("policy") or "aggregate"
    n_series = _panel_n_series(data)
    selected = [m for m in (selected or []) if m != "seasonal_naive"]
    est = estimate_cost(n_series, policy, selected)
    color, icon, label = _TIER[est["tier"]]

    detail = (f"{n_series:,} series × {len(selected)} model(s) · "
              f"{_POLICY_LABEL.get(policy, policy)}")
    kids = [html.Div([
        _cicon(icon), html.Strong(f"Estimated cost: {label}"),
        html.Span(f"  ({detail})", style={"fontSize": "0.8rem", "marginLeft": "4px"}),
    ], className="d-flex align-items-center")]

    if est["tier"] == "heavy":
        kids.append(html.Div(
            "Training this many per-series models can take a while. Prefer the global "
            "LightGBM policy or fewer series — or confirm to run it anyway.",
            style={"fontSize": "0.78rem", "marginTop": "6px"}))
        kids.append(dbc.Switch(id="switch-train-confirm", value=False,
                               label="Confirm — train anyway",
                               style={"fontSize": "0.82rem", "marginTop": "6px"}))
    else:
        # keep the id in the tree so btn-train's State always resolves
        kids.append(html.Div(dbc.Switch(id="switch-train-confirm", value=False),
                             style={"display": "none"}))
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
    picks = [sel.get("model"), sel.get("runner_up")]
    return [m for m in dict.fromkeys(picks) if m and m in option_ids] or option_ids[:1]


def _picker_card(data, sel):
    options = _eligible_options(sel)
    option_ids = [o["value"] for o in options]
    default = _default_models(sel, option_ids)
    return dbc.Card(dbc.CardBody([
        html.Div(html.Span([_cicon("bi-ui-checks"), "Models to train"], style=_SH),
                 className="mb-2"),
        html.P([f"Stage 5 recommends ", html.Strong(_label_of(sel, sel.get("model"))),
                ". Train it, or override — pick up to ", html.Strong(str(_MAX_MODELS)),
                " models. A seasonal-naive baseline is always trained for comparison."],
               style={"fontSize": "0.82rem", "color": "#94a3b8"}),
        dcc.Dropdown(id="dropdown-train-models", multi=True, options=options, value=default,
                     placeholder="Choose one or more models…", className="mb-3",
                     style={"fontSize": "0.85rem"}),
        html.Div(id="train-estimate-banner", children=build_estimate_banner(data, default)),
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
            notes.append("baseline")
        if e.get("error"):
            notes = ["error: " + e["error"][:40]]
        rows.append({
            "Model": e.get("label", e["model"]),
            "MASE": _fmt(m.get("mase")), "RMSE": _fmt(m.get("rmse"), 1),
            "sMAPE": _fmt(m.get("smape")), "MAPE": _fmt(m.get("mape")),
            "R²": _fmt(m.get("r2")), "n": m.get("n") or 0,
            "Notes": " · ".join(notes),
        })
    cols = ["Model", "MASE", "RMSE", "sMAPE", "MAPE", "R²", "n", "Notes"]
    return dash_table.DataTable(
        data=rows, columns=[{"name": c, "id": c} for c in cols],
        style_table={"overflowX": "auto", "borderRadius": "10px", "overflow": "hidden"},
        style_cell={"fontSize": "12px", "padding": "8px 12px", "textAlign": "left",
                    "backgroundColor": "#161b2e", "color": "#e2e8f0", "border": "none",
                    "borderBottom": "1px solid rgba(255,255,255,0.05)",
                    "fontFamily": "Inter, sans-serif", "whiteSpace": "normal", "height": "auto"},
        style_header={"fontWeight": "600", "backgroundColor": "#1e2235", "color": "#64748b",
                      "fontSize": "10px", "textTransform": "uppercase", "letterSpacing": "0.06em",
                      "border": "none", "borderBottom": "1px solid rgba(255,255,255,0.10)"},
        style_data_conditional=[
            {"if": {"filter_query": "{Notes} contains 'best'"},
             "backgroundColor": "rgba(5,150,105,0.14)"},
            {"if": {"filter_query": "{Notes} contains 'baseline'"}, "color": "#94a3b8"},
        ],
        page_size=12)


def _style_fig(fig, height=320):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=_CHART_BG,
        font=dict(family="Inter, sans-serif", size=11, color=_INK),
        margin=dict(l=55, r=15, t=20, b=30), height=height, hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10)),
        hoverlabel=dict(bgcolor="#1e2235", font=dict(color="#e2e8f0", size=11)))
    fig.update_xaxes(gridcolor=_GRID, zeroline=False, showline=False)
    fig.update_yaxes(gridcolor=_GRID, zeroline=False, showline=False)
    return fig


def _forecast_fig(entry, history):
    fig = go.Figure()
    if history and history.get("x"):
        fig.add_scatter(x=history["x"], y=history["y"], mode="lines", name="History",
                        line=dict(width=1.8, color=_C_HIST))
    bt = entry.get("backtest") or {}
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
    return _style_fig(fig)


def _details(report, data):
    stage6 = data.get("_stage6") or {}
    recipe = stage6.get("recipe") or {}
    rat = recipe.get("rationale") or {}
    feat_kids = [html.Div([html.Strong(f"{k}: "), html.Span(str(v))],
                          style={"fontSize": "0.76rem", "color": "#94a3b8", "marginBottom": "3px"})
                 for k, v in rat.items()]
    n_feat = stage6.get("n_features", 0)
    feat_kids.append(html.Div(
        f"{n_feat} features materialized" if n_feat else
        "Feature matrix is materialized only when an ML model (LightGBM/XGBoost) is trained.",
        style={"fontSize": "0.72rem", "color": "#64748b", "marginTop": "4px"}))

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
    return html.Details([
        html.Summary("Feature recipe, hyperparameters & CV detail",
                     style={"cursor": "pointer", "color": "#64748b", "fontSize": "0.8rem"}),
        html.Div(meta, style={"fontSize": "0.74rem", "color": "#64748b", "margin": "8px 0"}),
        html.Div(html.Span([_cicon("bi-diagram-3"), "EDA-seeded features"], style=_SH),
                 className="mt-2 mb-1"),
        html.Div(feat_kids),
        html.Div(html.Span([_cicon("bi-sliders"), "Model parameters"], style=_SH),
                 className="mt-3 mb-1"),
        html.Div(params_rows or [html.Span("—", style={"color": "#64748b"})]),
    ], className="mb-2")


def _results_block(data, report):
    scope = report.get("scope")
    results = report.get("results") or []
    trained = [e for e in results if not e.get("error")]
    hero = next((e for e in results if e.get("is_best")), trained[0] if trained else None)

    kids = []
    if hero:
        m = hero.get("metrics") or {}
        kids.append(dbc.Card(dbc.CardBody([
            html.Div(html.Span([_cicon("bi-trophy"), "Best model"], style=_SH), className="mb-1"),
            html.Div([
                html.Span(hero.get("label", hero["model"]),
                          style={"fontSize": "1.4rem", "fontWeight": "700", "color": _TEXT,
                                 "marginRight": "12px"}),
                dbc.Badge(f"MASE {_fmt(m.get('mase'))}", color="success", className="me-1"),
                dbc.Badge(f"sMAPE {_fmt(m.get('smape'))}%", color="secondary"),
            ], className="d-flex align-items-center flex-wrap"),
            html.P("MASE < 1 means the model beats the seasonal-naive benchmark.",
                   className="mb-0 mt-1", style={"fontSize": "0.78rem", "color": "#64748b"}),
        ]), className="mb-3", style={"borderLeft": "3px solid #059669"}))

        history = None
        if scope != "per_series":
            history = (((data.get("_stage4") or {}).get("eda") or {}).get("full") or {}) \
                .get("aggregate", {}).get("plot_data", {}).get("series")
        kids.append(dbc.Card(dbc.CardBody([
            html.Div(html.Span([_cicon("bi-graph-up-arrow"),
                                f"{hero.get('label', hero['model'])} — history, backtest & forecast"],
                               style=_SH), className="mb-2"),
            dcc.Graph(figure=_forecast_fig(hero, history),
                      config={"displayModeBar": False},
                      style={"borderRadius": "10px", "overflow": "hidden"}),
        ]), className="mb-3"))

    kids.append(dbc.Card(dbc.CardBody([
        html.Div(html.Span([_cicon("bi-list-ol"), "Leaderboard (ranked by MASE)"], style=_SH),
                 className="mb-2"),
        _leaderboard(report),
    ]), className="mb-3"))
    kids.append(_details(report, data))
    return kids


# ── Tab entry point ───────────────────────────────────────────────────────────

def render_training_tab(data):
    if not data:
        return html.Div(
            [_cicon("bi-info-circle", fontSize="2rem", color="#334155",
                    display="block", marginBottom="12px", marginRight="0"),
             html.P("Load a dataset on the Data tab first.",
                    style={"color": "#64748b", "fontSize": "0.9rem"})],
            className="text-center mt-5 pt-4")

    sel = data.get("_stage5")
    if not sel:
        return html.Div([
            html.Div([html.Span([_cicon("bi-lightning-fill"), "Training"],
                                className="fw-semibold section-title")], className="mb-3"),
            dbc.Alert([_cicon("bi-info-circle"),
                       "Run model selection first (Pipeline Setup → confirm, or the Model "
                       "Select tab) — training needs the recommendation and eligibility list."],
                      color="secondary", className="py-2"),
        ])

    header = html.Div([html.Span([_cicon("bi-lightning-fill"), "Training"],
                                 className="fw-semibold section-title")], className="mb-3")
    report = data.get("_stage7")
    blocks = [header, _picker_card(data, sel)]
    if report:
        blocks += _results_block(data, report)
    else:
        blocks.append(dbc.Alert(
            [_cicon("bi-arrow-up-circle"),
             "Choose your models above and click Train — walk-forward backtest, a "
             "seasonal-naive baseline and a horizon forecast will appear here."],
            color="secondary", className="py-2"))
    return html.Div(blocks)
