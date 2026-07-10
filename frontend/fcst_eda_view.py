"""Forecast EDA tab — pure rendering (no callbacks; those live in callbacks.py).

Results-only since the intent-first refactor: every choice (target, scope, series key,
frequency, horizon, exog) was confirmed on the Pipeline Setup tab and Stage 4 ran as
part of the confirm chain. This tab shows the outcome from ``_stage4.eda`` plus a
compact recap of the confirmed intent, and offers a frequency/horizon-only re-run
(those two don't affect cleaning, so adjusting them must not force a re-clean).

Chart styling: dark-surface palette validated against #161b2e (indigo #6366f1 series,
emerald #059669 trend, amber #d97706 seasonal, red #e66767 residual — all ≥3:1 contrast,
CVD-checked). Identity never rides on color alone: every component gets its own titled
panel, and the multi-series overlay carries a legend.
"""
import json

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html, dash_table
from plotly.subplots import make_subplots

from frontend.ui import collapse_section, horizon_datepicker

_TEXT = "var(--bs-body-color)"
_CHART_BG = "#161b2e"
_GRID = "rgba(255,255,255,0.06)"
_INK = "#94a3b8"

_C_SERIES = "#6366f1"
_C_TREND = "#059669"
_C_SEASONAL = "#d97706"
_C_RESID = "#e66767"
_C_OVERLAY = [_C_SERIES, _C_TREND, _C_SEASONAL, _C_RESID]  # fixed slot order, never cycled

_SH = {"color": "#94a3b8", "fontSize": "0.78rem", "textTransform": "uppercase",
       "letterSpacing": "0.06em"}

_CONF_BADGE = {
    "high": ("auto", "success"),
    "medium": ("suggested", "info"),
    "low": ("confirm", "warning"),
}


def _cicon(name, **style):
    return html.I(className=f"bi {name}", style={"marginRight": "5px", **style})


def _lbl(text, confidence=None):
    """Form label with an optional confidence badge from detect()."""
    kids = [html.Span(text)]
    if confidence in _CONF_BADGE:
        word, color = _CONF_BADGE[confidence]
        kids.append(dbc.Badge(word, color=color, className="ms-2",
                              style={"fontSize": "0.6rem", "verticalAlign": "middle"}))
    return html.Label(kids, className="form-label",
                      style={"fontSize": "0.8rem", "color": "#94a3b8"})


def _datatable(rows, columns):
    # Mirrors callbacks._datatable (kept local — importing callbacks here would be circular).
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


def _fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return str(v)


# ── Plotly figure base ───────────────────────────────────────────────────────

# Interactive but unobtrusive: toolbar on hover only, wheel-zoom on, double-click resets.
_GRAPH_CONFIG = {
    "displayModeBar": "hover",
    "scrollZoom": True,
    "doubleClick": "reset",
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}


def _style_fig(fig, height=280):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=_CHART_BG,
        font=dict(family="Inter, sans-serif", size=11, color=_INK),
        margin=dict(l=50, r=15, t=30, b=30), height=height,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1e2235", font=dict(color="#e2e8f0", size=11)),
        dragmode="zoom",
        modebar=dict(bgcolor="rgba(0,0,0,0)", color="#64748b",
                     activecolor="#e2e8f0", orientation="v"),
    )
    fig.update_xaxes(gridcolor=_GRID, zeroline=False, linecolor=_GRID, showline=False)
    fig.update_yaxes(gridcolor=_GRID, zeroline=False, linecolor=_GRID, showline=False)
    return fig


def _graph(fig):
    return dcc.Graph(figure=fig, config=_GRAPH_CONFIG,
                     style={"borderRadius": "10px", "overflow": "hidden"})


def _chart_card(title, icon, graph, subtitle=None):
    body = [html.H6([_cicon(icon), title], className="fw-semibold mb-1", style=_SH)]
    if subtitle:
        body.append(html.P(subtitle, style={"fontSize": "0.72rem", "color": "#64748b",
                                            "marginBottom": "6px"}))
    body.append(graph)
    return dbc.Card(dbc.CardBody(body), className="mb-3")


def _series_fig(plot_series, target, height=280):
    fig = go.Figure(go.Scatter(
        x=plot_series["x"], y=plot_series["y"], mode="lines", name=target,
        line=dict(width=2, color=_C_SERIES),
    ))
    fig.update_layout(showlegend=False)  # single series — the card title names it
    return _style_fig(fig, height)


def _decomposition_fig(decomp, series):
    parts = [
        ("Observed", series, _C_SERIES),
        ("Trend", decomp["trend"], _C_TREND),
        ("Seasonal", decomp["seasonal"], _C_SEASONAL),
        ("Residual", decomp["resid"], _C_RESID),
    ]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=[p[0] for p in parts])
    for i, (name, data, color) in enumerate(parts, start=1):
        fig.add_trace(
            go.Scatter(x=data["x"], y=data["y"], mode="lines", name=name,
                       line=dict(width=1.5, color=color)),
            row=i, col=1,
        )
    fig.update_layout(showlegend=False)
    fig.update_annotations(font=dict(size=11, color=_INK))
    return _style_fig(fig, height=560)


def _acf_fig(values, conf_bound, title):
    lags = list(range(1, len(values) + 1))
    fig = go.Figure()
    if conf_bound:
        fig.add_hrect(y0=-conf_bound, y1=conf_bound,
                      fillcolor="rgba(148,163,184,0.12)", line_width=0)
    fig.add_trace(go.Bar(x=lags, y=values, name=title,
                         marker=dict(color=_C_SERIES), width=0.5))
    fig.update_layout(showlegend=False, bargap=0.4)
    fig.update_xaxes(title_text="lag", title_font=dict(size=10))
    return _style_fig(fig, height=230)


def _overlay_fig(deep_dives, target):
    fig = go.Figure()
    for i, d in enumerate(deep_dives[:4]):
        ps = d["stats"]["plot_data"]["series"]
        label = d["key"] if len(d["key"]) <= 28 else d["key"][:26] + "…"
        fig.add_trace(go.Scatter(x=ps["x"], y=ps["y"], mode="lines", name=label,
                                 line=dict(width=1.5, color=_C_OVERLAY[i])))
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
    )
    return _style_fig(fig, height=300)


# ── Stat tiles ───────────────────────────────────────────────────────────────

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


_INTERMITTENCY_COLOR = {"smooth": "success", "intermittent": "warning",
                        "erratic": "warning", "lumpy": "danger", "no_demand": "secondary"}


def _verdict_tiles(payload):
    """The four numbers that actually drive the model decision. Everything else
    (stationarity, SNR, CV, length) lives in the collapsible statistical-detail table."""
    fcast = payload.get("forecastability")
    seas_s = payload.get("seasonality_strength")
    trend_s = payload.get("trend_strength")
    icls = payload.get("intermittency_class") or "—"
    tiles = dbc.Row([
        _tile("bi-magic", "Forecastability",
              _fmt(fcast, 2), sub="1 − spectral entropy (1 = clean signal)"),
        _tile("bi-arrow-repeat", "Seasonality",
              _fmt(seas_s, 2), sub=f"period {payload.get('seasonality_period') or '—'}"),
        _tile("bi-graph-up-arrow", "Trend",
              _fmt(trend_s, 2), sub=payload.get("trend_direction", "—")),
        _tile("bi-activity", "Demand pattern",
              dbc.Badge(icls, color=_INTERMITTENCY_COLOR.get(icls, "secondary")),
              sub=f"ADI {_fmt(payload.get('adi'), 2)} · CV² {_fmt(payload.get('cv2'), 2)}"),
    ])
    return dbc.Card(dbc.CardBody(tiles), className="mb-3",
                    style={"borderLeft": "3px solid #6366f1"})


# ── Confirmed-setup summary ──────────────────────────────────────────────────

_AGG_LABEL = {"sum": "sum per period", "mean": "mean per period",
              "nunique": "distinct count per period", "count": "count per period"}
_SCOPE_LABEL = {"aggregate": "Overall (aggregate)", "per_series": "Per-series (panel)"}


def _setup_summary(selections):
    """One-line recap of the confirmed forecast setup (the full form lives on Pipeline
    Setup — repeating it here as a tile card was pure duplication)."""
    if not selections or not selections.get("target_col"):
        return None
    group_cols = selections.get("group_cols") or []
    exog = selections.get("exog_cols") or []
    agg = selections.get("agg") or ""
    if selections.get("agg_forced"):
        agg += "·auto-corrected"
    parts = [
        f"Target {selections['target_col']} ({agg})",
        str(selections.get("forecast_frequency") or "—"),
        f"horizon {selections.get('horizon') or '—'}",
        _SCOPE_LABEL.get(selections.get("scope"), selections.get("scope") or "—"),
    ]
    if group_cols:
        parts.append("key " + " × ".join(group_cols))
    parts.append("exog: " + (", ".join(exog) if exog else "none"))
    return html.Div([
        html.Span([_cicon("bi-sliders"), "  ·  ".join(parts)],
                  style={"fontSize": "0.78rem", "color": "#94a3b8"}),
        html.Small("change on the Pipeline Setup tab",
                   style={"color": "#64748b", "fontSize": "0.68rem"}),
    ], className="d-flex justify-content-between align-items-center mb-2 px-1")


# ── Frequency/horizon adjust mini-form (Stage-4-only re-run) ─────────────────

def _data_end(data):
    """Last actual timestamp for the 'forecast until' picker: the CLEANED grid's end from
    Stage 4 when available, else the raw-data end recorded by intent detection."""
    stage4_end = ((((data.get("_stage4") or {}).get("eda") or {}).get("full") or {})
                  .get("aggregate") or {}).get("end")
    return stage4_end or (data.get("_intent") or {}).get("suggestions", {}).get("data_end")


def _adjust_form(data, eda_exists):
    """Stage-4-only re-run controls. Scope / frequency / horizon don't affect cleaning, so
    changing them here re-runs from forecast EDA onward (re-chaining model selection)
    WITHOUT the expensive re-clean. The series key / target / agg — which DO affect cleaning
    — stay on the Pipeline Setup tab."""
    intent = data.get("_intent") or {}
    selections = intent.get("selections") or {}
    suggestions = intent.get("suggestions") or {}
    freq_current = (selections.get("forecast_frequency")
                    or suggestions.get("frequency", {}).get("suggested"))
    freq_opts = suggestions.get("frequency", {}).get("options") or (
        [freq_current] if freq_current else [])
    horizon_current = selections.get("horizon") or suggestions.get("horizon", {}).get("suggested", 12)
    scope_current = (selections.get("scope")
                     or suggestions.get("scope", {}).get("suggested", "aggregate"))
    group_cols = selections.get("group_cols") or suggestions.get("group", {}).get("suggested") or []
    has_key = bool(group_cols)

    _radio_lbl = {"display": "block", "marginBottom": "4px", "cursor": "pointer",
                  "color": "#94a3b8", "fontSize": "0.82rem"}
    key_hint = None
    if not has_key:
        key_hint = html.Div(
            "Per-series needs a series key — set one on the Pipeline Setup tab (that re-cleans).",
            style={"fontSize": "0.72rem", "color": "#d97706", "marginTop": "2px"})

    # Exogenous re-selection: also a Stage-4-only concern (cleaning never reads exog
    # for row decisions; the sanitizer merely protects selected exog from mutation).
    # Options = the intent detector's exog candidates ∪ whatever is currently selected;
    # the API validates against the cleaned columns.
    target_col = selections.get("target_col") or suggestions.get("target", {}).get("suggested")
    exog_current = selections.get("exog_cols") or []
    exog_opts = sorted({*(suggestions.get("exogenous", {}).get("candidates") or []),
                        *exog_current} - {target_col, None})

    return dbc.Card(dbc.CardBody([
        html.P("Scope, frequency, horizon and exogenous drivers re-run from here "
               "without re-cleaning.",
               style={"fontSize": "0.72rem", "color": "#64748b", "marginBottom": "8px"}),
        dbc.Row([
            dbc.Col([
                _lbl("Scope"),
                dcc.RadioItems(
                    id="radio-fcst-scope",
                    options=[{"label": "  Overall (aggregate)", "value": "aggregate"},
                             {"label": "  Per-series (panel)", "value": "per_series",
                              "disabled": not has_key}],
                    value=scope_current,
                    inputStyle={"marginRight": "6px", "accentColor": "#6366f1"},
                    labelStyle=_radio_lbl),
                key_hint,
            ], md=5),
            dbc.Col([
                _lbl("Forecast frequency"),
                dcc.Dropdown(id="dropdown-fcst-freq",
                             options=[{"label": str(f).capitalize(), "value": f} for f in freq_opts],
                             value=freq_current, clearable=False,
                             style={"fontSize": "0.85rem"}),
            ], md=4),
            dbc.Col([
                _lbl("Horizon"),
                dbc.Input(id="input-fcst-horizon", type="number", min=1, step=1,
                          value=horizon_current, size="sm"),
                *horizon_datepicker(_data_end(data),
                                    "date-fcst-horizon-end", "fcst-horizon-note"),
            ], md=3),
        ]),
        dbc.Row([dbc.Col([
            _lbl("Exogenous drivers (optional)"),
            dcc.Dropdown(id="dropdown-fcst-exog", multi=True,
                         options=[{"label": c, "value": c} for c in exog_opts],
                         value=exog_current,
                         placeholder="No exogenous drivers",
                         style={"fontSize": "0.85rem"}),
        ])], className="mt-2"),
        dcc.Loading(
            dbc.Button(
                [_cicon("bi-graph-up"),
                 "Re-run Forecast EDA" if eda_exists else "Run Forecast EDA"],
                id="btn-fcst-rerun", color="primary", className="w-100 mt-3",
                style={"fontWeight": "600"},
            ),
            type="circle", color="#6366f1", delay_show=100,
            target_components={"cleaning-status": "children"},
        ),
    ]), className="mb-3")


# ── Results sections ─────────────────────────────────────────────────────────

def _payload_details(payload):
    return html.Details([
        html.Summary(
            [_cicon("bi-filetype-json", color="#6366f1"), "View model selection payload"],
            style={"cursor": "pointer", "fontSize": "0.78rem", "color": "#94a3b8",
                   "userSelect": "none"},
        ),
        html.Pre(
            json.dumps(payload, indent=2),
            style={"backgroundColor": "#161b2e", "color": "#e2e8f0", "fontSize": "0.72rem",
                   "lineHeight": "1.5", "padding": "10px 14px", "borderRadius": "10px",
                   "marginTop": "6px", "marginBottom": "0", "maxHeight": "320px",
                   "overflowY": "auto", "whiteSpace": "pre-wrap",
                   "border": "1px solid rgba(99, 102, 241, 0.25)"},
        ),
    ], className="mb-3")


def _seasonality_section(agg_stats):
    seas = agg_stats.get("seasonality", {})
    rows = [
        {"Period": c["period"],
         "Seasonal strength": _fmt(c.get("seasonal_strength")),
         "Trend strength": _fmt(c.get("trend_strength")),
         "SNR": _fmt(c.get("snr"), 1)}
        for c in seas.get("candidates", [])
    ]
    fft_rows = [
        {"FFT period": p["period"], "Power share": _fmt(p.get("power_share"))}
        for p in seas.get("fft_peaks", [])
    ]
    if not rows and not fft_rows:
        return []
    cols = []
    if rows:
        cols.append(dbc.Col([
            html.H6([_cicon("bi-arrow-repeat"), "Seasonal Periods Tested (STL)"],
                    className="fw-semibold mb-2", style=_SH),
            _datatable(rows, ["Period", "Seasonal strength", "Trend strength", "SNR"]),
        ], md=7))
    if fft_rows:
        cols.append(dbc.Col([
            html.H6([_cicon("bi-broadcast"), "FFT Dominant Cycles"],
                    className="fw-semibold mb-2", style=_SH),
            _datatable(fft_rows, ["FFT period", "Power share"]),
        ], md=5))
    return [dbc.Row(cols, className="mb-3")]


def _detail_tables(agg_stats, payload):
    st = agg_stats.get("stationarity", {})
    fc = agg_stats.get("forecastability", {})
    het = agg_stats.get("heteroskedasticity", {})
    tr = agg_stats.get("trend", {})
    brk = agg_stats.get("structural_breaks", {})
    rows = [
        # Formerly their own verdict tiles — demoted here to keep the tile card to the
        # four decision-driving numbers.
        {"Statistic": "Stationary", "Value": _fmt(payload.get("stationarity")),
         "Reading": f"ndiffs {_fmt(payload.get('ndiffs'))} · "
                    f"nsdiffs {_fmt(payload.get('nsdiffs'))}"},
        {"Statistic": "Signal / noise ratio", "Value": _fmt(payload.get("snr"), 1),
         "Reading": "Var(trend+seasonal) / Var(resid)"},
        {"Statistic": "Coefficient of variation", "Value": _fmt(payload.get("cv"), 3),
         "Reading": "std / mean"},
        {"Statistic": "Series length", "Value": f"{payload.get('series_length', 0):,}",
         "Reading": f"{payload.get('forecast_frequency', '')} · "
                    f"{_fmt(payload.get('missing_pct'), 1)}% grid gaps"},
        {"Statistic": "ADF p-value", "Value": _fmt(st.get("adf_p"), 4),
         "Reading": "< 0.05 rejects unit root (stationary)"},
        {"Statistic": "KPSS p-value", "Value": _fmt(st.get("kpss_p"), 4),
         "Reading": "> 0.05 fails to reject stationarity"},
        {"Statistic": "Differences needed (ADF/KPSS/PP consensus)",
         "Value": _fmt(st.get("ndiffs")), "Reading": "d for ARIMA"},
        {"Statistic": "Seasonal differences needed", "Value": _fmt(st.get("nsdiffs")),
         "Reading": "D for SARIMA"},
        {"Statistic": "Spectral entropy", "Value": _fmt(fc.get("spectral_entropy")),
         "Reading": "0 = pure signal, 1 = white noise"},
        {"Statistic": "Sample entropy", "Value": _fmt(fc.get("sample_entropy")),
         "Reading": "lower = more regular/predictable"},
        {"Statistic": "Approximate entropy", "Value": _fmt(fc.get("approx_entropy")),
         "Reading": "lower = more regular"},
        {"Statistic": "DFA alpha", "Value": _fmt(fc.get("dfa_alpha")),
         "Reading": "> 0.5 = persistent / trending memory"},
        {"Statistic": "Overall slope", "Value": f"{_fmt(tr.get('slope_pct_per_period'))}%/period",
         "Reading": f"recent (last 20%): {_fmt(tr.get('recent_slope_pct_per_period'))}%/period"},
        {"Statistic": "Heteroskedastic", "Value": _fmt(het.get("heteroskedastic")),
         "Reading": f"transform: {het.get('transform_recommendation', 'none')}"},
        {"Statistic": "Residual Ljung-Box p", "Value": _fmt(payload.get("residual_ljung_box_p"), 4),
         "Reading": "< 0.05 = structure left for ARMA terms"},
        {"Statistic": "Structural breaks",
         "Value": str(len(brk.get("break_dates", []))),
         "Reading": ("recent break — consider shorter training window"
                     if brk.get("recent_break") else
                     ", ".join(brk.get("break_dates", [])[:4]) or "none detected")},
    ]
    return _datatable(rows, ["Statistic", "Value", "Reading"])


def _exog_section(exog):
    if not exog or not exog.get("has_exogenous"):
        return []
    cross_rows = [
        {"Driver": r["col"], "Corr (same period)": _fmt(r.get("corr_lag0")),
         "Best lead": f"{r['best_lag']} periods" if r.get("best_lag") else "0 (coincident)",
         "Corr at best lead": _fmt(r.get("best_corr"))}
        for r in exog.get("cross_correlation", [])
    ]
    vif_rows = [
        {"Regressor": r["col"], "VIF": _fmt(r.get("vif"), 1),
         "Collinear": "yes — drop or combine" if (r.get("vif") or 0) > 10 else "no"}
        for r in exog.get("vif", [])
    ]
    out = [
        html.H6([_cicon("bi-diagram-3"), "Exogenous Drivers"],
                className="fw-semibold mb-2 mt-3", style=_SH),
        html.P(exog.get("note") or "", style={"fontSize": "0.72rem", "color": "#64748b",
                                              "marginBottom": "6px"}),
    ]
    cols = []
    if cross_rows:
        cols.append(dbc.Col([_datatable(
            cross_rows, ["Driver", "Corr (same period)", "Best lead", "Corr at best lead"])],
            md=7))
    if vif_rows:
        cols.append(dbc.Col([_datatable(vif_rows, ["Regressor", "VIF", "Collinear"])], md=5))
    out.append(dbc.Row(cols, className="mb-3"))
    return out


def _panel_section(full):
    panel = full.get("panel")
    if not panel:
        return []
    mix = panel.get("intermittency_mix", {})
    mix_badges = [
        dbc.Badge(f"{cls}: {share * 100:.1f}%",
                  color=_INTERMITTENCY_COLOR.get(cls, "secondary"), className="me-2 mb-1")
        for cls, share in sorted(mix.items(), key=lambda kv: -kv[1])
    ]
    lengths = panel.get("series_length", {})
    tiles = dbc.Row([
        _tile("bi-collection", "Series in panel", f"{panel.get('n_series', 0):,}"),
        _tile("bi-rulers", "Series length",
              f"{lengths.get('median', 0):,}",
              sub=f"min {lengths.get('min', 0):,} · max {lengths.get('max', 0):,}"),
        _tile("bi-pie-chart", "Top-10 volume share",
              _fmt(panel.get("top_series_volume_share"), 3),
              sub="low = demand spread evenly across series"),
        dbc.Col([
            html.Div([html.I(className="bi bi-activity",
                             style={"color": "#64748b", "marginRight": "4px",
                                    "fontSize": "0.7rem"}),
                      html.Span("Demand pattern mix",
                                style={"fontSize": "0.65rem", "color": "#94a3b8",
                                       "textTransform": "uppercase",
                                       "letterSpacing": "0.06em"})],
                     className="d-flex align-items-center mb-1"),
            html.Div(mix_badges),
        ], width=3),
    ])

    deep = full.get("per_series_deep_dive", [])
    deep_rows = [
        {"Series": d["key"],
         "Pattern": d["stats"]["intermittency"]["class"],
         "Seasonal strength": _fmt(d["stats"].get("seasonal_strength")),
         "Trend strength": _fmt(d["stats"].get("trend_strength")),
         "Forecastability": _fmt(d["stats"]["forecastability"].get("score"), 2),
         "CV": _fmt(d["stats"].get("cv"))}
        for d in deep
    ]

    out = [
        html.Hr(className="my-4"),
        html.H6([_cicon("bi-collection"), "Panel Profile (per-series scope)"],
                className="fw-semibold mb-2", style=_SH),
        dbc.Card(dbc.CardBody(tiles), className="mb-3",
                 style={"borderLeft": "3px solid #d97706"}),
    ]
    if deep:
        target = full.get("selections", {}).get("target_col", "")
        out.append(_chart_card(
            "Top Series by Volume", "bi-graph-up",
            _graph(_overlay_fig(deep, target)),
            subtitle="The highest-volume individual series, for comparison against the aggregate.",
        ))
        out.append(_datatable(deep_rows, ["Series", "Pattern", "Seasonal strength",
                                          "Trend strength", "Forecastability", "CV"]))
    return out


def _results_section(stage4):
    eda = stage4.get("eda") or {}
    payload = eda.get("payload")
    full = eda.get("full")
    if not payload or not full:
        return []

    agg = full.get("aggregate", {})
    plot_data = agg.get("plot_data", {})
    selections = full.get("selections", {})
    target = selections.get("target_col", "")

    sections = [html.Hr(className="my-4")]

    warns = payload.get("warnings", [])
    if warns:
        sections.append(dbc.Alert(
            [html.Div([_cicon("bi-exclamation-triangle-fill"),
                       html.Strong("Forecasting risks detected")], className="mb-1")] +
            [html.Div(w, style={"fontSize": "0.8rem"}) for w in warns],
            color="warning", className="py-2 mb-3",
        ))

    sections.append(_verdict_tiles(payload))

    if plot_data.get("series"):
        agg_label = {"sum": "sum", "mean": "mean", "nunique": "distinct count",
                     "count": "count"}.get(selections.get("agg"), "")
        scope_label = ("aggregate of all series" if selections.get("scope") == "per_series"
                       else "overall series")
        sections.append(_chart_card(
            f"{target} — Analysis Series", "bi-graph-up",
            _graph(_series_fig(plot_data["series"], target)),
            subtitle=f"{agg_label} per {selections.get('forecast_frequency', '')} period "
                     f"({scope_label}).",
        ))

    if plot_data.get("decomposition"):
        d = plot_data["decomposition"]
        sections.append(_chart_card(
            f"Decomposition (period = {d.get('period')})", "bi-layers",
            _graph(_decomposition_fig(d, plot_data["series"])),
            subtitle="Trend + seasonal + residual split behind the strength scores above.",
        ))

    ac = agg.get("autocorrelation", {})
    if ac.get("acf"):
        acf_col = dbc.Col([_graph(_acf_fig(ac["acf"], ac.get("conf_bound"), "ACF"))], md=6)
        cols = [acf_col]
        if ac.get("pacf"):
            cols.append(dbc.Col([_graph(_acf_fig(ac["pacf"], ac.get("conf_bound"), "PACF"))],
                                md=6))
        sections.append(_chart_card(
            "Autocorrelation (ACF / PACF)", "bi-bar-chart-line",
            dbc.Row(cols),
            subtitle="Bars outside the gray band are statistically significant lags.",
        ))

    seas_rows = _seasonality_section(agg)
    if seas_rows:
        sections.append(collapse_section("Seasonality tables (STL periods + FFT)",
                                         seas_rows, icon="bi-arrow-repeat"))
    sections.append(collapse_section("Statistical detail (16 tests)",
                                     _detail_tables(agg, payload),
                                     icon="bi-clipboard-data"))
    sections += _exog_section(full.get("exogenous"))
    sections += _panel_section(full)
    sections.append(html.Hr(className="my-3"))
    sections.append(_payload_details(payload))
    return sections


# ── Tab entry point ──────────────────────────────────────────────────────────

def render_fcst_eda_tab(data):
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
    intent = data.get("_intent") or {}
    selections = intent.get("selections") or {}

    if not eda_exists and not data.get("_stage3"):
        return html.Div(
            [_cicon("bi-info-circle", fontSize="2rem", color="#334155",
                    display="block", marginBottom="12px", marginRight="0"),
             html.P("Confirm the forecast setup on the Pipeline Setup tab — the pipeline "
                    "runs cleaning, validation, forecast EDA and model selection in one go.",
                    style={"color": "#64748b", "fontSize": "0.9rem"})],
            className="text-center mt-5 pt-4",
        )

    header = html.Div([
        html.Span([_cicon("bi-graph-up"), "Forecast EDA"],
                  className="fw-semibold section-title"),
    ], className="mb-3")

    validation = (data.get("_stage3") or {}).get("_validation", {})
    val_warn = []
    if validation and not validation.get("passed", True):
        val_warn = [dbc.Alert(
            [_cicon("bi-shield-exclamation"),
             "The validation gate FAILED for this cleaning run — these statistics may "
             "not be trustworthy. Consider re-running cleaning first."],
            color="danger", className="py-2 mb-3",
        )]

    summary = _setup_summary(
        selections or (stage4.get("eda") or {}).get("full", {}).get("selections") or {})

    if not eda_exists:
        return html.Div([
            header, *val_warn,
            *( [summary] if summary else [] ),
            dbc.Alert(
                [_cicon("bi-info-circle"),
                 "Cleaning is done but forecast EDA hasn't produced results yet — "
                 "run it below."],
                color="secondary", className="py-2 mb-3"),
            _adjust_form(data, eda_exists=False),
        ])

    return html.Div([
        header,
        *val_warn,
        *( [summary] if summary else [] ),
        _adjust_form(data, eda_exists=True),
        *_results_section(stage4),
    ])
