"""Results tab (Stage 8) — pure rendering; callbacks live in callbacks.py.

Model picker (trained non-error models, best pre-selected) + an OPT-IN LLM insight
report (explicit note: derived statistics and forecast summaries are sent to the
configured LLM — never raw dataset rows; no rule-based fallback, an LLM failure is
shown as an error) + forecast downloads (CSV / Parquet via GET /runs/{id}/forecast-file).

Reads ``_stage7`` for the model list and ``_stage8`` for a previously generated report
(both reconstructed by /summary, so a report survives reloads and Past Runs loads).
"""
import dash_bootstrap_components as dbc
from dash import dcc, html

_SH = {"color": "var(--ink-muted)", "fontSize": "0.78rem", "textTransform": "uppercase",
       "letterSpacing": "0.06em"}
_HIDDEN = {"display": "none"}


def _cicon(name, **style):
    return html.I(className=f"bi {name}", style={"marginRight": "5px", **style})


def _trained_options(report):
    return [{"label": e.get("label", e["model"]), "value": e["model"]}
            for e in report.get("results") or [] if not e.get("error")]


def _report_card(report):
    options = _trained_options(report)
    default = report.get("best_model")
    if default not in [o["value"] for o in options]:
        default = options[0]["value"] if options else None
    return dbc.Card(dbc.CardBody([
        html.Div(html.Span([_cicon("bi-file-earmark-text"), "Insight report"], style=_SH),
                 className="mb-2"),
        html.P("A plain-language forecast report: what's expected and when, in the "
               "target's real units — growth vs recent history, best/worst case, and "
               "what to keep in mind. No pipeline statistics.",
               style={"fontSize": "0.82rem", "color": "var(--ink-muted)"}),
        dbc.Row([
            dbc.Col(dcc.Dropdown(id="dropdown-results-model", options=options,
                                 value=default, clearable=False,
                                 style={"fontSize": "0.85rem"}), md=7),
            dbc.Col(dcc.Loading(
                dbc.Button([_cicon("bi-stars"), "Generate report"],
                           id="btn-generate-report", color="primary", className="w-100",
                           style={"fontWeight": "600"}),
                type="circle", color="#6366f1", delay_show=100,
                target_components={"cleaning-status": "children"}), md=5),
        ], className="mb-2 g-2"),
        html.Small([_cicon("bi-shield-lock", fontSize="0.7rem"),
                    "Opt-in: clicking Generate sends computed statistics, model metrics "
                    "and forecast summaries for this run to the configured report LLM — "
                    "never raw dataset rows. Nothing is sent until you click."],
                   style={"color": "var(--ink-faint)", "fontSize": "0.72rem"}),
    ]), className="mb-3")


def _report_body(stage8):
    llm = stage8.get("llm") or {}
    meta = " · ".join(x for x in [
        stage8.get("model_label") or stage8.get("model"),
        llm.get("model"),
        f"generated {str(stage8.get('generated_at', ''))[:19].replace('T', ' ')} UTC",
    ] if x)
    return dbc.Card(dbc.CardBody([
        html.Div(meta, style={"fontSize": "0.72rem", "color": "var(--ink-faint)"},
                 className="mb-2"),
        dcc.Markdown(stage8.get("markdown") or "", className="results-report-md",
                     link_target="_blank"),
    ]), className="mb-3")


def _download_card(report):
    per_series = bool(report.get("has_series_forecasts")) \
        or report.get("scope") == "per_series"
    switch = dbc.Switch(id="switch-download-series", value=False,
                        label="Per-series rows instead of the aggregate",
                        style={"fontSize": "0.82rem"})
    # Keep the switch mounted even when irrelevant so the download callback's State
    # always resolves (same trick as switch-train-confirm on the Training tab).
    switch_wrap = html.Div(switch) if per_series else html.Div(switch, style=_HIDDEN)
    return dbc.Card(dbc.CardBody([
        html.Div(html.Span([_cicon("bi-download"), "Download forecast"], style=_SH),
                 className="mb-2"),
        html.P("The selected model's horizon forecast (dates, predicted values and "
               "interval bounds where available).",
               style={"fontSize": "0.82rem", "color": "var(--ink-muted)"}),
        switch_wrap,
        dbc.Row([
            dbc.Col(dbc.Button([_cicon("bi-filetype-csv"), "CSV"],
                               id="btn-download-csv", color="secondary", outline=True,
                               className="w-100"), md=6),
            dbc.Col(dbc.Button([_cicon("bi-file-earmark-binary"), "Parquet"],
                               id="btn-download-parquet", color="secondary", outline=True,
                               className="w-100"), md=6),
        ], className="g-2 mt-1"),
    ]), className="mb-3")


def render_results_tab(data):
    if not data:
        return html.Div(
            [_cicon("bi-info-circle", fontSize="2rem", color="var(--ink-ghost)",
                    display="block", marginBottom="12px", marginRight="0"),
             html.P("Load a dataset on the Data tab first.",
                    style={"color": "var(--ink-faint)", "fontSize": "0.9rem"})],
            className="text-center mt-5 pt-4")

    header = html.Div([html.Span([_cicon("bi-clipboard-check"), "Results"],
                                 className="fw-semibold section-title")], className="mb-3")
    report = data.get("_stage7")
    if not report:
        return html.Div([
            header,
            dbc.Alert([_cicon("bi-info-circle"),
                       "Train at least one model on the Training tab first — the results "
                       "report and forecast download read the training output."],
                      color="secondary", className="py-2"),
        ])

    stage8 = data.get("_stage8")
    body = [header,
            dbc.Row([
                dbc.Col(_report_card(report), md=8),
                dbc.Col(_download_card(report), md=4),
            ])]
    if stage8:
        body.append(_report_body(stage8))
    else:
        body.append(dbc.Alert(
            [_cicon("bi-file-earmark-text"),
             "No report yet — pick a model and generate one (optional, uses the LLM)."],
            color="secondary", className="py-2"))
    return html.Div(body)
