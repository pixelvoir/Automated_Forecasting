"""Dash callbacks — all interactivity lives here."""
import base64
import os
import tempfile
from pathlib import Path

import requests
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, dash_table, no_update, callback, ALL, ctx, clientside_callback

API_URL = os.environ.get("API_URL", "http://localhost:8000")


# ── Theme toggle (clientside — zero server round-trip) ───────────────────────
# Applies saved theme on page load; toggles on button click.

clientside_callback(
    """
    function(data) {
        if (data === 'light') {
            document.body.classList.add('light-mode');
            return 'bi bi-moon-stars-fill';
        } else {
            document.body.classList.remove('light-mode');
            return 'bi bi-sun';
        }
    }
    """,
    Output("theme-toggle-icon", "className"),
    Input("theme-store", "data"),
)

clientside_callback(
    """
    function(n) {
        const body = document.body;
        const isLight = body.classList.toggle('light-mode');
        return isLight ? 'light' : 'dark';
    }
    """,
    Output("theme-store", "data"),
    Input("btn-theme-toggle", "n_clicks"),
    prevent_initial_call=True,
)


def _save_upload(contents: str, filename: str) -> str:
    """Decode a dcc.Upload base64 payload and save to a temp file. Returns the file path."""
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    suffix = Path(filename).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="upload_")
    try:
        tmp.write(decoded)
    finally:
        tmp.close()
    return tmp.name


# ── 0. Update stage pills when results-store changes ─────────────────────

@callback(
    Output("pill-preclean", "color"),
    Input("results-store", "data"),
)
def update_pills(data):
    if not data:
        return "secondary"
    s2 = data.get("_stage2") or {}
    # _stage2 from a live run: {"decision_payload": {...}, "status": "completed", ...}
    # _stage2 from a past run: {"decision_payload": {...}}
    if s2.get("decision_payload") is not None or s2.get("status") == "completed":
        return "success"
    return "secondary"


# ── 1. Toggle DB / File sections based on source radio ────────────────────

_SHOW = {"display": "block"}
_HIDE = {"display": "none"}

@callback(
    Output("section-db", "style"),
    Output("section-file", "style"),
    Input("source-radio", "value"),
)
def toggle_source(source):
    if source == "database":
        return _SHOW, _HIDE
    return _HIDE, _SHOW


# ── 1b. Show uploaded filename ────────────────────────────────────────────

@callback(
    Output("upload-filename", "children"),
    Input("upload-file", "filename"),
    prevent_initial_call=True,
)
def show_upload_filename(filename):
    if filename:
        return f"Selected: {filename}"
    return ""


# ── 2. Connect to database (credentials from UI, never stored) ────────────
# Credentials flow: browser input → callback State → POST body → FastAPI RAM
# → DB connection → disposed. Nothing is persisted anywhere.

@callback(
    Output("dropdown-table", "options"),
    Output("dropdown-table", "placeholder"),
    Output("connection-alert", "children"),
    Output("btn-connect", "disabled"),
    Input("btn-connect", "n_clicks"),
    State("input-host", "value"),
    State("input-port", "value"),
    State("input-database", "value"),
    State("input-user", "value"),
    State("input-password", "value"),
    prevent_initial_call=True,
)
def connect_db(_, host, port, database, user, password):
    missing = [name for name, val in [
        ("Host", host), ("Database", database), ("Username", user), ("Password", password)
    ] if not val or not str(val).strip()]
    if missing:
        alert = dbc.Alert(f"Fill in: {', '.join(missing)}", color="warning", className="mb-2 py-2 small")
        return [], "Connect first…", alert, False

    payload = {
        "host": host.strip(),
        "port": int(port) if port else 5432,
        "database": database.strip(),
        "user": user.strip(),
        "password": password,
    }
    try:
        resp = requests.post(f"{API_URL}/runs/tables-with-creds", json=payload, timeout=10)
        if resp.ok:
            tables = resp.json()["tables"]
            options = [{"label": t["qualified"], "value": t["qualified"]} for t in tables]
            placeholder = f"{len(tables)} table(s) found — select one" if options else "No tables found"
            alert = dbc.Alert(
                f"Connected — {len(tables)} table(s) available.",
                color="success", className="mb-2 py-2 small", dismissable=True,
            )
            return options, placeholder, alert, False
        detail = resp.json().get("detail", "Connection failed")
        alert = dbc.Alert(detail, color="danger", className="mb-2 py-2 small")
        return [], "Connect first…", alert, False
    except requests.exceptions.ConnectionError:
        alert = dbc.Alert("Cannot reach API — is run_dev.bat running?", color="danger", className="mb-2 py-2 small")
        return [], "Connect first…", alert, False
    except Exception as e:
        alert = dbc.Alert(str(e), color="danger", className="mb-2 py-2 small")
        return [], "Connect first…", alert, False


# ── 3. Toggle custom SQL query textarea ───────────────────────────────────

@callback(
    Output("collapse-query", "is_open"),
    Input("btn-toggle-query", "n_clicks"),
    State("collapse-query", "is_open"),
    prevent_initial_call=True,
)
def toggle_query(_, is_open):
    return not is_open


# ── 4. Trigger ingestion run ───────────────────────────────────────────────
# Credentials are passed as State (read from the input fields at click time).
# They are never stored between requests.

@callback(
    Output("results-store", "data"),
    Output("alert-div", "children"),
    Output("btn-run", "disabled"),
    Input("btn-run", "n_clicks"),
    State("source-radio", "value"),
    State("dropdown-table", "value"),
    State("input-query", "value"),
    State("input-host", "value"),
    State("input-port", "value"),
    State("input-database", "value"),
    State("input-user", "value"),
    State("input-password", "value"),
    State("upload-file", "contents"),
    State("upload-file", "filename"),
    prevent_initial_call=True,
)
def trigger_run(_, source, table, query, host, port, database, user, password,
                upload_contents, upload_filename):
    if source == "file":
        if not upload_contents:
            return no_update, dbc.Alert(
                "Upload a file first — drag & drop or click Browse.", color="warning", dismissable=True
            ), False
        try:
            tmp_path = _save_upload(upload_contents, upload_filename)
        except Exception as e:
            return no_update, dbc.Alert(f"Could not process upload: {e}", color="danger", dismissable=True), False
        payload = {"source": "file", "file_path": tmp_path}

    else:  # database
        if not all([host, database, user, password]):
            return no_update, dbc.Alert(
                "Fill in all credential fields and click Connect first.",
                color="warning", dismissable=True,
            ), False
        if query and query.strip():
            payload = {"source": "database", "query": query.strip()}
        elif table:
            payload = {"source": "database", "table": table}
        else:
            return no_update, dbc.Alert(
                "Select a table from the dropdown or enter a custom query.",
                color="warning", dismissable=True,
            ), False
        payload["credentials"] = {
            "host": host.strip(),
            "port": int(port) if port else 5432,
            "database": database.strip(),
            "user": user.strip(),
            "password": password,
        }

    try:
        run_resp = requests.post(f"{API_URL}/runs", json=payload, timeout=120)
        if not run_resp.ok:
            detail = run_resp.json().get("detail", "Unknown error")
            return no_update, dbc.Alert(f"Run failed: {detail}", color="danger", dismissable=True), False

        run_data = run_resp.json()
        run_id = run_data["run_id"]

        meta_resp = requests.get(f"{API_URL}/runs/{run_id}/metadata", timeout=15)
        if not meta_resp.ok:
            return no_update, dbc.Alert(
                f"Run completed ({run_id}) but metadata fetch failed.", color="warning", dismissable=True
            ), False

        full_data = meta_resp.json()
        full_data["_summary"] = run_data

        # Auto-trigger Stage 2 immediately after Stage 1
        try:
            eda_resp = requests.post(f"{API_URL}/runs/{run_id}/pre-clean-eda", timeout=60)
            if eda_resp.ok:
                full_data["_stage2"] = eda_resp.json()
        except Exception:
            pass

        return full_data, dbc.Alert(
            f"Completed: {run_id}", color="success", dismissable=True, duration=6000
        ), False

    except requests.exceptions.Timeout:
        return no_update, dbc.Alert(
            "Request timed out — the table may be very large. "
            "Set 'row_limit' in config/settings.yaml to cap rows during dev.",
            color="danger", dismissable=True,
        ), False
    except requests.exceptions.ConnectionError:
        return no_update, dbc.Alert(
            "Cannot reach API — is run_dev.bat running?", color="danger", dismissable=True
        ), False
    except Exception as e:
        return no_update, dbc.Alert(f"Unexpected error: {e}", color="danger", dismissable=True), False


# ── 5. Render results from store ───────────────────────────────────────────

@callback(
    Output("results-panel", "children"),
    Input("results-store", "data"),
)
def render_results(data):
    if not data:
        return html.P(
            "Select a data source, connect, then click Run Ingestion.",
            className="text-muted mt-4 ms-2",
        )

    shape = data.get("shape", {})
    schema = data.get("schema", [])
    nulls = data.get("nulls", {})
    numeric_stats = data.get("numeric_stats", {})
    datetime_cols = data.get("datetime_cols", [])
    frequency = data.get("frequency", {})
    duplicates = data.get("duplicates", {})

    def _metric(icon, label, value, col_width=2):
        return dbc.Col([
            html.Div(
                [html.I(className=f"bi {icon}", style={"color": "#64748b", "marginRight": "4px", "fontSize": "0.7rem"}),
                 html.Span(label, style={"fontSize": "0.65rem", "color": "#94a3b8", "textTransform": "uppercase",
                                         "letterSpacing": "0.06em"})],
                className="d-flex align-items-center mb-1",
            ),
            html.Div(str(value), style={"fontSize": "1rem", "fontWeight": "700",
                                        "letterSpacing": "-0.02em"}),
        ], width=col_width)

    shape_card = dbc.Card(
        dbc.CardBody(
            dbc.Row([
                _metric("bi-fingerprint", "Run ID", data.get("run_id", "—"), col_width=4),
                _metric("bi-rows",        "Rows",   f"{shape.get('rows', 0):,}"),
                _metric("bi-columns-gap", "Cols",   shape.get("cols", 0)),
                _metric("bi-memory",      "Memory", f"{shape.get('memory_mb', 0)} MB"),
                _metric("bi-copy",        "Dupes",  duplicates.get("row_count", 0)),
            ])
        ),
        className="mb-3",
        style={"borderLeft": "3px solid #10b981"},
    )

    schema_rows = [
        {
            "Column": s["col"],
            "Inferred type": s["dtype_inferred"],
            "DB dtype": s["dtype_raw"],
            "Null count": nulls.get(s["col"], {}).get("count", 0),
            "Null %": nulls.get(s["col"], {}).get("pct", 0),
            "Frequency": frequency.get(s["col"], "—") if s["col"] in datetime_cols else "—",
        }
        for s in schema
    ]

    numeric_section = []
    if numeric_stats:
        numeric_rows = [
            {
                "Column": col,
                "Min": _fmt(s["min"]), "Max": _fmt(s["max"]),
                "Mean": _fmt(s["mean"]), "Median": _fmt(s["median"]),
                "Std": _fmt(s["std"]), "Skew": _fmt(s["skew"]),
                "Kurtosis": _fmt(s["kurtosis"]), "IQR": _fmt(s["iqr"]),
            }
            for col, s in numeric_stats.items()
        ]
        numeric_section = [
            html.H6(
                [html.I(className="bi bi-bar-chart-line", style={"marginRight": "5px"}), "Numeric Statistics"],
                className="mt-4 mb-2 fw-semibold",
                style={"color": "#94a3b8", "fontSize": "0.78rem", "textTransform": "uppercase", "letterSpacing": "0.06em"},
            ),
            _datatable(numeric_rows, ["Column", "Min", "Max", "Mean", "Median", "Std", "Skew", "Kurtosis", "IQR"]),
        ]

    stage2 = data.get("_stage2", {})
    stage2_section = _render_stage2(stage2) if stage2 else []

    def _icon(name):
        return html.I(className=f"bi {name}", style={"marginRight": "5px"})

    header = html.Div([
        html.Span([
            _icon("bi-cloud-download"),
            "Stage 1 — Ingestion Results",
        ], className="fw-semibold", style={"color": "#c7d2fe"}),
        dbc.Button(
            [html.I(className="bi bi-arrow-counterclockwise", style={"marginRight": "4px"}), "New Run"],
            id="btn-clear-results",
            color="outline-secondary",
            size="sm",
            className="text-decoration-none",
        ),
    ], className="d-flex justify-content-between align-items-center mb-3")

    return [
        header,
        shape_card,
        html.H6([_icon("bi-table"), "Schema"], className="mb-2 fw-semibold",
                style={"color": "#94a3b8", "fontSize": "0.78rem", "textTransform": "uppercase", "letterSpacing": "0.06em"}),
        _datatable(schema_rows, ["Column", "Inferred type", "DB dtype", "Null count", "Null %", "Frequency"]),
        *numeric_section,
        *stage2_section,
    ]


# ── 6. Clear results (< New Run link in results panel) ────────────────────

@callback(
    Output("results-store", "data", allow_duplicate=True),
    Input("btn-clear-results", "n_clicks"),
    prevent_initial_call=True,
)
def clear_results(_):
    return None


# ── 7. Toggle ingestion form ↔ completion state ────────────────────────────

@callback(
    Output("ingestion-form", "style"),
    Output("ingestion-done", "style"),
    Output("ingestion-done-text", "children"),
    Input("results-store", "data"),
)
def toggle_ingestion_panel(data):
    if data:
        source = data.get("source", {})
        if source.get("table"):
            text = f"Table: {source['table']}"
        elif source.get("file"):
            text = f"File: {Path(source['file']).name}"
        else:
            text = f"Run: {data.get('run_id', '')}"
        return _HIDE, _SHOW, text
    return _SHOW, _HIDE, ""


# ── 8. "Load Different Dataset" button in completion state ─────────────────

@callback(
    Output("results-store", "data", allow_duplicate=True),
    Input("btn-new-run", "n_clicks"),
    prevent_initial_call=True,
)
def new_dataset(_):
    return None


# ── 9. Populate past-runs list (updates after every run + on page load) ────

@callback(
    Output("past-runs-list", "children"),
    Input("results-store", "data"),
)
def populate_past_runs(_):
    try:
        resp = requests.get(f"{API_URL}/runs", timeout=5)
        if not resp.ok:
            return html.P("Could not load runs.", className="text-muted small mb-0")
        runs = resp.json().get("runs", [])
        if not runs:
            return html.P("No runs yet.", className="text-muted small mb-0")

        items = []
        for r in runs:
            source = r.get("source", {})
            label = (source.get("table")
                     or Path(source.get("file", "")).name
                     or r["run_id"])
            badge = dbc.Badge("S2", color="success", className="ms-1") if r.get("has_stage2") else None
            items.append(
                html.Div([
                    dbc.Button(
                        [html.Div([label, badge], className="small fw-semibold text-truncate"),
                         html.Div(f"{r['rows']:,} rows x {r['cols']} cols",
                                  className="text-muted", style={"fontSize": "11px"})],
                        id={"type": "btn-past-run", "run_id": r["run_id"]},
                        color="link",
                        size="sm",
                        className="text-start p-1 text-decoration-none w-100",
                    ),
                ], className="border-bottom")
            )
        return items
    except Exception:
        return html.P("Could not reach API.", className="text-muted small mb-0")


# ── 10. Load a past run from disk (no DB, reads local JSON) ────────────────

@callback(
    Output("results-store", "data", allow_duplicate=True),
    Input({"type": "btn-past-run", "run_id": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def load_past_run(n_clicks_list):
    if not ctx.triggered_id or not any(v for v in n_clicks_list if v):
        return no_update
    run_id = ctx.triggered_id["run_id"]
    try:
        resp = requests.get(f"{API_URL}/runs/{run_id}/summary", timeout=10)
        if resp.ok:
            return resp.json()
    except Exception:
        pass
    return no_update


def _render_stage2(stage2: dict) -> list:
    dp = stage2.get("decision_payload", {})
    if not dp:
        return []

    _sh = {"color": "#94a3b8", "fontSize": "0.78rem", "textTransform": "uppercase", "letterSpacing": "0.06em"}

    sections = [
        html.Hr(className="my-4"),
        html.H5([html.I(className="bi bi-search", style={"marginRight": "7px", "color": "#6366f1"}),
                 "Stage 2 — Pre-clean EDA"],
                className="fw-semibold mb-3", style={"color": "#c7d2fe"}),
    ]

    missing = dp.get("missing", {})
    if missing:
        missing_rows = [
            {"Column": col, "Null %": f"{v['pct']}%", "Max consecutive": v["max_consecutive"], "Pattern": v["pattern"]}
            for col, v in missing.items()
        ]
        sections += [
            html.H6([html.I(className="bi bi-exclamation-triangle", style={"marginRight": "5px"}), "Missing Data"],
                    className="fw-semibold mb-2", style=_sh),
            _datatable(missing_rows, ["Column", "Null %", "Max consecutive", "Pattern"]),
        ]

    outliers = dp.get("outliers", {})
    if outliers:
        outlier_rows = [
            {"Column": col, "IQR outlier %": f"{v['iqr_pct']}%", "Z-score outlier %": f"{v['zscore_pct']}%", "MAD outlier %": f"{v['mad_pct']}%"}
            for col, v in outliers.items()
        ]
        sections += [
            html.H6([html.I(className="bi bi-diagram-3", style={"marginRight": "5px"}), "Outlier Analysis"],
                    className="fw-semibold mb-2 mt-3", style=_sh),
            _datatable(outlier_rows, ["Column", "IQR outlier %", "Z-score outlier %", "MAD outlier %"]),
        ]

    flags = []
    dupes = dp.get("duplicates", {})
    if dupes.get("rows", 0) > 0:
        flags.append(dbc.Badge(f"{dupes['rows']} duplicate rows", color="warning", className="me-2 mb-1"))
    if dupes.get("timestamps", 0) > 0:
        flags.append(dbc.Badge(f"{dupes['timestamps']} duplicate timestamps", color="warning", className="me-2 mb-1"))
    for issue in dp.get("dtype_issues", []):
        flags.append(dbc.Badge(f"{issue['col']}: {issue['issue']}", color="danger", className="me-2 mb-1"))
    for col in dp.get("constant_cols", []):
        flags.append(dbc.Badge(f"{col}: constant/near-constant", color="secondary", className="me-2 mb-1"))
    for bp in dp.get("breakpoints", []):
        flags.append(dbc.Badge(f"Structural break: {bp}", color="info", className="me-2 mb-1"))

    if flags:
        sections += [
            html.H6([html.I(className="bi bi-flag", style={"marginRight": "5px"}), "Flags"],
                    className="fw-semibold mb-2 mt-3", style=_sh),
            html.Div(flags),
        ]
    elif not missing and not outliers:
        sections.append(dbc.Alert(
            [html.I(className="bi bi-check-circle-fill", style={"marginRight": "6px"}), "No quality issues detected."],
            color="success", className="mt-2",
        ))

    return sections


def _datatable(rows, columns):
    return dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in columns],
        style_table={"overflowX": "auto", "borderRadius": "10px", "overflow": "hidden"},
        style_cell={
            "fontSize": "12px",
            "padding": "8px 12px",
            "textAlign": "left",
            "backgroundColor": "#161b2e",
            "color": "#e2e8f0",
            "border": "none",
            "borderBottom": "1px solid rgba(255,255,255,0.05)",
            "fontFamily": "Inter, sans-serif",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_header={
            "fontWeight": "600",
            "backgroundColor": "#1e2235",
            "color": "#64748b",
            "fontSize": "10px",
            "textTransform": "uppercase",
            "letterSpacing": "0.06em",
            "border": "none",
            "borderBottom": "1px solid rgba(255,255,255,0.10)",
            "fontFamily": "Inter, sans-serif",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#1a1f30"},
        ],
        page_size=25,
    )


def _fmt(v: float) -> str:
    return f"{v:,.4f}" if abs(v) < 1e6 else f"{v:,.2f}"
