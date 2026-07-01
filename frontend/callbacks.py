"""Dash callbacks — all interactivity lives here."""
import base64
import os
import tempfile
from pathlib import Path

import requests
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, dash_table, no_update, callback, ALL, ctx, clientside_callback

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# CSS variable, not a literal color — resolves per current theme so metric numbers stay
# legible in both dark and light mode instead of silently inheriting an invisible color.
_TEXT = "var(--bs-body-color)"


def _cancel_running():
    """Ask the API to terminate any heavy stage still running for a previous dataset.
    Called whenever the user starts a new run or switches datasets so old work stops
    immediately instead of pinning the CPU in the background."""
    try:
        requests.post(f"{API_URL}/runs/cancel", timeout=5)
    except Exception:
        pass


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
    Input("btn-connect", "n_clicks"),
    State("input-host", "value"),
    State("input-port", "value"),
    State("input-database", "value"),
    State("input-user", "value"),
    State("input-password", "value"),
    running=[(Output("btn-connect", "disabled"), True, False)],
    prevent_initial_call=True,
)
def connect_db(_, host, port, database, user, password):
    missing = [name for name, val in [
        ("Host", host), ("Database", database), ("Username", user), ("Password", password)
    ] if not val or not str(val).strip()]
    if missing:
        alert = dbc.Alert(f"Fill in: {', '.join(missing)}", color="warning", className="mb-2 py-2 small")
        return [], "Connect first…", alert

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
            return options, placeholder, alert
        detail = resp.json().get("detail", "Connection failed")
        alert = dbc.Alert(detail, color="danger", className="mb-2 py-2 small")
        return [], "Connect first…", alert
    except requests.exceptions.ConnectionError:
        alert = dbc.Alert("Cannot reach API — is run_dev.bat running?", color="danger", className="mb-2 py-2 small")
        return [], "Connect first…", alert
    except Exception as e:
        alert = dbc.Alert(str(e), color="danger", className="mb-2 py-2 small")
        return [], "Connect first…", alert


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
    running=[(Output("btn-run", "disabled"), True, False)],
    prevent_initial_call=True,
)
def trigger_run(_, source, table, query, host, port, database, user, password,
                upload_contents, upload_filename):
    if source == "file":
        if not upload_contents:
            return no_update, dbc.Alert(
                "Upload a file first — drag & drop or click Browse.", color="warning", dismissable=True
            )
        try:
            tmp_path = _save_upload(upload_contents, upload_filename)
        except Exception as e:
            return no_update, dbc.Alert(f"Could not process upload: {e}", color="danger", dismissable=True)
        payload = {"source": "file", "file_path": tmp_path}

    else:  # database
        if not all([host, database, user, password]):
            return no_update, dbc.Alert(
                "Fill in all credential fields and click Connect first.",
                color="warning", dismissable=True,
            )
        if query and query.strip():
            payload = {"source": "database", "query": query.strip()}
        elif table:
            payload = {"source": "database", "table": table}
        else:
            return no_update, dbc.Alert(
                "Select a table from the dropdown or enter a custom query.",
                color="warning", dismissable=True,
            )
        payload["credentials"] = {
            "host": host.strip(),
            "port": int(port) if port else 5432,
            "database": database.strip(),
            "user": user.strip(),
            "password": password,
        }

    try:
        run_resp = requests.post(f"{API_URL}/runs", json=payload, timeout=1800)  # 30 min for large tables
        if not run_resp.ok:
            detail = run_resp.json().get("detail", "Unknown error")
            return no_update, dbc.Alert(f"Run failed: {detail}", color="danger", dismissable=True)

        run_data = run_resp.json()
        run_id = run_data["run_id"]

        meta_resp = requests.get(f"{API_URL}/runs/{run_id}/metadata", timeout=15)
        if not meta_resp.ok:
            return no_update, dbc.Alert(
                f"Run completed ({run_id}) but metadata fetch failed.", color="warning", dismissable=True
            )

        full_data = meta_resp.json()
        full_data["_summary"] = run_data

        # Auto-trigger Stage 2 immediately after Stage 1. If this fails or gets preempted,
        # say so explicitly — the "Run Pre-clean EDA" button on the Data tab lets the user
        # retry rather than silently ending up with no Stage 2 and no explanation.
        try:
            eda_resp = requests.post(f"{API_URL}/runs/{run_id}/pre-clean-eda", timeout=600)  # 10 min for large tables
            if eda_resp.ok:
                full_data["_stage2"] = eda_resp.json()
            else:
                detail = eda_resp.json().get("detail", "unknown error")
                return full_data, dbc.Alert(
                    f"Ingested {run_id}, but pre-clean EDA failed: {detail}. "
                    "Use the \"Run Pre-clean EDA\" button below to retry.",
                    color="warning", dismissable=True,
                )
        except Exception as e:
            return full_data, dbc.Alert(
                f"Ingested {run_id}, but pre-clean EDA failed: {e}. "
                "Use the \"Run Pre-clean EDA\" button below to retry.",
                color="warning", dismissable=True,
            )

        return full_data, dbc.Alert(
            f"Completed: {run_id}", color="success", dismissable=True, duration=6000
        )

    except requests.exceptions.Timeout:
        return no_update, dbc.Alert(
            "Request timed out after 30 minutes — the dataset may be too large to ingest fully. "
            "Set 'row_limit' in config/settings.yaml to cap rows (e.g. row_limit: 500000).",
            color="danger", dismissable=True,
        )
    except requests.exceptions.ConnectionError:
        return no_update, dbc.Alert(
            "Cannot reach API — is run_dev.bat running?", color="danger", dismissable=True
        )
    except Exception as e:
        return no_update, dbc.Alert(f"Unexpected error: {e}", color="danger", dismissable=True)


# ── 5. Render the active tab ───────────────────────────────────────────────
# tab-content is rebuilt whenever the active tab OR the store changes. All stage
# builders read from the persistent results-store, so switching tabs never loses state.

@callback(
    Output("tab-content", "children"),
    Input("stage-tabs", "value"),
    Input("results-store", "data"),
)
def render_tab(active_tab, data):
    if active_tab == "tab-clean":
        return _render_clean_tab(data)
    if active_tab in ("tab-fcst-eda", "tab-model", "tab-training", "tab-results"):
        return _placeholder_tab()
    return _render_data_tab(data)


def _cicon(name, **style):
    return html.I(className=f"bi {name}", style={"marginRight": "5px", **style})


def _placeholder_tab():
    return html.Div(
        [
            _cicon("bi-hourglass-split", fontSize="2rem", color="#334155",
                   display="block", marginBottom="12px", marginRight="0"),
            html.P("This stage is not implemented yet.",
                   style={"color": "#64748b", "fontSize": "0.9rem"}),
        ],
        className="text-center mt-5 pt-4",
    )


# ── Tab 1: Data & Pre-clean EDA ────────────────────────────────────────────

def _render_data_tab(data):
    left_col = dbc.Col(width=3, children=[
        _ingestion_card(data),
        _past_runs_card(),
    ])

    if not data:
        right_body = html.Div(
            [
                _cicon("bi-arrow-left-circle", fontSize="2rem", color="#334155",
                       display="block", marginBottom="12px", marginRight="0"),
                html.P("Select a data source and click Run Ingestion.",
                       style={"color": "#64748b", "fontSize": "0.9rem"}),
            ],
            className="text-center mt-5 pt-4",
        )
        return dbc.Row([left_col, dbc.Col(width=9, children=right_body)])

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
                                        "letterSpacing": "-0.02em", "color": _TEXT}),
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

    header = html.Div([
        html.H5(
            [_cicon("bi-database-check", color="#10b981", marginRight="7px"), "Dataset"],
            className="fw-semibold mb-0", style={"color": "#c7d2fe"},
        ),
        dbc.Button(
            [html.I(className="bi bi-arrow-counterclockwise", style={"marginRight": "4px"}), "New Run"],
            id="btn-clear-results",
            color="outline-secondary",
            size="sm",
            className="text-decoration-none",
        ),
    ], className="d-flex justify-content-between align-items-center mb-3")

    right_col = dbc.Col(width=9, children=[
        header,
        shape_card,
        html.H6([_cicon("bi-table"), "Schema"], className="mb-2 fw-semibold",
                style={"color": "#94a3b8", "fontSize": "0.78rem", "textTransform": "uppercase", "letterSpacing": "0.06em"}),
        _datatable(schema_rows, ["Column", "Inferred type", "DB dtype", "Null count", "Null %", "Frequency"]),
        *numeric_section,
        *stage2_section,
        _cleaning_hint(data),
    ])

    return dbc.Row([left_col, right_col])


def _cleaning_hint(data):
    """Below the schema/stats: either a retry button (pre-clean EDA hasn't run — this
    normally happens automatically after ingestion, but can fail or get preempted) or a
    prompt to move on to the Cleaning tab."""
    if "_stage2" not in data:
        return html.Div(
            [
                html.P(
                    "Pre-clean EDA hasn't run for this dataset yet.",
                    style={"color": "#94a3b8", "fontSize": "0.85rem", "marginBottom": "8px"},
                ),
                dcc.Loading(
                    dbc.Button(
                        [_cicon("bi-play-fill"), "Run Pre-clean EDA"],
                        id="btn-run-eda", color="primary", size="sm",
                    ),
                    type="circle", color="#6366f1", delay_show=100,
                    target_components={"cleaning-status": "children"},
                ),
            ],
            className="mt-4",
        )
    return dbc.Alert(
        [_cicon("bi-arrow-right-circle", color="#6366f1"),
         "Pre-clean EDA complete — open the ", html.Strong("Cleaning"), " tab to continue."],
        color="info", className="mt-4 py-2",
    )


def _lbl(text):
    return dbc.Label(
        text, className="fw-semibold small mb-1",
        style={"fontSize": "0.68rem", "letterSpacing": "0.07em",
               "textTransform": "uppercase", "color": "#94a3b8"},
    )


def _ingestion_card(data):
    """Left-column card: shows the input form until a run is loaded, then a compact
    'dataset loaded' state with a reset button."""
    if data:
        source = data.get("source", {})
        if source.get("table"):
            text = f"Table: {source['table']}"
        elif source.get("file"):
            text = f"File: {Path(source['file']).name}"
        else:
            text = f"Run: {data.get('run_id', '')}"
        body = html.Div([
            dbc.Alert(
                [
                    html.Div([
                        _cicon("bi-check-circle-fill", color="#10b981"),
                        html.Strong("Dataset loaded"),
                    ], className="d-flex align-items-center mb-1"),
                    html.Small(text, style={"color": "#6ee7b7", "fontSize": "0.78rem"}),
                ],
                color="success", className="py-2 mb-3",
            ),
            dbc.Button(
                [_cicon("bi-arrow-counterclockwise"), "Load Different Dataset"],
                id="btn-new-run",
                color="outline-secondary",
                size="sm",
                className="w-100",
            ),
        ])
    else:
        body = _ingestion_form()

    return dbc.Card([
        dbc.CardHeader([
            _cicon("bi-cloud-download", color="#6366f1"),
            "Data Source",
        ]),
        dbc.CardBody(body),
    ])


def _ingestion_form():
    return html.Div(id="ingestion-form", children=[
        _lbl("Data source"),
        dcc.RadioItems(
            id="source-radio",
            options=[
                {"label": "  PostgreSQL database", "value": "database"},
                {"label": "  CSV / Excel / Parquet file", "value": "file"},
            ],
            value="database",
            className="mb-3",
            inputStyle={"marginRight": "6px", "accentColor": "#6366f1"},
            labelStyle={"display": "block", "marginBottom": "6px",
                        "cursor": "pointer", "color": "#94a3b8", "fontSize": "0.85rem"},
        ),

        html.Div(id="section-db", children=[
            _lbl("Host"),
            dbc.Input(id="input-host", placeholder="192.168.1.10", size="sm", className="mb-2"),
            dbc.Row([
                dbc.Col([
                    _lbl("Port"),
                    dbc.Input(id="input-port", value="5432", type="number", size="sm", className="mb-2"),
                ], width=4),
                dbc.Col([
                    _lbl("Database"),
                    dbc.Input(id="input-database", placeholder="db_name", size="sm", className="mb-2"),
                ], width=8),
            ]),
            _lbl("Username"),
            dbc.Input(id="input-user", placeholder="postgres", size="sm", className="mb-2", autocomplete="username"),
            _lbl("Password"),
            dbc.Input(id="input-password", type="password", placeholder="password", size="sm",
                      className="mb-2", autocomplete="new-password"),

            dbc.Button(
                [_cicon("bi-plug-fill"), "Connect"],
                id="btn-connect", color="outline-secondary", size="sm",
                className="w-100 mb-2", disabled=False,
            ),
            dcc.Loading(
                children=html.Div(id="connection-alert"),
                type="circle", color="#6366f1", delay_show=150, className="mb-2",
            ),

            _lbl("Table"),
            dcc.Dropdown(
                id="dropdown-table", clearable=True, placeholder="Connect first…",
                className="mb-2", style={"fontSize": "0.85rem"},
            ),

            dbc.Button(
                [_cicon("bi-code-slash"), "Custom SQL query"],
                id="btn-toggle-query", color="link", size="sm",
                className="p-0 mb-1 text-decoration-none",
            ),
            dbc.Collapse(
                dbc.Textarea(
                    id="input-query",
                    placeholder="SELECT * FROM schema.table WHERE …",
                    style={"height": "90px", "fontSize": "12px", "fontFamily": "monospace"},
                    className="mb-2",
                ),
                id="collapse-query", is_open=False,
            ),
        ]),

        html.Div(id="section-file", style={"display": "none"}, children=[
            _lbl("Upload file"),
            dcc.Upload(
                id="upload-file",
                children=html.Div([
                    _cicon("bi-cloud-upload", fontSize="1.4rem", color="#4b5563",
                           display="block", marginBottom="4px", marginRight="0"),
                    html.Span("Drag & drop or ", style={"color": "#94a3b8"}),
                    html.A("browse", style={"color": "#6366f1", "cursor": "pointer"}),
                ], style={"paddingTop": "12px"}),
                style={
                    "width": "100%", "height": "80px", "borderWidth": "1.5px",
                    "borderStyle": "dashed", "borderRadius": "10px", "textAlign": "center",
                    "borderColor": "rgba(255,255,255,0.12)", "cursor": "pointer",
                    "fontSize": "0.82rem", "transition": "all 0.2s ease",
                },
                accept=".csv,.xlsx,.xls,.parquet", multiple=False, className="mb-1",
            ),
            html.Div(id="upload-filename",
                     style={"fontSize": "0.75rem", "color": "#94a3b8"}, className="mb-2"),
            html.P("Supported: .csv  .xlsx  .xls  .parquet",
                   style={"fontSize": "0.7rem", "color": "#64748b"}, className="mb-0"),
        ]),

        html.Hr(className="my-3"),

        dcc.Loading(
            children=[
                dbc.Button(
                    [_cicon("bi-play-fill"), "Run Ingestion"],
                    id="btn-run", color="primary", className="w-100", disabled=False,
                    style={"fontWeight": "600", "letterSpacing": "0.02em"},
                ),
                html.Div(id="alert-div", className="mt-2"),
            ],
            type="circle", color="#6366f1", delay_show=100,
        ),
    ])


def _past_runs_card():
    return dbc.Card(className="mt-3", children=[
        dbc.CardHeader([
            _cicon("bi-clock-history", color="#475569"),
            "Past Runs",
        ]),
        dbc.CardBody(
            id="past-runs-list",
            children=_build_runs_list(),
            style={"maxHeight": "280px", "overflowY": "auto", "padding": "0.5rem"},
        ),
    ])


# ── 6. New Run / reset — clears the store and returns to the first tab ─────
# Both the right-panel "New Run" button and the left "Load Different Dataset"
# button reset everything, which also re-locks the Cleaning tab (see gate_clean_tab).

@callback(
    Output("results-store", "data", allow_duplicate=True),
    Output("stage-tabs", "value", allow_duplicate=True),
    Input("btn-clear-results", "n_clicks"),
    running=[(Output("btn-clear-results", "disabled"), True, False)],
    prevent_initial_call=True,
)
def clear_results(_):
    _cancel_running()
    return None, "tab-data"


@callback(
    Output("results-store", "data", allow_duplicate=True),
    Output("stage-tabs", "value", allow_duplicate=True),
    Input("btn-new-run", "n_clicks"),
    running=[(Output("btn-new-run", "disabled"), True, False)],
    prevent_initial_call=True,
)
def new_dataset(_):
    _cancel_running()
    return None, "tab-data"


# ── 9. Past-runs list ─────────────────────────────────────────────────────

def _build_runs_list() -> list:
    """Render past-run items with load + delete buttons. Shared by two callbacks."""
    try:
        resp = requests.get(f"{API_URL}/runs", timeout=5)
        if not resp.ok:
            return [html.P("Could not load runs.", className="text-muted small mb-0",
                           style={"fontSize": "0.78rem"})]
        runs = resp.json().get("runs", [])
        if not runs:
            return [html.P("No runs yet.", className="text-muted small mb-0",
                           style={"fontSize": "0.78rem"})]

        items = []
        for r in runs:
            source = r.get("source", {})
            label = (source.get("table") or Path(source.get("file", "")).name or r["run_id"])
            s2_badge = dbc.Badge("S2", color="success", className="ms-1") if r.get("has_stage2") else None
            s3_badge = dbc.Badge("S3", color="info", className="ms-1") if r.get("has_stage3") else None
            items.append(
                html.Div([
                    dbc.Button(
                        [
                            html.Div([label, s2_badge, s3_badge],
                                     className="small fw-semibold text-truncate"),
                            html.Div(f"{r['rows']:,} rows × {r['cols']} cols",
                                     className="text-muted", style={"fontSize": "11px"}),
                        ],
                        id={"type": "btn-past-run", "run_id": r["run_id"]},
                        color="link", size="sm",
                        className="text-start p-1 text-decoration-none flex-grow-1",
                        style={"minWidth": 0, "overflow": "hidden"},
                    ),
                    dbc.Button(
                        html.I(className="bi bi-trash3"),
                        id={"type": "btn-delete-run", "run_id": r["run_id"]},
                        color="link", size="sm",
                        className="p-1 ms-1",
                        style={"color": "#ef4444", "flexShrink": "0"},
                        title=f"Delete {r['run_id']}",
                    ),
                ], className="d-flex align-items-center border-bottom")
            )
        return items
    except Exception:
        return [html.P("Could not reach API.", className="text-muted small mb-0",
                       style={"fontSize": "0.78rem"})]


# Both callbacks below disable EVERY past-run button (load + delete) for the duration of
# the request, not just the one clicked. This is what closes the "tab unlocks then re-locks"
# bug: without it, a second click (on the same or a different past run) while a summary GET
# is still in flight can return LATER than a subsequent click's response, silently
# overwriting the store with the older/wrong run's data a few seconds after it looked loaded.
# Disabling the whole list makes that overlap structurally impossible — a click can't even
# register while another load/delete is pending.
_PAST_RUN_RUNNING = [
    (Output({"type": "btn-past-run", "run_id": ALL}, "disabled"), True, False),
    (Output({"type": "btn-delete-run", "run_id": ALL}, "disabled"), True, False),
]


@callback(
    Output("past-runs-list", "children", allow_duplicate=True),
    Input({"type": "btn-delete-run", "run_id": ALL}, "n_clicks"),
    running=_PAST_RUN_RUNNING,
    prevent_initial_call=True,
)
def delete_run(n_clicks_list):
    if not ctx.triggered_id or not any(v for v in n_clicks_list if v):
        return no_update
    run_id = ctx.triggered_id["run_id"]
    try:
        requests.delete(f"{API_URL}/runs/{run_id}", timeout=10)
    except Exception:
        pass
    return _build_runs_list()


# ── 10. Load a past run from disk (no DB, reads local JSON) ────────────────
# Only updates the store; toggle_ingestion_panel handles hiding the form.

@callback(
    Output("results-store", "data", allow_duplicate=True),
    Output("cleaning-status", "children", allow_duplicate=True),
    Input({"type": "btn-past-run", "run_id": ALL}, "n_clicks"),
    running=_PAST_RUN_RUNNING,
    prevent_initial_call=True,
)
def load_past_run(n_clicks_list):
    if not ctx.triggered_id or not any(v for v in n_clicks_list if v):
        return no_update, no_update
    # Switching datasets stops any heavy stage still crunching for the current one.
    _cancel_running()
    run_id = ctx.triggered_id["run_id"]
    try:
        resp = requests.get(f"{API_URL}/runs/{run_id}/summary", timeout=10)
        if resp.ok:
            return resp.json(), ""
        detail = resp.json().get("detail", "Could not load this run.")
        return no_update, dbc.Alert(detail, color="danger", dismissable=True, className="mb-2")
    except Exception as e:
        return no_update, dbc.Alert(f"Could not load run: {e}", color="danger", dismissable=True, className="mb-2")


# ── Retry pre-clean EDA from the Data tab (auto-run after ingest can fail/get preempted) ──

@callback(
    Output("results-store", "data", allow_duplicate=True),
    Output("cleaning-status", "children", allow_duplicate=True),
    Input("btn-run-eda", "n_clicks"),
    State("results-store", "data"),
    running=[(Output("btn-run-eda", "disabled"), True, False)],
    prevent_initial_call=True,
)
def run_eda_retry(n_clicks, store_data):
    if not n_clicks or not store_data:
        return no_update, no_update
    run_id = store_data.get("run_id")
    if not run_id:
        return no_update, dbc.Alert("No active run found.", color="warning", className="mb-2")
    try:
        resp = requests.post(f"{API_URL}/runs/{run_id}/pre-clean-eda", timeout=600)
        if resp.status_code == 409:
            return no_update, dbc.Alert(
                "This request was cancelled by a newer action. Try again.",
                color="warning", dismissable=True, className="mb-2",
            )
        if not resp.ok:
            detail = resp.json().get("detail", "Pre-clean EDA failed")
            return no_update, dbc.Alert(detail, color="danger", dismissable=True, className="mb-2")
        new_store = {**store_data, "_stage2": resp.json()}
        return new_store, ""
    except requests.exceptions.ConnectionError:
        return no_update, dbc.Alert(
            "Cannot reach API — is run_dev.bat running?", color="danger", dismissable=True, className="mb-2"
        )
    except Exception as e:
        return no_update, dbc.Alert(str(e), color="danger", dismissable=True, className="mb-2")


def _render_stage2(stage2: dict) -> list:
    dp = stage2.get("decision_payload", {})
    if not dp:
        return []

    _sh = {"color": "#94a3b8", "fontSize": "0.78rem", "textTransform": "uppercase", "letterSpacing": "0.06em"}

    sections = [
        html.Hr(className="my-4"),
        html.H5([html.I(className="bi bi-search", style={"marginRight": "7px", "color": "#10b981"}),
                 "Pre-clean EDA"],
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
            {
                "Column": col,
                "IQR %": f"{v['iqr_pct']}%",
                "Z-score %": f"{v['zscore_pct']}%",
                "MAD %": f"{v['mad_pct']}%",
                "Temporal %": f"{v['temporal_pct']}%" if v.get("temporal_pct") is not None else "—",
            }
            for col, v in outliers.items()
        ]
        sections += [
            html.H6([html.I(className="bi bi-diagram-3", style={"marginRight": "5px"}), "Outlier Analysis"],
                    className="fw-semibold mb-2 mt-3", style=_sh),
            html.P(
                "Temporal % uses rolling-window IQR — significantly lower than IQR % signals seasonal inflation, not noise.",
                style={"fontSize": "0.72rem", "color": "#64748b", "marginBottom": "6px"},
            ),
            _datatable(outlier_rows, ["Column", "IQR %", "Z-score %", "MAD %", "Temporal %"]),
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


# ── 11. Stage 3: run cleaning ─────────────────────────────────────────────────
# Outputs to cleaning-status (persistent in layout) instead of btn-run-cleaning
# or cleaning-alert (both are dynamically rendered), so the store update always
# reaches render_results regardless of component lifecycle timing in Dash 4.

@callback(
    Output("results-store", "data", allow_duplicate=True),
    Output("cleaning-status", "children"),
    Input("btn-run-cleaning", "n_clicks"),
    State("results-store", "data"),
    State("dropdown-ts-confirm", "value"),
    State("switch-use-llm", "value"),
    running=[(Output("btn-run-cleaning", "disabled"), True, False)],
    prevent_initial_call=True,
)
def run_cleaning(n_clicks, store_data, ts_val, use_llm):
    if not n_clicks or not store_data:
        return no_update, no_update
    run_id = store_data.get("run_id")
    if not run_id:
        return no_update, dbc.Alert("No active run found.", color="warning", className="mb-2")

    try:
        clean_resp = requests.post(
            f"{API_URL}/runs/{run_id}/clean",
            json={"timestamp_col": ts_val, "use_llm": bool(use_llm)},
            timeout=1800,  # 30 min — large datasets can take time
        )
        if clean_resp.status_code == 409:
            return no_update, dbc.Alert(
                "This request was cancelled by a newer action. Click Run Cleaning again.",
                color="warning", dismissable=True, className="mb-2",
            )
        if not clean_resp.ok:
            detail = clean_resp.json().get("detail", "Cleaning failed")
            return no_update, dbc.Alert(f"Cleaning error: {detail}", color="danger", dismissable=True, className="mb-2")

        stage3_data = clean_resp.json()

        validate_resp = requests.post(f"{API_URL}/runs/{run_id}/validate", timeout=30)
        if validate_resp.ok:
            stage3_data["_validation"] = validate_resp.json()

        new_store = {**store_data, "_stage3": stage3_data}
        return new_store, ""

    except requests.exceptions.Timeout:
        return no_update, dbc.Alert(
            "Cleaning timed out after 5 minutes.", color="danger", dismissable=True, className="mb-2"
        )
    except requests.exceptions.ConnectionError:
        return no_update, dbc.Alert(
            "Cannot reach API — is run_dev.bat running?", color="danger", dismissable=True, className="mb-2"
        )
    except Exception as e:
        return no_update, dbc.Alert(str(e), color="danger", dismissable=True, className="mb-2")


# ── Tab 2: Cleaning ────────────────────────────────────────────────────────

def _llm_status_banner(stage3: dict):
    """Surface whether the LLM produced the recipe, or why it fell back."""
    if not stage3:
        return dbc.Alert(
            [_cicon("bi-info-circle"),
             "Cleaning has not been run yet. Confirm the timestamp column and run it below."],
            color="secondary", className="py-2 mb-3",
        )
    src = stage3.get("recipe_source", "unknown")
    err = stage3.get("recipe_error")
    if src == "llm":
        return dbc.Alert(
            [_cicon("bi-check-circle-fill", color="#10b981"),
             html.Strong("LLM connected"),
             html.Span(" — cleaning recipe generated by the model.",
                       style={"fontSize": "0.85rem", "marginLeft": "4px"})],
            color="success", className="py-2 mb-3",
        )
    if src == "fallback" and err:
        return dbc.Alert(
            [
                html.Div([_cicon("bi-x-circle-fill", color="#ef4444"),
                          html.Strong("LLM connection failed — used rule-based fallback")],
                         className="d-flex align-items-center mb-1"),
                html.Code(err, style={"fontSize": "0.75rem", "color": "#fca5a5",
                                      "wordBreak": "break-all"}),
            ],
            color="danger", className="py-2 mb-3",
        )
    if src == "fallback":
        return dbc.Alert(
            [_cicon("bi-cpu", color="#f59e0b"),
             html.Strong("Rule-based recipe"),
             html.Span(" — LLM was skipped for this run.",
                       style={"fontSize": "0.85rem", "marginLeft": "4px"})],
            color="warning", className="py-2 mb-3",
        )
    return dbc.Alert(
        [_cicon("bi-question-circle"), "Recipe source unknown (older run)."],
        color="secondary", className="py-2 mb-3",
    )


def _render_clean_tab(data):
    if not data or "_stage2" not in data:
        return html.Div(
            [_cicon("bi-info-circle", fontSize="2rem", color="#334155",
                    display="block", marginBottom="12px", marginRight="0"),
             html.P("Run pre-clean EDA on the Data tab first.",
                    style={"color": "#64748b", "fontSize": "0.9rem"})],
            className="text-center mt-5 pt-4",
        )

    datetime_cols = data.get("datetime_cols", [])
    stage3 = data.get("_stage3", {})
    recipe = stage3.get("recipe", {}) if stage3 else {}
    default_ts = recipe.get("timestamp_col") or (datetime_cols[0] if datetime_cols else None)
    ts_options = [{"label": c, "value": c} for c in datetime_cols]
    already_run = bool(stage3)

    controls = dbc.Card(
        dbc.CardBody([
            html.P([_cicon("bi-calendar-check"), "Confirm Cleaning Settings"],
                   className="fw-semibold mb-3", style={"color": "#c7d2fe"}),
            dbc.Row([
                dbc.Col([
                    html.Label("Timestamp Column", className="form-label",
                               style={"fontSize": "0.8rem", "color": "#94a3b8"}),
                    dcc.Dropdown(
                        id="dropdown-ts-confirm",
                        options=ts_options, value=default_ts,
                        clearable=False, style={"fontSize": "0.85rem"},
                    ),
                ], md=6),
                dbc.Col([
                    html.Label("LLM", className="form-label",
                               style={"fontSize": "0.8rem", "color": "#94a3b8"}),
                    dbc.Switch(
                        id="switch-use-llm", value=True,
                        label="Use LLM (off = rule-based only)",
                        className="mt-1",
                    ),
                ], md=6),
            ], className="mb-3"),
            dcc.Loading(
                dbc.Button(
                    [_cicon("bi-scissors"),
                     "Re-run Cleaning" if already_run else "Run Cleaning"],
                    id="btn-run-cleaning", color="primary", className="w-100",
                    disabled=default_ts is None,
                    style={"fontWeight": "600", "letterSpacing": "0.02em"},
                ),
                type="circle", color="#6366f1", delay_show=100,
                target_components={"cleaning-status": "children"},
            ),
        ]),
        className="mb-3",
        style={"background": "rgba(30, 41, 59, 0.7)",
               "border": "1px solid rgba(99, 102, 241, 0.3)"},
    )

    header = html.Div([
        html.Span([_cicon("bi-scissors"), "Cleaning"],
                  className="fw-semibold", style={"color": "#c7d2fe"}),
    ], className="mb-3")

    return html.Div([
        header,
        _llm_status_banner(stage3),
        controls,
        *(_render_stage3(stage3) if stage3 else []),
    ])


def _render_stage3(stage3: dict) -> list:
    if not stage3:
        return []

    _sh = {"color": "#94a3b8", "fontSize": "0.78rem", "textTransform": "uppercase", "letterSpacing": "0.06em"}

    rows_before = stage3.get("rows_before", "—")
    rows_after = stage3.get("rows_after", "—")
    row_loss_pct = stage3.get("row_loss_pct", "—")
    cols_dropped = stage3.get("cols_dropped", [])
    recipe = stage3.get("recipe", {})
    validation = stage3.get("_validation", {})

    sections = [
        html.Hr(className="my-4"),
        html.H6(
            [html.I(className="bi bi-clipboard-data", style={"marginRight": "6px"}),
             "Cleaning Results"],
            className="fw-semibold mb-3", style=_sh,
        ),
    ]

    # Before/after summary
    sections.append(
        dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col([
                html.Div([html.I(className="bi bi-table", style={"marginRight": "4px", "color": "#64748b", "fontSize": "0.7rem"}),
                          html.Span("Rows before", style={"fontSize": "0.65rem", "color": "#94a3b8", "textTransform": "uppercase"})],
                         className="d-flex align-items-center mb-1"),
                html.Div(f"{rows_before:,}" if isinstance(rows_before, int) else str(rows_before),
                         style={"fontSize": "1rem", "fontWeight": "700", "color": _TEXT}),
            ], width=3),
            dbc.Col([
                html.Div([html.I(className="bi bi-table", style={"marginRight": "4px", "color": "#64748b", "fontSize": "0.7rem"}),
                          html.Span("Rows after", style={"fontSize": "0.65rem", "color": "#94a3b8", "textTransform": "uppercase"})],
                         className="d-flex align-items-center mb-1"),
                html.Div(f"{rows_after:,}" if isinstance(rows_after, int) else str(rows_after),
                         style={"fontSize": "1rem", "fontWeight": "700", "color": _TEXT}),
            ], width=3),
            dbc.Col([
                html.Div([html.I(className="bi bi-percent", style={"marginRight": "4px", "color": "#64748b", "fontSize": "0.7rem"}),
                          html.Span("Row loss", style={"fontSize": "0.65rem", "color": "#94a3b8", "textTransform": "uppercase"})],
                         className="d-flex align-items-center mb-1"),
                html.Div(f"{row_loss_pct}%", style={"fontSize": "1rem", "fontWeight": "700", "color": _TEXT}),
            ], width=3),
            dbc.Col([
                html.Div([html.I(className="bi bi-columns-gap", style={"marginRight": "4px", "color": "#64748b", "fontSize": "0.7rem"}),
                          html.Span("Cols dropped", style={"fontSize": "0.65rem", "color": "#94a3b8", "textTransform": "uppercase"})],
                         className="d-flex align-items-center mb-1"),
                html.Div(str(len(cols_dropped)), style={"fontSize": "1rem", "fontWeight": "700", "color": _TEXT}),
            ], width=3),
        ])), className="mb-3", style={"borderLeft": "3px solid #6366f1"}),
    )

    # Cleaning recipe table
    col_recipes = recipe.get("columns", {})
    if col_recipes:
        recipe_rows = [
            {
                "Column": col,
                "Action": r.get("action", "—"),
                "Missing strategy": r.get("missing_strategy", "—"),
                "Outlier strategy": r.get("outlier_strategy", "—"),
                "Type fix": r.get("type_fix", "—"),
            }
            for col, r in col_recipes.items()
        ]
        sections += [
            html.H6([html.I(className="bi bi-list-check", style={"marginRight": "5px"}), "Cleaning Recipe"],
                    className="fw-semibold mb-2", style=_sh),
            _datatable(recipe_rows, ["Column", "Action", "Missing strategy", "Outlier strategy", "Type fix"]),
        ]

    # Validation gate results
    if validation:
        checks = validation.get("checks", {})
        passed_overall = validation.get("passed", False)
        gate_badge = dbc.Badge(
            "Validation PASSED" if passed_overall else "Validation FAILED",
            color="success" if passed_overall else "danger",
            className="ms-2",
        )
        sections.append(
            html.H6(
                [html.I(className="bi bi-shield-check", style={"marginRight": "5px"}),
                 "Stage 3.5 — Validation Gate", gate_badge],
                className="fw-semibold mb-2 mt-3 d-flex align-items-center", style=_sh,
            )
        )
        check_rows = [
            {
                "Check": name.replace("_", " ").title(),
                "Result": "✓ Pass" if c["passed"] else "✗ Fail",
                "Detail": c.get("detail", ""),
            }
            for name, c in checks.items()
        ]
        sections.append(_datatable(check_rows, ["Check", "Result", "Detail"]))

    return sections
