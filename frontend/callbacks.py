"""Dash callbacks — all interactivity lives here.

Hard-won rules this file follows (violating either resurfaces real bugs):

1. **Tab switching is clientside-only.** All six panes stay mounted in the layout;
   ``switch_pane`` just toggles ``style.display``. Server callbacks re-render pane
   *bodies* only when ``results-store`` changes. Rebuilding tab content on tab switch
   (the old ``render_tab`` pattern) unmounted/remounted components constantly, which
   triggered rule 2 and 3 bugs on every switch.

2. **Every button callback guards on ``n_clicks``.** Dash fires a callback whose Input
   component is inserted by another callback's render even with
   ``prevent_initial_call=True``, unless all of its Outputs are inside the same inserted
   chunk. Unguarded, the reset buttons fired on mount and wiped the store (and their
   ``_cancel_running()`` call killed in-flight jobs).

3. **Never use pattern-matching ids in a ``running=`` spec.** When a callback with
   ALL-pattern inputs fires because those components were added/removed by a re-render,
   ``changedPropIds`` is empty and the Dash 4.3.0 renderer crashes in ``replacePMC``
   ("Cannot read properties of undefined (reading 'run_id')"). The past-runs lock
   targets the list container's plain string id instead — string-id ``running`` targets
   are safe even while unmounted (the renderer explicitly tolerates them).
"""
import base64
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import requests
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, dash_table, no_update, callback, ALL, ctx, clientside_callback

from frontend import fcst_eda_view, intent_view, model_select_view, results_view, training_view, ui
from frontend.layout import RUNS_LIST_STYLE, RUNS_LIST_STYLE_LOCKED, TAB_VALUES

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# CSS variable, not a literal color — resolves per current theme so metric numbers stay
# legible in both dark and light mode instead of silently inheriting an invisible color.
_TEXT = "var(--bs-body-color)"

_SHOW = {"display": "block"}
_HIDE = {"display": "none"}

# "Forecast until <date>" → integer horizon. pd.Period subtraction counts whole periods
# regardless of intra-period position (mid-month vs month-start can't skew the count).
_PERIOD_FREQ = {"hourly": "h", "daily": "D", "weekly": "W",
                "monthly": "M", "quarterly": "Q", "yearly": "Y"}


def _periods_until(data_end, target_date, freq_label):
    f = _PERIOD_FREQ.get(freq_label or "")
    if not (f and data_end and target_date):
        return None
    try:
        n = (pd.Period(str(target_date)[:10], f) - pd.Period(str(data_end)[:19], f)).n
        return max(int(n), 1)
    except Exception:
        return None


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


# ── Pane switching (clientside — zero server round-trip) ─────────────────────
# All panes stay mounted; only style.display changes. This is what makes tab switching
# instant and free, and it's also load-bearing for correctness (see module docstring).

clientside_callback(
    f"""
    function(tab) {{
        const order = {json.dumps(TAB_VALUES)};
        return order.map(t => (t === tab ? {{display: 'block'}} : {{display: 'none'}}));
    }}
    """,
    [Output(v.replace("tab-", "pane-"), "style") for v in TAB_VALUES],
    Input("stage-tabs", "value"),
)


def _fetch_intent(run_id: str) -> dict | None:
    """Stage 2.5 auto-chain: detect + LLM-refine the forecast-intent suggestions right
    after pre-clean EDA. Failure is non-fatal — the Pipeline Setup tab has a Detect
    button as the retry path."""
    try:
        resp = requests.post(f"{API_URL}/runs/{run_id}/forecast-intent", timeout=900)
        if resp.ok:
            return resp.json().get("intent")
    except Exception:
        pass
    return None


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
def connect_db(n_clicks, host, port, database, user, password):
    if not n_clicks:
        return no_update, no_update, no_update
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
def toggle_query(n_clicks, is_open):
    if not n_clicks:
        return no_update
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
    State("input-file-path", "value"),
    running=[(Output("btn-run", "disabled"), True, False)],
    prevent_initial_call=True,
)
def trigger_run(n_clicks, source, table, query, host, port, database, user, password,
                upload_contents, upload_filename, file_path):
    if not n_clicks:
        return no_update, no_update
    if source == "file":
        # A pasted local path wins: zero copies — the API streams straight from disk
        # (the upload route base64-inflates the whole file through browser + Dash RAM,
        # which is why it is capped at 100 MB).
        path = (file_path or "").strip().strip('"').strip("'")
        if path:
            p = Path(path)
            if not p.is_file():
                return no_update, dbc.Alert(
                    f"File not found: {path} — the path must exist on the machine "
                    "running this app.", color="warning", dismissable=True)
            if p.suffix.lower() not in (".csv", ".xlsx", ".xls", ".parquet"):
                return no_update, dbc.Alert(
                    "Unsupported file type — use .csv, .xlsx, .xls or .parquet.",
                    color="warning", dismissable=True)
            payload = {"source": "file", "file_path": str(p)}
        elif upload_contents:
            try:
                tmp_path = _save_upload(upload_contents, upload_filename)
            except Exception as e:
                return no_update, dbc.Alert(f"Could not process upload: {e}", color="danger", dismissable=True)
            payload = {"source": "file", "file_path": tmp_path}
        else:
            return no_update, dbc.Alert(
                "Provide a file: paste a local path (any size), or upload one up to "
                "100 MB.", color="warning", dismissable=True)

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
                # Auto-chain Stage 2.5: intent suggestions are ready by the time the
                # user opens the Pipeline Setup tab.
                intent = _fetch_intent(run_id)
                if intent:
                    full_data["_intent"] = {"suggestions": intent}
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


# ── 5. Data pane — everything driven by the store, in one round trip ────────
# Fires at page load (rebuilding the UI from session storage — this is what makes a
# refresh or past-run "resume" work) and on every store change. Tab switching does NOT
# re-run this.

@callback(
    Output("data-tab-results", "children"),
    Output("ingestion-loaded", "style"),
    Output("ingestion-form-wrap", "style"),
    Output("loaded-source-text", "children"),
    Output("past-runs-list", "children"),
    Input("results-store", "data"),
)
def render_data_pane(data):
    runs_list = _build_runs_list()
    if not data:
        return _no_data_placeholder(), _HIDE, _SHOW, "", runs_list
    return _render_data_results(data), _SHOW, _HIDE, _loaded_source_text(data), runs_list


@callback(
    Output("clean-tab-body", "children"),
    Input("results-store", "data"),
)
def render_clean_body(data):
    return _render_clean_tab(data)


@callback(
    Output("fcst-eda-tab-body", "children"),
    Input("results-store", "data"),
)
def render_fcst_eda_body(data):
    return fcst_eda_view.render_fcst_eda_tab(data)


@callback(
    Output("model-tab-body", "children"),
    Input("results-store", "data"),
)
def render_model_body(data):
    return model_select_view.render_model_tab(data)


_TRAIN_CFG_CACHE: dict = {}


def _training_default_profile() -> str:
    """The configured training.accuracy_profile, fetched once from the API (the Dash
    process never reads the API's settings.yaml directly)."""
    if not _TRAIN_CFG_CACHE:
        try:
            r = requests.get(f"{API_URL}/runs/training-config", timeout=3)
            if r.ok:
                _TRAIN_CFG_CACHE.update(r.json())
        except requests.RequestException:
            pass
    return _TRAIN_CFG_CACHE.get("accuracy_profile") or "balanced"


@callback(
    Output("training-tab-body", "children"),
    Input("results-store", "data"),
)
def render_training_body(data):
    return training_view.render_training_tab(data, default_profile=_training_default_profile())


@callback(
    Output("input-password", "value"),
    Input("results-store", "data"),
    prevent_initial_call=True,
)
def clear_password(data):
    """Once a dataset is loaded, the DB password has served its purpose — the backend
    has already pulled the data to a local parquet and disposed the connection. Wipe it
    from the form so it doesn't linger in the browser for the rest of the session."""
    if data:
        return ""
    return no_update


def _loaded_source_text(data):
    source = data.get("source", {})
    if source.get("table"):
        return f"Table: {source['table']}"
    if source.get("file"):
        return f"File: {Path(source['file']).name}"
    return f"Run: {data.get('run_id', '')}"


def _cicon(name, **style):
    return html.I(className=f"bi {name}", style={"marginRight": "5px", **style})


def _no_data_placeholder():
    return html.Div(
        [
            _cicon("bi-arrow-left-circle", fontSize="2rem", color="#334155",
                   display="block", marginBottom="12px", marginRight="0"),
            html.P("Select a data source and click Run Ingestion.",
                   style={"color": "#64748b", "fontSize": "0.9rem"}),
        ],
        className="text-center mt-5 pt-4",
    )


# ── Tab 1 right column: dataset metrics, schema, pre-clean EDA ─────────────

def _render_data_results(data):
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
        numeric_section = [ui.collapse_section(
            f"Numeric statistics ({len(numeric_rows)} columns)",
            _datatable(numeric_rows, ["Column", "Min", "Max", "Mean", "Median", "Std",
                                      "Skew", "Kurtosis", "IQR"]),
            icon="bi-bar-chart-line")]

    stage2 = data.get("_stage2", {})
    stage2_section = _render_stage2(stage2) if stage2 else []

    header = html.Div([
        html.H5(
            [_cicon("bi-database-check", color="#10b981", marginRight="7px"), "Dataset"],
            className="fw-semibold mb-0 section-title",
        ),
        dbc.Button(
            [html.I(className="bi bi-arrow-counterclockwise", style={"marginRight": "4px"}), "New Run"],
            id="btn-clear-results",
            color="outline-secondary",
            size="sm",
            className="text-decoration-none",
        ),
    ], className="d-flex justify-content-between align-items-center mb-3")

    return html.Div([
        header,
        shape_card,
        ui.collapse_section(
            f"Schema ({len(schema_rows)} columns)",
            _datatable(schema_rows, ["Column", "Inferred type", "DB dtype", "Null count",
                                     "Null %", "Frequency"]),
            icon="bi-table"),
        *numeric_section,
        *stage2_section,
        _cleaning_hint(data),
    ])


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


# ── 6. New Run / reset — clears the store; render_data_pane restores the form ──
# Both the right-panel "New Run" button and the left "Load Different Dataset" button
# reset everything. The n_clicks guards are load-bearing: btn-clear-results is rendered
# dynamically, so this callback also fires when the button MOUNTS (see module
# docstring) — without the guard that spurious fire wipes the store and cancels
# whatever job is running.

@callback(
    Output("results-store", "data", allow_duplicate=True),
    Output("stage-tabs", "value", allow_duplicate=True),
    Input("btn-clear-results", "n_clicks"),
    running=[(Output("btn-clear-results", "disabled"), True, False)],
    prevent_initial_call=True,
)
def clear_results(n_clicks):
    if not n_clicks:
        return no_update, no_update
    _cancel_running()
    return None, "tab-data"


@callback(
    Output("results-store", "data", allow_duplicate=True),
    Output("stage-tabs", "value", allow_duplicate=True),
    Input("btn-new-run", "n_clicks"),
    running=[(Output("btn-new-run", "disabled"), True, False)],
    prevent_initial_call=True,
)
def new_dataset(n_clicks):
    if not n_clicks:
        return no_update, no_update
    _cancel_running()
    return None, "tab-data"


# ── 9. Past-runs list ─────────────────────────────────────────────────────

def _build_runs_list() -> list:
    """Render past-run items with load + delete buttons. Called by render_data_pane
    (page load + every store change) and by delete_run."""
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
            s4_badge = dbc.Badge("S4", color="primary", className="ms-1") if r.get("has_stage4") else None
            s5_badge = dbc.Badge("S5", color="warning", className="ms-1") if r.get("has_stage5") else None
            items.append(
                html.Div([
                    dbc.Button(
                        [
                            html.Div([label, s2_badge, s3_badge, s4_badge, s5_badge],
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


# Both callbacks below lock the WHOLE past-runs list (via pointer-events on the always-
# mounted container) for the duration of the request, not just the button clicked. This
# closes a real race: a second click while a summary GET is still in flight can return
# LATER than a subsequent click's response, silently overwriting the store with the
# older run's data. Locking the container makes that overlap structurally impossible.
# Deliberately NOT a pattern-matching running spec — see rule 3 in the module docstring.
_RUNS_LIST_LOCK = [(Output("past-runs-list", "style"), RUNS_LIST_STYLE_LOCKED, RUNS_LIST_STYLE)]


@callback(
    Output("past-runs-list", "children", allow_duplicate=True),
    Input({"type": "btn-delete-run", "run_id": ALL}, "n_clicks"),
    running=_RUNS_LIST_LOCK,
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
# "Resume": /runs/{id}/summary returns metadata + _stage2 + _stage3 in exactly the
# shape the store uses, so writing it to the store re-renders every pane as if the
# stages had just run.

@callback(
    Output("results-store", "data", allow_duplicate=True),
    Output("cleaning-status", "children", allow_duplicate=True),
    Input({"type": "btn-past-run", "run_id": ALL}, "n_clicks"),
    running=_RUNS_LIST_LOCK,
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
        intent = _fetch_intent(run_id)
        if intent:
            new_store["_intent"] = {"suggestions": intent}
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
                className="fw-semibold mb-3 section-title"),
    ]

    missing = dp.get("missing", {})
    if missing:
        missing_rows = [
            {"Column": col, "Null %": f"{v['pct']}%", "Max consecutive": v["max_consecutive"], "Pattern": v["pattern"]}
            for col, v in missing.items()
        ]
        sections.append(ui.collapse_section(
            f"Missing data ({len(missing_rows)} columns)",
            _datatable(missing_rows, ["Column", "Null %", "Max consecutive", "Pattern"]),
            icon="bi-exclamation-triangle"))

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
        sections.append(ui.collapse_section(
            f"Outlier analysis ({len(outlier_rows)} columns)",
            [html.P(
                "Temporal % uses rolling-window IQR — significantly lower than IQR % "
                "signals seasonal inflation, not noise.",
                style={"fontSize": "0.72rem", "color": "#64748b", "marginBottom": "6px"}),
             _datatable(outlier_rows, ["Column", "IQR %", "Z-score %", "MAD %", "Temporal %"])],
            icon="bi-diagram-3"))

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


# ── 11. Confirm intent → clean → validate → forecast EDA (one chain) ─────────
# The Pipeline Setup tab's confirm button drives the whole pipeline through Stage 5.
# Outputs to cleaning-status (persistent in layout) instead of any dynamically rendered
# component, so the store update always reaches the pane renderers regardless of
# component lifecycle timing in Dash 4. On full success the UI lands on Forecast EDA.

@callback(
    Output("results-store", "data", allow_duplicate=True),
    Output("cleaning-status", "children"),
    Output("stage-tabs", "value", allow_duplicate=True),
    Input("btn-confirm-run", "n_clicks"),
    State("results-store", "data"),
    State("dropdown-intent-ts", "value"),
    State("dropdown-intent-target", "value"),
    State("dropdown-intent-agg", "value"),
    State("radio-intent-scope", "value"),
    State("dropdown-intent-group", "value"),
    State("dropdown-intent-exog", "value"),
    State("dropdown-intent-freq", "value"),
    State("input-intent-horizon", "value"),
    State("switch-use-llm", "value"),
    running=[(Output("btn-confirm-run", "disabled"), True, False)],
    prevent_initial_call=True,
)
def confirm_and_run(n_clicks, store_data, ts, target, agg, scope, group_cols,
                    exog, freq, horizon, use_llm):
    if not n_clicks or not store_data:
        return no_update, no_update, no_update
    run_id = store_data.get("run_id")
    if not run_id:
        return no_update, dbc.Alert("No active run found.", color="warning", className="mb-2"), no_update
    if not ts or not target:
        return no_update, dbc.Alert(
            "Confirm the timestamp and target columns first.",
            color="warning", dismissable=True, className="mb-2"), no_update
    if scope == "per_series" and not group_cols:
        return no_update, dbc.Alert(
            "Per-series scope needs at least one series-key column (or switch to Overall).",
            color="warning", dismissable=True, className="mb-2"), no_update

    body = {
        "timestamp_col": ts,
        "target_col": target,
        "agg": agg or "sum",
        "scope": scope or "aggregate",
        "group_cols": group_cols or [],
        "exog_cols": exog or [],
        "forecast_frequency": freq,
        "horizon": int(horizon) if horizon else None,
        "use_llm": bool(use_llm),
    }
    try:
        clean_resp = requests.post(f"{API_URL}/runs/{run_id}/clean", json=body,
                                   timeout=1800)  # 30 min — large datasets take time
        if clean_resp.status_code == 409:
            return no_update, dbc.Alert(
                "This request was cancelled by a newer action. Click the run button again.",
                color="warning", dismissable=True, className="mb-2"), no_update
        if not clean_resp.ok:
            detail = clean_resp.json().get("detail", "Cleaning failed")
            return no_update, dbc.Alert(
                f"Cleaning error: {detail}", color="danger", dismissable=True,
                className="mb-2"), no_update

        stage3_data = clean_resp.json()
        validate_resp = requests.post(f"{API_URL}/runs/{run_id}/validate", timeout=300)
        if validate_resp.ok:
            stage3_data["_validation"] = validate_resp.json()

        new_store = {**store_data, "_stage3": stage3_data}
        new_store["_intent"] = {
            **(store_data.get("_intent") or {}),
            "selections": {k: v for k, v in body.items() if k != "use_llm"},
        }
        # The cleaned parquet was just rewritten — earlier Stage 4-8 results describe
        # data that no longer exists (the backend purged them on disk too).
        for k in ("_stage4", "_stage5", "_stage6", "_stage7", "_stage8"):
            new_store.pop(k, None)

        eda_resp = requests.post(f"{API_URL}/runs/{run_id}/forecast-eda", json={},
                                 timeout=1800)
        if eda_resp.status_code == 409:
            return new_store, dbc.Alert(
                "Cleaning + validation finished, but forecast EDA was preempted by a "
                "newer action — re-run it from the Forecast EDA tab.",
                color="warning", dismissable=True, className="mb-2"), no_update
        if not eda_resp.ok:
            detail = eda_resp.json().get("detail", "Forecast EDA failed")
            return new_store, dbc.Alert(
                f"Cleaning + validation finished, but forecast EDA failed: {detail}",
                color="danger", dismissable=True, className="mb-2"), no_update

        result = eda_resp.json()
        # run() may refine the selections (e.g. force nunique for a non-numeric target)
        if result.get("selections"):
            new_store["_intent"]["selections"] = result["selections"]
        new_store["_stage4"] = {"eda": {"payload": result.get("payload"),
                                        "full": result.get("full")}}

        msel_resp = requests.post(f"{API_URL}/runs/{run_id}/model-select",
                                  json={"use_llm": bool(use_llm)}, timeout=300)
        if msel_resp.ok:
            new_store["_stage5"] = msel_resp.json().get("selection")
            return new_store, "", "tab-fcst-eda"
        # EDA succeeded — keep it, but surface the selection failure visibly (house
        # rule: a 409 must never look like "nothing happened").
        if msel_resp.status_code == 409:
            msel_msg = "model selection was preempted by a newer action"
        else:
            msel_msg = ("model selection failed: "
                        + msel_resp.json().get("detail", "unknown error"))
        return new_store, dbc.Alert(
            f"Pipeline finished through forecast EDA, but {msel_msg} — "
            "re-run it from the Model Select tab.",
            color="warning", dismissable=True, className="mb-2"), "tab-fcst-eda"

    except requests.exceptions.Timeout:
        return no_update, dbc.Alert(
            "The pipeline timed out after 30 minutes.", color="danger",
            dismissable=True, className="mb-2"), no_update
    except requests.exceptions.ConnectionError:
        return no_update, dbc.Alert(
            "Cannot reach API — is run_dev.bat running?", color="danger",
            dismissable=True, className="mb-2"), no_update
    except Exception as e:
        return no_update, dbc.Alert(str(e), color="danger", dismissable=True, className="mb-2"), no_update


# ── Tab 2: Cleaning ────────────────────────────────────────────────────────

def _llm_response_details(stage3: dict):
    """Collapsible, theme-matched view of the model's raw JSON response. Shown on
    success (proof the LLM actually produced the recipe) and on validation failures
    (so the user can see what the model returned that got rejected)."""
    resp = stage3.get("llm_response")
    if not resp:
        return None
    return html.Details(
        [
            html.Summary(
                [_cicon("bi-chat-square-text", color="#6366f1"), "View LLM response"],
                style={"cursor": "pointer", "fontSize": "0.78rem", "color": "#94a3b8",
                       "userSelect": "none"},
            ),
            html.Pre(
                json.dumps(resp, indent=2),
                style={
                    "backgroundColor": "#161b2e", "color": "#e2e8f0",
                    "fontSize": "0.72rem", "lineHeight": "1.5",
                    "padding": "10px 14px", "borderRadius": "10px", "marginTop": "6px",
                    "marginBottom": "0", "maxHeight": "320px", "overflowY": "auto",
                    "whiteSpace": "pre-wrap",
                    "border": "1px solid rgba(99, 102, 241, 0.25)",
                },
            ),
        ],
        className="mb-3 mt-n2",
    )


def _llm_status_banner(stage3: dict):
    """Surface whether the LLM produced the cleaning recipe, or why it fell back."""
    if not stage3:
        return None
    src = stage3.get("recipe_source", "unknown")
    err = stage3.get("recipe_error")
    model = stage3.get("llm_model")
    details = _llm_response_details(stage3)
    if src == "llm":
        return html.Div([
            dbc.Alert(
                [_cicon("bi-check-circle-fill", color="#10b981"),
                 html.Strong("LLM connected"),
                 html.Span(
                     f" — cleaning recipe generated by {model}." if model
                     else " — cleaning recipe generated by the model.",
                     style={"fontSize": "0.85rem", "marginLeft": "4px"})],
                color="success", className="py-2 mb-2",
            ),
            *( [details] if details else [] ),
        ])
    if src == "fallback" and err:
        return html.Div([
            dbc.Alert(
                [
                    html.Div([_cicon("bi-x-circle-fill", color="#ef4444"),
                              html.Strong(
                                  f"LLM failed ({model}) — used rule-based fallback" if model
                                  else "LLM connection failed — used rule-based fallback")],
                             className="d-flex align-items-center mb-1"),
                    html.Code(err, style={"fontSize": "0.75rem", "color": "#fca5a5",
                                          "wordBreak": "break-all"}),
                ],
                color="danger", className="py-2 mb-2",
            ),
            *( [details] if details else [] ),
        ])
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
    """Pipeline Setup: the pipeline's single user checkpoint. Intent form (Stage 2.5
    suggestions, LLM-refined) on top; cleaning recipe/results + validation gate below
    once the confirm chain has run."""
    if not data or "_stage2" not in data:
        return html.Div(
            [_cicon("bi-info-circle", fontSize="2rem", color="#334155",
                    display="block", marginBottom="12px", marginRight="0"),
             html.P("Run pre-clean EDA on the Data tab first.",
                    style={"color": "#64748b", "fontSize": "0.9rem"})],
            className="text-center mt-5 pt-4",
        )

    intent = data.get("_intent") or {}
    suggestions = intent.get("suggestions")
    stage3 = data.get("_stage3", {})

    header = html.Div([
        html.Span([_cicon("bi-sliders"), "Pipeline Setup"],
                  className="fw-semibold section-title"),
    ], className="mb-3")

    if not suggestions:
        # Auto-detection after Stage 2 failed or this is an older run — offer it here.
        return html.Div([
            header,
            intent_view.detect_prompt_card(),
            _llm_status_banner(stage3),
            *(_render_stage3(stage3) if stage3 else []),
        ])

    return html.Div([
        header,
        intent_view.llm_intent_banner(suggestions),
        intent_view.build_intent_form(suggestions, intent.get("selections"),
                                      already_run=bool(stage3)),
        _llm_status_banner(stage3),
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

    # Target-total preservation — a level-mutating recipe (clipping a sum-target) shows
    # up here as a large drift, red-highlighted. Only rendered when the cleaner tracked
    # a target (intent-aware runs).
    drift = stage3.get("target_total_drift_pct")
    if stage3.get("target_total_before") is not None and drift is not None:
        drift_bad = abs(drift) > 5
        drift_color = "#e66767" if drift_bad else "#059669"
        sections.append(dbc.Card(dbc.CardBody(html.Div([
            html.Span(f"Target '{stage3.get('target_col')}' total: ",
                      style={"fontSize": "0.8rem", "color": "#94a3b8"}),
            html.Span(f"{stage3['target_total_before']:,.0f} → "
                      f"{(stage3.get('target_total_after') or 0):,.0f} ",
                      style={"fontSize": "0.85rem", "fontWeight": "600", "color": _TEXT}),
            html.Span(f"({drift:+}%)", style={"fontSize": "0.85rem", "fontWeight": "700",
                                              "color": drift_color}),
            html.Div("Cleaning shifted the target's total significantly — check the "
                     "outlier strategy on the target column.",
                     style={"fontSize": "0.72rem", "color": "#e66767", "marginTop": "3px"})
            if drift_bad else None,
        ])), className="mb-3",
            style={"borderLeft": f"3px solid {drift_color}"}))

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
        warn_fails = [
            name for name, c in checks.items()
            if c.get("severity") == "warning" and not c.get("passed", True)
        ]
        if passed_overall and warn_fails:
            badge_text, badge_color = f"PASSED — {len(warn_fails)} warning(s)", "warning"
        elif passed_overall:
            badge_text, badge_color = "Validation PASSED", "success"
        else:
            badge_text, badge_color = "Validation FAILED", "danger"
        gate_badge = dbc.Badge(badge_text, color=badge_color, className="ms-2")
        sections.append(
            html.H6(
                [html.I(className="bi bi-shield-check", style={"marginRight": "5px"}),
                 "Stage 3.5 — Validation Gate", gate_badge],
                className="fw-semibold mb-2 mt-3 d-flex align-items-center", style=_sh,
            )
        )

        def _result_cell(c):
            if c.get("passed"):
                return "✓ Pass"
            return "⚠ Warn" if c.get("severity") == "warning" else "✗ Fail"

        check_rows = [
            {
                "Check": name.replace("_", " ").title(),
                "Severity": c.get("severity", "blocking").title(),
                "Result": _result_cell(c),
                "Detail": c.get("detail", ""),
            }
            for name, c in checks.items()
        ]
        sections.append(_datatable(check_rows, ["Check", "Severity", "Result", "Detail"]))

    return sections


# ── Stage 2.5 / Stage 4 auxiliary callbacks ───────────────────────────────────
# Rendering lives in intent_view.py / fcst_eda_view.py; these handle interactivity.
# Same hard-won rules as the rest of this file: every button guards n_clicks (all these
# controls are dynamically rendered), running= uses string ids only, and status output
# goes to the persistent root-level cleaning-status div.

@callback(
    Output("results-store", "data", allow_duplicate=True),
    Output("cleaning-status", "children", allow_duplicate=True),
    Input("btn-intent-redetect", "n_clicks"),
    State("results-store", "data"),
    running=[(Output("btn-intent-redetect", "disabled"), True, False)],
    prevent_initial_call=True,
)
def run_intent_redetect(n_clicks, store_data):
    """Stage 2.5 on demand: (re-)detect the intent suggestions. Keeps the user's
    previously confirmed selections — only the suggestions refresh."""
    if not n_clicks or not store_data:
        return no_update, no_update
    run_id = store_data.get("run_id")
    if not run_id:
        return no_update, dbc.Alert("No active run found.", color="warning", className="mb-2")
    try:
        resp = requests.post(f"{API_URL}/runs/{run_id}/forecast-intent", timeout=900)
        if resp.status_code == 409:
            return no_update, dbc.Alert(
                "This request was cancelled by a newer action. Try again.",
                color="warning", dismissable=True, className="mb-2",
            )
        if not resp.ok:
            detail = resp.json().get("detail", "Intent detection failed")
            return no_update, dbc.Alert(detail, color="danger", dismissable=True, className="mb-2")
        intent = {**(store_data.get("_intent") or {}), "suggestions": resp.json().get("intent")}
        return {**store_data, "_intent": intent}, ""
    except requests.exceptions.ConnectionError:
        return no_update, dbc.Alert(
            "Cannot reach API — is run_dev.bat running?", color="danger", dismissable=True, className="mb-2"
        )
    except Exception as e:
        return no_update, dbc.Alert(str(e), color="danger", dismissable=True, className="mb-2")


# ── Date-based horizon pickers ────────────────────────────────────────────────
# The integer horizon stays the canonical field end-to-end; picking an end date just
# fills the horizon input (with a note showing the equivalence). Both callbacks have
# dynamically inserted Inputs → the `not date` guard is load-bearing (rule 2 above),
# and they skip when the computed count equals the current value so manual typing is
# never fought by a pane re-render.

@callback(
    Output("input-intent-horizon", "value"),
    Output("intent-horizon-note", "children"),
    Input("date-intent-horizon-end", "date"),
    Input("dropdown-intent-freq", "value"),
    State("results-store", "data"),
    State("input-intent-horizon", "value"),
    prevent_initial_call=True,
)
def sync_intent_horizon(date, freq, store_data, current):
    if not date or not store_data:
        return no_update, no_update
    data_end = ((store_data.get("_intent") or {}).get("suggestions") or {}).get("data_end")
    n = _periods_until(data_end, date, freq)
    if n is None or n == current:
        return no_update, no_update
    return n, f"= {n} {freq} period(s) after {str(data_end)[:10]}"


@callback(
    Output("input-fcst-horizon", "value"),
    Output("fcst-horizon-note", "children"),
    Input("date-fcst-horizon-end", "date"),
    Input("dropdown-fcst-freq", "value"),
    State("results-store", "data"),
    State("input-fcst-horizon", "value"),
    prevent_initial_call=True,
)
def sync_fcst_horizon(date, freq, store_data, current):
    if not date or not store_data:
        return no_update, no_update
    data_end = fcst_eda_view._data_end(store_data)
    n = _periods_until(data_end, date, freq)
    if n is None or n == current:
        return no_update, no_update
    return n, f"= {n} {freq} period(s) after {str(data_end)[:10]}"


@callback(
    Output("results-store", "data", allow_duplicate=True),
    Output("cleaning-status", "children", allow_duplicate=True),
    Input("btn-fcst-rerun", "n_clicks"),
    State("results-store", "data"),
    State("dropdown-fcst-freq", "value"),
    State("input-fcst-horizon", "value"),
    State("radio-fcst-scope", "value"),
    State("dropdown-fcst-exog", "value"),
    State("switch-use-llm", "value"),
    running=[(Output("btn-fcst-rerun", "disabled"), True, False)],
    prevent_initial_call=True,
)
def rerun_fcst_eda(n_clicks, store_data, freq, horizon, scope, exog, use_llm):
    """Stage-4-only re-run with scope / frequency / horizon / exog overrides — none of
    these affect cleaning (cleaner.py never reads them), so adjusting them must NOT force
    a re-clean. Model selection is re-chained afterward (fresh evidence invalidates the
    old decision), and the now-stale Stage 6/7 features+training are dropped."""
    if not n_clicks or not store_data:
        return no_update, no_update
    run_id = store_data.get("run_id")
    if not run_id:
        return no_update, dbc.Alert("No active run found.", color="warning", className="mb-2")
    # Pre-refactor runs render detect_prompt_card without the LLM switch — treat a
    # missing switch as "on" rather than silently disabling the LLM.
    use_llm = True if use_llm is None else bool(use_llm)
    body = {"forecast_frequency": freq, "horizon": int(horizon) if horizon else None,
            "scope": scope or None,
            "exog_cols": list(exog) if exog is not None else None}
    try:
        resp = requests.post(f"{API_URL}/runs/{run_id}/forecast-eda", json=body, timeout=1800)
        if resp.status_code == 409:
            return no_update, dbc.Alert(
                "This request was cancelled by a newer action. Click Re-run again.",
                color="warning", dismissable=True, className="mb-2",
            )
        if not resp.ok:
            detail = resp.json().get("detail", "Forecast EDA failed")
            return no_update, dbc.Alert(
                f"Forecast EDA error: {detail}", color="danger", dismissable=True, className="mb-2"
            )
        result = resp.json()
        new_store = {**store_data,
                     "_stage4": {"eda": {"payload": result.get("payload"),
                                         "full": result.get("full")}}}
        if result.get("selections"):
            new_store["_intent"] = {**(store_data.get("_intent") or {}),
                                    "selections": result["selections"]}
        # Fresh Stage 4 evidence invalidates the old model decision + features + training
        # + report (the backend purged them on disk) — drop them, re-select below.
        for k in ("_stage5", "_stage6", "_stage7", "_stage8"):
            new_store.pop(k, None)
        msel_resp = requests.post(f"{API_URL}/runs/{run_id}/model-select",
                                  json={"use_llm": use_llm}, timeout=300)
        if msel_resp.ok:
            new_store["_stage5"] = msel_resp.json().get("selection")
            return new_store, ""
        msel_msg = ("was preempted by a newer action" if msel_resp.status_code == 409
                    else "failed: " + msel_resp.json().get("detail", "unknown error"))
        return new_store, dbc.Alert(
            f"Forecast EDA finished, but model selection {msel_msg} — "
            "re-run it from the Model Select tab.",
            color="warning", dismissable=True, className="mb-2")
    except requests.exceptions.Timeout:
        return no_update, dbc.Alert(
            "Forecast EDA timed out after 30 minutes.", color="danger", dismissable=True, className="mb-2"
        )
    except requests.exceptions.ConnectionError:
        return no_update, dbc.Alert(
            "Cannot reach API — is run_dev.bat running?", color="danger", dismissable=True, className="mb-2"
        )
    except Exception as e:
        return no_update, dbc.Alert(str(e), color="danger", dismissable=True, className="mb-2")


@callback(
    Output("results-store", "data", allow_duplicate=True),
    Output("cleaning-status", "children", allow_duplicate=True),
    Input("btn-model-rerun", "n_clicks"),
    State("results-store", "data"),
    State("switch-model-llm", "value"),
    running=[(Output("btn-model-rerun", "disabled"), True, False)],
    prevent_initial_call=True,
)
def run_model_select(n_clicks, store_data, use_llm):
    """Stage-5-only re-run from the Model Select tab (e.g. to compare the rules-only
    pick against the LLM's, or after a chain failure). Selection is cheap — one JSON
    read plus at most one LLM call."""
    if not n_clicks or not store_data:
        return no_update, no_update
    run_id = store_data.get("run_id")
    if not run_id:
        return no_update, dbc.Alert("No active run found.", color="warning", className="mb-2")
    try:
        resp = requests.post(f"{API_URL}/runs/{run_id}/model-select",
                             json={"use_llm": bool(use_llm)}, timeout=300)
        if resp.status_code == 409:
            return no_update, dbc.Alert(
                "This request was cancelled by a newer action. Click Re-run Selection again.",
                color="warning", dismissable=True, className="mb-2",
            )
        if not resp.ok:
            detail = resp.json().get("detail", "Model selection failed")
            return no_update, dbc.Alert(
                f"Model selection error: {detail}", color="danger", dismissable=True, className="mb-2"
            )
        new_store = {**store_data, "_stage5": resp.json().get("selection")}
        return new_store, ""
    except requests.exceptions.Timeout:
        return no_update, dbc.Alert(
            "Model selection timed out.", color="danger", dismissable=True, className="mb-2"
        )
    except requests.exceptions.ConnectionError:
        return no_update, dbc.Alert(
            "Cannot reach API — is run_dev.bat running?", color="danger", dismissable=True, className="mb-2"
        )
    except Exception as e:
        return no_update, dbc.Alert(str(e), color="danger", dismissable=True, className="mb-2")


# ── Tab 5: Training (Stage 6 + 7) ────────────────────────────────────────────
# The user picks one or more eligible models; the Stage 5 pick is only the default. The
# estimate banner updates live as the selection changes; training itself is job-managed
# (cancellable) with a heavy-cost confirm gate.

@callback(
    Output("train-estimate-banner", "children"),
    Input("dropdown-train-models", "value"),
    Input("dropdown-train-profile", "value"),
    State("results-store", "data"),
    prevent_initial_call=True,
)
def update_train_estimate(models, profile, store_data):
    if not store_data:
        return no_update
    return training_view.build_estimate_banner(store_data, models or [], profile)


@callback(
    Output("graph-train-chart", "figure"),
    Output("train-insights-strip", "children"),
    Input("dropdown-train-chart-model", "value"),
    Input("dropdown-train-chart-series", "value"),
    State("results-store", "data"),
    prevent_initial_call=True,
)
def update_train_chart(model_id, series, store_data):
    """Chart any trained model (not just the best); with a series selected on a
    per-series run, fetch that single series' backtest+forecast from the API. The
    target-insights strip follows the model dropdown (aggregate view only — the
    insights are panel-level numbers, so a single-series chart leaves them as-is)."""
    store_data = store_data or {}
    report = store_data.get("_stage7") or {}
    if not model_id or not report:
        return no_update, no_update

    if series:
        run_id = report.get("run_id") or store_data.get("run_id")
        try:
            resp = requests.get(f"{API_URL}/runs/{run_id}/series-forecast",
                                params={"model": model_id, "series": series}, timeout=60)
            if resp.ok:
                js = resp.json()
                return training_view.forecast_figure(
                    {"backtest": js.get("backtest"), "forecast": js.get("forecast")},
                    None), no_update
        except requests.RequestException:
            pass
        return no_update, no_update

    entry = next((e for e in report.get("results") or [] if e["model"] == model_id), None)
    if not entry:
        return no_update, no_update
    return (training_view.forecast_figure(entry, training_view.history_for(store_data, report)),
            training_view._insights_strip(entry))


@callback(
    Output("results-store", "data", allow_duplicate=True),
    Output("cleaning-status", "children", allow_duplicate=True),
    Output("stage-tabs", "value", allow_duplicate=True),
    Input("btn-train", "n_clicks"),
    State("results-store", "data"),
    State("dropdown-train-models", "value"),
    State("switch-train-confirm", "value"),
    State("dropdown-train-profile", "value"),
    running=[(Output("btn-train", "disabled"), True, False)],
    prevent_initial_call=True,
)
def run_training(n_clicks, store_data, models, confirm, profile):
    """Stages 6+7 for the user-selected models. Lands on the Training tab on success; a
    heavy-cost 400 tells the user to enable the confirm switch and retry."""
    if not n_clicks or not store_data:
        return no_update, no_update, no_update
    run_id = store_data.get("run_id")
    if not run_id:
        return no_update, dbc.Alert("No active run found.", color="warning", className="mb-2"), no_update
    if not models:
        return no_update, dbc.Alert(
            "Select at least one model to train.", color="warning",
            dismissable=True, className="mb-2"), no_update
    body = {"models": models, "confirm_heavy": bool(confirm), "profile": profile or None}
    try:
        resp = requests.post(f"{API_URL}/runs/{run_id}/train", json=body, timeout=3600)
        if resp.status_code == 409:
            return no_update, dbc.Alert(
                "This request was cancelled by a newer action. Click Train again.",
                color="warning", dismissable=True, className="mb-2"), no_update
        if resp.status_code == 400:
            detail = resp.json().get("detail")
            # The heavy-cost gate returns a dict detail carrying the estimate.
            if isinstance(detail, dict) and detail.get("needs_confirm"):
                est = detail.get("estimate", {})
                return no_update, dbc.Alert(
                    [_cicon("bi-exclamation-triangle"),
                     f"{detail.get('message', 'Heavy training request.')} "
                     f"(~{est.get('units')} units across {est.get('n_series')} series). "
                     "Enable “Confirm — train anyway” and click Train again."],
                    color="warning", dismissable=True, className="mb-2"), no_update
            msg = detail if isinstance(detail, str) else "Training request rejected."
            return no_update, dbc.Alert(msg, color="danger", dismissable=True, className="mb-2"), no_update
        if not resp.ok:
            detail = resp.json().get("detail", "Training failed")
            return no_update, dbc.Alert(
                f"Training error: {detail}", color="danger", dismissable=True,
                className="mb-2"), no_update
        payload = resp.json()
        new_store = {**store_data, "_stage7": payload.get("report")}
        if payload.get("feature_report"):
            new_store["_stage6"] = payload["feature_report"]
        # A retrain invalidates the Stage 8 narrative (the route deleted it on disk).
        new_store.pop("_stage8", None)
        return new_store, "", "tab-training"
    except requests.exceptions.Timeout:
        return no_update, dbc.Alert(
            "Training timed out after 60 minutes.", color="danger",
            dismissable=True, className="mb-2"), no_update
    except requests.exceptions.ConnectionError:
        return no_update, dbc.Alert(
            "Cannot reach API — is run_dev.bat running?", color="danger",
            dismissable=True, className="mb-2"), no_update
    except Exception as e:
        return no_update, dbc.Alert(str(e), color="danger", dismissable=True, className="mb-2"), no_update


# ── Tab 6: Results (Stage 8) ─────────────────────────────────────────────────
# Opt-in LLM insight report + forecast download. The report POST is NOT job-managed
# server-side (network-bound; must never preempt a running training job).

@callback(
    Output("results-tab-body", "children"),
    Input("results-store", "data"),
)
def render_results_body(data):
    return results_view.render_results_tab(data)


@callback(
    Output("results-store", "data", allow_duplicate=True),
    Output("cleaning-status", "children", allow_duplicate=True),
    Input("btn-generate-report", "n_clicks"),
    State("results-store", "data"),
    State("dropdown-results-model", "value"),
    running=[(Output("btn-generate-report", "disabled"), True, False)],
    prevent_initial_call=True,
)
def generate_results_report(n_clicks, store_data, model):
    """Stage 8 on demand. LLM-only: a failure surfaces verbatim, there is no fallback
    report. Success merges _stage8 into the store (persisted server-side too)."""
    if not n_clicks or not store_data:
        return no_update, no_update
    run_id = store_data.get("run_id")
    if not run_id or not model:
        return no_update, dbc.Alert("Pick a trained model first.", color="warning",
                                    dismissable=True, className="mb-2")
    try:
        resp = requests.post(f"{API_URL}/runs/{run_id}/report",
                             json={"model": model}, timeout=300)
        if not resp.ok:
            detail = resp.json().get("detail", "Report generation failed")
            color = "danger" if resp.status_code == 502 else "warning"
            return no_update, dbc.Alert(str(detail), color=color, dismissable=True,
                                        className="mb-2")
        return {**store_data, "_stage8": resp.json().get("report")}, ""
    except requests.exceptions.Timeout:
        return no_update, dbc.Alert(
            "Report generation timed out — try again or switch llm_report to a faster "
            "model.", color="danger", dismissable=True, className="mb-2")
    except requests.exceptions.ConnectionError:
        return no_update, dbc.Alert(
            "Cannot reach API — is run_dev.bat running?", color="danger",
            dismissable=True, className="mb-2")
    except Exception as e:
        return no_update, dbc.Alert(str(e), color="danger", dismissable=True, className="mb-2")


@callback(
    Output("download-forecast", "data"),
    Output("cleaning-status", "children", allow_duplicate=True),
    Input("btn-download-csv", "n_clicks"),
    Input("btn-download-parquet", "n_clicks"),
    State("results-store", "data"),
    State("dropdown-results-model", "value"),
    State("switch-download-series", "value"),
    running=[(Output("btn-download-csv", "disabled"), True, False),
             (Output("btn-download-parquet", "disabled"), True, False)],
    prevent_initial_call=True,
)
def download_forecast(n_csv, n_pq, store_data, model, series_level):
    """Fetch the forecast file server-side (keeps all API access out of the browser) and
    hand it to the root-level dcc.Download."""
    if not (n_csv or n_pq) or not store_data or not ctx.triggered_id:
        return no_update, no_update
    run_id = store_data.get("run_id")
    if not run_id or not model:
        return no_update, dbc.Alert("Pick a trained model first.", color="warning",
                                    dismissable=True, className="mb-2")
    fmt = "parquet" if ctx.triggered_id == "btn-download-parquet" else "csv"
    level = "series" if series_level else "aggregate"
    try:
        resp = requests.get(f"{API_URL}/runs/{run_id}/forecast-file",
                            params={"model": model, "fmt": fmt, "level": level},
                            timeout=120)
        if not resp.ok:
            detail = resp.json().get("detail", "Download failed")
            return no_update, dbc.Alert(str(detail), color="warning", dismissable=True,
                                        className="mb-2")
        fname = f"{run_id}_{model}_forecast_{level}.{fmt}"
        return dcc.send_bytes(resp.content, fname), ""
    except requests.exceptions.ConnectionError:
        return no_update, dbc.Alert(
            "Cannot reach API — is run_dev.bat running?", color="danger",
            dismissable=True, className="mb-2")
    except Exception as e:
        return no_update, dbc.Alert(str(e), color="danger", dismissable=True, className="mb-2")
