"""Dash page layout.

The body is a 3-section interface (Setup / Pipeline / Results). **All three section
panes are always mounted** — a clientside callback in callbacks.py toggles their
``style.display`` when the active section changes, so switching never touches the
server and never unmounts components.

This matters for correctness, not just speed: Dash fires callbacks whose Inputs are
dynamically added or removed by another callback's render, *ignoring*
``prevent_initial_call`` (and for pattern-matching ALL inputs the renderer's
``replacePMC`` crashes outright when such a fire meets a pattern-matching ``running=``
spec — "Cannot read properties of undefined (reading 'run_id')"). Keeping everything
mounted removes the whole class of bugs.

Section anatomy (all six pre-refactor body div ids SURVIVE, re-parented, so every
store-driven render callback keeps working):
- **Setup**: ingestion form + past runs (left) · dataset profile ``data-tab-results``
  + forecast-intent confirmation ``intent-form-body`` (right).
- **Pipeline**: live stage rail ``stage-rail`` + log ``pipeline-log`` (left) · four
  static ``ui.stage_card`` expanders wrapping ``clean-tab-body`` /
  ``fcst-eda-tab-body`` / ``model-tab-body`` / ``training-tab-body`` (right).
- **Results**: ``results-tab-body`` (leaderboard + charts + LLM report + downloads).

Tabs are always clickable — no disabled/locked state. When a section's prerequisite
hasn't run yet, its content shows a short line of text saying so.
"""
import dash_bootstrap_components as dbc
from dash import dcc, html

from frontend import ui

# Tab definitions: (label, tab-id/value, icon). Each tab value maps to a pane id via
# "tab-" → "pane-" (see the clientside pane switcher in callbacks.py).
_TABS = [
    ("Setup",    "tab-setup",    "bi-cloud-download"),
    ("Pipeline", "tab-pipeline", "bi-diagram-3"),
    ("Results",  "tab-results",  "bi-trophy"),
]

TAB_VALUES = [value for _, value, _ in _TABS]

# Past-runs list styles. The locked variant is applied via the `running=` argument of the
# load/delete callbacks so the whole list ignores clicks while a load or delete is in
# flight — a plain string-id target, because pattern-matching (ALL) `running=` outputs
# crash the Dash 4.3.0 renderer when the callback fires without a concrete trigger id.
RUNS_LIST_STYLE = {"maxHeight": "280px", "overflowY": "auto", "padding": "0.5rem"}
RUNS_LIST_STYLE_LOCKED = {**RUNS_LIST_STYLE, "pointerEvents": "none", "opacity": "0.55"}

_HIDDEN = {"display": "none"}


def _icon(name, **kwargs):
    return html.I(className=f"bi {name}", **kwargs)


def _cicon(name, **style):
    return html.I(className=f"bi {name}", style={"marginRight": "5px", **style})


def _lbl(text):
    return dbc.Label(
        text, className="fw-semibold small mb-1",
        style={"fontSize": "0.68rem", "letterSpacing": "0.07em",
               "textTransform": "uppercase", "color": "var(--ink-muted)"},
    )


def _tab_label(icon, text):
    return html.Span([_icon(icon, style={"marginRight": "8px"}), text])


def _build_tabs():
    """Static tab bar — every tab is always clickable. Content-level placeholders tell
    the user what to do first, rather than the tab itself being blocked."""
    return [
        dcc.Tab(
            label=_tab_label(icon, label),
            value=value,
            id=value,
            className="stage-tab",
            selected_className="stage-tab--selected",
        )
        for label, value, icon in _TABS
    ]


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
                        "cursor": "pointer", "color": "var(--ink-muted)", "fontSize": "0.85rem"},
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
            _lbl("Local file path (recommended for large files)"),
            dbc.Input(
                id="input-file-path", type="text",
                placeholder=r"e.g. C:\data\my_data.csv", className="mb-1",
                style={"fontSize": "0.82rem"},
            ),
            html.P("Read directly from disk — no upload, no size limit. The path must "
                   "exist on the machine running this app.",
                   style={"fontSize": "0.7rem", "color": "var(--ink-faint)"}, className="mb-2"),
            html.Div("— or —", className="mb-2 text-center",
                     style={"fontSize": "0.7rem", "color": "var(--ink-ghost)"}),
            _lbl("Upload file (up to 100 MB)"),
            dcc.Upload(
                id="upload-file",
                children=html.Div([
                    _cicon("bi-cloud-upload", fontSize="1.4rem", color="#4b5563",
                           display="block", marginBottom="4px", marginRight="0"),
                    html.Span("Drag & drop or ", style={"color": "var(--ink-muted)"}),
                    html.A("browse", style={"color": "#6366f1", "cursor": "pointer"}),
                ], style={"paddingTop": "12px"}),
                style={
                    "width": "100%", "height": "80px", "borderWidth": "1.5px",
                    "borderStyle": "dashed", "borderRadius": "10px", "textAlign": "center",
                    "borderColor": "rgba(255,255,255,0.12)", "cursor": "pointer",
                    "fontSize": "0.82rem", "transition": "all 0.2s ease",
                },
                accept=".csv,.xlsx,.xls,.parquet", multiple=False, className="mb-1",
                # a browser upload base64-encodes the WHOLE file in memory — large files
                # OOM'd the Dash process before ingest ever saw them. Oversize files are
                # silently ignored by dcc.Upload, so the copy below must carry the limit.
                max_size=100 * 1024 * 1024,
            ),
            html.Div(id="upload-filename",
                     style={"fontSize": "0.75rem", "color": "var(--ink-muted)"}, className="mb-2"),
            html.P("Supported: .csv  .xlsx  .xls  .parquet — files over 100 MB are "
                   "ignored by the uploader; paste the file path above instead.",
                   style={"fontSize": "0.7rem", "color": "var(--ink-faint)"}, className="mb-0"),
        ]),

        html.Hr(className="my-3"),

        # Auto-run: one click from ingest to report. Static/always mounted → safe as a
        # State in every chain callback (no mount-fire). The one pause is the forecast-
        # intent confirmation; the final report step calls the configured LLM.
        dbc.Switch(
            id="switch-auto-run", value=True,
            label="Run everything automatically (pauses once to confirm the forecast "
                  "setup; generates the LLM report at the end)",
            className="mb-2", style={"fontSize": "0.78rem"},
        ),

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


def _data_source_card():
    """Left-column card. Both states — the input form and the compact 'dataset loaded'
    banner — are always mounted; toggle_ingestion_panel (callbacks.py) switches which
    one is visible. Keeping the form mounted also preserves typed credentials across
    tab switches and dataset loads."""
    loaded_state = html.Div(id="ingestion-loaded", style=_HIDDEN, children=[
        dbc.Alert(
            [
                html.Div([
                    _cicon("bi-check-circle-fill", color="#10b981"),
                    html.Strong("Dataset loaded"),
                ], className="d-flex align-items-center mb-1"),
                html.Small(id="loaded-source-text",
                           style={"color": "#6ee7b7", "fontSize": "0.78rem"}),
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

    return dbc.Card([
        dbc.CardHeader([
            _cicon("bi-cloud-download", color="#6366f1"),
            "Data Source",
        ]),
        dbc.CardBody([
            loaded_state,
            html.Div(id="ingestion-form-wrap", children=_ingestion_form()),
        ]),
    ])


def _past_runs_card():
    return dbc.Card(className="mt-3", children=[
        dbc.CardHeader([
            _cicon("bi-clock-history", color="var(--ink-ghost)"),
            "Past Runs",
        ]),
        # Children filled by render_data_pane (callbacks.py) at page load and on every
        # store change, so the list stays fresh without the layout hitting the API.
        dbc.CardBody(
            id="past-runs-list",
            children=[],
            style=RUNS_LIST_STYLE,
        ),
    ])


def _setup_pane():
    return dbc.Row([
        dbc.Col(width=3, children=[
            _data_source_card(),
            _past_runs_card(),
        ]),
        # Right column: dataset profile (rebuilt on store change) then the forecast-
        # intent confirmation form — the pipeline's single user checkpoint lives on
        # Setup so the whole decision surface is one screen.
        dbc.Col(width=9, children=[
            html.Div(id="data-tab-results"),
            html.Div(id="intent-form-body", className="mt-3"),
        ]),
    ])


def _pipeline_pane():
    """Stage rail + log (left) · per-stage result cards (right). Card chrome is static —
    only the badges and the body divs inside are updated by callbacks."""
    return dbc.Row([
        dbc.Col(width=3, children=[
            dbc.Card([
                dbc.CardHeader([_cicon("bi-diagram-3", color="#6366f1"), "Pipeline progress"]),
                dbc.CardBody(html.Div(id="stage-rail"), className="py-2"),
            ]),
            dbc.Card(className="mt-3", children=[
                dbc.CardHeader([_cicon("bi-terminal", color="var(--ink-ghost)"), "Log"]),
                dbc.CardBody(html.Div(id="pipeline-log"), className="p-2"),
            ]),
        ]),
        dbc.Col(width=9, children=[
            ui.stage_card("clean", "Cleaning & Validation", "bi-scissors",
                          html.Div(id="clean-tab-body")),
            ui.stage_card("forecast_eda", "Forecast EDA", "bi-graph-up",
                          html.Div(id="fcst-eda-tab-body")),
            ui.stage_card("model_select", "Model Selection", "bi-cpu",
                          html.Div(id="model-tab-body")),
            ui.stage_card("train", "Training", "bi-lightning-fill",
                          html.Div(id="training-tab-body"), open=True),
        ]),
    ])


def create_layout():
    return dbc.Container(
        fluid=True,
        style={"minHeight": "100vh"},
        children=[

            # ── Navbar ────────────────────────────────────────────────────
            dbc.Navbar(
                dbc.Container([
                    dbc.NavbarBrand([
                        _icon("bi-graph-up-arrow", style={"color": "#10b981", "marginRight": "8px", "fontSize": "1.1rem"}),
                        "Forecasting Agent",
                    ], className="fw-bold"),

                    dbc.Nav([
                        dbc.NavItem(
                            dbc.Button(
                                html.I(id="theme-toggle-icon", className="bi bi-sun"),
                                id="btn-theme-toggle",
                                color="link",
                                size="sm",
                                className="ms-2",
                                title="Toggle light / dark mode",
                            )
                        ),
                    ], className="ms-auto align-items-center", navbar=True),
                ], fluid=True),
                color="dark", dark=True, className="mb-3",
            ),

            # storage_type="session" — survives a page refresh (memory default does not),
            # so the loaded run/results aren't silently lost if the browser reloads.
            dcc.Store(id="results-store", storage_type="session"),
            dcc.Store(id="theme-store", data="dark", storage_type="local"),
            # Auto-run state machine (phase: await_intent/train_pending/…). Session so a
            # refresh mid-auto-run can resume; every dataset switch resets it.
            dcc.Store(id="auto-run-store", storage_type="session"),
            # Relay for the confirm-click jump to the Pipeline section. Always mounted:
            # a clientside callback may NOT pair a dynamically-rendered Input with an
            # allow_duplicate output directly — same-input-signature callbacks collide
            # on the duplicate-output hash, and a second (unmounted) Input makes the
            # renderer throw "nonexistent object used in an Input" on every dispatch.
            dcc.Store(id="jump-pipeline-store"),
            # Polls GET /runs/{id}/progress while a chain callback is in flight — every
            # long callback enables it via running=. Drives ONLY the rail/log render.
            # 500ms, not more: a dcc.Interval fires its FIRST tick a full period after
            # being enabled, so this period is the floor on click→first-log-line latency.
            # Idle cost is zero — the interval is disabled outside chain callbacks.
            dcc.Interval(id="progress-interval", interval=500, disabled=True),
            # Always-on, cheap (one small JSON read per tick): re-arms progress-interval
            # whenever the server says a job is active for the loaded run, independent of
            # whether THIS browser tab has an in-flight chain callback. The running= specs
            # above only cover a poll started by this tab's own request — a page refresh or
            # reopened tab loses that in-flight callback while the job keeps running
            # server-side, so nothing was left to notice it finished (2026-07-20).
            dcc.Interval(id="job-watchdog-interval", interval=3000, disabled=False),
            # Root-level and always mounted: the download target must never be inside a
            # re-renderable pane body (outputting to an unmounted component fails silently).
            dcc.Download(id="download-forecast"),

            # ── Section tabs (static — always clickable, see _build_tabs docstring) ──
            dcc.Tabs(
                id="stage-tabs",
                value="tab-setup",
                persistence=True,
                className="stage-tabs",
                parent_className="stage-tabs-parent",
                children=_build_tabs(),
            ),

            # ── Persistent status area (targeted by run_cleaning; must stay mounted) ──
            # overlay_style applies to the (hidden) children div ONLY while loading: the
            # status div is normally empty (~0px tall), and dcc.Loading centers its 40px
            # spinner within the wrapped content's height — without the reserved height
            # the spinner straddled the container's top edge, half-hidden under the tab
            # bar on every section.
            dcc.Loading(
                html.Div(id="cleaning-status", className="mb-2 mt-3"),
                type="circle",
                color="#6366f1",
                delay_show=100,
                overlay_style={"minHeight": "44px"},
                target_components={"cleaning-status": "children"},
            ),

            # ── Tab panes: all mounted, visibility toggled clientside ───────
            # target_components covers the actual slow work (ingest / clean / load past
            # run all write results-store.data) so the overlay stays up for the whole
            # operation, not just the fast re-render that follows it.
            # NOTE: the rail/log divs are deliberately NOT in target_components — the
            # 1.5s progress polls must not flash the full-pane spinner.
            dcc.Loading(
                html.Div(id="tab-content", className="mt-2", children=[
                    html.Div(id="pane-setup", children=_setup_pane()),
                    html.Div(id="pane-pipeline", style=_HIDDEN, children=_pipeline_pane()),
                    html.Div(id="pane-results", style=_HIDDEN,
                             children=html.Div(id="results-tab-body")),
                ]),
                type="circle",
                color="#6366f1",
                delay_show=200,
                target_components={
                    "results-store": "data",
                    "cleaning-status": "children",
                    "past-runs-list": "children",
                },
            ),
        ],
    )
