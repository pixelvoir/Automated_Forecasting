"""Dash page layout.

The body is a tabbed interface: one tab per pipeline stage. **All six tab panes are
always mounted** — a clientside callback in callbacks.py toggles their ``style.display``
when the active tab changes, so switching tabs never touches the server and never
unmounts components.

This matters for correctness, not just speed: Dash fires callbacks whose Inputs are
dynamically added or removed by another callback's render, *ignoring*
``prevent_initial_call`` (and for pattern-matching ALL inputs the renderer's
``replacePMC`` crashes outright when such a fire meets a pattern-matching ``running=``
spec — "Cannot read properties of undefined (reading 'run_id')"). Keeping everything
mounted removes the whole class of bugs.

Only the data-dependent bodies (``data-tab-results``, ``clean-tab-body``,
``past-runs-list``) are re-rendered by server callbacks, and only when
``results-store`` actually changes.

Tabs are always clickable — there is no disabled/locked state. When a tab's prerequisite
stage hasn't run yet, its content simply shows a short line of text saying so instead of
blocking access to the tab itself.
"""
import dash_bootstrap_components as dbc
from dash import dcc, html

# Tab definitions: (label, tab-id/value, icon). Each tab value maps to a pane id via
# "tab-" → "pane-" (see the clientside pane switcher in callbacks.py).
_TABS = [
    ("Data & Pre-clean EDA", "tab-data",     "bi-search"),
    ("Cleaning",             "tab-clean",     "bi-scissors"),
    ("Forecast EDA",         "tab-fcst-eda",  "bi-graph-up"),
    ("Model Select",         "tab-model",     "bi-cpu"),
    ("Training",             "tab-training",  "bi-lightning-fill"),
    ("Results",              "tab-results",   "bi-trophy"),
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
               "textTransform": "uppercase", "color": "#94a3b8"},
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
            _cicon("bi-clock-history", color="#475569"),
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


def _data_pane():
    return dbc.Row([
        dbc.Col(width=3, children=[
            _data_source_card(),
            _past_runs_card(),
        ]),
        # Right column: rebuilt by render_data_results whenever results-store changes.
        dbc.Col(width=9, children=html.Div(id="data-tab-results")),
    ])


def _placeholder_pane():
    return html.Div(
        [
            _cicon("bi-hourglass-split", fontSize="2rem", color="#334155",
                   display="block", marginBottom="12px", marginRight="0"),
            html.P("This stage is not implemented yet.",
                   style={"color": "#64748b", "fontSize": "0.9rem"}),
        ],
        className="text-center mt-5 pt-4",
    )


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

            # ── Stage tabs (static — always clickable, see _build_tabs docstring) ──
            dcc.Tabs(
                id="stage-tabs",
                value="tab-data",
                persistence=True,
                className="stage-tabs",
                parent_className="stage-tabs-parent",
                children=_build_tabs(),
            ),

            # ── Persistent status area (targeted by run_cleaning; must stay mounted) ──
            dcc.Loading(
                html.Div(id="cleaning-status", className="mb-2 mt-3"),
                type="circle",
                color="#6366f1",
                delay_show=100,
                target_components={"cleaning-status": "children"},
            ),

            # ── Tab panes: all mounted, visibility toggled clientside ───────
            # target_components covers the actual slow work (ingest / clean / load past
            # run all write results-store.data) so the overlay stays up for the whole
            # operation, not just the fast re-render that follows it.
            dcc.Loading(
                html.Div(id="tab-content", className="mt-2", children=[
                    html.Div(id="pane-data", children=_data_pane()),
                    html.Div(id="pane-clean", style=_HIDDEN,
                             children=html.Div(id="clean-tab-body")),
                    html.Div(id="pane-fcst-eda", style=_HIDDEN, children=_placeholder_pane()),
                    html.Div(id="pane-model", style=_HIDDEN, children=_placeholder_pane()),
                    html.Div(id="pane-training", style=_HIDDEN, children=_placeholder_pane()),
                    html.Div(id="pane-results", style=_HIDDEN, children=_placeholder_pane()),
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
