"""Dash page layout."""
from dash import dcc, html
import dash_bootstrap_components as dbc

# Stage pill definitions: (label, id, icon, default-color)
_STAGES = [
    ("Ingest",        "pill-ingest",     "bi-cloud-download",    "success"),
    ("Pre-clean EDA", "pill-preclean",   "bi-search",            "success"),
    ("Cleaning",      "pill-cleaning",   "bi-scissors",          "success"),
    ("Validation",    "pill-validation", "bi-shield-check",      "success"),
    ("Forecast EDA",  "pill-fcst-eda",   "bi-graph-up",          "success"),
    ("Model Select",  "pill-model",      "bi-cpu",               "success"),
    ("Training",      "pill-training",   "bi-lightning-fill",    "success"),
    ("Results",       "pill-results",    "bi-trophy",            "success"),
]

_lbl = lambda text: dbc.Label(
    text, className="fw-semibold small mb-1",
    style={"fontSize": "0.68rem", "letterSpacing": "0.07em",
           "textTransform": "uppercase", "color": "#94a3b8"},
)

def _icon(name, **kwargs):
    return html.I(className=f"bi {name}", **kwargs)


def create_layout():
    return dbc.Container(
        fluid=True,
        style={"minHeight": "100vh"},
        children=[

            # ── Navbar ────────────────────────────────────────────────────
            dbc.Navbar(
                dbc.Container([
                    dbc.NavbarBrand([
                        _icon("bi-graph-up-arrow", style={"color": "#6366f1", "marginRight": "8px", "fontSize": "1.1rem"}),
                        "Forecasting Agent",
                    ], className="fw-bold"),

                    dbc.Nav([
                        dbc.NavItem(
                            dbc.Badge(
                                [_icon(icon, style={"fontSize": "0.65rem", "marginRight": "4px"}), label],
                                id=pid,
                                color=color,
                                className="me-1 px-2 py-1",
                            )
                        )
                        for label, pid, icon, color in _STAGES
                    ] + [
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

            dcc.Store(id="results-store"),
            dcc.Store(id="theme-store", data="dark", storage_type="local"),

            # ── Main row ──────────────────────────────────────────────────
            dbc.Row([

                # ── Left panel ────────────────────────────────────────────
                dbc.Col(width=3, children=[

                    # Stage 1 card
                    dbc.Card([
                        dbc.CardHeader([
                            _icon("bi-cloud-download", style={"marginRight": "6px", "color": "#6366f1"}),
                            "Stage 1 — Ingestion",
                        ]),
                        dbc.CardBody([

                            # ── Stage 1 complete state ─────────────────────
                            html.Div(id="ingestion-done", style={"display": "none"}, children=[
                                dbc.Alert(
                                    [
                                        html.Div([
                                            _icon("bi-check-circle-fill", style={"color": "#10b981", "marginRight": "6px"}),
                                            html.Strong("Stage 1 complete"),
                                        ], className="d-flex align-items-center mb-1"),
                                        html.Small(id="ingestion-done-text",
                                                   style={"color": "#6ee7b7", "fontSize": "0.78rem"}),
                                    ],
                                    color="success",
                                    className="py-2 mb-3",
                                ),
                                dbc.Button(
                                    [_icon("bi-arrow-counterclockwise"), " Load Different Dataset"],
                                    id="btn-new-run",
                                    color="outline-secondary",
                                    size="sm",
                                    className="w-100",
                                ),
                            ]),

                            # ── Ingestion form ─────────────────────────────
                            html.Div(id="ingestion-form", children=[

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
                                                "cursor": "pointer", "color": "#94a3b8",
                                                "fontSize": "0.85rem"},
                                ),

                                # ── DB credential form ─────────────────────
                                html.Div(id="section-db", children=[
                                    _lbl("Host"),
                                    dbc.Input(id="input-host", placeholder="192.168.1.10",
                                              size="sm", className="mb-2"),
                                    dbc.Row([
                                        dbc.Col([
                                            _lbl("Port"),
                                            dbc.Input(id="input-port", value="5432", type="number",
                                                      size="sm", className="mb-2"),
                                        ], width=4),
                                        dbc.Col([
                                            _lbl("Database"),
                                            dbc.Input(id="input-database", placeholder="db_name",
                                                      size="sm", className="mb-2"),
                                        ], width=8),
                                    ]),
                                    _lbl("Username"),
                                    dbc.Input(id="input-user", placeholder="postgres",
                                              size="sm", className="mb-2", autocomplete="username"),
                                    _lbl("Password"),
                                    dbc.Input(id="input-password", type="password",
                                              placeholder="password", size="sm", className="mb-2",
                                              autocomplete="new-password"),

                                    dbc.Button(
                                        [_icon("bi-plug-fill"), " Connect"],
                                        id="btn-connect",
                                        color="outline-secondary",
                                        size="sm",
                                        className="w-100 mb-2",
                                        disabled=False,
                                    ),
                                    dcc.Loading(
                                        children=html.Div(id="connection-alert"),
                                        type="circle",
                                        color="#6366f1",
                                        delay_show=150,
                                        className="mb-2",
                                    ),

                                    _lbl("Table"),
                                    dcc.Dropdown(
                                        id="dropdown-table",
                                        clearable=True,
                                        placeholder="Connect first…",
                                        className="mb-2",
                                        style={"fontSize": "0.85rem"},
                                    ),

                                    dbc.Button(
                                        [_icon("bi-code-slash"), " Custom SQL query"],
                                        id="btn-toggle-query",
                                        color="link",
                                        size="sm",
                                        className="p-0 mb-1 text-decoration-none",
                                    ),
                                    dbc.Collapse(
                                        dbc.Textarea(
                                            id="input-query",
                                            placeholder="SELECT * FROM schema.table WHERE …",
                                            style={"height": "90px", "fontSize": "12px",
                                                   "fontFamily": "monospace"},
                                            className="mb-2",
                                        ),
                                        id="collapse-query",
                                        is_open=False,
                                    ),
                                ]),

                                # ── File upload ────────────────────────────
                                html.Div(id="section-file", style={"display": "none"}, children=[
                                    _lbl("Upload file"),
                                    dcc.Upload(
                                        id="upload-file",
                                        children=html.Div([
                                            _icon("bi-cloud-upload",
                                                  style={"fontSize": "1.4rem", "color": "#4b5563",
                                                         "display": "block", "marginBottom": "4px"}),
                                            html.Span("Drag & drop or ", style={"color": "#94a3b8"}),
                                            html.A("browse", style={"color": "#6366f1", "cursor": "pointer"}),
                                        ], style={"paddingTop": "12px"}),
                                        style={
                                            "width": "100%",
                                            "height": "80px",
                                            "borderWidth": "1.5px",
                                            "borderStyle": "dashed",
                                            "borderRadius": "10px",
                                            "textAlign": "center",
                                            "borderColor": "rgba(255,255,255,0.12)",
                                            "cursor": "pointer",
                                            "fontSize": "0.82rem",
                                            "transition": "all 0.2s ease",
                                        },
                                        accept=".csv,.xlsx,.xls,.parquet",
                                        multiple=False,
                                        className="mb-1",
                                    ),
                                    html.Div(id="upload-filename",
                                             style={"fontSize": "0.75rem", "color": "#94a3b8"},
                                             className="mb-2"),
                                    html.P("Supported: .csv  .xlsx  .xls  .parquet",
                                           style={"fontSize": "0.7rem", "color": "#64748b"},
                                           className="mb-0"),
                                ]),

                                html.Hr(className="my-3"),

                                dcc.Loading(
                                    children=[
                                        dbc.Button(
                                            [_icon("bi-play-fill"), " Run Ingestion"],
                                            id="btn-run",
                                            color="primary",
                                            className="w-100",
                                            disabled=False,
                                            style={"fontWeight": "600", "letterSpacing": "0.02em"},
                                        ),
                                        html.Div(id="alert-div", className="mt-2"),
                                    ],
                                    type="circle",
                                    color="#6366f1",
                                    delay_show=100,
                                ),

                            ]),  # end ingestion-form
                        ]),
                    ]),

                    # Past Runs card
                    dbc.Card(className="mt-3", children=[
                        dbc.CardHeader([
                            _icon("bi-clock-history", style={"marginRight": "6px", "color": "#475569"}),
                            "Past Runs",
                        ]),
                        dbc.CardBody(
                            id="past-runs-list",
                            children=[
                                html.P("No runs yet.",
                                       className="text-muted small mb-0",
                                       style={"fontSize": "0.78rem"}),
                            ],
                            style={"maxHeight": "280px", "overflowY": "auto", "padding": "0.5rem"},
                        ),
                    ]),
                ]),

                # ── Right panel ───────────────────────────────────────────
                dbc.Col(width=9, children=[
                    dcc.Loading(
                        html.Div(
                            id="cleaning-status",
                            className="mb-2",
                            style={"minHeight": "40px"},
                        ),
                        type="circle",
                        color="#6366f1",
                        delay_show=100,
                        target_components={"cleaning-status": "children"},
                    ),
                    dbc.Spinner(
                        html.Div(
                            id="results-panel",
                            children=[
                                html.Div([
                                    _icon("bi-arrow-left-circle",
                                          style={"fontSize": "2rem", "color": "#334155",
                                                 "display": "block", "marginBottom": "12px"}),
                                    html.P("Select a data source and click Run Ingestion.",
                                           style={"color": "#64748b", "fontSize": "0.9rem"}),
                                ], className="text-center mt-5 pt-4"),
                            ],
                        ),
                        color="#6366f1",
                        delay_show=300,
                    ),
                ]),
            ]),
        ],
    )
