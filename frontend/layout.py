"""Dash page layout."""
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

_STAGE_PILLS = [
    ("Ingest", "success"),
    ("Pre-clean EDA", "secondary"),
    ("Cleaning", "secondary"),
    ("Validation", "secondary"),
    ("Forecast EDA", "secondary"),
    ("Model Select", "secondary"),
    ("Training", "secondary"),
    ("Results", "secondary"),
]

layout = dbc.Container(
    fluid=True,
    children=[
        # ── Navbar ────────────────────────────────────────────────────────
        dbc.Navbar(
            dbc.Container([
                dbc.NavbarBrand("Automated Forecasting Agent", className="fw-bold"),
                dbc.Nav([
                    dbc.NavItem(
                        dbc.Badge(label, color=color, className="me-1 px-2 py-1"),
                    )
                    for label, color in _STAGE_PILLS
                ], className="ms-auto", navbar=True),
            ], fluid=True),
            color="dark", dark=True, className="mb-3",
        ),

        # ── Hidden helpers ─────────────────────────────────────────────────
        dcc.Interval(id="page-load", interval=500, max_intervals=1),
        dcc.Store(id="results-store"),

        # ── Main row ───────────────────────────────────────────────────────
        dbc.Row([

            # Left panel — controls
            dbc.Col(width=3, children=[
                dbc.Card([
                    dbc.CardHeader(html.Strong("Stage 1 — Ingestion")),
                    dbc.CardBody([
                        html.Div(id="connection-alert"),
                        dbc.Label("Select table", html_for="dropdown-table", className="fw-semibold"),
                        dcc.Dropdown(
                            id="dropdown-table",
                            clearable=True,
                            placeholder="Loading tables…",
                            className="mb-2",
                        ),

                        dbc.Button(
                            "Use custom query instead ▾",
                            id="btn-toggle-query",
                            color="link",
                            size="sm",
                            className="p-0 mb-1 text-decoration-none",
                        ),
                        dbc.Collapse(
                            dbc.Textarea(
                                id="input-query",
                                placeholder="SELECT * FROM schema.table WHERE …",
                                style={"height": "110px", "fontSize": "13px"},
                                className="mb-2",
                            ),
                            id="collapse-query",
                            is_open=False,
                        ),

                        dbc.Button(
                            "Run Ingestion",
                            id="btn-run",
                            color="primary",
                            className="w-100 mt-1",
                        ),
                        html.Div(id="alert-div", className="mt-2"),
                    ]),
                ]),

                dbc.Card(className="mt-3", children=[
                    dbc.CardHeader(html.Strong("Past Runs")),
                    dbc.CardBody(
                        id="past-runs-list",
                        children=[html.P("No runs yet.", className="text-muted small mb-0")],
                        style={"maxHeight": "280px", "overflowY": "auto"},
                    ),
                ]),
            ]),

            # Right panel — results
            dbc.Col(width=9, children=[
                dbc.Spinner(
                    html.Div(
                        id="results-panel",
                        children=[
                            html.P(
                                "Select a table from the dropdown and click Run Ingestion.",
                                className="text-muted mt-4 ms-2",
                            )
                        ],
                    ),
                    color="primary",
                    delay_show=300,
                ),
            ]),

        ]),
    ],
)
