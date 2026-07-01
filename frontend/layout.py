"""Dash page layout.

The body is a tabbed interface: one tab per pipeline stage. All run state lives in the
root-level ``results-store``; ``tab-content`` is rebuilt by the ``render_tab`` callback in
callbacks.py whenever the active tab or the store changes. Keep the stores, cleaning-status
and tab-content at the root so cross-tab callbacks always have their targets mounted.

Tabs are always clickable — there is no disabled/locked state. When a tab's prerequisite
stage hasn't run yet, its content simply shows a short line of text saying so instead of
blocking access to the tab itself.
"""
from dash import dcc, html
import dash_bootstrap_components as dbc

# Tab definitions: (label, tab-id/value, icon)
_TABS = [
    ("Data & Pre-clean EDA", "tab-data",     "bi-search"),
    ("Cleaning",             "tab-clean",     "bi-scissors"),
    ("Forecast EDA",         "tab-fcst-eda",  "bi-graph-up"),
    ("Model Select",         "tab-model",     "bi-cpu"),
    ("Training",             "tab-training",  "bi-lightning-fill"),
    ("Results",              "tab-results",   "bi-trophy"),
]


def _icon(name, **kwargs):
    return html.I(className=f"bi {name}", **kwargs)


def _tab_label(icon, text):
    return html.Span([_icon(icon, style={"marginRight": "8px"}), text])


def _build_tabs():
    """Static tab bar — every tab is always clickable. Content-level placeholders (in
    callbacks.py) tell the user what to do first, rather than the tab itself being blocked."""
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

            # ── Active tab body (rebuilt by render_tab callback) ───────────
            # target_components covers the actual slow work (ingest / clean / load past
            # run all write results-store.data) so the overlay stays up for the whole
            # operation, not just the fast re-render that follows it.
            dcc.Loading(
                html.Div(id="tab-content", className="mt-2"),
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
