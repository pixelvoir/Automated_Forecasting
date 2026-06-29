"""Dash callbacks — all interactivity lives here."""
import os

import requests
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, dash_table, no_update

from frontend.app import app

API_URL = os.environ.get("API_URL", "http://localhost:8000")


# ── 1. Populate table dropdown on page load ────────────────────────────────

@app.callback(
    Output("dropdown-table", "options"),
    Output("dropdown-table", "placeholder"),
    Output("connection-alert", "children"),
    Input("page-load", "n_intervals"),
)
def load_tables(_):
    try:
        resp = requests.get(f"{API_URL}/runs/tables", timeout=10)
        if resp.ok:
            tables = resp.json()["tables"]
            options = [{"label": t["qualified"], "value": t["qualified"]} for t in tables]
            placeholder = "Select a table…" if options else "No tables found in database"
            return options, placeholder, None
        detail = resp.json().get("detail", "unknown error")
        alert = dbc.Alert(detail, color="danger", className="mb-2 py-2 small")
        return [], "—", alert
    except requests.exceptions.ConnectionError:
        alert = dbc.Alert("Cannot reach API — is run_dev.bat running?", color="danger", className="mb-2 py-2 small")
        return [], "—", alert
    except Exception as e:
        alert = dbc.Alert(str(e), color="danger", className="mb-2 py-2 small")
        return [], "—", alert


# ── 2. Toggle custom query textarea ───────────────────────────────────────

@app.callback(
    Output("collapse-query", "is_open"),
    Input("btn-toggle-query", "n_clicks"),
    State("collapse-query", "is_open"),
    prevent_initial_call=True,
)
def toggle_query(_, is_open):
    return not is_open


# ── 3. Trigger ingestion run ───────────────────────────────────────────────

@app.callback(
    Output("results-store", "data"),
    Output("alert-div", "children"),
    Input("btn-run", "n_clicks"),
    State("dropdown-table", "value"),
    State("input-query", "value"),
    prevent_initial_call=True,
)
def trigger_run(_, table, query):
    # Determine payload — custom query overrides dropdown if both are filled
    if query and query.strip():
        payload = {"query": query.strip()}
    elif table:
        payload = {"table": table}
    else:
        return no_update, dbc.Alert(
            "Select a table from the dropdown or enter a custom query.", color="warning", dismissable=True
        )

    try:
        run_resp = requests.post(f"{API_URL}/runs", json=payload, timeout=120)
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
        return full_data, dbc.Alert(
            f"Completed: {run_id}", color="success", dismissable=True, duration=6000
        )

    except requests.exceptions.Timeout:
        return no_update, dbc.Alert(
            "Request timed out — the table may be very large. "
            "Set 'row_limit' in config/settings.yaml to cap rows during dev.",
            color="danger", dismissable=True,
        )
    except requests.exceptions.ConnectionError:
        return no_update, dbc.Alert("Cannot reach API — is run_dev.bat running?", color="danger", dismissable=True)
    except Exception as e:
        return no_update, dbc.Alert(f"Unexpected error: {e}", color="danger", dismissable=True)


# ── 4. Render results from store ───────────────────────────────────────────

@app.callback(
    Output("results-panel", "children"),
    Input("results-store", "data"),
)
def render_results(data):
    if not data:
        return html.P(
            "Select a table from the dropdown and click Run Ingestion.",
            className="text-muted mt-4 ms-2",
        )

    summary = data.get("_summary", {})
    shape = data.get("shape", {})
    schema = data.get("schema", [])
    nulls = data.get("nulls", {})
    numeric_stats = data.get("numeric_stats", {})
    datetime_cols = data.get("datetime_cols", [])
    frequency = data.get("frequency", {})
    duplicates = data.get("duplicates", {})

    # ── Shape summary card ─────────────────────────────────────────────────
    shape_card = dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div(className="text-muted small", children="Run ID"),
                    html.Strong(data.get("run_id", "—"), style={"fontSize": "13px"}),
                ], width=4),
                dbc.Col([
                    html.Div(className="text-muted small", children="Rows"),
                    html.Strong(f"{shape.get('rows', 0):,}"),
                ], width=2),
                dbc.Col([
                    html.Div(className="text-muted small", children="Columns"),
                    html.Strong(str(shape.get("cols", 0))),
                ], width=2),
                dbc.Col([
                    html.Div(className="text-muted small", children="Memory"),
                    html.Strong(f"{shape.get('memory_mb', 0)} MB"),
                ], width=2),
                dbc.Col([
                    html.Div(className="text-muted small", children="Duplicate rows"),
                    html.Strong(str(duplicates.get("row_count", 0))),
                ], width=2),
            ]),
        ]),
        className="mb-3 border-success",
    )

    # ── Schema table ───────────────────────────────────────────────────────
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

    schema_table = _datatable(
        schema_rows,
        ["Column", "Inferred type", "DB dtype", "Null count", "Null %", "Frequency"],
    )

    # ── Numeric stats table ────────────────────────────────────────────────
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
            html.H6("Numeric Statistics", className="mt-4 mb-2 fw-semibold"),
            _datatable(numeric_rows, ["Column", "Min", "Max", "Mean", "Median", "Std", "Skew", "Kurtosis", "IQR"]),
        ]

    return [shape_card, html.H6("Schema", className="mb-2 fw-semibold"), schema_table, *numeric_section]


def _datatable(rows: list[dict], columns: list[str]) -> dash_table.DataTable:
    return dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in columns],
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": "13px", "padding": "6px 10px", "textAlign": "left"},
        style_header={"fontWeight": "600", "backgroundColor": "#f1f3f5", "borderBottom": "2px solid #dee2e6"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"}],
        page_size=25,
    )


def _fmt(v: float) -> str:
    return f"{v:,.4f}" if abs(v) < 1e6 else f"{v:,.2f}"
