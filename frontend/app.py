"""Dash application entry point."""
import dash
import dash_bootstrap_components as dbc
from dotenv import load_dotenv

load_dotenv()

from frontend.layout import create_layout

app = dash.Dash(
    __name__,
    external_stylesheets=[],   # themes served from assets/ — no CDN needed
    suppress_callback_exceptions=True,
    title="Forecasting Agent",
)
app.layout = create_layout()


@app.server.after_request
def _no_cache_html(resp):
    """Never let the browser cache the index page or the layout/dependency payloads.

    The index HTML embeds the inline clientside callback functions. Dash serves it with
    no cache headers, so a browser can reuse a cached copy while fetching fresh
    /_dash-dependencies from a newer server — the stale page then lacks any clientside
    function added since, crashing with "Cannot read properties of undefined (reading
    'apply')" the moment that callback fires. Component-suite JS bundles are fingerprinted
    and deliberately left cacheable."""
    from flask import request
    if resp.mimetype == "text/html" or request.path in ("/_dash-layout", "/_dash-dependencies"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


# Importing callbacks registers them via the global @callback decorator (Dash 4.x pattern).
# Must happen AFTER app is created so Dash knows which app to attach them to.
import frontend.callbacks  # noqa: E402, F401

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
