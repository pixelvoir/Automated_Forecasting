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

# Backstop for the 100 MB dcc.Upload cap: werkzeug rejects an oversized POST on its
# Content-Length header BEFORE buffering the body, so the Dash process can never OOM on
# a huge upload again (a 254 MB CSV once base64-inflated to ~930 MB of transient copies
# here). 150 MB = 100 MB file × ~1.37 base64 inflation + the JSON envelope.
app.server.config["MAX_CONTENT_LENGTH"] = 150 * 1024 * 1024


@app.server.errorhandler(413)
def _too_large(_e):
    return {"error": "Upload too large — uploads are capped at 100 MB. Paste the local "
                     "file path in the ingestion form instead (no size limit)."}, 413


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
