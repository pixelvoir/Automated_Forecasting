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

# Importing callbacks registers them via the global @callback decorator (Dash 4.x pattern).
# Must happen AFTER app is created so Dash knows which app to attach them to.
import frontend.callbacks  # noqa: E402, F401

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
