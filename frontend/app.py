"""Dash application entry point."""
import dash
import dash_bootstrap_components as dbc
from dotenv import load_dotenv

load_dotenv()

from frontend.layout import layout

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=False,
    title="Forecasting Agent",
)
app.layout = layout

# Import callbacks after app is defined — registers them as a side-effect.
# The circular import (callbacks imports app) is intentional and safe here
# because `app` is fully constructed before this line executes.
import frontend.callbacks  # noqa: E402, F401

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
