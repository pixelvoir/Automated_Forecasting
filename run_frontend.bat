@echo off
:: Start the Dash UI at http://localhost:8050
:: Requires run_dev.bat (FastAPI at :8000) to be running first.
call .venv\Scripts\activate.bat
:: Dev mode: werkzeug reloader + debugger on. start_app.bat runs without this.
set APP_DEBUG=1
python -m frontend.app
