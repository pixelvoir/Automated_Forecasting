@echo off
:: One-time setup: creates .venv and installs all dependencies.
:: After this, double-click start_app.bat (or use run_dev.bat/run_frontend.bat for dev).

python -m venv .venv
call .venv\Scripts\activate.bat

pip install --upgrade pip

:: torch is CPU-only here (powers the NHITS backend); the CPU wheel index keeps the
:: download far smaller than the default CUDA build.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
pip install -e .

echo.
echo Setup complete. Double-click start_app.bat to launch the app.
