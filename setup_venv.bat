@echo off
:: One-time setup: creates .venv and installs all dependencies. Run once, then use run_dev.bat.

python -m venv .venv
call .venv\Scripts\activate.bat

pip install --upgrade pip

:: NOTE: prophet requires C++ build tools (pystan backend) and is excluded from requirements.txt.
:: Option A: install Visual C++ Build Tools, then: pip install "prophet>=1.1.4"
:: Option B: conda install -c conda-forge prophet

pip install -r requirements.txt
pip install -e .

echo.
echo Setup complete. Run run_dev.bat to start the API server.
