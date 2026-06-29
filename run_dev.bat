@echo off
:: Start the FastAPI dev server (auto-reloads on file changes). Requires setup_venv.bat first.
call .venv\Scripts\activate.bat
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
