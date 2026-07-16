@echo off
:: ─────────────────────────────────────────────────────────────────────────────
::  One-click launcher: starts the API + UI (production mode, no dev reload/debug
::  — the file watchers burn constant CPU) and opens the app in your browser.
::  Stop everything with stop_app.bat. Developers: use run_dev.bat/run_frontend.bat.
:: ─────────────────────────────────────────────────────────────────────────────
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [setup] No virtual environment found - running one-time setup...
    call setup_venv.bat
    if not exist ".venv\Scripts\python.exe" (
        echo [error] Setup failed - see messages above.
        pause
        exit /b 1
    )
)

echo [start] Launching Forecasting API on http://127.0.0.1:8000 ...
start "Forecasting API" /MIN cmd /c ".venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000"

echo [start] Launching Forecasting UI on http://127.0.0.1:8050 ...
start "Forecasting UI" /MIN cmd /c ".venv\Scripts\python.exe -m frontend.app"

echo [wait] Waiting for the API to come up (up to 60s)...
powershell -NoProfile -Command "$ok=$false; foreach ($i in 1..60) { try { $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:8000/health; if ($r.StatusCode -eq 200) { $ok=$true; break } } catch {}; Start-Sleep -Seconds 1 }; if (-not $ok) { exit 1 }"
if errorlevel 1 (
    echo [error] The API did not respond on http://127.0.0.1:8000/health within 60s.
    echo         Check the "Forecasting API" window for the error.
    pause
    exit /b 1
)

echo [ok] Opening http://127.0.0.1:8050 in your browser...
start "" http://127.0.0.1:8050
echo.
echo  The app is running. Both servers run minimized in the taskbar
echo  ("Forecasting API" / "Forecasting UI"). Run stop_app.bat to shut down.
:: stdin-free sleep (timeout.exe fails when the script runs without a console stdin)
ping -n 6 127.0.0.1 >nul
