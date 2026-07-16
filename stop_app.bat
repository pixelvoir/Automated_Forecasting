@echo off
:: Shut down both servers started by start_app.bat.
:: Primary: kill by the window titles the launcher assigned. Fallback: kill
:: whatever still listens on ports 8000/8050 (covers manually-started servers).
echo [stop] Closing Forecasting API / UI windows...
taskkill /FI "WINDOWTITLE eq Forecasting API*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Forecasting UI*" /T /F >nul 2>&1

powershell -NoProfile -Command "$pids = @(); foreach ($port in 8000, 8050) { try { $pids += (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction Stop).OwningProcess } catch {} }; $pids | Sort-Object -Unique | ForEach-Object { try { Stop-Process -Id $_ -Force -ErrorAction Stop; Write-Host ('[stop] killed PID ' + $_) } catch {} }"

echo [ok] Done.
:: stdin-free sleep (timeout.exe fails when the script runs without a console stdin)
ping -n 4 127.0.0.1 >nul
