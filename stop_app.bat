@echo off
:: Shut down both servers started by start_app.bat.
:: Primary: kill by the window titles the launcher assigned. Fallback: kill
:: whatever still listens on ports 8000/8050 (covers manually-started servers).
echo [stop] Closing Forecasting API / UI windows...
taskkill /FI "WINDOWTITLE eq Forecasting API*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Forecasting UI*" /T /F >nul 2>&1

powershell -NoProfile -Command "$pids = @(); foreach ($port in 8000, 8050) { try { $pids += (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction Stop).OwningProcess } catch {} }; $pids | Sort-Object -Unique | ForEach-Object { taskkill /PID $_ /T /F 2>$null | Out-Null; Write-Host ('[stop] killed PID tree ' + $_) }"

:: Safety net: kill any orphaned pipeline job worker (a multiprocessing spawn child) left
:: behind if a server was force-killed before its parent-death watchdog could fire. These
:: only ever exist as children of our API, so matching the spawn bootstrap is safe.
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'multiprocessing.spawn|multiprocessing-fork' } | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop; Write-Host ('[stop] killed orphaned job worker PID ' + $_.ProcessId) } catch {} }"

echo [ok] Done.
:: stdin-free sleep (timeout.exe fails when the script runs without a console stdin)
ping -n 4 127.0.0.1 >nul
