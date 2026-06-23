@echo off
setlocal

set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not exist "%PYTHON_EXE%" (
    for /f "delims=" %%I in ('where python 2^>nul') do (
        set "PYTHON_EXE=%%I"
        goto :found_python
    )
)
:found_python

if not exist "%PYTHON_EXE%" (
    echo Python was not found. Install Python 3.12 or set PYTHON_EXE.
    exit /b 1
)

"%PYTHON_EXE%" -m automated_forecasting.cli %*
exit /b %ERRORLEVEL%