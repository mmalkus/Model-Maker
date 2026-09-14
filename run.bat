@echo off
setlocal
cd /d "%~dp0"

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv || goto :error
)

call .venv\Scripts\activate.bat || goto :error

echo Installing/updating Model Maker...
python -m pip install --upgrade quantology-modelmaker || goto :error

echo Starting Model Maker at http://127.0.0.1:8001
modelmaker-api
goto :eof

:error
echo.
echo Setup failed - see the error above.
pause
exit /b 1
