@echo off
setlocal
cd /d "%~dp0"

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv || goto :error
)

call .venv\Scripts\activate.bat || goto :error

echo Installing backend dependencies...
python -m pip install -e ".[dev]" || goto :error

echo Installing frontend dependencies...
pushd frontend
call npm install --no-audit --prefer-offline || goto :error

echo Building frontend...
call npm run build || goto :error
popd

echo Starting Model Maker at http://127.0.0.1:8001
modelmaker-api
goto :eof

:error
echo.
echo Setup failed - see the error above.
pause
exit /b 1
