@echo off
cd /d "%~dp0"

:: Try the project-root venv first (no console window)
if exist "%~dp0.venv\Scripts\pythonw.exe" (
    start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0run.py"
    exit /b 0
)

:: Fall back to pythonw on PATH (conda, system Python, differently-named venv)
where pythonw >nul 2>&1
if %errorlevel% == 0 (
    start "" pythonw "%~dp0run.py"
    exit /b 0
)

:: pythonw not available - try plain python (shows a console window)
where python >nul 2>&1
if %errorlevel% == 0 (
    start "" python "%~dp0run.py"
    exit /b 0
)

:: Nothing found - show an error the user can actually read
echo.
echo ERROR: Python not found.
echo.
echo Tried in order:
echo   1. %~dp0.venv\Scripts\pythonw.exe  (project venv)
echo   2. pythonw  (PATH)
echo   3. python   (PATH)
echo.
echo To fix this, either:
echo   - Create a venv at the project root:  python -m venv .venv
echo   - Or activate your environment before running this script
echo.
pause
exit /b 1
