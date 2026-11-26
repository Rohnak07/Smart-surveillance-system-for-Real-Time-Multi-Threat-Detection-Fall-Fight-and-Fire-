@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Smart surveillance system for Real-Time Multi-Threat Detection (Fall, Fight, and Fire) Python environment was not found.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" run_phone.py %*
if errorlevel 1 pause
