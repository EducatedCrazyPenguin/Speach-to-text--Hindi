@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo The app is not installed yet. Running setup...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
  if errorlevel 1 pause & exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "%~dp0app.py"

