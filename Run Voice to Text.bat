@echo off
setlocal
title Private Conversation Transcriber
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
  echo First-time setup: installing the speech-to-text app...
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
  if errorlevel 1 (
    echo.
    echo Setup failed. Review the error above, then try again.
    pause
    exit /b 1
  )
)

echo Starting Private Conversation Transcriber...
start "" ".venv\Scripts\pythonw.exe" "%~dp0app.py"
exit /b 0
