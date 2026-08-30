@echo off
setlocal
title Private Conversation Transcriber - GPU
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo First-time setup: installing the speech-to-text app...
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
  if errorlevel 1 goto :failed
)

".venv\Scripts\python.exe" -c "import sys, torch; sys.exit(0 if torch.version.cuda else 1)" >nul 2>&1
if errorlevel 1 (
  echo Installing the NVIDIA CUDA runtime. This is a large first-time download...
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-gpu.ps1"
  if errorlevel 1 goto :failed
)

echo Starting Private Conversation Transcriber with automatic GPU detection...
start "" ".venv\Scripts\pythonw.exe" "%~dp0app.py"
exit /b 0

:failed
echo.
echo GPU setup failed. Review the error above, then try again.
pause
exit /b 1
