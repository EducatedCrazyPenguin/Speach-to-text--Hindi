@echo off
setlocal
title Private Conversation Transcriber - Maximum Accuracy GPU
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo First-time setup: installing the local app...
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
  if errorlevel 1 goto :failed
)

".venv\Scripts\python.exe" -c "import sys, torch; sys.exit(0 if torch.version.cuda else 1)" >nul 2>&1
if errorlevel 1 (
  echo Installing the NVIDIA CUDA runtime. This is a large first-time download...
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-gpu.ps1"
  if errorlevel 1 goto :failed
)

".venv\Scripts\python.exe" -c "import qwen_asr, transformers, pyloudnorm, uroman" >nul 2>&1
if errorlevel 1 (
  echo Installing optional maximum-accuracy model runtimes...
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-accuracy.ps1"
  if errorlevel 1 goto :failed
)

echo Checking GPU, recovery models, readable model, and saved speaker credentials...
echo First use may download about 15.5 GB; downloads resume automatically.
".venv\Scripts\python.exe" -m voice_to_text.preflight --require-gpu --ensure-qwen --ensure-alignment --ensure-recovery --ensure-readable
if errorlevel 1 goto :failed

echo Starting Maximum Local Accuracy mode...
start "" ".venv\Scripts\pythonw.exe" "%~dp0app.py"
exit /b 0

:failed
echo.
echo Maximum-accuracy setup failed. Review the error above, then try again.
pause
exit /b 1
