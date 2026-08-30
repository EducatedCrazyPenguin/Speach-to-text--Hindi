$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Run setup.ps1 before installing GPU support."
}

Set-Location -LiteralPath $ProjectRoot
Write-Host "Installing CUDA 12.9 runtime libraries through PyTorch (large download)..."
& $Python -m pip install torch==2.8.0+cu129 --index-url https://download.pytorch.org/whl/cu129
Write-Host "Installing GPU inference for the AI4Bharat ONNX models..."
& $Python -m pip install onnxruntime==1.23.2 coloredlogs
& $Python -m pip install --force-reinstall --no-deps onnxruntime-gpu==1.23.2
Write-Host "GPU runtimes installed. Restart the app and leave Device set to auto."
