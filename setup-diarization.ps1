param(
    [switch]$Gpu
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Run setup.ps1 before installing speaker identification."
}

Set-Location -LiteralPath $ProjectRoot
Write-Host "Installing optional local two-speaker identification (large download)..."
if ($Gpu) {
    Write-Host "Installing the CUDA-enabled PyTorch runtime for the NVIDIA GPU..."
    & (Join-Path $ProjectRoot "setup-gpu.ps1")
    & $Python -m pip install torchaudio==2.8.0+cu129 --index-url https://download.pytorch.org/whl/cu129
}
& $Python -m pip install -r (Join-Path $ProjectRoot "requirements-diarization.txt")
Write-Host "Speaker identification installed. Accept the model terms and create a token as described in README.md."
