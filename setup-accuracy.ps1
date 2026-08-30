param(
    [switch]$DownloadRecommendedModels,
    [switch]$DownloadReadableModel
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    & (Join-Path $ProjectRoot "setup.ps1")
}

Set-Location -LiteralPath $ProjectRoot
Write-Host "Installing maximum-accuracy local model runtimes..."
& $Python -m pip install -r (Join-Path $ProjectRoot "requirements-accuracy.txt")
if ($LASTEXITCODE -ne 0) { throw "Accuracy runtime installation failed." }

& $Python -m pip check
if ($LASTEXITCODE -ne 0) { throw "Python dependency check failed." }

if ($DownloadRecommendedModels) {
    Write-Host "Downloading Qwen3-ASR 1.7B. This uses about 4.7 GB..."
    $env:HF_HUB_DISABLE_XET = "1"
    & $Python -m voice_to_text.preflight --ensure-qwen --ensure-alignment
    if ($LASTEXITCODE -ne 0) { throw "Recommended ASR model download failed." }
}

if ($DownloadReadableModel) {
    Write-Host "Downloading the optional Qwen3.5 readable-copy model..."
    $env:HF_HUB_DISABLE_XET = "1"
    & $Python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B')"
    if ($LASTEXITCODE -ne 0) { throw "Readable-copy model download failed." }
}

Write-Host "Maximum-accuracy runtime is ready. Hindi word alignment is enabled automatically."
