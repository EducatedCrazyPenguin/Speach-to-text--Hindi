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
    Write-Host "Downloading the ASR, alignment, recovery, and readable models..."
    $env:HF_HUB_DISABLE_XET = "1"
    & $Python -m voice_to_text.preflight --ensure-qwen --ensure-alignment --ensure-recovery --ensure-readable
    if ($LASTEXITCODE -ne 0) { throw "Recommended ASR model download failed." }
}

if ($DownloadReadableModel) {
    Write-Host "Downloading the Qwen3.5 readable-copy model and isolated runtime..."
    $env:HF_HUB_DISABLE_XET = "1"
    & $Python -m voice_to_text.preflight --ensure-readable
    if ($LASTEXITCODE -ne 0) { throw "Readable-copy model download failed." }
}

Write-Host "Maximum-accuracy runtime is ready. Hindi word alignment is enabled automatically."
