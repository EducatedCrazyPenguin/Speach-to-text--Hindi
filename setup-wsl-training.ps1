param([switch]$InstallWsl)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$Distributions = & wsl.exe --list --quiet 2>$null
if (-not $Distributions) {
    if (-not $InstallWsl) {
        throw "WSL is not installed. Re-run this script from an Administrator PowerShell with -InstallWsl; Windows may require a restart."
    }
    Write-Host "Installing WSL 2 and Ubuntu 24.04. Windows may request a restart..."
    & wsl.exe --install -d Ubuntu-24.04
    exit $LASTEXITCODE
}

Write-Host "Preparing the official SraVaani fine-tuning environment in Ubuntu..."
$LinuxCommand = @'
set -e
sudo apt-get update
sudo apt-get install -y python3-venv git ffmpeg
mkdir -p ~/sravaani-personal
cd ~/sravaani-personal
if [ ! -d SraVaani/.git ]; then git clone https://github.com/ARTPARK-Speech-Models/SraVaani.git; fi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "nemo_toolkit[asr,cu12]" pytorch-lightning omegaconf
echo "WSL training environment is ready at ~/sravaani-personal"
'@
& wsl.exe -d Ubuntu-24.04 -- bash -lc $LinuxCommand
if ($LASTEXITCODE -ne 0) { throw "WSL SraVaani setup failed." }

Write-Host "Next: collect/correct 3 hours, run training\prepare_corrections.py, and follow training\README.md."
