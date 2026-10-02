# Sets up the `nerfly` conda env for this project on Windows. Idempotent: safe to re-run.
# Requires conda (Miniforge/Anaconda) already on PATH; run from an Anaconda/Miniforge Prompt.

$ErrorActionPreference = "Stop"
$EnvName = "nerfly"
$PyVersion = "3.11"
$BoxmotCommit = "259625c8"

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Error "nvidia-smi not found. A GPU driver must already be installed system-wide (this script will not install/modify drivers). Ask IT/the lab owner."
    exit 1
}
$cudaLine = (nvidia-smi | Select-String "CUDA Version:").ToString()
$cudaVer = ($cudaLine -split "CUDA Version:\s*")[1].Trim().Split(" ")[0]
Write-Host "Driver reports CUDA $cudaVer"
if (-not $cudaVer.StartsWith("13")) {
    Write-Warning "This project was verified against CUDA 13.x. Torch install below targets cu130; if it fails to use the GPU, pick a matching index from https://pytorch.org/get-started/locally/"
}

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "conda not found on PATH. Install Miniforge first, then re-run from its Prompt."
    exit 1
}

$envs = conda env list
if (-not ($envs -match "^\Q$EnvName\E\s")) {
    conda create -y -n $EnvName "python=$PyVersion"
}
conda activate $EnvName

pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -e .
pip install --no-deps trackeval==1.3.0

if (-not (Test-Path boxmot)) {
    git clone https://github.com/mikel-brostrom/boxmot.git
}
git -C boxmot checkout $BoxmotCommit
pip install -e ./boxmot

python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '-', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
Write-Host "Done. Activate with: conda activate $EnvName"
