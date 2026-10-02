#!/usr/bin/env bash
# Sets up the `nerfly` conda env for this project on Linux. Idempotent: safe to re-run.
set -euo pipefail

ENV_NAME="nerfly"
PY_VERSION="3.11"
BOXMOT_COMMIT="259625c8"

if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. A GPU driver must already be installed system-wide" >&2
  echo "(this script will not install/modify drivers). Ask the machine owner/IT." >&2
  exit 1
fi
CUDA_VER=$(nvidia-smi | grep -oE "CUDA Version: [0-9.]+" | cut -d' ' -f3)
echo "Driver reports CUDA $CUDA_VER"
if [[ "$CUDA_VER" != 13.* ]]; then
  echo "NOTE: this project was verified against CUDA 13.x. Torch install below targets cu130;" >&2
  echo "if it fails to use the GPU, pick a matching index from https://pytorch.org/get-started/locally/" >&2
fi

if ! command -v conda &>/dev/null; then
  echo "conda not found, installing Miniforge to \$HOME/miniforge3 (user-space, no sudo)"
  curl -fsSL -o /tmp/miniforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
  bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
  rm /tmp/miniforge.sh
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

if ! conda env list | grep -q "^${ENV_NAME} "; then
  conda create -y -n "$ENV_NAME" "python=${PY_VERSION}"
fi
conda activate "$ENV_NAME"

pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -e .
pip install --no-deps trackeval==1.3.0

if [[ ! -d boxmot ]]; then
  git clone https://github.com/mikel-brostrom/boxmot.git
fi
git -C boxmot checkout "$BOXMOT_COMMIT"
pip install -e ./boxmot

python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '-', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
echo "Done. Activate with: conda activate $ENV_NAME"
