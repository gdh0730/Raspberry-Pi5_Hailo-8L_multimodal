#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Run this script manually if you want to move from Phase-1 baseline
# to deep-model training (B1/B2/B3/B4).
#
# Requirements:
# - sudo access
# - internet access

sudo apt update
sudo apt install -y python3-pip python3-venv ffmpeg

python3 -m pip install --user virtualenv
python3 -m virtualenv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Core research stack
python -m pip install numpy pandas scikit-learn

# Optional deep learning stack
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] NVIDIA GPU detected. Installing CUDA-enabled PyTorch wheels (cu126) ..."
  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
else
  echo "[INFO] NVIDIA GPU not detected. Installing CPU-only PyTorch wheels ..."
  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo "Environment ready. Activate with: source .venv/bin/activate"
