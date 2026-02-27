#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x .venv/bin/python ]]; then
  echo "[ERROR] .venv missing. Run setup first."
  exit 1
fi

VENV_PY="$ROOT_DIR/.venv/bin/python"

if ! "$VENV_PY" -c "import numpy, sklearn" >/dev/null 2>&1; then
  echo "[INFO] Installing required Python packages ..."
  "$VENV_PY" -m pip install --upgrade pip
  "$VENV_PY" -m pip install numpy pandas scikit-learn
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  if ! "$VENV_PY" -c "import torch; import torchvision; import torchaudio; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    echo "[INFO] Installing CUDA-enabled torch/torchvision/torchaudio (cu126) ..."
    "$VENV_PY" -m pip install --upgrade pip
    "$VENV_PY" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  fi
else
  if ! "$VENV_PY" -c "import torch, torchvision, torchaudio" >/dev/null 2>&1; then
    echo "[INFO] Installing CPU torch/torchvision/torchaudio ..."
    "$VENV_PY" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  fi
fi

echo "[NEXT-V8 STEP 1/4] Build cache_v5_hubert audio (GPU auto) ..."
mkdir -p derived/features/cache_v5_hubert/audio derived/features/cache_v5_hubert/video
"$VENV_PY" scripts/prepare_advanced_features.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --cache-dir derived/features/cache_v5_hubert \
  --source-cache-dir derived/features/cache_v3 \
  --device auto \
  --no-prefer-source-cache \
  --fallback-raw \
  --audio-pretrained-backbone hubert_base \
  --audio-pretrained-max-samples 32000 \
  --kind audio \
  --overwrite \
  --progress-every 200

echo "[NEXT-V8 STEP 2/4] Reuse cache_v3 video features into cache_v5_hubert ..."
if [[ -d derived/features/cache_v3/video ]]; then
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete derived/features/cache_v3/video/ derived/features/cache_v5_hubert/video/
  else
    rm -rf derived/features/cache_v5_hubert/video
    mkdir -p derived/features/cache_v5_hubert/video
    cp -a derived/features/cache_v3/video/. derived/features/cache_v5_hubert/video/
  fi
else
  echo "[ERROR] cache_v3 video directory missing."
  exit 1
fi

echo "[NEXT-V8 STEP 3/4] FP32 main (gated wide, cache_v5_hubert, GPU auto) ..."
"$VENV_PY" scripts/train_fp32_multitask.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --out-dir derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_main \
  --cache-dir derived/features/cache_v5_hubert \
  --mode fusion \
  --device auto \
  --fusion-type gated \
  --modality-dropout-p 0.15 \
  --num-folds 5 \
  --epochs 24 \
  --batch-size 128 \
  --hidden-dim 384 \
  --emb-dim 192 \
  --dropout 0.30 \
  --lr 0.0006 \
  --emotion-loss ce \
  --label-smoothing 0.12 \
  --weighted-sampler \
  --n-bootstrap 200 \
  --progress-every 2000

echo "[NEXT-V8 STEP 4/4] Analyze v8 runs ..."
"$VENV_PY" scripts/analyze_phase35_next_v8.py --out-dir derived/reports

echo "Phase-3.5 next-v8 (HuBERT main) pipeline completed."
