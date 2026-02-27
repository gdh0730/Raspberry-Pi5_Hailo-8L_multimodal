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

echo "[NEXT-V7 STEP 1/5] FP32 main (gated CE+LS+WS) ..."
"$VENV_PY" scripts/train_fp32_multitask.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --out-dir derived/results/fp32_multitask_phase35_v7_ce_ls_ws_gated_main \
  --cache-dir derived/features/cache_v4 \
  --mode fusion \
  --device auto \
  --fusion-type gated \
  --modality-dropout-p 0.10 \
  --num-folds 5 \
  --epochs 16 \
  --batch-size 128 \
  --hidden-dim 256 \
  --emb-dim 128 \
  --dropout 0.25 \
  --lr 0.0008 \
  --emotion-loss ce \
  --label-smoothing 0.08 \
  --weighted-sampler \
  --n-bootstrap 200 \
  --progress-every 2000

echo "[NEXT-V7 STEP 2/5] FP32 main (gated focal+WS) ..."
"$VENV_PY" scripts/train_fp32_multitask.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --out-dir derived/results/fp32_multitask_phase35_v7_focal_ws_gated_main \
  --cache-dir derived/features/cache_v4 \
  --mode fusion \
  --device auto \
  --fusion-type gated \
  --modality-dropout-p 0.10 \
  --num-folds 5 \
  --epochs 16 \
  --batch-size 128 \
  --hidden-dim 256 \
  --emb-dim 128 \
  --dropout 0.25 \
  --lr 0.0008 \
  --emotion-loss focal \
  --focal-gamma 1.5 \
  --weighted-sampler \
  --n-bootstrap 200 \
  --progress-every 2000

echo "[NEXT-V7 STEP 3/5] FP32 main (gated CE+LS+WS, wider) ..."
"$VENV_PY" scripts/train_fp32_multitask.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --out-dir derived/results/fp32_multitask_phase35_v7_ce_ls_ws_gated_wide_main \
  --cache-dir derived/features/cache_v4 \
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

echo "[NEXT-V7 STEP 4/5] ML main (cache_v4, rbf_svm fusion) ..."
"$VENV_PY" scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --out-dir derived/results/ml_baselines_phase35_v7_rbfsvm_main \
  --cache-dir derived/features/cache_v4 \
  --modalities fusion \
  --classifier rbf_svm \
  --num-folds 5 \
  --n-bootstrap 200 \
  --progress-every 2000

echo "[NEXT-V7 STEP 5/5] Analyze v7 runs ..."
"$VENV_PY" scripts/analyze_phase35_next_v7.py --out-dir derived/reports

echo "Phase-3.5 next-v7 pipeline completed."
