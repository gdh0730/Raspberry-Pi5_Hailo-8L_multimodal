#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x .venv/bin/python ]]; then
  echo "[ERROR] .venv missing. Run setup first."
  exit 1
fi

VENV_PY="$ROOT_DIR/.venv/bin/python"

if ! "$VENV_PY" -c "import torch, numpy, sklearn" >/dev/null 2>&1; then
  echo "[INFO] Installing required packages for phase-3.5 ..."
  "$VENV_PY" -m pip install --upgrade pip
  "$VENV_PY" -m pip install numpy pandas scikit-learn
  if command -v nvidia-smi >/dev/null 2>&1; then
    "$VENV_PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu126
  else
    "$VENV_PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  fi
fi

# Candidate A: CE loss + no weighted sampler (reference run)
"$VENV_PY" scripts/train_fp32_multitask.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --out-dir derived/results/fp32_multitask_phase35_ce_main \
  --cache-dir derived/features/cache_v1 \
  --mode fusion \
  --device auto \
  --num-folds 5 \
  --epochs 12 \
  --batch-size 128 \
  --lr 0.001 \
  --emotion-loss ce \
  --n-bootstrap 200 \
  --progress-every 2000

# Candidate B: Focal loss + weighted sampler (imbalance-oriented)
"$VENV_PY" scripts/train_fp32_multitask.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --out-dir derived/results/fp32_multitask_phase35_focal_ws_main \
  --cache-dir derived/features/cache_v1 \
  --mode fusion \
  --device auto \
  --num-folds 5 \
  --epochs 12 \
  --batch-size 128 \
  --lr 0.0008 \
  --dropout 0.25 \
  --emotion-loss focal \
  --focal-gamma 2.0 \
  --weighted-sampler \
  --n-bootstrap 200 \
  --progress-every 2000

"$VENV_PY" scripts/analyze_phase35_results.py \
  --out-dir derived/reports \
  --candidate-dirs \
    derived/results/fp32_multitask_phase35_ce_main,derived/results/fp32_multitask_phase35_focal_ws_main

echo "Phase-3.5 experiments completed."
