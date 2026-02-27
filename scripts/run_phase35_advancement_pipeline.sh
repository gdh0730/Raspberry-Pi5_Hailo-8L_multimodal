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
  echo "[INFO] Installing required packages ..."
  "$VENV_PY" -m pip install --upgrade pip
  "$VENV_PY" -m pip install numpy pandas scikit-learn
  if command -v nvidia-smi >/dev/null 2>&1; then
    "$VENV_PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu126
  else
    "$VENV_PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  fi
fi

echo "[STEP 1/5] Build advanced cache_v2 ..."
"$VENV_PY" scripts/prepare_advanced_features.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --cache-dir derived/features/cache_v2 \
  --source-cache-dir derived/features/cache_v1 \
  --device auto \
  --kind both \
  --overwrite \
  --progress-every 300

echo "[STEP 2/5] Train ML baseline (logreg, cache_v2) ..."
"$VENV_PY" scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --out-dir derived/results/ml_baselines_phase35_v2_logreg_main \
  --cache-dir derived/features/cache_v2 \
  --modalities audio,video,fusion \
  --classifier logreg \
  --num-folds 5 \
  --n-bootstrap 200 \
  --progress-every 2000

echo "[STEP 3/5] Train ML baseline (random_forest, cache_v2) ..."
"$VENV_PY" scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --out-dir derived/results/ml_baselines_phase35_v2_rf_main \
  --cache-dir derived/features/cache_v2 \
  --modalities audio,video,fusion \
  --classifier random_forest \
  --num-folds 5 \
  --n-bootstrap 200 \
  --progress-every 2000

echo "[STEP 4/5] Train FP32 candidate on cache_v2 ..."
"$VENV_PY" scripts/train_fp32_multitask.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --out-dir derived/results/fp32_multitask_phase35_v2_ce_main \
  --cache-dir derived/features/cache_v2 \
  --mode fusion \
  --device auto \
  --num-folds 5 \
  --epochs 12 \
  --batch-size 128 \
  --lr 0.001 \
  --emotion-loss ce \
  --n-bootstrap 200 \
  --progress-every 2000

echo "[STEP 5/5] Analyze phase35 advancement results ..."
"$VENV_PY" scripts/analyze_phase35_advancement.py --out-dir derived/reports

echo "Phase-3.5 advancement pipeline completed."
