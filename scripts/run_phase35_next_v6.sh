#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x .venv/bin/python ]]; then
  echo "[ERROR] .venv missing. Run setup first."
  exit 1
fi

VENV_PY="$ROOT_DIR/.venv/bin/python"

echo "[NEXT-V6 STEP 1/3] FP32 main (CE + label smoothing + weighted sampler) ..."
"$VENV_PY" scripts/train_fp32_multitask.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --out-dir derived/results/fp32_multitask_phase35_v6_ce_ls_ws_main \
  --cache-dir derived/features/cache_v4 \
  --mode fusion \
  --device auto \
  --num-folds 5 \
  --epochs 12 \
  --batch-size 128 \
  --lr 0.001 \
  --emotion-loss ce \
  --label-smoothing 0.1 \
  --weighted-sampler \
  --n-bootstrap 200 \
  --progress-every 2000

echo "[NEXT-V6 STEP 2/3] FP32 main (Focal + weighted sampler) ..."
"$VENV_PY" scripts/train_fp32_multitask.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --out-dir derived/results/fp32_multitask_phase35_v6_focal_ws_main \
  --cache-dir derived/features/cache_v4 \
  --mode fusion \
  --device auto \
  --num-folds 5 \
  --epochs 12 \
  --batch-size 128 \
  --lr 0.001 \
  --emotion-loss focal \
  --focal-gamma 2.0 \
  --weighted-sampler \
  --n-bootstrap 200 \
  --progress-every 2000

echo "[NEXT-V6 STEP 3/3] Analyze v6 runs ..."
"$VENV_PY" scripts/analyze_phase35_next_v6.py --out-dir derived/reports

echo "Phase-3.5 next-v6 pipeline completed."
