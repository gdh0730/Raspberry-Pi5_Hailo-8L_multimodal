#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x .venv/bin/python ]]; then
  echo "[ERROR] .venv missing. Run setup first."
  exit 1
fi

PY="$ROOT_DIR/.venv/bin/python"

if ! "$PY" -c "import onnx, onnxscript" >/dev/null 2>&1; then
  echo "[INFO] Installing ONNX dependencies in .venv ..."
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install onnx onnxscript
fi

RUN_DIR="derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune4"
CKPT_PATH="${RUN_DIR}/checkpoints/best_fold_0.pt"
CACHE_DIR="derived/features/cache_v5_hubert"
MANIFEST="derived/manifests/manifest_multimodal_common6_av.jsonl"
SPLIT_LIST="derived/splits/groupkfold5_all/fold_0_train.txt"
ONNX_DIR="derived/hailo/onnx"
CALIB_DIR="derived/hailo/calib/fold0_train_1024"

echo "[E4 STEP 1/2] Export ONNX from best fold checkpoint ..."
"$PY" hailo/export_onnx.py \
  --run-dir "$RUN_DIR" \
  --fold 0 \
  --onnx-dir "$ONNX_DIR"

echo "[E4 STEP 2/2] Dump calibration npy directory ..."
"$PY" hailo/calib_dump_npy_dir.py \
  --manifest "$MANIFEST" \
  --cache-dir "$CACHE_DIR" \
  --split-list "$SPLIT_LIST" \
  --out-dir "$CALIB_DIR" \
  --max-samples 1024 \
  --seed 1337 \
  --normalize-checkpoint "$CKPT_PATH"

echo "[OK] E4 local prep completed."
echo " - ONNX dir : $ONNX_DIR"
echo " - Calib dir: $CALIB_DIR"
echo "Next: run hailo/compile_hef.sh on Hailo DFC/Model Zoo host."
