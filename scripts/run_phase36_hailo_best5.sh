#!/usr/bin/env bash
set -euo pipefail

# Build Hailo-ready artifacts for best Phase36 runs (best mode per track):
# 1) choose best run per track from progress csv
# 2) export ONNX
# 3) build normalized train calibration npy set
# 4) compile HEF locally (optional)
#
# Example:
#   bash scripts/run_phase36_hailo_best5.sh
#   bash scripts/run_phase36_hailo_best5.sh --tracks id_all,id_crema --compile 1
#   bash scripts/run_phase36_hailo_best5.sh --compile 0

PROGRESS_CSV="derived/reports/phase36_fp32_test_progress.csv"
RESULTS_ROOT="derived/results/phase36"
SPLIT_DIR="derived/splits/phase36_id_ood"
MANIFEST="derived/manifests/manifest_multimodal_common6_av.jsonl"
CACHE_DIR="derived/features/cache_v5_hubert"
ONNX_DIR="derived/hailo/onnx/phase36"
CALIB_ROOT="derived/hailo/calib/phase36"
BUILD_ROOT="derived/hailo/build/phase36"
OUT_META="derived/hailo/phase36_best5_build_meta.csv"
OUT_CANDIDATES="derived/hailo/phase36_best5_candidates.csv"
MAX_CALIB=1024
TRACKS=""
ALLOWED_MODES="audio,video,fusion"
COMPILE=1
SKIP_EXISTING=1
VENV_FP32=".venv"
VENV_HAILO=".venv-hailo"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --progress-csv) PROGRESS_CSV="$2"; shift 2 ;;
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --split-dir) SPLIT_DIR="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --onnx-dir) ONNX_DIR="$2"; shift 2 ;;
    --calib-root) CALIB_ROOT="$2"; shift 2 ;;
    --build-root) BUILD_ROOT="$2"; shift 2 ;;
    --out-meta) OUT_META="$2"; shift 2 ;;
    --out-candidates) OUT_CANDIDATES="$2"; shift 2 ;;
    --max-calib) MAX_CALIB="$2"; shift 2 ;;
    --tracks) TRACKS="$2"; shift 2 ;;
    --allowed-modes) ALLOWED_MODES="$2"; shift 2 ;;
    --compile) COMPILE="$2"; shift 2 ;;
    --skip-existing) SKIP_EXISTING="$2"; shift 2 ;;
    --venv-fp32) VENV_FP32="$2"; shift 2 ;;
    --venv-hailo) VENV_HAILO="$2"; shift 2 ;;
    *) echo "[ERROR] Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ ! -x "$VENV_FP32/bin/python" ]]; then
  echo "[ERROR] Missing fp32 python: $VENV_FP32/bin/python"
  exit 1
fi
if [[ "$COMPILE" == "1" && ! -x "$VENV_HAILO/bin/python" ]]; then
  echo "[ERROR] Missing hailo python: $VENV_HAILO/bin/python"
  exit 1
fi
if [[ ! -f "$PROGRESS_CSV" ]]; then
  echo "[ERROR] Missing progress csv: $PROGRESS_CSV"
  exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "[ERROR] Missing manifest: $MANIFEST"
  exit 1
fi
if [[ ! -d "$CACHE_DIR" ]]; then
  echo "[ERROR] Missing cache dir: $CACHE_DIR"
  exit 1
fi
if [[ ! -d "$SPLIT_DIR" ]]; then
  echo "[ERROR] Missing split dir: $SPLIT_DIR"
  exit 1
fi

mkdir -p "$ONNX_DIR" "$CALIB_ROOT" "$BUILD_ROOT" "$(dirname "$OUT_META")" "$(dirname "$OUT_CANDIDATES")"

SELECT_CSV="$(mktemp)"
"$VENV_FP32/bin/python" - "$PROGRESS_CSV" "$TRACKS" "$ALLOWED_MODES" "$SELECT_CSV" <<'PY'
import csv
import sys
from collections import defaultdict

progress_csv = sys.argv[1]
tracks_arg = sys.argv[2].strip()
modes_arg = sys.argv[3].strip()
out_csv = sys.argv[4]

wanted = None
if tracks_arg:
    wanted = {x.strip() for x in tracks_arg.split(",") if x.strip()}
allowed_modes = {x.strip() for x in modes_arg.split(",") if x.strip()}

best = {}
with open(progress_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        t = row["track"].strip()
        m = row["mode"].strip()
        if wanted is not None and t not in wanted:
            continue
        if allowed_modes and m not in allowed_modes:
            continue
        f1 = float(row["macro_f1"])
        cur = best.get(t)
        if cur is None or f1 > cur[0]:
            best[t] = (f1, row)

rows = [v[1] for _, v in sorted(best.items(), key=lambda x: x[0])]
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["track", "mode", "run_name", "accuracy", "macro_f1"])
    w.writeheader()
    for r in rows:
        w.writerow(
            {
                "track": r["track"],
                "mode": r["mode"],
                "run_name": r["run_name"],
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
            }
        )
print(f"selected={len(rows)}")
PY

echo "track,mode,run_name,fp32_acc,fp32_f1,checkpoint,onnx_full,calib_dir,hef_path,build_status" > "$OUT_META"
echo "name,hef_local" > "$OUT_CANDIDATES"

while IFS=, read -r TRACK MODE RUN_NAME ACC F1; do
  if [[ "$TRACK" == "track" ]]; then
    continue
  fi
  TRACK="${TRACK//$'\r'/}"
  MODE="${MODE//$'\r'/}"
  RUN_NAME="${RUN_NAME//$'\r'/}"
  ACC="${ACC//$'\r'/}"
  F1="${F1//$'\r'/}"
  RUN_DIR="${RESULTS_ROOT}/${RUN_NAME}"
  CKPT="${RUN_DIR}/checkpoints/best_fold_0.pt"
  ONNX_FULL="${ONNX_DIR}/${RUN_NAME}_full.onnx"
  CALIB_DIR="${CALIB_ROOT}/${RUN_NAME}_train${MAX_CALIB}"
  HEF_PATH="${BUILD_ROOT}/${RUN_NAME}/${RUN_NAME}.hef"
  BUILD_STATUS="ok"

  if [[ ! -f "$CKPT" ]]; then
    BUILD_STATUS="missing_checkpoint"
    echo "${TRACK},${MODE},${RUN_NAME},${ACC},${F1},${CKPT},${ONNX_FULL},${CALIB_DIR},${HEF_PATH},${BUILD_STATUS}" >> "$OUT_META"
    continue
  fi

  TRAIN_LIST="${SPLIT_DIR}/${TRACK}_train.txt"
  if [[ ! -f "$TRAIN_LIST" ]]; then
    BUILD_STATUS="missing_train_list"
    echo "${TRACK},${MODE},${RUN_NAME},${ACC},${F1},${CKPT},${ONNX_FULL},${CALIB_DIR},${HEF_PATH},${BUILD_STATUS}" >> "$OUT_META"
    continue
  fi

  echo "[PHASE36-HAILO] export onnx: ${RUN_NAME}"
  if [[ "$SKIP_EXISTING" == "1" && -f "$ONNX_FULL" ]]; then
    echo "[PHASE36-HAILO] skip onnx (exists): ${ONNX_FULL}"
  else
    "$VENV_FP32/bin/python" hailo/export_onnx.py \
      --run-dir "$RUN_DIR" \
      --fold 0 \
      --onnx-dir "$ONNX_DIR" \
      --name "$RUN_NAME" >/dev/null
  fi

  echo "[PHASE36-HAILO] build calib: ${RUN_NAME}"
  if [[ "$SKIP_EXISTING" == "1" && -f "$CALIB_DIR/index.csv" ]]; then
    echo "[PHASE36-HAILO] skip calib (exists): ${CALIB_DIR}/index.csv"
  else
    "$VENV_FP32/bin/python" hailo/calib_dump_npy_dir.py \
      --manifest "$MANIFEST" \
      --cache-dir "$CACHE_DIR" \
      --split-list "$TRAIN_LIST" \
      --out-dir "$CALIB_DIR" \
      --max-samples "$MAX_CALIB" \
      --seed 1337 \
      --normalize-checkpoint "$CKPT" >/dev/null
  fi

  if [[ "$COMPILE" == "1" ]]; then
    echo "[PHASE36-HAILO] compile hef: ${RUN_NAME}"
    if [[ "$SKIP_EXISTING" == "1" && -f "$HEF_PATH" ]]; then
      echo "[PHASE36-HAILO] skip compile (exists): ${HEF_PATH}"
    else
      if ! bash scripts/run_e4_compile_local.sh \
        --venv "$VENV_HAILO" \
        --network-name "$RUN_NAME" \
        --onnx "$ONNX_FULL" \
        --calib-dir "$CALIB_DIR" \
        --out-dir "${BUILD_ROOT}/${RUN_NAME}" \
        --optimization-level 0 \
        --compression-level 0 \
        --max-calib "$MAX_CALIB" >/dev/null; then
        BUILD_STATUS="compile_failed"
      fi
    fi
  else
    BUILD_STATUS="prepared_no_compile"
  fi

  if [[ -f "$HEF_PATH" ]]; then
    echo "${RUN_NAME},${HEF_PATH}" >> "$OUT_CANDIDATES"
  fi
  echo "${TRACK},${MODE},${RUN_NAME},${ACC},${F1},${CKPT},${ONNX_FULL},${CALIB_DIR},${HEF_PATH},${BUILD_STATUS}" >> "$OUT_META"
done < "$SELECT_CSV"

rm -f "$SELECT_CSV"

echo "[OK] phase36 hailo prep done"
echo " - meta       : $OUT_META"
echo " - candidates : $OUT_CANDIDATES"
