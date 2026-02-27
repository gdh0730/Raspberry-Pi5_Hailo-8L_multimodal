#!/usr/bin/env bash
set -euo pipefail

# Run one Phase-36 experiment:
# 1) Train FP32 model on track train/val split
# 2) Build normalized test npy/index
# 3) Evaluate checkpoint on test index
#
# Example:
#   bash scripts/run_phase36_fp32_single.sh \
#     --track ood_c2r \
#     --mode fusion \
#     --run-name phase36_ood_c2r_fusion_v5_lnfree

TRACK=""
MODE=""
RUN_NAME=""
MANIFEST="derived/manifests/manifest_multimodal_common6_av.jsonl"
CACHE_DIR="derived/features/cache_v5_hubert"
SPLIT_DIR="derived/splits/phase36_id_ood"
OUT_ROOT="derived/results/phase36"
EVAL_ROOT="derived/hailo/eval/phase36"
EPOCHS=40
BATCH_SIZE=128
LR=0.00045
WEIGHT_DECAY=0.0001
HIDDEN_DIM=512
EMB_DIM=256
DROPOUT=0.2
MODALITY_DROPOUT_P=0.05
LABEL_SMOOTHING=0.08
SEED=1337
N_BOOTSTRAP=100
DEVICE="auto"
FUSION_TYPE="gated"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --track) TRACK="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --split-dir) SPLIT_DIR="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --eval-root) EVAL_ROOT="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --hidden-dim) HIDDEN_DIM="$2"; shift 2 ;;
    --emb-dim) EMB_DIM="$2"; shift 2 ;;
    --dropout) DROPOUT="$2"; shift 2 ;;
    --modality-dropout-p) MODALITY_DROPOUT_P="$2"; shift 2 ;;
    --label-smoothing) LABEL_SMOOTHING="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --n-bootstrap) N_BOOTSTRAP="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --fusion-type) FUSION_TYPE="$2"; shift 2 ;;
    *) echo "[ERROR] Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$TRACK" ]]; then
  echo "[ERROR] --track is required"
  exit 1
fi
if [[ -z "$MODE" ]]; then
  echo "[ERROR] --mode is required"
  exit 1
fi
if [[ "$MODE" != "audio" && "$MODE" != "video" && "$MODE" != "fusion" ]]; then
  echo "[ERROR] --mode must be one of: audio, video, fusion"
  exit 1
fi
if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="phase36_${TRACK}_${MODE}_v5_lnfree"
fi

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[ERROR] .venv/bin/python missing"
  exit 1
fi
PY=".venv/bin/python"

TRAIN_LIST="${SPLIT_DIR}/${TRACK}_train.txt"
VAL_LIST="${SPLIT_DIR}/${TRACK}_val.txt"
TEST_LIST="${SPLIT_DIR}/${TRACK}_test.txt"
if [[ ! -f "$TRAIN_LIST" || ! -f "$VAL_LIST" || ! -f "$TEST_LIST" ]]; then
  echo "[ERROR] Missing split files for track=${TRACK} under ${SPLIT_DIR}"
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

OUT_DIR="${OUT_ROOT}/${RUN_NAME}"
EVAL_DIR="${EVAL_ROOT}/${RUN_NAME}_test"
mkdir -p "$OUT_DIR" "$EVAL_DIR"

if [[ "$MODE" != "fusion" ]]; then
  # Fusion-only option is irrelevant in single-modality mode.
  FUSION_TYPE="concat"
  MODALITY_DROPOUT_P="0.0"
fi

echo "[PHASE36 1/3] Train FP32 track=${TRACK} mode=${MODE} run=${RUN_NAME}"
"$PY" scripts/train_fp32_multitask.py \
  --manifest "$MANIFEST" \
  --train-list "$TRAIN_LIST" \
  --val-list "$VAL_LIST" \
  --out-dir "$OUT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --mode "$MODE" \
  --fusion-type "$FUSION_TYPE" \
  --modality-dropout-p "$MODALITY_DROPOUT_P" \
  --num-folds 1 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --hidden-dim "$HIDDEN_DIM" \
  --emb-dim "$EMB_DIM" \
  --dropout "$DROPOUT" \
  --emotion-loss ce \
  --label-smoothing "$LABEL_SMOOTHING" \
  --weighted-sampler \
  --no-head-layernorm \
  --seed "$SEED" \
  --n-bootstrap "$N_BOOTSTRAP" \
  --device "$DEVICE"

CKPT_PATH="${OUT_DIR}/checkpoints/best_fold_0.pt"
if [[ ! -f "$CKPT_PATH" ]]; then
  echo "[ERROR] Missing checkpoint: $CKPT_PATH"
  exit 1
fi

echo "[PHASE36 2/3] Build normalized test eval set (track=${TRACK})"
"$PY" hailo/calib_dump_npy_dir.py \
  --manifest "$MANIFEST" \
  --cache-dir "$CACHE_DIR" \
  --split-list "$TEST_LIST" \
  --out-dir "$EVAL_DIR" \
  --max-samples 0 \
  --seed "$SEED" \
  --normalize-checkpoint "$CKPT_PATH" >/dev/null

echo "[PHASE36 3/3] Evaluate FP32 on test index (same samples for future Hailo compare)"
"$PY" scripts/eval_fp32_checkpoint_on_index.py \
  --checkpoint "$CKPT_PATH" \
  --index-csv "$EVAL_DIR/index.csv" \
  --calib-dir "$EVAL_DIR" \
  --manifest "$MANIFEST" \
  --out-json "$OUT_DIR/fp32_test_eval.json" \
  --out-pred-csv "$OUT_DIR/fp32_test_predictions.csv" \
  --device cpu >/dev/null

"$PY" - "$TRACK" "$MODE" "$RUN_NAME" "$OUT_DIR" "$EVAL_DIR" "$TRAIN_LIST" "$VAL_LIST" "$TEST_LIST" <<'PY'
import json
import sys
from pathlib import Path

track, mode, run_name, out_dir, eval_dir, train_list, val_list, test_list = sys.argv[1:]
out = Path(out_dir)
eval_p = Path(eval_dir)
train_n = len([ln for ln in Path(train_list).read_text(encoding="utf-8").splitlines() if ln.strip()])
val_n = len([ln for ln in Path(val_list).read_text(encoding="utf-8").splitlines() if ln.strip()])
test_n = len([ln for ln in Path(test_list).read_text(encoding="utf-8").splitlines() if ln.strip()])
fp32 = json.loads((out / "fp32_test_eval.json").read_text(encoding="utf-8"))
meta = {
    "track": track,
    "mode": mode,
    "run_name": run_name,
    "split_counts": {"train": train_n, "val": val_n, "test": test_n},
    "paths": {
        "out_dir": str(out),
        "checkpoint": str(out / "checkpoints" / "best_fold_0.pt"),
        "train_summary": str(out / "summary.json"),
        "test_eval_json": str(out / "fp32_test_eval.json"),
        "test_pred_csv": str(out / "fp32_test_predictions.csv"),
        "eval_index_csv": str(eval_p / "index.csv"),
        "eval_dir": str(eval_p),
    },
    "fp32_test_emotion6": fp32.get("emotion6", {}),
}
(out / "phase36_run_meta.json").write_text(json.dumps(meta, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
print(json.dumps(meta, ensure_ascii=True, indent=2))
PY

echo "[OK] Phase36 single run completed: ${RUN_NAME}"

