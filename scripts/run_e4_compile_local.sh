#!/usr/bin/env bash
set -euo pipefail

# Run E4 compile directly on local host (x86 + hailomz installed).
#
# Example:
#   bash scripts/run_e4_compile_local.sh \
#     --network-name fp32_v8_fold0 \
#     --venv .venv-hailo

VENV_DIR=".venv-hailo"
NETWORK_NAME=""
ONNX_PATH="derived/hailo/onnx/fp32_multitask_phase35_v8_hubert_gated_wide_tune4_fold0_full.onnx"
CALIB_DIR="derived/hailo/calib/fold0_train_1024"
OUT_DIR=""
STRIP_LAYERNORM=1
OPTIMIZATION_LEVEL=0
COMPRESSION_LEVEL=0
MAX_CALIB=1024

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VENV_DIR="$2"; shift 2 ;;
    --network-name) NETWORK_NAME="$2"; shift 2 ;;
    --onnx) ONNX_PATH="$2"; shift 2 ;;
    --calib-dir) CALIB_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --no-strip-layernorm) STRIP_LAYERNORM=0; shift ;;
    --optimization-level) OPTIMIZATION_LEVEL="$2"; shift 2 ;;
    --compression-level) COMPRESSION_LEVEL="$2"; shift 2 ;;
    --max-calib) MAX_CALIB="$2"; shift 2 ;;
    *)
      echo "[ERROR] Unknown arg: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$NETWORK_NAME" ]]; then
  echo "[ERROR] --network-name is required"
  exit 2
fi
if [[ ! -f "$ONNX_PATH" ]]; then
  echo "[ERROR] Missing ONNX: $ONNX_PATH"
  exit 3
fi
if [[ ! -d "$CALIB_DIR" ]]; then
  echo "[ERROR] Missing calib dir: $CALIB_DIR"
  exit 4
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "[ERROR] Missing venv python: $VENV_DIR/bin/python"
  echo "Run setup first: bash scripts/setup_hailo_compile_env.sh ..."
  exit 5
fi
if ! "$VENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1
from hailo_sdk_client import ClientRunner  # noqa: F401
PY
then
  echo "[ERROR] hailo_sdk_client import failed in $VENV_DIR"
  echo "Run setup first: bash scripts/setup_hailo_compile_env.sh ..."
  exit 6
fi

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="derived/hailo/build/${NETWORK_NAME}"
fi
mkdir -p "$OUT_DIR"

echo "[LOCAL-E4] Using venv: $VENV_DIR"
echo "[LOCAL-E4] ONNX      : $ONNX_PATH"
echo "[LOCAL-E4] CALIB     : $CALIB_DIR"
echo "[LOCAL-E4] OUT       : $OUT_DIR"
echo "[LOCAL-E4] STRIP_LN  : $STRIP_LAYERNORM"
echo "[LOCAL-E4] OPT_LEVEL : $OPTIMIZATION_LEVEL"
echo "[LOCAL-E4] COMP_LEVEL: $COMPRESSION_LEVEL"
echo "[LOCAL-E4] MAX_CALIB : $MAX_CALIB"

if [[ "$VENV_DIR" = /* ]]; then
  VENV_BIN="$VENV_DIR/bin"
else
  VENV_BIN="$PWD/$VENV_DIR/bin"
fi

CMD=(
  bash hailo/compile_hef.sh
  --onnx "$ONNX_PATH"
  --calib-dir "$CALIB_DIR"
  --out-dir "$OUT_DIR"
  --network-name "$NETWORK_NAME"
  --optimization-level "$OPTIMIZATION_LEVEL"
  --compression-level "$COMPRESSION_LEVEL"
  --max-calib "$MAX_CALIB"
)
if [[ "$STRIP_LAYERNORM" -eq 0 ]]; then
  CMD+=(--no-strip-layernorm)
fi
CMD+=(--execute)

PY_BIN="$VENV_BIN/python" PATH="$VENV_BIN:$PATH" "${CMD[@]}"

echo "[OK] Local compile completed."
echo " - HEF: $OUT_DIR/${NETWORK_NAME}.hef"
