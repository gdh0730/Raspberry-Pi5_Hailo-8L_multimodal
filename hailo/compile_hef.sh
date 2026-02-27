#!/usr/bin/env bash
set -euo pipefail

# Hailo E4 compile entrypoint.
# Uses hailo_sdk_client directly (custom ONNX + npy calibration set).
#
# Usage:
#   bash hailo/compile_hef.sh \
#     --onnx derived/hailo/onnx/<name>_full.onnx \
#     --calib-dir derived/hailo/calib/fold0_train_1024 \
#     --out-dir derived/hailo/build/<name> \
#     --network-name <name> \
#     [--execute]

ONNX_PATH=""
CALIB_DIR=""
OUT_DIR=""
NETWORK_NAME=""
EXECUTE=0
HW_ARCH="hailo8l"
STRIP_LAYERNORM=1
OPTIMIZATION_LEVEL=0
COMPRESSION_LEVEL=0
MAX_CALIB=1024

while [[ $# -gt 0 ]]; do
  case "$1" in
    --onnx) ONNX_PATH="$2"; shift 2 ;;
    --calib-dir) CALIB_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --network-name) NETWORK_NAME="$2"; shift 2 ;;
    --hw-arch) HW_ARCH="$2"; shift 2 ;;
    --optimization-level) OPTIMIZATION_LEVEL="$2"; shift 2 ;;
    --compression-level) COMPRESSION_LEVEL="$2"; shift 2 ;;
    --max-calib) MAX_CALIB="$2"; shift 2 ;;
    --no-strip-layernorm) STRIP_LAYERNORM=0; shift ;;
    --execute) EXECUTE=1; shift ;;
    *) echo "[ERROR] Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$ONNX_PATH" || -z "$CALIB_DIR" || -z "$OUT_DIR" || -z "$NETWORK_NAME" ]]; then
  echo "[ERROR] Missing required args. See script header."
  exit 1
fi

mkdir -p "$OUT_DIR"
HAR_PATH="$OUT_DIR/${NETWORK_NAME}.har"
HEF_PATH="$OUT_DIR/${NETWORK_NAME}.hef"

PY_BIN="${PY_BIN:-}"
if [[ -z "$PY_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PY_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PY_BIN="python3"
  else
    echo "[ERROR] python/python3 not found in PATH."
    exit 2
  fi
fi

# Some installations (docker-derived wheels) need an explicit runtime lib path
# for libhailort.so at import time.
RUNTIME_LIB_DIR="${HAILO_RUNTIME_LIB_DIR:-third_party/hailo_runtime_libs}"
if [[ -d "$RUNTIME_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="$PWD/$RUNTIME_LIB_DIR:${LD_LIBRARY_PATH:-}"
  echo "[INFO] LD_LIBRARY_PATH += $PWD/$RUNTIME_LIB_DIR"
fi

if ! "$PY_BIN" - <<'PY' >/dev/null 2>&1
from hailo_sdk_client import ClientRunner  # noqa: F401
PY
then
  echo "[ERROR] hailo_sdk_client import failed in current python."
  echo "        Run setup first: bash scripts/setup_hailo_compile_env.sh ..."
  exit 3
fi

CMD=(
  "$PY_BIN" hailo/compile_custom_onnx_sdk.py
  --onnx "$ONNX_PATH"
  --calib-dir "$CALIB_DIR"
  --out-dir "$OUT_DIR"
  --network-name "$NETWORK_NAME"
  --hw-arch "$HW_ARCH"
  --optimization-level "$OPTIMIZATION_LEVEL"
  --compression-level "$COMPRESSION_LEVEL"
  --max-calib "$MAX_CALIB"
)
if [[ "$STRIP_LAYERNORM" -eq 1 ]]; then
  CMD+=(--strip-layernorm)
else
  CMD+=(--no-strip-layernorm)
fi

echo "[INFO] compile command: ${CMD[*]}"
echo "[INFO] outputs:"
echo "  - HAR: $HAR_PATH"
echo "  - HEF: $HEF_PATH"

if [[ "$EXECUTE" -ne 1 ]]; then
  echo "[INFO] Dry-run mode. Add --execute to run commands."
  exit 0
fi

"${CMD[@]}"

echo "[OK] HEF generated: $HEF_PATH"
