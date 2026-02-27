#!/usr/bin/env bash
set -euo pipefail

# One-shot local setup:
# 1) Prepare Hailo wheels from suite dir/archive
# 2) Setup local compile env with v2.17 model-zoo
#
# Example:
#   bash scripts/run_e4_local_setup.sh \
#     --suite-dir /mnt/c/Users/<you>/Downloads/hailo_ai_sw_suite_x.y.z

SUITE_DIR=""
SUITE_ARCHIVE=""
WHEEL_DIR="third_party/hailo_wheels"
VENV_DIR=".venv-hailo"
MODEL_ZOO_DIR="third_party/hailo_model_zoo"
MODEL_ZOO_REF="v2.17"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite-dir) SUITE_DIR="$2"; shift 2 ;;
    --suite-archive) SUITE_ARCHIVE="$2"; shift 2 ;;
    --wheel-dir) WHEEL_DIR="$2"; shift 2 ;;
    --venv) VENV_DIR="$2"; shift 2 ;;
    --model-zoo-dir) MODEL_ZOO_DIR="$2"; shift 2 ;;
    --model-zoo-ref) MODEL_ZOO_REF="$2"; shift 2 ;;
    *)
      echo "[ERROR] Unknown arg: $1"
      exit 1
      ;;
  esac
done

echo "[E4-LOCAL-SETUP 1/2] Prepare wheels ..."
PREP_ARGS=(--out-dir "$WHEEL_DIR")
if [[ -n "$SUITE_DIR" ]]; then
  PREP_ARGS+=(--suite-dir "$SUITE_DIR")
fi
if [[ -n "$SUITE_ARCHIVE" ]]; then
  PREP_ARGS+=(--suite-archive "$SUITE_ARCHIVE")
fi
bash scripts/prepare_hailo_wheels.sh "${PREP_ARGS[@]}"

echo "[E4-LOCAL-SETUP 2/2] Setup compile env ..."
bash scripts/setup_hailo_compile_env.sh \
  --wheel-dir "$WHEEL_DIR" \
  --venv "$VENV_DIR" \
  --model-zoo-dir "$MODEL_ZOO_DIR" \
  --model-zoo-ref "$MODEL_ZOO_REF"

echo "[OK] Local setup completed."
echo "Next:"
echo "  bash scripts/run_e4_compile_local.sh --network-name fp32_v8_fold0 --venv $VENV_DIR"
