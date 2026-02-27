#!/usr/bin/env bash
set -euo pipefail

# Run E4 HEF compilation on a remote x86 host that has Hailo SDK/DFC.
#
# Examples:
#   bash scripts/run_e4_compile_remote.sh \
#     --host user@x86-host \
#     --network-name fp32_v8_fold0
#
#   REMOTE_PASSWORD='***' bash scripts/run_e4_compile_remote.sh \
#     --host user@x86-host \
#     --network-name fp32_v8_fold0 \
#     --use-password

HOST=""
REMOTE_WORKDIR="~/work/hailo_compile_stage"
NETWORK_NAME=""
ONNX_PATH="derived/hailo/onnx/fp32_multitask_phase35_v8_hubert_gated_wide_tune4_fold0_full.onnx"
CALIB_DIR="derived/hailo/calib/fold0_train_1024"
LOCAL_OUT_DIR=""
USE_PASSWORD=0
STRIP_LAYERNORM=1
OPTIMIZATION_LEVEL=0
COMPRESSION_LEVEL=0
MAX_CALIB=1024

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --remote-workdir) REMOTE_WORKDIR="$2"; shift 2 ;;
    --network-name) NETWORK_NAME="$2"; shift 2 ;;
    --onnx) ONNX_PATH="$2"; shift 2 ;;
    --calib-dir) CALIB_DIR="$2"; shift 2 ;;
    --local-out-dir) LOCAL_OUT_DIR="$2"; shift 2 ;;
    --no-strip-layernorm) STRIP_LAYERNORM=0; shift ;;
    --optimization-level) OPTIMIZATION_LEVEL="$2"; shift 2 ;;
    --compression-level) COMPRESSION_LEVEL="$2"; shift 2 ;;
    --max-calib) MAX_CALIB="$2"; shift 2 ;;
    --use-password) USE_PASSWORD=1; shift ;;
    *) echo "[ERROR] Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$HOST" ]]; then
  echo "[ERROR] --host is required (example: user@x86-host)"
  exit 1
fi
if [[ -z "$NETWORK_NAME" ]]; then
  echo "[ERROR] --network-name is required"
  exit 1
fi
if [[ ! -f "$ONNX_PATH" ]]; then
  echo "[ERROR] Missing ONNX: $ONNX_PATH"
  exit 1
fi
if [[ ! -d "$CALIB_DIR" ]]; then
  echo "[ERROR] Missing calib dir: $CALIB_DIR"
  exit 1
fi

if [[ -z "$LOCAL_OUT_DIR" ]]; then
  LOCAL_OUT_DIR="derived/hailo/build/${NETWORK_NAME}"
fi
mkdir -p "$LOCAL_OUT_DIR"

if [[ "$USE_PASSWORD" -eq 1 ]]; then
  if [[ -z "${REMOTE_PASSWORD:-}" ]]; then
    echo "[ERROR] --use-password requested but REMOTE_PASSWORD is empty."
    exit 2
  fi
  if command -v sshpass >/dev/null 2>&1; then
    SSH_BASE=(sshpass -p "$REMOTE_PASSWORD" ssh -o StrictHostKeyChecking=accept-new)
    SCP_BASE=(sshpass -p "$REMOTE_PASSWORD" scp -o StrictHostKeyChecking=accept-new)
  else
    ASKPASS_SCRIPT="$(mktemp)"
    trap 'rm -f "${TMP_TAR:-}" "${ASKPASS_SCRIPT:-}"' EXIT
    cat >"$ASKPASS_SCRIPT" <<EOF
#!/usr/bin/env sh
echo '$REMOTE_PASSWORD'
EOF
    chmod 700 "$ASKPASS_SCRIPT"
    SSH_BASE=(env SSH_ASKPASS="$ASKPASS_SCRIPT" SSH_ASKPASS_REQUIRE=force DISPLAY=:0 setsid ssh -o StrictHostKeyChecking=accept-new)
    SCP_BASE=(env SSH_ASKPASS="$ASKPASS_SCRIPT" SSH_ASKPASS_REQUIRE=force DISPLAY=:0 setsid scp -o StrictHostKeyChecking=accept-new)
  fi
else
  SSH_BASE=(ssh -o StrictHostKeyChecking=accept-new)
  SCP_BASE=(scp -o StrictHostKeyChecking=accept-new)
fi

echo "[E4-REMOTE 0/4] Preflight remote toolchain ..."
REMOTE_PREFLIGHT_CMD=$(cat <<'EOF'
set -euo pipefail
ARCH="$(uname -m || true)"
MODEL="$(tr -d '\0' </proc/device-tree/model 2>/dev/null || true)"
HAS_HAILO=0
HAS_HAILORTCLI=0
command -v hailo >/dev/null 2>&1 && HAS_HAILO=1
command -v hailortcli >/dev/null 2>&1 && HAS_HAILORTCLI=1
PY_OK=0
if command -v python >/dev/null 2>&1; then
  python - <<'PY' >/dev/null 2>&1 && PY_OK=1 || PY_OK=0
from hailo_sdk_client import ClientRunner  # noqa: F401
PY
fi
printf "ARCH=%s\nMODEL=%s\nPY_SDK_OK=%s\nHAS_HAILO=%s\nHAS_HAILORTCLI=%s\n" "$ARCH" "$MODEL" "$PY_OK" "$HAS_HAILO" "$HAS_HAILORTCLI"
EOF
)

set +e
PREFLIGHT_OUT="$("${SSH_BASE[@]}" "$HOST" "$REMOTE_PREFLIGHT_CMD" 2>&1)"
PREFLIGHT_RC=$?
set -e
if [[ "$PREFLIGHT_RC" -ne 0 ]]; then
  echo "[ERROR] Remote preflight failed."
  echo "$PREFLIGHT_OUT"
  exit "$PREFLIGHT_RC"
fi
echo "$PREFLIGHT_OUT"
if echo "$PREFLIGHT_OUT" | grep -q "PY_SDK_OK=0"; then
  echo "[BLOCKED] Remote host python cannot import hailo_sdk_client."
  echo "         This host can run runtime inference checks, but cannot compile HEF."
  echo "         Install Hailo SDK/DFC there and rerun this script."
  exit 12
fi

echo "[E4-REMOTE 1/4] Create remote workspace ..."
"${SSH_BASE[@]}" "$HOST" "mkdir -p ${REMOTE_WORKDIR}/derived/hailo/onnx ${REMOTE_WORKDIR}/derived/hailo/calib ${REMOTE_WORKDIR}/hailo"

echo "[E4-REMOTE 2/4] Upload ONNX, calib, compile script (single archive) ..."
TMP_TAR="$(mktemp /tmp/e4_remote_stage.XXXXXX.tar.gz)"
trap 'rm -f "$TMP_TAR" "${ASKPASS_SCRIPT:-}"' EXIT
tar czf "$TMP_TAR" "$ONNX_PATH" "$CALIB_DIR" "hailo/compile_hef.sh"
"${SCP_BASE[@]}" "$TMP_TAR" "$HOST:${REMOTE_WORKDIR}/e4_stage.tar.gz"
"${SSH_BASE[@]}" "$HOST" "cd ${REMOTE_WORKDIR} && tar xzf e4_stage.tar.gz && rm -f e4_stage.tar.gz"

REMOTE_CMD=$(cat <<EOF
set -euo pipefail
cd ${REMOTE_WORKDIR}
chmod +x hailo/compile_hef.sh
if ! python - <<'PY' >/dev/null 2>&1; then
from hailo_sdk_client import ClientRunner  # noqa: F401
PY
  echo "[REMOTE-ERROR] hailo_sdk_client import failed."
  exit 12
fi
EXTRA_STRIP_FLAG=""
if [[ "${STRIP_LAYERNORM}" -eq 0 ]]; then
  EXTRA_STRIP_FLAG="--no-strip-layernorm"
fi
bash hailo/compile_hef.sh \\
  --onnx ${ONNX_PATH} \\
  --calib-dir ${CALIB_DIR} \\
  --out-dir derived/hailo/build/${NETWORK_NAME} \\
  --network-name ${NETWORK_NAME} \\
  --optimization-level ${OPTIMIZATION_LEVEL} \\
  --compression-level ${COMPRESSION_LEVEL} \\
  --max-calib ${MAX_CALIB} \\
  ${EXTRA_STRIP_FLAG} \\
  --execute
EOF
)

echo "[E4-REMOTE 3/4] Run compile on remote host ..."
set +e
"${SSH_BASE[@]}" "$HOST" "$REMOTE_CMD"
RC=$?
set -e
if [[ "$RC" -ne 0 ]]; then
  echo "[ERROR] Remote compile failed with code=${RC}."
  echo "Check remote environment (hailo_sdk_client/DFC install) and retry."
  exit 12
fi

echo "[E4-REMOTE 4/4] Download HEF artifacts ..."
"${SCP_BASE[@]}" -r "$HOST:${REMOTE_WORKDIR}/derived/hailo/build/${NETWORK_NAME}/." "$LOCAL_OUT_DIR/"

echo "[OK] Remote compile done."
echo " - Local artifacts: $LOCAL_OUT_DIR"
