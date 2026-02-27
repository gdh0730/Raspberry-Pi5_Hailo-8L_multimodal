#!/usr/bin/env bash
set -euo pipefail

# Run one real inference on Pi using hailo_platform Python API and fetch outputs.
#
# Example:
#   PI_PASSWORD='***' bash scripts/run_e5_infer_pi.sh \
#     --host wormhole@129.254.232.91 \
#     --hef-local derived/hailo/build/fp32_v8_fold0/fp32_v8_fold0.hef \
#     --audio-npy derived/hailo/calib/fold0_train_1024/audio/00000.npy \
#     --video-npy derived/hailo/calib/fold0_train_1024/video/00000.npy \
#     --name fp32_v8_fold0_sample0 \
#     --use-password

HOST=""
HEF_REMOTE=""
HEF_LOCAL=""
AUDIO_NPY="derived/hailo/calib/fold0_train_1024/audio/00000.npy"
VIDEO_NPY="derived/hailo/calib/fold0_train_1024/video/00000.npy"
NAME="fp32_v8_fold0_sample0"
REMOTE_WORKDIR="~/work/raspi_hailo_stage"
OUT_DIR="derived/hailo/pi_infer"
USE_PASSWORD=0
STRICT_SHAPE=0
EMOTION_CLASSES="angry,disgust,fearful,happy,neutral,sad"
NORMALIZE_CHECKPOINT=""

PY_BIN=""
if [[ -x ".venv/bin/python" ]]; then
  PY_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PY_BIN="python"
else
  echo "[ERROR] python executable not found."
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --hef) HEF_REMOTE="$2"; shift 2 ;;
    --hef-local) HEF_LOCAL="$2"; shift 2 ;;
    --audio-npy) AUDIO_NPY="$2"; shift 2 ;;
    --video-npy) VIDEO_NPY="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --remote-workdir) REMOTE_WORKDIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --emotion-classes) EMOTION_CLASSES="$2"; shift 2 ;;
    --normalize-checkpoint) NORMALIZE_CHECKPOINT="$2"; shift 2 ;;
    --strict-shape) STRICT_SHAPE=1; shift ;;
    --use-password) USE_PASSWORD=1; shift ;;
    *) echo "[ERROR] Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$HOST" ]]; then
  echo "[ERROR] --host is required"
  exit 1
fi
if [[ -n "$HEF_REMOTE" && -n "$HEF_LOCAL" ]]; then
  echo "[ERROR] Use either --hef or --hef-local, not both."
  exit 1
fi
if [[ -z "$HEF_REMOTE" && -z "$HEF_LOCAL" ]]; then
  echo "[ERROR] One of --hef / --hef-local is required."
  exit 1
fi
if [[ -n "$HEF_LOCAL" && ! -f "$HEF_LOCAL" ]]; then
  echo "[ERROR] Missing local HEF: $HEF_LOCAL"
  exit 1
fi
if [[ ! -f "$AUDIO_NPY" ]]; then
  echo "[ERROR] Missing local audio npy: $AUDIO_NPY"
  exit 1
fi
if [[ ! -f "$VIDEO_NPY" ]]; then
  echo "[ERROR] Missing local video npy: $VIDEO_NPY"
  exit 1
fi
if [[ -n "$NORMALIZE_CHECKPOINT" && ! -f "$NORMALIZE_CHECKPOINT" ]]; then
  echo "[ERROR] Missing --normalize-checkpoint file: $NORMALIZE_CHECKPOINT"
  exit 1
fi

mkdir -p "$OUT_DIR"

LOCAL_AUDIO_UPLOAD="$AUDIO_NPY"
LOCAL_VIDEO_UPLOAD="$VIDEO_NPY"
if [[ -n "$NORMALIZE_CHECKPOINT" ]]; then
  TMP_INPUT_DIR="$(mktemp -d)"
  trap 'rm -rf "${TMP_INPUT_DIR:-}" "${ASKPASS_SCRIPT:-}"' EXIT
  LOCAL_AUDIO_UPLOAD="${TMP_INPUT_DIR}/audio.npy"
  LOCAL_VIDEO_UPLOAD="${TMP_INPUT_DIR}/video.npy"
  "$PY_BIN" - "$NORMALIZE_CHECKPOINT" "$AUDIO_NPY" "$VIDEO_NPY" "$LOCAL_AUDIO_UPLOAD" "$LOCAL_VIDEO_UPLOAD" <<'PY'
import sys
from pathlib import Path
import numpy as np
import torch

ckpt = Path(sys.argv[1])
audio_in = Path(sys.argv[2])
video_in = Path(sys.argv[3])
audio_out = Path(sys.argv[4])
video_out = Path(sys.argv[5])
try:
    s = torch.load(ckpt, map_location="cpu", weights_only=False)
except TypeError:
    s = torch.load(ckpt, map_location="cpu")
mu_a = np.asarray(s["audio_mu"], dtype=np.float32).reshape(-1)
sd_a = np.asarray(s["audio_sd"], dtype=np.float32).reshape(-1)
mu_v = np.asarray(s["video_mu"], dtype=np.float32).reshape(-1)
sd_v = np.asarray(s["video_sd"], dtype=np.float32).reshape(-1)
sd_a = np.where(sd_a < 1e-6, 1.0, sd_a)
sd_v = np.where(sd_v < 1e-6, 1.0, sd_v)
xa = np.load(audio_in).astype(np.float32).reshape(1, -1)
xv = np.load(video_in).astype(np.float32).reshape(1, -1)
if xa.shape[1] != mu_a.shape[0]:
    raise ValueError(f"audio dim mismatch: {xa.shape[1]} vs {mu_a.shape[0]}")
if xv.shape[1] != mu_v.shape[0]:
    raise ValueError(f"video dim mismatch: {xv.shape[1]} vs {mu_v.shape[0]}")
np.save(audio_out, ((xa - mu_a) / sd_a).astype(np.float32))
np.save(video_out, ((xv - mu_v) / sd_v).astype(np.float32))
PY
fi

if [[ "$USE_PASSWORD" -eq 1 ]]; then
  if [[ -z "${PI_PASSWORD:-}" ]]; then
    echo "[ERROR] --use-password requested but PI_PASSWORD is empty."
    exit 2
  fi
  if command -v sshpass >/dev/null 2>&1; then
    SSH_BASE=(sshpass -p "$PI_PASSWORD" ssh -o StrictHostKeyChecking=accept-new)
    SCP_BASE=(sshpass -p "$PI_PASSWORD" scp -o StrictHostKeyChecking=accept-new)
  else
    ASKPASS_SCRIPT="$(mktemp)"
    trap 'rm -rf "${TMP_INPUT_DIR:-}" ; rm -f "${ASKPASS_SCRIPT:-}"' EXIT
    cat >"$ASKPASS_SCRIPT" <<EOF
#!/usr/bin/env sh
echo '$PI_PASSWORD'
EOF
    chmod 700 "$ASKPASS_SCRIPT"
    SSH_BASE=(env SSH_ASKPASS="$ASKPASS_SCRIPT" SSH_ASKPASS_REQUIRE=force DISPLAY=:0 setsid ssh -o StrictHostKeyChecking=accept-new)
    SCP_BASE=(env SSH_ASKPASS="$ASKPASS_SCRIPT" SSH_ASKPASS_REQUIRE=force DISPLAY=:0 setsid scp -o StrictHostKeyChecking=accept-new)
  fi
else
  SSH_BASE=(ssh -o StrictHostKeyChecking=accept-new)
  SCP_BASE=(scp -o StrictHostKeyChecking=accept-new)
fi

REMOTE_HEF="$HEF_REMOTE"
if [[ -n "$HEF_LOCAL" ]]; then
  REMOTE_HEF="${REMOTE_WORKDIR}/hefs/${NAME}.hef"
fi

REMOTE_AUDIO="${REMOTE_WORKDIR}/infer_inputs/${NAME}_audio.npy"
REMOTE_VIDEO="${REMOTE_WORKDIR}/infer_inputs/${NAME}_video.npy"
REMOTE_SCRIPT="${REMOTE_WORKDIR}/hailo/pi_infer_hailort.py"
REMOTE_OUT_JSON="${REMOTE_WORKDIR}/infer_outputs/${NAME}.json"
REMOTE_LOG="${REMOTE_WORKDIR}/infer_outputs/${NAME}.log"

echo "[E5-INFER 1/4] Upload artifacts to Pi ..."
"${SSH_BASE[@]}" "$HOST" "mkdir -p ${REMOTE_WORKDIR}/hefs ${REMOTE_WORKDIR}/infer_inputs ${REMOTE_WORKDIR}/infer_outputs ${REMOTE_WORKDIR}/hailo"
if [[ -n "$HEF_LOCAL" ]]; then
  "${SCP_BASE[@]}" "$HEF_LOCAL" "$HOST:${REMOTE_HEF}"
fi
"${SCP_BASE[@]}" "$LOCAL_AUDIO_UPLOAD" "$HOST:${REMOTE_AUDIO}"
"${SCP_BASE[@]}" "$LOCAL_VIDEO_UPLOAD" "$HOST:${REMOTE_VIDEO}"
"${SCP_BASE[@]}" "hailo/pi_infer_hailort.py" "$HOST:${REMOTE_SCRIPT}"

STRICT_ARG=""
if [[ "$STRICT_SHAPE" -eq 1 ]]; then
  STRICT_ARG="--strict-shape"
fi

REMOTE_CMD=$(cat <<EOF
set -euo pipefail
python3 ${REMOTE_SCRIPT} \
  --hef ${REMOTE_HEF} \
  --audio-npy ${REMOTE_AUDIO} \
  --video-npy ${REMOTE_VIDEO} \
  --emotion-classes '${EMOTION_CLASSES}' \
  --out-json ${REMOTE_OUT_JSON} \
  ${STRICT_ARG} | tee ${REMOTE_LOG}
EOF
)

echo "[E5-INFER 2/4] Run inference on Pi ..."
"${SSH_BASE[@]}" "$HOST" "$REMOTE_CMD"

echo "[E5-INFER 3/4] Download inference outputs ..."
"${SCP_BASE[@]}" "$HOST:${REMOTE_OUT_JSON}" "${OUT_DIR}/${NAME}.json"
"${SCP_BASE[@]}" "$HOST:${REMOTE_LOG}" "${OUT_DIR}/${NAME}.log"

echo "[E5-INFER 4/4] Done."
echo " - JSON: ${OUT_DIR}/${NAME}.json"
echo " - Log : ${OUT_DIR}/${NAME}.log"
