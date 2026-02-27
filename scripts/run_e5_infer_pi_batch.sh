#!/usr/bin/env bash
set -euo pipefail

# Run multi-sample Pi inference with progress logging, then evaluate results.
#
# Example:
#   PI_PASSWORD='***' bash scripts/run_e5_infer_pi_batch.sh \
#     --host wormhole@129.254.232.91 \
#     --hef-local derived/hailo/build/fp32_v8_fold0/fp32_v8_fold0.hef \
#     --max-samples 50 \
#     --use-password

HOST=""
HEF_REMOTE=""
HEF_LOCAL=""
NAME=""
CALIB_DIR="derived/hailo/calib/fold0_train_1024"
INDEX_CSV=""
MANIFEST="derived/manifests/manifest_multimodal_common6_av.jsonl"
START_IDX=0
MAX_SAMPLES=0
REMOTE_WORKDIR="~/work/raspi_hailo_stage"
OUT_DIR="derived/hailo/pi_infer_batch"
USE_PASSWORD=0
STRICT_SHAPE=0
EMOTION_CLASSES="angry,disgust,fearful,happy,neutral,sad"
BOOTSTRAP_N=500
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
    --name) NAME="$2"; shift 2 ;;
    --calib-dir) CALIB_DIR="$2"; shift 2 ;;
    --index-csv) INDEX_CSV="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --start-idx) START_IDX="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --remote-workdir) REMOTE_WORKDIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --emotion-classes) EMOTION_CLASSES="$2"; shift 2 ;;
    --strict-shape) STRICT_SHAPE=1; shift ;;
    --bootstrap-n) BOOTSTRAP_N="$2"; shift 2 ;;
    --normalize-checkpoint) NORMALIZE_CHECKPOINT="$2"; shift 2 ;;
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
if [[ -z "$INDEX_CSV" ]]; then
  INDEX_CSV="${CALIB_DIR}/index.csv"
fi
if [[ ! -f "$INDEX_CSV" ]]; then
  echo "[ERROR] Missing index CSV: $INDEX_CSV"
  exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "[ERROR] Missing manifest: $MANIFEST"
  exit 1
fi
if [[ -n "$NORMALIZE_CHECKPOINT" && ! -f "$NORMALIZE_CHECKPOINT" ]]; then
  echo "[ERROR] Missing --normalize-checkpoint file: $NORMALIZE_CHECKPOINT"
  exit 1
fi
if [[ -z "$NAME" ]]; then
  if [[ -n "$HEF_REMOTE" ]]; then
    NAME="$(basename "$HEF_REMOTE" .hef)"
  else
    NAME="$(basename "$HEF_LOCAL" .hef)"
  fi
fi

RUN_OUT_DIR="${OUT_DIR}/${NAME}"
JSON_DIR="${RUN_OUT_DIR}/json"
LOG_DIR="${RUN_OUT_DIR}/logs"
mkdir -p "$JSON_DIR" "$LOG_DIR"

RAW_PROGRESS_CSV="${RUN_OUT_DIR}/progress.csv"
PRED_CSV="${RUN_OUT_DIR}/predictions.csv"
SUMMARY_JSON="${RUN_OUT_DIR}/summary.json"
SUMMARY_BOOTSTRAP_JSON="${RUN_OUT_DIR}/summary_bootstrap.json"

printf "idx,clip_id,dataset,y_true_emotion,status,json_path,error\n" > "$RAW_PROGRESS_CSV"

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
    trap 'rm -f "${ASKPASS_SCRIPT:-}"' EXIT
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

echo "[E5-BATCH 1/5] Build sample list ..."
mapfile -t SAMPLE_LINES < <(
  "$PY_BIN" - "$INDEX_CSV" "$MANIFEST" "$START_IDX" "$MAX_SAMPLES" <<'PY'
import csv
import json
import sys
from pathlib import Path

index_csv = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
start_idx = int(sys.argv[3])
max_samples = int(sys.argv[4])

manifest = {}
with manifest_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        manifest[str(row["clip_id"])] = row

count = 0
with index_csv.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        idx = int(row["idx"])
        if idx < start_idx:
            continue
        clip_id = row["clip_id"]
        m = manifest.get(clip_id, {})
        dataset = row.get("dataset") or m.get("dataset", "")
        y_true = m.get("emotion6", "")
        audio_rel = row["audio_npy"]
        video_rel = row["video_npy"]
        print(f"{idx}\t{clip_id}\t{dataset}\t{y_true}\t{audio_rel}\t{video_rel}")
        count += 1
        if max_samples > 0 and count >= max_samples:
            break
PY
)

TOTAL="${#SAMPLE_LINES[@]}"
if [[ "$TOTAL" -eq 0 ]]; then
  echo "[ERROR] No samples selected. Check --start-idx/--max-samples/index CSV."
  exit 3
fi
echo "[E5-BATCH] selected_samples=${TOTAL}"

NORMALIZED_INPUT_DIR=""
if [[ -n "$NORMALIZE_CHECKPOINT" ]]; then
  NORMALIZED_INPUT_DIR="${RUN_OUT_DIR}/normalized_inputs"
  echo "[E5-BATCH 1.5/5] Build normalized local inputs ..."
  "$PY_BIN" scripts/normalize_hailo_inputs.py \
    --index-csv "$INDEX_CSV" \
    --calib-dir "$CALIB_DIR" \
    --checkpoint "$NORMALIZE_CHECKPOINT" \
    --out-dir "$NORMALIZED_INPUT_DIR" \
    --start-idx "$START_IDX" \
    --max-samples "$MAX_SAMPLES" >/dev/null
fi

REMOTE_HEF="$HEF_REMOTE"
if [[ -n "$HEF_LOCAL" ]]; then
  REMOTE_HEF="${REMOTE_WORKDIR}/hefs/${NAME}.hef"
fi

REMOTE_SCRIPT="${REMOTE_WORKDIR}/hailo/pi_infer_hailort.py"
REMOTE_INPUT_DIR="${REMOTE_WORKDIR}/infer_inputs"
REMOTE_OUT_DIR="${REMOTE_WORKDIR}/infer_outputs/${NAME}_batch"

echo "[E5-BATCH 2/5] Upload shared artifacts ..."
"${SSH_BASE[@]}" "$HOST" "mkdir -p ${REMOTE_WORKDIR}/hefs ${REMOTE_WORKDIR}/hailo ${REMOTE_INPUT_DIR} ${REMOTE_OUT_DIR}"
if [[ -n "$HEF_LOCAL" ]]; then
  "${SCP_BASE[@]}" "$HEF_LOCAL" "$HOST:${REMOTE_HEF}"
fi
"${SCP_BASE[@]}" "hailo/pi_infer_hailort.py" "$HOST:${REMOTE_SCRIPT}"

STRICT_ARG=""
if [[ "$STRICT_SHAPE" -eq 1 ]]; then
  STRICT_ARG="--strict-shape"
fi

echo "[E5-BATCH 3/5] Run per-sample inference on Pi ..."
OK_COUNT=0
FAIL_COUNT=0
for i in "${!SAMPLE_LINES[@]}"; do
  line="${SAMPLE_LINES[$i]}"
  IFS=$'\t' read -r IDX CLIP_ID DATASET Y_TRUE AUDIO_REL VIDEO_REL <<<"$line"
  CUR=$((i + 1))
  PCT=$((CUR * 100 / TOTAL))

  if [[ -n "$NORMALIZED_INPUT_DIR" ]]; then
    LOCAL_AUDIO="${NORMALIZED_INPUT_DIR}/audio/$(printf '%05d' "$IDX").npy"
    LOCAL_VIDEO="${NORMALIZED_INPUT_DIR}/video/$(printf '%05d' "$IDX").npy"
  else
    LOCAL_AUDIO="${CALIB_DIR}/${AUDIO_REL}"
    LOCAL_VIDEO="${CALIB_DIR}/${VIDEO_REL}"
  fi
  SAMPLE_TAG="$(printf '%s_%05d' "$NAME" "$IDX")"
  REMOTE_AUDIO="${REMOTE_INPUT_DIR}/${SAMPLE_TAG}_audio.npy"
  REMOTE_VIDEO="${REMOTE_INPUT_DIR}/${SAMPLE_TAG}_video.npy"
  REMOTE_JSON="${REMOTE_OUT_DIR}/${SAMPLE_TAG}.json"
  REMOTE_LOG="${REMOTE_OUT_DIR}/${SAMPLE_TAG}.log"
  LOCAL_JSON="${JSON_DIR}/$(printf '%05d' "$IDX").json"
  LOCAL_LOG="${LOG_DIR}/$(printf '%05d' "$IDX").log"

  echo "[E5-BATCH ${CUR}/${TOTAL} ${PCT}%] idx=${IDX} clip=${CLIP_ID}"
  STATUS="ok"
  ERR=""

  if [[ ! -f "$LOCAL_AUDIO" ]]; then
    STATUS="failed"
    ERR="missing_audio_npy"
  elif [[ ! -f "$LOCAL_VIDEO" ]]; then
    STATUS="failed"
    ERR="missing_video_npy"
  fi

  if [[ "$STATUS" == "ok" ]]; then
    set +e
    "${SCP_BASE[@]}" "$LOCAL_AUDIO" "$HOST:${REMOTE_AUDIO}" >/dev/null 2>&1
    rc_audio=$?
    "${SCP_BASE[@]}" "$LOCAL_VIDEO" "$HOST:${REMOTE_VIDEO}" >/dev/null 2>&1
    rc_video=$?
    if [[ "$rc_audio" -ne 0 || "$rc_video" -ne 0 ]]; then
      STATUS="failed"
      ERR="scp_input_failed"
    fi
    set -e
  fi

  if [[ "$STATUS" == "ok" ]]; then
    set +e
    "${SSH_BASE[@]}" "$HOST" "set -euo pipefail; python3 ${REMOTE_SCRIPT} --hef ${REMOTE_HEF} --audio-npy ${REMOTE_AUDIO} --video-npy ${REMOTE_VIDEO} --emotion-classes '${EMOTION_CLASSES}' --out-json ${REMOTE_JSON} ${STRICT_ARG} > ${REMOTE_LOG}" >/dev/null 2>&1
    rc_infer=$?
    set -e
    if [[ "$rc_infer" -ne 0 ]]; then
      STATUS="failed"
      ERR="remote_infer_failed"
    fi
  fi

  if [[ "$STATUS" == "ok" ]]; then
    set +e
    "${SCP_BASE[@]}" "$HOST:${REMOTE_JSON}" "$LOCAL_JSON" >/dev/null 2>&1
    rc_json=$?
    "${SCP_BASE[@]}" "$HOST:${REMOTE_LOG}" "$LOCAL_LOG" >/dev/null 2>&1
    rc_log=$?
    set -e
    if [[ "$rc_json" -ne 0 ]]; then
      STATUS="failed"
      ERR="scp_output_failed"
    elif [[ "$rc_log" -ne 0 ]]; then
      # log fetch failure is non-fatal when JSON exists.
      ERR="missing_remote_log"
    fi
  fi

  if [[ "$STATUS" == "ok" ]]; then
    OK_COUNT=$((OK_COUNT + 1))
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi

  printf "%s,%s,%s,%s,%s,%s,%s\n" \
    "$IDX" "$CLIP_ID" "$DATASET" "$Y_TRUE" "$STATUS" "$LOCAL_JSON" "$ERR" >> "$RAW_PROGRESS_CSV"
done

echo "[E5-BATCH 4/5] Evaluate predictions ..."
"$PY_BIN" scripts/eval_hailo_pi_infer.py \
  --index-csv "$INDEX_CSV" \
  --manifest "$MANIFEST" \
  --json-dir "$JSON_DIR" \
  --start-idx "$START_IDX" \
  --max-samples "$MAX_SAMPLES" \
  --emotion-classes "$EMOTION_CLASSES" \
  --out-pred-csv "$PRED_CSV" \
  --out-summary-json "$SUMMARY_JSON"

"$PY_BIN" scripts/eval_offline.py \
  --pred-csv "$PRED_CSV" \
  --out-json "$SUMMARY_BOOTSTRAP_JSON" \
  --n-bootstrap "$BOOTSTRAP_N" >/dev/null

echo "[E5-BATCH 5/5] Done."
echo " - run_dir          : $RUN_OUT_DIR"
echo " - progress_csv     : $RAW_PROGRESS_CSV"
echo " - predictions_csv  : $PRED_CSV"
echo " - summary_json     : $SUMMARY_JSON"
echo " - bootstrap_json   : $SUMMARY_BOOTSTRAP_JSON"
echo " - ok_count         : $OK_COUNT"
echo " - fail_count       : $FAIL_COUNT"
