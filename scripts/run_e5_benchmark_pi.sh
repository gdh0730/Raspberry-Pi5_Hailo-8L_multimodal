#!/usr/bin/env bash
set -euo pipefail

# Run E5 benchmark on Raspberry Pi + Hailo.
#
# Examples:
#   bash scripts/run_e5_benchmark_pi.sh \
#     --host wormhole@129.254.232.91 \
#     --hef /usr/share/hailo-models/resnet_v1_50_h8l.hef \
#     --name resnet50_h8l_ref
#
#   bash scripts/run_e5_benchmark_pi.sh \
#     --host wormhole@129.254.232.91 \
#     --hef-local derived/hailo/build/fp32_v8_fold0/fp32_v8_fold0.hef \
#     --name fp32_v8_fold0

HOST=""
HEF_REMOTE=""
HEF_LOCAL=""
NAME=""
TIME_TO_RUN=10
REMOTE_WORKDIR="~/work/raspi_hailo_stage"
OUT_DIR="derived/hailo/pi_bench"
USE_PASSWORD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --hef) HEF_REMOTE="$2"; shift 2 ;;
    --hef-local) HEF_LOCAL="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --time) TIME_TO_RUN="$2"; shift 2 ;;
    --remote-workdir) REMOTE_WORKDIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
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
if [[ -z "$NAME" ]]; then
  if [[ -n "$HEF_REMOTE" ]]; then
    NAME="$(basename "$HEF_REMOTE" .hef)"
  else
    NAME="$(basename "$HEF_LOCAL" .hef)"
  fi
fi

mkdir -p "$OUT_DIR"

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
    trap 'rm -f "$ASKPASS_SCRIPT"' EXIT
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
  echo "[E5 1/4] Upload HEF to Pi ..."
  "${SSH_BASE[@]}" "$HOST" "mkdir -p ${REMOTE_WORKDIR}/hefs ${REMOTE_WORKDIR}/bench"
  "${SCP_BASE[@]}" "$HEF_LOCAL" "$HOST:${REMOTE_HEF}"
else
  echo "[E5 1/4] Use existing remote HEF: ${REMOTE_HEF}"
  "${SSH_BASE[@]}" "$HOST" "mkdir -p ${REMOTE_WORKDIR}/bench"
fi

REMOTE_CSV="${REMOTE_WORKDIR}/bench/${NAME}_bench.csv"
REMOTE_PARSE="${REMOTE_WORKDIR}/bench/${NAME}_parse.txt"
REMOTE_BENCH_LOG="${REMOTE_WORKDIR}/bench/${NAME}_bench.log"
REMOTE_CMD=$(cat <<EOF
set -euo pipefail
if ! command -v hailo >/dev/null 2>&1; then
  echo "[REMOTE-ERROR] hailo cli not found."
  exit 11
fi
hailo parse-hef ${REMOTE_HEF} | tee ${REMOTE_PARSE}
set +e
hailo benchmark ${REMOTE_HEF} -t ${TIME_TO_RUN} --csv ${REMOTE_CSV} 2>&1 | tee ${REMOTE_BENCH_LOG}
BENCH_RC=\$?
set -e
if [[ "\$BENCH_RC" -ne 0 ]]; then
  if grep -q "HW Latency measurement is supported on networks with a single input" ${REMOTE_BENCH_LOG}; then
    CLEAN_LOG="\$(mktemp)"
    tr '\\r' '\\n' < ${REMOTE_BENCH_LOG} | sed -E 's/\\x1B\\[[0-9;]*[A-Za-z]//g' > "\$CLEAN_LOG"
    mapfile -t FINAL_LINES < <(grep "100% |" "\$CLEAN_LOG" | grep "FPS:" || true)
    rm -f "\$CLEAN_LOG"
    HW_ONLY_FPS=""
    HW_ONLY_FRAMES=""
    STREAMING_FPS=""
    STREAMING_FRAMES=""
    if [[ "\${#FINAL_LINES[@]}" -ge 1 ]]; then
      L1="\${FINAL_LINES[0]}"
      HW_ONLY_FRAMES="\$(echo "\$L1" | cut -d'|' -f2 | tr -d '[:space:]')"
      HW_ONLY_FPS="\$(echo "\$L1" | cut -d'|' -f3 | sed -E 's/.*FPS:[[:space:]]*([0-9.]+).*/\\1/')"
    fi
    if [[ "\${#FINAL_LINES[@]}" -ge 2 ]]; then
      L2="\${FINAL_LINES[1]}"
      STREAMING_FRAMES="\$(echo "\$L2" | cut -d'|' -f2 | tr -d '[:space:]')"
      STREAMING_FPS="\$(echo "\$L2" | cut -d'|' -f3 | sed -E 's/.*FPS:[[:space:]]*([0-9.]+).*/\\1/')"
    fi
    cat > ${REMOTE_CSV} <<CSV
net_name,fps,hw_only_fps,num_of_frames,num_of_frames_hw_only,hw_latency,overall_latency,min_power,average_power,max_power
${NAME},\${STREAMING_FPS},\${HW_ONLY_FPS},\${STREAMING_FRAMES},\${HW_ONLY_FRAMES},,,,,
CSV
    echo "[REMOTE-WARN] Latency step unsupported for multi-input network. Wrote fallback CSV from FPS logs."
  else
    echo "[REMOTE-ERROR] benchmark failed (rc=\$BENCH_RC)."
    exit "\$BENCH_RC"
  fi
fi
EOF
)

echo "[E5 2/4] Run parse-hef + benchmark on Pi ..."
"${SSH_BASE[@]}" "$HOST" "$REMOTE_CMD"

echo "[E5 3/4] Download benchmark artifacts ..."
"${SCP_BASE[@]}" "$HOST:${REMOTE_CSV}" "${OUT_DIR}/${NAME}_bench.csv"
"${SCP_BASE[@]}" "$HOST:${REMOTE_PARSE}" "${OUT_DIR}/${NAME}_parse.txt"
"${SCP_BASE[@]}" "$HOST:${REMOTE_BENCH_LOG}" "${OUT_DIR}/${NAME}_bench.log"

echo "[E5 4/4] Done."
echo " - CSV  : ${OUT_DIR}/${NAME}_bench.csv"
echo " - Parse: ${OUT_DIR}/${NAME}_parse.txt"
echo " - Log  : ${OUT_DIR}/${NAME}_bench.log"
