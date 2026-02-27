#!/usr/bin/env bash
set -euo pipefail

# Lightweight remote readiness check for Raspberry Pi + Hailo.
#
# Examples:
#   bash scripts/run_e5_pi_check.sh --host pi@192.168.0.50
#   PI_PASSWORD='***' bash scripts/run_e5_pi_check.sh --host pi@192.168.0.50 --use-password

HOST=""
USE_PASSWORD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --use-password) USE_PASSWORD=1; shift ;;
    *) echo "[ERROR] Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$HOST" ]]; then
  echo "[ERROR] --host is required (example: pi@192.168.0.50)"
  exit 1
fi

REMOTE_CMD='
set -e
echo "[REMOTE] host=$(hostname)"
echo "[REMOTE] uname=$(uname -a)"
if command -v hailortcli >/dev/null 2>&1; then
  echo "[REMOTE] hailortcli=OK"
  hailortcli fw-control identify || true
else
  echo "[REMOTE] hailortcli=MISSING"
fi
if command -v rpicam-hello >/dev/null 2>&1; then
  echo "[REMOTE] rpicam-hello=OK"
else
  echo "[REMOTE] rpicam-hello=MISSING"
fi
if command -v python3 >/dev/null 2>&1; then
  python3 - << "PY"
import sys
print("[REMOTE] python", sys.version.replace("\n"," "))
PY
fi
'

if [[ "$USE_PASSWORD" -eq 1 ]]; then
  if [[ -z "${PI_PASSWORD:-}" ]]; then
    echo "[ERROR] --use-password requested but PI_PASSWORD env var is empty."
    exit 2
  fi
  if command -v sshpass >/dev/null 2>&1; then
    sshpass -p "$PI_PASSWORD" ssh -o StrictHostKeyChecking=accept-new "$HOST" "$REMOTE_CMD"
  else
    ASKPASS_SCRIPT="$(mktemp)"
    trap 'rm -f "$ASKPASS_SCRIPT"' EXIT
    cat >"$ASKPASS_SCRIPT" <<EOF
#!/usr/bin/env sh
echo '$PI_PASSWORD'
EOF
    chmod 700 "$ASKPASS_SCRIPT"
    env SSH_ASKPASS="$ASKPASS_SCRIPT" SSH_ASKPASS_REQUIRE=force DISPLAY=:0 \
      setsid ssh -o StrictHostKeyChecking=accept-new "$HOST" "$REMOTE_CMD"
  fi
else
  ssh -o StrictHostKeyChecking=accept-new "$HOST" "$REMOTE_CMD"
fi
