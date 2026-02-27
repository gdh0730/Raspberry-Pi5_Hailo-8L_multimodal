#!/usr/bin/env bash
set -euo pipefail

# End-to-end local flow:
# 1) Wait/find Hailo SW Suite file or dir
# 2) Prepare wheels + setup local compile env
# 3) Compile HEF locally
# 4) (Optional) Run Pi benchmark with compiled HEF
#
# Example:
#   bash scripts/run_e4_end_to_end_local.sh \
#     --wait-seconds 3600 \
#     --network-name fp32_v8_fold0 \
#     --pi-host wormhole@129.254.232.91

SUITE_DIR=""
SUITE_ARCHIVE=""
SEARCH_ROOT="/mnt/c/Users/$(whoami)/Downloads"
WAIT_SECONDS=0
POLL_SECONDS=20
NETWORK_NAME=""
VENV_DIR=".venv-hailo"
PI_HOST=""
PI_TIME=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite-dir) SUITE_DIR="$2"; shift 2 ;;
    --suite-archive) SUITE_ARCHIVE="$2"; shift 2 ;;
    --search-root) SEARCH_ROOT="$2"; shift 2 ;;
    --wait-seconds) WAIT_SECONDS="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --network-name) NETWORK_NAME="$2"; shift 2 ;;
    --venv) VENV_DIR="$2"; shift 2 ;;
    --pi-host) PI_HOST="$2"; shift 2 ;;
    --pi-time) PI_TIME="$2"; shift 2 ;;
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
if [[ -n "$SUITE_DIR" && -n "$SUITE_ARCHIVE" ]]; then
  echo "[ERROR] Use either --suite-dir or --suite-archive."
  exit 3
fi

find_suite_archive() {
  local root="$1"
  find "$root" -maxdepth 5 -type f \
    \( -iname "*hailo*sw*suite*.tar.gz" -o -iname "*hailo*sw*suite*.zip" -o -iname "*software*suite*.tar.gz" -o -iname "*software*suite*.zip" \) \
    | sed -n '1,1p'
}

find_suite_dir() {
  local root="$1"
  find "$root" -maxdepth 5 -type d -iname "*hailo*sw*suite*" | sed -n '1,1p'
}

if [[ -z "$SUITE_DIR" && -z "$SUITE_ARCHIVE" && "$WAIT_SECONDS" -gt 0 ]]; then
  echo "[WAIT] Looking for Hailo SW Suite in: $SEARCH_ROOT"
  echo "[WAIT] Timeout=${WAIT_SECONDS}s, poll=${POLL_SECONDS}s"
  START_TS="$(date +%s)"
  while true; do
    if [[ -d "$SEARCH_ROOT" ]]; then
      SUITE_DIR="$(find_suite_dir "$SEARCH_ROOT" || true)"
      if [[ -z "$SUITE_DIR" ]]; then
        SUITE_ARCHIVE="$(find_suite_archive "$SEARCH_ROOT" || true)"
      fi
    fi
    if [[ -n "$SUITE_DIR" || -n "$SUITE_ARCHIVE" ]]; then
      break
    fi
    NOW_TS="$(date +%s)"
    ELAPSED="$((NOW_TS - START_TS))"
    if [[ "$ELAPSED" -ge "$WAIT_SECONDS" ]]; then
      echo "[ERROR] Timed out waiting for SW Suite under: $SEARCH_ROOT"
      exit 4
    fi
    sleep "$POLL_SECONDS"
  done
fi

SETUP_ARGS=()
if [[ -n "$SUITE_DIR" ]]; then
  SETUP_ARGS+=(--suite-dir "$SUITE_DIR")
fi
if [[ -n "$SUITE_ARCHIVE" ]]; then
  SETUP_ARGS+=(--suite-archive "$SUITE_ARCHIVE")
fi

if [[ "${#SETUP_ARGS[@]}" -eq 0 ]]; then
  echo "[ERROR] Missing SW Suite input."
  echo "Use one of:"
  echo "  --suite-dir <extracted_suite_dir>"
  echo "  --suite-archive <suite_archive>"
  echo "Or use --wait-seconds with --search-root."
  exit 5
fi

echo "[RUN 1/3] Setup local compile env ..."
bash scripts/run_e4_local_setup.sh "${SETUP_ARGS[@]}" --venv "$VENV_DIR"

echo "[RUN 2/3] Compile HEF locally ..."
bash scripts/run_e4_compile_local.sh \
  --network-name "$NETWORK_NAME" \
  --venv "$VENV_DIR"

if [[ -n "$PI_HOST" ]]; then
  echo "[RUN 3/3] Benchmark compiled HEF on Pi ..."
  bash scripts/run_e5_benchmark_pi.sh \
    --host "$PI_HOST" \
    --hef-local "derived/hailo/build/${NETWORK_NAME}/${NETWORK_NAME}.hef" \
    --name "$NETWORK_NAME" \
    --time "$PI_TIME"
else
  echo "[RUN 3/3] Skipped Pi benchmark (no --pi-host)."
fi

echo "[OK] End-to-end local E4 flow done."
