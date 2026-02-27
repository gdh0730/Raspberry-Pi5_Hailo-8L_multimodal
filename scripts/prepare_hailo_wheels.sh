#!/usr/bin/env bash
set -euo pipefail

# Prepare Hailo wheel directory from an extracted suite directory or archive.
#
# Example:
#   bash scripts/prepare_hailo_wheels.sh \
#     --suite-dir /mnt/c/Users/<you>/Downloads/hailo_ai_sw_suite_x.y.z \
#     --out-dir third_party/hailo_wheels
#
#   bash scripts/prepare_hailo_wheels.sh \
#     --suite-archive /mnt/c/Users/<you>/Downloads/hailo_ai_sw_suite_x.y.z.tar.gz \
#     --out-dir third_party/hailo_wheels

SUITE_DIR=""
SUITE_ARCHIVE=""
SEARCH_ROOT="/mnt/c/Users/$(whoami)/Downloads"
OUT_DIR="third_party/hailo_wheels"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite-dir) SUITE_DIR="$2"; shift 2 ;;
    --suite-archive) SUITE_ARCHIVE="$2"; shift 2 ;;
    --search-root) SEARCH_ROOT="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    *)
      echo "[ERROR] Unknown arg: $1"
      exit 1
      ;;
  esac
done

if [[ -n "$SUITE_DIR" && -n "$SUITE_ARCHIVE" ]]; then
  echo "[ERROR] Use either --suite-dir or --suite-archive, not both."
  exit 2
fi

WORK_DIR=""
cleanup() {
  if [[ -n "$WORK_DIR" && -d "$WORK_DIR" ]]; then
    rm -rf "$WORK_DIR"
  fi
}
trap cleanup EXIT

detect_suite_dir() {
  local root="$1"
  find "$root" -maxdepth 5 -type d -iname "*hailo*sw*suite*" | sed -n '1,1p'
}

if [[ -z "$SUITE_DIR" && -z "$SUITE_ARCHIVE" ]]; then
  if [[ -d "$SEARCH_ROOT" ]]; then
    SUITE_DIR="$(detect_suite_dir "$SEARCH_ROOT" || true)"
    if [[ -z "$SUITE_DIR" ]]; then
      SUITE_ARCHIVE="$(find "$SEARCH_ROOT" -maxdepth 5 -type f \( -iname "*hailo*sw*suite*.tar.gz" -o -iname "*hailo*sw*suite*.zip" \) | sed -n '1,1p')"
    fi
  fi
fi

if [[ -n "$SUITE_ARCHIVE" ]]; then
  if [[ ! -f "$SUITE_ARCHIVE" ]]; then
    echo "[ERROR] --suite-archive not found: $SUITE_ARCHIVE"
    exit 3
  fi
  WORK_DIR="$(mktemp -d /tmp/hailo_suite_extract.XXXXXX)"
  echo "[PREP 1/3] Extract archive: $SUITE_ARCHIVE"
  case "$SUITE_ARCHIVE" in
    *.tar.gz|*.tgz) tar xzf "$SUITE_ARCHIVE" -C "$WORK_DIR" ;;
    *.zip) unzip -q "$SUITE_ARCHIVE" -d "$WORK_DIR" ;;
    *)
      echo "[ERROR] Unsupported archive format: $SUITE_ARCHIVE"
      exit 4
      ;;
  esac
  SUITE_DIR="$(find "$WORK_DIR" -maxdepth 3 -type d -iname "*hailo*sw*suite*" | sed -n '1,1p')"
  if [[ -z "$SUITE_DIR" ]]; then
    SUITE_DIR="$WORK_DIR"
  fi
fi

if [[ -z "$SUITE_DIR" || ! -d "$SUITE_DIR" ]]; then
  cat <<EOF
[ERROR] Hailo Software Suite directory not found.
Provide one of:
  --suite-dir <extracted suite directory>
  --suite-archive <suite .tar.gz or .zip>

Or place it under:
  $SEARCH_ROOT
EOF
  exit 5
fi

echo "[PREP 2/3] Collect wheels from: $SUITE_DIR"
mkdir -p "$OUT_DIR"
mapfile -t WHEELS < <(
  find "$SUITE_DIR" -maxdepth 8 -type f \
    \( -iname "hailo*.whl" -o -iname "*dataflow*.whl" -o -iname "*sdk*.whl" \) \
    | sort
)

if [[ "${#WHEELS[@]}" -eq 0 ]]; then
  echo "[ERROR] No Hailo-related wheels found in: $SUITE_DIR"
  exit 6
fi

for w in "${WHEELS[@]}"; do
  cp -f "$w" "$OUT_DIR/"
done

echo "[PREP 3/3] Done"
echo " - wheel count: $(find "$OUT_DIR" -maxdepth 1 -type f -name '*.whl' | wc -l)"
echo " - wheel dir  : $OUT_DIR"
