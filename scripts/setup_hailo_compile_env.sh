#!/usr/bin/env bash
set -euo pipefail

# Setup local x86 Linux/WSL environment for Hailo HEF compilation.
# This script cannot bypass Hailo license/download requirements.
# It expects user-provided DFC/SDK wheels from Hailo Developer Zone package.
#
# Example:
#   bash scripts/setup_hailo_compile_env.sh \
#     --wheel-dir third_party/hailo_wheels \
#     --model-zoo-ref v2.17
#
# Or with extracted suite directory:
#   bash scripts/setup_hailo_compile_env.sh \
#     --suite-dir /mnt/c/Users/<you>/Downloads/hailo_ai_sw_suite_x.y.z

VENV_DIR=".venv-hailo"
PYTHON_BIN="python3"
WHEEL_DIR=""
SUITE_DIR=""
MODEL_ZOO_DIR="third_party/hailo_model_zoo"
MODEL_ZOO_REF="v2.17"
SKIP_MODEL_ZOO=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VENV_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --wheel-dir) WHEEL_DIR="$2"; shift 2 ;;
    --suite-dir) SUITE_DIR="$2"; shift 2 ;;
    --model-zoo-dir) MODEL_ZOO_DIR="$2"; shift 2 ;;
    --model-zoo-ref) MODEL_ZOO_REF="$2"; shift 2 ;;
    --skip-model-zoo) SKIP_MODEL_ZOO=1; shift ;;
    *)
      echo "[ERROR] Unknown arg: $1"
      exit 1
      ;;
  esac
done

ARCH="$(uname -m)"
if [[ "$ARCH" != "x86_64" ]]; then
  echo "[ERROR] Local compile host must be x86_64. Current arch: $ARCH"
  exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] Python not found: $PYTHON_BIN"
  exit 3
fi

if [[ -n "$SUITE_DIR" && ! -d "$SUITE_DIR" ]]; then
  echo "[ERROR] --suite-dir not found: $SUITE_DIR"
  exit 4
fi
if [[ -n "$WHEEL_DIR" && ! -d "$WHEEL_DIR" ]]; then
  echo "[ERROR] --wheel-dir not found: $WHEEL_DIR"
  exit 5
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[SETUP 1/5] Create venv: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

PY="$VENV_DIR/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "[ERROR] Broken venv: missing $PY"
  exit 6
fi

if ! "$PY" -m pip --version >/dev/null 2>&1; then
  echo "[WARN] Broken pip/venv detected at $VENV_DIR. Recreating venv."
  rm -rf "$VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  PY="$VENV_DIR/bin/python"
fi

echo "[SETUP 2/5] Bootstrap pip/setuptools/wheel"
"$PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$PY" -m pip install --upgrade pip setuptools wheel

detect_wheel_dir() {
  local base="$1"
  find "$base" -maxdepth 5 -type f \
    \( -iname "hailo*.whl" -o -iname "*dataflow*.whl" -o -iname "*sdk*.whl" \) \
    -print -quit
}

if [[ -z "$WHEEL_DIR" && -n "$SUITE_DIR" ]]; then
  hit="$(detect_wheel_dir "$SUITE_DIR" || true)"
  if [[ -n "$hit" ]]; then
    WHEEL_DIR="$(dirname "$hit")"
  fi
fi

if [[ -z "$WHEEL_DIR" ]]; then
  for cand in "$HOME/Downloads" "/mnt/c/Users/$(whoami)/Downloads"; do
    if [[ -d "$cand" ]]; then
      hit="$(detect_wheel_dir "$cand" || true)"
      if [[ -n "$hit" ]]; then
        WHEEL_DIR="$(dirname "$hit")"
        break
      fi
    fi
  done
fi

if [[ -x "$VENV_DIR/bin/hailomz" ]]; then
  echo "[SETUP 3/5] hailomz already available in venv PATH context."
else
  if [[ -z "$WHEEL_DIR" ]]; then
    cat <<EOF
[ERROR] Could not find Hailo DFC/SDK wheels.
Provide one of:
  --wheel-dir <directory containing *.whl>
  --suite-dir <extracted Hailo AI SW Suite directory>

Expected wheel examples (version may vary):
  hailo_sdk_client-*.whl
  hailo_dataflow_compiler-*.whl
EOF
    exit 7
  fi

  mapfile -t WHEELS < <(
    find "$WHEEL_DIR" -maxdepth 4 -type f \
      ! -path "*/.patched/*" \
      \( -iname "hailo*.whl" -o -iname "*dataflow*.whl" -o -iname "*sdk*.whl" \) \
      | sort
  )
  if [[ "${#WHEELS[@]}" -eq 0 ]]; then
    echo "[ERROR] No Hailo wheels found under: $WHEEL_DIR"
    exit 8
  fi

  echo "[SETUP 3/5] Resolve compatible Hailo wheels from: $WHEEL_DIR"
  mapfile -t COMPAT_WHEELS < <(
    "$PY" - <<'PY' "$WHEEL_DIR" "${WHEELS[@]}"
import sys
from pathlib import Path
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

wheel_dir = Path(sys.argv[1])
candidates = [Path(p) for p in sys.argv[2:]]
supported = set(sys_tags())
compatible = []

for w in candidates:
    name = w.name
    try:
        _, _, _, tags = parse_wheel_filename(name)
    except Exception:
        continue
    if tags & supported:
        compatible.append(str(w))

for p in sorted(set(compatible)):
    print(p)
PY
  )
  if [[ "${#COMPAT_WHEELS[@]}" -eq 0 ]]; then
    echo "[ERROR] No compatible Hailo wheels found for current Python/platform in: $WHEEL_DIR"
    exit 8
  fi

  PATCHED_DIR="$WHEEL_DIR/.patched"
  mkdir -p "$PATCHED_DIR"
  FINAL_WHEELS=()
  for w in "${COMPAT_WHEELS[@]}"; do
    bn="$(basename "$w")"
    if [[ "$bn" == hailo_dataflow_compiler-*.whl ]]; then
      patched="$PATCHED_DIR/$bn"
      echo "[SETUP 3/5] Patch DFC wheel to skip pygraphviz hard dependency: $bn"
      "$PY" scripts/patch_dfc_wheel_skip_pygraphviz.py --in-wheel "$w" --out-wheel "$patched"
      FINAL_WHEELS+=("$patched")
    else
      FINAL_WHEELS+=("$w")
    fi
  done

  echo "[SETUP 3/5] Install compatible wheels (${#COMPAT_WHEELS[@]}):"
  printf ' - %s\n' "${FINAL_WHEELS[@]}"
  "$PY" -m pip install "${FINAL_WHEELS[@]}"
fi

# Patch pygraphviz import guard for wheel-only setups without system graphviz dev headers.
echo "[SETUP 3.5/5] Patch hailo_sdk_client pygraphviz guard"
"$PY" scripts/patch_hailo_sdk_pygraphviz_guard.py --venv-dir "$VENV_DIR"

if [[ "$SKIP_MODEL_ZOO" -eq 0 ]]; then
  echo "[SETUP 4/5] Install Hailo Model Zoo CLI (hailomz)"
  mkdir -p "$(dirname "$MODEL_ZOO_DIR")"
  if [[ ! -d "$MODEL_ZOO_DIR/.git" ]]; then
    git clone https://github.com/hailo-ai/hailo_model_zoo.git "$MODEL_ZOO_DIR"
  fi
  (
    cd "$MODEL_ZOO_DIR"
    git fetch --tags --prune
    git checkout "$MODEL_ZOO_REF"
  )
  PIP_USE_PEP517=0 "$PY" -m pip install --no-build-isolation -e "$MODEL_ZOO_DIR"
fi

echo "[SETUP 5/5] Verify tools"
if ! "$PY" - <<'PY' >/dev/null 2>&1
from hailo_sdk_client import ClientRunner  # noqa: F401
PY
then
  echo "[ERROR] hailo_sdk_client import failed after install."
  exit 9
fi
if [[ -x "$VENV_DIR/bin/hailomz" ]]; then
  if ! HAILO_RUNTIME_LIB_DIR="${HAILO_RUNTIME_LIB_DIR:-third_party/hailo_runtime_libs}" \
    LD_LIBRARY_PATH="$PWD/${HAILO_RUNTIME_LIB_DIR:-third_party/hailo_runtime_libs}:${LD_LIBRARY_PATH:-}" \
    "$VENV_DIR/bin/hailomz" --help >/dev/null 2>&1; then
    echo "[WARN] hailomz exists but runtime libs are not fully wired."
  fi
else
  echo "[INFO] hailomz not installed (SKIP_MODEL_ZOO or optional). SDK compile path is still available."
fi

echo "[OK] Local compile environment is ready."
echo "Activate: source $VENV_DIR/bin/activate"
echo "Then compile:"
echo "  bash scripts/run_e4_compile_local.sh --network-name fp32_v8_fold0 --venv $VENV_DIR"
