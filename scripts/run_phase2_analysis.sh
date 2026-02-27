#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x .venv/bin/python ]]; then
  echo "[ERROR] .venv is missing"
  exit 1
fi

VENV_PY="$ROOT_DIR/.venv/bin/python"

if ! "$VENV_PY" -c "import numpy, pandas, sklearn" >/dev/null 2>&1; then
  echo "[ERROR] Required packages missing in .venv. Run setup first."
  exit 1
fi

"$VENV_PY" scripts/analyze_phase2_results.py --out-dir derived/reports --n-bootstrap 500

echo "Phase-2 analysis completed."
