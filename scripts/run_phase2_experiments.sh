#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bootstrap_pip() {
  local py="$1"
  # 1) try ensurepip (works on many venv builds)
  "$py" -m ensurepip --upgrade >/dev/null 2>&1 || true
  if "$py" -m pip --version >/dev/null 2>&1; then
    return 0
  fi

  # 2) fallback to get-pip.py
  local get_pip_py="/tmp/get-pip.py"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$get_pip_py"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$get_pip_py" https://bootstrap.pypa.io/get-pip.py
  else
    echo "[ERROR] Neither curl nor wget is available to bootstrap pip."
    return 1
  fi
  "$py" "$get_pip_py" >/dev/null
}

if [[ ! -x .venv/bin/python ]]; then
  echo "[ERROR] .venv is missing. Run: bash scripts/setup_ml_env.sh"
  exit 1
fi

VENV_PY="$ROOT_DIR/.venv/bin/python"

# Ensure pip exists in the exact interpreter used below.
if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
  echo "[INFO] pip is missing in .venv. Bootstrapping pip ..."
  if ! bootstrap_pip "$VENV_PY"; then
    echo "[ERROR] Failed to bootstrap pip inside .venv."
    echo "        Recreate venv with: bash scripts/setup_ml_env.sh"
    exit 1
  fi
fi

# Ensure required packages exist in the exact interpreter used below.
if ! "$VENV_PY" -c "import numpy, pandas, sklearn" >/dev/null 2>&1; then
  echo "[INFO] Installing missing Python packages into .venv ..."
  "$VENV_PY" -m pip install --upgrade pip
  "$VENV_PY" -m pip install numpy pandas scikit-learn
fi

"$VENV_PY" scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --out-dir derived/results/ml_baselines_main \
  --cache-dir derived/features/cache_v1 \
  --modalities audio,video,fusion \
  --num-folds 5 \
  --n-bootstrap 300

"$VENV_PY" scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_common6_all.jsonl \
  --train-list derived/splits/cross_dataset/train_crema_test_ravdess_common6_av_train.txt \
  --val-list derived/splits/cross_dataset/train_crema_test_ravdess_common6_av_test.txt \
  --out-dir derived/results/ml_baselines_cross_crema_to_ravdess \
  --cache-dir derived/features/cache_v1 \
  --modalities audio,video,fusion \
  --n-bootstrap 300

"$VENV_PY" scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_common6_all.jsonl \
  --train-list derived/splits/cross_dataset/test_crema_train_ravdess_common6_av_train.txt \
  --val-list derived/splits/cross_dataset/test_crema_train_ravdess_common6_av_test.txt \
  --out-dir derived/results/ml_baselines_cross_ravdess_to_crema \
  --cache-dir derived/features/cache_v1 \
  --modalities audio,video,fusion \
  --n-bootstrap 300

echo "Phase-2 experiments completed."
