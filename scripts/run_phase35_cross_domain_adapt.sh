#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x .venv/bin/python ]]; then
  echo "[ERROR] .venv missing. Run setup first."
  exit 1
fi

VENV_PY="$ROOT_DIR/.venv/bin/python"

if ! "$VENV_PY" -c "import numpy, sklearn" >/dev/null 2>&1; then
  echo "[INFO] Installing required packages for cross-domain adaptation ..."
  "$VENV_PY" -m pip install --upgrade pip
  "$VENV_PY" -m pip install numpy scikit-learn
fi

echo "[ADAPT STEP 1/3] CORAL cross eval: CREMA->RAVDESS (cache_v3, logreg fusion) ..."
"$VENV_PY" scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --train-list derived/splits/cross_dataset/train_crema_test_ravdess_common6_av_train.txt \
  --val-list derived/splits/cross_dataset/train_crema_test_ravdess_common6_av_test.txt \
  --out-dir derived/results/ml_baselines_phase35_v4_logreg_coral_cross_crema_to_ravdess \
  --cache-dir derived/features/cache_v3 \
  --modalities fusion \
  --classifier logreg \
  --domain-adapt coral \
  --coral-eps 1e-5 \
  --n-bootstrap 200 \
  --progress-every 2000

echo "[ADAPT STEP 2/3] CORAL cross eval: RAVDESS->CREMA (cache_v3, logreg fusion) ..."
"$VENV_PY" scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --train-list derived/splits/cross_dataset/test_crema_train_ravdess_common6_av_train.txt \
  --val-list derived/splits/cross_dataset/test_crema_train_ravdess_common6_av_test.txt \
  --out-dir derived/results/ml_baselines_phase35_v4_logreg_coral_cross_ravdess_to_crema \
  --cache-dir derived/features/cache_v3 \
  --modalities fusion \
  --classifier logreg \
  --domain-adapt coral \
  --coral-eps 1e-5 \
  --n-bootstrap 200 \
  --progress-every 2000

echo "[ADAPT STEP 3/3] Analyze cross adaptation results ..."
"$VENV_PY" scripts/analyze_phase35_cross_domain_adapt.py --out-dir derived/reports

echo "Phase-3.5 cross-domain adaptation pipeline completed."
