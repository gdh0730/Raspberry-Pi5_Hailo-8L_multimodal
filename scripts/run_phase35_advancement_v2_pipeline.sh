#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x .venv/bin/python ]]; then
  echo "[ERROR] .venv missing. Run setup first."
  exit 1
fi

VENV_PY="$ROOT_DIR/.venv/bin/python"

if ! "$VENV_PY" -c "import torch, numpy, sklearn, torchvision" >/dev/null 2>&1; then
  echo "[INFO] Installing required packages for phase35-v2 ..."
  "$VENV_PY" -m pip install --upgrade pip
  "$VENV_PY" -m pip install numpy pandas scikit-learn
  if command -v nvidia-smi >/dev/null 2>&1; then
    "$VENV_PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  else
    "$VENV_PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  fi
fi

echo "[V2 STEP 1/7] Build raw cache_v3 (+ pretrained video embedding) ..."
"$VENV_PY" scripts/prepare_advanced_features.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --cache-dir derived/features/cache_v3 \
  --device auto \
  --no-prefer-source-cache \
  --fallback-raw \
  --video-pretrained-backbone resnet18 \
  --video-pretrained-frames 4 \
  --video-pretrained-size 224 \
  --kind both \
  --overwrite \
  --progress-every 200

echo "[V2 STEP 2/7] ML baseline logreg (main, fusion only) ..."
"$VENV_PY" scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --out-dir derived/results/ml_baselines_phase35_v3_logreg_main \
  --cache-dir derived/features/cache_v3 \
  --modalities fusion \
  --classifier logreg \
  --num-folds 5 \
  --n-bootstrap 200 \
  --progress-every 2000

echo "[V2 STEP 3/7] ML baseline rbf_svm (main, fusion only) ..."
"$VENV_PY" scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --out-dir derived/results/ml_baselines_phase35_v3_rbfsvm_main \
  --cache-dir derived/features/cache_v3 \
  --modalities fusion \
  --classifier rbf_svm \
  --num-folds 5 \
  --n-bootstrap 200 \
  --progress-every 2000

echo "[V2 STEP 4/7] FP32 candidate (main) ..."
"$VENV_PY" scripts/train_fp32_multitask.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --out-dir derived/results/fp32_multitask_phase35_v3_ce_main \
  --cache-dir derived/features/cache_v3 \
  --mode fusion \
  --device auto \
  --num-folds 5 \
  --epochs 12 \
  --batch-size 128 \
  --lr 0.001 \
  --emotion-loss ce \
  --n-bootstrap 200 \
  --progress-every 2000

echo "[V2 STEP 5/7] Cross eval (logreg fusion) ..."
"$VENV_PY" scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --train-list derived/splits/cross_dataset/train_crema_test_ravdess_common6_av_train.txt \
  --val-list derived/splits/cross_dataset/train_crema_test_ravdess_common6_av_test.txt \
  --out-dir derived/results/ml_baselines_phase35_v3_logreg_cross_crema_to_ravdess \
  --cache-dir derived/features/cache_v3 \
  --modalities fusion \
  --classifier logreg \
  --n-bootstrap 200 \
  --progress-every 2000

"$VENV_PY" scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --train-list derived/splits/cross_dataset/test_crema_train_ravdess_common6_av_train.txt \
  --val-list derived/splits/cross_dataset/test_crema_train_ravdess_common6_av_test.txt \
  --out-dir derived/results/ml_baselines_phase35_v3_logreg_cross_ravdess_to_crema \
  --cache-dir derived/features/cache_v3 \
  --modalities fusion \
  --classifier logreg \
  --n-bootstrap 200 \
  --progress-every 2000

echo "[V2 STEP 6/7] Cross eval (rbf_svm fusion) ..."
"$VENV_PY" scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --train-list derived/splits/cross_dataset/train_crema_test_ravdess_common6_av_train.txt \
  --val-list derived/splits/cross_dataset/train_crema_test_ravdess_common6_av_test.txt \
  --out-dir derived/results/ml_baselines_phase35_v3_rbfsvm_cross_crema_to_ravdess \
  --cache-dir derived/features/cache_v3 \
  --modalities fusion \
  --classifier rbf_svm \
  --n-bootstrap 200 \
  --progress-every 2000

"$VENV_PY" scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --train-list derived/splits/cross_dataset/test_crema_train_ravdess_common6_av_train.txt \
  --val-list derived/splits/cross_dataset/test_crema_train_ravdess_common6_av_test.txt \
  --out-dir derived/results/ml_baselines_phase35_v3_rbfsvm_cross_ravdess_to_crema \
  --cache-dir derived/features/cache_v3 \
  --modalities fusion \
  --classifier rbf_svm \
  --n-bootstrap 200 \
  --progress-every 2000

echo "[V2 STEP 7/7] Analyze phase35-v2 results ..."
"$VENV_PY" scripts/analyze_phase35_advancement_v2.py --out-dir derived/reports

echo "Phase-3.5 advancement v2 pipeline completed."
