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
  echo "[INFO] Installing required Python packages ..."
  "$VENV_PY" -m pip install --upgrade pip
  "$VENV_PY" -m pip install numpy pandas scikit-learn
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  if ! "$VENV_PY" -c "import torch; import torchvision; import torchaudio; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    echo "[INFO] Installing CUDA-enabled torch/torchvision/torchaudio (cu126) ..."
    "$VENV_PY" -m pip install --upgrade pip
    "$VENV_PY" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  fi
else
  if ! "$VENV_PY" -c "import torch, torchvision, torchaudio" >/dev/null 2>&1; then
    echo "[INFO] Installing CPU torch/torchvision/torchaudio ..."
    "$VENV_PY" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  fi
fi

echo "[STRONG STEP 1/7] Build cache_v4 audio with pretrained wav2vec2 ..."
mkdir -p derived/features/cache_v4/audio derived/features/cache_v4/video

"$VENV_PY" scripts/prepare_advanced_features.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --cache-dir derived/features/cache_v4 \
  --source-cache-dir derived/features/cache_v3 \
  --device auto \
  --no-prefer-source-cache \
  --fallback-raw \
  --audio-pretrained-backbone wav2vec2_base \
  --audio-pretrained-max-samples 32000 \
  --kind audio \
  --overwrite \
  --progress-every 200

echo "[STRONG STEP 2/7] Reuse cache_v3 video features into cache_v4 ..."
if [[ -d derived/features/cache_v3/video ]]; then
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete derived/features/cache_v3/video/ derived/features/cache_v4/video/
  else
    rm -rf derived/features/cache_v4/video
    mkdir -p derived/features/cache_v4/video
    cp -a derived/features/cache_v3/video/. derived/features/cache_v4/video/
  fi
else
  echo "[ERROR] cache_v3 video directory missing."
  exit 1
fi

echo "[STRONG STEP 3/7] Main eval (logreg, fusion, cache_v4) ..."
"$VENV_PY" scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --out-dir derived/results/ml_baselines_phase35_v5_logreg_main \
  --cache-dir derived/features/cache_v4 \
  --modalities fusion \
  --classifier logreg \
  --num-folds 5 \
  --n-bootstrap 200 \
  --progress-every 2000

echo "[STRONG STEP 4/7] Main eval (linear_svm, fusion, cache_v4) ..."
"$VENV_PY" scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --out-dir derived/results/ml_baselines_phase35_v5_linsvm_main \
  --cache-dir derived/features/cache_v4 \
  --modalities fusion \
  --classifier linear_svm \
  --num-folds 5 \
  --n-bootstrap 200 \
  --progress-every 2000

echo "[STRONG STEP 5/7] Cross eval (none, logreg+linear_svm) ..."
"$VENV_PY" scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --train-list derived/splits/cross_dataset/train_crema_test_ravdess_common6_av_train.txt \
  --val-list derived/splits/cross_dataset/train_crema_test_ravdess_common6_av_test.txt \
  --out-dir derived/results/ml_baselines_phase35_v5_logreg_cross_crema_to_ravdess \
  --cache-dir derived/features/cache_v4 \
  --modalities fusion \
  --classifier logreg \
  --n-bootstrap 200 \
  --progress-every 2000

"$VENV_PY" scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --train-list derived/splits/cross_dataset/test_crema_train_ravdess_common6_av_train.txt \
  --val-list derived/splits/cross_dataset/test_crema_train_ravdess_common6_av_test.txt \
  --out-dir derived/results/ml_baselines_phase35_v5_logreg_cross_ravdess_to_crema \
  --cache-dir derived/features/cache_v4 \
  --modalities fusion \
  --classifier logreg \
  --n-bootstrap 200 \
  --progress-every 2000

"$VENV_PY" scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --train-list derived/splits/cross_dataset/train_crema_test_ravdess_common6_av_train.txt \
  --val-list derived/splits/cross_dataset/train_crema_test_ravdess_common6_av_test.txt \
  --out-dir derived/results/ml_baselines_phase35_v5_linsvm_cross_crema_to_ravdess \
  --cache-dir derived/features/cache_v4 \
  --modalities fusion \
  --classifier linear_svm \
  --n-bootstrap 200 \
  --progress-every 2000

"$VENV_PY" scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --train-list derived/splits/cross_dataset/test_crema_train_ravdess_common6_av_train.txt \
  --val-list derived/splits/cross_dataset/test_crema_train_ravdess_common6_av_test.txt \
  --out-dir derived/results/ml_baselines_phase35_v5_linsvm_cross_ravdess_to_crema \
  --cache-dir derived/features/cache_v4 \
  --modalities fusion \
  --classifier linear_svm \
  --n-bootstrap 200 \
  --progress-every 2000

echo "[STRONG STEP 6/7] Cross eval (CORAL, logreg+linear_svm) ..."
"$VENV_PY" scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --train-list derived/splits/cross_dataset/train_crema_test_ravdess_common6_av_train.txt \
  --val-list derived/splits/cross_dataset/train_crema_test_ravdess_common6_av_test.txt \
  --out-dir derived/results/ml_baselines_phase35_v5_logreg_coral_cross_crema_to_ravdess \
  --cache-dir derived/features/cache_v4 \
  --modalities fusion \
  --classifier logreg \
  --domain-adapt coral \
  --coral-eps 1e-5 \
  --n-bootstrap 200 \
  --progress-every 2000

"$VENV_PY" scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --train-list derived/splits/cross_dataset/test_crema_train_ravdess_common6_av_train.txt \
  --val-list derived/splits/cross_dataset/test_crema_train_ravdess_common6_av_test.txt \
  --out-dir derived/results/ml_baselines_phase35_v5_logreg_coral_cross_ravdess_to_crema \
  --cache-dir derived/features/cache_v4 \
  --modalities fusion \
  --classifier logreg \
  --domain-adapt coral \
  --coral-eps 1e-5 \
  --n-bootstrap 200 \
  --progress-every 2000

"$VENV_PY" scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --train-list derived/splits/cross_dataset/train_crema_test_ravdess_common6_av_train.txt \
  --val-list derived/splits/cross_dataset/train_crema_test_ravdess_common6_av_test.txt \
  --out-dir derived/results/ml_baselines_phase35_v5_linsvm_coral_cross_crema_to_ravdess \
  --cache-dir derived/features/cache_v4 \
  --modalities fusion \
  --classifier linear_svm \
  --domain-adapt coral \
  --coral-eps 1e-5 \
  --n-bootstrap 200 \
  --progress-every 2000

"$VENV_PY" scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --train-list derived/splits/cross_dataset/test_crema_train_ravdess_common6_av_train.txt \
  --val-list derived/splits/cross_dataset/test_crema_train_ravdess_common6_av_test.txt \
  --out-dir derived/results/ml_baselines_phase35_v5_linsvm_coral_cross_ravdess_to_crema \
  --cache-dir derived/features/cache_v4 \
  --modalities fusion \
  --classifier linear_svm \
  --domain-adapt coral \
  --coral-eps 1e-5 \
  --n-bootstrap 200 \
  --progress-every 2000

echo "[STRONG STEP 7/7] Analyze strong-v1 results ..."
"$VENV_PY" scripts/analyze_phase35_strong_v1.py --out-dir derived/reports

echo "Phase-3.5 strong-v1 pipeline completed."
