# Phase-3 FP32 Multitask Training

This stage continues after phase-2 baselines and trains an FP32 multitask model:
- emotion(common-6)
- arousal2
- arousal3 (masked when unavailable)

## 1) Main run (actor-independent 5-fold)

```bash
bash scripts/run_phase3_fp32_main.sh
```

Outputs:
- `derived/results/fp32_multitask_main/summary.json`
- `derived/results/fp32_multitask_main/predictions.csv`
- `derived/results/fp32_multitask_main/progress.json`
- `derived/results/fp32_multitask_main/checkpoints/best_fold_*.pt`

## 2) Cross-dataset runs (E3)

```bash
# CREMA-D train -> RAVDESS test
./.venv/bin/python scripts/train_fp32_multitask.py \
  --manifest derived/manifests/manifest_common6_all.jsonl \
  --train-list derived/splits/cross_dataset/train_crema_test_ravdess_common6_av_train.txt \
  --val-list derived/splits/cross_dataset/train_crema_test_ravdess_common6_av_test.txt \
  --out-dir derived/results/fp32_multitask_cross_crema_to_ravdess \
  --cache-dir derived/features/cache_v1 \
  --mode fusion \
  --epochs 10 \
  --batch-size 128 \
  --lr 0.001 \
  --n-bootstrap 200 \
  --progress-every 2000

# RAVDESS train -> CREMA-D test
./.venv/bin/python scripts/train_fp32_multitask.py \
  --manifest derived/manifests/manifest_common6_all.jsonl \
  --train-list derived/splits/cross_dataset/test_crema_train_ravdess_common6_av_train.txt \
  --val-list derived/splits/cross_dataset/test_crema_train_ravdess_common6_av_test.txt \
  --out-dir derived/results/fp32_multitask_cross_ravdess_to_crema \
  --cache-dir derived/features/cache_v1 \
  --mode fusion \
  --epochs 10 \
  --batch-size 128 \
  --lr 0.001 \
  --n-bootstrap 200 \
  --progress-every 2000
```

## 3) Analysis report

```bash
bash scripts/run_phase3_analysis.sh
```

Outputs:
- `derived/reports/phase3_global_metrics.csv`
- `derived/reports/phase3_vs_phase2_bootstrap.csv`
- `derived/reports/phase3_results.md`
- `derived/reports/phase3_emotion_f1.svg`
- `derived/reports/phase3_vs_phase2_delta_f1.svg`

## 4) Quick smoke run

```bash
./.venv/bin/python scripts/train_fp32_multitask.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --out-dir derived/results/fp32_multitask_smoke \
  --mode fusion \
  --num-folds 2 \
  --epochs 3 \
  --max-train-per-fold 600 \
  --max-val-per-fold 200 \
  --n-bootstrap 50 \
  --progress-every 200
```

## 5) Live progress monitoring

```bash
watch -n 1 cat derived/results/fp32_multitask_main/progress.json
```

The trainer also prints epoch-level status directly to terminal.
