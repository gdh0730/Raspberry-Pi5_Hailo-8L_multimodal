# Phase-2 ML Baselines (B1/B2/B3)

## Environment

```bash
source .venv/bin/activate
```

If `.venv` is missing:

```bash
bash scripts/setup_ml_env.sh
```

## Main experiment (E1): actor-independent 5-fold

```bash
python scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --out-dir derived/results/ml_baselines_main \
  --cache-dir derived/features/cache_v1 \
  --modalities audio,video,fusion \
  --num-folds 5 \
  --n-bootstrap 300
```

Outputs:
- `derived/results/ml_baselines_main/summary.json`
- `derived/results/ml_baselines_main/predictions.csv`
- `derived/results/ml_baselines_main/progress.json`

## Cross-dataset experiment (E3): CREMA -> RAVDESS(AV common6)

```bash
python scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_common6_all.jsonl \
  --train-list derived/splits/cross_dataset/train_crema_test_ravdess_common6_av_train.txt \
  --val-list derived/splits/cross_dataset/train_crema_test_ravdess_common6_av_test.txt \
  --out-dir derived/results/ml_baselines_cross_crema_to_ravdess \
  --cache-dir derived/features/cache_v1 \
  --modalities audio,video,fusion \
  --n-bootstrap 300
```

## Cross-dataset experiment (E3): RAVDESS(AV common6) -> CREMA

```bash
python scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_common6_all.jsonl \
  --train-list derived/splits/cross_dataset/test_crema_train_ravdess_common6_av_train.txt \
  --val-list derived/splits/cross_dataset/test_crema_train_ravdess_common6_av_test.txt \
  --out-dir derived/results/ml_baselines_cross_ravdess_to_crema \
  --cache-dir derived/features/cache_v1 \
  --modalities audio,video,fusion \
  --n-bootstrap 300
```

## Quick smoke mode (fast check)

```bash
python scripts/train_ml_baselines.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --out-dir derived/results/ml_baselines_smoke \
  --cache-dir derived/features/cache_v1 \
  --modalities audio,video,fusion \
  --max-train-per-fold 120 \
  --max-val-per-fold 60 \
  --n-bootstrap 50
```

## Notes

- First run is slower because ffmpeg feature cache is built.
- Cache directory is reusable across runs: `derived/features/cache_v1`.
- This is a classical-ML baseline step before deep FP32 models.
- During run, progress is visualized in terminal (fold/mode/progress bar).
- Live status file is updated continuously:
  - `derived/results/<out_dir>/progress.json`
- To poll progress from another terminal:

```bash
watch -n 1 cat derived/results/ml_baselines_main/progress.json
```

## Runtime benchmark (E5 draft, CPU)

```bash
python scripts/bench_phase2_runtime.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --mode fusion \
  --fold 0 \
  --max-train 1000 \
  --max-val 300 \
  --out-json derived/results/phase2_runtime_bench_fusion_fold0.json
```
