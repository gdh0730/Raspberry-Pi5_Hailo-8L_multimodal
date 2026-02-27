# Phase-3.5 Advancement (Before E4)

This stage improves preprocessing/modeling before quantization (E4).

## GPU usage

- `train_fp32_multitask.py` and `prepare_advanced_features.py` now support `--device auto` (default).
- On a CUDA host, `auto` resolves to `cuda`; otherwise it falls back to `cpu`.
- Quick check:

```bash
./.venv/bin/python - << 'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## Why this stage

- `phase3_main` underperforms `phase2_fusion_main` on macro-F1.
- Need stronger imbalance handling + comparison framework before PTQ/QAT.

## What is added now

- `train_fp32_multitask.py` new options:
  - `--emotion-loss {ce,focal}`
  - `--focal-gamma`
  - `--weighted-sampler`
- Runner:
  - `scripts/run_phase35_experiments.sh`
  - `scripts/run_phase35_advancement_pipeline.sh`
- Analyzer:
  - `scripts/analyze_phase35_results.py`
  - `scripts/analyze_phase35_advancement.py`

## Run

```bash
bash scripts/run_phase35_experiments.sh
```

Outputs:
- `derived/results/fp32_multitask_phase35_ce_main/summary.json`
- `derived/results/fp32_multitask_phase35_focal_ws_main/summary.json`
- `derived/reports/phase35_candidate_metrics.csv`
- `derived/reports/phase35_results.md`

## Advanced pipeline (cache_v2 + algorithm comparison)

```bash
bash scripts/run_phase35_advancement_pipeline.sh
```

Outputs:
- `derived/features/cache_v2/summary.json`
- `derived/results/ml_baselines_phase35_v2_logreg_main/summary.json`
- `derived/results/ml_baselines_phase35_v2_rf_main/summary.json`
- `derived/results/fp32_multitask_phase35_v2_ce_main/summary.json`
- `derived/reports/phase35_advancement_metrics.csv`
- `derived/reports/phase35_advancement_results.md`

## Advanced v2 pipeline (raw cache_v3 + pretrained video embedding)

```bash
bash scripts/run_phase35_advancement_v2_pipeline.sh
```

Key differences:
- Forces raw extraction path (`--no-prefer-source-cache --fallback-raw`)
- Adds pretrained video embedding (`resnet18`) into video features
- Compares `logreg` vs `rbf_svm` on fusion mode
- Includes cross-dataset runs for both classifiers

Outputs:
- `derived/features/cache_v3/summary.json`
- `derived/results/ml_baselines_phase35_v3_logreg_main/summary.json`
- `derived/results/ml_baselines_phase35_v3_rbfsvm_main/summary.json`
- `derived/results/fp32_multitask_phase35_v3_ce_main/summary.json`
- `derived/reports/phase35_advancement_v2_main_metrics.csv`
- `derived/reports/phase35_advancement_v2_cross_metrics.csv`
- `derived/reports/phase35_advancement_v2_results.md`

## Cross-domain adaptation follow-up (CORAL)

```bash
bash scripts/run_phase35_cross_domain_adapt.sh
```

Purpose:
- Keep the best v3 feature cache (`cache_v3`) and target cross-domain gap directly.
- Compare `logreg` baseline (`domain_adapt=none`) vs `CORAL` (`domain_adapt=coral`) on:
  - `CREMA -> RAVDESS`
  - `RAVDESS -> CREMA`

Outputs:
- `derived/results/ml_baselines_phase35_v4_logreg_coral_cross_crema_to_ravdess/summary.json`
- `derived/results/ml_baselines_phase35_v4_logreg_coral_cross_ravdess_to_crema/summary.json`
- `derived/reports/phase35_cross_domain_adapt_metrics.csv`
- `derived/reports/phase35_cross_domain_adapt_results.md`

## Strong v1 pipeline (audio pretrained + cache_v4)

```bash
bash scripts/run_phase35_strong_v1.sh
```

Purpose:
- Add pretrained audio embedding (`wav2vec2_base`) to build `cache_v4` audio.
- Reuse `cache_v3` video features for fast fusion experiments.
- Run main/cross with `logreg`, `linear_svm` and cross-domain `CORAL`.

Outputs:
- `derived/features/cache_v4/summary.json`
- `derived/results/ml_baselines_phase35_v5_logreg_main/summary.json`
- `derived/results/ml_baselines_phase35_v5_linsvm_main/summary.json`
- `derived/results/ml_baselines_phase35_v5_*_cross_*/summary.json`
- `derived/reports/phase35_strong_v1_main_metrics.csv`
- `derived/reports/phase35_strong_v1_cross_metrics.csv`
- `derived/reports/phase35_strong_v1_results.md`

## Next v6 (FP32 label-smoothing direction)

```bash
bash scripts/run_phase35_next_v6.sh
```

Purpose:
- Execute the next 0.7-direction upgrade on FP32 training:
  - CE + label smoothing + weighted sampler
  - Focal + weighted sampler
- Compare against `v5_logreg_main`.

Outputs:
- `derived/results/fp32_multitask_phase35_v6_ce_ls_ws_main/summary.json`
- `derived/results/fp32_multitask_phase35_v6_focal_ws_main/summary.json`
- `derived/reports/phase35_next_v6_metrics.csv`
- `derived/reports/phase35_next_v6_results.md`

## Next v7 (Gated fusion + long-run RBF-SVM)

```bash
bash scripts/run_phase35_next_v7.sh
```

Purpose:
- Add a stronger fusion architecture in FP32 (`--fusion-type gated`, modality dropout, wider hidden).
- Run long `rbf_svm` baseline on `cache_v4` without interruption.
- Compare all v7 candidates against v6 best and report gap to `0.7`.

Outputs:
- `derived/results/fp32_multitask_phase35_v7_ce_ls_ws_gated_main/summary.json`
- `derived/results/fp32_multitask_phase35_v7_focal_ws_gated_main/summary.json`
- `derived/results/fp32_multitask_phase35_v7_ce_ls_ws_gated_wide_main/summary.json`
- `derived/results/ml_baselines_phase35_v7_rbfsvm_main/summary.json`
- `derived/reports/phase35_next_v7_metrics.csv`
- `derived/reports/phase35_next_v7_results.md`

## Next v8 (HuBERT audio + gated wide)

```bash
bash scripts/run_phase35_next_v8_hubert_main.sh
```

Purpose:
- Build `cache_v5_hubert` with pretrained `hubert_base` audio embeddings on GPU (`--device auto`).
- Reuse `cache_v3` video features and run the best FP32 gated-wide setting.
- Compare v8 vs v7 baseline in a dedicated report.

Outputs:
- `derived/features/cache_v5_hubert/summary.json`
- `derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_main/summary.json`
- `derived/reports/phase35_next_v8_metrics.csv`
- `derived/reports/phase35_next_v8_results.md`

Additional tuning (single-model boost + ensemble):

```bash
.venv/bin/python scripts/train_fp32_multitask.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --cache-dir derived/features/cache_v5_hubert \
  --mode fusion --fusion-type gated --device auto \
  --weighted-sampler --emotion-loss ce \
  --out-dir derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune1

.venv/bin/python scripts/train_fp32_multitask.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --cache-dir derived/features/cache_v5_hubert \
  --mode fusion --fusion-type gated --device auto \
  --weighted-sampler --emotion-loss ce \
  --out-dir derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune2

.venv/bin/python scripts/train_fp32_multitask.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --cache-dir derived/features/cache_v5_hubert \
  --mode fusion --fusion-type gated --device auto \
  --weighted-sampler --emotion-loss ce \
  --out-dir derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune3

.venv/bin/python scripts/train_fp32_multitask.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --cache-dir derived/features/cache_v5_hubert \
  --mode fusion --fusion-type gated --device auto \
  --weighted-sampler --emotion-loss ce \
  --seed 1337 \
  --out-dir derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune4

.venv/bin/python scripts/build_vote_ensemble.py \
  --name fp32_v8_hubert_ensemble_vote3_main_t3_t4 \
  --out-dir derived/results/fp32_multitask_phase35_v8_hubert_ensemble_vote3_main_t3_t4 \
  --pred-csv \
  derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_main/predictions.csv \
  derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune3/predictions.csv \
  derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune4/predictions.csv

.venv/bin/python scripts/analyze_phase35_next_v8.py --out-dir derived/reports
```

Current best snapshot:
- single model: `fp32_v8_hubert_gated_wide_tune4` macro-F1 `0.6992`
- ensemble: `fp32_v8_hubert_ensemble_vote3_main_t3_t4` macro-F1 `0.7099`

## Interpret

- Primary metric: emotion macro-F1
- Promotion gate to E4:
  - best candidate macro-F1 >= phase2 fusion main macro-F1
  - arousal2 MAE not significantly degraded
