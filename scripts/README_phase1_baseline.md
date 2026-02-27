# Phase-1 Baseline (No Extra Python Packages)

This phase gives you reproducible baseline numbers immediately in the current environment.

## 1) Rebuild manifests/splits

```bash
python3 scripts/prepare_research_data.py --repo-root .
```

## 2) Run B0 majority baseline (main AV track)

```bash
python3 scripts/train_b0_majority.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --out-dir derived/results/b0_majority_av
```

Outputs:
- `derived/results/b0_majority_av/predictions.csv`
- `derived/results/b0_majority_av/summary.json`

## 3) Optional AO baseline (RAVDESS audio-only)

```bash
python3 scripts/train_b0_majority.py \
  --manifest derived/manifests/manifest_ravdess_audio_only_common6.jsonl \
  --fold-dir derived/splits/groupkfold5_ravdess \
  --out-dir derived/results/b0_majority_ravdess_ao
```

## 4) Re-evaluate any prediction CSV

```bash
python3 scripts/eval_offline.py \
  --pred-csv derived/results/b0_majority_av/predictions.csv \
  --out-json derived/results/b0_majority_av/eval_offline_summary.json
```

## Current limitations

- This phase is intentionally stdlib-only and does not include deep models.
- It provides B0 (majority) as the lower bound and validates your metric/report pipeline.

## Next required step (B1/B2/B3/B4)

Install a full ML stack first, then add model training:
- `torch`, `torchaudio`, `numpy`, `pandas`, `scikit-learn`
- feature pipeline (log-mel, frame sampling)
- model training scripts for audio-only/video-only/fusion
