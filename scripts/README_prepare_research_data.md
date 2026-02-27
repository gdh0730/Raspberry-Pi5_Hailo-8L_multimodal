# Research Data Preparation

## Run

```bash
python3 scripts/prepare_research_data.py --repo-root .
```

## Generated Files

- `derived/manifests/manifest_all.jsonl`
- `derived/manifests/manifest_crema_d.jsonl`
- `derived/manifests/manifest_ravdess.jsonl`
- `derived/manifests/manifest_common6_all.jsonl`
- `derived/manifests/manifest_multimodal_common6_av.jsonl`
- `derived/manifests/manifest_audio_enabled_common6.jsonl`
- `derived/manifests/manifest_video_enabled_common6.jsonl`
- `derived/manifests/manifest_ravdess_audio_only_common6.jsonl`
- `derived/manifests/manifest_ravdess_av_common6.jsonl`
- `derived/manifests/summary.json`
- `derived/splits/groupkfold5_*.csv`
- `derived/splits/groupkfold5_*/fold_{0..4}_{train,val}.txt`
- `derived/splits/cross_dataset/*.txt`

## Key Rules Used

- Common-6 labels: `neutral, happy, sad, angry, fearful, disgust`
- RAVDESS:
  - `modality_code=01` -> `modality=av`, `has_audio=true`
  - `modality_code=02` -> `modality=video_only`, `has_audio=false`
  - `modality_code=03` -> `modality=audio_only`, `has_audio=true`
- CREMA-D:
  - Intensity `LO/MD/HI` -> `arousal3=0/1/2`
  - Arousal-2: `HI=1`, `LO/MD=0`
  - Intensity `XX` -> arousal labels are null

## Notes

- `manifest_multimodal_common6_av.jsonl` is the safest default manifest for AV training.
- The script auto-loads both:
  - `datasets/ravdess/raw_video_speech` (`.mp4`)
  - `datasets/ravdess/raw_audio_speech` (`.wav`)
- `manifest_multimodal_common6_av.jsonl` stays AV-only by design.
