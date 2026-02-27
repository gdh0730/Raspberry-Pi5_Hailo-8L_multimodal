# Research Project Full Process and Results Report (as of 2026-02-24)

## 1. Document purpose
This report consolidates the entire executed process from initial data setup to the latest Phase-3.5 results.
It is intended to be a single source for:
- what was actually executed,
- what changed at each phase,
- what performance was achieved,
- what remains unfinished against the research designs,
- and what must be done next for the new `0.9` target track.

## 2. Scope and dataset status

### 2.1 Project scope actually executed
- Data preparation and split generation
- B0 majority baseline
- Phase-2 classical ML baselines
- Phase-3 FP32 multitask model
- Phase-3.5 advancement iterations (v2/v3/v4/v5/v6/v7/v8)
- Cross-domain adaptation experiments (CORAL)
- GPU migration to CUDA and re-runs
- Progress/analysis report automation

### 2.2 Data integrity summary
Source: `derived/manifests/summary.json`
- Total samples: 11,762
- CREMA-D: 7,442
- RAVDESS: 4,320
- Common-6 total: 10,610
- Multimodal AV (Common-6): 8,498
- RAVDESS AO (Common-6): 1,056

## 3. Chronological execution log (what was done)

### 3.1 Data phase
- Downloaded and extracted CREMA-D and RAVDESS.
- Resolved unzip wildcard mismatch issue by validating actual ZIP inventory.
- Built manifests/splits with `scripts/prepare_research_data.py`.
- Created groupkfold and cross-dataset split files under `derived/splits/`.

### 3.2 B0 baseline
- Ran `scripts/train_b0_majority.py` for AV and AO references.
- B0 macro-F1 remained near random-level reference (~0.05).

### 3.3 Environment stabilization (WSL/.venv)
- Fixed missing `numpy` and missing `pip` in `.venv`.
- Added bootstrap/install logic to run scripts.
- Standardized reproducible execution entry points.

### 3.4 Phase-2 ML baselines
- Ran main 5-fold and cross-domain baselines via `scripts/train_ml_baselines.py`.
- Added progress visibility (`progress.json` + staged logs).
- Generated phase-level analysis artifacts and plots.

### 3.5 Phase-3 FP32 multitask
- Introduced `scripts/train_fp32_multitask.py`.
- Fixed masked-label handling for arousal labels to prevent sample collapse.
- Completed main + cross runs and generated analysis.

### 3.6 Phase-3.5 advancement chain
- v2 (`cache_v2`): improved main vs old baseline.
- v3 (`cache_v3` + pretrained video embedding): improved main, but cross collapse observed.
- v4/v5: introduced CORAL and stronger cross-domain recovery.
- v6: label smoothing + weighted sampler in FP32 path.
- v7: gated fusion, wider head, stronger long-run baselines.
- GPU migration: fixed device resolution bug and moved pipelines to CUDA (`RTX 4090`).
- v8 HuBERT track: best performance jump to date.

### 3.7 Ensemble and cross re-measurement
- Created multiple v8 tunes and vote ensembles.
- Best ensemble reached macro-F1 `0.7099`.
- Re-measured cross-domain on v8 HuBERT cache with logreg+CORAL, and both directions improved.

## 4. Performance evolution (key numbers)

### 4.1 Main track evolution (emotion macro-F1)
| Stage | Model/Run | Macro-F1 |
|---|---|---:|
| B0 | majority | ~0.05 |
| Phase-2 | fusion baseline | 0.3950 |
| Phase-3 | fp32 multitask main | 0.3775 |
| Phase-3.5 v2 | ml_v2_logreg_fusion | 0.4205 |
| Phase-3.5 v3 | ml_v3_rbfsvm_fusion | 0.4807 |
| Phase-3.5 v5 | ml_v5_logreg_main | 0.5734 |
| Phase-3.5 v6 | fp32_v6_ce_ls_ws_main | 0.5985 |
| Phase-3.5 v7 | fp32_v7_ce_ls_ws_gated_wide_main | 0.6056 |
| Phase-3.5 v8 single best | fp32_v8_hubert_gated_wide_tune4 | 0.6992 |
| Phase-3.5 v8 ensemble best | fp32_v8_hubert_ensemble_vote3_main_t3_t4 | **0.7099** |

Sources:
- `derived/reports/phase2_global_metrics.csv`
- `derived/reports/phase3_global_metrics.csv`
- `derived/reports/phase35_next_v8_metrics.csv`

### 4.2 Cross-domain evolution (emotion macro-F1)
| Direction | Phase-2 fusion | v5 logreg+CORAL | v8 hubert logreg+CORAL |
|---|---:|---:|---:|
| CREMA->RAVDESS | 0.2288 | 0.3025 | **0.3207** |
| RAVDESS->CREMA | 0.0714 | 0.2724 | **0.3187** |

Source: `derived/reports/phase35_cross_domain_adapt_metrics.csv`

### 4.3 Latest top-3 main candidates (2026-02-24)
1. `fp32_v8_hubert_ensemble_vote3_main_t3_t4`: F1 0.7099, acc 0.7088
2. `fp32_v8_hubert_ensemble_vote3`: F1 0.7047, acc 0.7034
3. `fp32_v8_hubert_gated_wide_tune4` (single): F1 0.6992, acc 0.6974

Source: `derived/reports/phase35_next_v8_results.md`

## 5. Engineering fixes and reproducibility improvements

### 5.1 Critical fixes
- Masked arousal label handling in FP32 training path.
- Device selection bug fix (`auto/cpu/cuda`) in training and feature extraction.
- CUDA package migration to cu126 stack.

### 5.2 Runtime/monitoring improvements
- Introduced stage-wise progress outputs and `progress.json` in long jobs.
- Added analysis scripts for phase-wise metric consolidation.
- Added reusable vote-ensemble builder script (`scripts/build_vote_ensemble.py`).

## 6. Design-vs-implementation gap (important)

### 6.1 What is implemented
- log-mel/delta statistical audio features
- pretrained audio/video embedding append path
- classical baseline family (LR/LinearSVM/RBF-SVM/RF)
- gated fusion in FP32 model
- CE/Focal/weighted sampler/label smoothing options
- CORAL-based cross-domain adaptation for ML baselines

### 6.2 What is still incomplete against design
- P-track:
  - loudness norm, CMVN, SpecAugment full implementation
  - face ROI crop + face quality flags (blur/brightness/success ratio)
- A-track:
  - XGBoost/LightGBM comparison
  - cross-attention-lite fusion block
- T-track:
  - pretrained backbone fine-tuning (unfreeze top layers) not yet done
  - current path is mostly frozen embedding extraction + shallow learner
- Original deployment track:
  - PTQ/QAT and Hailo HEF compilation/benchmarks are still pending

Reference:
- `deep-research-report-advancement (3).md`
- `deep-research-report (3).md`

## 7. New objective split: 0.7 track vs 0.9 track

### 7.1 0.7 track status
- Achieved on ensemble basis (`0.7099`), single-model near-hit (`0.6992`).

### 7.2 0.9 track status
- Not achieved; current gap is large.
- Single-model gap: +0.2008
- Ensemble gap to 0.92 target: +0.2101

### 7.3 Implication
The current architecture family can push to high-0.6/low-0.7, but `0.9` requires a new stage:
- stronger preprocessing fidelity,
- fine-tuned pretrained encoders,
- more expressive multimodal fusion,
- and stronger domain-robust training.

## 8. Risks and constraints
- Domain shift between CREMA-D and RAVDESS remains substantial.
- Cross metrics improved, but still far from high-confidence production-grade levels.
- Overfitting risk rises as model complexity increases without additional data.
- Reaching `0.9` under current protocol may be unrealistic without major redesign or protocol scope adjustment.

## 9. Artifacts generated in this session
- Design update: `deep-research-report-advancement (3).md`
- Full report: `derived/reports/full_research_process_and_results_until_2026-02-24.md`
- Existing rolling progress log: `derived/reports/project_progress_until_2026-02-20.md`
- Main latest metrics: `derived/reports/phase35_next_v8_metrics.csv`
- Cross latest metrics: `derived/reports/phase35_cross_domain_adapt_metrics.csv`

## 10. Next execution recommendation (before new experiments)
1. Lock this report as baseline snapshot (`2026-02-24`).
2. Start `0.9` track with explicit experiment namespace (`phase35_v9_*`).
3. Implement missing P/A/T blocks in the order: ROI+quality -> cross-attention-lite -> fine-tuning.
4. Keep `0.7` track frozen for regression checks.

