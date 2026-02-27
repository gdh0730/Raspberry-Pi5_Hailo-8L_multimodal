# Phase-3.5 Candidate Report

## Reference
- phase2_fusion_main: macro-F1=0.3950, acc=0.4122
- phase3_main: macro-F1=0.3775, acc=0.4195

## Candidates
- `fp32_multitask_phase35_ce_main`: F1=0.3869, acc=0.4257, ΔvsP2=-0.0082, ΔvsP3=+0.0093, loss=ce, ws=False
- `fp32_multitask_phase35_focal_ws_main`: F1=0.3752, acc=0.4170, ΔvsP2=-0.0198, ΔvsP3=-0.0023, loss=focal, ws=True

## Best Candidate
- `fp32_multitask_phase35_ce_main` (macro-F1=0.3869)

## Output
- `derived/reports/phase35_candidate_metrics.csv`
- `derived/reports/phase35_results.md`
