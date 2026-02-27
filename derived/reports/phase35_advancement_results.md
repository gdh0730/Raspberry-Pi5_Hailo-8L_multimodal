# Phase-3.5 Advancement Result Report

## References
- phase2_fusion_main: F1=0.3950, acc=0.4122
- phase3_fp32_main: F1=0.3775, acc=0.4195

## Candidates
- `ml_v2_logreg_fusion`: F1=0.4205, acc=0.4408, ΔvsP2=+0.0255, ΔvsP3=+0.0430
- `ml_v2_rf_fusion`: F1=0.4066, acc=0.4287, ΔvsP2=+0.0116, ΔvsP3=+0.0291
- `fp32_v2_ce_fusion`: F1=0.3980, acc=0.4328, ΔvsP2=+0.0030, ΔvsP3=+0.0205
- `ml_v2_rf_audio`: F1=0.3918, acc=0.4074, ΔvsP2=-0.0032, ΔvsP3=+0.0143
- `ml_v2_logreg_audio`: F1=0.3858, acc=0.4123, ΔvsP2=-0.0093, ΔvsP3=+0.0082
- `ml_v2_logreg_video`: F1=0.2552, acc=0.2727, ΔvsP2=-0.1398, ΔvsP3=-0.1224
- `ml_v2_rf_video`: F1=0.2336, acc=0.2350, ΔvsP2=-0.1615, ΔvsP3=-0.1440

## Best Candidate
- `ml_v2_logreg_fusion` (F1=0.4205)

## Outputs
- `derived/reports/phase35_advancement_metrics.csv`
- `derived/reports/phase35_advancement_results.md`
