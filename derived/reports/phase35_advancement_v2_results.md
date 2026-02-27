# Phase-3.5 Advancement v2 Result Report

## References
- phase2_fusion_main: F1=0.3950, acc=0.4122
- phase3_fp32_main: F1=0.3775, acc=0.4195

## Main Candidates
- `ml_v3_rbfsvm_fusion`: F1=0.4807, acc=0.4872, ΔvsP2=+0.0857, ΔvsP3=+0.1032
- `fp32_v3_ce_fusion`: F1=0.4721, acc=0.4782, ΔvsP2=+0.0770, ΔvsP3=+0.0945
- `ml_v3_logreg_fusion`: F1=0.4149, acc=0.4158, ΔvsP2=+0.0199, ΔvsP3=+0.0374

## Cross Candidates
- `ml_v3_logreg_cross_crema_to_ravdess`(cross_crema_to_ravdess): F1=0.1190, phase2=0.2288, delta=-0.1098
- `ml_v3_logreg_cross_ravdess_to_crema`(cross_ravdess_to_crema): F1=0.1066, phase2=0.0714, delta=+0.0352
- `ml_v3_rbfsvm_cross_crema_to_ravdess`(cross_crema_to_ravdess): F1=0.0513, phase2=0.2288, delta=-0.1775
- `ml_v3_rbfsvm_cross_ravdess_to_crema`(cross_ravdess_to_crema): F1=0.0486, phase2=0.0714, delta=-0.0228

## Best Main Candidate
- `ml_v3_rbfsvm_fusion` (F1=0.4807)

## Outputs
- `derived/reports/phase35_advancement_v2_main_metrics.csv`
- `derived/reports/phase35_advancement_v2_cross_metrics.csv`
- `derived/reports/phase35_advancement_v2_results.md`
