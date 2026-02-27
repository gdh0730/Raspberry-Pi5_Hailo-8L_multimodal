# Phase-3.5 Next v8 Report

## Main Candidates
- `fp32_v8_hubert_ensemble_vote3_main_t3_t4`: F1=0.7099, acc=0.7088, device=ensemble, ΔvsV7=+0.1042, gap_to_0.7=-0.0099
- `fp32_v8_hubert_ensemble_vote3`: F1=0.7047, acc=0.7034, device=ensemble, ΔvsV7=+0.0990, gap_to_0.7=-0.0047
- `fp32_v8_hubert_gated_wide_tune4`: F1=0.6992, acc=0.6974, device=cuda, ΔvsV7=+0.0935, gap_to_0.7=0.0008
- `fp32_v8_hubert_gated_wide_tune2`: F1=0.6961, acc=0.6946, device=cuda, ΔvsV7=+0.0904, gap_to_0.7=0.0039
- `fp32_v8_hubert_gated_wide_tune1`: F1=0.6950, acc=0.6938, device=cuda, ΔvsV7=+0.0894, gap_to_0.7=0.0050
- `fp32_v8_hubert_gated_wide_tune3`: F1=0.6940, acc=0.6933, device=cuda, ΔvsV7=+0.0884, gap_to_0.7=0.0060
- `fp32_v8_hubert_gated_wide_main`: F1=0.6913, acc=0.6905, device=cuda, ΔvsV7=+0.0857, gap_to_0.7=0.0087
- `fp32_v7_best_ref`: F1=0.6056, acc=0.6061, device=cpu, ΔvsV7=+0.0000, gap_to_0.7=0.0944
- `fp32_v7_best_cuda_ref`: F1=0.5992, acc=0.5997, device=cuda, ΔvsV7=-0.0064, gap_to_0.7=0.1008

## Outputs
- `derived/reports/phase35_next_v8_metrics.csv`
- `derived/reports/phase35_next_v8_results.md`
