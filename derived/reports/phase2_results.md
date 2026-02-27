# Phase-2 Result Report

## Best Modalities
- Main(5-fold): `fusion` (macro-F1=0.3950)
- Cross CREMA->RAVDESS: `audio` (macro-F1=0.2386)
- Cross RAVDESS->CREMA: `audio` (macro-F1=0.1297)

## Key Files
- `derived/reports/phase2_global_metrics.csv`
- `derived/reports/phase2_pairwise_bootstrap.csv`
- `derived/reports/phase2_main_f1.svg`
- `derived/reports/phase2_cross_f1.svg`

## Interpretation
- In-domain(main), `fusion` is the strongest and improves over single modalities.
- Cross-dataset scores drop substantially, indicating domain gap.
- The asymmetry between CREMA->RAVDESS and RAVDESS->CREMA is notable.

## Next Step (Design Alignment)
- Proceed to FP32 deep baseline training (E1/E2 refined).
- Then run quantization path (E4) and on-device benchmark (E5).

