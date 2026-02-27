# Phase-3.5 Cross-Domain Adaptation Report

## Results
- `v4_logreg_coral_cross_crema_to_ravdess` [cross_crema_to_ravdess|coral]: F1=0.2604, acc=0.2642, ΔvsP2=+0.0316, ΔvsV3=+0.0937
- `v5_logreg_coral_cross_crema_to_ravdess` [cross_crema_to_ravdess|coral]: F1=0.3025, acc=0.3097, ΔvsP2=+0.0737, ΔvsV3=+0.1357
- `v8_hubert_logreg_coral_cross_crema_to_ravdess` [cross_crema_to_ravdess|coral]: F1=0.3207, acc=0.3229, ΔvsP2=+0.0919, ΔvsV3=+0.1539
- `v3_logreg_baseline_cross_crema_to_ravdess` [cross_crema_to_ravdess|none]: F1=0.1190, acc=0.2017, ΔvsP2=-0.1098, ΔvsV3=-0.0477
- `v5_logreg_baseline_cross_crema_to_ravdess` [cross_crema_to_ravdess|none]: F1=0.1667, acc=0.2330, ΔvsP2=-0.0621, ΔvsV3=+0.0000
- `v4_logreg_coral_cross_ravdess_to_crema` [cross_ravdess_to_crema|coral]: F1=0.2422, acc=0.2501, ΔvsP2=+0.1708, ΔvsV3=+0.0983
- `v5_logreg_coral_cross_ravdess_to_crema` [cross_ravdess_to_crema|coral]: F1=0.2724, acc=0.2781, ΔvsP2=+0.2010, ΔvsV3=+0.1285
- `v8_hubert_logreg_coral_cross_ravdess_to_crema` [cross_ravdess_to_crema|coral]: F1=0.3187, acc=0.3225, ΔvsP2=+0.2473, ΔvsV3=+0.1748
- `v3_logreg_baseline_cross_ravdess_to_crema` [cross_ravdess_to_crema|none]: F1=0.1066, acc=0.1910, ΔvsP2=+0.0352, ΔvsV3=-0.0373
- `v5_logreg_baseline_cross_ravdess_to_crema` [cross_ravdess_to_crema|none]: F1=0.1439, acc=0.2160, ΔvsP2=+0.0725, ΔvsV3=+0.0000

## Best for `cross_crema_to_ravdess`
- `v8_hubert_logreg_coral_cross_crema_to_ravdess` (coral) F1=0.3207

## Best for `cross_ravdess_to_crema`
- `v8_hubert_logreg_coral_cross_ravdess_to_crema` (coral) F1=0.3187

## Outputs
- `derived/reports/phase35_cross_domain_adapt_metrics.csv`
- `derived/reports/phase35_cross_domain_adapt_results.md`
