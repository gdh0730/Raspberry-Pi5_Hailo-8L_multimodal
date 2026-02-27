# Phase-3.5 Strong v1 Report

## Main Candidates
- `v5_logreg_main`: F1=0.5734, acc=0.5729, ΔvsP2=+0.1784, ΔvsP3=+0.1959
- `v5_linsvm_main`: F1=0.5346, acc=0.5338, ΔvsP2=+0.1396, ΔvsP3=+0.1571
- `v3_rbfsvm_main_ref`: F1=0.4807, acc=0.4872, ΔvsP2=+0.0857, ΔvsP3=+0.1032

## Cross Candidates
- `v5_linsvm_coral_cross_crema_to_ravdess` [cross_crema_to_ravdess|coral]: F1=0.2693, acc=0.2756, ΔvsP2=+0.0405, ΔvsV5None=+0.0610
- `v5_logreg_coral_cross_crema_to_ravdess` [cross_crema_to_ravdess|coral]: F1=0.3025, acc=0.3097, ΔvsP2=+0.0737, ΔvsV5None=+0.1357
- `v4_logreg_coral_ref_cross_crema_to_ravdess` [cross_crema_to_ravdess|coral_ref]: F1=0.2604, acc=0.2642, ΔvsP2=+0.0316, ΔvsV5None=+0.0937
- `v5_linsvm_cross_crema_to_ravdess` [cross_crema_to_ravdess|none]: F1=0.2084, acc=0.2765, ΔvsP2=-0.0204, ΔvsV5None=+0.0000
- `v5_logreg_cross_crema_to_ravdess` [cross_crema_to_ravdess|none]: F1=0.1667, acc=0.2330, ΔvsP2=-0.0621, ΔvsV5None=+0.0000
- `v5_linsvm_coral_cross_ravdess_to_crema` [cross_ravdess_to_crema|coral]: F1=0.2438, acc=0.2517, ΔvsP2=+0.1724, ΔvsV5None=+0.1164
- `v5_logreg_coral_cross_ravdess_to_crema` [cross_ravdess_to_crema|coral]: F1=0.2724, acc=0.2781, ΔvsP2=+0.2010, ΔvsV5None=+0.1285
- `v4_logreg_coral_ref_cross_ravdess_to_crema` [cross_ravdess_to_crema|coral_ref]: F1=0.2422, acc=0.2501, ΔvsP2=+0.1708, ΔvsV5None=+0.0983
- `v5_linsvm_cross_ravdess_to_crema` [cross_ravdess_to_crema|none]: F1=0.1274, acc=0.1881, ΔvsP2=+0.0560, ΔvsV5None=+0.0000
- `v5_logreg_cross_ravdess_to_crema` [cross_ravdess_to_crema|none]: F1=0.1439, acc=0.2160, ΔvsP2=+0.0725, ΔvsV5None=+0.0000

## Best Main
- `v5_logreg_main` F1=0.5734

## Best Cross `cross_crema_to_ravdess`
- `v5_logreg_coral_cross_crema_to_ravdess` (coral) F1=0.3025

## Best Cross `cross_ravdess_to_crema`
- `v5_logreg_coral_cross_ravdess_to_crema` (coral) F1=0.2724

## Outputs
- `derived/reports/phase35_strong_v1_main_metrics.csv`
- `derived/reports/phase35_strong_v1_cross_metrics.csv`
- `derived/reports/phase35_strong_v1_results.md`
