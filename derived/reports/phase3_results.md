# Phase-3 FP32 Result Report

## Best FP32 Run
- `main` (macro-F1=0.3775)

## Key Files
- `derived/reports/phase3_global_metrics.csv`
- `derived/reports/phase3_vs_phase2_bootstrap.csv`
- `derived/reports/phase3_emotion_f1.svg`
- `derived/reports/phase3_vs_phase2_delta_f1.svg`

## Phase-2 Fusion 대비 요약
- `main`: Δmacro-F1=-0.0175, phase3=0.3775, phase2_fusion=0.3950
- `cross_crema_to_ravdess`: Δmacro-F1=+0.0577, phase3=0.2865, phase2_fusion=0.2288
- `cross_ravdess_to_crema`: Δmacro-F1=-0.0009, phase3=0.0705, phase2_fusion=0.0714

## Interpretation
- Main(run=main)에서는 phase-2 fusion과 성능 차이를 직접 비교해 기준선 유지/개선 여부를 판단한다.
- Cross run 결과는 도메인 갭(E3)을 정량화하며, 방향별 비대칭을 확인한다.
- 다음 단계(E4)는 phase-3 체크포인트를 기준으로 PTQ/QAT 경량화 비교 실험이다.
