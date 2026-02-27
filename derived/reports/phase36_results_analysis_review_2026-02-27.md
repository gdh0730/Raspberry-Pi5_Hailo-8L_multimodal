# Phase36 결과 분석/리뷰 (2026-02-27)

## 1) 분석 범위
- 입력: `derived/reports/phase36_fp32_test_progress.csv`
- 기준: emotion6 `test` 정확도/매크로 F1
- 대상: 5개 트랙(`id_all`, `id_crema`, `id_ravdess`, `ood_c2r`, `ood_r2c`) x 3모드(`audio`, `video`, `fusion`)

## 2) 핵심 결과 요약
- ID 트랙 최상위:
  - `id_all`: fusion F1 `0.6261` (audio `0.6244`와 근접)
  - `id_crema`: audio F1 `0.6663` (fusion `0.6643`와 근접)
  - `id_ravdess`: fusion F1 `0.5811`
- OOD 트랙 최상위:
  - `ood_c2r`: video F1 `0.2658`
  - `ood_r2c`: fusion F1 `0.1278`
- 결론: 현재 파이프라인에서 `ID는 0.58~0.67`, `OOD는 0.09~0.27` 구간으로 성능 분리가 매우 큼.

## 3) 모드별 패턴 리뷰
1. ID에서는 audio/fusion이 반복적으로 우세
- `id_all`: fusion≈audio >> video
- `id_crema`: audio≈fusion >> video
- `id_ravdess`: fusion > audio > video

2. OOD에서는 트랙별로 우세 모드가 바뀜
- `ood_c2r`: video가 가장 강함
- `ood_r2c`: fusion이 근소 우세하지만 절대 성능 낮음

3. 해석
- 모달리티별 일반화 특성이 데이터셋 이동 방향(`CREMA->RAVDESS`, `RAVDESS->CREMA`)에 따라 달라짐.
- 현재 fusion은 ID에서 강점이 뚜렷하지만, OOD에서는 도메인 불일치를 충분히 상쇄하지 못함.

## 4) 연구 진행 관점의 리뷰 결론
1. 이번 Phase36의 목적(고정 train/val/test + ID/OOD 3모드 매트릭스)은 달성됨.
2. 다음 병목은 모델 구조 자체보다 `도메인 이동 대응`(OOD 강건성)임이 수치로 확인됨.
3. Hailo 단계는 이제 "best-per-track 모델"부터 연결하는 방식이 리스크/비용 대비 효율적임.

## 5) 의사결정(이번 리뷰 기준)
1. Hailo 비교는 15개 전체 동시 진입 대신, `트랙별 best 5개`를 먼저 실행한다.
2. 비교 일관성을 위해 FP32 test와 동일 index를 그대로 Pi 추론에 사용한다.
3. compile calibration은 각 run의 train split에서 생성(누수 방지), eval은 해당 run test index로 고정한다.
4. 현재 Hailo 파이프라인(`compile_custom_onnx_sdk.py`, `pi_infer_hailort.py`)은 2입력 AV 모델 기준이므로, 단일모달(audio/video) best가 선택된 트랙은 Hailo 단계에서 `fusion 대체`를 적용한다.

## 6) 즉시 다음 단계
1. best-per-track 5개 자동 선택
2. run별 ONNX export
3. run별 calibration(train 기반, checkpoint 정규화)
4. run별 HEF compile(local x86)
5. 생성된 HEF 목록으로 Pi batch inference 및 FP32-Hailo gap 집계

## 7) 근거 파일
- `derived/reports/phase36_fp32_test_progress.csv`
- `derived/reports/phase36_fp32_test_leaderboard_2026-02-26.md`
- `derived/reports/research_design_process_status_until_2026-02-26.md`
