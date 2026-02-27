# 연구 프로젝트 진행 통합 기록 (기준일: 2026-02-27)

## 1. 문서 목적
이 문서는 본 프로젝트의 전체 연구 과정을 처음 단계부터 현재 중단 시점(2026-02-27)까지 단일 타임라인으로 정리한 최신 상태 기록이다.  
핵심 목표는 다음 두 가지다.

- 연구 수행 이력(무엇을 왜 했는지)과 의사결정 근거를 추적 가능하게 남기는 것
- 지금 즉시 재개 가능한 수준으로 현재 상태(완료/보류/미완료)를 명확히 고정하는 것

---

## 2. 연구 목표 및 기준

- 과제: CREMA-D + RAVDESS 기반 영상+음성 감정인식(공통 6감정)
- 평가 축:
  - ID(in-domain): 동일 분포 내 일반화
  - OOD(cross-dataset): 데이터셋 간 전이 일반화
- 모드 축:
  - `audio`, `video`, `fusion`
- 핵심 지표:
  - emotion `macro-F1`(주지표), accuracy

핵심 설계 문서:
- `deep-research-report (3).md`
- `deep-research-report-advancement (3).md`

---

## 3. 단계별 진행 이력

## 3.1 데이터 준비/정합/분할

실행 파이프라인:
- `scripts/prepare_research_data.py`

주요 산출물:
- `derived/manifests/manifest_multimodal_common6_av.jsonl`
- `derived/manifests/summary.json`
- `derived/splits/groupkfold5_*/*`
- `derived/splits/cross_dataset/*`

확정된 데이터 규모(요약):
- all: 11,762
- CREMA-D: 7,442
- RAVDESS: 4,320
- common6_all: 10,610
- multimodal_common6_av: 8,498

핵심 의사결정:
- actor-independent 분할을 기본 원칙으로 유지
- ID와 OOD를 분리 평가
- train/val/test 누수 확인 루틴 유지

---

## 3.2 Phase-1/2: 기준선 및 고전 ML 비교

참고 결과:
- `derived/reports/phase2_global_metrics.csv`
- `derived/reports/phase2_results.md`

핵심 결과(Phase-2):
- main fusion macro-F1: `0.3950`
- cross C->R fusion macro-F1: `0.2288`
- cross R->C fusion macro-F1: `0.0714`

해석:
- main에서는 fusion 이득이 확인됨
- cross에서는 도메인 이동에 따른 급락이 큼

---

## 3.3 Phase-3: FP32 멀티태스크 초기 모델

핵심 코드:
- `scripts/train_fp32_multitask.py`

참고 결과:
- `derived/reports/phase3_global_metrics.csv`
- `derived/reports/phase3_results.md`

핵심 수치:
- main macro-F1: `0.3775`
- C->R macro-F1: `0.2865`
- R->C macro-F1: `0.0705`

해석:
- 초기 FP32 구조만으로는 main에서 Phase-2 fusion 대비 우세가 자동 보장되지 않음
- 이후 성능 향상은 구조 자체보다 전처리/표현/학습 전략 고도화가 핵심임을 확인

---

## 3.4 Phase-3.5: 고도화(전처리 + 사전학습 + 결합구조)

실행/분석 파이프라인:
- `scripts/run_phase35_advancement_pipeline.sh`
- `scripts/run_phase35_advancement_v2_pipeline.sh`
- `scripts/run_phase35_strong_v1.sh`
- `scripts/run_phase35_next_v8_hubert_main.sh`

참고 결과:
- `derived/reports/phase35_strong_v1_results.md`
- `derived/reports/phase35_cross_domain_adapt_results.md`
- `derived/reports/phase35_next_v8_results.md`

핵심 성과:
- 단일모델 최고(main):
  - `fp32_v8_hubert_gated_wide_tune4` -> macro-F1 `0.6992`
- 앙상블 최고(main):
  - `fp32_v8_hubert_ensemble_vote3_main_t3_t4` -> macro-F1 `0.7099`
- cross 최고(CORAL 포함):
  - C->R: `0.3207`
  - R->C: `0.3187`

핵심 의사결정:
- 오디오 표현을 HuBERT 계열로 고도화
- fusion을 gated/wide 계열로 확장
- OOD는 CORAL 기반 도메인 정렬을 명시적으로 적용

---

## 3.5 Hailo E4/E5 준비 및 전환

준비/자동화 구축:
- `hailo/export_onnx.py`
- `hailo/calib_dump_npy_dir.py`
- `hailo/compile_hef.sh`
- `scripts/run_e4_compile_remote.sh`
- `scripts/run_e5_infer_pi.sh`
- `scripts/run_e5_infer_pi_batch.sh`
- `scripts/eval_hailo_pi_infer.py`

환경/준비 상태 기록:
- `derived/reports/hailo_e4_prep_status_2026-02-25.md`

핵심 정리:
- Pi(Hailo-8L)는 추론 런타임 경로 확인 완료
- 컴파일은 x86 DFC 환경에서 수행 후 Pi로 배포하는 구조로 고정
- 입력 정규화/인덱스 일치 비교 정책을 파이프라인에 반영

---

## 3.6 Phase-36: FP32 ID/OOD 5트랙 x 3모드 매트릭스

핵심 산출물:
- `derived/reports/phase36_fp32_test_progress.csv`
- `derived/reports/phase36_fp32_test_leaderboard_2026-02-26.md`
- `derived/reports/phase36_results_analysis_review_2026-02-27.md`

test 기준 핵심 결과:
- `id_all`: audio 0.6244 / video 0.3065 / fusion 0.6261
- `id_crema`: audio 0.6663 / video 0.3244 / fusion 0.6643
- `id_ravdess`: audio 0.5359 / video 0.4072 / fusion 0.5811
- `ood_c2r`: audio 0.0898 / video 0.2658 / fusion 0.1080
- `ood_r2c`: audio 0.1018 / video 0.1242 / fusion 0.1278

해석:
- ID에서는 audio/fusion이 우세
- OOD는 절대 성능이 낮고, 이동 방향별 우세 모드가 바뀜
- 즉, 현재 병목은 도메인 이동 강건성

---

## 3.7 Phase-36 Hailo best5 비교(진행 중단 시점 상태)

비교 리포트:
- `derived/reports/phase36_fp32_vs_hailo_best5.csv`
- `derived/reports/phase36_fp32_vs_hailo_best5.md`

컴파일 준비 상태:
- `derived/hailo/phase36_best5_build_meta.csv` 기준 5개 트랙 모두 `build_status=ok`

현재 비교 상태(중단 시점):
- done: 4
  - `id_all` fusion: FP32 F1 0.6261 -> Hailo F1 0.5892 (Δ -0.0369)
  - `id_crema` fusion: FP32 F1 0.6643 -> Hailo F1 0.6162 (Δ -0.0482)
  - `id_ravdess` fusion: FP32 F1 0.5811 -> Hailo F1 0.5346 (Δ -0.0465)
  - `ood_c2r` fusion: FP32 F1 0.1080 -> Hailo F1 0.0550 (Δ -0.0530)
- pending: 1
  - `ood_r2c` fusion

`ood_r2c` 중단 당시 진행도:
- 파일: `derived/hailo/pi_infer_batch/phase36_best5/phase36_ood_r2c_fusion_v5_lnfree_hailo_test/progress.csv`
- 처리량: `3987 / 7441` (약 `53.58%`)
- 중단 시점 확인: 관련 실행 프로세스 없음

---

## 4. 핵심 의사결정 로그(요약)

1. 최종 평가는 train이 아니라 test 기준으로 고정
2. FP32 vs Hailo 비교는 반드시 동일 샘플 인덱스로 정렬
3. Hailo 파이프라인은 현 시점 2입력 AV(fusion) 경로를 표준으로 운영
4. Phase-36는 15개 FP32 매트릭스를 먼저 완주하고, Hailo는 best5 우선 연결
5. 장시간 실행은 진행 상태 파일(`progress.csv`) 중심으로 가시화

---

## 5. 현재 상태 요약 (2026-02-27 종료 시점)

완료:
- 데이터 준비/분할/기초 실험/고도화/FP32 15-run 매트릭스 완료
- Hailo E4/E5 준비 및 best5 중 4개 비교 완료

보류:
- `ood_r2c` Hailo 배치 추론 1건 미완료(중단 상태 유지)

연구 결론(현재까지):
- main은 0.7대(앙상블) 달성 이력 확보
- OOD는 여전히 낮아 도메인 적응이 다음 핵심 개선축
- Hailo는 FP32 대비 성능 하락이 일관되게 관찰되며(대략 F1 -0.04~-0.05), 동일 인덱스 기준 비교 체계는 확립됨

---

## 6. 관련 최신 문서 목록

- 통합 진행 기록(이전): `derived/reports/research_design_process_status_until_2026-02-26.md`
- 결과 리뷰: `derived/reports/phase36_results_analysis_review_2026-02-27.md`
- FP32 리더보드: `derived/reports/phase36_fp32_test_leaderboard_2026-02-26.md`
- FP32 vs Hailo: `derived/reports/phase36_fp32_vs_hailo_best5.csv`
- 발표 대본: `derived/reports/research_progress_presentation_script_2026-02-24_ko.md`

