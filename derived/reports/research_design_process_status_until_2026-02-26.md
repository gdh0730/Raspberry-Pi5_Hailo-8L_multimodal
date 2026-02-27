# 연구 설계/진행/의사결정 통합 기록 (기준일: 2026-02-26)

## 1. 문서 목적
이 문서는 본 프로젝트의 연구 설계, 단계별 실행 과정, 핵심 의사결정, 현재 상태를 단일 기준 문서로 통합 기록한다.  
근거는 실제 코드/설계서/실험 산출물 파일이며, 추정 서술을 최소화한다.

핵심 근거 문서:
- `deep-research-report (3).md`
- `deep-research-report-advancement (3).md`
- `derived/reports/project_progress_until_2026-02-20.md`
- `derived/reports/full_research_process_and_results_until_2026-02-24_ko.md`
- `scripts/README_phase35_advancement.md`
- `hailo/README_E4.md`

---

## 2. 연구 목표와 평가 축
프로젝트의 실질 목표는 다음 2축을 동시에 만족하는 것이다.

1. 모델 성능 축  
- CREMA-D + RAVDESS 기반 멀티모달 감정(6-class) 성능 개선  
- ID(in-domain)와 OOD(cross-dataset) 조건 분리 평가  
- 최종 평가는 test 기준으로 확정

2. 배포 축  
- FP32 모델을 Hailo-8L 추론 경로로 변환  
- 동일 샘플 기준 FP32 vs Hailo 성능 격차 정량화  
- Pi 실기기에서 배치 추론 재현성 확보

---

## 3. 데이터/분할 설계 이력

### 3.1 데이터 준비
실행: `scripts/prepare_research_data.py`  
핵심 산출:
- `derived/manifests/manifest_multimodal_common6_av.jsonl`
- `derived/manifests/summary.json`
- `derived/splits/groupkfold5_*/*`
- `derived/splits/cross_dataset/*`

핵심 수치(`derived/manifests/summary.json`):
- all: 11,762
- crema_d: 7,442
- ravdess: 4,320
- common6_all: 10,610
- multimodal_common6_av: 8,498
- ravdess_av_common6: 1,056
- ravdess_audio_only_common6: 1,056

### 3.2 분할 관련 핵심 의사결정
1. actor-independent를 기본 원칙으로 채택  
- 근거 구현: `scripts/prepare_research_data.py` (`assign_group_folds` 기반 분할)

2. cross-dataset 평가를 별도 트랙으로 분리  
- `train_crema_test_ravdess_common6_av_*`
- `test_crema_train_ravdess_common6_av_*`

3. train/val/test 누수 검증을 수동 점검 루틴으로 반복 수행  
- 최근 점검 결과: 주요 split 간 `clip_id`/`actor_id` 교집합 0 확인

---

## 4. 모델/전처리 설계 이력

### 4.1 Baseline 단계
1. B0 majority
- 산출: `derived/results/b0_majority_av/summary.json`
- 역할: 하한선 기준

2. Phase-2 ML baselines (audio/video/fusion)
- 산출: `derived/results/ml_baselines_main/summary.json`
- 분석: `derived/reports/phase2_results.md`
- 의사결정: in-domain에서는 fusion 우세, cross에서는 도메인 갭 큼

### 4.2 FP32 딥러닝 단계
1. 초기 FP32 멀티태스크
- 코드: `scripts/train_fp32_multitask.py`
- 산출: `derived/results/fp32_multitask_main/summary.json`

2. Phase-3.5 고도화(전처리/학습 레시피 고도화)
- 관련 파이프라인:
  - `scripts/run_phase35_advancement_pipeline.sh`
  - `scripts/run_phase35_advancement_v2_pipeline.sh`
  - `scripts/run_phase35_strong_v1.sh`
  - `scripts/run_phase35_next_v6.sh`
  - `scripts/run_phase35_next_v7.sh`
  - `scripts/run_phase35_next_v8_hubert_main.sh`

주요 의사결정:
1. 전처리 캐시 버전 전략(`cache_v1 -> v2 -> v3 -> v4 -> v5_hubert`) 채택  
2. audio pretrained를 wav2vec2/hubert로 확장  
3. fusion head를 gated/wide로 확장  
4. 학습 안정화(CE+label smoothing+weighted sampler) 조합 채택

주요 결과:
- 단일 모델 최고: `fp32_v8_hubert_gated_wide_tune4`  
  - `derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune4/summary.json`
  - global macro-F1: 0.6992
- 앙상블: `fp32_v8_hubert_ensemble_vote3_main_t3_t4`
  - `derived/results/fp32_multitask_phase35_v8_hubert_ensemble_vote3_main_t3_t4/summary.json`
  - global macro-F1: 0.7099

---

## 5. Hailo 전환(E4/E5) 이력

### 5.1 파이프라인 구축
구성 요소:
- ONNX export: `hailo/export_onnx.py`
- calibration/eval npy dump: `hailo/calib_dump_npy_dir.py`
- compile: `hailo/compile_hef.sh`, `scripts/run_e4_compile_remote.sh`
- Pi 추론: `hailo/pi_infer_hailort.py`, `scripts/run_e5_infer_pi_batch.sh`
- 평가: `scripts/eval_hailo_pi_infer.py`

### 5.2 핵심 기술 의사결정
1. Hailo 호환성 확보를 위해 LayerNorm 없는 LN-free 체크포인트 별도 확보  
- 산출: `derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune4_hailo_lnfree_fold0/checkpoints/best_fold_0.pt`

2. FP32와 Hailo 비교는 반드시 동일 표본(index.csv) 기준으로 수행  
- 최근 비교 파일:
  - `derived/hailo/pi_infer_batch/fp32_v8_lnfree_fold0_val1690/summary.json`
  - `derived/hailo/pi_infer_batch/fp32_v8_lnfree_fold0_test_ravdess1056/summary.json`
  - `derived/hailo/pi_infer_batch/fp32_v8_lnfree_fold0_test_ravdess1056/compare_fp32_vs_hailo.json`

3. 정규화 통계(train mu/sd)를 Hailo 입력 생성 시 동일 적용  
- 근거 코드: `hailo/calib_dump_npy_dir.py`

### 5.3 운영 중 발생 이슈와 해결
1. `No module named numpy` / `No module named pip`  
- 원인: venv 불완전  
- 해결: 환경 부트스트랩/의존성 설치 로직 강화

2. `--max-samples 0` 동작 버그  
- 증상: 전체가 아니라 1개 샘플만 수집  
- 해결: `hailo/calib_dump_npy_dir.py` 조건 수정

3. train 기반 평가 과대해석 리스크  
- 해결: valid/test 분리셋 재평가로 교정

### 5.4 최신 Hailo 결과(요약)
1. valid(1690)  
- Hailo: acc 0.6320 / f1 0.6316  
- FP32(동일표본): acc 0.6527 / f1 0.6509  

2. OOD test (CREMA->RAVDESS, 1056)  
- Hailo: acc 0.8286 / f1 0.8294  
- FP32(동일표본): acc 0.9252 / f1 0.9255  
- 격차: 약 0.096 (acc/f1)

---

## 6. 현재 상태(2026-02-26) 정리

### 6.1 확정된 것
1. 데이터/분할/전처리/학습/분석/발표 산출물이 전반적으로 재현 가능한 상태
2. Hailo 컴파일/추론 파이프라인은 실기기에서 동작 확인됨
3. FP32 vs Hailo를 동일 index 기준으로 비교하는 절차가 자리잡음

### 6.2 미완료 항목
1. ID + OOD 전 구간에서 `audio/video/fusion` 3모드를 동일 프로토콜로 완주한 Hailo 매트릭스 미완료
2. Phase-36 FP32 결과의 원인분석(특히 OOD 저성능 구간) 및 개선 실험 미완료
3. 0.9 목표(고난도 목표)에 대한 추가 구조 실험(사전학습 파인튜닝 고강도/도메인 적응 고도화) 미완료

---

## 7. 최근 의사결정 로그(핵심)
1. val 점수는 개발 지표로만 사용하고, 최종 판단은 test로 분리한다.
2. train 기반 calibration/eval 지표는 참고용으로만 두고, 보고 지표는 val/test로 재정렬한다.
3. Hailo 비교는 반드시 FP32와 동일 샘플 index를 사용한다.
4. ID/OOD x (audio/video/fusion) FP32 매트릭스를 선완주하고, 동일 index로 Hailo 추론을 연결한다.

---

## 8. 다음 실행 계획(즉시 착수)
1. Phase-36 FP32 전조합(5트랙x3모드) 완료 및 리더보드 고정
2. 각 조합별 ONNX/HEF 변환 자동화
3. Pi 배치 추론과 FP32 동표본 비교 리포트 자동화
4. OOD 저성능 구간에 대한 원인분석/개선 실험 루프 착수

이 문서 작성 이후, 위 1번부터 즉시 실행한다.

---

## 9. 실행 로그 업데이트 (2026-02-26, Phase-36 착수)

### 9.1 완료된 실행
1. split 재구축
- 명령: `python scripts/build_phase36_splits.py`
- 산출: `derived/splits/phase36_id_ood/summary.json`
- 핵심: 모든 트랙 train/val/test 간 clip/actor overlap 0

2. OOD C2R fusion
- 실행: `phase36_ood_c2r_fusion_v5_lnfree`
- 산출: `derived/results/phase36/phase36_ood_c2r_fusion_v5_lnfree/phase36_run_meta.json`
- test(1056): acc 0.2216 / macro-F1 0.1080

3. OOD C2R audio
- 실행: `phase36_ood_c2r_audio_v5_lnfree`
- 산출: `derived/results/phase36/phase36_ood_c2r_audio_v5_lnfree/phase36_run_meta.json`
- test(1056): acc 0.2008 / macro-F1 0.0898

### 9.2 현재 진행 항목
1. Phase-36 FP32 매트릭스 완료(5트랙 x 3모드 = 15 runs)
2. 통합 리더보드/CSV 산출 완료
3. 다음 단계: 동일 test index 기반 Hailo 배치 추론 매트릭스 실행

### 9.3 Phase-36 중간 결과(test, FP32)
완료 run 기준 요약:

| Track | Audio F1 | Video F1 | Fusion F1 | 비고 |
|---|---:|---:|---:|---|
| id_all | 0.6244 | 0.3065 | 0.6261 | fusion≈audio >> video |
| id_crema | 0.6663 | 0.3244 | 0.6643 | audio≈fusion >> video |
| id_ravdess | 0.5359 | 0.4072 | 0.5811 | fusion > audio > video |
| ood_c2r | 0.0898 | 0.2658 | 0.1080 | video > fusion > audio |
| ood_r2c | 0.1018 | 0.1242 | 0.1278 | fusion≈video > audio (절대값 낮음) |

해석(최신):
1. ID에서는 audio/fusion이 안정적 우세, video 단독은 일관되게 낮다.
2. OOD(CREMA->RAVDESS)에서는 audio 분포 적응이 크게 무너지고 video가 상대적으로 강하다.
3. OOD(RAVDESS->CREMA)에서도 세 모드 절대 성능이 모두 낮아, 도메인 이동 문제 자체가 주된 병목으로 확인된다.
4. 세부 수치 파일:
- `derived/reports/phase36_fp32_test_progress.csv`
- `derived/reports/phase36_fp32_test_leaderboard_2026-02-26.md`

### 9.4 남은 실행
1. 15개 run별 ONNX/HEF 변환 및 실행 가능성 검증
2. 동일 index로 Pi Hailo 배치 추론 매트릭스 수행
3. FP32-Hailo gap 리포트(트랙/모드별) 산출

---

## 10. 분석/리뷰 및 다음 단계 착수 (2026-02-27)

### 10.1 결과 리뷰 기록
리뷰 문서 작성:
- `derived/reports/phase36_results_analysis_review_2026-02-27.md`

핵심 결론:
1. ID는 `audio/fusion`이 우세, video 단독은 일관되게 낮음.
2. OOD는 절대 성능이 낮고, 이동 방향별 우세 모드가 달라 일반화 병목이 큼.
3. Hailo 단계는 15개 전량 동시보다 `track별 best` 우선 연결이 효율적.

### 10.2 다음 단계 실행(실행 완료/진행 중)
1. 자동화 스크립트 추가:
- `scripts/run_phase36_hailo_best5.sh`
- 기능: best run 자동선정 -> ONNX export -> train calib 생성 -> HEF compile

2. `id_all` best run 실제 실행 완료:
- run: `phase36_id_all_fusion_v5_lnfree`
- 산출:
  - ONNX: `derived/hailo/onnx/phase36/phase36_id_all_fusion_v5_lnfree_full.onnx`
  - calib: `derived/hailo/calib/phase36/phase36_id_all_fusion_v5_lnfree_train1024`
  - HEF: `derived/hailo/build/phase36/phase36_id_all_fusion_v5_lnfree/phase36_id_all_fusion_v5_lnfree.hef`
  - meta: `derived/hailo/phase36_best5_build_meta.csv`
  - candidates: `derived/hailo/phase36_best5_candidates.csv`

3. Pi 실기기 배치 추론 착수(진행 중):
- name: `phase36_id_all_fusion_v5_lnfree_hailo_test1690`
- 대상: `id_all` test 1690 샘플 전체
- 진행 파일:
  - `derived/hailo/pi_infer_batch/phase36_id_all_fusion_v5_lnfree_hailo_test1690/progress.csv`

4. Hailo 입력 제약 확인 및 트랙 선택 보정:
- 제약: 현재 컴파일/추론 스크립트는 2입력 AV 모델 전용
  - `hailo/compile_custom_onnx_sdk.py` (2입력 가정)
  - `hailo/pi_infer_hailort.py` (audio/video 2입력 가정)
- 영향: 단일모달 best(`id_crema/audio`, `ood_c2r/video`)는 Hailo compile 실패
- 조치: Hailo 단계는 `fusion-only best-per-track`로 재선정
  - 산출: `derived/hailo/phase36_best5_build_meta.csv`
  - 결과: 5트랙 모두 `build_status=ok`

5. FP32-Hailo 비교(현재 확정):
- `derived/reports/phase36_fp32_vs_hailo_best5.csv`
- `id_all/fusion`: FP32 F1 0.6261 -> Hailo F1 0.5892 (delta -0.0369)
- `id_ravdess/fusion`: FP32 F1 0.5811 -> Hailo F1 0.5346 (delta -0.0465)
- `id_crema/fusion`: FP32 F1 0.6643 -> Hailo F1 0.6162 (delta -0.0482)
- `ood_c2r/fusion`: FP32 F1 0.1080 -> Hailo F1 0.0550 (delta -0.0530)
- 부분 확정 리포트(ood_r2c 완료 전):  
  - `derived/reports/phase36_fp32_vs_hailo_best5_partial_until_ood_c2r.csv`
  - `derived/reports/phase36_fp32_vs_hailo_best5_partial_until_ood_c2r.md`

6. 현재 진행 중:
- `id_crema/fusion` 완료: Hailo F1 0.6162 (FP32 0.6643, delta -0.0482)
- `ood_c2r/fusion` 완료: Hailo F1 0.0550 (FP32 0.1080, delta -0.0530)
- `ood_r2c/fusion` Pi 전체 test(7441) 배치 추론 진행 중
  - progress: `derived/hailo/pi_infer_batch/phase36_best5/phase36_ood_r2c_fusion_v5_lnfree_hailo_test/progress.csv`
  - log: `derived/hailo/pi_infer_batch/phase36_best5/_logs/run_phase36_hailo_eval_ood_r2c_fg.log`
