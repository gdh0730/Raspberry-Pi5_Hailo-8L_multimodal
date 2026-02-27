# 연구 프로젝트 진행 현황 정리 (시점: 2026-02-20)

## 1) 프로젝트 목표 및 범위
- 목표: CREMA-D + RAVDESS 기반 멀티모달(오디오+비디오) 감정/각성 프록시 인식 연구를 단계적으로 수행하고, 최종적으로 온디바이스 최적화(Hailo-8L 방향)까지 연결 가능한 실험 체계를 구축.
- 현재 범위 완료: 데이터 준비 -> 베이스라인(B0) -> ML baselines(Phase-2) -> FP32 딥러닝(Phase-3) + 분석 리포트 자동화.

## 2) 데이터 준비 단계 (완료)

### 2.1 수행 내용
- CREMA-D, RAVDESS 원천 데이터 다운로드/압축해제 완료.
- 표준 전처리/매니페스트 생성 스크립트 실행:
  - `scripts/prepare_research_data.py`
- 문서화:
  - `scripts/README_prepare_research_data.md`

### 2.2 생성 산출물
- 매니페스트:
  - `derived/manifests/manifest_all.jsonl`
  - `derived/manifests/manifest_common6_all.jsonl`
  - `derived/manifests/manifest_multimodal_common6_av.jsonl`
  - `derived/manifests/manifest_ravdess_audio_only_common6.jsonl`
  - 그 외 관련 manifest 일체
- 스플릿:
  - `derived/splits/groupkfold5_all/*`
  - `derived/splits/groupkfold5_crema_d/*`
  - `derived/splits/groupkfold5_ravdess/*`
  - `derived/splits/cross_dataset/*`
- 요약:
  - `derived/manifests/summary.json`

### 2.3 데이터 정합 결과 (핵심 수치)
출처: `derived/manifests/summary.json`
- 전체 샘플: 11,762
- CREMA-D: 7,442
- RAVDESS: 4,320
- Common-6 전체: 10,610
- 멀티모달 AV(Common-6): 8,498
- RAVDESS AO(Common-6): 1,056
- RAVDESS AV(Common-6): 1,056

### 2.4 초기 이슈 및 처리
- `unzip` 경고(`filename not matched`) 발생:
  - 원인: 지정한 와일드카드 패턴에 해당하는 ZIP 파일(예: `Video_Speech_Actor_02.zip`~)이 실제 폴더에 없었음.
  - 조치: 실제 다운로드된 파일 목록 기준으로 압축해제 대상 재검증.
- RAVDESS AO 필요성:
  - 결론: 주 경로(AV 멀티모달)에는 필수 아님.
  - 단, AO 전용 베이스라인/비교 실험에는 유용하며 현재 프로젝트에 보조 실험 자산으로 반영됨.

## 3) Phase-1 베이스라인(B0) (완료)

### 3.1 수행 내용
- 최빈 클래스 기준선(majority) 학습/평가 수행.
- 스크립트:
  - `scripts/train_b0_majority.py`
  - `scripts/README_phase1_baseline.md`

### 3.2 결과 산출물
- AV 베이스라인:
  - `derived/results/b0_majority_av/summary.json`
  - `derived/results/b0_majority_av/predictions.csv`
- RAVDESS AO 베이스라인:
  - `derived/results/b0_majority_ravdess_ao/summary.json`
  - `derived/results/b0_majority_ravdess_ao/predictions.csv`

### 3.3 핵심 결과
- B0 AV(global): emotion6 macro-F1 약 0.049
- B0 AO(global): emotion6 macro-F1 약 0.051
- 해석: 랜덤/최빈 수준 기준선으로서 이후 모델 개선폭 측정 기준점 확보.

## 4) 환경/실행 안정화 (완료)

### 4.1 발생 오류
1. `ModuleNotFoundError: No module named 'numpy'`
2. `.venv/bin/python: No module named pip`

### 4.2 원인
- WSL 내 `.venv` 인터프리터 기준으로 의존성 및 `pip`가 불완전한 상태였음.

### 4.3 해결
- `scripts/run_phase2_experiments.sh`에 아래 안정화 로직 반영:
  - `.venv` 존재 검사
  - `ensurepip` + `get-pip.py` 기반 pip bootstrap
  - `numpy/pandas/scikit-learn` 자동 설치
- 결과: 환경 초기화 실패 없이 Phase-2를 단일 명령으로 재실행 가능.

## 5) Phase-2 ML Baselines (완료)

### 5.1 수행 내용
- 스크립트:
  - `scripts/train_ml_baselines.py`
  - `scripts/run_phase2_experiments.sh`
- 실험 세트:
  - Main 5-fold (audio/video/fusion)
  - Cross: CREMA->RAVDESS
  - Cross: RAVDESS->CREMA

### 5.2 진행 상태 가시화 개선
- 문제: 장시간 실행 중 터미널 출력 부족으로 진행률 확인 어려움.
- 개선:
  - `progress.json` 업데이트 구조 도입
  - 단계별 상태(`features_train`, `features_val`, `train`, `completed`) 기록
  - 정기 진행 메시지 출력

### 5.3 결과 산출물
- 메인:
  - `derived/results/ml_baselines_main/summary.json`
- 교차:
  - `derived/results/ml_baselines_cross_crema_to_ravdess/summary.json`
  - `derived/results/ml_baselines_cross_ravdess_to_crema/summary.json`
- 분석 리포트:
  - `scripts/analyze_phase2_results.py`
  - `scripts/run_phase2_analysis.sh`
  - `derived/reports/phase2_results.md`
  - `derived/reports/phase2_global_metrics.csv`
  - `derived/reports/phase2_pairwise_bootstrap.csv`
  - `derived/reports/phase2_main_f1.svg`
  - `derived/reports/phase2_cross_f1.svg`

### 5.4 핵심 성능 (global)
출처: `derived/reports/phase2_global_metrics.csv`
- Main
  - audio F1: 0.3671
  - video F1: 0.2536
  - fusion F1: 0.3950 (최고)
- Cross CREMA->RAVDESS
  - audio F1: 0.2386 (최고)
  - fusion F1: 0.2288
- Cross RAVDESS->CREMA
  - audio F1: 0.1297 (최고)
  - fusion F1: 0.0714

### 5.5 런타임 벤치마크(E5 초기)
출처: `derived/results/phase2_runtime_bench_fusion_fold0.json`
- mean latency: 9.623 ms
- p50 latency: 9.506 ms
- p95 latency: 11.192 ms
- max latency: 13.653 ms
- fps_equiv: 103.90

## 6) Phase-3 FP32 멀티태스크 딥러닝 (완료)

### 6.1 수행 내용
- 신규 학습 스크립트:
  - `scripts/train_fp32_multitask.py`
- 실행 스크립트:
  - `scripts/run_phase3_fp32_main.sh`
- 문서:
  - `scripts/README_phase3_fp32.md`

### 6.2 핵심 버그 수정
- 증상: 일부 실행에서 fold 학습 샘플 수가 비정상적으로 작음(예: 1931 수준).
- 원인: `arousal2` 결측 샘플이 학습셋에서 과도하게 제외되는 데이터 필터링.
- 조치:
  - 결측 라벨은 제거하지 않고 마스킹 처리(`-1`)하여 손실 계산 시 제외.
- 결과:
  - 학습 샘플 수 정상화(예: fold별 약 6.7k~6.9k train 사용).

### 6.3 실행 완료 결과
- Main 5-fold:
  - `derived/results/fp32_multitask_main/summary.json`
- Cross CREMA->RAVDESS:
  - `derived/results/fp32_multitask_cross_crema_to_ravdess/summary.json`
- Cross RAVDESS->CREMA:
  - `derived/results/fp32_multitask_cross_ravdess_to_crema/summary.json`
- 공통 진행 파일:
  - 각 결과 폴더의 `progress.json` (`status: completed`)

### 6.4 핵심 성능 (FP32 fusion)
- Main:
  - macro-F1: 0.3775
  - accuracy: 0.4195
  - arousal2 MAE: 0.2301
  - arousal3 MAE: 0.5868
- Cross CREMA->RAVDESS:
  - macro-F1: 0.2865
  - accuracy: 0.3201
  - arousal2 MAE: 0.3400
- Cross RAVDESS->CREMA:
  - macro-F1: 0.0705
  - accuracy: 0.1709
  - arousal2 MAE: 0.4667
  - arousal3 MAE: 0.7934

## 7) Phase-3 분석 자동화 (완료)

### 7.1 수행 내용
- 스크립트 추가:
  - `scripts/analyze_phase3_results.py`
  - `scripts/run_phase3_analysis.sh`
- 목적:
  - Phase-3 FP32 결과와 Phase-2 fusion baseline의 정량 비교 자동화

### 7.2 생성 리포트
- `derived/reports/phase3_results.md`
- `derived/reports/phase3_global_metrics.csv`
- `derived/reports/phase3_vs_phase2_bootstrap.csv`
- `derived/reports/phase3_emotion_f1.svg`
- `derived/reports/phase3_vs_phase2_delta_f1.svg`

### 7.3 비교 요약 (Phase-3 vs Phase-2 fusion)
출처: `derived/reports/phase3_global_metrics.csv`, `derived/reports/phase3_vs_phase2_bootstrap.csv`
- main: delta F1 = -0.0175 (FP32가 소폭 낮음)
- cross_crema_to_ravdess: delta F1 = +0.0577 (FP32가 개선)
- cross_ravdess_to_crema: delta F1 = -0.0009 (거의 동일)

## 8) 오늘 시점 최종 상태 (2026-02-20)

### 8.1 실행 상태
- 현재 장기 학습 프로세스 실행 중 아님(종료 완료 상태).
- 주요 결과 파일 모두 생성 완료 및 재확인 완료.

### 8.2 연구 단계별 완료 여부
- 데이터 준비: 완료
- 베이스라인(B0): 완료
- Phase-2 ML baseline + 분석: 완료
- Phase-3 FP32 main/cross + 분석: 완료
- 다음 단계(E4 경량화/PTQ/QAT): 미착수
- 온디바이스 HEF 컴파일/실측(E5 본실험): 미착수

## 9) 주요 타임스탬프(파일 기준, KST)
- 2026-02-19 17:18:22: `derived/manifests/summary.json`
- 2026-02-19 17:58:43: `derived/results/b0_majority_av/summary.json`
- 2026-02-20 16:57:35: `derived/results/ml_baselines_main/summary.json`
- 2026-02-20 17:36:44: `derived/reports/phase2_results.md`
- 2026-02-20 17:37:34: `derived/results/phase2_runtime_bench_fusion_fold0.json`
- 2026-02-20 18:06:47: `derived/results/fp32_multitask_main/summary.json`
- 2026-02-20 18:11:21: `derived/results/fp32_multitask_cross_crema_to_ravdess/summary.json`
- 2026-02-20 18:13:16: `derived/results/fp32_multitask_cross_ravdess_to_crema/summary.json`
- 2026-02-20 18:16:08: `derived/reports/phase3_results.md`

## 10) 다음 세션 시작점(권장)
1. E4 경량화 실험 설계 확정: PTQ -> (필요 시) QAT + KD.
2. FP32 체크포인트(`derived/results/fp32_multitask_main/checkpoints/*.pt`) 기준 ONNX export/양자화 파이프라인 구축.
3. E5 온디바이스 실측: p50/p95 latency, FPS, power, RSS를 replay/live로 분리 측정.
4. 최종 논문용 표/그림 템플릿에 Phase-2/Phase-3 결과 자동 반영.

---

## 11) 후속 고도화 설계 문서
- 고도화 설계서(Phase-3.5): `deep-research-report-advancement (3).md`
- 실행 가이드: `scripts/README_phase35_advancement.md`

---

## 12) Phase-3.5 고도화 실행 결과 (2026-02-23)

### 12.1 수행한 파이프라인
- 실행 스크립트: `scripts/run_phase35_advancement_pipeline.sh`
- 구성:
  1. `prepare_advanced_features.py`로 `cache_v2` 생성
  2. `train_ml_baselines.py`(logreg, random_forest) 비교
  3. `train_fp32_multitask.py`(FP32 CE) 실행
  4. `analyze_phase35_advancement.py`로 통합 비교 리포트 생성

### 12.2 전처리 V2 생성 결과
- 파일: `derived/features/cache_v2/summary.json`
- 결과: total 8,498 / 성공 8,498 / 실패 0

### 12.3 핵심 성능 (main, macro-F1)
출처: `derived/reports/phase35_advancement_metrics.csv`
- `ml_v2_logreg_fusion`: **0.4205** (최고)
- `ml_v2_rf_fusion`: 0.4066
- `fp32_v2_ce_fusion`: 0.3980
- 기준 비교:
  - phase2_fusion_main: 0.3950
  - phase3_fp32_main: 0.3775

### 12.4 해석
- V2 전처리 + 알고리즘 비교 결과, `ml_v2_logreg_fusion`이 현재 최고 성능 후보.
- phase2/phase3 기준선을 모두 초과했으므로 “E4 진입 전 성능 회복” 목표는 달성.
- 단, cross-dataset 재검증(동일 V2 조건)은 추가 수행 필요.

### 12.5 생성 리포트
- `derived/reports/phase35_advancement_metrics.csv`
- `derived/reports/phase35_advancement_results.md`

### 12.6 Cross-dataset 추가 검증 (logreg fusion, cache_v2)
출처: `derived/reports/phase35_advancement_cross_metrics.csv`
- CREMA->RAVDESS: F1 0.1610 (phase2 cross 대비 -0.0678)
- RAVDESS->CREMA: F1 0.1437 (phase2 cross 대비 +0.0723)
- 해석: V2 개선이 cross 양방향에서 일관되지는 않으며, 도메인 적응 전략이 추가로 필요함.

---

## 13) Phase-3.5 고도화 2차(v3 cache + pretrained video embedding) 결과 (2026-02-23)

### 13.1 수행한 파이프라인
- 실행 스크립트: `scripts/run_phase35_advancement_v2_pipeline.sh`
- 구성:
  1. `prepare_advanced_features.py`로 `cache_v3` 생성
  2. main 실험: `train_ml_baselines.py` (logreg / rbf_svm), `train_fp32_multitask.py` (CE)
  3. cross 실험: logreg / rbf_svm 양방향
  4. `analyze_phase35_advancement_v2.py` 통합 분석

### 13.2 전처리 V3 생성 결과
- 파일: `derived/features/cache_v3/summary.json`
- 결과: total 8,498 / 성공 8,497 / 실패 1 (`video_feature_fail`)
- 설정: raw decode 기반(`--no-prefer-source-cache --fallback-raw`) + pretrained video embedding(`resnet18`, 4 frames, 224)

### 13.3 핵심 성능 (main, macro-F1)
출처: `derived/reports/phase35_advancement_v2_main_metrics.csv`
- `ml_v3_rbfsvm_fusion`: **0.4807** (최고)
- `fp32_v3_ce_fusion`: 0.4721
- `ml_v3_logreg_fusion`: 0.4149
- 기준 비교:
  - phase2_fusion_main(0.3950) 대비 최대 +0.0857
  - phase3_fp32_main(0.3775) 대비 최대 +0.1032

### 13.4 Cross-dataset 결과 (macro-F1)
출처: `derived/reports/phase35_advancement_v2_cross_metrics.csv`
- `ml_v3_logreg_cross_crema_to_ravdess`: 0.1190
- `ml_v3_logreg_cross_ravdess_to_crema`: 0.1066
- `ml_v3_rbfsvm_cross_crema_to_ravdess`: 0.0513
- `ml_v3_rbfsvm_cross_ravdess_to_crema`: 0.0486
- 해석: main 성능은 크게 개선됐지만, cross 일반화는 오히려 크게 악화되어 도메인 적응/정규화 전략이 필수.

### 13.5 생성 리포트
- `derived/reports/phase35_advancement_v2_main_metrics.csv`
- `derived/reports/phase35_advancement_v2_cross_metrics.csv`
- `derived/reports/phase35_advancement_v2_results.md`

---

## 14) Cross-domain 적응 후속 실험 (CORAL, 2026-02-23)

### 14.1 수행 내용
- 스크립트: `scripts/run_phase35_cross_domain_adapt.sh`
- 모델/설정:
  - classifier: `logreg`
  - modality: `fusion`
  - cache: `derived/features/cache_v3`
  - domain adaptation: `none`(v3 baseline) vs `coral`(v4)
- 대상 split:
  - `cross_crema_to_ravdess`
  - `cross_ravdess_to_crema`

### 14.2 핵심 결과 (macro-F1)
출처: `derived/reports/phase35_cross_domain_adapt_metrics.csv`
- CREMA->RAVDESS:
  - v3 baseline: 0.1190
  - v4 CORAL: **0.2604** (Δ +0.1414 vs v3, Δ +0.0316 vs phase2 cross)
- RAVDESS->CREMA:
  - v3 baseline: 0.1066
  - v4 CORAL: **0.2422** (Δ +0.1356 vs v3, Δ +0.1708 vs phase2 cross)

### 14.3 해석
- v3에서 붕괴됐던 cross-domain 일반화가 CORAL 적용 후 양방향 모두 큰 폭 회복.
- Main 최고 성능(v3 rbf_svm, F1 0.4807)과 별개로, cross 일반화용 후보는 `v4 logreg + CORAL`이 현시점 최선.
- 다음 단계(E4) 전 비교 기준은 `main`과 `cross`를 분리해 이원화하는 것이 타당:
  - main-optimal: `ml_v3_rbfsvm_fusion`
  - cross-optimal: `ml_v4_logreg_coral_fusion`

### 14.4 생성 산출물
- `derived/results/ml_baselines_phase35_v4_logreg_coral_cross_crema_to_ravdess/summary.json`
- `derived/results/ml_baselines_phase35_v4_logreg_coral_cross_ravdess_to_crema/summary.json`
- `derived/reports/phase35_cross_domain_adapt_metrics.csv`
- `derived/reports/phase35_cross_domain_adapt_results.md`

---

## 15) Strong v1 (audio pretrained cache_v4) 실행 결과 (2026-02-23)

### 15.1 수행 내용
- 오디오 사전학습 임베딩 추가:
  - `prepare_advanced_features.py`에 `wav2vec2_base` 오디오 임베딩 경로 추가
  - `cache_v4` 오디오 생성(8,498/8,498 성공), 비디오는 `cache_v3` 재사용
- 실험:
  - main: `v5_logreg_main`, `v5_linsvm_main` (5-fold)
  - cross: `v5_logreg`, `v5_linsvm` none/coral 양방향

### 15.2 핵심 성능
출처: `derived/reports/phase35_strong_v1_main_metrics.csv`, `derived/reports/phase35_strong_v1_cross_metrics.csv`
- Main:
  - `v5_logreg_main`: **macro-F1 0.5734**, acc 0.5729
  - `v5_linsvm_main`: macro-F1 0.5346, acc 0.5338
  - 기존 최고(v3 rbf_svm 0.4807) 대비 +0.0927
- Cross:
  - CREMA->RAVDESS:
    - logreg none: 0.1667
    - linsvm none: 0.2084
    - logreg coral: **0.3025** (best)
    - linsvm coral: 0.2693
  - RAVDESS->CREMA:
    - logreg none: 0.1439
    - linsvm none: 0.1274
    - logreg coral: **0.2724** (best)
    - linsvm coral: 0.2438

### 15.3 해석
- 오디오 pretrained 추가로 main/cross 모두 유의미하게 개선됨.
- 그러나 목표치인 macro-F1 0.7에는 아직 미달.
- 현시점 best 실험:
  - main: `v5_logreg_main` (0.5734)
  - cross: `v5_logreg + CORAL` (0.3025 / 0.2724)
- `v5_linsvm_*`는 실행 시간이 매우 길었고 수렴 경고가 반복됐으나, 요청 조건에 따라 중단 없이 완주함.

### 15.4 생성 산출물
- `derived/features/cache_v4/summary.json`
- `derived/results/ml_baselines_phase35_v5_logreg_main/summary.json`
- `derived/results/ml_baselines_phase35_v5_linsvm_main/summary.json`
- `derived/results/ml_baselines_phase35_v5_logreg_cross_crema_to_ravdess/summary.json`
- `derived/results/ml_baselines_phase35_v5_logreg_cross_ravdess_to_crema/summary.json`
- `derived/results/ml_baselines_phase35_v5_logreg_coral_cross_crema_to_ravdess/summary.json`
- `derived/results/ml_baselines_phase35_v5_logreg_coral_cross_ravdess_to_crema/summary.json`
- `derived/results/ml_baselines_phase35_v5_linsvm_cross_crema_to_ravdess/summary.json`
- `derived/results/ml_baselines_phase35_v5_linsvm_cross_ravdess_to_crema/summary.json`
- `derived/results/ml_baselines_phase35_v5_linsvm_coral_cross_crema_to_ravdess/summary.json`
- `derived/results/ml_baselines_phase35_v5_linsvm_coral_cross_ravdess_to_crema/summary.json`
- `derived/reports/phase35_strong_v1_main_metrics.csv`
- `derived/reports/phase35_strong_v1_cross_metrics.csv`
- `derived/reports/phase35_strong_v1_results.md`

---

## 16) Next v6 (FP32 label-smoothing 방향) 결과 (2026-02-23)

### 16.1 수행 내용
- `train_fp32_multitask.py`에 `--label-smoothing` 옵션 추가 후 main 5-fold 실행
- 실행 후보:
  - `fp32_v6_ce_ls_ws_main` (CE + label smoothing + weighted sampler)
  - `fp32_v6_focal_ws_main` (Focal + weighted sampler)
- 분석:
  - `scripts/analyze_phase35_next_v6.py`

### 16.2 핵심 성능 (main, macro-F1)
출처: `derived/reports/phase35_next_v6_metrics.csv`
- `fp32_v6_ce_ls_ws_main`: **0.5985** (best)
- `fp32_v6_focal_ws_main`: 0.5905
- `v5_logreg_main_ref`: 0.5734

### 16.3 해석
- v6 고도화로 main 성능이 추가 개선되어 현시점 최고는 `0.5985`.
- 여전히 목표치 0.7에는 미달이며, 다음 병목은 모델 구조(고급 fusion/attention)와 품질플래그 활용 미적용.
- cross 최고는 여전히 `v5 logreg + CORAL` (0.3025 / 0.2724).

### 16.4 생성 산출물
- `derived/results/fp32_multitask_phase35_v6_ce_ls_ws_main/summary.json`
- `derived/results/fp32_multitask_phase35_v6_focal_ws_main/summary.json`
- `derived/reports/phase35_next_v6_metrics.csv`
- `derived/reports/phase35_next_v6_results.md`

---

## 17) Next v7 (Gated fusion + 장기 RBF-SVM) 결과 (2026-02-24)

### 17.1 수행 내용
- `train_fp32_multitask.py` 고도화:
  - `--fusion-type {concat,gated}`
  - `--modality-dropout-p`
  - fusion MLP 확장(LayerNorm + deeper head)
- `prepare_advanced_features.py` 백본 확장:
  - audio: `hubert_base`, `wavlm_base_plus` 옵션 추가
  - video: `resnet34`, `efficientnet_b0` 옵션 추가
- 실행 스크립트:
  - `scripts/run_phase35_next_v7.sh`
  - `scripts/analyze_phase35_next_v7.py`
- 실험 정책:
  - `rbf_svm(cache_v4, 5-fold)` 장시간 구간을 중단 없이 완주

### 17.2 핵심 성능 (main, macro-F1)
출처: `derived/reports/phase35_next_v7_metrics.csv`
- `fp32_v7_ce_ls_ws_gated_wide_main`: **0.6056** (신규 최고)
- `ml_v7_rbfsvm_main`: 0.6002
- `fp32_v6_ce_ls_ws_main`(기존 최고 ref): 0.5985
- `fp32_v7_ce_ls_ws_gated_main`: 0.5929
- `fp32_v7_focal_ws_gated_main`: 0.5889

### 17.3 해석
- v7에서 gated fusion + wider 설정이 v6 대비 개선:
  - ΔF1 = +0.0071 (`0.5985 -> 0.6056`)
- 장시간 `rbf_svm(cache_v4)`도 완주했고 `0.6002`로 경쟁력 확인.
- 다만 목표치 `0.7`까지는 아직 `0.0944` 부족.
- 다음 고효율 우선순위:
  1. `hubert_base`/`wavlm_base_plus` 기반 cache 재생성 후 동일 프로토콜 비교
  2. ROI/품질플래그(blur/brightness/motion quality) feature 추가
  3. cross-attention-lite 또는 gated+quality fusion 추가

### 17.4 생성 산출물
- `derived/results/fp32_multitask_phase35_v7_ce_ls_ws_gated_main/summary.json`
- `derived/results/fp32_multitask_phase35_v7_focal_ws_gated_main/summary.json`
- `derived/results/fp32_multitask_phase35_v7_ce_ls_ws_gated_wide_main/summary.json`
- `derived/results/ml_baselines_phase35_v7_rbfsvm_main/summary.json`
- `derived/reports/phase35_next_v7_metrics.csv`
- `derived/reports/phase35_next_v7_results.md`

---

## 18) GPU 전환 및 CUDA 실험 검증 (2026-02-24)

### 18.1 문제 원인
- WSL에서 NVIDIA GPU(RTX 4090)는 인식됐지만, `.venv`의 PyTorch가 `2.10.0+cpu`라 학습이 CPU로 강제됨.
- 또한 `train_fp32_multitask.py`에 장치 선택 버그가 있어 `--device`와 무관하게 CPU를 사용하던 문제가 있었음.

### 18.2 수정 사항
- 장치 선택 로직 수정:
  - `scripts/train_fp32_multitask.py`
    - `--device` 기본값을 `auto`로 변경
    - `resolve_device(auto/cpu/cuda)` 추가
    - 학습/평가 텐서 이동을 비동기(`non_blocking`)로 개선
    - 요약에 `device_requested`, `device_resolved`, `cuda_name` 기록
- 사전학습 특징 추출도 GPU 사용 가능하도록 수정:
  - `scripts/prepare_advanced_features.py`
    - `--device auto` 추가
    - pretrained embedder 모델/입력 텐서를 선택 장치로 이동
    - 요약에 장치 정보 기록
- 실행 스크립트 CUDA 설치 경로 갱신:
  - `scripts/setup_ml_env.sh`, `scripts/run_phase35_*`, `scripts/run_phase3_fp32_main.sh`
  - GPU 감지 시 `cu126` 휠, 미감지 시 `cpu` 휠 설치
  - FP32 실행 커맨드에 `--device auto` 적용

### 18.3 환경 복구 결과
- `.venv` 재설치 결과:
  - `torch 2.10.0+cu126`
  - `torchvision 0.25.0+cu126`
  - `torchaudio 2.10.0+cu126`
  - `torch.cuda.is_available() == True`
  - GPU: `NVIDIA GeForce RTX 4090`

### 18.4 GPU 실험 결과 (실행 완료)
- 스모크 검증:
  - `derived/results/fp32_gpu_smoke/summary.json`
  - `device_resolved: cuda` 확인
- 실험 재실행(v7 gated wide, 5-fold/24epoch, CUDA):
  - 결과: `derived/results/fp32_multitask_phase35_v7_ce_ls_ws_gated_wide_main_cuda/summary.json`
  - global macro-F1: **0.5992**
  - `device_resolved: cuda`, `cuda_name: NVIDIA GeForce RTX 4090`

### 18.5 해석
- GPU 전환 자체는 완전 정상(학습/특징추출 모두 CUDA 사용 확인).
- 성능은 기존 CPU 최고(`0.6056`) 대비 소폭 낮게 관측(`0.5992`):
  - CUDA 비결정성 및 학습 경로 차이(연산 순서/커널) 영향 가능.
  - 필요 시 재현성 고정 모드(cudnn deterministic)로 재검증 권장.
- 다음 단계 실행 스크립트 준비 완료:
  - `scripts/run_phase35_next_v8_hubert_main.sh`
  - `scripts/analyze_phase35_next_v8.py`

---

## 19) Next v8 (HuBERT 강화 + 0.7 달성) 결과 (2026-02-24)

### 19.1 수행 내용
- `scripts/run_phase35_next_v8_hubert_main.sh` 실행:
  1. `cache_v5_hubert` 오디오 특징 생성(`hubert_base`, GPU auto)
  2. `cache_v3` 비디오 특징 재사용
  3. FP32 gated-wide main 학습
  4. `scripts/analyze_phase35_next_v8.py`로 통합 분석
- 후속 튜닝 실행:
  - `fp32_v8_hubert_gated_wide_tune1`
  - `fp32_v8_hubert_gated_wide_tune2`
  - `fp32_v8_hubert_gated_wide_tune3`
  - `fp32_v8_hubert_gated_wide_tune4`
- 3개 모델 OOF 예측 다수결 앙상블 생성:
  - `fp32_v8_hubert_ensemble_vote3`
  - `fp32_v8_hubert_ensemble_vote3_main_t3_t4`

### 19.2 핵심 성능 (main, macro-F1)
출처: `derived/reports/phase35_next_v8_metrics.csv`
- `fp32_v8_hubert_ensemble_vote3_main_t3_t4`: **0.7099** (앙상블 최고)
- `fp32_v8_hubert_ensemble_vote3`: 0.7047
- `fp32_v8_hubert_gated_wide_tune4`: **0.6992** (단일모델 최고)
- `fp32_v8_hubert_gated_wide_tune2`: 0.6961
- `fp32_v8_hubert_gated_wide_tune3`: 0.6940
- `fp32_v8_hubert_gated_wide_tune1`: 0.6950
- `fp32_v8_hubert_gated_wide_main`: 0.6913
- `fp32_v7_best_ref`: 0.6056

### 19.3 해석
- HuBERT 기반 특징으로 v7 대비 큰 폭 성능 향상:
  - 단일모델 기준 최대 `+0.0935`p (`0.6056 -> 0.6992`)
- 앙상블 기준으로 목표치 0.7을 초과:
  - 최고 `0.7099` (`gap_to_0.7 = -0.0099`)
- 남은 과제:
  1. 단일모델 0.7 상회 재현(시드/하이퍼파라미터 안정화)
  2. cross-domain 성능 동반 개선(현재는 v8 CORAL 구간으로 개선됨)
  3. ROI/품질플래그 + cross-attention-lite 반영 후 재평가

### 19.4 생성 산출물
- `derived/features/cache_v5_hubert/summary.json`
- `derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_main/summary.json`
- `derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune1/summary.json`
- `derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune2/summary.json`
- `derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune3/summary.json`
- `derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune4/summary.json`
- `derived/results/fp32_multitask_phase35_v8_hubert_ensemble_vote3/summary.json`
- `derived/results/fp32_multitask_phase35_v8_hubert_ensemble_vote3_main_t3_t4/summary.json`
- `derived/reports/phase35_next_v8_metrics.csv`
- `derived/reports/phase35_next_v8_results.md`

---

## 20) v8 HuBERT cache 기반 Cross-domain 재측정 (2026-02-24)

### 20.1 수행 내용
- v8에서 생성한 `cache_v5_hubert`를 cross-domain ML baseline에 적용.
- 실험 설정:
  - classifier: `logreg`
  - modalities: `fusion`
  - domain adaptation: `CORAL`
- 실행 결과:
  - `ml_baselines_phase35_v8_hubert_logreg_coral_cross_crema_to_ravdess`
  - `ml_baselines_phase35_v8_hubert_logreg_coral_cross_ravdess_to_crema`
- 통합 비교 리포트 재생성:
  - `scripts/analyze_phase35_cross_domain_adapt.py`

### 20.2 핵심 성능 (cross, macro-F1)
출처: `derived/reports/phase35_cross_domain_adapt_metrics.csv`
- CREMA->RAVDESS: **0.3207** (`v8_hubert_logreg_coral_cross_crema_to_ravdess`)
- RAVDESS->CREMA: **0.3187** (`v8_hubert_logreg_coral_cross_ravdess_to_crema`)

### 20.3 해석
- 기존 최고(v5 CORAL) 대비 양방향 모두 개선:
  - CREMA->RAVDESS: `0.3025 -> 0.3207` (`+0.0182`)
  - RAVDESS->CREMA: `0.2724 -> 0.3187` (`+0.0463`)
- v8 HuBERT audio feature가 in-domain뿐 아니라 cross-domain 일반화에도 유효함을 확인.

### 20.4 생성 산출물
- `derived/results/ml_baselines_phase35_v8_hubert_logreg_coral_cross_crema_to_ravdess/summary.json`
- `derived/results/ml_baselines_phase35_v8_hubert_logreg_coral_cross_ravdess_to_crema/summary.json`
- `derived/reports/phase35_cross_domain_adapt_metrics.csv`
- `derived/reports/phase35_cross_domain_adapt_results.md`

---

## 21) 0.9 목표 버전 분리 및 종합 보고/발표자료 생성 (2026-02-24)

### 21.1 수행 내용
- 고도화 설계서에서 `0.7 목표`와 분리된 `0.9 목표 버전` 섹션 추가:
  - 파일: `deep-research-report-advancement (3).md`
  - 내용: 목표 수치, 현재 격차, 전제 조건, 운영 원칙
- 현재까지의 전체 연구 프로세스/결과를 단일 종합 보고서로 정리:
  - 파일: `derived/reports/full_research_process_and_results_until_2026-02-24.md`
- 발표자료 자동 생성 스크립트 추가 및 산출물 생성:
  - 스크립트: `scripts/generate_full_progress_presentation.py`
  - PPTX: `derived/reports/research_progress_summary_2026-02-24.pptx`
  - PDF: `derived/reports/research_progress_summary_2026-02-24.pdf`

### 21.2 핵심 요약(보고서 기준)
- Main 최고 단일모델: `0.6992`
- Main 최고 앙상블: `0.7099`
- Cross 최고:
  - CREMA->RAVDESS: `0.3207`
  - RAVDESS->CREMA: `0.3187`
- 해석:
  - `0.7` 트랙은 앙상블 기준 달성.
  - `0.9` 트랙은 대규모 미구현 항목(ROI/품질플래그, attention 고도화, backbone fine-tuning 등) 선행이 필수.

### 21.3 보완판 생성 (사용자 피드백 반영)
- 종합 보고서 보완:
  - 기존 요약형 보고서를 단계별 목적/수행/결과/의사결정 구조로 확장
  - 파일: `derived/reports/full_research_process_and_results_until_2026-02-24_ko.md`
- 발표자료 보완:
  - 영어 위주 요약 슬라이드를 한국어 발표형 구성으로 전면 개편
  - 16슬라이드(배경/프로토콜/로드맵/차트/표/결론)로 재생성
  - 파일:
    - `derived/reports/research_progress_summary_2026-02-24.pptx`
    - `derived/reports/research_progress_summary_2026-02-24.pdf`
  - 생성 스크립트:
    - `scripts/generate_full_progress_presentation.py`
