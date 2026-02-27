# 고도화 연구 설계서 (Advancement v3)

## 0. 문서 목적
이 문서는 `deep-research-report (3).md`의 후속 고도화 설계서이며, 현재 저장소 상태(Phase-2/Phase-3 결과)를 기준으로 **E4(경량화: PTQ/QAT) 착수 전 성능 고도화 단계(Phase-3.5)** 를 정의한다.

핵심 목표는 다음 3가지다.
1. 멀티모달 전처리 품질을 올려 모델 입력 표현력을 강화한다.
2. 알고리즘 비교 실험을 체계화해 성능 병목을 정확히 분리한다.
3. 사전학습 표현(오디오/비디오)을 도입해 macro-F1 중심 성능을 개선한다.

---

## 0.1 적용 점검 (2026-02-24, 코드/결과 기준)

아래는 “0.7 목표를 위한 강한 버전” 관점에서, 실제 반영 여부를 코드/결과로 점검한 표다.

| 트랙 | 설계 요구 | 현재 적용 상태 | 근거 |
|---|---|---|---|
| P-track | 오디오 log-mel/delta 기반 고도화 | **부분 적용** | `scripts/prepare_advanced_features.py`에 log-mel(80)+delta 통계 특징 존재 |
| P-track | loudness norm, VAD, CMVN, SpecAugment | **미적용** | 해당 처리/옵션 없음 |
| P-track | 얼굴 ROI/품질플래그(blur/brightness) | **미적용** | ROI/품질플래그 추출 로직 없음 |
| P-track | raw 기반 재추출 + 사전학습 비디오 임베딩 | **적용** | `cache_v3`, `resnet18` 임베딩 사용 |
| A-track | LR/RF/RBF-SVM 비교 | **적용** | `scripts/train_ml_baselines.py` |
| A-track | XGBoost/LightGBM | **미적용** | 지원 없음 |
| A-track | Gated Fusion / Cross-attention-lite | **부분 적용** | `train_fp32_multitask.py`에 `--fusion-type gated`, `--modality-dropout-p` 반영 |
| A-track | CE vs Focal, weighted sampler | **적용** | `scripts/train_fp32_multitask.py` 옵션 존재 |
| A-track | label smoothing | **부분 적용** | `train_fp32_multitask.py` CE 경로에 옵션 적용(`--label-smoothing`) |
| T-track | Pretrained Video 백본 | **부분 적용** | `resnet18` 반영, `resnet34/efficientnet_b0` 옵션 추가 |
| T-track | Pretrained Audio(wav2vec2/HuBERT/PANN) | **부분 적용(강화)** | `wav2vec2_base` 실험 + `hubert_base` 실측 완료, `wavlm_base_plus` 옵션 추가 |
| Domain Adapt | Cross-domain 적응 | **적용(후속)** | `CORAL` 추가 및 cross 성능 개선 확인 |

핵심 결론:
- “강한 버전”은 **부분 적용** 상태다.
- 즉, “이미 전부 적용됨”이 아니라, 고성능 핵심 항목(ROI/품질플래그, gated/cross-attention, 오디오 사전학습의 감정특화 fine-tuning)은 아직 남아 있다.

## 0.2 0.7 목표 기준 현황

- Main 최고 단일모델 (2026-02-24 최신): macro-F1 **0.6992**
  - `fp32_v8_hubert_gated_wide_tune4`
- Main 최고 앙상블 (2026-02-24 최신): macro-F1 **0.7099**
  - `fp32_v8_hubert_ensemble_vote3_main_t3_t4` (3-model majority vote)
- Cross 최고 (2026-02-24):
  - CREMA->RAVDESS: **0.3207** (`v8 hubert logreg+CORAL`)
  - RAVDESS->CREMA: **0.3187** (`v8 hubert logreg+CORAL`)

따라서 `0.7` 목표는 **앙상블 기준으로 달성**했다.
단, 단일모델 기준으로는 아직 `0.0008` 부족하므로 다음 우선순위는 **단일모델 0.7 상회 재현 + cross 일반화 강화**다.

---

## 0.2b 0.9 목표 버전 (신규, 2026-02-24)

이 절은 기존 `0.7` 목표와 분리된 상위 목표 버전이다.

### 0.2b.1 목표 정의
- Main 단일모델 macro-F1 `>= 0.90`
- Main 앙상블 macro-F1 `>= 0.92`
- Cross macro-F1 양방향 각각 `>= 0.60`

### 0.2b.2 현재 대비 격차
- 단일모델 최고: `0.6992` -> `0.90`까지 `+0.2008`
- 앙상블 최고: `0.7099` -> `0.92`까지 `+0.2101`
- Cross 최고:
  - CREMA->RAVDESS: `0.3207` -> `0.60`까지 `+0.2793`
  - RAVDESS->CREMA: `0.3187` -> `0.60`까지 `+0.2813`

### 0.2b.3 전제 조건
- 현 구조(고정 캐시 특징 + 얕은 MLP/ML 중심)만으로는 0.9 도달 가능성이 낮다.
- 0.9는 아래 미완료 항목의 대규모 구현이 선행돼야 한다:
  1. P-track: ROI/품질플래그 + CMVN/SpecAugment + 강화된 오디오/비디오 증강
  2. A-track: cross-attention-lite, boosting 계열(XGBoost/LightGBM), 품질플래그 기반 gated fusion
  3. T-track: 오디오/비디오 pretrained backbone의 부분 unfreeze fine-tuning
  4. Domain Adapt: deep feature-level adaptation + consistency/self-training

### 0.2b.4 운영 원칙
- 0.7 버전 결과는 기준선으로 유지하고, 0.9 버전 실험은 별도 실험군(prefix: `phase35_v9_*`)으로 분리 관리한다.
- 목표 달성 판단은 단일 수치가 아니라 `main + cross + 분산(시드)`를 함께 충족할 때만 인정한다.

---

## 0.3 실행 정책 (장기 실험)

- 수렴 경고가 있더라도 프로세스가 정상 CPU 사용/진행 상태를 보이면 **중단하지 않고 완주**한다.
- 장시간 실험(예: `linear_svm` 고차원 feature)은 progress 파일과 시스템 프로세스를 병행 모니터링한다.
- 중단은 사용자 명시 요청 또는 실제 비정상 정지(프로세스 종료/오류 루프)에서만 수행한다.

---

## 0.4 최신 실행 로그 요약 (2026-02-24)

- Strong v1(`cache_v4`)에서 누락됐던 `linear_svm` main/cross를 재실행하여 모두 완주:
  - main: `v5_linsvm_main` F1 0.5346
  - cross best(linsvm): 0.2693 / 0.2438 (CORAL)
- Next v6(FP32 고도화) 실행 완료:
  - `fp32_v6_ce_ls_ws_main` F1 0.5985
  - `fp32_v6_focal_ws_main` F1 0.5905
- Next v7(고급 fusion + 미실행 장기실험) 실행 완료:
  - `fp32_v7_ce_ls_ws_gated_wide_main` F1 **0.6056** (신규 main 최고)
  - `ml_v7_rbfsvm_main` F1 0.6002
  - `fp32_v7_ce_ls_ws_gated_main` F1 0.5929
  - `fp32_v7_focal_ws_gated_main` F1 0.5889
- GPU 전환(CUDA) 완료:
  - `.venv`를 `torch 2.10.0+cu126`로 교체했고 `device=auto`가 `cuda`로 해석됨을 실험으로 검증
  - CUDA 재실행 `fp32_v7_ce_ls_ws_gated_wide_main_cuda` F1 0.5992 (`device_resolved=cuda`)
- Next v8(HuBERT) 실행 완료:
  - `fp32_v8_hubert_gated_wide_main` F1 **0.6913**
  - `fp32_v8_hubert_gated_wide_tune1` F1 **0.6950**
  - `fp32_v8_hubert_gated_wide_tune2` F1 **0.6961**
  - `fp32_v8_hubert_gated_wide_tune3` F1 **0.6940** (추가 탐색, 최고 미갱신)
  - `fp32_v8_hubert_gated_wide_tune4` F1 **0.6992** (단일모델 최고 갱신)
  - `fp32_v8_hubert_ensemble_vote3` F1 **0.7047**
  - `fp32_v8_hubert_ensemble_vote3_main_t3_t4` F1 **0.7099** (앙상블 최고 갱신)
- 결과 리포트:
  - `derived/reports/phase35_next_v8_metrics.csv`
  - `derived/reports/phase35_next_v8_results.md`
- v8 cache 기반 cross-domain 재측정(`logreg + CORAL`) 완료:
  - `v8_hubert_logreg_coral_cross_crema_to_ravdess` F1 **0.3207**
  - `v8_hubert_logreg_coral_cross_ravdess_to_crema` F1 **0.3187**
  - 기존 v5 최고(0.3025 / 0.2724) 대비 양방향 모두 개선
- 결론: HuBERT 기반 고도화로 성능이 크게 상승했고, 0.7은 앙상블에서 달성했다. 남은 핵심 과제는 단일모델 0.7 상회 재현과 cross-domain 성능 동반 개선이다.

---

## 1. 현재 상태 요약 (기준 시점: 2026-02-20)

### 1.1 데이터/분할
- 데이터 정리 완료: `derived/manifests/summary.json`
- 학습 주 매니페스트: `derived/manifests/manifest_multimodal_common6_av.jsonl`
- 평가 분할:
  - Actor-independent 5-fold: `derived/splits/groupkfold5_all/*`
  - Cross-dataset: `derived/splits/cross_dataset/*`

### 1.2 현재 성능 핵심
출처: `derived/reports/phase3_global_metrics.csv`, `derived/reports/phase3_vs_phase2_bootstrap.csv`
- Main
  - Phase-2 fusion macro-F1: 0.3950
  - Phase-3 FP32 macro-F1: 0.3775 (delta -0.0175)
- Cross CREMA->RAVDESS
  - Phase-2 fusion macro-F1: 0.2288
  - Phase-3 FP32 macro-F1: 0.2865 (delta +0.0577)
- Cross RAVDESS->CREMA
  - Phase-2 fusion macro-F1: 0.0714
  - Phase-3 FP32 macro-F1: 0.0705 (거의 동일)

### 1.3 문제 정의
- Main 기준 macro-F1이 기준선(Phase-2 fusion)보다 낮아짐.
- 정확도는 일부 향상되나 macro-F1이 낮아 소수 클래스 대응 약함.
- Cross 성능의 방향 비대칭이 커 도메인 갭 문제가 큼.

---

## 2. 왜 점수가 높지 않은가 (원인 분석)

### 2.1 전처리/특징의 표현력 한계
- 현재 ML baseline 특징은 통계 기반 저차원 요약(오디오 16차원, 비디오 14차원) 중심.
- 얼굴 ROI, 표정 디테일, 시간적 패턴, 주파수 패턴 정보가 충분히 반영되지 않음.

### 2.2 모델 용량 및 강한 사전학습/튜닝 부족
- `scripts/train_fp32_multitask.py`는 경량 MLP 기반 듀얼 인코더 구조.
- `wav2vec2_base`/ImageNet 임베딩은 일부 반영됐지만, HuBERT/WavLM 실측 비교와 감정특화 fine-tuning은 미완료라 일반화 여력이 제한됨.

### 2.3 학습 전략의 한계
- class imbalance 대응(샘플링, focal, balanced softmax) 고도화 부족.
- macro-F1 최적화 관점의 하이퍼파라미터 탐색이 충분치 않음.

### 2.4 도메인 갭
- CREMA-D와 RAVDESS 간 촬영/발화 스타일 차이가 커 cross 일반화가 불안정.

---

## 3. 고도화 목표 (Phase-3.5 KPI)

### 3.1 1차 목표 (E4 진입 전 승급 조건)
1. Main macro-F1이 최소 `0.3950`(Phase-2 fusion) 이상 회복.
2. Cross 실험 2방향 중 최소 1방향에서 bootstrap CI 기준 유의 개선.
3. arousal2 MAE는 현재 값 이상 악화되지 않도록 제한.

### 3.2 2차 목표
- 모델 복잡도 증가 대비 추후 INT8 변환 가능성(연산/구조 단순성) 유지.

---

## 4. Phase-3.5 실험 설계

## 4.1 전처리 고도화 트랙 (P-track)

### P1. 오디오 전처리 V2
- 16kHz mono 고정 + loudness normalization
- VAD 기반 유효 구간 비율 추출
- log-mel(80) + delta + delta2
- utterance-level CMVN
- 훈련 시 증강:
  - SpecAugment (time/freq masking)
  - 소량 잡음/볼륨 perturbation

### P2. 비디오 전처리 V2
- 얼굴 중심 ROI 추출(검출 실패 시 center crop fallback)
- 2초 윈도우에서 16/32 프레임 고정 샘플링 비교
- 프레임 정규화 + 약한 color jitter/occlusion 증강
- 품질 플래그 기록:
  - face_detect_success_ratio
  - blur/brightness proxy

### P3. 멀티모달 정합
- 오디오/비디오 타임 윈도우 정렬 규칙 고정
- 품질 플래그를 fusion 입력에 추가

---

## 4.2 알고리즘 비교 트랙 (A-track)

### A1. Classical 비교군
- Logistic Regression (existing)
- SVM (RBF)
- Gradient Boosting 계열(가능 시 XGBoost/LightGBM)

### A2. Deep 비교군
- MLP (existing improved)
- Audio CNN + Video CNN + Late Fusion
- Gated Fusion (품질 플래그 활용)
- (선택) Cross-attention-lite

### A3. 손실/학습 전략 비교
- CrossEntropy vs Focal Loss
- class-balanced sampler on/off
- label smoothing on/off
- macro-F1 early stopping 기준 고정

---

## 4.3 사전학습 트랙 (T-track)

### T1. 오디오 사전학습 임베딩
- 후보: wav2vec2/HuBERT/PANN
- 1단계: 백본 freeze + 얕은 head 학습
- 2단계: 상위 일부 레이어만 unfreeze

### T2. 비디오 사전학습 백본
- 후보: MobileNet/EfficientNet(ImageNet) + temporal pooling
- 가능 시 FER 사전학습 체크포인트 활용

### T3. 멀티모달 결합
- pretrained encoder feature concat
- gated fusion으로 modality reliability 반영

---

## 5. 실험 프로토콜 (고정)

### 5.1 데이터/분할 고정
- In-domain: `groupkfold5_all`
- Cross-domain:
  - CREMA->RAVDESS
  - RAVDESS->CREMA

### 5.2 공통 지표
- Emotion: accuracy, macro-F1, OVR-AUC
- Arousal: MAE
- 통계: bootstrap CI

### 5.3 공정 비교 원칙
- 전처리 버전별 feature cache 분리 (`cache_v1`, `cache_v2`, ...)
- 실험마다 seed 고정
- 동일 split에서 모델만 바꿔 비교

---

## 6. 실행 단계 (정확한 순서)

## Step 1. 전처리 V2 파이프라인 구현
- 목표: `derived/features/cache_v2` 생성
- 산출물:
  - V2 feature manifest/meta
  - 전처리 품질 리포트(추출 성공률, 누락률)

## Step 2. 알고리즘 비교 실험 실행
- 최소 실험군:
  - `ML-LR(v2)`
  - `ML-SVM(v2)`
  - `DL-MLP(v2)`
  - `DL-GatedFusion(v2)`
- 결과 저장:
  - `derived/results/phase35_*`

## Step 3. 사전학습 모델 도입 실험
- 최소 실험군:
  - `Pretrain-Audio + VideoBase`
  - `Pretrain-Video + AudioBase`
  - `Pretrain-Audio + Pretrain-Video + GatedFusion`

## Step 4. Phase-3.5 통합 분석
- 리포트 생성:
  - `derived/reports/phase35_global_metrics.csv`
  - `derived/reports/phase35_pairwise_bootstrap.csv`
  - `derived/reports/phase35_results.md`
  - 성능/효율 그래프(svg)

## Step 5. E4 착수 판단
- 3.1 승급 조건 충족 시 E4(PTQ/QAT) 시작
- 미충족 시:
  - 데이터 증강 강도/샘플링/손실 가중 재탐색

---

## 7. 구현 원칙 (이 저장소 기준)

### 7.1 코드 구조 권장
- `scripts/prepare_advanced_features.py`
- `scripts/train_phase35_models.py`
- `scripts/run_phase35_experiments.sh`
- `scripts/analyze_phase35_results.py`
- `scripts/README_phase35_advancement.md`

### 7.2 기존 코드 재사용
- split/manifest 로딩 로직은 기존 재사용 (`train_ml_baselines.py`)
- 통계/CI는 `scripts/research_metrics.py` 재사용
- 진행 상태 출력은 `progress.json` 패턴 동일 적용

### 7.3 재현성
- 모든 실험에 run config JSON 저장
- random seed, 데이터 버전, feature cache 버전 명시

---

## 8. 리스크 및 대응
- 리스크 1: 사전학습 모델 도입 시 연산량 과다
  - 대응: freeze 전략, 경량 head 우선
- 리스크 2: Cross 성능 불안정
  - 대응: domain augmentation + calibration + gated fusion
- 리스크 3: E4 변환성 저하
  - 대응: Hailo-friendly 연산 구조 유지(과도한 custom op 회피)

---

## 9. 최종 결론
현재 단계에서 바로 E4로 가는 것보다, **Phase-3.5(전처리/모델/사전학습 고도화 + 체계적 비교)** 를 선행하는 것이 타당하다.

본 문서의 승급 조건을 만족한 모델을 기준으로 E4(PTQ/QAT)와 E5(온디바이스 실측)로 넘어가야, 최종 연구 결과의 설득력(정확도-지연-전력 트레이드오프)이 높아진다.
