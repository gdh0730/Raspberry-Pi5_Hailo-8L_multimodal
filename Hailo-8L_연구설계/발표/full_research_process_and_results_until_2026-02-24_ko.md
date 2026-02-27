# 영상+음성 멀티모달 연구 전체 프로세스 종합 보고서 (방법론 보강판)

- 작성일: 2026-02-24
- 기준 저장소: `Raspberry-Pi+Hailo-8`
- 기준 문서: `deep-research-report (3).md`, `deep-research-report-advancement (3).md`
- 보고 범위: 데이터 준비 -> Phase-1/2/3 -> Phase-3.5(v2~v8) -> GPU 전환 -> 최신 종합 해석
- 현재 정책: 사용자 요청에 따라 **신규 실험은 일시 보류**, 분석/보고 품질 보강에 집중

---

## 0. Executive Summary

### 0.1 한 줄 결론
본 프로젝트는 멀티모달 전처리-모델-도메인적응을 단계적으로 고도화해 main macro-F1을 `0.3950 -> 0.7099`까지 끌어올렸고, cross도 양방향 `0.32` 수준까지 회복했다. 다만 `0.9`는 현재 파이프라인의 부분 구현 상태로는 구조적으로 어려우며, 설계서의 미완료 고강도 트랙을 실제 코드로 확장해야 한다.

### 0.2 최신 최고 성능
출처: `derived/reports/phase35_next_v8_metrics.csv`, `derived/reports/phase35_cross_domain_adapt_metrics.csv`

- Main 단일모델 최고: `fp32_v8_hubert_gated_wide_tune4` -> macro-F1 `0.6992`
- Main 앙상블 최고: `fp32_v8_hubert_ensemble_vote3_main_t3_t4` -> macro-F1 `0.7099`
- Cross 최고:
  - CREMA->RAVDESS: `0.3207`
  - RAVDESS->CREMA: `0.3187`

### 0.3 해석 요약
- `0.7`은 앙상블 기준 달성, 단일모델도 임계점 근접.
- 성능 개선의 핵심 기여는 단일 요인보다 “표현 강화(HuBERT) + 결합 구조(gated/wide) + 도메인 정렬(CORAL)”의 조합 효과.
- `0.9` 목표는 단순 하이퍼파라미터 탐색이 아니라 데이터 품질 제어/사전학습 파인튜닝/강한 도메인적응의 동시 업그레이드가 필요.

---

## 1. 연구 배경 (영상+음성 멀티모달 관점)

### 1.1 문제 정의
감정은 단일 신호가 아니라, 음성(억양/에너지/리듬)과 영상(표정/시선/미세근육 변화)의 상호작용으로 나타난다. 따라서 단일모달 모델은 다음 상황에서 취약하다.

- 오디오 취약: 무성 구간, 배경 소음, 마이크/레벨 편차
- 비디오 취약: 얼굴 검출 실패, 모션 블러, 조명 편차
- 데이터셋 전이 취약: CREMA-D와 RAVDESS의 촬영/발화 스타일 차이

### 1.2 본 연구의 핵심 연구질문(RQ)
- RQ1: 멀티모달 결합이 단일모달 대비 일관된 성능 이득을 제공하는가?
- RQ2: 전처리/표현을 고도화하면 성능 병목이 얼마나 해소되는가?
- RQ3: 사전학습 특징과 결합 구조 고도화가 `0.7` 구간 진입에 유효한가?
- RQ4: 도메인적응 없이도 cross 일반화가 가능한가, 아니면 필수적인가?

### 1.3 연구 운영 원칙
- main(in-domain)과 cross(out-of-domain)를 분리 측정
- macro-F1 중심 평가(클래스 불균형 대응)
- 단계별 고립 비교(같은 분할/같은 지표에서 한 요소씩 변경)
- 모든 실험 산출물(`summary.json`, `predictions.csv`, `progress.json`) 저장 및 추적

---

## 2. 데이터 전처리 방법론

### 2.1 데이터 구성과 라벨 정합
출처: `derived/manifests/summary.json`

- 전체 샘플: `11,762`
- CREMA-D: `7,442`
- RAVDESS: `4,320`
- Common-6 전체: `10,610`
- AV(Common-6): `8,498`
- RAVDESS AO(Common-6): `1,056`

라벨 체계:
- Emotion: common-6(`angry/disgust/fearful/happy/neutral/sad`)로 통일
- Arousal: `arousal2`, `arousal3`를 보조 태스크로 병행

### 2.2 분할/검증 설계
- Main: `GroupKFold(5)` actor-independent 분할
- Cross:
  - CREMA->RAVDESS
  - RAVDESS->CREMA

방법론적 의미:
- 같은 모델을 두 가지 일반화 축에서 분리 검증
- 단순 랜덤 분할 대비 배우 누수(leakage) 위험을 통제

### 2.3 전처리 파이프라인 설계 원리
핵심은 “표현력 향상”과 “재현성 보장”의 균형이다.

- 표현력 향상: log-mel/delta, pretrained embedding, modality 결합
- 재현성 보장: 캐시 버전(`cache_v1~v5_hubert`) 고정, 실험별 분리

### 2.4 캐시 버전별 방법론적 차이
- `cache_v1`: 초기 통계 특징(저비용, 저표현)
- `cache_v2`: 고급 통계 특징(log-mel/delta 기반) 추가
- `cache_v3`: raw 재추출 + pretrained video embedding
- `cache_v4`: audio pretrained(wav2vec2 계열) 강화
- `cache_v5_hubert`: HuBERT 오디오 표현 강화

### 2.5 전처리 품질 관리(현재/미완료)
현재 강점:
- 캐시 체계 덕분에 전처리 변경 효과를 모델 변경 효과와 분리 가능
- 멀티모달/단일모달 동시 비교를 동일 분할에서 수행 가능

설계 대비 미완료:
- 얼굴 ROI crop + 품질플래그(blur/brightness/success ratio)
- CMVN, SpecAugment, loudness normalization의 full training pipeline

결론:
- 현재 성능 상승은 “기본 고도화 + pretrained embedding”의 효과가 크고,
- `0.9`를 위해서는 품질 제어형 전처리(quality-aware preprocessing)가 필요.

---

## 3. 모델 설계/학습 방법론

### 3.1 비교군 설계 철학
단계별로 모델 복잡도를 점진 증가시키며, 반드시 하위 기준선을 유지한다.

- B0: majority baseline
- ML baseline: LR / LinearSVM / RBF-SVM / RF
- FP32 multitask: dual-encoder + fusion + multi-head

### 3.2 FP32 멀티태스크 구조
핵심 구현: `scripts/train_fp32_multitask.py`

- 입력: 오디오 특징 + 비디오 특징
- fusion:
  - `concat`
  - `gated`(`--fusion-type gated`) + modality dropout
- 출력 head:
  - emotion(주태스크)
  - arousal2, arousal3(보조태스크)
- 결측 라벨 처리:
  - `-1` 마스킹으로 손실 계산 제외(샘플 자체는 유지)

방법론적 의의:
- 결측 라벨 샘플을 버리지 않아 데이터 효율 유지
- multitask regularization을 통해 표현 공유 유도

### 3.3 학습 전략 고도화 요소
- 손실: CE / Focal 비교
- 불균형 대응: weighted sampler
- 안정화: label smoothing, gradient clipping, cosine scheduler
- 선택적 구조 강화: wider head, gated fusion

### 3.4 사전학습 활용 방식
- 오디오: wav2vec2 -> HuBERT 순으로 표현 강화
- 비디오: pretrained embedding(resnet/efficientnet 계열)
- 현재 중심은 “특징 추출 후 후단 학습”이며, backbone unfreeze fine-tuning은 제한적

### 3.5 도메인적응 방법
- CORAL 적용으로 출발/도착 도메인 공분산 차이를 완화
- same classifier(logreg) 기준 `none vs coral`을 비교하여 적응 효과를 분리

---

## 4. 비교/평가 방법론

### 4.1 핵심 평가 지표
- Emotion: macro-F1(주지표), accuracy, OVR-AUC
- Arousal: MAE

이유:
- macro-F1은 다중 클래스 불균형에서 실제 인식 품질을 더 잘 반영
- accuracy 단독 해석의 함정(다수 클래스 편향)을 보정

### 4.2 단계별 비교 설계
각 단계는 아래 통제 원칙을 따른다.

- 동일한 분할(main/cross)
- 동일한 라벨 정의(common-6)
- 동일한 주요 지표(macro-F1)
- 이전 최고 모델 대비 delta를 명시

### 4.3 통계적 신뢰성
- bootstrap 기반 비교 리포트 유지
- 실험 산출물의 파일 단위 보존으로 재현성 확보

### 4.4 운영 가시화
- 장시간 실험에 `progress.json` 도입
- 실행 중 무출력 구간 문제를 상태 파일과 단계 로그로 해결

---

## 5. 단계별 연구 진행: 방법론 + 비교/평가 + 인사이트

아래는 모든 단계를 동일 템플릿으로 정리한 내용이다.

- 연구질문: 이 단계가 검증하려는 가설
- 방법론: 데이터/모델/학습/평가 설계
- 비교/평가: 핵심 수치와 이전 단계 대비 차이
- 결론(인사이트): 왜 그런 결과가 나왔는지와 다음 단계 설계 근거

### 5.1 Phase-1(B0): 기준선 확립

연구질문:
- 최소 기준선은 어디인가?

방법론:
- 최빈 클래스 예측(majority)으로 하한선 정의

비교/평가:
- B0 macro-F1 약 `0.05` 수준

결론(인사이트):
- 후속 단계의 개선폭을 정량 해석할 수 있는 기준점 확보
- 이 단계의 목적은 성능이 아니라 비교 기준의 통계적 앵커(anchor) 설정

### 5.2 Phase-2(ML baseline): 멀티모달 기본 효과 검증

연구질문:
- 고전 모델에서도 fusion 이득이 나타나는가?

방법론:
- LR/SVM/RF를 audio/video/fusion으로 병렬 비교
- main + cross를 함께 기록

비교/평가:
출처: `derived/reports/phase2_global_metrics.csv`

- Main
  - audio: `0.3671`
  - video: `0.2536`
  - fusion: `0.3950`(최고)
- Cross
  - C->R fusion: `0.2288`
  - R->C fusion: `0.0714`

결론(인사이트):
- main에서는 멀티모달 결합 이득이 명확
- cross는 방향 비대칭이 매우 커, 도메인 차이가 핵심 병목임을 확인

### 5.3 Phase-3(FP32 multitask): 딥러닝 구조의 초기 검증

연구질문:
- dual-encoder multitask가 ML baseline을 안정적으로 초과하는가?

방법론:
- emotion + arousal2 + arousal3 멀티태스크 학습
- 결측 arousal 라벨 마스킹 처리로 샘플 보존

비교/평가:
출처: `derived/reports/phase3_global_metrics.csv`

- main: `0.3775` (phase2 fusion 대비 `-0.0175`)
- C->R: `0.2865` (phase2 fusion 대비 `+0.0577`)
- R->C: `0.0705` (phase2 fusion 대비 `-0.0009`)

결론(인사이트):
- 딥러닝 구조가 항상 main 우세를 보장하지는 않음
- 그러나 cross 한 방향 개선이 확인되어, 구조 자체보다 “표현/전처리/학습전략”이 성패를 좌우한다는 신호 확보

### 5.4 Phase-3.5 v2: 전처리 고도화 1차

연구질문:
- 특징 표현 강화가 성능 저하 구간을 회복할 수 있는가?

방법론:
- `cache_v2` 기반 고급 통계 특징
- ML/FP32 동시 비교로 모델 의존성 분리

비교/평가:
출처: `derived/reports/phase35_advancement_metrics.csv`

- 최고(main): `ml_v2_logreg_fusion = 0.4205`

결론(인사이트):
- phase2(`0.3950`)와 phase3(`0.3775`)를 모두 초과
- 병목은 모델 크기보다 입력 표현 품질에 더 크게 걸려 있었음

### 5.5 Phase-3.5 v3: pretrained video 확장

연구질문:
- pretrained video embedding 추가가 main/cross 모두에 이득인가?

방법론:
- `cache_v3`(raw 재추출 + pretrained video)
- main/cross 동시 측정

비교/평가:
출처: `derived/reports/phase35_advancement_v2_main_metrics.csv`, `derived/reports/phase35_cross_domain_adapt_metrics.csv`

- main 최고: `ml_v3_rbfsvm_fusion = 0.4807`
- cross baseline(logreg, no adapt) 예시:
  - C->R: `0.1190`
  - R->C: `0.1066`

결론(인사이트):
- main은 크게 상승했지만 cross는 붕괴 위험 노출
- 즉, 표현 강화만으로는 전이 일반화가 보장되지 않으며 적응 메커니즘이 필요

### 5.6 Phase-3.5 v4/v5: domain adaptation(CORAL) + strong v1

연구질문:
- CORAL이 cross 붕괴를 회복할 수 있는가?

방법론:
- 동일 logreg에 대해 `none` vs `coral` 비교
- strong v1(audio pretrained 강화)와 결합

비교/평가:
출처: `derived/reports/phase35_strong_v1_main_metrics.csv`, `derived/reports/phase35_cross_domain_adapt_metrics.csv`

- main: `v5_logreg_main = 0.5734`
- cross best(v5+CORAL):
  - C->R: `0.3025`
  - R->C: `0.2724`

결론(인사이트):
- CORAL은 cross 개선에 실효적
- main과 cross를 동시에 올리려면 “표현 강화 + 도메인 정렬”의 결합이 필요

### 5.7 Phase-3.5 v6: 학습전략 정교화

연구질문:
- loss/샘플링/스케줄 조정만으로도 유의미한 성능 도약이 가능한가?

방법론:
- CE+label smoothing+weighted sampler
- Focal+weighted sampler 비교

비교/평가:
출처: `derived/reports/phase35_next_v6_metrics.csv`

- `fp32_v6_ce_ls_ws_main = 0.5985`
- `fp32_v6_focal_ws_main = 0.5905`

결론(인사이트):
- 단순 구조 변경이 아닌 학습 recipe 조정만으로도 +0.02~0.03 상승 가능
- 이 구간부터는 “표현+학습” 결합 최적화의 효과가 누적되기 시작

### 5.8 Phase-3.5 v7: fusion 구조 강화 + 장기 실험 완주

연구질문:
- gated/wide 구조가 `0.6` 이상 구간을 안정화하는가?

방법론:
- gated fusion + wider head
- 장시간 실험(예: linear_svm)도 중단 없이 완주 정책 적용

비교/평가:
출처: `derived/reports/phase35_next_v7_metrics.csv`

- 최고(main): `fp32_v7_ce_ls_ws_gated_wide_main = 0.6056`
- 대안(main): `ml_v7_rbfsvm_main = 0.6002`

결론(인사이트):
- 구조 강화 효과는 유효하나 단독으로는 `0.7`까지 부족
- 더 강한 오디오 표현(pretrained) 도입의 필요성이 명확해짐

### 5.9 GPU 전환(CUDA): 탐색 효율 인프라 확보

연구질문:
- 고도화 실험 반복을 GPU 기준으로 안정화할 수 있는가?

방법론:
- `torch +cu126` 전환
- `device=auto` 해석/학습 경로 점검

비교/평가:
- CUDA 인식 및 학습 경로 정상 동작 확인

결론(인사이트):
- 이 단계는 성능 직접 상승보다 실험 회전율 향상이 목적
- 이후 v8 다중 튜닝/앙상블 탐색이 가능해지는 기반 단계

### 5.10 Phase-3.5 v8(HuBERT): 0.7 구간 진입

연구질문:
- HuBERT 기반 오디오 표현이 임계 구간(`0.7`) 진입을 가능하게 하는가?

방법론:
- `cache_v5_hubert`
- FP32 gated-wide tune1~tune4
- vote ensemble 조합 탐색

비교/평가:
출처: `derived/reports/phase35_next_v8_metrics.csv`

- 단일모델 최고: `0.6992` (tune4)
- 앙상블 최고: `0.7099` (main+t3+t4)

결론(인사이트):
- `0.7`은 앙상블 기준 달성
- 단일모델도 문턱 직전이므로, 미완료 전처리/미세튜닝이 이어지면 단일 `0.7+` 가능성 높음

### 5.11 v8 기반 cross 재측정: 전이 일반화 재확인

연구질문:
- v8 표현 개선이 cross에도 일관되게 전달되는가?

방법론:
- `cache_v5_hubert + logreg + CORAL` 양방향 재측정

비교/평가:
출처: `derived/reports/phase35_cross_domain_adapt_metrics.csv`

- C->R: `0.3207` (v5+CORAL 대비 `+0.0182`)
- R->C: `0.3187` (v5+CORAL 대비 `+0.0463`)

결론(인사이트):
- v8은 main뿐 아니라 cross에서도 개선
- 도메인적응을 병행하면 표현 고도화의 이득이 전이 성능으로 연결됨

---

## 6. 단계별 성능 추세 요약

### 6.1 Main 추세

| 구간 | 대표 실험 | macro-F1 | 이전 구간 대비 해석 |
|---|---|---:|---|
| Phase-2 | fusion baseline | 0.3950 | 멀티모달 기본 이득 확인 |
| Phase-3 | fp32_main | 0.3775 | 구조만 변경 시 성능 저하 가능성 확인 |
| v2 | ml_v2_logreg | 0.4205 | 전처리 개선 효과 확인 |
| v3 | ml_v3_rbfsvm | 0.4807 | pretrained video로 대폭 상승 |
| v5 | ml_v5_logreg | 0.5734 | strong audio + 적응 체계 결합 효과 |
| v6 | fp32_v6_ce_ls_ws | 0.5985 | 학습 recipe 최적화 효과 |
| v7 | fp32_v7_gated_wide | 0.6056 | 결합 구조 강화의 추가 이득 |
| v8(single) | fp32_v8_tune4 | 0.6992 | HuBERT로 임계점 근접 |
| v8(ensemble) | vote3_main_t3_t4 | **0.7099** | 0.7 달성 |

### 6.2 Cross 추세

| 방향 | phase2 fusion | v5(logreg+CORAL) | v8(hubert+CORAL) | 해석 |
|---|---:|---:|---:|---|
| CREMA->RAVDESS | 0.2288 | 0.3025 | **0.3207** | 도메인 정렬 + 표현 강화 동시 필요 |
| RAVDESS->CREMA | 0.0714 | 0.2724 | **0.3187** | 초기 붕괴 구간에서 의미 있는 회복 |

---

## 7. 설계서 대비 구현 상태 점검 (멀티모달 고도화 관점)

출처: `deep-research-report-advancement (3).md`

| 트랙 | 설계 요구 | 현재 상태 | 실무적 의미 |
|---|---|---|---|
| P-track | mel/delta + ROI + 품질플래그 + CMVN/SpecAug | 부분 적용 | mel/delta/임베딩은 적용, 품질 제어형 전처리는 미완료 |
| A-track | LR/SVM/RF + boosting + 고급 fusion | 부분 적용 | LR/SVM/RF/gated 완료, boosting/cross-attention-lite 미완료 |
| T-track | pretrained + 부분 unfreeze fine-tuning | 부분 적용 | embedding 기반 강화 성공, backbone fine-tuning은 제한적 |
| Domain | cross 적응 | 적용 | CORAL의 일관된 개선 확인 |

결론:
- 지금까지의 상승은 실제로 유효하지만, 설계서의 “강한 버전” 전체가 완결된 상태는 아니다.
- 따라서 `0.9`를 목표로 할 때는 이미 성공한 경로를 반복하기보다 미완료 트랙을 코드 수준으로 채워야 한다.

---

## 8. 0.7 목표와 0.9 목표를 분리한 해석

### 8.1 0.7 목표(현재)
- 상태: 달성(앙상블), 단일모델 임계점 근접
- 연구적 의미: 현재 파이프라인이 “동작한다”는 증거 확보

### 8.2 0.9 목표(다음)
- 단일모델 기준 격차: `+0.2008`
- 앙상블 기준 격차(0.92): `+0.2101`

필수 조건:
- 품질 제어 전처리(ROI/quality-aware)
- pretrained backbone 부분 unfreeze fine-tuning
- attention 기반 멀티모달 결합
- CORAL 이상의 feature-level adaptation/self-training

### 8.3 방법론적 결론
`0.9`는 “같은 실험을 더 오래”로 얻기 어렵다. 전처리/모델/적응을 동시에 상향하는 구조 전환이 전제되어야 한다.

---

## 9. 산출물 인덱스

### 9.1 핵심 보고서
- `derived/reports/full_research_process_and_results_until_2026-02-24_ko.md`
- `derived/reports/full_research_process_and_results_until_2026-02-24.md`
- `derived/reports/project_progress_until_2026-02-20.md`

### 9.2 단계별 결과 요약 파일
- `derived/reports/phase2_global_metrics.csv`
- `derived/reports/phase3_global_metrics.csv`
- `derived/reports/phase35_advancement_metrics.csv`
- `derived/reports/phase35_advancement_v2_main_metrics.csv`
- `derived/reports/phase35_strong_v1_main_metrics.csv`
- `derived/reports/phase35_next_v6_metrics.csv`
- `derived/reports/phase35_next_v7_metrics.csv`
- `derived/reports/phase35_next_v8_metrics.csv`
- `derived/reports/phase35_cross_domain_adapt_metrics.csv`

### 9.3 발표자료
- `derived/reports/research_progress_summary_2026-02-24.pptx`
- `derived/reports/research_progress_summary_2026-02-24.pdf`
- 생성 스크립트: `scripts/generate_full_progress_presentation.py`

---

## 10. 최종 정리

이 프로젝트는 멀티모달 연구에서 흔히 발생하는 문제(표현력 부족, 도메인 갭, 학습 불안정)를 단계별로 분해해 해결해 왔다. 특히 성능 상승은 단일 기법이 아니라, 전처리-표현-결합-적응의 누적 개선 효과로 설명된다. 즉, 현재 결과는 “우연한 최고점”이 아니라 방법론적 개선의 결과이며, 다음 목표(`0.9`) 역시 같은 원리로 미완료 고강도 트랙을 구현할 때 현실적으로 접근 가능하다.
