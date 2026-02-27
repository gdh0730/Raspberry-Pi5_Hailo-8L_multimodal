# 발표 Q&A 확장본 (2026-02-24 기준)

이 문서는 `derived/reports/research_progress_summary_2026-02-24.pptx`(45장) 발표용 확장 질의응답만 정리한 자료다.  
답변은 실제 코드/설정/결과 파일을 근거로 작성했다.

---

## Q01. 이 연구의 핵심 목표는 무엇인가?
짧은 답변: CREMA-D + RAVDESS 기반 멀티모달(영상+음성) 감정인식에서 main 성능과 cross-domain 일반화를 동시에 끌어올리는 것이다.

세부 답변: main은 actor-independent 5-fold에서 macro-F1을 올리고, cross는 CREMA->RAVDESS와 RAVDESS->CREMA 양방향 성능 붕괴를 완화하는 것이 핵심 KPI다.

근거 파일: `deep-research-report-advancement (3).md`, `derived/reports/phase2_global_metrics.csv`, `derived/reports/phase35_cross_domain_adapt_metrics.csv`

## Q02. 왜 단일모달이 아니라 멀티모달인가?
짧은 답변: 감정 단서가 음성(억양/에너지)과 영상(표정/움직임)에 분산되어 있어서다.

세부 답변: 실제 baseline에서도 audio-only와 video-only보다 fusion이 안정적으로 높다. phase-2 main에서 audio F1 0.3671, video 0.2536, fusion 0.3950으로 fusion 이점이 확인된다.

근거 파일: `derived/reports/phase2_global_metrics.csv`

## Q03. 현재 최고 성능은 정확히 얼마인가?
짧은 답변: 단일모델 최고 macro-F1은 0.6992, 앙상블 최고는 0.7099다.

세부 답변: 단일 최고는 `fp32_v8_hubert_gated_wide_tune4`, 앙상블 최고는 `fp32_v8_hubert_ensemble_vote3_main_t3_t4`다.

근거 파일: `derived/reports/phase35_next_v8_metrics.csv`

## Q04. 0.7 목표는 달성한 건가?
짧은 답변: 앙상블 기준으로 달성했다. 단일모델은 0.6992로 0.0008 부족했다.

세부 답변: 목표 정의를 단일/앙상블로 분리해 보면, 단일은 임계값 직전, 앙상블은 초과 달성이다.

근거 파일: `derived/reports/phase35_next_v8_metrics.csv`, `deep-research-report-advancement (3).md`

## Q05. 0.9 목표와 현재 격차는?
짧은 답변: 단일기준 +0.2008, 앙상블기준 +0.2101이 필요하다.

세부 답변: 현재 best single 0.6992, best ensemble 0.7099이고, 0.9/0.92 목표와의 차이를 설계서에 분리 정의했다.

근거 파일: `deep-research-report-advancement (3).md`

## Q06. 성능이 단계적으로 실제 얼마나 올랐나?
짧은 답변: phase-2 main fusion 0.3950 -> v5 0.5734 -> v7 0.6056 -> v8 single 0.6992 -> v8 ensemble 0.7099다.

세부 답변: 개선은 단일 요인이 아니라 cache 고도화 + 학습레시피 + pretrained 오디오(HuBERT) + gated fusion의 누적 효과로 나타났다.

근거 파일: `derived/reports/phase2_global_metrics.csv`, `derived/reports/phase35_strong_v1_main_metrics.csv`, `derived/reports/phase35_next_v7_metrics.csv`, `derived/reports/phase35_next_v8_metrics.csv`

## Q07. 그런데 phase-3가 phase-2보다 낮았던 이유는?
짧은 답변: 초기 FP32 세팅이 아직 표현/학습 레시피가 충분히 고도화되지 않아 macro-F1이 일시 하락했다.

세부 답변: phase-3 main F1은 0.3775로 phase-2 fusion 0.3950 대비 -0.0175였고, 이후 phase-3.5에서 설정/특징을 재구성하면서 회복 및 초과했다.

근거 파일: `derived/reports/phase3_global_metrics.csv`

## Q08. 데이터 규모는 어떻게 되나?
짧은 답변: 전체 11,762, CREMA-D 7,442, RAVDESS 4,320이다.

세부 답변: 연구의 핵심 학습 매니페스트(공통 6감정 AV)는 8,498건이다.

근거 파일: `derived/manifests/summary.json`

## Q09. 왜 8,498(AV common6)만 학습에 쓰나?
짧은 답변: 멀티모달 정합(오디오+비디오 동시 존재)과 공통 감정 라벨 일관성을 맞추기 위해서다.

세부 답변: AO(video 없음) 또는 video-only(audio 없음)는 멀티모달 실험의 주 비교 축에서는 제외하고 별도 데이터 자산으로 유지한다.

근거 파일: `derived/manifests/summary.json`, `derived/manifests/manifest_multimodal_common6_av.jsonl`

## Q10. RAVDESS AO(audio-only)는 완전히 불필요한가?
짧은 답변: 완전히 불필요하진 않지만, 현재 핵심 멀티모달 본 실험에는 직접 투입하지 않았다.

세부 답변: AO 1,440건은 향후 audio-only 보조학습/도메인 적응 실험용으로 활용 가능하다. 현재 main/cross AV 프로토콜의 주 학습셋은 AV 정합셋이다.

근거 파일: `derived/manifests/summary.json`

## Q11. 분할 방식은 누수(leakage) 없이 설계됐나?
짧은 답변: actor 기준 GroupKFold와 고정 cross split을 사용해 누수 위험을 낮췄다.

세부 답변: in-domain은 `groupkfold5_all`, cross는 사전에 생성된 train/test list를 사용해 방향별 일반화 성능을 평가한다.

근거 파일: `derived/splits/groupkfold5_all/fold_0_val.txt`, `derived/splits/cross_dataset/train_crema_test_ravdess_common6_av_train.txt`, `scripts/train_ml_baselines.py`

## Q12. 라벨 불균형은 어느 정도인가?
짧은 답변: neutral(1375)가 다른 감정보다 적어 macro-F1 기준에서 난도가 높다.

세부 답변: angry/disgust/fearful/happy/sad는 각 1847로 균등하지만 neutral이 상대적으로 적다.

근거 파일: `derived/manifests/summary.json`

## Q13. cache_v1 특징은 무엇인가?
짧은 답변: 얕은 통계형 baseline 특징이다.

세부 답변: audio 16차원(시간/주파수 요약), video 14차원(밝기/프레임 통계)으로 구성된다.

근거 파일: `scripts/train_ml_baselines.py`, `derived/features/cache_v1`

## Q14. cache_v2는 무엇이 달라졌나?
짧은 답변: 특징량을 확장한 advanced cache 버전이다.

세부 답변: v2는 source-cache lifting 또는 raw 추출 경로를 지원하며, 현재 저장된 샘플 기준 차원은 audio 56, video 49다.

근거 파일: `scripts/prepare_advanced_features.py`, `derived/features/cache_v2/summary.json`

## Q15. cache_v3는 어떤 의미인가?
짧은 답변: raw 재추출 + 비디오 pretrained 임베딩(resnet18) 결합 버전이다.

세부 답변: 샘플 기준 차원은 audio 328, video 1044로 크게 증가했다. 다만 8,498건 중 1건 video feature fail이 발생했다.

근거 파일: `derived/features/cache_v3/summary.json`, `scripts/prepare_advanced_features.py`

## Q16. cache_v4와 cache_v5 차이는?
짧은 답변: 둘 다 오디오 pretrained를 붙인 고차원 cache지만, v4는 wav2vec2, v5는 HuBERT다.

세부 답변: 둘 다 audio 1864차원으로 같지만 백본이 다르다. v4는 `wav2vec2_base`, v5는 `hubert_base`를 사용했다.

근거 파일: `scripts/run_phase35_strong_v1.sh`, `scripts/run_phase35_next_v8_hubert_main.sh`, `derived/features/cache_v4/summary.json`, `derived/features/cache_v5_hubert/summary.json`

## Q17. wav2vec2는 정확히 어디서 썼나?
짧은 답변: strong-v1의 cache_v4 생성 단계에서 사용했다.

세부 답변: `scripts/run_phase35_strong_v1.sh` STEP 1에 `--audio-pretrained-backbone wav2vec2_base`가 명시되어 있다.

근거 파일: `scripts/run_phase35_strong_v1.sh`

## Q18. HuBERT는 정확히 어디서 썼나?
짧은 답변: v8의 cache_v5_hubert 생성 단계와 이를 입력으로 하는 main/cross 실험에서 사용했다.

세부 답변: `run_phase35_next_v8_hubert_main.sh` STEP 1에서 `hubert_base`로 cache를 만들고, 이후 FP32 main과 cross logreg+CORAL 재측정에 사용했다.

근거 파일: `scripts/run_phase35_next_v8_hubert_main.sh`, `derived/results/ml_baselines_phase35_v8_hubert_logreg_coral_cross_crema_to_ravdess/summary.json`

## Q19. 오디오 전처리에서 mel은 실제 적용됐나?
짧은 답변: 적용됐다.

세부 답변: `audio_feature_v2`에 log-mel(80), delta 통계, 에너지 기반 보조 통계가 구현되어 있다.

근거 파일: `scripts/prepare_advanced_features.py`

## Q20. 비디오 crop(얼굴 ROI)은 적용됐나?
짧은 답변: 현재 구현에는 얼굴 ROI 검출/크롭이 없다.

세부 답변: 현재는 ffmpeg로 고정 스케일링(예: 96x96, pretrained용 224x224)하며, 설계서의 ROI/품질플래그 항목은 미적용 상태다.

근거 파일: `scripts/prepare_advanced_features.py`, `deep-research-report-advancement (3).md`

## Q21. VAD/CMVN/SpecAugment는 적용됐나?
짧은 답변: 완전 적용은 아니다.

세부 답변: 현재는 energy 통계/voiced_ratio 등 일부 성분만 있고, 설계서에서 제안한 full VAD+CMVN+SpecAugment 파이프라인은 미완료로 기록되어 있다.

근거 파일: `scripts/prepare_advanced_features.py`, `deep-research-report-advancement (3).md`

## Q22. 비디오 pretrained는 어느 백본까지 실험했나?
짧은 답변: 코드 옵션은 `resnet18/resnet34/efficientnet_b0`까지 있지만, 실측 핵심 결과는 resnet18 경로가 중심이다.

세부 답변: v3에서 `video_pretrained_backbone: resnet18`이 사용됐고, 이후 v8에서는 video를 v3 cache에서 재사용했다.

근거 파일: `derived/features/cache_v3/summary.json`, `scripts/prepare_advanced_features.py`, `scripts/run_phase35_next_v8_hubert_main.sh`

## Q23. 오디오 pretrained max samples=32000의 의미는?
짧은 답변: 16kHz 기준 최대 2초 구간을 pretrained 오디오 임베딩 입력으로 쓴다는 뜻이다.

세부 답변: 실험 전처리 구간 자체가 2초로 고정되어 있어 시간창 일관성이 유지된다.

근거 파일: `scripts/prepare_advanced_features.py`, `derived/features/cache_v5_hubert/summary.json`

## Q24. FP32 멀티태스크 모델 구조는?
짧은 답변: dual encoder(audio/video) + fusion(concat/gated) + emotion/a2/a3 multi-head 구조다.

세부 답변: `DualEncoderMultiTask`에서 audio/video 인코더를 각각 통과한 뒤 결합하고, emotion(6클래스), arousal2(2클래스), arousal3(3클래스) head를 함께 학습한다.

근거 파일: `scripts/train_fp32_multitask.py`

## Q25. gated fusion은 수식적으로 어떤 의미인가?
짧은 답변: 모달별 임베딩을 gate로 혼합해 신뢰도 기반 결합을 유도한다.

세부 답변: 구현상 `blend = g*za + (1-g)*zv`, 추가로 `|za-zv|`, `za*zv`를 concat해 상보성과 불일치 정보를 함께 준다.

근거 파일: `scripts/train_fp32_multitask.py`

## Q26. modality dropout은 왜 넣었나?
짧은 답변: 한 모달 의존 과적합을 줄이기 위해서다.

세부 답변: 학습 중 확률적으로 audio/video 중 일부를 drop해 결측/열화 상황에서도 견고한 결합을 유도한다.

근거 파일: `scripts/train_fp32_multitask.py`

## Q27. CE vs Focal은 어느 쪽이 유리했나?
짧은 답변: 현재 best는 CE 계열이었다.

세부 답변: v7에서 `fp32_v7_ce_ls_ws_gated_wide_main` F1 0.6056, `fp32_v7_focal_ws_gated_main` F1 0.5889로 CE 경로가 높았다.

근거 파일: `derived/reports/phase35_next_v7_metrics.csv`

## Q28. weighted sampler는 실제 적용됐나?
짧은 답변: 적용됐다.

세부 답변: v7/v8 주요 FP32 run config에 `weighted_sampler: true`가 저장되어 있고, 코드에서 `WeightedRandomSampler`를 생성한다.

근거 파일: `scripts/train_fp32_multitask.py`, `derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune4/summary.json`

## Q29. v8 단일 최고(tune4)의 핵심 하이퍼파라미터는?
짧은 답변: epochs 40, lr 4.5e-4, hidden 512, emb 256, dropout 0.2, modality_dropout 0.05, label_smoothing 0.08, weighted sampler다.

세부 답변: 이 설정이 HuBERT cache 입력과 결합되면서 single 최고 0.6992를 만들었다.

근거 파일: `derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune4/summary.json`

## Q30. v8 앙상블은 어떤 방식인가?
짧은 답변: 3개 모델 예측의 다수결(vote) 앙상블이다.

세부 답변: source는 `main`, `tune3`, `tune4`의 predictions.csv이고 tie는 source order로 처리한다.

근거 파일: `derived/results/fp32_multitask_phase35_v8_hubert_ensemble_vote3_main_t3_t4/summary.json`

## Q31. GPU는 실제로 사용됐나?
짧은 답변: 사용됐다.

세부 답변: 주요 v8 run summary에 `device_resolved: cuda`, `cuda_name: NVIDIA GeForce RTX 4090`가 기록되어 있다.

근거 파일: `derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune4/summary.json`, `derived/features/cache_v5_hubert/summary.json`

## Q32. 진행 상태는 어떻게 추적했나?
짧은 답변: 각 학습/추출 스크립트가 `progress.json`을 지속 갱신한다.

세부 답변: stage/fold/epoch/percent/elapsed가 저장되며 비대화형 환경에서도 주요 체크포인트를 출력하도록 설계됐다.

근거 파일: `scripts/train_ml_baselines.py`, `scripts/train_fp32_multitask.py`

## Q33. 왜 실행 중 출력이 없던 문제가 있었나?
짧은 답변: non-interactive shell에서는 진행라인 출력이 제한적으로 동작하기 때문이다.

세부 답변: 이후 `progress.json` 기반 모니터링과 단계별 로그를 통해 가시성을 보완했다.

근거 파일: `scripts/train_ml_baselines.py`, `derived/results/ml_baselines_progress_check/progress.json`

## Q34. cross-domain에서 CORAL은 어떤 역할을 했나?
짧은 답변: source(train) 특징 공분산을 target(val) 특징 공분산에 정렬해 도메인 갭을 줄였다.

세부 답변: 타깃 라벨은 쓰지 않고 feature covariance만 맞추는 비지도 정렬이며, 코드에 `coral_align_train_to_val`로 구현돼 있다.

근거 파일: `scripts/train_ml_baselines.py`

## Q35. CORAL 효과는 수치로 얼마나 컸나? (CREMA->RAVDESS)
짧은 답변: phase-2 fusion 0.2288 대비 v8 HuBERT+CORAL 0.3207로 +0.0919 상승했다.

세부 답변: v5 CORAL(0.3025)보다도 추가 개선이 있어, CORAL + HuBERT 결합이 유효했다.

근거 파일: `derived/reports/phase2_global_metrics.csv`, `derived/reports/phase35_cross_domain_adapt_metrics.csv`

## Q36. CORAL 효과는 수치로 얼마나 컸나? (RAVDESS->CREMA)
짧은 답변: phase-2 fusion 0.0714 대비 v8 HuBERT+CORAL 0.3187로 +0.2473 상승했다.

세부 답변: 개선 폭이 큰 이유는 초기 성능이 매우 낮았기 때문이며, 여전히 main 대비 절대 성능 갭은 남아 있다.

근거 파일: `derived/reports/phase2_global_metrics.csv`, `derived/reports/phase35_cross_domain_adapt_metrics.csv`

## Q37. 그런데 cross가 아직 낮은 이유는?
짧은 답변: 데이터셋 촬영/발화 스타일 차이(도메인 갭)가 매우 크고, 현재는 얕은 feature-level 정렬(CORAL) 중심이기 때문이다.

세부 답변: 설계서에서 제안한 deep adaptation/self-training/consistency는 아직 본격 적용 전이라 ceiling이 남아 있다.

근거 파일: `deep-research-report-advancement (3).md`

## Q38. 왜 일부 결과에서 n=8498이 아니라 n=8497인가?
짧은 답변: cache_v3 생성 시 1건의 `video_feature_fail`이 발생해 이후 파이프라인에서 해당 샘플이 제외됐기 때문이다.

세부 답변: v8이 reuse한 video cache가 v3 기반이어서 이 1건 누락이 이어졌다.

근거 파일: `derived/features/cache_v3/summary.json`, `derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune4/summary.json`

## Q39. 통계 신뢰성(신뢰구간)은 어떻게 확보했나?
짧은 답변: bootstrap CI를 사용했다.

세부 답변: ML baseline은 기본 n_bootstrap=300, 일부 FP32/v8 비교는 200으로 기록되어 있으며 fold 및 global 지표에 CI가 저장된다.

근거 파일: `scripts/research_metrics.py`, `scripts/train_ml_baselines.py`, `derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune4/summary.json`

## Q40. v8의 fold별 변동은 큰가?
짧은 답변: 과도한 붕괴 없이 비교적 안정적이다.

세부 답변: tune4 fold macro-F1은 약 0.6679~0.7162 범위로 분포한다.

근거 파일: `derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune4/summary.json`

## Q41. arousal 멀티태스크는 왜 넣었나?
짧은 답변: emotion 단일학습의 과적합을 줄이고 정서 관련 보조신호를 제공하기 위해서다.

세부 답변: 손실은 emotion + lambda_a2*a2 + lambda_a3*a3 형태로 합산한다.

근거 파일: `scripts/train_fp32_multitask.py`

## Q42. arousal3 지표가 비어있는 경우는 왜 생기나?
짧은 답변: 해당 평가셋에서 arousal3 라벨이 없거나 매핑 불가인 샘플은 마스킹되기 때문이다.

세부 답변: cross 실험 summary에서 arousal3 n=0이 확인된다.

근거 파일: `derived/results/ml_baselines_phase35_v8_hubert_logreg_coral_cross_crema_to_ravdess/summary.json`

## Q43. linear SVM을 오래 걸려도 계속 돌린 이유는?
짧은 답변: 고차원 feature에서 수렴 시간이 길어도 정상 진행이면 완주하는 정책으로 전환했기 때문이다.

세부 답변: 설계서 실행 정책에 장기 실험 완주 원칙이 반영되어 있다.

근거 파일: `deep-research-report-advancement (3).md`, `scripts/run_phase35_strong_v1.sh`

## Q44. 재현성은 어떻게 보장했나?
짧은 답변: split 고정, seed 고정, cache 버전 분리, run summary 저장으로 보장했다.

세부 답변: 모든 핵심 실험은 `summary.json`에 하이퍼파라미터/장치/지표가 기록된다.

근거 파일: `derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune4/summary.json`, `derived/manifests/summary.json`

## Q45. “wav2vec2와 HuBERT 둘 다 썼냐” 질문에는 어떻게 답하나?
짧은 답변: “네, 둘 다 썼고 v4는 wav2vec2, v8은 HuBERT입니다.”

세부 답변: wav2vec2는 strong-v1(cache_v4), HuBERT는 next-v8(cache_v5_hubert)에서 사용했으며, 최고 성능은 HuBERT 경로에서 나왔다.

근거 파일: `scripts/run_phase35_strong_v1.sh`, `scripts/run_phase35_next_v8_hubert_main.sh`, `derived/reports/phase35_next_v8_metrics.csv`

## Q46. 지금 모델 성능이 절대적으로 낮게 보이는 이유는?
짧은 답변: 초기에는 얕은 통계형 특징과 단순 모델 구조 중심이었기 때문이다.

세부 답변: cache_v1/phase-2에서는 표현력이 제한적이었고, pretrained feature 도입 이후 급상승이 나타났다.

근거 파일: `derived/reports/phase2_global_metrics.csv`, `derived/reports/phase35_next_v8_metrics.csv`

## Q47. 0.7을 넘기려면 단일모델에서 무엇을 우선해야 하나?
짧은 답변: v8 tune4 설정 재현 + seed 안정화 + 추가 튜닝(학습률/드롭아웃/모달드롭아웃)을 우선해야 한다.

세부 답변: 이미 0.6992까지 왔기 때문에 파라미터 미세 조정과 fold 분산 관리가 단기적으로 가장 효율적이다.

근거 파일: `derived/reports/phase35_next_v8_metrics.csv`

## Q48. 0.9를 위해 지금 반드시 추가해야 하는 것은?
짧은 답변: ROI/품질플래그 기반 전처리, pretrained backbone partial unfreeze fine-tuning, deep domain adaptation이 필요하다.

세부 답변: 현재 설계 점검표에서 이 항목들은 “부분 적용/미적용”으로 남아 있어 구조적 확장이 필수다.

근거 파일: `deep-research-report-advancement (3).md`

## Q49. 지금 상태에서 바로 E4(PTQ/QAT)로 가도 되나?
짧은 답변: 권장되지 않는다.

세부 답변: 성능 구조를 더 끌어올린 뒤 경량화해야 정확도-지연-전력 트레이드오프 해석이 설득력 있다.

근거 파일: `deep-research-report-advancement (3).md`

## Q50. 발표장에서 “다음 실험 지금 바로 시작하냐”는 질문에는?
짧은 답변: “현재는 Q&A/자료 보강 단계이며, 실험 재개는 합의된 우선순위(0.7 안정화 또는 0.9 전환)에 따라 시작합니다.”

세부 답변: 현재 요청사항이 “다음 실험 보류 + 발표자료 보강”이었기 때문에, 문서와 발표 준비를 먼저 완결한 뒤 실험을 재개하는 것이 맞다.

근거 파일: `derived/reports/research_progress_presentation_script_2026-02-24_ko.md`, `deep-research-report-advancement (3).md`
