# CREMA-D와 RAVDESS만으로 완성하는 Hailo-8L 온디바이스 비전+음성 감정·인지상태 모니터링 연구 설계서

## 요약
**(A, 500자 이내)** CREMA-D(7,442클립·91배우·6감정+강도)와 RAVDESS(7,356파일·24배우·AV/VO/AO·8감정+강도)만으로 ‘감정 분류’와 ‘인지상태 프록시=각성/alertness(강도)’를 멀티태스크로 학습한다. STFT/VAD/동기화는 CPU, 비전·오디오 CNN 추론은 Hailo-8L INT8(HEF)로 수행하며 FP32·CPU-only·NPU를 정확도/지연(p50,p95)/FPS/전력/메모리로 비교해 논문·데모·벤치마크를 완결한다. citeturn12view1turn6view0turn8view1turn17view0turn11view0

## 연구 설계 개요
**목적(논문형 정의)**  
본 연구는 (1) 비전+음성 기반 **감정 인식**과 (2) ‘인지 상태’의 직접 라벨이 없는 공개 데이터 제약 하에서 **각성/alertness 축을 인지상태 프록시로 정식화**하여, Raspberry Pi+Hailo-8L 환경에서 **정확도–지연–전력** 트레이드오프를 정량 보고하는 것을 목표로 한다. citeturn8view1turn9view2turn17view0turn11view0  

**인지상태 프록시의 근거(핵심 가정 명시)**  
감정의 차원모형(원형모형)에서 수직축은 arousal/activation이며 “alertness(각성)”로도 설명된다. 따라서 RAVDESS/CREMA-D가 제공하는 **감정 강도(intensity/level)** 를 “각성 기반 인지상태 프록시” 라벨로 사용한다(임상 진단이 아니라 모니터링 지표). citeturn8view1turn6view0turn12view0  

**가설**  
- H1(멀티모달 이득): 배우 독립(actor-independent) 분할에서 비전+오디오 융합이 단일모달 대비 macro-F1/OVR-AUC를 유의하게 개선한다. citeturn12view1turn5view1turn13search0turn13search7  
- H2(INT8/경량화): PTQ→(필요 시)QAT+지식증류 적용 시 FP32 대비 성능 저하를 제한하면서 Pi에서 p95 지연/FPS/전력 목표를 달성한다. citeturn11view0turn10search1turn10search3turn17view0  
- H3(실시간 안정화): VAD 게이팅+버퍼 드롭 정책이 장시간 실행에서 p95 지연을 안정화한다(큐 적체 방지). citeturn16search0turn7search3  

**기여(취업·논문화 관점에서 “평가 가능성” 확정 포인트)**  
- CREMA-D·RAVDESS의 라벨 체계를 **공통 6감정 + 각성(강도) 프록시**로 정규화하고, 교차 데이터셋 일반화까지 포함한 평가 프로토콜을 제시한다. citeturn12view1turn6view0turn5view2turn8view1  
- entity["company","Hailo","edge ai accelerator vendor"] Model Zoo가 제시하는 “Parse→Profile(FPS/latency/power)→Quantize(4/8/16-bit)→Compile(HEF)” 흐름을 그대로 사용해, 온디바이스 최적화·예측·실측을 연결한다. citeturn11view0turn11view1  
- Raspberry Pi 공식 설치 체인(`hailo-all`, `hailortcli`)과 카메라 스택(rpicam-apps/Hailo postprocess)을 기반으로 “재현 가능한 데모/벤치마크 앱”까지 포함한다. citeturn9view2turn9view4turn17view0  

## 데이터 설계
### 공개 데이터셋 목록과 메타
| 항목 | CREMA-D | RAVDESS |
|---|---|---|
| 규모 | 7,442 clips, 91 actors(20–74) | 7,356 files, 24 actors(성비 균형) citeturn12view1turn5view1 |
| 라벨 | 6 emotions + 4 levels(Low/Med/High/Unspecified), AV/V/A 조건으로 평가 | emotions(8) + intensity(normal/strong; neutral은 strong 없음), AV/VO/AO 제공 citeturn6view0turn12view1turn5view2 |
| 포맷 | Git LFS 기반: wav/mp3/flash video + csv/투표 결과 | audio-only(16bit, 48kHz wav), AV(720p H.264 + AAC 48kHz mp4), VO(no sound) citeturn12view1turn6view0 |
| 라이선스 | ODbL 1.0 + DbCL 1.0 | CC BY-NC-SA 4.0(+상업 라이선스 별도 구매) citeturn12view0turn6view0 |
| 주의 | repo zip은 LFS 링크만 포함(실데이터는 clone 필요) | “비상업·동일조건변경허락” 제약 → 포트폴리오 공개물에 원본 미디어 포함 금지 citeturn12view1turn6view0 |

### 대체계획(“두 데이터만 사용” 조건 내)
- CREMA-D 다운로드/Git LFS 장애 시: RAVDESS만으로 6감정(common-6)과 8감정(내부 평가)을 수행하되, 교차 일반화 실험(E3)에서 “CREMA-D→RAVDESS” 방향만 제외. citeturn12view1turn6view0  
- RAVDESS의 CC BY-NC-SA 제약으로 데모 배포 범위가 제한될 경우: 학습·평가는 동일하게 진행하되, 공개 저장소에는 **코드/가중치/정량 결과/전처리 스크립트/manifest만** 포함(원본·샘플 영상/음성 미포함). citeturn6view0  

### 윤리·동의(데모 입력)
- 공개 데이터 실험은 해당 라이선스 준수(재배포 금지 포함)로 종료. citeturn12view0turn6view0  
- 실시간 데모에서 사용자 얼굴·음성을 입력받을 경우 기본 설정은 “저장 안 함”; 저장 기능을 켤 때는 목적·보관기간·암호화·삭제·철회 포함 동의서를 제공(윤리 섹션 예시 참조). citeturn17view0  

### 포맷 표준화와 라벨 정합 규칙
**입력 윈도우(고정)**  
- 실시간·학습 공통: 2초 윈도우, hop 0.2초(실시간) / 학습용 hop 1.0초(옵션).  
- 비디오: 224×224 RGB, 16fps(1초=16프레임) 또는 2초=32프레임(온디바이스 지연 목표에 따라 둘 다 실험).  
- 오디오: 48kHz→16kHz mono resample → log-mel(80 bins, win 25ms, hop 10ms). RAVDESS의 오디오 포맷(48kHz wav/AAC)이 명시되어 있다. citeturn6view0  

**라벨 정합(평가 가능성 최우선)**  
- 공통 감정(common-6): `neutral, happy, sad, angry, fearful, disgust`  
- RAVDESS 전용 확장: `calm, surprised`는 “RAVDESS 전용 헤드”로만 평가(교차 데이터셋 실험에 혼입 금지). citeturn6view0turn5view1  
- 각성(인지상태 프록시) 라벨  
  - RAVDESS: intensity {normal, strong} → arousal_2cls(0/1)  
  - CREMA-D: level {low, medium, high} → arousal_3cls(0/1/2); 2클래스 실험에서는 high=1, low+medium=0(사전등록). citeturn6view0turn12view0turn8view1  

### 전처리·증강·저장·보안
- 오디오 증강(훈련만): SpecAugment(time/freq masking) + (선택)볼륨 스케일/배경잡음. citeturn7search14  
- 비디오 증강(훈련만): 약한 brightness/contrast jitter, 랜덤 가림(마스크·손), 랜덤 좌우반전(단, 발화-입모양 의존 시 반전 ablation 포함).  
- 저장 구조(재현성): `raw/ derived/ manifests/ splits/ models_fp32/ models_int8/ hef/ bench/` + sha256.  
- 보안/공개 정책: 공개 레포에는 원본 미디어 미포함(특히 RAVDESS NC-SA). citeturn6view0  

## 모델과 학습, 경량화
### 모델 아키텍처(온디바이스·컴파일 안정성 우선)
모델은 “듀얼 인코더 + CPU late fusion + 멀티태스크 헤드”를 기본으로 확정한다(두 인코더는 HEF로 Hailo-8L 실행, 융합·헤드는 CPU). Hailo Model Zoo가 HEF를 HailoRT로 로드해 사용하는 구조를 명시한다. citeturn11view0turn9view1  

| 블록 | 입력 | 권장 백본 | 출력 | 배치 |
|---|---|---|---|---|
| 비전 인코더 | (T,224,224,3) | MobileNetV2(폭 0.5~1.0) + temporal avg pooling | 128D | Hailo(HEF#1) citeturn10search0turn11view0 |
| 오디오 인코더 | (80,~200,1) | MobileNetV2(1ch) 또는 경량 CNN | 128D | Hailo(HEF#2) |
| 융합/헤드 | 256D(+품질플래그) | MLP(256→128→heads) | 감정(common-6) + 각성(2/3cls) | CPU |

### 멀티모달 융합 방식(실험 확정)
- Late fusion(기본): `z = concat(z_v, z_a, q)` → MLP.  
- Gated fusion(ablation): `q`에 VAD voiced ratio, 얼굴 검출 성공률, 밝기/SNR 등을 넣고 모달 가중을 동적으로 조절(실시간 안정성 향상 가설 H3 연결). citeturn7search3turn16search0  

### 손실함수·학습 스케줄·데이터 분할
- 감정: Weighted Cross-Entropy(불균형 보정)  
- 각성(2cls): CE 또는 BCE  
- 각성(3cls, CREMA-D): CE(+label smoothing)  
- 총손실: `L = L_emotion + λ L_arousal2 + μ L_arousal3`(λ,μ는 val-fold에서 고정)  

**분할과 교차검증(누수 방지 고정)**  
- Actor-independent split: 배우 ID를 그룹으로 GroupKFold(5-fold).  
- 교차 데이터셋 일반화(필수): Train(CREMA-D)→Test(RAVDESS common-6), Train(RAVDESS common-6)→Test(CREMA-D). RAVDESS는 24 배우의 lexical-matched 발화를 제공하고, 6/8 감정 및 강도 조건을 명시한다. citeturn5view1turn12view1turn5view2  

### 베이스라인 모델(논문 비교군 확정)
- B0: 단순 기준선(항상 최빈 감정 / 각성은 항상 normal)  
- B1: audio-only, B2: video-only  
- B3: late fusion(제안), B4: gated fusion(제안+)  
- B5: FP32(서버) vs CPU-only(Pi) vs INT8(Hailo-8L)  
- (옵션) 다른 NPU 비교: 보유 시 Edge TPU/Jetson 등과 FPS·전력 비교(논문에는 “optional hardware”로 명시)

### 경량화 기술(지식증류·프루닝·QAT/PTQ)
- 지식증류(KD): 앙상블/대형 모델 지식을 소형 모델로 이전해 배포 비용을 줄이는 대표 기법. citeturn10search1  
- 프루닝: Deep Compression이 제시한 pruning→(trained quantization) 흐름을 “구조적 채널 프루닝→미세조정→양자화”로 사용(하드웨어 친화). citeturn10search2  
- QAT/PTQ: integer-only 추론을 위한 QAT 절차와 정확도–지연 트레이드오프 개선이 MobileNet 계열에서도 보고된다. citeturn10search3  

## Hailo-8L 최적화와 구현
### 하드웨어·소프트웨어 스택(공식 문서 기반)
Raspberry Pi AI Kit 제품 브리프는 (a) AI 모듈이 Hailo-8L 기반 13 TOPS이고 (b) M.2 2242이며 (c) M.2 HAT+가 Pi 5 PCIe 2.0과 M.2 인터페이스를 중계하고 (d) 최신 OS에서 자동 감지되며 (e) rpicam-apps가 NPU로 호환 post-processing을 수행한다고 설명한다. citeturn9view2  

설치/검증은 Raspberry Pi 가이드의 절차를 그대로 고정한다: `sudo apt install hailo-all` → reboot → `hailortcli fw-control identify`. 또한 PCIe Gen 3.0 활성화는 “옵션이지만 강력 권장”으로 명시된다. citeturn17view0  

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["Raspberry Pi AI Kit M.2 HAT+ Hailo-8L installed on Raspberry Pi 5","Raspberry Pi M.2 HAT+ on Raspberry Pi 5 assembly","Hailo-8L M.2 module close-up"],"num_per_query":1}

### 지원 연산·정밀도(실행 관점의 정의)
- “지원 연산”은 DFC/Model Zoo의 **Parse→Quantize→Compile 성공 여부**로 판정한다(버전 의존). citeturn11view0turn11view1  
- 정밀도: Model Zoo 문서 흐름에서 Quantize 단계가 4/8/16-bit 정수 precision으로 변환함을 명시한다. citeturn11view0  

### ONNX→DFC→HEF 파이프라인(확정)
Hailo Model Zoo(공식 블로그)는 ONNX/TF 모델을 HAR로 변환(Parse)하고, 성능/전력/지연을 포함한 리포트를 생성(Profile), 정수화(Quantize) 후, HEF를 생성(Compile)하는 전체 흐름을 명시한다. citeturn11view0turn9view1  
또한 Hailo-8/8L은 Model Zoo v2.x + DFC v3.x 조합을 요구하므로 해당 브랜치 조합을 고정한다. citeturn11view1  

**INT8 캘리브레이션(실행 규칙)**  
- 최소 1024개 캘리브레이션 샘플을 준비하고, 메모리 제약 시 `npy_dir`로 배치 처리한다. citeturn11view3  

**HEF 아키텍처 검증(실패 예방 필수)**  
- `hailortcli parse-hef`로 HEF가 HAILO8L 대상 컴파일인지 확인하고(“HEF arch mismatch” 방지), 배포 파이프라인에 CI 체크로 포함한다. citeturn11view2  

### TAPPAS/GStreamer 통합(실시간 비전 파이프라인)
TAPPAS는 GStreamer 기반 플러그인/파이프라인으로 Hailo 디바이스를 이용한 지능형 비디오 처리 파이프라인을 구성하도록 설계되었고, User Guide는 GStreamer 개념과 플러그인 사용을 설명한다. citeturn16search7turn16search0  

### 멀티모달 실시간 전처리(동기화·버퍼링·VAD·face align·ASR)
- 동기화: monotonic timestamp를 기준으로 오디오/비디오를 정렬(Δt 기록).  
- 버퍼링: 비디오 프레임 큐(≤1s, 초과 시 drop), 오디오 ring buffer(≥10s)로 p95 지연 폭주를 방지(TAPPAS 파이프라인 구성 원칙과 정합). citeturn16search0turn16search7  
- VAD: WebRTC VAD로 음성 구간을 검출해 무음 구간 오디오 추론을 스킵(전력·지연 절감). citeturn7search3  
- face align: 오프라인(데이터셋)에서는 중앙 crop(배우 중심 프레이밍)→(옵션) face detect/align를 실험, 온라인(데모)에서는 “중앙 crop 모드”와 “검출+추적 모드”를 병렬 제공(지연/정확도 trade-off 리포트). citeturn5view2turn6view0  
- ASR 품질(선택 모듈, 데이터는 동일): RAVDESS는 2개 문장(lexically-matched)과 파일명 규칙(Statement ID)이 명시되어 GT 텍스트를 안정적으로 구성 가능. citeturn6view0turn5view1  

**WER/CER/RTF 정의(요구 지표 충족)**  
WER은 (삭제+삽입+치환)/참조 단어 수로 NIST가 정의한다. citeturn2search8  
CER은 (S+D+I)/N로 TorchMetrics 문서에 정의되어 있다. citeturn2search1  
RTF는 처리시간/오디오길이의 비로 Microsoft 문서가 정의한다. citeturn2search10  

### 시스템 아키텍처 다이어그램(확정)
```text
Camera(libcamera/rpicam-apps) -> GStreamer/TAPPAS -> (ROI: center/face) -> hailonet(video.hef) -> v_emb,t_v
Mic(ALSA) -> RingBuffer -> WebRTC VAD -> log-mel(STFT, CPU) -> HailoRT(audio.hef) -> a_emb,t_a
Sync(|t_v-t_a|) + QualityFlags -> CPU FusionHead -> emotion(common-6)+arousal(2/3cls)+smoothing
Overlay(UI) + Logs(JSONL/Parquet) + Bench(latency p50/p95, FPS, power, RSS)
```

### 코드 구조(예시)
```text
edge_affect_arousal/
  configs/
  data/
    crema_download.md
    ravdess_download.md
    manifest_build.py
    splits_groupkfold.py
  preprocess/
    video_decode.py
    face_roi.py
    audio_resample.py
    logmel.py
    vad_webrtc.py
    sync_buffers.py
  models/
    video_mnv2.py
    audio_mnv2.py
    fusion_head.py
  train/
    train_fp32.py
    distill.py
    prune.py
    qat.py
  hailo/
    export_onnx.py
    calib_dump_npy_dir.py
    compile_hef.sh
  runtime_pi/
    gst_video.py
    hailort_audio.py
    fusion_runtime.py
    metrics_runtime.py
  scripts/
    eval_offline.py
    bench_pi.py
```

## 실험, 평가, 재현성, 윤리, 일정
### 실험 계획표
**(C) 실험 계획표(표)**

| 실험 | 목적/가설 | 데이터 | 비교군 | 지표 | 통계검정 |
|---|---|---|---|---|---|
| E1 멀티모달 이득 | H1 | (각 데이터셋) 공통 6감정 | A-only vs V-only vs AV | Acc, macro-F1, OVR-AUC | AUC: DeLong, 분류: McNemar(이진화 보조), 95% CI: bootstrap citeturn13search0turn13search7turn13search12 |
| E2 각성 프록시 | H1/H2 | RAVDESS intensity, CREMA-D level | A/V/AV | arousal Acc/F1 + (서열)MAE | paired bootstrap, Wilcoxon(옵션) citeturn6view0turn12view0turn13search12 |
| E3 교차 일반화 | 도메인 강건성 | Train CREMA→Test RAVDESS(common-6) 등 | fusion vs single | macro-F1, AUC | paired bootstrap + (AUC) DeLong(OVR) citeturn13search0turn13search12 |
| E4 경량화 | H2 | 동일 split | FP32 vs PTQ-INT8 vs QAT-INT8 vs QAT+KD | ΔF1/ΔAUC/ΔMAE | paired bootstrap(효과크기 포함) citeturn10search1turn10search3turn13search12 |
| E5 온디바이스 벤치 | H2/H3 | Pi에서 “replay(라벨有)” + “live(라벨無)” | CPU-only vs INT8(HEF) | latency p50/p95, FPS, power, memory | 분포 비교 + Bland–Altman(예측↔실측) citeturn11view0turn13search1 |
| E6 ASR 품질(선택) | 요구 지표 충족 | CREMA/RAVDESS 고정 문장 | Whisper tiny/base 등 | WER, CER, RTF | 기술통계+CI citeturn2search8turn2search1turn2search10 |

### 구현 체크리스트
**(D) 단계별 구현 체크리스트**

1) **데이터 준비**  
- CREMA-D: Git LFS 설치 후 clone(영상/음성 포함), csv(투표/문장/메타) 무결성 확인. citeturn12view1  
- RAVDESS: Zenodo에서 AO/AV/VO zip 다운로드, 파일명 규칙 파싱(모달/감정/강도/문장/배우). citeturn6view0turn5view1  

2) **전처리 파이프라인 고정(재현성 핵심)**  
- `manifest.jsonl` 생성(clip_id, actor_id, emotion6, emotion8(opt), arousal2, arousal3(opt), paths, window offsets).  
- log-mel 파라미터 고정(80 mel, 25ms/10ms), 비디오 fps(16)·해상도(224) 고정. citeturn6view0  

3) **학습/평가(서버 FP32)**  
- GroupKFold(actor) 5-fold 학습 + 교차 데이터셋(E3) 평가 실행. citeturn5view2turn12view1  

4) **경량화**  
- KD(teacher→student) → 채널 프루닝 → PTQ → 목표 미달 시 QAT. citeturn10search1turn10search2turn10search3  

5) **HEF 생성(x86)**  
- Model Zoo v2.x + DFC v3.x 조합 고정. citeturn11view1  
- 캘리브레이션 ≥1024, 메모리 제약 시 `npy_dir` 사용. citeturn11view3  

6) **Pi 런타임 구성**  
- `sudo apt install hailo-all` → reboot → `hailortcli fw-control identify`. citeturn17view0  
- `rpicam-apps` 설치/검증(카메라 스택 + Hailo postprocess 포함). citeturn9view4  
- `hailortcli parse-hef`로 HAILO8L 대상 HEF 확인(배포 전 필수). citeturn11view2  

7) **벤치/로그**  
- latency(p50/p95), FPS, RSS, 전력(USB 전력계) 로그 자동 저장(JSONL/CSV).  
- DFC profile report의 (FPS/latency/power) 예측치와 실측치 비교(Bland–Altman). citeturn11view0turn13search1  

### 예상 결과와 해석 가이드
**(E) 예상 결과·해석 가이드(표·차트 제안)**

| 관측 | 해석 | 우선 조치 | 차트 제안 |
|---|---|---|---|
| AV가 단일모달 대비 F1/AUC↑ | 얼굴+발화 운율 결합 이득(H1 지지) | gated fusion/품질플래그 ablation | macro-F1 막대 + OVR ROC citeturn13search0 |
| 교차 데이터셋에서 성능↓ | acted 데이터·촬영·코덱 차이로 도메인 갭 | BN 통계 적응(가벼운 affine), 증강 강화 | Train→Test 성능 매트릭스 |
| PTQ에서 성능 급락 | 캘리브레이션 분포 불일치 | calib 재구성(≥1024), QAT+KD | ΔF1 vs FPS 산점도 citeturn11view3turn10search3 |
| p95 지연 폭주 | 큐 적체/동기화 실패 | drop 정책, VAD 스킵, hop 조정 | latency 히스토그램+타임라인 citeturn16search0turn7search3 |
| 프로파일 예측과 실측 power 차이 | “칩 추정” vs “시스템 전력” 범위 차이 | Bland–Altman로 합치도/한계 명시 | Bland–Altman plot citeturn13search1turn11view0 |

### 간단한 실험 스크립트 예시
**오프라인 평가 스켈레톤(교차검증 + 통계)**  
```python
# eval_offline.py (개념): fold별 예측 저장 → macro-F1/AUC/MAE → bootstrap CI
import json, numpy as np
from sklearn.metrics import f1_score, roc_auc_score, mean_absolute_error

def compute_metrics(y_true_cls, y_pred_proba, y_true_ar, y_pred_ar):
    f1 = f1_score(y_true_cls, y_pred_proba.argmax(1), average="macro")
    auc = roc_auc_score(y_true_cls, y_pred_proba, multi_class="ovr")
    mae = mean_absolute_error(y_true_ar, y_pred_ar)
    return {"macro_f1": f1, "ovr_auc": auc, "mae_arousal": mae}
```

**Pi 벤치 스크립트(지연 p50/p95, FPS, 메모리)**  
```python
# bench_pi.py (개념): replay 입력으로 end-to-end 지연 측정
import time, psutil
lat = []
p = psutil.Process()
t0 = time.time()

for _ in range(600):  # 약 1분
    s = time.perf_counter_ns()
    # y = pipeline_step()  # VAD/log-mel + HEF infer + fusion
    e = time.perf_counter_ns()
    lat.append((e - s) / 1e6)
    time.sleep(0.1)

lat.sort()
print({
  "p50_ms": lat[int(0.50*len(lat))],
  "p95_ms": lat[int(0.95*len(lat))],
  "fps_equiv": len(lat)/(time.time()-t0),
  "rss_mb": p.memory_info().rss/(1024**2)
})
```

### 재현성 패키지(공개 가능한 범위)
- 공개: 코드, 전처리 스크립트, 라벨 매핑 규칙, split 파일, 학습 설정(yaml), HEF 생성 스크립트(단, DFC/Model Zoo는 설치 조건이 있을 수 있어 버전·해시를 고정 문서화). citeturn11view1turn11view0  
- 비공개/미포함: RAVDESS 원본 미디어(라이선스 NC-SA), CREMA-D 원본 파일(사용 정책/용량), 대신 manifest만 포함. citeturn6view0turn12view0  

### 윤리·한계
- 데이터 한계: 두 데이터셋은 “배우 기반 표준화 연기(acted)” 성격이 강해 실제 시니어 환경으로의 일반화가 제한될 수 있다(논문 한계로 명확히 기술). citeturn5view2turn12view1  
- 사용 한계: “인지상태”는 alertness 기반 프록시이며 임상적 인지장애 진단이 아니다(데모 UI/논문에서 비진단 고지). citeturn8view1  
- 동의서 예시(데모 저장 기능 제공 시 핵심 문구)  
  - 수집: 얼굴 영상, 음성(선택), 추론 결과(감정/각성)  
  - 목적: 로컬 데모 표시 및 성능 벤치마크  
  - 보관: 기본 저장 없음, 저장 시 기간(예: 24h) 후 자동 삭제  
  - 철회: 즉시 삭제 버튼 제공, 로그도 함께 삭제  

### 일정·자원
| 단계 | 기간 | 산출물 | 리스크/대응 |
|---|---:|---|---|
| 데이터·전처리 고정 | 2주 | manifest/splits/전처리 스크립트 | CREMA Git LFS 이슈 → GitLab mirror/분할 다운로드 citeturn12view1turn7view0 |
| FP32 학습/교차검증 | 2주 | baseline 성능표+CI | 누수 방지(actor split) 자동 검사 |
| 경량화(QAT/KD/Prune) | 2–3주 | INT8 후보 2–3종 | PTQ 급락 → QAT+KD 전환 citeturn10search3turn10search1 |
| HEF 생성·검증(x86) | 1–2주 | HEF+profile report | calib≥1024, `npy_dir`로 RAM 절감 citeturn11view3turn11view0 |
| Pi 통합·벤치/데모 | 2주 | 실시간 앱+벤치 로그 | PCIe Gen3 설정, `hailo-all` 버전 고정 citeturn17view0 |
| 논문화/포트폴리오 | 1–2주 | 논문 초안+데모 영상 | RAVDESS NC-SA 준수(원본 미디어 미배포) citeturn6view0 |

## 참고문헌과 우선순위 소스
**(F) 우선순위 소스(원문 링크는 인용 클릭으로 제공)**  
- Raspberry Pi AI Kit 제품 브리프(13 TOPS, M.2 HAT+, PCIe 2.0 중계, 자동 감지, rpicam-apps 통합). citeturn9view2  
- Raspberry Pi AI 소프트웨어 문서(rpicam-apps 설치 및 Hailo postprocess 포함). citeturn9view4  
- Raspberry Pi AI Kit 설치 가이드(`hailo-all`, `hailortcli`, PCIe Gen3 권장). citeturn17view0  
- Hailo Model Zoo 전체 플로우 및 프로파일 리포트(FPS/latency/power, 4/8/16-bit quantize, HEF 생성). citeturn11view0  
- Hailo Model Zoo 호환성(Hailo-8/8L은 v2.x + DFC v3.x). citeturn11view1  
- TAPPAS User Guide(GStreamer 기반 파이프라인 개요). citeturn16search0  
- CREMA-D 공식 리포(규모/문장/감정/레벨/평가 방식, ODbL/DbCL 라이선스, Git LFS 안내). citeturn12view1turn12view0  
- entity["organization","Zenodo","research data repository"] RAVDESS 레코드(7356 files, 포맷/강도/파일명 규칙, CC BY-NC-SA 4.0). citeturn6view0turn5view1  
- RAVDESS 원 논문(구성/평정/신뢰도, 자유 다운로드 근거). citeturn5view2  
- 감정 차원모형(arousal=alertness 축 근거). citeturn8view1  
- MobileNetV2(경량 백본 근거). citeturn10search0  
- 지식증류/프루닝/QAT(경량화 이론 근거). citeturn10search1turn10search2turn10search3  
- WER/CER/RTF 정의(NIST/TorchMetrics/Microsoft). citeturn2search8turn2search1turn2search10  
- 통계 검정(DeLong, Bland–Altman). citeturn13search0turn13search1