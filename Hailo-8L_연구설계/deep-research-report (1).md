# Hailo-8L 온디바이스 멀티모달 시니어 마인드케어 연구 설계서

## 요약
본 설계서는 Raspberry Pi M.2 HAT+ + Hailo-8L 환경에서 **오픈 데이터만**으로 “논문 수준의 평가 가능성”을 최우선으로 두 프로젝트를 최종 확정한다. 프로젝트1은 공개 오디오-비디오 데이터 **AVCAffe**의 정답(정서·인지부하 라벨)을 활용해 **감정(Valence/Arousal)+인지부하(mental demand 등)** 를 **멀티태스크**로 예측하고, Hailo-8L에는 **비전/스펙트로그램 CNN**을 올리며 융합·후처리는 CPU에서 수행한다. AVCAffe는 108시간·58,112 클립·106명·오디오/비디오·정서/인지부하 정답을 제공해 정량평가가 명확하다. citeturn8view2turn17view1 프로젝트2는 AI-Hub “노인 정신건강 영상 데이터”(온라인 안심존)의 **실제 임상 인지검사 상황**에서 수집된 mp4/이미지/음성 기반 라벨(정상/비정상인지)로 **대화 기반 인지능력 평가(비전+텍스트)** 를 구성하되, 텍스트는 공개 ASR(Whisper)로 생성해 재현을 보장한다. citeturn11view0turn9view0turn15search0

## 프로젝트1 감정과 인지부하 실시간 모니터링
### 목적·가설·기여
**목적**: 카메라+마이크 기반으로 (i) 감정(Valence/Arousal), (ii) 인지부하(mental demand 등)를 **동시에** 추정하고, Hailo-8L에서 **지연·전력 예산을 만족**하는 실시간 파이프라인을 제공한다. AVCAffe는 정답 라벨이 포함된 오디오-비디오 데이터셋이므로 “정확도/지연/전력” 3축 평가가 가능하다. citeturn8view2turn17view1  
**가설 H1**: 멀티모달(비전+오디오) 융합이 단일모달 대비 AUC/F1(분류) 또는 MAE(회귀)를 유의미하게 개선한다(동일 split에서 비교). citeturn8view2  
**가설 H2**: Hailo DFC 기반 INT8(PTQ/QAT) + 지식증류를 적용하면, FP32 서버 대비 성능 저하를 제한하면서(예: ΔAUC≤0.02 또는 ΔMAE≤5%) 온디바이스 지연/전력을 달성한다(Profiler 예측 + 실측 검증). citeturn17view0turn14search4  
**기여(논문 포인트)**: (1) AVCAffe에서 정서·인지부하를 **동일 모델**로 다루는 멀티태스크+멀티모달 설계, (2) Hailo-8L에서 “2-네트워크 이상”을 운영하는 스트리밍 최적화(버퍼링/동기화) 레시피, (3) **정확도·RTF·지연·전력**을 통합 보고하는 재현 패키지. citeturn0search1turn14search4  

### 데이터
**최종 확정(평가 가능성 최상) 데이터셋: AVCAffe**
- **라벨/정답**: arousal, valence, mental demand, temporal demand, effort, 등(자기보고 기반) citeturn8view2turn17view1  
- **규모/형식**: 106명, 108시간, 58,112개 클립. 원본 mp4(640×360, 2.5–10분) + 6초 내외 세그먼트 avi + face crops 제공. citeturn8view2turn17view1  
- **접근/라이선스**: 비상업 연구 목적의 이용 조건 및 Dataverse 기반 접근 요청 절차 존재(재현성에 포함). citeturn8view2turn17view1  

**보조(도메인/감정 다양성 보강, 대체 가능) 데이터셋**
- CREMA-D(오디오-비디오 감정): 7,442 클립, 91명(20–74세). “시니어(60+) 포함 가능” 측면에서 보조 사전학습에 유리. citeturn12search0turn12search20  
- RAVDESS(오디오 또는 오디오-비디오 감정): 24명 배우 기반 감정 음성/비디오 자료(감정 분류 베이스라인 용). citeturn12search5turn12search1  

**전처리·증강(재현 가능 규격)**
- 오디오: 16kHz mono, 25ms window/10ms hop log-mel(예: 80 bins) + SpecAugment(시간/주파수 마스킹), time-stretch(±5%), SNR 기반 노이즈 믹싱(실내 잡음).  
- 비디오: 얼굴 검출→정렬→크롭(예: 224×224), 프레임 16fps로 리샘플, 랜덤 수평반전/밝기·대비 jitter(훈련만).  
- 동기화: AVCAffe “clip 단위(6초)”를 기본 샘플로 하여 오디오·비디오를 동일 구간으로 묶음(데이터 누수 방지). citeturn17view1  

**저장·보안**
- 원본 데이터는 read-only 스토리지에 저장, 파생 특징/체크포인트는 해시 기반 버전관리(DVC 또는 단순 manifest.json), 접근키/토큰은 분리 보관. AVCAffe는 접근 통제 절차가 있으므로 토큰/접근 로그를 별도 관리. citeturn17view1turn8view2  

### 모델·학습
**아키텍처(온디바이스 친화 + Hailo 호환 우선)**
- 오디오 인코더: log-mel을 “단일 채널 이미지”로 보고 MobileNetV3/EfficientNet-lite 계열 CNN → 128D 임베딩  
- 비전 인코더: face crop 기반 MobileNetV3/EfficientNet-lite → 128D 임베딩  
- 융합: late fusion(오디오/비전 임베딩 concat) + MLP(2–3 layer) → 멀티태스크 헤드  
  - (감정) valence/arousal: (a) 회귀(MAE) 또는 (b) 3클래스(저/중/고) 분류(AUC/F1)  
  - (인지부하) mental demand/effort 등: (a) 3클래스 또는 (b) 회귀  
이 구성은 Hailo-8L이 “비전 AI 모델” 실행에 최적화되고(텍스트·대형 생성형은 AI HAT+ 2에서 강조) 다중 네트워크를 파이프라인에서 병렬 추론할 수 있다는 전제와 잘 맞는다. citeturn17view2turn14search4  

**손실함수**
- 분류: Cross-Entropy(+ class weight 또는 focal)  
- 회귀: MAE(요구 지표 포함) + Huber(안정성)  
- 멀티태스크 합성: λ-weighted sum(λ는 검증셋 기반으로 고정)  

**데이터 분할·교차검증**
- 원칙: “participant-independent” split(동일 참가자의 다른 클립이 train/test에 동시에 등장하지 않도록)  
- AVCAffe가 제공하는 split 정보를 1차로 사용하고, k-fold(예: 5-fold)로 반복해 CI 산출(부트스트랩/fold 평균). citeturn17view1  

**베이스라인(필수 비교군)**
- 단일모달: (A) 오디오-only CNN, (B) 비전-only CNN  
- 고전 ML: MFCC 통계 + SVM/LogReg, 얼굴 랜드마크 통계 + XGBoost  
- 서버 FP32 기준: 동일 Backbone을 FP32로 평가한 상한선(온디바이스 대비 성능 하락량 보고)  

### Hailo-8L 최적화
**Hailo 호환성(핵심 제약 정리)**
- Hailo-8/8L은 Hailo Model Zoo **v2.x + DFC v3.x** 조합이 전제(버전 불일치가 재현성/컴파일 실패의 핵심 리스크). citeturn12search2  
- DFC 플로우: Parse→Profile(FPS/지연/전력 예측 포함)→Quantize(정수 비트 정밀도 4/8/16)→Compile(HEF 생성). citeturn17view0  
- 런타임 통합: GStreamer 플러그인의 hailonet은 “파이프라인 내 다중 사용”으로 병렬 네트워크 추론을 지원. citeturn14search4turn14search16  
- 지원 연산은 CNN 중심(Conv/DWConv/Pooling/엘리먼트와이즈 등)으로 알려져 있으며, 복잡한 STFT·동적 제어흐름은 NPU 밖(CPU)에서 처리하는 설계가 안전하다. citeturn1search1  

**ONNX 변환**
- 학습 프레임워크(Pytorch)→ONNX export(정적 입력)→DFC(또는 Model Zoo 빌드 플로우)로 HEF 생성. DFC 변환은 리소스 소모가 커 x86 Linux 환경에서 수행하는 것이 일반적이다. citeturn17view0turn1search5  

**INT8 PTQ/QAT**
- 1차: PTQ(대표 데이터셋으로 calibration)  
- 성능 저하가 크면: QAT(가짜양자화 포함 재학습) + 지식증류(Teacher=FP32, Student=INT8 대상)  
- “평가 가능성”을 위해 PTQ→QAT 적용 전후를 모두 리포팅(Ablation).  

**지연·전력 예측 방법(논문에 넣을 수 있게)**
- 예측: DFC/Profile 리포트에서 FPS/latency/power를 1차 확보하고(모델별 HTML breakdown 포함), citeturn17view0  
- 실측: Pi에서 end-to-end latency(p50/p95), FPS, 메모리 RSS, 전력(USB 전력계 또는 INA219) 측정  
- 정합: (예측값, 실측값) 쌍을 3개 이상 모델로 모아 “예측-실측 상관/편향”을 보고(결론: host PCIe/버퍼링 영향 설명). PCIe Gen 3.0 활성화는 성능 최적화에 중요하다. citeturn17view2turn18view0  

### 구현
**소프트웨어 스택(확정)**
- OS: 64-bit Raspberry Pi OS(Trixie) 기준(문서 기준선). citeturn17view2  
- 설치: `sudo apt install hailo-all` 로 드라이버/펌웨어, HailoRT, TAPPAS core, rpicam-apps postprocess 설치. citeturn18view0  
- 검증: `hailortcli fw-control identify`로 장치 인식 확인. citeturn18view0  

**실시간 파이프라인 아키텍처(설계 다이어그램)**

```text
[Camera(libcamera/rpicam)] --> [GStreamer] --> [Face Crop/Preproc] --> [hailonet(Video CNN, HEF)]
                                                         |--> (video embedding)
[Mic(ALSA)] --> [Audio RingBuffer] --> [log-mel CPU] --> [HailoRT(Python, Audio CNN HEF)]
                                                         |--> (audio embedding)
(video emb, audio emb, timestamps) --> [Fusion on CPU] --> [State Output + Logging]
                                          |--> (UI/REST/MQTT optional)
```

GStreamer 기반 비전 추론은 TAPPAS의 “비디오 프레임 추론 템플릿”과 다중 파이프라인 예제(멀티 네트워크/멀티 스트림) 개념을 직접 차용한다. citeturn0search1turn14search16  

**예제 코드 구조(권장)**

```text
repo/
  configs/
    model_video.yaml
    model_audio.yaml
    fusion.yaml
  src/
    ingest/
      camera_gst.py
      mic_alsa.py
    preprocess/
      face_align.py
      logmel.py
    infer/
      hailo_video.py
      hailo_audio.py
    fusion/
      late_fusion.py
    eval/
      metrics.py
      benchmark_pi.py
  scripts/
    export_onnx.py
    build_hef.sh
    run_demo.sh
```

**GStreamer/TAPPAS 파이프라인 스니펫(개념)**
```bash
gst-launch-1.0 \
  libcamerasrc ! videoconvert ! videoscale ! \
  hailonet hef-path=video_model.hef ! \
  fakesink
```

hailonet은 “구성된 네트워크에 따라 프레임 추론”을 수행하며, 동일 파이프라인에서 여러 번 사용해 병렬 네트워크 구성이 가능하다는 점이 멀티모달(최소 2망) 구성의 근거다. citeturn14search4turn14search16  

### 실험·평가
**정확도 지표**
- 분류: Acc, Macro-F1, ROC-AUC(OVR)  
- 회귀: MAE(필수), 추가로 RMSE(선택)  
**실시간 지표**
- RTF: 처리시간/오디오길이(실시간성 판단). citeturn13search9turn13search2  
- FPS, end-to-end latency(p50/p95), 메모리(RSS), 전력(W)  

**통계 검정**
- 동일 테스트셋 2모델 비교: McNemar(이진 분류) citeturn13search16turn13search3  
- AUC 비교: DeLong(상관 ROC) citeturn14search0turn14search6  
- CI: participant 단위 bootstrap(권장) citeturn14search11  

**비교군**
- 서버 FP32(최대 성능 기준), Pi CPU-only(온디바이스 기준), Hailo-8L(가속 기준)  
- Ablation: (오디오-only, 비전-only, late fusion, 증류/프루닝/INT8 전후)

## 프로젝트2 대화 기반 인지능력 평가
### 목적·가설·기여
**목적**: “검사자-피검자 질의응답(인지검사)” 상황의 영상·음성으로부터 **정상 vs 비정상인지(경도인지+치매)** 를 판별하고, 결과를 **설명 가능한 형태(질문별 근거, 신호 품질, 불확실성)** 로 제시한다. 데이터는 의료기관에서 인지기능검사 중 얻어진 mp4/jpg/png와 라벨(json)로 구성되며, 평가 라벨이 명확하다. citeturn9view0turn9view1  
**가설 H1**: 텍스트(ASR 기반 응답 내용) 단독보다 **비전(얼굴표현/행동 단서)** 을 결합하면 AUC/F1이 개선된다. 비전은 음성 잡음·ASR 오류가 큰 구간에서 보완 역할을 한다. citeturn9view1turn15news46  
**가설 H2**: 질문별(26개 응답 구간) 분해 후 “질문-아키텍처”를 적용하면(예: 시간지남·기억·주의 질문에서 텍스트 가중↑, 얼굴표현 이상구간에서 비전 가중↑) 단일 통합보다 성능/설명가능성이 함께 개선된다. citeturn9view1  
**기여**: (1) AI-Hub 안심존 의료 데이터로부터 “비전+텍스트(ASR 생성)” 파이프라인을 **재현 가능한 절차**로 정식화, (2) Hailo-8L 기반 **다중 입력 신호**의 온디바이스 지연/전력 측정, (3) 의료/생체정보 프라이버시 요구사항을 반영한 배포 설계. citeturn11view0turn16search6  

### 데이터
**최종 확정(도메인 적합 + 라벨 명확) 데이터셋: 노인 정신건강 영상 데이터(AI-Hub, 온라인 안심존)**
- **형식/출처/규모**: jpg/png/mp4, 2023년 구축, 총 43,894 규모(표기), 의료기관 다기관 수집. citeturn9view0turn11view0  
- **분포**: 원시 기준 1,203명(여 63.59%, 남 36.41%), 50대~80대 이상 4클래스, 정상 43.06% vs 비정상인지 56.94%. citeturn9view0  
- **라벨 구조(핵심)**: `diagnosis`가 인지기능상태 0(정상) / 1(비정상인지=경도인지+치매)로 제공. citeturn9view1turn9view0  
- **세부 데이터 구성(연구 설계에 유리한 점)**  
  - 음성: 질문별 응답 구간 start/end 라벨링(1140명×26개=29,640 구간) + MFCC 이미지 생성/분류 힌트 citeturn9view1turn5view3  
  - 얼굴표현: “정상/이상 얼굴표현” 구간 라벨 + 14개 랜드마크 자동 추출(pkl) citeturn9view1turn9view0  
  - 오각형(도형 그리기): png 이미지(5장/피험자) + 정상/비정상 라벨 citeturn9view1turn5view3  

**텍스트 생성(오픈소스, 재현성 확보)**
- 본 데이터는 텍스트 전사가 직접 명시되지 않으므로, 음성 구간을 Whisper로 전사해 “텍스트 모달리티”를 구성한다. Whisper는 코드·가중치가 MIT 라이선스로 공개되어 재현이 가능하다. citeturn15search0turn15search17  
- 단, Whisper는 의료/고위험 도메인에서 “허구 문장 삽입(환각)” 위험이 보고되어 있으므로, 연구 설계상 **품질 게이팅/휴먼-인-더-루프**를 포함한다. citeturn15news46turn13search7  

**ASR 품질 평가용 보조 데이터(라벨/전사 존재)**
- 인지기능 장애 진단 음성/대화: 음성 5,769건, 스크립트 5,769건, 672시간, 진단별 정상/MCI/AD 분포 제공(ASR WER/CER 계산에 적합). citeturn10view0turn11view1  

**윤리·접근(안심존)**
- 두 AI-Hub 의료 데이터는 “온라인 안심존”으로 제공되며, 이용 절차에 IRB 문서/보안서약서 등 제출이 포함된다. citeturn11view0turn16search6turn16search2  
- 연구 설계서에 “안심존 내 학습/평가 + (필요 시) 모델 반출 신청”을 공식 단계로 포함(리스크 관리). citeturn11view0turn16search6  

### 모델·학습
**입력·출력 정의(평가 가능하게 고정)**
- 입력(비전): 얼굴표현 구간 비디오 클립(또는 랜드마크 시퀀스), 오각형 png(선택)  
- 입력(텍스트): 질문별 응답 음성→ASR 텍스트, 질문 ID, 음성 품질 지표(SNR, 무음비율)  
- 출력: 정상(0) / 비정상인지(1) 이진 분류 + 불확실성(예: temperature-scaled probability)

**멀티모달 융합(온디바이스 현실 반영)**
- (Hailo-8L) 비전망 1: 얼굴표현 클립→(프레임 샘플링)→CNN(or 2D+Temporal pooling)→128D  
- (Hailo-8L) 비전망 2(선택): 오각형 png→CNN→64D  
- (CPU) 텍스트망:  
  - Baseline A: TF-IDF + Linear SVM/LogReg(빠르고 재현성 높음)  
  - Baseline B: 소형 Transformer(8-bit 양자화)  
- 융합: late fusion(임베딩 concat) + logistic head  
- **질문별 가중 게이팅**: 26개 질문에서 “언어 의존 질문/표현 의존 질문” 가중을 다르게 두는 ablation을 설계(설명가능성 강화). citeturn9view1  

**손실·스케줄**
- 손실: BCEWithLogits(+ class weight), calibration을 위해 ECE(선택)  
- 스케줄: cosine decay + early stopping, seed 3회 반복  
- 분할: subject-independent stratified split(기관/성별/연령대 분포를 가능한 유지). citeturn9view0  

**베이스라인**
- (영상-only) 얼굴표현 라벨 구간 기반 CNN/랜드마크 LSTM  
- (텍스트-only) TF-IDF + Linear  
- (음성-only) 제공 MFCC 이미지(DenseNet121 등 공개된 구축 측 모델 설정을 1개 baseline으로 재현) citeturn9view0turn5view3  

### Hailo-8L 최적화
- 비전망(얼굴/오각형)은 Hailo-8L에 적합한 CNN으로 고정하고, DFC의 Profile→Quantize→Compile을 수행한다. citeturn17view0turn12search2  
- 다중 네트워크 운영: 동일 파이프라인에서 hailonet 다중 사용(얼굴/오각형) 또는 프로세스 분리 후 HailoRT로 스케줄링. hailonet의 다중 사용은 공식 컴포넌트 설명에 근거한다. citeturn14search4turn14search16  
- 텍스트는 Hailo-8L 대상이 아니므로(비전 중심), CPU 최적화(ONNX Runtime, int8 CPU quant)로 목표 지연을 맞춘다. citeturn17view2  

### 구현
**안심존 제약을 반영한 구현 전략**
- 1) 안심존 내: 전처리/학습/평가(정확도·WER/CER) 수행  
- 2) 반출물: (i) 파생 특징(개인식별 제거 수준), (ii) 모델 가중치/ONNX/HEF, (iii) 통계 리포트(집계값)만 반출 신청(가능 여부는 안심존 정책에 따름) citeturn16search6turn11view0  
- 3) Pi 데모: 반출된 HEF/모델로 라이브 입력에 대해 실시간 동작 및 지연·전력 측정

**ASR/텍스트 파이프라인(품질 게이팅 포함)**
- Whisper 전사 후, (a) “무음 비율”, (b) 평균 logprob(가능 시), (c) 길이/반복 토큰을 이용해 “전사 신뢰도 점수”를 만들고, 낮으면 텍스트 모달리티 가중을 낮추는 gating을 적용한다(Whisper 환각 리스크 완화). citeturn15news46turn15search0  

### 실험·평가
**지표(필수 포함)**
- 분류: Acc/F1/AUC(+ sensitivity/specificity)  
- ASR:  
  - WER = (D+I+S)/N (NIST 정의) citeturn13search7  
  - CER = (S+D+I)/N (torchmetrics 정의) citeturn13search11  
- 실시간: RTF(ASR) 및 end-to-end latency/FPS/power/memory. citeturn13search9turn13search2  

**통계검정**
- AUC 차이: DeLong(상관 ROC) citeturn14search0turn14search6  
- 이진 분류 성능 비교: McNemar citeturn13search16turn13search3  
- CI: participant bootstrap citeturn14search11  

## 실험 계획표
| 단계 | 목표 산출물 | 데이터/환경 | 핵심 비교(평가 가능성) | 주요 지표 |
|---|---|---|---|---|
| 데이터 접근 확정 | 안심존 승인·다운로드/분석환경 | AI-Hub 안심존(의료) | 승인/접근 리스크 식별 | 승인 리드타임, 재현 로그 |
| Project1 베이스라인 | AVCAffe 단일모달/융합 baseline | AVCAffe | 오디오-only vs 비전-only vs 융합 | Acc/F1/AUC, MAE |
| Project1 온디바이스 | HEF 2개(오디오/비전) + Pi 실시간 | Pi OS + hailo-all | CPU-only vs Hailo | latency/FPS/power/memory |
| Project2 텍스트 생성·검증 | Whisper 전사 + WER/CER 리포트 | 인지기능 음성/대화(스크립트) | Whisper vs 규칙기반(또는 다른 ASR) | WER/CER, RTF |
| Project2 멀티모달 분류 | 비전+텍스트 융합 모델 | 노인 정신건강 영상 | 텍스트-only vs 비전-only vs 융합 | AUC/F1 + DeLong/McNemar |
| 통합 데모 | Pi 데모 앱/대시보드 | Pi + 라이브 입력 | 지연·전력·안정성 | p95 latency, uptime |

## 구현 체크리스트
1) **Pi 환경 고정**: 64-bit Raspberry Pi OS(Trixie) 기준 확인 → AI Kit(M.2 HAT+) 사용 시 PCIe Gen3 활성화. citeturn17view2turn18view0  
2) **필수 패키지 설치**: `hailo-all` 설치(HailoRT/TAPPAS/rpicam-apps postprocess 포함) → `hailortcli fw-control identify`로 장치 확인. citeturn18view0  
3) **모델 빌드 체인 고정**: Hailo-8/8L 대상은 Model Zoo v2.x + DFC v3.x 조합 준수(버전 매트릭스 문서화). citeturn12search2  
4) **Project1 HEF 생성**: (오디오CNN, 비전CNN) ONNX export → DFC Profile/Quantize/Compile → HEF 2개 생성. citeturn17view0turn1search5  
5) **Project1 실시간 파이프라인**: 비전은 GStreamer(hailonet), 오디오는 HailoRT Python 또는 별도 hailonet 입력 파이프 구성 → 타임스탬프 동기화/링버퍼/드롭 정책 구현. citeturn14search4turn0search1  
6) **Project2 안심존 분석 파이프라인**: 질문별 구간 로딩→Whisper 전사→텍스트 전처리→비전 특징 추출→융합 학습/평가. citeturn9view1turn15search0  
7) **지표/통계 자동화**: WER/CER/RTF 계산(정의 고정) + AUC DeLong + McNemar + bootstrap CI 스크립트화. citeturn13search7turn13search11turn14search0turn13search16  
8) **재현 패키지**: 컨테이너(학습), 하드웨어 명세, seed, split manifest, HEF 해시, Pi 벤치마크 로그를 묶어 릴리스.

## 예상 결과·해석 가이드
| 관찰 결과 | 해석(가능 원인) | 다음 액션(연구적으로 유효) | 제안 차트 |
|---|---|---|---|
| 융합이 단일모달보다 AUC↑ | 상보적 정보 결합 성공(가설 H1 지지) | 질문별 게이팅/불확실성 보정 추가 | ROC(모델별), ablation bar |
| INT8 후 성능↓ 크다 | quant noise/대표셋 부족 | calibration set 확대, QAT/증류 | FP32 vs INT8 scatter |
| Profiler 예측과 Pi 실측 괴리 | PCIe/버퍼링/파이프 병목 | Gen3 확인, GStreamer queue 튜닝 | 예측-실측 Bland–Altman citeturn17view0turn18view0 |
| Whisper WER/CER 높음 | 도메인/잡음/발화 특성 | 텍스트 게이팅, 질문별 키워드 기반 피처 보강 | WER/CER vs AUC 상관 citeturn13search7turn13search11turn15news46 |
| Project2에서 비전이 특히 도움 | ASR 오류 구간 보완 | “ASR 신뢰도-가중” 함수 학습 | AUC by ASR-quality bin |

## 참고문헌·우선순위 소스
- Raspberry Pi AI 소프트웨어/전제조건(64-bit OS, AI Kit, PCIe Gen3, 비전 AI, AI HAT+2 GenAI 구분). citeturn17view2  
- Raspberry Pi AI Kit 설치(hailo-all 구성요소, hailortcli 검증, Gen3 권장). citeturn18view0  
- Hailo Model Zoo/DFC 파이프라인(Parse/Profile/Quantize/Compile, Profile에 FPS·지연·전력 포함). citeturn17view0  
- Hailo 런타임 통합(hailonet 다중 사용, pyHailoRT 등). citeturn14search4turn14search16  
- Hailo-8/8L 호환 버전(모델주/DFC 브랜치 제약). citeturn12search2  
- AVCAffe 공식 레포/통계(오디오·비디오, 108h, 58k clips, 파일 포맷/구조). citeturn8view2turn17view1  
- AI-Hub “노인 정신건강 영상 데이터” 메타·분포·라벨 구조·질문별 구간 라벨링. citeturn9view0turn9view1turn11view0  
- AI-Hub “인지기능 장애 진단 음성/대화” 규모·진단 라벨·스크립트 존재(ASR 평가 기반). citeturn10view0turn11view1  
- 안심존 이용 절차/서류(IRB 등). citeturn16search6turn16search2  
- WER 정의(NIST). citeturn13search7  
- CER 정의(torchmetrics). citeturn13search11  
- RTF 정의(embedded speech 성능 평가). citeturn13search9  
- AUC 통계검정(DeLong). citeturn14search0turn14search6  
- McNemar(분류기 비교). citeturn13search16turn13search3  
- 생체정보·개인정보(한국 개인정보보호법/생체정보 보호 가이드라인). citeturn16search4turn16search1  
- Whisper 공개/라이선스(MIT) 및 고위험 도메인 환각 리스크 보도(윤리·한계 근거). citeturn15search0turn15news46