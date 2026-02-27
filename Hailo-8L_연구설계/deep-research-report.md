# Raspberry Pi + Hailo-8 기반 시니어 마인드케어 멀티모달 연구 설계

## 요약
본 설계는 (1) **감정·인지 상태 실시간 모니터링(비전+음성)**, (2) **대화 기반 인지능력 평가(비전+텍스트)**를 entity["company","Hailo Technologies","edge ai accelerator vendor"] NPU(HEF) 기반 온디바이스로 구현하고, **멀티모달 융합이 단일모달 대비 유의미한 성능 향상**을 만든다는 가설을 검증한다. Hailo-8(26TOPS, typical 2.5W)citeturn10view0 및 Raspberry Pi 계열 AI 스택(Driver/HailoRT/TAPPAS, 버전 매칭 필수)citeturn0search8turn4view1을 전제로, 공공(특히 한국어·노인) 데이터 + IRB 기반 자체 수집으로 데이터/학습/양자화/실험 재현성을 논문 수준으로 고도화한다. fileciteturn0file0

## 상세 설계

**전제 확인(하드웨어/스택)**
- Hailo 계열 가속은 Raspberry Pi 5에서 PCIe로 연결되며, AI Kit는 Hailo-8L(13TOPS) 기반 M.2 2242 모듈을 M.2 HAT+에 실장하고, HAT+가 Pi 5의 PCIe 2.0과 브리지한다. citeturn11view0turn4view0  
- Hailo-8은 26TOPS, typical 2.5W를 명시하고(“best-in-class power efficiency”) 딥러닝 프레임워크 및 컴파일러/런타임/예제(TAPPAS/Model Zoo/DFC/HailoRT)를 포함한 SW Suite 구성을 제시한다. citeturn10view0  
- Raspberry Pi 문서 기준, **AI HAT+ (Hailo-8/8L)** 는 “vision AI models”, **AI HAT+ 2(Hailo-10H)** 는 LLM/VLM 등 GenAI를 추가로 지원한다. 즉 Hailo-8 환경에서 텍스트(LLM급)까지 전부 NPU로 처리한다는 가정은 위험하며, **텍스트는 CPU(또는 경량 모델) 중심**으로 설계하되, 필요 시 CLIP 등 VLM/텍스트 인코더 활용 가능성을 옵션으로 둔다. citeturn4view0turn6view0  

### 목적·가설·기여(두 프로젝트 공통 프레임)
아래 표는 “논문형”으로 기술 가능한 목적/가설/기여를 **정량 검증 가능 형태**로 쪼갠 것이다.

| 항목 | 프로젝트 1: 감정·인지 실시간(비전+음성) | 프로젝트 2: 대화 기반 인지평가(비전+텍스트) |
|---|---|---|
| 목적 | 노인 사용자 상호작용 중 **정서(감정/스트레스)** 및 **인지부하(혼란/주의저하)** 징후를 초단위로 추정해 “상태 변화”를 감지 | 대화/과제 기반 발화로 **인지기능 저하(정상/MCI/AD)** 및 **인지점수(MMSE 등) 회귀**를 추정(ADReSS 방식 재현+확장) |
| 핵심 가설 | (H1) 비전+음성 융합이 단일모달 대비 Macro-F1/AUC를 유의미하게 개선 (H2) INT8+증류로 정확도 저하 ≤X%에서 지연/전력 목표 달성 | (H1) 자동 전사(ASR) 기반 텍스트에서도 강건 특징(ASR conf, disfluency 등)을 쓰면 수기 전사 대비 성능 저하를 부분 상쇄 (H2) 얼굴/시선 기반 비전 특징이 언어 특징과 상보적 |
| 1차 기여(학술) | “엣지 NPU 실시간” 조건에서 멀티모달 상태 추정 파이프라인 및 지연/전력-정확도 트레이드오프 보고 | 한국어 노인 음성/대화 기반 인지저하 평가 벤치마크(공개 가능 범위) + ASR 오류(WER/CER)가 다운스트림 성능에 미치는 영향 정량화 |
| 2차 기여(공학) | Hailo TAPPAS(GStreamer) 기반 실시간 동기화/버퍼링 설계와 프로파일링 레시피 확립 citeturn0search1turn0search5turn0search13 | 온디바이스에서 비전은 Hailo, 텍스트는 CPU로 분리(또는 경량)한 “하이브리드 멀티모달” 아키텍처 제시(현실적 배치) citeturn4view0 |

### 데이터 설계
#### 공개 데이터셋 추천(한국어 우선)
- **인지기능 장애 진단 음성/대화(AI-Hub)**: 1,002명, 음성 5,769건(672h), 정상/MCI/AD 라벨, 스크립트 제공(타자 음성 제거 파일도 제공). 임상 진단 어노테이션 활용 가능. citeturn12view0  
- **자유대화 음성(노인남여, AI-Hub)**: 60세 이상 1,000명+, 3,000h+ 규모의 노인 자유대화 음성+전사, 노인 발화 특성(사투리/억양 등) 반영 필요성 명시. ASR 적응/언어 특징 사전학습에 적합. citeturn12view1  
- **감정 분류를 위한 대화 음성(AI-Hub)**: 16bit/48kHz wav, 7감정(5명 라벨), 메타에 상황/ASR 결과/감정 라벨/성별·나이 포함. 단, 의료 데이터는 안심존/IRB 요구 가능성을 안내. citeturn13view0  
- (영어·멀티모달 사전학습) **IEMOCAP**(오디오/비디오 기반 감정)citeturn1search0turn1search28, **RAVDESS**(검증된 멀티모달 감정 DB)citeturn1search5  
- (인지 평가 국제 벤치마크) **ADReSS/ADReSSo**: 자발화로 AD 분류 + MMSE 회귀 과제 정의(표준화 부족 문제 해결 목적)citeturn0search7turn2search1turn0search3, **DementiaBank Pitt(쿠키 도둑 과제 등)**citeturn2search0turn2search8  

#### 자체 수집(연구/논문 수준 고도화용, 권장)
- **수집 목표(현실적 최소)**: 정상(60+) 30명, MCI 30명, AD(경증 위주) 30명(총 90명) + 동일 인물 반복 세션(2~3회)로 종단성 신호 확보. (임상 라벨은 의료기관 협력 또는 AI-Hub 임상 데이터 활용으로 대체 가능)citeturn12view0turn2search5  
- **과제 설계(프로젝트 2 중심)**: Cookie Theft 유사 그림 설명, 범주 유창성(동물/과일), 이야기 회상, 자유대화(가족/일상), 지시 따르기(2~3단계), 정서 질문(가벼운 기분척도). ADReSS가 “그림 설명 기반 자발화” 및 MMSE 회귀를 과제로 삼는 점을 재현한다. citeturn2search1turn2search17  

#### 윤리·동의(필수 절차)
- 얼굴/음성은 **생체정보/민감정보로 취급될 소지가 큰 데이터**이므로, entity["organization","개인정보보호위원회","korea privacy regulator"] 가이드라인에 맞춘 최소수집·목적명시·보관기간·접근통제·가명처리 원칙을 설계에 반영한다. citeturn5search24turn5search20turn5search0  
- 의료·취약계층(노인, 인지장애) 연구는 **IRB 심의/동의 방식 검토**가 핵심이며, 전자 파일(녹음/영상) 기반 동의도 가능하나 기관위원회의 연구계획서 기반 심의가 필요하다는 Q&A가 존재한다. citeturn5search25  
- 공개 장소 촬영/이동형 장치(로봇 등) 사용 시 개인영상정보 보호 안내서 및 PIPA 관련 제한을 점검한다(연구는 가능한 한 통제된 공간에서 수행). citeturn5search11turn5search7  
- 산출물은 “진단”이 아니라 “스크리닝/모니터링 보조”로 표현하고, 상용·의료기기화 시 entity["organization","식품의약품안전처","korea mfds regulator"] 가이드라인/허가 프로세스 고려(본 연구는 비임상/비의료기기 범위 명시). citeturn5search18turn5search3  

#### 장비·센서·포맷(온디바이스 동기화 가능한 수준)
- 카메라: CSI(권장) 또는 UVC. Raspberry Pi 문서는 비전 AI 모델 실행에 “supported camera” 연결을 권장한다. citeturn4view0  
- 마이크: USB 오디오 인터페이스(권장) 또는 I2S.  
- 권장 포맷:
  - 비디오: 1280×720@30fps, YUYV/RGB(파이프라인 내부 RGB로 통일), 각 프레임에 `t_video`(monotonic ns) 부여
  - 오디오: PCM 16kHz, 16bit, mono, 20ms hop 스트림 버퍼, 각 청크에 `t_audio_start/end` 부여
  - 텍스트: ASR 출력 + 토큰 단위 타임스탬프(가능 시) + ASR confidence  
- 저장 포맷(재현성): `manifest.jsonl`(세션 단위 메타), `frames.mp4`(또는 개별 JPEG), `audio.wav`, `transcript.json`, `labels.csv`(표준 스키마).  

#### 라벨링 지침(최소 충돌, 최대 재현)
- 감정 라벨: (A) 범주(7감정 등), (B) 차원(valence/arousal) 중 1개 이상 선택. AI-Hub 대화 음성은 7감정/다중 라벨러 구조를 제공하므로, **다수결+불일치 샘플 별도 태그**를 채택한다. citeturn13view0  
- 인지 라벨: 정상/MCI/AD(분류) + 점수(회귀, 가능 시). AI-Hub 인지장애 음성/대화는 진단 라벨 분포를 제공한다. citeturn12view0  
- 품질 라벨: “마스크/역광/잡음/중첩발화/침묵비율” 등 파이프라인 실패 원인을 라벨로 남겨 ablation에 활용.

#### 전처리·증강
- 오디오: VAD로 발화 구간 분리(ADReSS 사이트가 VAD 기반 세그먼트 구성 정보를 공개함)citeturn0search15, noise/RIR 증강, 스피치 속도 perturbation(±5%), 볼륨 스케일.  
- 비디오: face detection→alignment(눈/코 기준), 랜덤 밝기/콘트라스트, motion blur(약하게), 랜덤 occlusion(마스크/손) 시뮬레이션.  
- 텍스트: ASR 오류를 반영한 “노이즈 주입”(동음이의 치환, 조사 삭제)로 강건 학습(단, 의미 보존 범위 제한).

### 모델·학습 설계
#### 공통 입력/출력 정의
- 프로젝트 1 출력(권장):  
  - 분류: `emotion_class ∈ {neutral, happy, sad, angry, fear, disgust, surprise}`(또는 4~7클래스)  
  - 회귀(선택): `valence ∈ [-1,1]`, `arousal ∈ [0,1]`  
  - 이벤트: “급격한 부정 정서 변화” 탐지(변화점 기반)
- 프로젝트 2 출력(권장):  
  - 분류: `cognitive_state ∈ {normal, MCI, AD}`(AI-Hub 라벨과 정합)citeturn12view0  
  - 회귀(선택): `MMSE_hat`(ADReSS가 MMSE 회귀 과제로 사용)citeturn2search17turn0search7  

#### 아키텍처(현실적 Hailo 배치)
**프로젝트 1(비전+음성)**
- 비전 가지: `FaceDet (Hailo Model Zoo 또는 경량 모델) → Crop/Align → MobileNetV3-Small + Temporal(1D-TCN/TSM) → embedding_v`  
- 음성 가지: `Log-Mel(CPU) → 2D-CNN(MobileNet/EfficientNet-lite) → embedding_a`  
- 융합: `concat(embedding_v, embedding_a, q_v, q_a) → MLP(2~3층) → logits`  
- 결측 모달리티 처리: ModDrop(학습 중 랜덤으로 한 모달 제거) + 품질 점수(q_v/q_a)로 게이팅.

**프로젝트 2(비전+텍스트)**
- 비전 가지: `Face/landmark/시선 proxy(가능 시) → MobileNetV3-Small → embedding_v`  
- 텍스트 가지(온디바이스 안정성 우선):  
  - 기본: `텍스트 특징(어휘 다양도, 평균 발화 길이, pause 비율, 반복/수정) + TF-IDF` + 경량 분류기(Linear/MLP)  
  - 고급(선택): DistilBERT/Korean ELECTRA 계열을 CPU INT8로(또는 TextCNN을 NPU 가능성 탐색)  
- 융합: “후기 융합(Decision-level)”을 기본으로 하되, 논문 기여를 위해 “중간 융합(embedding concat)”도 실험.

> 참고: ADReSS는 “음성만(자동 특징) vs 수기 전사 기반 언어 특징”에서 성능 격차(텍스트가 유리)를 보고한다. 자동 전사 기반 프로젝트 2는 **WER가 다운스트림 성능의 병목**이 될 수 있으므로, WER/CER를 직접 측정·보고하는 설계가 필수다. citeturn2search17turn2search7turn2search23  

#### 손실함수·학습 스케줄
- 분류: `CrossEntropy`(불균형 시 class weight 또는 Focal)  
- 회귀(MMSE/valence/arousal): `Huber` 또는 `MSE`  
- 멀티태스크: `L = L_cls + λ L_reg + μ L_consistency`(모달 dropout 시 예측 일관성 regularization)  
- 스케줄: AdamW + cosine decay, warmup 5%, early stopping(검증 AUC).  
- 데이터 분할: **participant-independent split**(세션이 아닌 사람 단위로 train/val/test). AD/노인 데이터는 누출에 취약하므로 필수. citeturn0search7turn2search20  
- 교차검증: 5-fold stratified group k-fold(그룹=participant).  

#### 베이스라인(논문 필수 구성)
- 프로젝트 1: (a) 비전-only, (b) 음성-only, (c) 단순 late fusion(평균), (d) 제안 게이팅/중간융합.  
- 프로젝트 2: (a) 텍스트-only(수기 전사/ASR 전사 비교), (b) 음성-only(ADReSS 스타일), (c) 비전-only, (d) 융합. ADReSS/ADReSSo의 과제 정의 및 베이스라인 비교를 명시한다. citeturn0search7turn0search3turn2search1  

### 경량화·온디바이스 최적화(Hailo-8)
#### Hailo-8 호환성 핵심(지원 연산·정밀도)
- Hailo는 “dataflow compiler” 기반으로 모델을 Hailo 실행 형식(HEF)으로 변환하고, 런타임(HailoRT)에서 실행한다. citeturn10view0turn3search22turn3search6  
- Hailo-8L 기준 지원 연산 예로 conv/depthwise conv, elementwise, pooling, InstanceNorm, 일부 activation(Mish/SiLU/GeLU preview 등)이 언급된다. 다만 STFT 같은 신호처리 연산은 일반적으로 NPU 연산으로 보기 어렵고, **오디오 전처리(STFT/Mel)는 CPU에서 수행**하는 것이 안전하다. citeturn3search0  

#### 컴파일·배포 파이프라인(권장 표준)
1) 학습(Pytorch) → 2) Export(ONNX 또는 TF) → 3) DFC에서 **INT8 양자화+컴파일(캘리브레이션 데이터 필요)** → 4) `.hef` 생성 → 5) Raspberry Pi에 `.hef` 배포 → 6) HailoRT/TAPPAS로 실시간 실행.  
- DFC는 “calibration data로 INT8 quantize & compile to HEF” 흐름을 명시(EdgeImpulse 가이드)한다. citeturn0search2  
- DFC 입력은 도구/버전에 따라 TF/TFLite/ONNX 등을 지원한다고 안내되며(예: RidgeRun 문서), 최종 호환은 DFC 리포트로 검증한다. citeturn0search10turn10view0  
- Raspberry Pi 문서는 패키지/드라이버 **버전 불일치 시 동작 불가**를 경고하며, 특정 버전 설치 예시(hailo-tappas-core/hailort/hailo-dkms/python3-hailort)를 제시한다. citeturn4view1  

#### INT8 양자화 이후 정확도 복원(필수 실험 축)
- **PTQ(후학습 양자화)**: 캘리브레이션 셋을 “실제 입력 분포(조명/잡음/억양/사투리)”로 구성. citeturn0search2turn12view1  
- **지식증류**: teacher(서버 FP32 큰 모델) → student(모바일/경량) → 동일 데이터에서 KL-divergence + CE 혼합.  
- **프루닝/구조 변경**: Depthwise/1×1 conv 비중 확대로 Hailo 친화적 설계(연산 지원/메모리 측면). (지원 연산 범위는 DFC에서 최종 확인)citeturn3search0  
- **모델 스크립트(컴파일러 튜닝)**: DFC 동작 커스터마이징을 “Model Scripts”로 수행 가능하다고 안내된다(최적화/컴파일 옵션 조정 근거). citeturn0search26  

#### 메모리·지연·전력 예측(실측 중심)
- Hailo-8은 typical 2.5W 및 “multi-stream/multi-model” 동시 처리를 강조한다. 실험은 “2-network 동시 실행(얼굴+감정/ASR 전처리)”을 포함해야 논문 가치가 커진다. citeturn10view0  
- 실측 항목: end-to-end latency(p50/p95), FPS, CPU 사용률, RSS 메모리, NPU utilization(가능 시), 전력(USB 전력계/INA219)  
- 예측/사전 점검: Hailo Model Zoo는 “full precision accuracy, quantized accuracy(Emulator), hardware accuracy 측정 및 HEF 생성” 기능을 제공한다고 설명한다(양자화 전후 정확도 스냅샷 확보). citeturn3search1  

### 구현(소프트웨어 스택·실시간 파이프라인)
#### 소프트웨어 스택(재현 가능한 고정 조합)
- OS: 64-bit Raspberry Pi OS 기반 + Hailo 스택(문서 가이드 준수)citeturn4view0turn4view1  
- 런타임: HailoRT(오픈소스 런타임, host CPU에서 동작)citeturn3search6turn10view0  
- 비전 파이프라인: TAPPAS(GStreamer 플러그인 기반) citeturn0search4turn0search5turn0search1  
- 개발 참고: Hailo RPi 예제 repo 및 “Hailo Apps Infra”가 최신 예제 인프라로 안내됨. citeturn6view0turn8search14  

#### 시스템 아키텍처 다이어그램(구현 관점)
```text
[프로젝트 1: 비전+음성]
Camera -> (decode/resize) -> FaceDet(HEF,NPU) -> FaceCrop -> EmoNet(HEF,NPU) -> e_v ->\
                                                                                      -> Fusion(MLP) -> State -> UI/Log
Mic    -> RingBuffer -> STFT/Mel(CPU) -> AudNet(HEF,NPU 또는 CPU) -> e_a -----------/

[프로젝트 2: 비전+텍스트]
Camera -> Face/HeadPose(HEF,NPU) -> e_v ------------------------------\
                                                                    -> Fusion -> {Normal/MCI/AD, MMSE_hat}
Mic -> RingBuffer -> Streaming ASR(CPU) -> Transcript+Conf -> TextFeat -> e_t --/
```
- **동기화 규칙**: `t_video` 기준으로 최근접 `t_audio` 윈도우를 매칭(±250ms).  
- **버퍼링 전략**: 영상은 leaky queue(지연 제한), 오디오는 fixed-size ring buffer(ASR/특징 추출 안정). TAPPAS는 GStreamer 파이프라인에서 queue/buffer 플러시 등 동작을 문서에서 다룸. citeturn0search1turn0search13  

#### 예시 코드 스니펫(HEF 로딩·추론 최소 예제)
아래는 Hailo 커뮤니티에 공유된 “최소 동작 예제” 형태(hef 로드→VDevice→InferVStreams)이며, 실제 파이프라인에 inference 호출부를 삽입하는 기준점으로 사용한다. citeturn9view0  
```python
# HailoRT Python minimal example (concept: load .hef and run infer)
import numpy as np
import hailo_platform as hpf

hef = hpf.HEF("my_model.hef")
with hpf.VDevice() as target:
    params = hpf.ConfigureParams.create_from_hef(hef, interface=hpf.HailoStreamInterface.PCIe)
    network_group = target.configure(hef, params)[0]
    ng_params = network_group.create_params()

    in_info = hef.get_input_vstream_infos()[0]
    out_info = hef.get_output_vstream_infos()[0]
    in_params = hpf.InputVStreamParams.make_from_network_group(network_group, quantized=False,
                                                              format_type=hpf.FormatType.FLOAT32)
    out_params = hpf.OutputVStreamParams.make_from_network_group(network_group, quantized=False,
                                                                 format_type=hpf.FormatType.FLOAT32)

    with network_group.activate(ng_params):
        with hpf.InferVStreams(network_group, in_params, out_params) as pipe:
            x = np.random.rand(*in_info.shape).astype(np.float32)
            y = pipe.infer({in_info.name: np.expand_dims(x, 0)})[out_info.name]
            print(y)
```

#### GStreamer 기반 비전 파이프라인(개념 스니펫)
TAPPAS는 hailonet/hailofilter 등 플러그인 기반으로 비전 파이프라인을 구성하며, 예제에서 queue 설정과 파이프라인 템플릿을 제공한다. citeturn0search1turn0search21turn0search5  
```bash
# concept only: source -> decode -> preprocess -> hailonet(.hef) -> postproc(hailofilter) -> display
gst-launch-1.0 v4l2src ! videoconvert ! queue ! hailonet hef-path=model.hef ! hailofilter so-path=post.so ! autovideosink
```

## 실험 계획표

| 실험 | 목적 | 데이터 | 모델/조건 | 지표(정확도 외) | 통계/비교 |
|---|---|---|---|---|---|
| E1 | 멀티모달 이득 검증(프로젝트1) | AI-Hub 감정 음성 + 자체 얼굴영상(또는 IEMOCAP 사전학습) citeturn13view0turn1search0 | 비전-only / 음성-only / fusion | Acc, Macro-F1, AUC, latency(p95), FPS, power | McNemar(분류), DeLong(AUC), 부트스트랩 CI |
| E2 | INT8 영향/복원 | 캘리브레이션 셋(현실 분포) | FP32(서버) vs PTQ INT8 vs 증류+INT8 | ΔF1, ΔAUC, HEF size, FPS | 동일 fold에서 paired test |
| E3 | ASR 오류 전파(프로젝트2) | AI-Hub 인지장애 음성/대화(라벨 제공)citeturn12view0 | 수기 전사(가능 시) vs ASR 전사 | WER/CERciteturn2search7turn2search23, downstream AUC/F1 | WER 구간별 성능 곡선 |
| E4 | 인지 점수 회귀 | ADReSS 방식(가능 데이터)citeturn0search7turn2search1 | regression head 추가 | MAE/RMSE, R^2, calibration | Bland–Altman(가능 시) |
| E5 | 실시간성(온디바이스) | Pi 실측 | 단일모델 vs 2모델 동시 | RTF(ASR)citeturn3search35turn3search15, end-to-end latency, FPS, memory | CPU-only vs NPU 가속 |
| E6 | Ablation | 전부 | 모달 dropout, 게이팅 제거, 텍스트 특징 제거 | Δ성능, 실패 케이스 분해 | 오류 유형 분석 |

## 구현 체크리스트

| 단계 | 완료 기준 | 핵심 리스크/대응 |
|---|---|---|
| 환경 구축 | Pi OS 64-bit + Hailo 패키지 버전 고정(apt hold)citeturn4view1 | 버전 불일치 → 문서대로 hold, 동일 버전 HEF 사용 |
| NPU 검증 | 제공 예제(detection_simple 등) 실행, HEF 로딩 성공citeturn6view0turn9view0 | 카메라/PCIe 설정(Gen3 등) 누락 |
| 데이터 파이프라인 | manifest/저장 스키마 생성, 익명화/접근통제 적용citeturn5search24turn5search0 | 개인정보 유출 → 암호화/권한 최소화 |
| 학습 코드 | GroupKFold, 재현 seed, 실험 로그(MLflow 등) | 누출 → participant 단위 split |
| DFC 컴파일 | calibration 셋으로 INT8 컴파일, HEF 생성citeturn0search2turn10view0 | 미지원 op → CPU 전처리로 우회(특히 STFT)citeturn3search0 |
| 실시간 동기화 | 오디오 ring buffer + 영상 leaky queue 안정화citeturn0search1turn0search13 | 지연 누적 → 드롭 정책/레이트 리밋 |
| 성능 계측 | FPS/latency/power/RTF(WER 포함) 자동 수집citeturn2search7turn3search15 | 측정 편향 → 동일 조건 반복/CI |

## 예상 결과·해석 가이드

| 관측 결과 | 해석(원인 후보) | 대응/추가 실험 |
|---|---|---|
| Fusion이 unimodal보다 AUC↑ | 모달 상보성 성립(가설 지지) | 모달 품질(q_v/q_a)별 조건부 성능 보고 |
| INT8에서 성능 급락 | 캘리브레이션 분포 불일치 or 민감 레이어 | calib 재구성, 증류, 마지막 레이어 FP 유지(가능 범위) |
| ASR WER↑일 때 인지 분류 급락 | 텍스트 특징이 ASR 오류에 취약 | ASR conf 기반 샘플 가중, 음성 프로소디 특징 병합(프로젝트2에 음성-only 베이스라인 포함)citeturn2search17turn2search23 |
| 실시간 FPS는 충분하나 latency p95↑ | 버퍼/큐 적체 또는 CPU 전처리 병목 | 오디오 전처리 최적화, queue leaky, 멀티프로세싱 |
| 오류가 특정 얼굴(마스크/조명)에서 집중 | 데이터 편향/도메인 갭 | occlusion 증강, 테스트 환경 다양화, 실패 케이스 라벨링 |

**차트 제안(논문 그림 세트)**
- (C1) **Accuracy/AUC vs Latency(p95)** 산점도(모델/양자화/동시실행 조건별)  
- (C2) **WER 구간별 다운스트림 AUC 곡선**(ASR 오류 전파 시각화; WER은 NIST 정의 기반)citeturn2search7turn2search23  
- (C3) **Ablation 막대그래프**(모달 제거/게이팅 제거/증류 유무)  
- (C4) **전력-성능 곡선**(Hailo-8 typical power는 제품 브리프 근거로 참고 축 설정)citeturn10view0  

## 참고문헌·우선순위 소스

**한국어/공식 1순위(데이터·법/윤리)**
- AI-Hub: 인지기능 장애 진단 음성/대화(정상/MCI/AD, 5,769건, 672h) citeturn12view0  
- AI-Hub: 자유대화 음성(노인남여, 3,000h+ 전사 포함) citeturn12view1  
- AI-Hub: 감정 분류 대화 음성(7감정, 5명 라벨, wav 포맷/메타 포함) citeturn13view0  
- 개인정보 보호법(목적/보호 원칙 근거) citeturn5search0  
- 개인정보보호위원회: 생체정보 보호 가이드라인/안내서(얼굴 등 생체정보 처리 원칙) citeturn5search24turn5search20  
- 식약처: 디지털의료기기/소프트웨어 가이드라인 공지(상용화 시 규제 고려) citeturn5search18turn5search3  

**플랫폼/툴체인 1순위(공식 문서·코드)**
- Raspberry Pi “AI software”: AI HAT+/AI HAT+2 범위(vision vs LLM)와 설치 전제 citeturn4view0turn4view1  
- Hailo-8 제품 브리프(26TOPS, typical 2.5W, SW suite 구성) citeturn10view0  
- TAPPAS: GStreamer 기반 아키텍처/유저가이드(파이프라인·queue 등) citeturn0search5turn0search1  
- HailoRT(오픈소스 런타임) citeturn3search6  
- DFC 파이프라인/INT8 캘리브레이션 근거(커뮤니티 가이드) citeturn0search2  
- Hailo Python 최소 예제(HEF→InferVStreams) citeturn9view0  

**학술 1순위(인지 평가/지표)**
- ADReSS(AD 분류 + MMSE 회귀, 표준화 목적) citeturn0search7turn2search17  
- ADReSSo(원시 음성 기반, 자동 전사 허용/권장) citeturn0search3turn0search11  
- DementiaBank Pitt(쿠키도둑 과제 등) citeturn2search0turn2search8  
- WER 정의(entity["organization","National Institute of Standards and Technology","us standards institute"] 평가계획 문서) citeturn2search7  
- RTF 정의(ASR 실시간성 지표; RTF≤1 조건) citeturn3search35turn3search15