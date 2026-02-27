# Hailo E4 준비 상태 (2026-02-25)

## 1) 오늘 완료한 항목

- ONNX export 파이프라인 구현
  - `hailo/export_onnx.py`
- Calibration npy dump 파이프라인 구현
  - `hailo/calib_dump_npy_dir.py`
- 로컬 E4 준비 러너 추가
  - `scripts/run_e4_hailo_prep.sh`
- Hailo compile 템플릿 추가(기본 dry-run)
  - `hailo/compile_hef.sh`
- Pi 원격 점검 스크립트 추가
  - `scripts/run_e5_pi_check.sh`
- 원격 x86 컴파일 자동화 스크립트 추가
  - `scripts/run_e4_compile_remote.sh`
- 로컬 x86(WSL/PC) 컴파일 환경 세팅 스크립트 추가
  - `scripts/setup_hailo_compile_env.sh`
- 로컬 x86(WSL/PC) 컴파일 실행 스크립트 추가
  - `scripts/run_e4_compile_local.sh`
- Hailo SW Suite에서 wheel 추출/정리 스크립트 추가
  - `scripts/prepare_hailo_wheels.sh`
- 로컬 원클릭 세팅 스크립트 추가
  - `scripts/run_e4_local_setup.sh`
- 다운로드 대기 포함 E2E 로컬 실행 스크립트 추가
  - `scripts/run_e4_end_to_end_local.sh`
- Pi 벤치 자동화 스크립트 추가
  - `scripts/run_e5_benchmark_pi.sh`

## 2) 실제 생성 산출물

- ONNX:
  - `derived/hailo/onnx/fp32_multitask_phase35_v8_hubert_gated_wide_tune4_fold0_full.onnx`
  - `derived/hailo/onnx/fp32_multitask_phase35_v8_hubert_gated_wide_tune4_fold0_audio_encoder.onnx`
  - `derived/hailo/onnx/fp32_multitask_phase35_v8_hubert_gated_wide_tune4_fold0_video_encoder.onnx`
  - `derived/hailo/onnx/fp32_multitask_phase35_v8_hubert_gated_wide_tune4_fold0_export_meta.json`
- Calibration:
  - `derived/hailo/calib/fold0_train_1024/audio/*.npy` (1024개)
  - `derived/hailo/calib/fold0_train_1024/video/*.npy` (1024개)
  - `derived/hailo/calib/fold0_train_1024/pairs/*.npz` (1024개)
  - `derived/hailo/calib/fold0_train_1024/index.csv`
  - `derived/hailo/calib/fold0_train_1024/summary.json`

## 3) 현재 환경 제약

- 이 WSL 환경에는 Hailo 도구가 없음:
  - `hailortcli` 없음
  - `hailomz` 없음
- 따라서 HEF compile(E4 후반)와 Pi 실측(E5)은 Pi/DFC 환경에서 실행 필요.

## 4) Pi 원격 점검 결과 (2026-02-25)

- 접속 호스트: `wormhole@129.254.232.91`
- 확인 결과:
  - `hailortcli`: OK
  - 디바이스 인식: `Device Architecture: HAILO8L`
  - `rpicam-hello`: OK
  - `hailomz`: missing (Pi에서 HEF 컴파일은 불가)
- 샘플 HEF 벤치 정상 수행:
  - HEF: `/usr/share/hailo-models/resnet_v1_50_h8l.hef`
  - 결과 CSV(로컬 복사): `derived/hailo/pi_bench/resnet50_h8l_bench.csv`
  - 주요 수치:
    - streaming FPS: `47.3894`
    - hw_only FPS: `47.1888`
    - hw latency: `15.4423 ms`
- 자동화 스크립트로 재실행 검증:
  - 스크립트: `scripts/run_e5_benchmark_pi.sh`
  - 결과:
    - `derived/hailo/pi_bench/resnet50_h8l_ref2_parse.txt`
    - `derived/hailo/pi_bench/resnet50_h8l_ref2_bench.csv`
  - 주요 수치:
    - streaming FPS: `47.3894`
    - hw_only FPS: `47.3889`
    - hw latency: `15.4419 ms`
- 추가 재검증(동일 조건):
  - `derived/hailo/pi_bench/resnet50_h8l_ref3_parse.txt`
  - `derived/hailo/pi_bench/resnet50_h8l_ref3_bench.csv`
  - 주요 수치:
    - streaming FPS: `47.3894`
    - hw_only FPS: `47.3896`
    - hw latency: `15.4416 ms`

## 4.1) 컴파일/추론 역할 분리 최종 확인

- `scripts/run_e4_compile_remote.sh`에 원격 preflight를 추가함.
- 실제 `wormhole@129.254.232.91` preflight 결과:
  - `ARCH=aarch64`
  - `MODEL=Raspberry Pi 5 Model B Rev 1.0`
  - `HAS_HAILOMZ=0`, `HAS_HAILO=1`, `HAS_HAILORTCLI=1`
- 결론:
  - 해당 Pi는 Hailo 런타임/추론은 가능.
  - HEF 컴파일은 불가(`hailomz` 없음).
  - HEF 컴파일은 x86 DFC/Model Zoo 호스트에서 수행 후 Pi로 배포해야 함.

## 4.2) 로컬 WSL(x86_64) 컴파일 가능성 확인

- 아키텍처: `x86_64` (컴파일 호스트 조건 자체는 충족)
- 현재 상태: `hailomz` 미설치
- 공개 경로 점검:
  - `pip index`에서 `hailo-model-zoo`, `hailo-sdk-client` 배포본 미확인
  - Ubuntu 기본 apt repo에도 DFC 패키지 없음
- 결론:
  - 이 WSL에서 컴파일은 가능하나, Hailo Developer Zone에서 받은 DFC/SDK wheel이 필요함.
  - 해당 wheel 경로를 `scripts/setup_hailo_compile_env.sh --wheel-dir ...`로 주면 로컬 컴파일 경로로 전환 가능.
  - 원클릭 경로:
    - `scripts/run_e4_local_setup.sh --suite-dir <...>`
  - E2E 경로(다운로드 대기 포함):
    - `scripts/run_e4_end_to_end_local.sh --wait-seconds 3600 --network-name fp32_v8_fold0 --pi-host <...>`

## 5) 바로 다음 실행 순서

1. Pi 연결 확인
```bash
bash scripts/run_e5_pi_check.sh --host <user@pi_host>
```

2. DFC/Model Zoo 환경에서 HEF compile
```bash
bash hailo/compile_hef.sh \
  --onnx derived/hailo/onnx/fp32_multitask_phase35_v8_hubert_gated_wide_tune4_fold0_full.onnx \
  --calib-dir derived/hailo/calib/fold0_train_1024 \
  --out-dir derived/hailo/build/fp32_v8_fold0 \
  --network-name fp32_v8_fold0 \
  --execute
```

3. 생성 HEF 아키텍처 확인
```bash
hailortcli parse-hef derived/hailo/build/fp32_v8_fold0/fp32_v8_fold0.hef
```

4. Pi에서 E5 실측(지연/FPS/전력/RSS) 실행 및 로그 저장
