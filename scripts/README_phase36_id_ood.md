# Phase-36 ID/OOD 3-Mode 실행 가이드

## 목적
- ID/OOD 트랙에서 `audio / video / fusion` 3모드를 동일 프로토콜로 학습/평가한다.
- Hailo 비교를 위해 test 입력셋(`index.csv + npy`)을 FP32 checkpoint 정규화 기준으로 함께 생성한다.

## 1) 분할 생성
```bash
.venv/bin/python scripts/build_phase36_splits.py
```

산출:
- `derived/splits/phase36_id_ood/*_train.txt`
- `derived/splits/phase36_id_ood/*_val.txt`
- `derived/splits/phase36_id_ood/*_test.txt`
- `derived/splits/phase36_id_ood/summary.json`

트랙:
- `id_all`
- `id_crema`
- `id_ravdess`
- `ood_c2r` (CREMA -> RAVDESS)
- `ood_r2c` (RAVDESS -> CREMA)

## 2) 단일 실험 실행 (FP32 train + test eval set + FP32 test 평가)
```bash
bash scripts/run_phase36_fp32_single.sh \
  --track ood_c2r \
  --mode fusion \
  --run-name phase36_ood_c2r_fusion_v5_lnfree
```

산출(예시):
- `derived/results/phase36/phase36_ood_c2r_fusion_v5_lnfree/summary.json`
- `derived/results/phase36/phase36_ood_c2r_fusion_v5_lnfree/fp32_test_eval.json`
- `derived/results/phase36/phase36_ood_c2r_fusion_v5_lnfree/phase36_run_meta.json`
- `derived/hailo/eval/phase36/phase36_ood_c2r_fusion_v5_lnfree_test/index.csv`

## 3) 매트릭스 실행
```bash
bash scripts/run_phase36_fp32_matrix.sh
```

부분 실행:
```bash
bash scripts/run_phase36_fp32_matrix.sh \
  --tracks id_crema,ood_c2r \
  --modes fusion \
  --epochs 40 \
  --device auto

# 이미 완료된 run 자동 건너뛰기(기본값)
bash scripts/run_phase36_fp32_matrix.sh --skip-done 1

# 실패 run이 있어도 나머지 계속
bash scripts/run_phase36_fp32_matrix.sh --continue-on-error 1
```

## 4) Hailo 비교 연결
단일 run의 test index를 그대로 사용해 Pi 배치 추론:
```bash
PI_PASSWORD='***' bash scripts/run_e5_infer_pi_batch.sh \
  --host wormhole@129.254.232.91 \
  --hef-local derived/hailo/build/<network>/<network>.hef \
  --name <run_name>_hailo \
  --calib-dir derived/hailo/eval/phase36/<run_name>_test \
  --index-csv derived/hailo/eval/phase36/<run_name>_test/index.csv \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --max-samples 0 \
  --use-password
```

## 5) Phase36 best-per-track Hailo 준비(자동)
Phase36 test 결과에서 트랙별 best run을 자동 선택하여
ONNX/export + train-calib 생성 + (옵션) HEF compile을 수행한다.

```bash
bash scripts/run_phase36_hailo_best5.sh
```

부분 실행(compile 없이 준비만):
```bash
bash scripts/run_phase36_hailo_best5.sh --compile 0
```

출력:
- `derived/hailo/phase36_best5_build_meta.csv`
- `derived/hailo/phase36_best5_candidates.csv`

## 6) Phase36 best-per-track Pi 추론 + FP32/Hailo 비교
best5 메타를 읽어 run별 test index로 Pi 배치 추론을 수행하고
FP32 대비 성능 차이 리포트를 만든다.

```bash
PI_PASSWORD='***' bash scripts/run_phase36_hailo_eval_best5.sh \
  --host wormhole@129.254.232.91 \
  --use-password
```

출력:
- `derived/reports/phase36_fp32_vs_hailo_best5.csv`
- `derived/reports/phase36_fp32_vs_hailo_best5.md`
