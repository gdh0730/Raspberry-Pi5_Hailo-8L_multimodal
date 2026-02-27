# Hailo E4 Starter (Current Repository)

This folder contains the first executable bridge from current FP32 results to E4 (PTQ/QAT preparation).

## Exact role split (important)

- Current WSL/PC: prepare ONNX + calibration data.
- x86 host with Hailo DFC/SDK (`hailo_sdk_client`): compile ONNX -> HEF.
- Raspberry Pi + Hailo-8L: run inference/benchmark using HEF.

In short: **Pi is inference runtime, not compile host** (unless you separately install a full compile toolchain there).

## What is implemented now

- `hailo/export_onnx.py`
  - exports ONNX from FP32 checkpoint:
    - full multitask model (`*_full.onnx`)
    - audio encoder (`*_audio_encoder.onnx`)
    - video encoder (`*_video_encoder.onnx`)
- `hailo/calib_dump_npy_dir.py`
  - builds representative calibration samples from cached features (`audio/*.npy`, `video/*.npy`, `pairs/*.npz`)
- `hailo/compile_hef.sh`
  - compile entrypoint for custom ONNX + npy calibration (dry-run by default)
- `hailo/compile_custom_onnx_sdk.py`
  - SDK-based compiler path used by `compile_hef.sh`
- `scripts/run_e4_compile_remote.sh`
  - uploads ONNX/calib to remote x86 compile host and runs `compile_hef.sh --execute`
- `scripts/run_e5_benchmark_pi.sh`
  - runs `hailo parse-hef` + `hailo benchmark` on Raspberry Pi and pulls CSV/logs back
- `scripts/run_e5_infer_pi.sh`
  - runs single-sample real inference on Pi using `hailo_platform` and pulls JSON/log back
- `scripts/run_e5_infer_pi_batch.sh`
  - runs multi-sample Pi inference with progress output and auto-evaluation (`predictions.csv`, `summary.json`)
- `scripts/run_e5_compare_models.sh`
  - runs multiple HEF candidates with identical evaluation setup and aggregates comparison table/report

## Quick start (local, current environment)

```bash
bash scripts/run_e4_hailo_prep.sh
```

Outputs:

- `derived/hailo/onnx/*`
- `derived/hailo/calib/fold0_train_1024/*`

## Compile on this WSL/PC directly (if you install DFC/Model Zoo here)

Fast path (one command):

```bash
bash scripts/run_e4_local_setup.sh \
  --suite-dir <extracted_hailo_ai_sw_suite_dir>
```

End-to-end (wait for download -> setup -> compile -> optional Pi benchmark):

```bash
bash scripts/run_e4_end_to_end_local.sh \
  --wait-seconds 3600 \
  --network-name fp32_v8_fold0 \
  --pi-host wormhole@129.254.232.91
```

Manual path:

1. Prepare wheel directory from Hailo SW Suite (x86 host):

```bash
bash scripts/prepare_hailo_wheels.sh \
  --suite-dir <extracted_hailo_ai_sw_suite_dir> \
  --out-dir third_party/hailo_wheels
```

2. Setup local compile env (x86 only):

```bash
bash scripts/setup_hailo_compile_env.sh \
  --wheel-dir third_party/hailo_wheels \
  --model-zoo-ref v2.17
```

3. Run local compile:

```bash
bash scripts/run_e4_compile_local.sh \
  --network-name fp32_v8_fold0 \
  --venv .venv-hailo \
  --optimization-level 0 \
  --compression-level 0 \
  --max-calib 1024
```

If SDK import fails, the setup script will stop with exact missing requirement.

## Then on Hailo DFC/SDK host (x86)

```bash
bash hailo/compile_hef.sh \
  --onnx derived/hailo/onnx/fp32_multitask_phase35_v8_hubert_gated_wide_tune4_fold0_full.onnx \
  --calib-dir derived/hailo/calib/fold0_train_1024 \
  --out-dir derived/hailo/build/fp32_v8_fold0 \
  --network-name fp32_v8_fold0 \
  --execute
```

Or run via automation wrapper:

```bash
bash scripts/run_e4_compile_remote.sh \
  --host <user@x86-host> \
  --network-name fp32_v8_fold0
```

`run_e4_compile_remote.sh` now performs a preflight check and stops immediately if python cannot import `hailo_sdk_client` on the remote host.

## Run benchmark on Pi (E5 smoke / runtime check)

Use built-in reference HEF:

```bash
bash scripts/run_e5_benchmark_pi.sh \
  --host wormhole@129.254.232.91 \
  --hef /usr/share/hailo-models/resnet_v1_50_h8l.hef \
  --name resnet50_h8l_ref \
  --time 10
```

Use your compiled HEF:

```bash
bash scripts/run_e5_benchmark_pi.sh \
  --host wormhole@129.254.232.91 \
  --hef-local derived/hailo/build/fp32_v8_fold0/fp32_v8_fold0.hef \
  --name fp32_v8_fold0 \
  --time 10
```

## Run real inference on Pi (single sample)

```bash
PI_PASSWORD='***' bash scripts/run_e5_infer_pi.sh \
  --host wormhole@129.254.232.91 \
  --hef-local derived/hailo/build/fp32_v8_fold0/fp32_v8_fold0.hef \
  --audio-npy derived/hailo/calib/fold0_train_1024/audio/00000.npy \
  --video-npy derived/hailo/calib/fold0_train_1024/video/00000.npy \
  --name fp32_v8_fold0_sample0 \
  --use-password
```

## Run real inference on Pi (batch + metrics)

```bash
PI_PASSWORD='***' bash scripts/run_e5_infer_pi_batch.sh \
  --host wormhole@129.254.232.91 \
  --hef-local derived/hailo/build/fp32_v8_fold0/fp32_v8_fold0.hef \
  --normalize-checkpoint derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune4/checkpoints/best_fold_0.pt \
  --max-samples 100 \
  --use-password
```

Outputs:

- `derived/hailo/pi_infer_batch/<name>/progress.csv`
- `derived/hailo/pi_infer_batch/<name>/predictions.csv`
- `derived/hailo/pi_infer_batch/<name>/summary.json`
- `derived/hailo/pi_infer_batch/<name>/summary_bootstrap.json`

## Compare multiple HEFs on Pi (same split/protocol)

1. Create candidate csv:

```csv
name,hef_local
fp32_v8_fold0,derived/hailo/build/fp32_v8_fold0/fp32_v8_fold0.hef
sdk_probe4,derived/hailo/build/sdk_probe4/fp32_v8_fold0.hef
```

2. Run:

```bash
PI_PASSWORD='***' bash scripts/run_e5_compare_models.sh \
  --host wormhole@129.254.232.91 \
  --candidates-csv derived/hailo/compare_candidates.csv \
  --normalize-checkpoint derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune4/checkpoints/best_fold_0.pt \
  --max-samples 200 \
  --use-password
```

Outputs:

- `derived/hailo/pi_compare/compare_metrics.csv`
- `derived/hailo/pi_compare/compare_report.md`

## High-accuracy path used in this repo (LayerNorm-free training)

If Hailo accuracy is much lower than WSL FP32 and compile logs show layer-norm/conv shape issues,
use a Hailo-friendly model variant trained without fusion-head LayerNorm.

1. Train fold0 model (GPU) without head LayerNorm:

```bash
.venv/bin/python scripts/train_fp32_multitask.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --fold-dir derived/splits/groupkfold5_all \
  --out-dir derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune4_hailo_lnfree_fold0 \
  --cache-dir derived/features/cache_v5_hubert \
  --mode fusion --fusion-type gated --modality-dropout-p 0.05 \
  --num-folds 1 --epochs 40 --batch-size 128 \
  --lr 0.00045 --weight-decay 0.0001 \
  --hidden-dim 512 --emb-dim 256 --dropout 0.2 \
  --emotion-loss ce --label-smoothing 0.08 --weighted-sampler \
  --seed 1337 --device cuda --no-head-layernorm
```

2. Export ONNX + normalized calibration:

```bash
.venv/bin/python hailo/export_onnx.py \
  --run-dir derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune4_hailo_lnfree_fold0 \
  --fold 0 \
  --onnx-dir derived/hailo/onnx \
  --name fp32_multitask_phase35_v8_hubert_gated_wide_tune4_hailo_lnfree_fold0

.venv/bin/python hailo/calib_dump_npy_dir.py \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --cache-dir derived/features/cache_v5_hubert \
  --split-list derived/splits/groupkfold5_all/fold_0_train.txt \
  --out-dir derived/hailo/calib/fold0_train_1024_lnfree \
  --max-samples 1024 --seed 1337 \
  --normalize-checkpoint derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune4_hailo_lnfree_fold0/checkpoints/best_fold_0.pt
```

3. Compile + Pi evaluate:

```bash
bash scripts/run_e4_compile_local.sh \
  --venv .venv-hailo \
  --network-name fp32_v8_lnfree_fold0 \
  --onnx derived/hailo/onnx/fp32_multitask_phase35_v8_hubert_gated_wide_tune4_hailo_lnfree_fold0_full.onnx \
  --calib-dir derived/hailo/calib/fold0_train_1024_lnfree \
  --out-dir derived/hailo/build/fp32_v8_lnfree_fold0 \
  --optimization-level 0 --compression-level 0 --max-calib 1024

PI_PASSWORD='***' bash scripts/run_e5_infer_pi_batch.sh \
  --host wormhole@129.254.232.91 \
  --hef-local derived/hailo/build/fp32_v8_lnfree_fold0/fp32_v8_lnfree_fold0.hef \
  --name fp32_v8_lnfree_fold0 \
  --calib-dir derived/hailo/calib/fold0_train_1024_lnfree \
  --index-csv derived/hailo/calib/fold0_train_1024_lnfree/index.csv \
  --manifest derived/manifests/manifest_multimodal_common6_av.jsonl \
  --max-samples 1024 \
  --use-password
```

## Notes

- `compile_hef.sh` uses `hailo_sdk_client` directly for parse/optimize/compile.
- For this model, `LayerNormalization` is converted to `Identity` during compile-prep by default (`--strip-layernorm`) to avoid current DFC layer-norm decomposition failures on this topology.
- This FP32 model was trained with fold-wise feature normalization (`audio_mu/sd`, `video_mu/sd`). Hailo calibration/inference inputs must use the same normalization for fair performance comparison.
- Current local WSL environment does not include Hailo toolchain.
