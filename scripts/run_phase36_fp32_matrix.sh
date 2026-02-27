#!/usr/bin/env bash
set -euo pipefail

# Run Phase-36 FP32 matrix:
# tracks = id_all, id_crema, id_ravdess, ood_c2r, ood_r2c
# modes  = audio, video, fusion
#
# Example:
#   bash scripts/run_phase36_fp32_matrix.sh
#   bash scripts/run_phase36_fp32_matrix.sh --tracks id_crema,ood_c2r --modes fusion

TRACKS="id_all,id_crema,id_ravdess,ood_c2r,ood_r2c"
MODES="audio,video,fusion"
EPOCHS=40
DEVICE="auto"
OUT_ROOT="derived/results/phase36"
SKIP_DONE=1
CONTINUE_ON_ERROR=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tracks) TRACKS="$2"; shift 2 ;;
    --modes) MODES="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --skip-done) SKIP_DONE="$2"; shift 2 ;;
    --continue-on-error) CONTINUE_ON_ERROR="$2"; shift 2 ;;
    *) echo "[ERROR] Unknown arg: $1"; exit 1 ;;
  esac
done

IFS=',' read -r -a TRACK_ARR <<< "$TRACKS"
IFS=',' read -r -a MODE_ARR <<< "$MODES"

for track in "${TRACK_ARR[@]}"; do
  track="$(echo "$track" | xargs)"
  [[ -z "$track" ]] && continue
  for mode in "${MODE_ARR[@]}"; do
    mode="$(echo "$mode" | xargs)"
    [[ -z "$mode" ]] && continue
    run_name="phase36_${track}_${mode}_v5_lnfree"
    meta="${OUT_ROOT}/${run_name}/phase36_run_meta.json"
    if [[ "$SKIP_DONE" == "1" ]] && [[ -f "$meta" ]]; then
      if rg -q '"fp32_test_emotion6"' "$meta"; then
        echo "[SKIP] already completed: ${run_name}"
        continue
      fi
    fi
    echo "[MATRIX] track=${track} mode=${mode} run=${run_name}"
    if ! bash scripts/run_phase36_fp32_single.sh \
      --track "$track" \
      --mode "$mode" \
      --run-name "$run_name" \
      --epochs "$EPOCHS" \
      --device "$DEVICE" \
      --out-root "$OUT_ROOT"; then
      echo "[ERROR] run failed: ${run_name}"
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
        exit 1
      fi
    fi
  done
done

echo "[OK] Phase36 matrix completed."
