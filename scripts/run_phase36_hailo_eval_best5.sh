#!/usr/bin/env bash
set -euo pipefail

# Evaluate Phase36 best-per-track HEFs on Pi with exact test indices.
#
# Input meta: derived/hailo/phase36_best5_build_meta.csv
# For each build_status=ok and existing HEF:
#   - run Pi batch inference (or skip if summary exists)
#   - collect FP32/Hailo gap table
#
# Example:
#   PI_PASSWORD='***' bash scripts/run_phase36_hailo_eval_best5.sh \
#     --host wormhole@129.254.232.91 \
#     --use-password

HOST=""
META_CSV="derived/hailo/phase36_best5_build_meta.csv"
MANIFEST="derived/manifests/manifest_multimodal_common6_av.jsonl"
EVAL_ROOT="derived/hailo/eval/phase36"
OUT_ROOT="derived/hailo/pi_infer_batch/phase36_best5"
REPORT_CSV="derived/reports/phase36_fp32_vs_hailo_best5.csv"
REPORT_MD="derived/reports/phase36_fp32_vs_hailo_best5.md"
MAX_SAMPLES=0
USE_PASSWORD=0
TRACKS=""
SKIP_DONE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --meta-csv) META_CSV="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --eval-root) EVAL_ROOT="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --report-csv) REPORT_CSV="$2"; shift 2 ;;
    --report-md) REPORT_MD="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --tracks) TRACKS="$2"; shift 2 ;;
    --skip-done) SKIP_DONE="$2"; shift 2 ;;
    --use-password) USE_PASSWORD=1; shift ;;
    *) echo "[ERROR] Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$HOST" ]]; then
  echo "[ERROR] --host is required"
  exit 1
fi
if [[ ! -f "$META_CSV" ]]; then
  echo "[ERROR] Missing meta csv: $META_CSV"
  exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "[ERROR] Missing manifest: $MANIFEST"
  exit 1
fi
if [[ ! -x ".venv/bin/python" ]]; then
  echo "[ERROR] Missing .venv/bin/python"
  exit 1
fi

mkdir -p "$OUT_ROOT" "$(dirname "$REPORT_CSV")" "$(dirname "$REPORT_MD")"
echo "track,mode,run_name,n,fp32_acc,fp32_f1,hailo_acc,hailo_f1,delta_acc,delta_f1,hailo_summary_json" > "$REPORT_CSV"

while IFS=, read -r TRACK MODE RUN_NAME FP32_ACC FP32_F1 CKPT ONNX_FULL CALIB_DIR HEF_PATH BUILD_STATUS; do
  if [[ "$TRACK" == "track" ]]; then
    continue
  fi
  TRACK="${TRACK//$'\r'/}"
  MODE="${MODE//$'\r'/}"
  RUN_NAME="${RUN_NAME//$'\r'/}"
  FP32_ACC="${FP32_ACC//$'\r'/}"
  FP32_F1="${FP32_F1//$'\r'/}"
  HEF_PATH="${HEF_PATH//$'\r'/}"
  BUILD_STATUS="${BUILD_STATUS//$'\r'/}"

  if [[ -n "$TRACKS" ]]; then
    case ",$TRACKS," in
      *",$TRACK,"*) ;;
      *) continue ;;
    esac
  fi

  if [[ "$BUILD_STATUS" != "ok" ]]; then
    echo "[WARN] skip ${RUN_NAME} (build_status=${BUILD_STATUS})"
    continue
  fi
  if [[ ! -f "$HEF_PATH" ]]; then
    echo "[WARN] skip ${RUN_NAME} (missing HEF: $HEF_PATH)"
    continue
  fi

  TEST_DIR="${EVAL_ROOT}/${RUN_NAME}_test"
  INDEX_CSV="${TEST_DIR}/index.csv"
  if [[ ! -f "$INDEX_CSV" ]]; then
    echo "[WARN] skip ${RUN_NAME} (missing index: $INDEX_CSV)"
    continue
  fi

  PI_NAME="${RUN_NAME}_hailo_test"
  PI_DIR="${OUT_ROOT}/${PI_NAME}"
  SUMMARY_JSON="${PI_DIR}/summary.json"
  if [[ "$SKIP_DONE" == "1" && -f "$SUMMARY_JSON" ]]; then
    echo "[SKIP] already done: ${PI_NAME}"
  else
    echo "[RUN] ${PI_NAME}"
    ARGS=(
      --host "$HOST"
      --hef-local "$HEF_PATH"
      --name "$PI_NAME"
      --calib-dir "$TEST_DIR"
      --index-csv "$INDEX_CSV"
      --manifest "$MANIFEST"
      --max-samples "$MAX_SAMPLES"
      --out-dir "$OUT_ROOT"
    )
    if [[ "$USE_PASSWORD" == "1" ]]; then
      ARGS+=(--use-password)
    fi
    bash scripts/run_e5_infer_pi_batch.sh "${ARGS[@]}"
  fi

  if [[ ! -f "$SUMMARY_JSON" ]]; then
    echo "[WARN] missing summary after run: $SUMMARY_JSON"
    continue
  fi

  LINE=$(.venv/bin/python - "$TRACK" "$MODE" "$RUN_NAME" "$FP32_ACC" "$FP32_F1" "$SUMMARY_JSON" <<'PY'
import json
import sys
track, mode, run_name, fp32_acc, fp32_f1, summary_json = sys.argv[1:]
s = json.loads(open(summary_json, "r", encoding="utf-8").read())
e = s.get("emotion6", {})
hacc = float(e.get("accuracy", 0.0))
hf1 = float(e.get("macro_f1", 0.0))
n = int(e.get("n", 0))
facc = float(fp32_acc)
ff1 = float(fp32_f1)
print(
    f"{track},{mode},{run_name},{n},{facc:.12f},{ff1:.12f},{hacc:.12f},{hf1:.12f},{(hacc-facc):.12f},{(hf1-ff1):.12f},{summary_json}"
)
PY
)
  echo "$LINE" >> "$REPORT_CSV"
done < "$META_CSV"

.venv/bin/python - "$REPORT_CSV" "$REPORT_MD" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
md_path = Path(sys.argv[2])
rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))

rows.sort(key=lambda r: float(r["delta_f1"]))
lines = [
    "# Phase36 FP32 vs Hailo (Best5)",
    "",
    f"- rows: {len(rows)}",
    "",
    "| Track | Mode | FP32 F1 | Hailo F1 | Delta F1(H-F) | FP32 Acc | Hailo Acc | Delta Acc(H-F) |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        "| {track} | {mode} | {fp32_f1:.4f} | {hailo_f1:.4f} | {delta_f1:.4f} | {fp32_acc:.4f} | {hailo_acc:.4f} | {delta_acc:.4f} |".format(
            track=r["track"],
            mode=r["mode"],
            fp32_f1=float(r["fp32_f1"]),
            hailo_f1=float(r["hailo_f1"]),
            delta_f1=float(r["delta_f1"]),
            fp32_acc=float(r["fp32_acc"]),
            hailo_acc=float(r["hailo_acc"]),
            delta_acc=float(r["delta_acc"]),
        )
    )
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(str(md_path))
PY

echo "[OK] phase36 fp32-vs-hailo report updated"
echo " - csv : $REPORT_CSV"
echo " - md  : $REPORT_MD"
