#!/usr/bin/env bash
set -euo pipefail

# Compare multiple HEF candidates on the same Pi inference/eval protocol.
#
# candidates CSV format:
#   name,hef_local
#   fp32_v8_fold0,derived/hailo/build/fp32_v8_fold0/fp32_v8_fold0.hef
#   probe4,derived/hailo/build/sdk_probe4/fp32_v8_fold0.hef

HOST=""
CANDIDATES_CSV=""
START_IDX=0
MAX_SAMPLES=0
CALIB_DIR="derived/hailo/calib/fold0_train_1024"
INDEX_CSV=""
MANIFEST="derived/manifests/manifest_multimodal_common6_av.jsonl"
OUT_DIR="derived/hailo/pi_compare"
USE_PASSWORD=0
NORMALIZE_CHECKPOINT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --candidates-csv) CANDIDATES_CSV="$2"; shift 2 ;;
    --start-idx) START_IDX="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --calib-dir) CALIB_DIR="$2"; shift 2 ;;
    --index-csv) INDEX_CSV="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --normalize-checkpoint) NORMALIZE_CHECKPOINT="$2"; shift 2 ;;
    --use-password) USE_PASSWORD=1; shift ;;
    *) echo "[ERROR] Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$HOST" ]]; then
  echo "[ERROR] --host is required"
  exit 1
fi
if [[ -z "$CANDIDATES_CSV" || ! -f "$CANDIDATES_CSV" ]]; then
  echo "[ERROR] --candidates-csv is required and must exist"
  exit 1
fi
if [[ -z "$INDEX_CSV" ]]; then
  INDEX_CSV="${CALIB_DIR}/index.csv"
fi
if [[ ! -f "$INDEX_CSV" ]]; then
  echo "[ERROR] Missing index CSV: $INDEX_CSV"
  exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "[ERROR] Missing manifest: $MANIFEST"
  exit 1
fi
if [[ -n "$NORMALIZE_CHECKPOINT" && ! -f "$NORMALIZE_CHECKPOINT" ]]; then
  echo "[ERROR] Missing --normalize-checkpoint file: $NORMALIZE_CHECKPOINT"
  exit 1
fi

mkdir -p "$OUT_DIR"
RUN_LOG="${OUT_DIR}/runs.txt"
: > "$RUN_LOG"

echo "[E5-COMPARE] Start candidates from: $CANDIDATES_CSV"
tail -n +2 "$CANDIDATES_CSV" | while IFS=, read -r NAME HEF_LOCAL; do
  NAME="$(echo "$NAME" | xargs)"
  HEF_LOCAL="$(echo "$HEF_LOCAL" | xargs)"
  if [[ -z "$NAME" || -z "$HEF_LOCAL" ]]; then
    continue
  fi
  if [[ ! -f "$HEF_LOCAL" ]]; then
    echo "[WARN] skip $NAME (missing hef: $HEF_LOCAL)"
    continue
  fi
  echo "[E5-COMPARE] Run candidate: $NAME"

  ARGS=(
    --host "$HOST"
    --hef-local "$HEF_LOCAL"
    --name "$NAME"
    --calib-dir "$CALIB_DIR"
    --index-csv "$INDEX_CSV"
    --manifest "$MANIFEST"
    --start-idx "$START_IDX"
    --max-samples "$MAX_SAMPLES"
    --out-dir "$OUT_DIR/runs"
  )
  if [[ "$USE_PASSWORD" -eq 1 ]]; then
    ARGS+=(--use-password)
  fi
  if [[ -n "$NORMALIZE_CHECKPOINT" ]]; then
    ARGS+=(--normalize-checkpoint "$NORMALIZE_CHECKPOINT")
  fi

  bash scripts/run_e5_infer_pi_batch.sh "${ARGS[@]}"
  echo "$NAME,$OUT_DIR/runs/$NAME/summary.json,$OUT_DIR/runs/$NAME/summary_bootstrap.json" >> "$RUN_LOG"
done

PY_BIN=""
if [[ -x ".venv/bin/python" ]]; then
  PY_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY_BIN="python3"
else
  PY_BIN="python"
fi

"$PY_BIN" - "$RUN_LOG" "$OUT_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_log = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
rows = []
if run_log.exists():
    with run_log.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, summary_json, boot_json = [x.strip() for x in line.split(",", 2)]
            sp = Path(summary_json)
            bp = Path(boot_json)
            if not sp.exists():
                continue
            s = json.loads(sp.read_text(encoding="utf-8"))
            b = json.loads(bp.read_text(encoding="utf-8")) if bp.exists() else {}
            e = s.get("emotion6", {})
            eci = b.get("emotion6", {})
            rows.append(
                {
                    "name": name,
                    "n": e.get("n"),
                    "acc": e.get("accuracy"),
                    "macro_f1": e.get("macro_f1"),
                    "acc_ci95": eci.get("accuracy_ci95"),
                    "macro_f1_ci95": eci.get("macro_f1_ci95"),
                    "summary_json": str(sp),
                }
            )

rows.sort(key=lambda r: (r["macro_f1"] is not None, r["macro_f1"]), reverse=True)
csv_path = out_dir / "compare_metrics.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(
        f,
        fieldnames=["name", "n", "acc", "macro_f1", "acc_ci95", "macro_f1_ci95", "summary_json"],
    )
    w.writeheader()
    w.writerows(rows)

md_path = out_dir / "compare_report.md"
lines = ["# E5 Pi HEF Compare", "", f"- candidates: {len(rows)}", ""]
for r in rows:
    lines.append(
        "- `{}`: F1={:.6f}, acc={:.6f}, n={}".format(
            r["name"],
            float(r["macro_f1"]) if r["macro_f1"] is not None else float("nan"),
            float(r["acc"]) if r["acc"] is not None else float("nan"),
            r["n"],
        )
    )
lines.append("")
lines.append(f"- csv: `{csv_path}`")
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(str(csv_path))
print(str(md_path))
PY

echo "[E5-COMPARE] Done."
echo " - run log : $RUN_LOG"
echo " - metrics : $OUT_DIR/compare_metrics.csv"
echo " - report  : $OUT_DIR/compare_report.md"
