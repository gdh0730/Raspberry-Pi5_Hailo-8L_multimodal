#!/usr/bin/env python3
"""Generate Phase36 FP32 vs Hailo comparison report from available summaries.

This script is non-destructive and does not run inference.
It merges:
- FP32 metrics from `fp32_test_eval.json`
- Hailo metrics from `pi_infer_batch/.../summary.json` when available
and marks missing runs as pending.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Phase36 FP32 vs Hailo report")
    p.add_argument("--meta-csv", type=Path, default=Path("derived/hailo/phase36_best5_build_meta.csv"))
    p.add_argument("--out-csv", type=Path, default=Path("derived/reports/phase36_fp32_vs_hailo_best5.csv"))
    p.add_argument("--out-md", type=Path, default=Path("derived/reports/phase36_fp32_vs_hailo_best5.md"))
    p.add_argument(
        "--hailo-out-root",
        type=Path,
        default=Path("derived/hailo/pi_infer_batch/phase36_best5"),
        help="Root containing <run_name>_hailo_test/summary.json",
    )
    return p.parse_args()


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if not args.meta_csv.exists():
        raise FileNotFoundError(f"Missing meta csv: {args.meta_csv}")

    rows: List[Dict[str, object]] = []
    with args.meta_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            track = r["track"].strip()
            mode = r["mode"].strip()
            run_name = r["run_name"].strip()
            fp32_eval = Path(f"derived/results/phase36/{run_name}/fp32_test_eval.json")
            hailo_summary = args.hailo_out_root / f"{run_name}_hailo_test" / "summary.json"

            row: Dict[str, object] = {
                "track": track,
                "mode": mode,
                "run_name": run_name,
                "status": "pending",
                "n": "",
                "fp32_acc": "",
                "fp32_f1": "",
                "hailo_acc": "",
                "hailo_f1": "",
                "delta_acc": "",
                "delta_f1": "",
                "hailo_summary_json": str(hailo_summary),
            }

            if fp32_eval.exists():
                f = read_json(fp32_eval).get("emotion6", {})
                row["fp32_acc"] = f.get("accuracy", "")
                row["fp32_f1"] = f.get("macro_f1", "")

            if hailo_summary.exists():
                h = read_json(hailo_summary).get("emotion6", {})
                hacc = h.get("accuracy", "")
                hf1 = h.get("macro_f1", "")
                row["n"] = h.get("n", "")
                row["hailo_acc"] = hacc
                row["hailo_f1"] = hf1
                if row["fp32_acc"] != "" and row["fp32_f1"] != "" and hacc != "" and hf1 != "":
                    row["delta_acc"] = float(hacc) - float(row["fp32_acc"])
                    row["delta_f1"] = float(hf1) - float(row["fp32_f1"])
                row["status"] = "done"

            rows.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "track",
                "mode",
                "run_name",
                "status",
                "n",
                "fp32_acc",
                "fp32_f1",
                "hailo_acc",
                "hailo_f1",
                "delta_acc",
                "delta_f1",
                "hailo_summary_json",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    done = [r for r in rows if r["status"] == "done"]
    pending = [r for r in rows if r["status"] != "done"]
    done_sorted = sorted(done, key=lambda r: float(r["delta_f1"]))

    lines = [
        "# Phase36 FP32 vs Hailo (Best5)",
        "",
        f"- done: {len(done)}",
        f"- pending: {len(pending)}",
        "",
        "| Track | Mode | Status | FP32 F1 | Hailo F1 | Delta F1(H-F) | FP32 Acc | Hailo Acc | Delta Acc(H-F) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in done_sorted:
        lines.append(
            "| {track} | {mode} | {status} | {fp32_f1:.4f} | {hailo_f1:.4f} | {delta_f1:.4f} | {fp32_acc:.4f} | {hailo_acc:.4f} | {delta_acc:.4f} |".format(
                track=r["track"],
                mode=r["mode"],
                status=r["status"],
                fp32_f1=float(r["fp32_f1"]),
                hailo_f1=float(r["hailo_f1"]),
                delta_f1=float(r["delta_f1"]),
                fp32_acc=float(r["fp32_acc"]),
                hailo_acc=float(r["hailo_acc"]),
                delta_acc=float(r["delta_acc"]),
            )
        )
    for r in pending:
        fp32_f1 = f"{float(r['fp32_f1']):.4f}" if r["fp32_f1"] != "" else "-"
        fp32_acc = f"{float(r['fp32_acc']):.4f}" if r["fp32_acc"] != "" else "-"
        lines.append(
            f"| {r['track']} | {r['mode']} | {r['status']} | {fp32_f1} | - | - | {fp32_acc} | - | - |"
        )

    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(args.out_csv))
    print(str(args.out_md))


if __name__ == "__main__":
    main()
