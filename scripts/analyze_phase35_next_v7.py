#!/usr/bin/env python3
"""
Analyze phase-3.5 next-direction v7 runs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


TARGET_F1 = 0.7


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze phase35 next v7 runs")
    p.add_argument("--out-dir", type=Path, default=Path("derived/reports"))
    return p.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_fp32(path: Path) -> Dict[str, Optional[float]]:
    d = load_json(path)
    g = d["global"]
    return {
        "emotion_acc": g["emotion"]["accuracy"],
        "emotion_macro_f1": g["emotion"]["macro_f1"],
        "arousal2_mae": g["arousal2"]["mae"],
        "arousal3_mae": g["arousal3"]["mae"],
    }


def read_ml(path: Path) -> Dict[str, Optional[float]]:
    d = load_json(path)
    g = d["global_metrics"]["fusion"]
    return {
        "emotion_acc": g["emotion6"]["accuracy"],
        "emotion_macro_f1": g["emotion6"]["macro_f1"],
        "arousal2_mae": g["arousal2"]["mae"],
        "arousal3_mae": g["arousal3"]["mae"],
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    refs = {
        "phase2_main_fusion": read_ml(Path("derived/results/ml_baselines_main/summary.json")),
        "phase3_main_fp32": read_fp32(Path("derived/results/fp32_multitask_main/summary.json")),
        "phase35_v5_logreg_main": read_ml(Path("derived/results/ml_baselines_phase35_v5_logreg_main/summary.json")),
        "phase35_v6_best": read_fp32(Path("derived/results/fp32_multitask_phase35_v6_ce_ls_ws_main/summary.json")),
    }

    specs = [
        ("fp32_v7_ce_ls_ws_gated_main", Path("derived/results/fp32_multitask_phase35_v7_ce_ls_ws_gated_main/summary.json"), "fp32"),
        ("fp32_v7_focal_ws_gated_main", Path("derived/results/fp32_multitask_phase35_v7_focal_ws_gated_main/summary.json"), "fp32"),
        ("fp32_v7_ce_ls_ws_gated_wide_main", Path("derived/results/fp32_multitask_phase35_v7_ce_ls_ws_gated_wide_main/summary.json"), "fp32"),
        ("ml_v7_rbfsvm_main", Path("derived/results/ml_baselines_phase35_v7_rbfsvm_main/summary.json"), "ml"),
        ("fp32_v6_best_ref", Path("derived/results/fp32_multitask_phase35_v6_ce_ls_ws_main/summary.json"), "fp32"),
        ("v5_logreg_main_ref", Path("derived/results/ml_baselines_phase35_v5_logreg_main/summary.json"), "ml"),
    ]
    rows: List[Dict[str, object]] = []
    for name, path, kind in specs:
        if not path.exists():
            continue
        m = read_fp32(path) if kind == "fp32" else read_ml(path)
        f1 = float(m["emotion_macro_f1"])
        rows.append(
            {
                "name": name,
                "kind": kind,
                "summary_json": str(path),
                "emotion_acc": m["emotion_acc"],
                "emotion_macro_f1": m["emotion_macro_f1"],
                "arousal2_mae": m["arousal2_mae"],
                "arousal3_mae": m["arousal3_mae"],
                "delta_f1_vs_phase2_main": f1 - float(refs["phase2_main_fusion"]["emotion_macro_f1"]),
                "delta_f1_vs_phase3_main": f1 - float(refs["phase3_main_fp32"]["emotion_macro_f1"]),
                "delta_f1_vs_v5_logreg_main": f1 - float(refs["phase35_v5_logreg_main"]["emotion_macro_f1"]),
                "delta_f1_vs_v6_best": f1 - float(refs["phase35_v6_best"]["emotion_macro_f1"]),
                "gap_to_0_7": TARGET_F1 - f1,
            }
        )

    rows.sort(key=lambda x: float(x["emotion_macro_f1"]), reverse=True)

    out_csv = args.out_dir / "phase35_next_v7_metrics.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "kind",
                "summary_json",
                "emotion_acc",
                "emotion_macro_f1",
                "arousal2_mae",
                "arousal3_mae",
                "delta_f1_vs_phase2_main",
                "delta_f1_vs_phase3_main",
                "delta_f1_vs_v5_logreg_main",
                "delta_f1_vs_v6_best",
                "gap_to_0_7",
            ],
        )
        w.writeheader()
        for row in rows:
            w.writerow(row)

    lines: List[str] = []
    lines.append("# Phase-3.5 Next v7 Report")
    lines.append("")
    lines.append("## Main Candidates")
    for row in rows:
        lines.append(
            "- `{}`: F1={:.4f}, acc={:.4f}, ΔvsV6={:+.4f}, gap_to_0.7={:.4f}".format(
                row["name"],
                float(row["emotion_macro_f1"]),
                float(row["emotion_acc"]),
                float(row["delta_f1_vs_v6_best"]),
                float(row["gap_to_0_7"]),
            )
        )
    lines.append("")
    if rows:
        best = max(rows, key=lambda x: float(x["emotion_macro_f1"]))
        lines.append("## Best")
        lines.append(
            "- `{}` F1={:.4f}, gap_to_0.7={:.4f}".format(
                best["name"],
                float(best["emotion_macro_f1"]),
                float(best["gap_to_0_7"]),
            )
        )
        lines.append("")
    lines.append("## Outputs")
    lines.append("- `derived/reports/phase35_next_v7_metrics.csv`")
    lines.append("- `derived/reports/phase35_next_v7_results.md`")

    out_md = args.out_dir / "phase35_next_v7_results.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "metrics_csv": str(out_csv),
                "report_md": str(out_md),
                "num_rows": len(rows),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
