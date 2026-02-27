#!/usr/bin/env python3
"""Analyze phase-3.5 next-direction v8 HuBERT run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


TARGET_F1 = 0.7


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze phase35 next v8 runs")
    p.add_argument("--out-dir", type=Path, default=Path("derived/reports"))
    return p.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_fp32(path: Path) -> Dict[str, Optional[float]]:
    d = load_json(path)
    g = d["global"]
    run = d.get("run", {})
    return {
        "emotion_acc": g["emotion"]["accuracy"],
        "emotion_macro_f1": g["emotion"]["macro_f1"],
        "arousal2_mae": g["arousal2"]["mae"],
        "arousal3_mae": g["arousal3"]["mae"],
        "device": run.get("device_resolved", run.get("device")),
    }


def read_ensemble(path: Path) -> Dict[str, Optional[float]]:
    d = load_json(path)
    g = d["global"]
    return {
        "emotion_acc": g["emotion"]["accuracy"],
        "emotion_macro_f1": g["emotion"]["macro_f1"],
        "arousal2_mae": g["arousal2"]["mae"],
        "arousal3_mae": g["arousal3"]["mae"],
        "device": "ensemble",
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    refs = {
        "phase2_main_fusion": read_fp32(Path("derived/results/fp32_multitask_main/summary.json")),
        "v7_best": read_fp32(Path("derived/results/fp32_multitask_phase35_v7_ce_ls_ws_gated_wide_main/summary.json")),
        "v7_best_cuda": read_fp32(Path("derived/results/fp32_multitask_phase35_v7_ce_ls_ws_gated_wide_main_cuda/summary.json")),
    }

    specs = [
        ("fp32_v8_hubert_gated_wide_main", Path("derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_main/summary.json"), "fp32"),
        ("fp32_v8_hubert_gated_wide_tune1", Path("derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune1/summary.json"), "fp32"),
        ("fp32_v8_hubert_gated_wide_tune2", Path("derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune2/summary.json"), "fp32"),
        ("fp32_v8_hubert_gated_wide_tune3", Path("derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune3/summary.json"), "fp32"),
        ("fp32_v8_hubert_gated_wide_tune4", Path("derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune4/summary.json"), "fp32"),
        ("fp32_v8_hubert_ensemble_vote3", Path("derived/results/fp32_multitask_phase35_v8_hubert_ensemble_vote3/summary.json"), "ensemble"),
        (
            "fp32_v8_hubert_ensemble_vote3_main_t3_t4",
            Path("derived/results/fp32_multitask_phase35_v8_hubert_ensemble_vote3_main_t3_t4/summary.json"),
            "ensemble",
        ),
        ("fp32_v7_best_ref", Path("derived/results/fp32_multitask_phase35_v7_ce_ls_ws_gated_wide_main/summary.json"), "fp32"),
        ("fp32_v7_best_cuda_ref", Path("derived/results/fp32_multitask_phase35_v7_ce_ls_ws_gated_wide_main_cuda/summary.json"), "fp32"),
    ]

    rows: List[Dict[str, object]] = []
    for name, path, kind in specs:
        if not path.exists():
            continue
        if kind == "ensemble":
            m = read_ensemble(path)
        else:
            m = read_fp32(path)
        f1 = float(m["emotion_macro_f1"])
        rows.append(
            {
                "name": name,
                "summary_json": str(path),
                "device": m["device"],
                "emotion_acc": m["emotion_acc"],
                "emotion_macro_f1": m["emotion_macro_f1"],
                "arousal2_mae": m["arousal2_mae"],
                "arousal3_mae": m["arousal3_mae"],
                "delta_f1_vs_v7_best": f1 - float(refs["v7_best"]["emotion_macro_f1"]),
                "delta_f1_vs_v7_best_cuda": f1 - float(refs["v7_best_cuda"]["emotion_macro_f1"]),
                "gap_to_0_7": TARGET_F1 - f1,
            }
        )

    rows.sort(key=lambda x: float(x["emotion_macro_f1"]), reverse=True)

    out_csv = args.out_dir / "phase35_next_v8_metrics.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "summary_json",
                "device",
                "emotion_acc",
                "emotion_macro_f1",
                "arousal2_mae",
                "arousal3_mae",
                "delta_f1_vs_v7_best",
                "delta_f1_vs_v7_best_cuda",
                "gap_to_0_7",
            ],
        )
        w.writeheader()
        for row in rows:
            w.writerow(row)

    lines: List[str] = []
    lines.append("# Phase-3.5 Next v8 Report")
    lines.append("")
    lines.append("## Main Candidates")
    for row in rows:
        lines.append(
            "- `{}`: F1={:.4f}, acc={:.4f}, device={}, ΔvsV7={:+.4f}, gap_to_0.7={:.4f}".format(
                row["name"],
                float(row["emotion_macro_f1"]),
                float(row["emotion_acc"]),
                row["device"],
                float(row["delta_f1_vs_v7_best"]),
                float(row["gap_to_0_7"]),
            )
        )
    lines.append("")
    lines.append("## Outputs")
    lines.append("- `derived/reports/phase35_next_v8_metrics.csv`")
    lines.append("- `derived/reports/phase35_next_v8_results.md`")

    out_md = args.out_dir / "phase35_next_v8_results.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"metrics_csv": str(out_csv), "report_md": str(out_md), "num_rows": len(rows)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
