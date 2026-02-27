#!/usr/bin/env python3
"""
Analyze phase-3.5 advancement runs (cache_v2 preprocessing + algorithm comparisons).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze phase35 advancement runs")
    p.add_argument("--out-dir", type=Path, default=Path("derived/reports"))
    return p.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def phase2_main_fusion() -> Dict[str, float]:
    d = load_json(Path("derived/results/ml_baselines_main/summary.json"))
    g = d["global_metrics"]["fusion"]
    return {
        "emotion_acc": g["emotion6"]["accuracy"],
        "emotion_macro_f1": g["emotion6"]["macro_f1"],
        "arousal2_mae": g["arousal2"]["mae"],
        "arousal3_mae": g["arousal3"]["mae"],
    }


def phase3_main_fp32() -> Dict[str, float]:
    d = load_json(Path("derived/results/fp32_multitask_main/summary.json"))
    g = d["global"]
    return {
        "emotion_acc": g["emotion"]["accuracy"],
        "emotion_macro_f1": g["emotion"]["macro_f1"],
        "arousal2_mae": g["arousal2"]["mae"],
        "arousal3_mae": g["arousal3"]["mae"],
    }


def read_ml_run(path: Path, mode: str) -> Dict[str, float]:
    d = load_json(path)
    g = d["global_metrics"][mode]
    return {
        "emotion_acc": g["emotion6"]["accuracy"],
        "emotion_macro_f1": g["emotion6"]["macro_f1"],
        "arousal2_mae": g["arousal2"]["mae"],
        "arousal3_mae": g["arousal3"]["mae"],
    }


def read_fp32_run(path: Path) -> Dict[str, float]:
    d = load_json(path)
    g = d["global"]
    return {
        "emotion_acc": g["emotion"]["accuracy"],
        "emotion_macro_f1": g["emotion"]["macro_f1"],
        "arousal2_mae": g["arousal2"]["mae"],
        "arousal3_mae": g["arousal3"]["mae"],
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    p2 = phase2_main_fusion()
    p3 = phase3_main_fp32()

    runs: List[Dict[str, object]] = []
    candidates = [
        ("ml_v2_logreg_audio", Path("derived/results/ml_baselines_phase35_v2_logreg_main/summary.json"), "ml", "audio"),
        ("ml_v2_logreg_video", Path("derived/results/ml_baselines_phase35_v2_logreg_main/summary.json"), "ml", "video"),
        ("ml_v2_logreg_fusion", Path("derived/results/ml_baselines_phase35_v2_logreg_main/summary.json"), "ml", "fusion"),
        ("ml_v2_rf_audio", Path("derived/results/ml_baselines_phase35_v2_rf_main/summary.json"), "ml", "audio"),
        ("ml_v2_rf_video", Path("derived/results/ml_baselines_phase35_v2_rf_main/summary.json"), "ml", "video"),
        ("ml_v2_rf_fusion", Path("derived/results/ml_baselines_phase35_v2_rf_main/summary.json"), "ml", "fusion"),
        ("fp32_v2_ce_fusion", Path("derived/results/fp32_multitask_phase35_v2_ce_main/summary.json"), "fp32", "fusion"),
    ]

    for name, path, kind, mode in candidates:
        if not path.exists():
            continue
        m = read_ml_run(path, mode) if kind == "ml" else read_fp32_run(path)
        runs.append(
            {
                "name": name,
                "kind": kind,
                "mode": mode,
                "summary_json": str(path),
                "emotion_acc": m["emotion_acc"],
                "emotion_macro_f1": m["emotion_macro_f1"],
                "arousal2_mae": m["arousal2_mae"],
                "arousal3_mae": m["arousal3_mae"],
                "delta_f1_vs_phase2_fusion_main": m["emotion_macro_f1"] - p2["emotion_macro_f1"],
                "delta_f1_vs_phase3_fp32_main": m["emotion_macro_f1"] - p3["emotion_macro_f1"],
            }
        )

    runs.sort(key=lambda r: float(r["emotion_macro_f1"]), reverse=True)

    out_csv = args.out_dir / "phase35_advancement_metrics.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "kind",
                "mode",
                "summary_json",
                "emotion_acc",
                "emotion_macro_f1",
                "arousal2_mae",
                "arousal3_mae",
                "delta_f1_vs_phase2_fusion_main",
                "delta_f1_vs_phase3_fp32_main",
            ],
        )
        w.writeheader()
        for r in runs:
            w.writerow(r)

    lines: List[str] = []
    lines.append("# Phase-3.5 Advancement Result Report")
    lines.append("")
    lines.append("## References")
    lines.append(
        "- phase2_fusion_main: F1={:.4f}, acc={:.4f}".format(
            p2["emotion_macro_f1"], p2["emotion_acc"]
        )
    )
    lines.append(
        "- phase3_fp32_main: F1={:.4f}, acc={:.4f}".format(
            p3["emotion_macro_f1"], p3["emotion_acc"]
        )
    )
    lines.append("")
    lines.append("## Candidates")
    for r in runs:
        lines.append(
            "- `{}`: F1={:.4f}, acc={:.4f}, ΔvsP2={:+.4f}, ΔvsP3={:+.4f}".format(
                r["name"],
                float(r["emotion_macro_f1"]),
                float(r["emotion_acc"]),
                float(r["delta_f1_vs_phase2_fusion_main"]),
                float(r["delta_f1_vs_phase3_fp32_main"]),
            )
        )
    lines.append("")
    if runs:
        best = runs[0]
        lines.append("## Best Candidate")
        lines.append("- `{}` (F1={:.4f})".format(best["name"], float(best["emotion_macro_f1"])))
    lines.append("")
    lines.append("## Outputs")
    lines.append("- `derived/reports/phase35_advancement_metrics.csv`")
    lines.append("- `derived/reports/phase35_advancement_results.md`")

    out_md = args.out_dir / "phase35_advancement_results.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "metrics_csv": str(out_csv),
                "report_md": str(out_md),
                "num_candidates": len(runs),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
