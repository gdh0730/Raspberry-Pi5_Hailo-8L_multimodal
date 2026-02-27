#!/usr/bin/env python3
"""
Analyze phase-3.5 candidate runs and compare against existing references.

References:
- phase2_fusion_main: derived/results/ml_baselines_main/summary.json
- phase3_main: derived/results/fp32_multitask_main/summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze phase-3.5 candidate runs")
    p.add_argument(
        "--candidate-dirs",
        type=str,
        required=True,
        help="Comma-separated result directories each containing summary.json",
    )
    p.add_argument("--out-dir", type=Path, default=Path("derived/reports"))
    return p.parse_args()


def read_phase2_main(path: Path) -> Dict[str, float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    g = d["global_metrics"]["fusion"]
    return {
        "emotion_acc": g["emotion6"]["accuracy"],
        "emotion_macro_f1": g["emotion6"]["macro_f1"],
        "arousal2_mae": g["arousal2"]["mae"],
        "arousal3_mae": g["arousal3"]["mae"],
    }


def read_phase3_main(path: Path) -> Dict[str, float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    g = d["global"]
    return {
        "emotion_acc": g["emotion"]["accuracy"],
        "emotion_macro_f1": g["emotion"]["macro_f1"],
        "arousal2_mae": g["arousal2"]["mae"],
        "arousal3_mae": g["arousal3"]["mae"],
    }


def read_candidate(path: Path) -> Dict[str, object]:
    d = json.loads(path.read_text(encoding="utf-8"))
    g = d["global"]
    run = d["run"]
    return {
        "name": path.parent.name,
        "summary_json": str(path),
        "emotion_acc": g["emotion"]["accuracy"],
        "emotion_macro_f1": g["emotion"]["macro_f1"],
        "arousal2_mae": g["arousal2"]["mae"],
        "arousal3_mae": g["arousal3"]["mae"],
        "epochs": run.get("epochs"),
        "lr": run.get("lr"),
        "dropout": run.get("dropout"),
        "emotion_loss": run.get("emotion_loss", "ce"),
        "focal_gamma": run.get("focal_gamma"),
        "weighted_sampler": run.get("weighted_sampler", False),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    phase2_ref = read_phase2_main(Path("derived/results/ml_baselines_main/summary.json"))
    phase3_ref = read_phase3_main(Path("derived/results/fp32_multitask_main/summary.json"))

    candidate_dirs = [Path(p.strip()) for p in args.candidate_dirs.split(",") if p.strip()]
    rows: List[Dict[str, object]] = []
    for cdir in candidate_dirs:
        summary = cdir / "summary.json"
        if not summary.exists():
            raise FileNotFoundError(f"Missing candidate summary: {summary}")
        row = read_candidate(summary)
        row["delta_f1_vs_phase2_fusion"] = row["emotion_macro_f1"] - phase2_ref["emotion_macro_f1"]
        row["delta_f1_vs_phase3_main"] = row["emotion_macro_f1"] - phase3_ref["emotion_macro_f1"]
        rows.append(row)

    rows.sort(key=lambda r: float(r["emotion_macro_f1"]), reverse=True)

    out_csv = args.out_dir / "phase35_candidate_metrics.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "summary_json",
                "emotion_acc",
                "emotion_macro_f1",
                "arousal2_mae",
                "arousal3_mae",
                "delta_f1_vs_phase2_fusion",
                "delta_f1_vs_phase3_main",
                "epochs",
                "lr",
                "dropout",
                "emotion_loss",
                "focal_gamma",
                "weighted_sampler",
            ],
        )
        w.writeheader()
        for row in rows:
            w.writerow(row)

    best = rows[0] if rows else None
    lines: List[str] = []
    lines.append("# Phase-3.5 Candidate Report")
    lines.append("")
    lines.append("## Reference")
    lines.append(
        "- phase2_fusion_main: macro-F1={:.4f}, acc={:.4f}".format(
            phase2_ref["emotion_macro_f1"], phase2_ref["emotion_acc"]
        )
    )
    lines.append(
        "- phase3_main: macro-F1={:.4f}, acc={:.4f}".format(
            phase3_ref["emotion_macro_f1"], phase3_ref["emotion_acc"]
        )
    )
    lines.append("")
    lines.append("## Candidates")
    for r in rows:
        lines.append(
            "- `{}`: F1={:.4f}, acc={:.4f}, ΔvsP2={:+.4f}, ΔvsP3={:+.4f}, loss={}, ws={}".format(
                r["name"],
                float(r["emotion_macro_f1"]),
                float(r["emotion_acc"]),
                float(r["delta_f1_vs_phase2_fusion"]),
                float(r["delta_f1_vs_phase3_main"]),
                r["emotion_loss"],
                r["weighted_sampler"],
            )
        )
    lines.append("")
    if best is not None:
        lines.append("## Best Candidate")
        lines.append(
            "- `{}` (macro-F1={:.4f})".format(best["name"], float(best["emotion_macro_f1"]))
        )
    lines.append("")
    lines.append("## Output")
    lines.append("- `derived/reports/phase35_candidate_metrics.csv`")
    lines.append("- `derived/reports/phase35_results.md`")

    out_md = args.out_dir / "phase35_results.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "csv": str(out_csv),
                "md": str(out_md),
                "num_candidates": len(rows),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
