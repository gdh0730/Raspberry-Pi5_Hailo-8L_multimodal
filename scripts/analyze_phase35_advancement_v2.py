#!/usr/bin/env python3
"""
Analyze phase-3.5 advancement v2 runs (raw cache_v3 + pretrained video embedding).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze phase35 advancement v2 runs")
    p.add_argument("--out-dir", type=Path, default=Path("derived/reports"))
    return p.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_phase2_main_fusion() -> Dict[str, float]:
    d = load_json(Path("derived/results/ml_baselines_main/summary.json"))
    g = d["global_metrics"]["fusion"]
    return {
        "emotion_acc": g["emotion6"]["accuracy"],
        "emotion_macro_f1": g["emotion6"]["macro_f1"],
        "arousal2_mae": g["arousal2"]["mae"],
        "arousal3_mae": g["arousal3"]["mae"],
    }


def read_phase3_main_fp32() -> Dict[str, float]:
    d = load_json(Path("derived/results/fp32_multitask_main/summary.json"))
    g = d["global"]
    return {
        "emotion_acc": g["emotion"]["accuracy"],
        "emotion_macro_f1": g["emotion"]["macro_f1"],
        "arousal2_mae": g["arousal2"]["mae"],
        "arousal3_mae": g["arousal3"]["mae"],
    }


def read_ml(path: Path, mode: str = "fusion") -> Dict[str, float]:
    d = load_json(path)
    g = d["global_metrics"][mode]
    return {
        "emotion_acc": g["emotion6"]["accuracy"],
        "emotion_macro_f1": g["emotion6"]["macro_f1"],
        "arousal2_mae": g["arousal2"]["mae"],
        "arousal3_mae": g["arousal3"]["mae"],
    }


def read_fp32(path: Path) -> Dict[str, float]:
    d = load_json(path)
    g = d["global"]
    return {
        "emotion_acc": g["emotion"]["accuracy"],
        "emotion_macro_f1": g["emotion"]["macro_f1"],
        "arousal2_mae": g["arousal2"]["mae"],
        "arousal3_mae": g["arousal3"]["mae"],
    }


def try_read_main_candidates() -> List[Dict[str, object]]:
    cand_specs = [
        ("ml_v3_logreg_fusion", Path("derived/results/ml_baselines_phase35_v3_logreg_main/summary.json"), "ml"),
        ("ml_v3_rbfsvm_fusion", Path("derived/results/ml_baselines_phase35_v3_rbfsvm_main/summary.json"), "ml"),
        ("fp32_v3_ce_fusion", Path("derived/results/fp32_multitask_phase35_v3_ce_main/summary.json"), "fp32"),
    ]
    rows: List[Dict[str, object]] = []
    for name, path, kind in cand_specs:
        if not path.exists():
            continue
        m = read_ml(path, mode="fusion") if kind == "ml" else read_fp32(path)
        rows.append(
            {
                "name": name,
                "kind": kind,
                "mode": "fusion",
                "summary_json": str(path),
                "emotion_acc": m["emotion_acc"],
                "emotion_macro_f1": m["emotion_macro_f1"],
                "arousal2_mae": m["arousal2_mae"],
                "arousal3_mae": m["arousal3_mae"],
            }
        )
    return rows


def try_read_cross_candidates() -> List[Dict[str, object]]:
    cross_specs = [
        ("ml_v3_logreg_cross_crema_to_ravdess", Path("derived/results/ml_baselines_phase35_v3_logreg_cross_crema_to_ravdess/summary.json"), "cross_crema_to_ravdess"),
        ("ml_v3_logreg_cross_ravdess_to_crema", Path("derived/results/ml_baselines_phase35_v3_logreg_cross_ravdess_to_crema/summary.json"), "cross_ravdess_to_crema"),
        ("ml_v3_rbfsvm_cross_crema_to_ravdess", Path("derived/results/ml_baselines_phase35_v3_rbfsvm_cross_crema_to_ravdess/summary.json"), "cross_crema_to_ravdess"),
        ("ml_v3_rbfsvm_cross_ravdess_to_crema", Path("derived/results/ml_baselines_phase35_v3_rbfsvm_cross_ravdess_to_crema/summary.json"), "cross_ravdess_to_crema"),
    ]
    rows: List[Dict[str, object]] = []
    for name, path, run_name in cross_specs:
        if not path.exists():
            continue
        m = read_ml(path, mode="fusion")
        rows.append(
            {
                "name": name,
                "run": run_name,
                "summary_json": str(path),
                "emotion_acc": m["emotion_acc"],
                "emotion_macro_f1": m["emotion_macro_f1"],
                "arousal2_mae": m["arousal2_mae"],
                "arousal3_mae": m["arousal3_mae"],
            }
        )
    return rows


def read_phase2_cross_ref() -> Dict[str, float]:
    refs: Dict[str, float] = {}
    with Path("derived/reports/phase2_global_metrics.csv").open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["mode"] != "fusion":
                continue
            refs[row["run"]] = float(row["emotion_macro_f1"])
    return refs


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    p2 = read_phase2_main_fusion()
    p3 = read_phase3_main_fp32()

    main_rows = try_read_main_candidates()
    for row in main_rows:
        row["delta_f1_vs_phase2_fusion_main"] = row["emotion_macro_f1"] - p2["emotion_macro_f1"]
        row["delta_f1_vs_phase3_fp32_main"] = row["emotion_macro_f1"] - p3["emotion_macro_f1"]
    main_rows.sort(key=lambda x: float(x["emotion_macro_f1"]), reverse=True)

    main_csv = args.out_dir / "phase35_advancement_v2_main_metrics.csv"
    with main_csv.open("w", newline="", encoding="utf-8") as f:
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
        for row in main_rows:
            w.writerow(row)

    cross_rows = try_read_cross_candidates()
    cross_ref = read_phase2_cross_ref()
    for row in cross_rows:
        ref = cross_ref.get(row["run"])
        row["phase2_fusion_f1"] = ref
        row["delta_f1_vs_phase2_cross"] = None if ref is None else row["emotion_macro_f1"] - ref

    cross_csv = args.out_dir / "phase35_advancement_v2_cross_metrics.csv"
    with cross_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "run",
                "summary_json",
                "emotion_acc",
                "emotion_macro_f1",
                "arousal2_mae",
                "arousal3_mae",
                "phase2_fusion_f1",
                "delta_f1_vs_phase2_cross",
            ],
        )
        w.writeheader()
        for row in cross_rows:
            w.writerow(row)

    lines: List[str] = []
    lines.append("# Phase-3.5 Advancement v2 Result Report")
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
    lines.append("## Main Candidates")
    for row in main_rows:
        lines.append(
            "- `{}`: F1={:.4f}, acc={:.4f}, ΔvsP2={:+.4f}, ΔvsP3={:+.4f}".format(
                row["name"],
                float(row["emotion_macro_f1"]),
                float(row["emotion_acc"]),
                float(row["delta_f1_vs_phase2_fusion_main"]),
                float(row["delta_f1_vs_phase3_fp32_main"]),
            )
        )
    lines.append("")
    lines.append("## Cross Candidates")
    for row in cross_rows:
        ref = row.get("phase2_fusion_f1")
        if ref is None:
            lines.append(
                "- `{}`({}): F1={:.4f}".format(
                    row["name"], row["run"], float(row["emotion_macro_f1"])
                )
            )
        else:
            lines.append(
                "- `{}`({}): F1={:.4f}, phase2={:.4f}, delta={:+.4f}".format(
                    row["name"],
                    row["run"],
                    float(row["emotion_macro_f1"]),
                    float(ref),
                    float(row["delta_f1_vs_phase2_cross"]),
                )
            )
    lines.append("")
    if main_rows:
        lines.append("## Best Main Candidate")
        lines.append(
            "- `{}` (F1={:.4f})".format(main_rows[0]["name"], float(main_rows[0]["emotion_macro_f1"]))
        )
    lines.append("")
    lines.append("## Outputs")
    lines.append("- `derived/reports/phase35_advancement_v2_main_metrics.csv`")
    lines.append("- `derived/reports/phase35_advancement_v2_cross_metrics.csv`")
    lines.append("- `derived/reports/phase35_advancement_v2_results.md`")

    out_md = args.out_dir / "phase35_advancement_v2_results.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "main_csv": str(main_csv),
                "cross_csv": str(cross_csv),
                "report_md": str(out_md),
                "num_main_candidates": len(main_rows),
                "num_cross_candidates": len(cross_rows),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
