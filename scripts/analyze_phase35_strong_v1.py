#!/usr/bin/env python3
"""
Analyze phase-3.5 strong-v1 runs (audio-pretrained cache_v4).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze phase35 strong-v1 runs")
    p.add_argument("--out-dir", type=Path, default=Path("derived/reports"))
    return p.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_ml(path: Path, mode: str = "fusion") -> Dict[str, Optional[float]]:
    d = load_json(path)
    g = d["global_metrics"][mode]
    return {
        "emotion_acc": g["emotion6"]["accuracy"],
        "emotion_macro_f1": g["emotion6"]["macro_f1"],
        "arousal2_mae": g["arousal2"]["mae"],
        "arousal3_mae": g["arousal3"]["mae"],
    }


def read_phase2_main() -> float:
    d = load_json(Path("derived/results/ml_baselines_main/summary.json"))
    return float(d["global_metrics"]["fusion"]["emotion6"]["macro_f1"])


def read_phase3_main() -> float:
    d = load_json(Path("derived/results/fp32_multitask_main/summary.json"))
    return float(d["global"]["emotion"]["macro_f1"])


def read_phase2_cross_ref() -> Dict[str, float]:
    refs: Dict[str, float] = {}
    p = Path("derived/reports/phase2_global_metrics.csv")
    if not p.exists():
        return refs
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("mode") != "fusion":
                continue
            refs[row["run"]] = float(row["emotion_macro_f1"])
    return refs


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    p2_main_f1 = read_phase2_main()
    p3_main_f1 = read_phase3_main()

    main_specs = [
        ("v5_logreg_main", Path("derived/results/ml_baselines_phase35_v5_logreg_main/summary.json")),
        ("v5_linsvm_main", Path("derived/results/ml_baselines_phase35_v5_linsvm_main/summary.json")),
        ("v3_rbfsvm_main_ref", Path("derived/results/ml_baselines_phase35_v3_rbfsvm_main/summary.json")),
    ]
    main_rows: List[Dict[str, object]] = []
    for name, path in main_specs:
        if not path.exists():
            continue
        m = read_ml(path, mode="fusion")
        row = {
            "name": name,
            "summary_json": str(path),
            "emotion_acc": m["emotion_acc"],
            "emotion_macro_f1": m["emotion_macro_f1"],
            "arousal2_mae": m["arousal2_mae"],
            "arousal3_mae": m["arousal3_mae"],
            "delta_f1_vs_phase2_main": None if m["emotion_macro_f1"] is None else float(m["emotion_macro_f1"]) - p2_main_f1,
            "delta_f1_vs_phase3_main": None if m["emotion_macro_f1"] is None else float(m["emotion_macro_f1"]) - p3_main_f1,
        }
        main_rows.append(row)
    main_rows.sort(key=lambda x: float(x["emotion_macro_f1"]), reverse=True)

    main_csv = args.out_dir / "phase35_strong_v1_main_metrics.csv"
    with main_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "summary_json",
                "emotion_acc",
                "emotion_macro_f1",
                "arousal2_mae",
                "arousal3_mae",
                "delta_f1_vs_phase2_main",
                "delta_f1_vs_phase3_main",
            ],
        )
        w.writeheader()
        for row in main_rows:
            w.writerow(row)

    cross_specs = [
        ("v5_logreg_cross_crema_to_ravdess", "cross_crema_to_ravdess", "none", Path("derived/results/ml_baselines_phase35_v5_logreg_cross_crema_to_ravdess/summary.json")),
        ("v5_logreg_cross_ravdess_to_crema", "cross_ravdess_to_crema", "none", Path("derived/results/ml_baselines_phase35_v5_logreg_cross_ravdess_to_crema/summary.json")),
        ("v5_logreg_coral_cross_crema_to_ravdess", "cross_crema_to_ravdess", "coral", Path("derived/results/ml_baselines_phase35_v5_logreg_coral_cross_crema_to_ravdess/summary.json")),
        ("v5_logreg_coral_cross_ravdess_to_crema", "cross_ravdess_to_crema", "coral", Path("derived/results/ml_baselines_phase35_v5_logreg_coral_cross_ravdess_to_crema/summary.json")),
        ("v5_linsvm_cross_crema_to_ravdess", "cross_crema_to_ravdess", "none", Path("derived/results/ml_baselines_phase35_v5_linsvm_cross_crema_to_ravdess/summary.json")),
        ("v5_linsvm_cross_ravdess_to_crema", "cross_ravdess_to_crema", "none", Path("derived/results/ml_baselines_phase35_v5_linsvm_cross_ravdess_to_crema/summary.json")),
        ("v5_linsvm_coral_cross_crema_to_ravdess", "cross_crema_to_ravdess", "coral", Path("derived/results/ml_baselines_phase35_v5_linsvm_coral_cross_crema_to_ravdess/summary.json")),
        ("v5_linsvm_coral_cross_ravdess_to_crema", "cross_ravdess_to_crema", "coral", Path("derived/results/ml_baselines_phase35_v5_linsvm_coral_cross_ravdess_to_crema/summary.json")),
        ("v4_logreg_coral_ref_cross_crema_to_ravdess", "cross_crema_to_ravdess", "coral_ref", Path("derived/results/ml_baselines_phase35_v4_logreg_coral_cross_crema_to_ravdess/summary.json")),
        ("v4_logreg_coral_ref_cross_ravdess_to_crema", "cross_ravdess_to_crema", "coral_ref", Path("derived/results/ml_baselines_phase35_v4_logreg_coral_cross_ravdess_to_crema/summary.json")),
    ]
    phase2_cross = read_phase2_cross_ref()

    cross_rows: List[Dict[str, object]] = []
    baseline_by_key: Dict[str, float] = {}
    for name, run, adapt, path in cross_specs:
        if not path.exists():
            continue
        m = read_ml(path, mode="fusion")
        row = {
            "name": name,
            "run": run,
            "domain_adapt": adapt,
            "summary_json": str(path),
            "emotion_acc": m["emotion_acc"],
            "emotion_macro_f1": m["emotion_macro_f1"],
            "arousal2_mae": m["arousal2_mae"],
            "arousal3_mae": m["arousal3_mae"],
        }
        cross_rows.append(row)
        if adapt == "none" and m["emotion_macro_f1"] is not None:
            key = f"{run}|{name.split('_')[1]}"  # run|logreg or run|linsvm
            baseline_by_key[key] = float(m["emotion_macro_f1"])

    for row in cross_rows:
        run = str(row["run"])
        f1 = row["emotion_macro_f1"]
        p2 = phase2_cross.get(run)
        row["phase2_fusion_f1"] = p2
        row["delta_f1_vs_phase2_cross"] = None if (p2 is None or f1 is None) else float(f1) - p2

        key_model = "logreg" if "logreg" in str(row["name"]) else ("linsvm" if "linsvm" in str(row["name"]) else None)
        if key_model is None:
            row["delta_f1_vs_v5_none"] = None
        else:
            key = f"{run}|{key_model}"
            b = baseline_by_key.get(key)
            row["delta_f1_vs_v5_none"] = None if (b is None or f1 is None) else float(f1) - b

    cross_rows.sort(key=lambda x: (x["run"], x["domain_adapt"], x["name"]))

    cross_csv = args.out_dir / "phase35_strong_v1_cross_metrics.csv"
    with cross_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "run",
                "domain_adapt",
                "summary_json",
                "emotion_acc",
                "emotion_macro_f1",
                "arousal2_mae",
                "arousal3_mae",
                "phase2_fusion_f1",
                "delta_f1_vs_phase2_cross",
                "delta_f1_vs_v5_none",
            ],
        )
        w.writeheader()
        for row in cross_rows:
            w.writerow(row)

    lines: List[str] = []
    lines.append("# Phase-3.5 Strong v1 Report")
    lines.append("")
    lines.append("## Main Candidates")
    for row in main_rows:
        lines.append(
            "- `{}`: F1={:.4f}, acc={:.4f}, ΔvsP2={:+.4f}, ΔvsP3={:+.4f}".format(
                row["name"],
                float(row["emotion_macro_f1"]),
                float(row["emotion_acc"]),
                float(row["delta_f1_vs_phase2_main"]),
                float(row["delta_f1_vs_phase3_main"]),
            )
        )
    lines.append("")
    lines.append("## Cross Candidates")
    for row in cross_rows:
        d2 = row["delta_f1_vs_phase2_cross"]
        d5 = row["delta_f1_vs_v5_none"]
        lines.append(
            "- `{}` [{}|{}]: F1={:.4f}, acc={:.4f}, ΔvsP2={}, ΔvsV5None={}".format(
                row["name"],
                row["run"],
                row["domain_adapt"],
                float(row["emotion_macro_f1"]),
                float(row["emotion_acc"]),
                "n/a" if d2 is None else f"{float(d2):+0.4f}",
                "n/a" if d5 is None else f"{float(d5):+0.4f}",
            )
        )
    lines.append("")
    if main_rows:
        best_main = max(main_rows, key=lambda x: float(x["emotion_macro_f1"]))
        lines.append("## Best Main")
        lines.append(
            "- `{}` F1={:.4f}".format(best_main["name"], float(best_main["emotion_macro_f1"]))
        )
        lines.append("")
    for run in ["cross_crema_to_ravdess", "cross_ravdess_to_crema"]:
        sub = [r for r in cross_rows if r["run"] == run]
        if not sub:
            continue
        best = max(sub, key=lambda x: float(x["emotion_macro_f1"]))
        lines.append(f"## Best Cross `{run}`")
        lines.append(
            "- `{}` ({}) F1={:.4f}".format(
                best["name"], best["domain_adapt"], float(best["emotion_macro_f1"])
            )
        )
        lines.append("")
    lines.append("## Outputs")
    lines.append("- `derived/reports/phase35_strong_v1_main_metrics.csv`")
    lines.append("- `derived/reports/phase35_strong_v1_cross_metrics.csv`")
    lines.append("- `derived/reports/phase35_strong_v1_results.md`")

    out_md = args.out_dir / "phase35_strong_v1_results.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "main_csv": str(main_csv),
                "cross_csv": str(cross_csv),
                "report_md": str(out_md),
                "num_main": len(main_rows),
                "num_cross": len(cross_rows),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
