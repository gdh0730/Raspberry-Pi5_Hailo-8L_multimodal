#!/usr/bin/env python3
"""
Analyze phase-3.5 cross-domain adaptation runs (CORAL vs baseline).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze cross-domain adaptation runs")
    p.add_argument("--out-dir", type=Path, default=Path("derived/reports"))
    return p.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_ml_fusion(path: Path) -> Dict[str, Optional[float]]:
    d = load_json(path)
    g = d["global_metrics"]["fusion"]
    return {
        "emotion_acc": g["emotion6"]["accuracy"],
        "emotion_macro_f1": g["emotion6"]["macro_f1"],
        "arousal2_mae": g["arousal2"]["mae"],
        "arousal3_mae": g["arousal3"]["mae"],
    }


def read_phase2_cross_ref() -> Dict[str, float]:
    refs: Dict[str, float] = {}
    path = Path("derived/reports/phase2_global_metrics.csv")
    if not path.exists():
        return refs
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("mode") != "fusion":
                continue
            refs[row["run"]] = float(row["emotion_macro_f1"])
    return refs


def build_rows() -> List[Dict[str, object]]:
    specs = [
        (
            "v3_logreg_baseline_cross_crema_to_ravdess",
            "cross_crema_to_ravdess",
            "none",
            Path("derived/results/ml_baselines_phase35_v3_logreg_cross_crema_to_ravdess/summary.json"),
        ),
        (
            "v3_logreg_baseline_cross_ravdess_to_crema",
            "cross_ravdess_to_crema",
            "none",
            Path("derived/results/ml_baselines_phase35_v3_logreg_cross_ravdess_to_crema/summary.json"),
        ),
        (
            "v4_logreg_coral_cross_crema_to_ravdess",
            "cross_crema_to_ravdess",
            "coral",
            Path("derived/results/ml_baselines_phase35_v4_logreg_coral_cross_crema_to_ravdess/summary.json"),
        ),
        (
            "v4_logreg_coral_cross_ravdess_to_crema",
            "cross_ravdess_to_crema",
            "coral",
            Path("derived/results/ml_baselines_phase35_v4_logreg_coral_cross_ravdess_to_crema/summary.json"),
        ),
        (
            "v5_logreg_baseline_cross_crema_to_ravdess",
            "cross_crema_to_ravdess",
            "none",
            Path("derived/results/ml_baselines_phase35_v5_logreg_cross_crema_to_ravdess/summary.json"),
        ),
        (
            "v5_logreg_baseline_cross_ravdess_to_crema",
            "cross_ravdess_to_crema",
            "none",
            Path("derived/results/ml_baselines_phase35_v5_logreg_cross_ravdess_to_crema/summary.json"),
        ),
        (
            "v5_logreg_coral_cross_crema_to_ravdess",
            "cross_crema_to_ravdess",
            "coral",
            Path("derived/results/ml_baselines_phase35_v5_logreg_coral_cross_crema_to_ravdess/summary.json"),
        ),
        (
            "v5_logreg_coral_cross_ravdess_to_crema",
            "cross_ravdess_to_crema",
            "coral",
            Path("derived/results/ml_baselines_phase35_v5_logreg_coral_cross_ravdess_to_crema/summary.json"),
        ),
        (
            "v8_hubert_logreg_coral_cross_crema_to_ravdess",
            "cross_crema_to_ravdess",
            "coral",
            Path("derived/results/ml_baselines_phase35_v8_hubert_logreg_coral_cross_crema_to_ravdess/summary.json"),
        ),
        (
            "v8_hubert_logreg_coral_cross_ravdess_to_crema",
            "cross_ravdess_to_crema",
            "coral",
            Path("derived/results/ml_baselines_phase35_v8_hubert_logreg_coral_cross_ravdess_to_crema/summary.json"),
        ),
    ]

    rows: List[Dict[str, object]] = []
    for name, run, adapt, path in specs:
        if not path.exists():
            continue
        m = read_ml_fusion(path)
        rows.append(
            {
                "name": name,
                "run": run,
                "domain_adapt": adapt,
                "summary_json": str(path),
                "emotion_acc": m["emotion_acc"],
                "emotion_macro_f1": m["emotion_macro_f1"],
                "arousal2_mae": m["arousal2_mae"],
                "arousal3_mae": m["arousal3_mae"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows()
    phase2_ref = read_phase2_cross_ref()

    baseline_by_run: Dict[str, float] = {}
    for row in rows:
        if row["domain_adapt"] == "none" and row["emotion_macro_f1"] is not None:
            baseline_by_run[str(row["run"])] = float(row["emotion_macro_f1"])

    for row in rows:
        run = str(row["run"])
        f1 = row["emotion_macro_f1"]
        p2 = phase2_ref.get(run)
        base = baseline_by_run.get(run)
        row["phase2_fusion_f1"] = p2
        row["delta_f1_vs_phase2"] = None if (p2 is None or f1 is None) else float(f1) - p2
        row["delta_f1_vs_v3_baseline"] = None if (base is None or f1 is None) else float(f1) - base

    rows.sort(key=lambda x: (x["run"], x["domain_adapt"]))

    out_csv = args.out_dir / "phase35_cross_domain_adapt_metrics.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
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
                "delta_f1_vs_phase2",
                "delta_f1_vs_v3_baseline",
            ],
        )
        w.writeheader()
        for row in rows:
            w.writerow(row)

    lines: List[str] = []
    lines.append("# Phase-3.5 Cross-Domain Adaptation Report")
    lines.append("")
    lines.append("## Results")
    for row in rows:
        f1 = row["emotion_macro_f1"]
        acc = row["emotion_acc"]
        d2 = row["delta_f1_vs_phase2"]
        db = row["delta_f1_vs_v3_baseline"]
        lines.append(
            "- `{}` [{}|{}]: F1={:.4f}, acc={:.4f}, ΔvsP2={}, ΔvsV3={}".format(
                row["name"],
                row["run"],
                row["domain_adapt"],
                float(f1) if f1 is not None else float("nan"),
                float(acc) if acc is not None else float("nan"),
                "n/a" if d2 is None else f"{float(d2):+0.4f}",
                "n/a" if db is None else f"{float(db):+0.4f}",
            )
        )
    lines.append("")

    for run in ["cross_crema_to_ravdess", "cross_ravdess_to_crema"]:
        subset = [r for r in rows if r["run"] == run and r["emotion_macro_f1"] is not None]
        if not subset:
            continue
        best = max(subset, key=lambda x: float(x["emotion_macro_f1"]))
        lines.append(
            "## Best for `{}`".format(run)
        )
        lines.append(
            "- `{}` ({}) F1={:.4f}".format(
                best["name"], best["domain_adapt"], float(best["emotion_macro_f1"])
            )
        )
        lines.append("")

    lines.append("## Outputs")
    lines.append("- `derived/reports/phase35_cross_domain_adapt_metrics.csv`")
    lines.append("- `derived/reports/phase35_cross_domain_adapt_results.md`")

    out_md = args.out_dir / "phase35_cross_domain_adapt_results.md"
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
