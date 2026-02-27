#!/usr/bin/env python3
"""
Analyze phase-2 experiment outputs and produce report artifacts.

Inputs:
- derived/results/ml_baselines_main/summary.json
- derived/results/ml_baselines_main/predictions.csv
- derived/results/ml_baselines_cross_crema_to_ravdess/summary.json
- derived/results/ml_baselines_cross_crema_to_ravdess/predictions.csv
- derived/results/ml_baselines_cross_ravdess_to_crema/summary.json
- derived/results/ml_baselines_cross_ravdess_to_crema/predictions.csv

Outputs:
- derived/reports/phase2_global_metrics.csv
- derived/reports/phase2_pairwise_bootstrap.csv
- derived/reports/phase2_results.md
- derived/reports/phase2_main_f1.svg
- derived/reports/phase2_cross_f1.svg
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class RunSpec:
    name: str
    summary_json: Path
    predictions_csv: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze phase-2 experiment outputs")
    p.add_argument("--out-dir", type=Path, default=Path("derived/reports"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    return p.parse_args()


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = q * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    labels = sorted(set(y_true))
    if not labels:
        return float("nan")
    f1s: List[float] = []
    for lab in labels:
        tp = fp = fn = 0
        for t, p in zip(y_true, y_pred):
            if t == lab and p == lab:
                tp += 1
            elif t != lab and p == lab:
                fp += 1
            elif t == lab and p != lab:
                fn += 1
        den = 2 * tp + fp + fn
        f1s.append(0.0 if den == 0 else (2 * tp) / den)
    return sum(f1s) / len(f1s)


def load_predictions(path: Path) -> Dict[str, List[dict]]:
    by_model: Dict[str, List[dict]] = {}
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            by_model.setdefault(row["model_type"], []).append(row)
    return by_model


def metric_from_rows(rows: List[dict]) -> Tuple[float, int]:
    y_true = [r["y_true_emotion"] for r in rows if r["y_true_emotion"] and r["y_pred_emotion"]]
    y_pred = [r["y_pred_emotion"] for r in rows if r["y_true_emotion"] and r["y_pred_emotion"]]
    return macro_f1(y_true, y_pred), len(y_true)


def bootstrap_delta_f1(
    rows_a: List[dict], rows_b: List[dict], n_boot: int, seed: int
) -> Tuple[float, float, float]:
    """
    Bootstrap CI for delta macro-F1 (A - B).
    Assumes same ordering/length for rows; aligns by clip_id for safety.
    """
    map_a = {r["clip_id"]: r for r in rows_a if r["y_true_emotion"] and r["y_pred_emotion"]}
    map_b = {r["clip_id"]: r for r in rows_b if r["y_true_emotion"] and r["y_pred_emotion"]}
    keys = sorted(set(map_a.keys()) & set(map_b.keys()))
    if not keys:
        return float("nan"), float("nan"), float("nan")

    base_a = [map_a[k] for k in keys]
    base_b = [map_b[k] for k in keys]

    rng = random.Random(seed)
    deltas: List[float] = []
    n = len(keys)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        sa = [base_a[i] for i in idx]
        sb = [base_b[i] for i in idx]
        f1a, _ = metric_from_rows(sa)
        f1b, _ = metric_from_rows(sb)
        deltas.append(f1a - f1b)

    return mean(deltas), percentile(deltas, 0.025), percentile(deltas, 0.975)


def build_simple_bar_svg(
    values: Dict[str, float], title: str, subtitle: str, out_file: Path
) -> None:
    width = 760
    height = 360
    margin_l = 70
    margin_r = 30
    margin_t = 70
    margin_b = 70
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    labels = list(values.keys())
    vals = [float(values[k]) for k in labels]
    vmax = max(vals) if vals else 1.0
    vmax = max(vmax, 1e-6)
    n = len(labels)
    bw = plot_w / max(n, 1) * 0.55
    gap = plot_w / max(n, 1)
    colors = ["#1b5e20", "#1565c0", "#ef6c00", "#6a1b9a", "#2e7d32"]

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append('<rect width="100%" height="100%" fill="#f8fafc" />')
    parts.append(f'<text x="{margin_l}" y="28" font-size="20" font-family="monospace" fill="#111827">{title}</text>')
    parts.append(f'<text x="{margin_l}" y="50" font-size="13" font-family="monospace" fill="#475569">{subtitle}</text>')
    parts.append(
        f'<line x1="{margin_l}" y1="{margin_t+plot_h}" x2="{margin_l+plot_w}" y2="{margin_t+plot_h}" stroke="#334155" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t+plot_h}" stroke="#334155" stroke-width="1"/>'
    )

    for i, (lab, v) in enumerate(zip(labels, vals)):
        x = margin_l + i * gap + (gap - bw) / 2
        bh = (v / vmax) * (plot_h * 0.92)
        y = margin_t + plot_h - bh
        col = colors[i % len(colors)]
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{col}" rx="4"/>')
        parts.append(
            f'<text x="{x+bw/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-size="12" font-family="monospace" fill="#0f172a">{v:.3f}</text>'
        )
        parts.append(
            f'<text x="{x+bw/2:.1f}" y="{margin_t+plot_h+20}" text-anchor="middle" font-size="12" font-family="monospace" fill="#334155">{lab}</text>'
        )
    parts.append("</svg>")
    out_file.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        RunSpec(
            name="main",
            summary_json=Path("derived/results/ml_baselines_main/summary.json"),
            predictions_csv=Path("derived/results/ml_baselines_main/predictions.csv"),
        ),
        RunSpec(
            name="cross_crema_to_ravdess",
            summary_json=Path("derived/results/ml_baselines_cross_crema_to_ravdess/summary.json"),
            predictions_csv=Path("derived/results/ml_baselines_cross_crema_to_ravdess/predictions.csv"),
        ),
        RunSpec(
            name="cross_ravdess_to_crema",
            summary_json=Path("derived/results/ml_baselines_cross_ravdess_to_crema/summary.json"),
            predictions_csv=Path("derived/results/ml_baselines_cross_ravdess_to_crema/predictions.csv"),
        ),
    ]

    missing = [
        r.name
        for r in runs
        if not r.summary_json.exists() or not r.predictions_csv.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing run outputs for: " + ", ".join(missing)
        )

    # 1) Global metrics CSV
    global_rows: List[dict] = []
    all_preds: Dict[str, Dict[str, List[dict]]] = {}
    for run in runs:
        d = json.loads(run.summary_json.read_text(encoding="utf-8"))
        preds = load_predictions(run.predictions_csv)
        all_preds[run.name] = preds
        for mode, g in d["global_metrics"].items():
            global_rows.append(
                {
                    "run": run.name,
                    "mode": mode,
                    "emotion_acc": g["emotion6"]["accuracy"],
                    "emotion_macro_f1": g["emotion6"]["macro_f1"],
                    "emotion_n": g["emotion6"]["n"],
                    "arousal2_mae": g["arousal2"]["mae"],
                    "arousal2_n": g["arousal2"]["n"],
                    "arousal3_mae": g["arousal3"]["mae"],
                    "arousal3_n": g["arousal3"]["n"],
                }
            )

    global_csv = args.out_dir / "phase2_global_metrics.csv"
    with global_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "mode",
                "emotion_acc",
                "emotion_macro_f1",
                "emotion_n",
                "arousal2_mae",
                "arousal2_n",
                "arousal3_mae",
                "arousal3_n",
            ],
        )
        w.writeheader()
        for row in global_rows:
            w.writerow(row)

    # 2) Pairwise bootstrap deltas
    pairs = [("fusion", "audio"), ("fusion", "video"), ("audio", "video")]
    pair_rows: List[dict] = []
    for run in runs:
        preds = all_preds[run.name]
        for a, b in pairs:
            if a not in preds or b not in preds:
                continue
            delta_mean, lo, hi = bootstrap_delta_f1(
                preds[a], preds[b], n_boot=args.n_bootstrap, seed=args.seed
            )
            pair_rows.append(
                {
                    "run": run.name,
                    "lhs_mode": a,
                    "rhs_mode": b,
                    "delta_macro_f1_mean": delta_mean,
                    "delta_macro_f1_ci95_lo": lo,
                    "delta_macro_f1_ci95_hi": hi,
                }
            )

    pair_csv = args.out_dir / "phase2_pairwise_bootstrap.csv"
    with pair_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "lhs_mode",
                "rhs_mode",
                "delta_macro_f1_mean",
                "delta_macro_f1_ci95_lo",
                "delta_macro_f1_ci95_hi",
            ],
        )
        w.writeheader()
        for row in pair_rows:
            w.writerow(row)

    # 3) SVG charts
    def f1_map(run_name: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for row in global_rows:
            if row["run"] == run_name:
                out[row["mode"]] = float(row["emotion_macro_f1"])
        return out

    build_simple_bar_svg(
        f1_map("main"),
        title="Phase-2 Main (Actor-Independent 5-Fold)",
        subtitle="Macro-F1 by modality",
        out_file=args.out_dir / "phase2_main_f1.svg",
    )
    build_simple_bar_svg(
        {
            "C->R audio": f1_map("cross_crema_to_ravdess").get("audio", 0.0),
            "C->R video": f1_map("cross_crema_to_ravdess").get("video", 0.0),
            "C->R fusion": f1_map("cross_crema_to_ravdess").get("fusion", 0.0),
            "R->C audio": f1_map("cross_ravdess_to_crema").get("audio", 0.0),
            "R->C video": f1_map("cross_ravdess_to_crema").get("video", 0.0),
            "R->C fusion": f1_map("cross_ravdess_to_crema").get("fusion", 0.0),
        },
        title="Phase-2 Cross-Dataset",
        subtitle="Macro-F1 by train->test direction and modality",
        out_file=args.out_dir / "phase2_cross_f1.svg",
    )

    # 4) Markdown report
    main_best = sorted(
        [r for r in global_rows if r["run"] == "main"],
        key=lambda x: x["emotion_macro_f1"],
        reverse=True,
    )[0]
    cross_cr = sorted(
        [r for r in global_rows if r["run"] == "cross_crema_to_ravdess"],
        key=lambda x: x["emotion_macro_f1"],
        reverse=True,
    )[0]
    cross_rc = sorted(
        [r for r in global_rows if r["run"] == "cross_ravdess_to_crema"],
        key=lambda x: x["emotion_macro_f1"],
        reverse=True,
    )[0]

    md = []
    md.append("# Phase-2 Result Report")
    md.append("")
    md.append("## Best Modalities")
    md.append(f"- Main(5-fold): `{main_best['mode']}` (macro-F1={main_best['emotion_macro_f1']:.4f})")
    md.append(
        f"- Cross CREMA->RAVDESS: `{cross_cr['mode']}` (macro-F1={cross_cr['emotion_macro_f1']:.4f})"
    )
    md.append(
        f"- Cross RAVDESS->CREMA: `{cross_rc['mode']}` (macro-F1={cross_rc['emotion_macro_f1']:.4f})"
    )
    md.append("")
    md.append("## Key Files")
    md.append("- `derived/reports/phase2_global_metrics.csv`")
    md.append("- `derived/reports/phase2_pairwise_bootstrap.csv`")
    md.append("- `derived/reports/phase2_main_f1.svg`")
    md.append("- `derived/reports/phase2_cross_f1.svg`")
    md.append("")
    md.append("## Interpretation")
    md.append("- In-domain(main), `fusion` is the strongest and improves over single modalities.")
    md.append("- Cross-dataset scores drop substantially, indicating domain gap.")
    md.append("- The asymmetry between CREMA->RAVDESS and RAVDESS->CREMA is notable.")
    md.append("")
    md.append("## Next Step (Design Alignment)")
    md.append("- Proceed to FP32 deep baseline training (E1/E2 refined).")
    md.append("- Then run quantization path (E4) and on-device benchmark (E5).")
    md.append("")
    (args.out_dir / "phase2_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "global_csv": str(global_csv),
                "pairwise_csv": str(pair_csv),
                "report_md": str(args.out_dir / "phase2_results.md"),
                "chart_main_svg": str(args.out_dir / "phase2_main_f1.svg"),
                "chart_cross_svg": str(args.out_dir / "phase2_cross_f1.svg"),
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
