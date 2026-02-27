#!/usr/bin/env python3
"""
Analyze phase-3 FP32 outputs and compare with phase-2 fusion baseline.

Inputs:
- derived/results/fp32_multitask_main/{summary.json,predictions.csv}
- derived/results/fp32_multitask_cross_crema_to_ravdess/{summary.json,predictions.csv}
- derived/results/fp32_multitask_cross_ravdess_to_crema/{summary.json,predictions.csv}
- derived/results/ml_baselines_main/{summary.json,predictions.csv}
- derived/results/ml_baselines_cross_crema_to_ravdess/{summary.json,predictions.csv}
- derived/results/ml_baselines_cross_ravdess_to_crema/{summary.json,predictions.csv}

Outputs:
- derived/reports/phase3_global_metrics.csv
- derived/reports/phase3_vs_phase2_bootstrap.csv
- derived/reports/phase3_results.md
- derived/reports/phase3_emotion_f1.svg
- derived/reports/phase3_vs_phase2_delta_f1.svg
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Tuple


@dataclass
class RunSpec:
    name: str
    phase3_summary: Path
    phase3_predictions: Path
    phase2_summary: Path
    phase2_predictions: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze phase-3 FP32 outputs")
    p.add_argument("--out-dir", type=Path, default=Path("derived/reports"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    return p.parse_args()


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    r = q * (len(s) - 1)
    lo = int(r)
    hi = min(lo + 1, len(s) - 1)
    frac = r - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


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
    return float(sum(f1s) / len(f1s))


def load_pred_rows(path: Path, model_type: str) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("model_type") == model_type:
                out.append(row)
    return out


def phase3_global_metrics(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    g = d["global"]
    return {
        "emotion_acc": g["emotion"]["accuracy"],
        "emotion_macro_f1": g["emotion"]["macro_f1"],
        "emotion_n": g["emotion"]["n"],
        "arousal2_mae": g["arousal2"]["mae"],
        "arousal2_n": g["arousal2"]["n"],
        "arousal3_mae": g["arousal3"]["mae"],
        "arousal3_n": g["arousal3"]["n"],
    }


def phase2_fusion_metrics(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    g = d["global_metrics"]["fusion"]
    return {
        "emotion_acc": g["emotion6"]["accuracy"],
        "emotion_macro_f1": g["emotion6"]["macro_f1"],
        "emotion_n": g["emotion6"]["n"],
        "arousal2_mae": g["arousal2"]["mae"],
        "arousal2_n": g["arousal2"]["n"],
        "arousal3_mae": g["arousal3"]["mae"],
        "arousal3_n": g["arousal3"]["n"],
    }


def bootstrap_delta_macro_f1(
    phase3_rows: List[dict],
    phase2_rows: List[dict],
    n_boot: int,
    seed: int,
) -> Tuple[float, float, float, int]:
    p3 = {
        r["clip_id"]: r
        for r in phase3_rows
        if r.get("y_true_emotion") and r.get("y_pred_emotion")
    }
    p2 = {
        r["clip_id"]: r
        for r in phase2_rows
        if r.get("y_true_emotion") and r.get("y_pred_emotion")
    }
    keys = sorted(set(p3.keys()) & set(p2.keys()))
    n = len(keys)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0

    rng = random.Random(seed)
    deltas: List[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        y_true = [p3[keys[i]]["y_true_emotion"] for i in idx]
        y3 = [p3[keys[i]]["y_pred_emotion"] for i in idx]
        y2 = [p2[keys[i]]["y_pred_emotion"] for i in idx]
        deltas.append(macro_f1(y_true, y3) - macro_f1(y_true, y2))
    return mean(deltas), percentile(deltas, 0.025), percentile(deltas, 0.975), n


def bar_svg(values: Dict[str, float], title: str, subtitle: str, out_path: Path) -> None:
    width = 820
    height = 380
    m_l = 75
    m_r = 25
    m_t = 72
    m_b = 80
    pw = width - m_l - m_r
    ph = height - m_t - m_b

    labels = list(values.keys())
    vals = [float(values[k]) for k in labels]
    vmax = max(vals) if vals else 1.0
    vmax = max(vmax, 1e-6)
    n = len(labels) if labels else 1
    gap = pw / n
    bw = gap * 0.55
    colors = ["#0f766e", "#0369a1", "#a16207", "#166534"]

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append('<rect width="100%" height="100%" fill="#f8fafc"/>')
    parts.append(f'<text x="{m_l}" y="30" font-size="20" font-family="monospace" fill="#0f172a">{title}</text>')
    parts.append(f'<text x="{m_l}" y="52" font-size="13" font-family="monospace" fill="#475569">{subtitle}</text>')
    parts.append(
        f'<line x1="{m_l}" y1="{m_t+ph}" x2="{m_l+pw}" y2="{m_t+ph}" stroke="#334155" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{m_l}" y1="{m_t}" x2="{m_l}" y2="{m_t+ph}" stroke="#334155" stroke-width="1"/>'
    )

    for i, (lab, v) in enumerate(zip(labels, vals)):
        x = m_l + i * gap + (gap - bw) / 2
        bh = (v / vmax) * (ph * 0.92)
        y = m_t + ph - bh
        c = colors[i % len(colors)]
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{c}" rx="4"/>')
        parts.append(
            f'<text x="{x+bw/2:.1f}" y="{y-7:.1f}" text-anchor="middle" font-size="12" font-family="monospace" fill="#111827">{v:.3f}</text>'
        )
        parts.append(
            f'<text x="{x+bw/2:.1f}" y="{m_t+ph+22}" text-anchor="middle" font-size="12" font-family="monospace" fill="#334155">{lab}</text>'
        )
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def delta_svg(values: Dict[str, float], title: str, subtitle: str, out_path: Path) -> None:
    width = 820
    height = 380
    m_l = 75
    m_r = 25
    m_t = 72
    m_b = 80
    pw = width - m_l - m_r
    ph = height - m_t - m_b

    labels = list(values.keys())
    vals = [float(values[k]) for k in labels]
    vmax = max([abs(v) for v in vals] + [1e-6])
    y0 = m_t + ph / 2
    n = len(labels) if labels else 1
    gap = pw / n
    bw = gap * 0.5

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append('<rect width="100%" height="100%" fill="#f8fafc"/>')
    parts.append(f'<text x="{m_l}" y="30" font-size="20" font-family="monospace" fill="#0f172a">{title}</text>')
    parts.append(f'<text x="{m_l}" y="52" font-size="13" font-family="monospace" fill="#475569">{subtitle}</text>')
    parts.append(f'<line x1="{m_l}" y1="{y0:.1f}" x2="{m_l+pw}" y2="{y0:.1f}" stroke="#334155" stroke-width="1"/>')
    parts.append(
        f'<line x1="{m_l}" y1="{m_t}" x2="{m_l}" y2="{m_t+ph}" stroke="#334155" stroke-width="1"/>'
    )

    for i, (lab, v) in enumerate(zip(labels, vals)):
        x = m_l + i * gap + (gap - bw) / 2
        bh = (abs(v) / vmax) * (ph * 0.45)
        y = y0 - bh if v >= 0 else y0
        color = "#166534" if v >= 0 else "#b91c1c"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{color}" rx="4"/>')
        ty = y - 7 if v >= 0 else y + bh + 15
        parts.append(
            f'<text x="{x+bw/2:.1f}" y="{ty:.1f}" text-anchor="middle" font-size="12" font-family="monospace" fill="#111827">{v:+.3f}</text>'
        )
        parts.append(
            f'<text x="{x+bw/2:.1f}" y="{m_t+ph+22}" text-anchor="middle" font-size="12" font-family="monospace" fill="#334155">{lab}</text>'
        )
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        RunSpec(
            name="main",
            phase3_summary=Path("derived/results/fp32_multitask_main/summary.json"),
            phase3_predictions=Path("derived/results/fp32_multitask_main/predictions.csv"),
            phase2_summary=Path("derived/results/ml_baselines_main/summary.json"),
            phase2_predictions=Path("derived/results/ml_baselines_main/predictions.csv"),
        ),
        RunSpec(
            name="cross_crema_to_ravdess",
            phase3_summary=Path("derived/results/fp32_multitask_cross_crema_to_ravdess/summary.json"),
            phase3_predictions=Path("derived/results/fp32_multitask_cross_crema_to_ravdess/predictions.csv"),
            phase2_summary=Path("derived/results/ml_baselines_cross_crema_to_ravdess/summary.json"),
            phase2_predictions=Path("derived/results/ml_baselines_cross_crema_to_ravdess/predictions.csv"),
        ),
        RunSpec(
            name="cross_ravdess_to_crema",
            phase3_summary=Path("derived/results/fp32_multitask_cross_ravdess_to_crema/summary.json"),
            phase3_predictions=Path("derived/results/fp32_multitask_cross_ravdess_to_crema/predictions.csv"),
            phase2_summary=Path("derived/results/ml_baselines_cross_ravdess_to_crema/summary.json"),
            phase2_predictions=Path("derived/results/ml_baselines_cross_ravdess_to_crema/predictions.csv"),
        ),
    ]

    missing = [
        r.name
        for r in runs
        if not r.phase3_summary.exists()
        or not r.phase3_predictions.exists()
        or not r.phase2_summary.exists()
        or not r.phase2_predictions.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing result files for runs: " + ", ".join(missing))

    global_rows: List[dict] = []
    boot_rows: List[dict] = []

    for i, run in enumerate(runs):
        p3 = phase3_global_metrics(run.phase3_summary)
        p2 = phase2_fusion_metrics(run.phase2_summary)
        global_rows.append(
            {
                "run": run.name,
                "phase3_emotion_acc": p3["emotion_acc"],
                "phase3_emotion_macro_f1": p3["emotion_macro_f1"],
                "phase3_emotion_n": p3["emotion_n"],
                "phase3_arousal2_mae": p3["arousal2_mae"],
                "phase3_arousal2_n": p3["arousal2_n"],
                "phase3_arousal3_mae": p3["arousal3_mae"],
                "phase3_arousal3_n": p3["arousal3_n"],
                "phase2_fusion_emotion_acc": p2["emotion_acc"],
                "phase2_fusion_emotion_macro_f1": p2["emotion_macro_f1"],
                "phase2_fusion_emotion_n": p2["emotion_n"],
                "phase2_fusion_arousal2_mae": p2["arousal2_mae"],
                "phase2_fusion_arousal2_n": p2["arousal2_n"],
                "phase2_fusion_arousal3_mae": p2["arousal3_mae"],
                "phase2_fusion_arousal3_n": p2["arousal3_n"],
                "delta_emotion_macro_f1": p3["emotion_macro_f1"] - p2["emotion_macro_f1"],
                "delta_arousal2_mae": (
                    None
                    if p3["arousal2_mae"] is None or p2["arousal2_mae"] is None
                    else p3["arousal2_mae"] - p2["arousal2_mae"]
                ),
                "delta_arousal3_mae": (
                    None
                    if p3["arousal3_mae"] is None or p2["arousal3_mae"] is None
                    else p3["arousal3_mae"] - p2["arousal3_mae"]
                ),
            }
        )

        p3_rows = load_pred_rows(run.phase3_predictions, "fp32_fusion")
        p2_rows = load_pred_rows(run.phase2_predictions, "fusion")
        d_mean, d_lo, d_hi, n_aligned = bootstrap_delta_macro_f1(
            p3_rows, p2_rows, n_boot=args.n_bootstrap, seed=args.seed + i * 17
        )
        boot_rows.append(
            {
                "run": run.name,
                "metric": "emotion_macro_f1",
                "delta_mean_phase3_minus_phase2fusion": d_mean,
                "ci95_low": d_lo,
                "ci95_high": d_hi,
                "n_aligned": n_aligned,
            }
        )

    global_csv = args.out_dir / "phase3_global_metrics.csv"
    with global_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "phase3_emotion_acc",
                "phase3_emotion_macro_f1",
                "phase3_emotion_n",
                "phase3_arousal2_mae",
                "phase3_arousal2_n",
                "phase3_arousal3_mae",
                "phase3_arousal3_n",
                "phase2_fusion_emotion_acc",
                "phase2_fusion_emotion_macro_f1",
                "phase2_fusion_emotion_n",
                "phase2_fusion_arousal2_mae",
                "phase2_fusion_arousal2_n",
                "phase2_fusion_arousal3_mae",
                "phase2_fusion_arousal3_n",
                "delta_emotion_macro_f1",
                "delta_arousal2_mae",
                "delta_arousal3_mae",
            ],
        )
        w.writeheader()
        for row in global_rows:
            w.writerow(row)

    boot_csv = args.out_dir / "phase3_vs_phase2_bootstrap.csv"
    with boot_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "metric",
                "delta_mean_phase3_minus_phase2fusion",
                "ci95_low",
                "ci95_high",
                "n_aligned",
            ],
        )
        w.writeheader()
        for row in boot_rows:
            w.writerow(row)

    p3_f1 = {row["run"]: float(row["phase3_emotion_macro_f1"]) for row in global_rows}
    delta_f1 = {row["run"]: float(row["delta_emotion_macro_f1"]) for row in global_rows}
    bar_svg(
        p3_f1,
        title="Phase-3 FP32 Macro-F1",
        subtitle="main + cross runs",
        out_path=args.out_dir / "phase3_emotion_f1.svg",
    )
    delta_svg(
        delta_f1,
        title="Delta Macro-F1 (Phase-3 FP32 - Phase-2 fusion)",
        subtitle="positive means phase-3 is better",
        out_path=args.out_dir / "phase3_vs_phase2_delta_f1.svg",
    )

    best_run = max(global_rows, key=lambda r: float(r["phase3_emotion_macro_f1"]))
    md_lines: List[str] = []
    md_lines.append("# Phase-3 FP32 Result Report")
    md_lines.append("")
    md_lines.append("## Best FP32 Run")
    md_lines.append(
        f"- `{best_run['run']}` (macro-F1={float(best_run['phase3_emotion_macro_f1']):.4f})"
    )
    md_lines.append("")
    md_lines.append("## Key Files")
    md_lines.append("- `derived/reports/phase3_global_metrics.csv`")
    md_lines.append("- `derived/reports/phase3_vs_phase2_bootstrap.csv`")
    md_lines.append("- `derived/reports/phase3_emotion_f1.svg`")
    md_lines.append("- `derived/reports/phase3_vs_phase2_delta_f1.svg`")
    md_lines.append("")
    md_lines.append("## Phase-2 Fusion 대비 요약")
    for row in global_rows:
        md_lines.append(
            "- `{run}`: Δmacro-F1={df:+.4f}, "
            "phase3={p3:.4f}, phase2_fusion={p2:.4f}".format(
                run=row["run"],
                df=float(row["delta_emotion_macro_f1"]),
                p3=float(row["phase3_emotion_macro_f1"]),
                p2=float(row["phase2_fusion_emotion_macro_f1"]),
            )
        )
    md_lines.append("")
    md_lines.append("## Interpretation")
    md_lines.append("- Main(run=main)에서는 phase-2 fusion과 성능 차이를 직접 비교해 기준선 유지/개선 여부를 판단한다.")
    md_lines.append("- Cross run 결과는 도메인 갭(E3)을 정량화하며, 방향별 비대칭을 확인한다.")
    md_lines.append("- 다음 단계(E4)는 phase-3 체크포인트를 기준으로 PTQ/QAT 경량화 비교 실험이다.")
    (args.out_dir / "phase3_results.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "global_csv": str(global_csv),
                "bootstrap_csv": str(boot_csv),
                "report_md": str(args.out_dir / "phase3_results.md"),
                "f1_svg": str(args.out_dir / "phase3_emotion_f1.svg"),
                "delta_svg": str(args.out_dir / "phase3_vs_phase2_delta_f1.svg"),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
