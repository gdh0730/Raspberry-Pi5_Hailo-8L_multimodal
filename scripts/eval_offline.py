#!/usr/bin/env python3
"""Evaluate prediction CSV and produce metrics + bootstrap CI."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

from research_metrics import accuracy_score, bootstrap_ci, macro_f1_score, mean_absolute_error


def parse_optional_int(value: str) -> Optional[int]:
    value = value.strip()
    if value == "" or value.lower() == "none":
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate offline prediction CSV")
    p.add_argument("--pred-csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, default=Path("derived/results/eval_offline_summary.json"))
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    with args.pred_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    y_true_e = [r.get("y_true_emotion") or None for r in rows]
    y_pred_e = [r.get("y_pred_emotion") or None for r in rows]
    valid_e = [(t, p) for t, p in zip(y_true_e, y_pred_e) if t is not None and p is not None]
    yt_e = [t for t, _ in valid_e]
    yp_e = [p for _, p in valid_e]

    if yt_e:
        acc = accuracy_score(yt_e, yp_e)
        f1 = macro_f1_score(yt_e, yp_e)
        acc_ci = bootstrap_ci(
            yt_e, yp_e, accuracy_score, n_boot=args.n_bootstrap, seed=args.seed
        )
        f1_ci = bootstrap_ci(
            yt_e, yp_e, macro_f1_score, n_boot=args.n_bootstrap, seed=args.seed
        )
    else:
        acc = None
        f1 = None
        acc_ci = (None, None)
        f1_ci = (None, None)

    y_true_a2 = [parse_optional_int(r.get("y_true_arousal2", "")) for r in rows]
    y_pred_a2 = [parse_optional_int(r.get("y_pred_arousal2", "")) for r in rows]
    valid_a2 = [(t, p) for t, p in zip(y_true_a2, y_pred_a2) if t is not None and p is not None]
    if valid_a2:
        yt = [t for t, _ in valid_a2]
        yp = [p for _, p in valid_a2]
        mae_a2 = mean_absolute_error(yt, yp)
        mae_a2_ci = bootstrap_ci(
            yt, yp, mean_absolute_error, n_boot=args.n_bootstrap, seed=args.seed
        )
    else:
        mae_a2 = None
        mae_a2_ci = (None, None)

    y_true_a3 = [parse_optional_int(r.get("y_true_arousal3", "")) for r in rows]
    y_pred_a3 = [parse_optional_int(r.get("y_pred_arousal3", "")) for r in rows]
    valid_a3 = [(t, p) for t, p in zip(y_true_a3, y_pred_a3) if t is not None and p is not None]
    if valid_a3:
        yt = [t for t, _ in valid_a3]
        yp = [p for _, p in valid_a3]
        mae_a3 = mean_absolute_error(yt, yp)
        mae_a3_ci = bootstrap_ci(
            yt, yp, mean_absolute_error, n_boot=args.n_bootstrap, seed=args.seed
        )
    else:
        mae_a3 = None
        mae_a3_ci = (None, None)

    result = {
        "input": {"pred_csv": str(args.pred_csv), "n_rows": len(rows)},
        "emotion6": {
            "accuracy": acc,
            "accuracy_ci95": list(acc_ci),
            "macro_f1": f1,
            "macro_f1_ci95": list(f1_ci),
            "n": len(valid_e),
        },
        "arousal2": {"mae": mae_a2, "mae_ci95": list(mae_a2_ci), "n": len(valid_a2)},
        "arousal3": {"mae": mae_a3, "mae_ci95": list(mae_a3_ci), "n": len(valid_a3)},
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
