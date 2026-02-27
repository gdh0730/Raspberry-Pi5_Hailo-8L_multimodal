#!/usr/bin/env python3
"""Build majority-vote ensemble predictions and summary from FP32 run CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build vote ensemble from prediction CSV files")
    p.add_argument("--name", type=str, default="vote_ensemble")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--pred-csv",
        type=Path,
        nargs="+",
        required=True,
        help="Prediction CSV paths in tie-break priority order.",
    )
    return p.parse_args()


def load_rows(path: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows[row["clip_id"]] = row
    return rows


def vote_labels(labels: List[str]) -> str:
    c = Counter(labels)
    top = max(c.values())
    tied = {k for k, v in c.items() if v == top}
    for label in labels:
        if label in tied:
            return label
    return sorted(tied)[0]


def vote_optional_int(values: List[str]) -> str:
    vals = [v for v in values if v not in ("", "None", "none", None)]
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sources = [p.resolve() for p in args.pred_csv]
    loaded = [load_rows(p) for p in sources]
    common_ids = sorted(set.intersection(*[set(d.keys()) for d in loaded]))
    if not common_ids:
        raise RuntimeError("No common clip_id across provided prediction files")

    out_rows: List[dict] = []
    y_true_e: List[str] = []
    y_pred_e: List[str] = []
    y_true_a2: List[int] = []
    y_pred_a2: List[int] = []
    y_true_a3: List[int] = []
    y_pred_a3: List[int] = []

    for cid in common_ids:
        picks = [d[cid] for d in loaded]
        first = picks[0]

        emo_labels = [r["y_pred_emotion"] for r in picks]
        emo_pred = vote_labels(emo_labels)

        a2_pred = vote_optional_int([r.get("y_pred_arousal2", "") for r in picks])
        a3_pred = vote_optional_int([r.get("y_pred_arousal3", "") for r in picks])

        row = {
            "model_type": args.name,
            "fold": first.get("fold", ""),
            "clip_id": cid,
            "dataset": first.get("dataset", ""),
            "actor_id": first.get("actor_id", ""),
            "y_true_emotion": first.get("y_true_emotion", ""),
            "y_pred_emotion": emo_pred,
            "y_true_arousal2": first.get("y_true_arousal2", ""),
            "y_pred_arousal2": a2_pred,
            "y_true_arousal3": first.get("y_true_arousal3", ""),
            "y_pred_arousal3": a3_pred,
        }
        out_rows.append(row)

        yt_e = row["y_true_emotion"]
        yp_e = row["y_pred_emotion"]
        if yt_e and yp_e:
            y_true_e.append(yt_e)
            y_pred_e.append(yp_e)

        yt_a2 = row["y_true_arousal2"]
        yp_a2 = row["y_pred_arousal2"]
        if yt_a2 not in ("", None) and yp_a2 not in ("", None):
            y_true_a2.append(int(yt_a2))
            y_pred_a2.append(int(yp_a2))

        yt_a3 = row["y_true_arousal3"]
        yp_a3 = row["y_pred_arousal3"]
        if yt_a3 not in ("", None) and yp_a3 not in ("", None):
            y_true_a3.append(int(yt_a3))
            y_pred_a3.append(int(yp_a3))

    pred_csv = args.out_dir / "predictions.csv"
    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model_type",
                "fold",
                "clip_id",
                "dataset",
                "actor_id",
                "y_true_emotion",
                "y_pred_emotion",
                "y_true_arousal2",
                "y_pred_arousal2",
                "y_true_arousal3",
                "y_pred_arousal3",
            ],
        )
        w.writeheader()
        w.writerows(out_rows)

    summary = {
        "run": {
            "name": args.name,
            "type": "ensemble_vote",
            "sources": [str(p) for p in sources],
            "n_models": len(sources),
            "tie_breaker": "source_order",
        },
        "global": {
            "emotion": {
                "accuracy": float(accuracy_score(y_true_e, y_pred_e)) if y_true_e else None,
                "macro_f1": float(f1_score(y_true_e, y_pred_e, average="macro")) if y_true_e else None,
                "n": len(y_true_e),
            },
            "arousal2": {
                "mae": float(mean_absolute_error(y_true_a2, y_pred_a2)) if y_true_a2 else None,
                "n": len(y_true_a2),
            },
            "arousal3": {
                "mae": float(mean_absolute_error(y_true_a3, y_pred_a3)) if y_true_a3 else None,
                "n": len(y_true_a3),
            },
            "n_predictions": len(out_rows),
        },
        "outputs": {"predictions_csv": str(pred_csv)},
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
