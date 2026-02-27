#!/usr/bin/env python3
"""
Run B0 majority baseline on actor-independent folds.

This script is dependency-light (stdlib only) so it works in the current
environment without PyTorch/NumPy installation.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from research_metrics import (
    accuracy_score,
    bootstrap_ci,
    macro_f1_score,
    majority_vote,
    mean_absolute_error,
)


@dataclass
class PredRow:
    clip_id: str
    fold: int
    dataset: str
    actor_id: str
    modality: str
    y_true_emotion: Optional[str]
    y_pred_emotion: Optional[str]
    y_true_arousal2: Optional[int]
    y_pred_arousal2: Optional[int]
    y_true_arousal3: Optional[int]
    y_pred_arousal3: Optional[int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B0 majority baseline")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("derived/manifests/manifest_multimodal_common6_av.jsonl"),
        help="Manifest JSONL path",
    )
    p.add_argument(
        "--fold-dir",
        type=Path,
        default=Path("derived/splits/groupkfold5_all"),
        help="Directory containing fold_{k}_{train,val}.txt",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("derived/results/b0_majority"),
        help="Output directory",
    )
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_manifest(path: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows[obj["clip_id"]] = obj
    return rows


def load_clip_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def eval_cls(
    y_true: Sequence[Optional[str]],
    y_pred: Sequence[Optional[str]],
    n_bootstrap: int,
    seed: int,
) -> dict:
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t is not None and p is not None]
    if not pairs:
        return {
            "accuracy": None,
            "accuracy_ci95": [None, None],
            "macro_f1": None,
            "macro_f1_ci95": [None, None],
            "n": 0,
        }
    yt = [t for t, _ in pairs]
    yp = [p for _, p in pairs]
    acc = accuracy_score(yt, yp)
    f1 = macro_f1_score(yt, yp)
    acc_lo, acc_hi = bootstrap_ci(
        yt, yp, accuracy_score, n_boot=n_bootstrap, seed=seed
    )
    f1_lo, f1_hi = bootstrap_ci(
        yt, yp, macro_f1_score, n_boot=n_bootstrap, seed=seed
    )
    return {
        "accuracy": acc,
        "accuracy_ci95": [acc_lo, acc_hi],
        "macro_f1": f1,
        "macro_f1_ci95": [f1_lo, f1_hi],
        "n": len(yt),
    }


def eval_arousal(
    y_true: Sequence[Optional[int]],
    y_pred: Sequence[Optional[int]],
    n_bootstrap: int,
    seed: int,
) -> dict:
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t is not None and p is not None]
    if not pairs:
        return {"mae": None, "mae_ci95": [None, None], "n": 0}
    yt = [t for t, _ in pairs]
    yp = [p for _, p in pairs]
    mae = mean_absolute_error(yt, yp)
    lo, hi = bootstrap_ci(yt, yp, mean_absolute_error, n_boot=n_bootstrap, seed=seed)
    return {"mae": mae, "mae_ci95": [lo, hi], "n": len(yt)}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)

    pred_rows: List[PredRow] = []
    folds_summary = []

    for fold in range(args.num_folds):
        train_file = args.fold_dir / f"fold_{fold}_train.txt"
        val_file = args.fold_dir / f"fold_{fold}_val.txt"
        if not train_file.exists() or not val_file.exists():
            raise FileNotFoundError(f"Missing split files for fold {fold} in {args.fold_dir}")

        train_ids = [cid for cid in load_clip_ids(train_file) if cid in manifest]
        val_ids = [cid for cid in load_clip_ids(val_file) if cid in manifest]

        train_rows = [manifest[cid] for cid in train_ids]
        val_rows = [manifest[cid] for cid in val_ids]

        maj_emotion = majority_vote(r.get("emotion6") for r in train_rows)
        maj_arousal2 = majority_vote(r.get("arousal2") for r in train_rows)
        maj_arousal3 = majority_vote(r.get("arousal3") for r in train_rows)

        fold_preds: List[PredRow] = []
        for r in val_rows:
            row = PredRow(
                clip_id=r["clip_id"],
                fold=fold,
                dataset=r.get("dataset", ""),
                actor_id=r.get("actor_id", ""),
                modality=r.get("modality", ""),
                y_true_emotion=r.get("emotion6"),
                y_pred_emotion=maj_emotion,
                y_true_arousal2=r.get("arousal2"),
                y_pred_arousal2=maj_arousal2,
                y_true_arousal3=r.get("arousal3"),
                y_pred_arousal3=maj_arousal3,
            )
            fold_preds.append(row)
            pred_rows.append(row)

        fold_emotion = eval_cls(
            [p.y_true_emotion for p in fold_preds],
            [p.y_pred_emotion for p in fold_preds],
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + fold,
        )
        fold_arousal2 = eval_arousal(
            [p.y_true_arousal2 for p in fold_preds],
            [p.y_pred_arousal2 for p in fold_preds],
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + fold,
        )
        fold_arousal3 = eval_arousal(
            [p.y_true_arousal3 for p in fold_preds],
            [p.y_pred_arousal3 for p in fold_preds],
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + fold,
        )

        folds_summary.append(
            {
                "fold": fold,
                "train_n": len(train_rows),
                "val_n": len(val_rows),
                "majority_emotion6": maj_emotion,
                "majority_arousal2": maj_arousal2,
                "majority_arousal3": maj_arousal3,
                "emotion6": fold_emotion,
                "arousal2": fold_arousal2,
                "arousal3": fold_arousal3,
            }
        )

    # Global metrics across all fold validation samples
    global_emotion = eval_cls(
        [p.y_true_emotion for p in pred_rows],
        [p.y_pred_emotion for p in pred_rows],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    global_arousal2 = eval_arousal(
        [p.y_true_arousal2 for p in pred_rows],
        [p.y_pred_arousal2 for p in pred_rows],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    global_arousal3 = eval_arousal(
        [p.y_true_arousal3 for p in pred_rows],
        [p.y_pred_arousal3 for p in pred_rows],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    pred_csv = args.out_dir / "predictions.csv"
    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "clip_id",
                "fold",
                "dataset",
                "actor_id",
                "modality",
                "y_true_emotion",
                "y_pred_emotion",
                "y_true_arousal2",
                "y_pred_arousal2",
                "y_true_arousal3",
                "y_pred_arousal3",
            ]
        )
        for p in pred_rows:
            w.writerow(
                [
                    p.clip_id,
                    p.fold,
                    p.dataset,
                    p.actor_id,
                    p.modality,
                    p.y_true_emotion,
                    p.y_pred_emotion,
                    p.y_true_arousal2,
                    p.y_pred_arousal2,
                    p.y_true_arousal3,
                    p.y_pred_arousal3,
                ]
            )

    summary = {
        "run": {
            "manifest": str(args.manifest),
            "fold_dir": str(args.fold_dir),
            "num_folds": args.num_folds,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
        },
        "folds": folds_summary,
        "global": {
            "emotion6": global_emotion,
            "arousal2": global_arousal2,
            "arousal3": global_arousal3,
            "n_predictions": len(pred_rows),
        },
        "outputs": {"predictions_csv": str(pred_csv)},
    }

    summary_json = args.out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
