#!/usr/bin/env python3
"""
Build fixed Train/Val/Test split files for phase-36 ID/OOD experiments.

Outputs (default):
- derived/splits/phase36_id_ood/id_all_{train,val,test}.txt
- derived/splits/phase36_id_ood/id_crema_{train,val,test}.txt
- derived/splits/phase36_id_ood/id_ravdess_{train,val,test}.txt
- derived/splits/phase36_id_ood/ood_c2r_{train,val,test}.txt
- derived/splits/phase36_id_ood/ood_r2c_{train,val,test}.txt
- derived/splits/phase36_id_ood/summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build phase-36 ID/OOD split files")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("derived/manifests/manifest_multimodal_common6_av.jsonl"),
    )
    p.add_argument("--splits-root", type=Path, default=Path("derived/splits"))
    p.add_argument("--out-dir", type=Path, default=Path("derived/splits/phase36_id_ood"))
    p.add_argument("--id-test-fold", type=int, default=0)
    p.add_argument("--id-val-fold", type=int, default=1)
    p.add_argument("--ood-source-val-fold", type=int, default=0)
    return p.parse_args()


def load_manifest(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row["clip_id"])] = row
    return out


def load_group_csv(path: Path, allowed_ids: Set[str]) -> Dict[int, List[str]]:
    rows_by_fold: Dict[int, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = str(row["clip_id"])
            if cid not in allowed_ids:
                continue
            fold = int(row["fold"])
            rows_by_fold.setdefault(fold, []).append(cid)
    for k in rows_by_fold:
        rows_by_fold[k] = sorted(set(rows_by_fold[k]))
    return rows_by_fold


def load_id_list(path: Path, allowed_ids: Set[str]) -> List[str]:
    ids = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return sorted([cid for cid in ids if cid in allowed_ids])


def folds_to_triplet(
    rows_by_fold: Dict[int, List[str]],
    val_fold: int,
    test_fold: int,
) -> Tuple[List[str], List[str], List[str]]:
    all_folds = sorted(rows_by_fold.keys())
    train: List[str] = []
    val = rows_by_fold.get(val_fold, [])
    test = rows_by_fold.get(test_fold, [])
    for fd in all_folds:
        if fd in {val_fold, test_fold}:
            continue
        train.extend(rows_by_fold[fd])
    return sorted(set(train)), sorted(set(val)), sorted(set(test))


def source_to_train_val(
    rows_by_fold: Dict[int, List[str]],
    val_fold: int,
) -> Tuple[List[str], List[str]]:
    train: List[str] = []
    val = rows_by_fold.get(val_fold, [])
    for fd, rows in rows_by_fold.items():
        if fd == val_fold:
            continue
        train.extend(rows)
    return sorted(set(train)), sorted(set(val))


def overlap(a: Set[str], b: Set[str]) -> int:
    return len(a.intersection(b))


def actor_set(ids: Set[str], manifest: Dict[str, dict]) -> Set[str]:
    return {str(manifest[cid].get("actor_id", "")) for cid in ids if cid in manifest}


def dump_list(path: Path, ids: List[str]) -> None:
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.id_val_fold == args.id_test_fold:
        raise ValueError("--id-val-fold and --id-test-fold must be different.")

    manifest = load_manifest(args.manifest)
    allowed_ids = set(manifest.keys())
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_root = args.splits_root
    group_all = load_group_csv(splits_root / "groupkfold5_all.csv", allowed_ids)
    group_crema = load_group_csv(splits_root / "groupkfold5_crema_d.csv", allowed_ids)
    group_ravdess = load_group_csv(splits_root / "groupkfold5_ravdess.csv", allowed_ids)

    # ID tracks: fixed val/test folds from groupkfold.
    id_all = folds_to_triplet(group_all, val_fold=args.id_val_fold, test_fold=args.id_test_fold)
    id_crema = folds_to_triplet(group_crema, val_fold=args.id_val_fold, test_fold=args.id_test_fold)
    id_ravdess = folds_to_triplet(group_ravdess, val_fold=args.id_val_fold, test_fold=args.id_test_fold)

    # OOD tracks: source train/val from source folds, target test from predefined cross split.
    c2r_train_src, c2r_val_src = source_to_train_val(group_crema, val_fold=args.ood_source_val_fold)
    r2c_train_src, r2c_val_src = source_to_train_val(group_ravdess, val_fold=args.ood_source_val_fold)

    c2r_test = load_id_list(
        splits_root / "cross_dataset" / "train_crema_test_ravdess_common6_av_test.txt",
        allowed_ids,
    )
    r2c_test = load_id_list(
        splits_root / "cross_dataset" / "test_crema_train_ravdess_common6_av_test.txt",
        allowed_ids,
    )

    tracks = {
        "id_all": {"train": id_all[0], "val": id_all[1], "test": id_all[2]},
        "id_crema": {"train": id_crema[0], "val": id_crema[1], "test": id_crema[2]},
        "id_ravdess": {"train": id_ravdess[0], "val": id_ravdess[1], "test": id_ravdess[2]},
        "ood_c2r": {"train": c2r_train_src, "val": c2r_val_src, "test": c2r_test},
        "ood_r2c": {"train": r2c_train_src, "val": r2c_val_src, "test": r2c_test},
    }

    summary = {
        "manifest": str(args.manifest),
        "id_val_fold": args.id_val_fold,
        "id_test_fold": args.id_test_fold,
        "ood_source_val_fold": args.ood_source_val_fold,
        "tracks": {},
    }

    for name, parts in tracks.items():
        tr, va, te = parts["train"], parts["val"], parts["test"]
        dump_list(out_dir / f"{name}_train.txt", tr)
        dump_list(out_dir / f"{name}_val.txt", va)
        dump_list(out_dir / f"{name}_test.txt", te)

        tr_set, va_set, te_set = set(tr), set(va), set(te)
        tr_actor = actor_set(tr_set, manifest)
        va_actor = actor_set(va_set, manifest)
        te_actor = actor_set(te_set, manifest)
        summary["tracks"][name] = {
            "counts": {"train": len(tr), "val": len(va), "test": len(te)},
            "clip_overlap": {
                "train_val": overlap(tr_set, va_set),
                "train_test": overlap(tr_set, te_set),
                "val_test": overlap(va_set, te_set),
            },
            "actor_overlap": {
                "train_val": overlap(tr_actor, va_actor),
                "train_test": overlap(tr_actor, te_actor),
                "val_test": overlap(va_actor, te_actor),
            },
            "files": {
                "train": str(out_dir / f"{name}_train.txt"),
                "val": str(out_dir / f"{name}_val.txt"),
                "test": str(out_dir / f"{name}_test.txt"),
            },
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

