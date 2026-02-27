#!/usr/bin/env python3
"""
Dump representative calibration samples into npy/npz directories.

Output layout:
- <out_dir>/audio/*.npy  (shape: [1, in_audio])
- <out_dir>/video/*.npy  (shape: [1, in_video])
- <out_dir>/pairs/*.npz  (keys: xa, xv)
- <out_dir>/index.csv
- <out_dir>/summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import List

import numpy as np
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_ml_baselines as tmb  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump calibration npy directory from cached features")
    p.add_argument("--manifest", type=Path, default=Path("derived/manifests/manifest_multimodal_common6_av.jsonl"))
    p.add_argument("--cache-dir", type=Path, default=Path("derived/features/cache_v5_hubert"))
    p.add_argument(
        "--split-list",
        type=Path,
        default=Path("derived/splits/groupkfold5_all/fold_0_train.txt"),
        help="Optional list of clip IDs. If missing, all manifest rows are considered.",
    )
    p.add_argument("--out-dir", type=Path, default=Path("derived/hailo/calib/fold0_train_1024"))
    p.add_argument("--max-samples", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument(
        "--normalize-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint (.pt) that contains audio_mu/audio_sd/video_mu/video_sd.",
    )
    return p.parse_args()


def load_ids(path: Path) -> List[str]:
    if not path.exists():
        return []
    return tmb.load_ids(path)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    manifest = tmb.load_manifest(args.manifest)
    store = tmb.FeatureStore(repo_root=repo_root, cache_dir=args.cache_dir)

    if args.split_list.exists():
        ids = [cid for cid in load_ids(args.split_list) if cid in manifest]
        source = str(args.split_list)
    else:
        ids = sorted(manifest.keys())
        source = "manifest_all"

    rng = random.Random(args.seed)
    rng.shuffle(ids)

    norm_stats = None
    if args.normalize_checkpoint is not None:
        ckpt_path = args.normalize_checkpoint.resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing normalize checkpoint: {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        required = ("audio_mu", "audio_sd", "video_mu", "video_sd")
        for k in required:
            if k not in ckpt:
                raise KeyError(f"Checkpoint missing '{k}': {ckpt_path}")
        mu_a = np.asarray(ckpt["audio_mu"], dtype=np.float32).reshape(-1)
        sd_a = np.asarray(ckpt["audio_sd"], dtype=np.float32).reshape(-1)
        mu_v = np.asarray(ckpt["video_mu"], dtype=np.float32).reshape(-1)
        sd_v = np.asarray(ckpt["video_sd"], dtype=np.float32).reshape(-1)
        sd_a = np.where(sd_a < 1e-6, 1.0, sd_a)
        sd_v = np.where(sd_v < 1e-6, 1.0, sd_v)
        norm_stats = {"mu_a": mu_a, "sd_a": sd_a, "mu_v": mu_v, "sd_v": sd_v, "checkpoint": str(ckpt_path)}

    out_dir = args.out_dir.resolve()
    audio_dir = out_dir / "audio"
    video_dir = out_dir / "video"
    pair_dir = out_dir / "pairs"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    skip_missing = 0
    rows = []
    for cid in ids:
        row = manifest[cid]
        fa = store.get_audio(row)
        fv = store.get_video(row)
        if fa is None or fv is None:
            skip_missing += 1
            continue

        xa = fa.astype(np.float32).reshape(1, -1)
        xv = fv.astype(np.float32).reshape(1, -1)
        if norm_stats is not None:
            if xa.shape[1] != norm_stats["mu_a"].shape[0]:
                raise ValueError(
                    f"audio dim mismatch: feature={xa.shape[1]} stats={norm_stats['mu_a'].shape[0]} clip={cid}"
                )
            if xv.shape[1] != norm_stats["mu_v"].shape[0]:
                raise ValueError(
                    f"video dim mismatch: feature={xv.shape[1]} stats={norm_stats['mu_v'].shape[0]} clip={cid}"
                )
            xa = ((xa - norm_stats["mu_a"]) / norm_stats["sd_a"]).astype(np.float32)
            xv = ((xv - norm_stats["mu_v"]) / norm_stats["sd_v"]).astype(np.float32)
        stem = f"{kept:05d}"
        pa = audio_dir / f"{stem}.npy"
        pv = video_dir / f"{stem}.npy"
        pp = pair_dir / f"{stem}.npz"
        np.save(pa, xa)
        np.save(pv, xv)
        np.savez(pp, xa=xa, xv=xv)

        rows.append(
            {
                "idx": kept,
                "clip_id": cid,
                "dataset": row.dataset,
                "actor_id": row.actor_id,
                "audio_npy": str(pa.relative_to(out_dir)),
                "video_npy": str(pv.relative_to(out_dir)),
                "pair_npz": str(pp.relative_to(out_dir)),
            }
        )
        kept += 1
        # max-samples <= 0 means "use all selected IDs"
        if args.max_samples > 0 and kept >= args.max_samples:
            break

    with (out_dir / "index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["idx", "clip_id", "dataset", "actor_id", "audio_npy", "video_npy", "pair_npz"],
        )
        w.writeheader()
        w.writerows(rows)

    summary = {
        "manifest": str(args.manifest),
        "cache_dir": str(args.cache_dir),
        "split_source": source,
        "requested_max_samples": args.max_samples,
        "seed": args.seed,
        "collected": kept,
        "skipped_missing_feature": skip_missing,
        "out_dir": str(out_dir),
        "audio_dir": str(audio_dir),
        "video_dir": str(video_dir),
        "pair_dir": str(pair_dir),
        "index_csv": str(out_dir / "index.csv"),
        "normalized": norm_stats is not None,
        "normalize_checkpoint": (None if norm_stats is None else norm_stats["checkpoint"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
