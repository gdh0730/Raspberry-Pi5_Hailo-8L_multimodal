#!/usr/bin/env python3
"""Build normalized Hailo input npy files from raw calib npy files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalize raw Hailo input npy files with checkpoint stats")
    p.add_argument("--index-csv", type=Path, required=True)
    p.add_argument("--calib-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=0, help="0 means all")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.index_csv.exists():
        raise FileNotFoundError(f"Missing index CSV: {args.index_csv}")
    if not args.calib_dir.exists():
        raise FileNotFoundError(f"Missing calib dir: {args.calib_dir}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    for k in ("audio_mu", "audio_sd", "video_mu", "video_sd"):
        if k not in ckpt:
            raise KeyError(f"Checkpoint missing '{k}': {args.checkpoint}")

    mu_a = np.asarray(ckpt["audio_mu"], dtype=np.float32).reshape(-1)
    sd_a = np.asarray(ckpt["audio_sd"], dtype=np.float32).reshape(-1)
    mu_v = np.asarray(ckpt["video_mu"], dtype=np.float32).reshape(-1)
    sd_v = np.asarray(ckpt["video_sd"], dtype=np.float32).reshape(-1)
    sd_a = np.where(sd_a < 1e-6, 1.0, sd_a)
    sd_v = np.where(sd_v < 1e-6, 1.0, sd_v)

    out_audio = args.out_dir / "audio"
    out_video = args.out_dir / "video"
    out_audio.mkdir(parents=True, exist_ok=True)
    out_video.mkdir(parents=True, exist_ok=True)

    n_done = 0
    with args.index_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["idx"])
            if idx < args.start_idx:
                continue
            if args.max_samples > 0 and n_done >= args.max_samples:
                break

            pa = args.calib_dir / row["audio_npy"]
            pv = args.calib_dir / row["video_npy"]
            if not pa.exists() or not pv.exists():
                raise FileNotFoundError(f"Missing input npy at idx={idx}: {pa} / {pv}")

            xa = np.load(pa).astype(np.float32).reshape(1, -1)
            xv = np.load(pv).astype(np.float32).reshape(1, -1)
            if xa.shape[1] != mu_a.shape[0]:
                raise ValueError(f"audio dim mismatch at idx={idx}: {xa.shape[1]} vs {mu_a.shape[0]}")
            if xv.shape[1] != mu_v.shape[0]:
                raise ValueError(f"video dim mismatch at idx={idx}: {xv.shape[1]} vs {mu_v.shape[0]}")

            xa_n = ((xa - mu_a) / sd_a).astype(np.float32)
            xv_n = ((xv - mu_v) / sd_v).astype(np.float32)

            stem = f"{idx:05d}.npy"
            np.save(out_audio / stem, xa_n)
            np.save(out_video / stem, xv_n)
            n_done += 1

    print(
        {
            "index_csv": str(args.index_csv),
            "calib_dir": str(args.calib_dir),
            "checkpoint": str(args.checkpoint),
            "out_dir": str(args.out_dir),
            "start_idx": args.start_idx,
            "max_samples": args.max_samples,
            "normalized_samples": n_done,
        }
    )


if __name__ == "__main__":
    main()
