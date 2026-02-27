#!/usr/bin/env python3
"""
Benchmark runtime latency for phase-2 baseline pipeline on CPU.

Measures end-to-end per-sample latency including:
- feature load/extract (via cache store)
- sklearn inference
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import List

import numpy as np

import train_ml_baselines as tmb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark phase-2 runtime")
    p.add_argument("--manifest", type=Path, default=Path("derived/manifests/manifest_multimodal_common6_av.jsonl"))
    p.add_argument("--fold-dir", type=Path, default=Path("derived/splits/groupkfold5_all"))
    p.add_argument("--mode", type=str, default="fusion", choices=["audio", "video", "fusion"])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--cache-dir", type=Path, default=Path("derived/features/cache_v1"))
    p.add_argument("--max-train", type=int, default=1000)
    p.add_argument("--max-val", type=int, default=400)
    p.add_argument("--out-json", type=Path, default=Path("derived/results/phase2_runtime_bench.json"))
    return p.parse_args()


def pct(v: List[float], q: float) -> float:
    if not v:
        return float("nan")
    s = sorted(v)
    if len(s) == 1:
        return s[0]
    rank = q * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    f = rank - lo
    return s[lo] * (1 - f) + s[hi] * f


def main() -> None:
    args = parse_args()
    repo_root = Path(".").resolve()
    manifest = tmb.load_manifest(args.manifest)
    store = tmb.FeatureStore(repo_root=repo_root, cache_dir=args.cache_dir)

    train_ids = [cid for cid in tmb.load_ids(args.fold_dir / f"fold_{args.fold}_train.txt") if cid in manifest]
    val_ids = [cid for cid in tmb.load_ids(args.fold_dir / f"fold_{args.fold}_val.txt") if cid in manifest]
    train_ids = train_ids[: args.max_train] if args.max_train > 0 else train_ids
    val_ids = val_ids[: args.max_val] if args.max_val > 0 else val_ids

    train_rows = [manifest[cid] for cid in train_ids]
    val_rows = [manifest[cid] for cid in val_ids]

    # Train a simple classifier for timing inference stage
    Xtr, ytr_e, _, _, _ = tmb.build_xy(train_rows, store, args.mode)
    tr_idx_e = [i for i, y in enumerate(ytr_e) if y is not None]
    Xtr_e = Xtr[tr_idx_e]
    ytr_e2 = [ytr_e[i] for i in tr_idx_e]
    clf = tmb.fit_logreg_classifier(Xtr_e, ytr_e2)

    lat_ms: List[float] = []
    t0 = time.perf_counter()
    n_ok = 0

    for row in val_rows:
        s = time.perf_counter()
        fa = store.get_audio(row)
        fv = store.get_video(row)
        if args.mode == "audio":
            if fa is None:
                continue
            feat = fa
        elif args.mode == "video":
            if fv is None:
                continue
            feat = fv
        else:
            if fa is None or fv is None:
                continue
            feat = np.concatenate([fa, fv], axis=0)
        _ = clf.predict(feat.reshape(1, -1))
        e = time.perf_counter()
        lat_ms.append((e - s) * 1000.0)
        n_ok += 1

    t1 = time.perf_counter()
    total_s = t1 - t0
    fps = n_ok / total_s if total_s > 0 else float("nan")

    result = {
        "mode": args.mode,
        "fold": args.fold,
        "n_train_used": len(train_rows),
        "n_val_used": len(val_rows),
        "n_benched": n_ok,
        "latency_ms": {
            "mean": statistics.fmean(lat_ms) if lat_ms else None,
            "p50": pct(lat_ms, 0.5) if lat_ms else None,
            "p95": pct(lat_ms, 0.95) if lat_ms else None,
            "max": max(lat_ms) if lat_ms else None,
        },
        "fps_equiv": fps,
        "total_wall_sec": total_s,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
