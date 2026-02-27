#!/usr/bin/env python3
"""
Evaluate a saved FP32 multitask checkpoint on an index.csv of prebuilt npy inputs.

This is intended for fair Hailo-vs-FP32 comparison on the exact same sample set.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

import train_ml_baselines as tmb
import train_fp32_multitask as fp32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate FP32 checkpoint on index.csv npy inputs")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--index-csv", type=Path, required=True)
    p.add_argument("--calib-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path("derived/manifests/manifest_multimodal_common6_av.jsonl"))
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-pred-csv", type=Path, default=None)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def infer_model_from_ckpt(ckpt: dict) -> fp32.DualEncoderMultiTask:
    sd = ckpt["state_dict"]
    emo_classes = ckpt["emotion_classes"]
    mode = ckpt.get("mode", "fusion")
    use_head_ln = bool(ckpt.get("use_head_layernorm", True))
    fusion_type = "gated" if any(k.startswith("gate.") for k in sd.keys()) else "concat"

    in_audio = int(sd["audio_enc.0.weight"].shape[1])
    in_video = int(sd["video_enc.0.weight"].shape[1])
    hidden_dim = int(sd["audio_enc.0.weight"].shape[0])
    emb_dim = int(sd["audio_enc.3.weight"].shape[0])

    model = fp32.DualEncoderMultiTask(
        in_audio=in_audio,
        in_video=in_video,
        n_emotions=len(emo_classes),
        mode=mode,
        fusion_type=fusion_type,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        dropout=0.0,
        use_head_layernorm=use_head_ln,
    )
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def to_int_or_none(v: Optional[object]) -> Optional[int]:
    return None if v is None else int(v)


def main() -> None:
    args = parse_args()
    ckpt = load_checkpoint(args.checkpoint)
    model = infer_model_from_ckpt(ckpt)
    emotion_classes: List[str] = list(ckpt["emotion_classes"])
    emo_map: Dict[str, int] = {k: i for i, k in enumerate(emotion_classes)}

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but CUDA is unavailable.")
    model = model.to(device)

    manifest = tmb.load_manifest(args.manifest)

    rows = []
    with args.index_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    xa_list = []
    xv_list = []
    y_true_emo = []
    y_true_a2 = []
    y_true_a3 = []
    clip_ids = []
    datasets = []
    actor_ids = []
    valid_mask = []

    for r in rows:
        clip_id = r["clip_id"]
        m = manifest.get(clip_id)
        if m is None or m.emotion6 is None or m.emotion6 not in emo_map:
            continue
        pa = args.calib_dir / r["audio_npy"]
        pv = args.calib_dir / r["video_npy"]
        if not pa.exists() or not pv.exists():
            continue

        xa = np.load(pa).astype(np.float32).reshape(1, -1)
        xv = np.load(pv).astype(np.float32).reshape(1, -1)
        xa_list.append(xa[0])
        xv_list.append(xv[0])
        y_true_emo.append(int(emo_map[m.emotion6]))
        y_true_a2.append(-1 if m.arousal2 is None else int(m.arousal2))
        y_true_a3.append(-1 if m.arousal3 is None else int(m.arousal3))
        clip_ids.append(clip_id)
        datasets.append(m.dataset)
        actor_ids.append(m.actor_id)
        valid_mask.append(True)

    if not xa_list:
        raise RuntimeError("No valid rows found for evaluation.")

    xa_np = np.stack(xa_list).astype(np.float32)
    xv_np = np.stack(xv_list).astype(np.float32)
    ye_np = np.asarray(y_true_emo, dtype=np.int64)
    ya2_np = np.asarray(y_true_a2, dtype=np.int64)
    ya3_np = np.asarray(y_true_a3, dtype=np.int64)

    with torch.no_grad():
        ta = torch.from_numpy(xa_np).to(device)
        tv = torch.from_numpy(xv_np).to(device)
        out = model(ta, tv)
        emo_pred_idx = out["emotion"].argmax(dim=1).cpu().numpy()
        a2_pred_idx = out["a2"].argmax(dim=1).cpu().numpy()
        a3_pred_idx = out["a3"].argmax(dim=1).cpu().numpy()

    emo_true = [emotion_classes[i] for i in ye_np.tolist()]
    emo_pred = [emotion_classes[i] for i in emo_pred_idx.tolist()]
    emo_acc = float(accuracy_score(emo_true, emo_pred))
    emo_f1 = float(f1_score(emo_true, emo_pred, average="macro"))

    m2 = ya2_np >= 0
    a2_mae = float(mean_absolute_error(ya2_np[m2], a2_pred_idx[m2])) if m2.any() else None
    m3 = ya3_np >= 0
    a3_mae = float(mean_absolute_error(ya3_np[m3], a3_pred_idx[m3])) if m3.any() else None

    if args.out_pred_csv is not None:
        args.out_pred_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_pred_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
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
            for i in range(len(clip_ids)):
                w.writerow(
                    {
                        "clip_id": clip_ids[i],
                        "dataset": datasets[i],
                        "actor_id": actor_ids[i],
                        "y_true_emotion": emo_true[i],
                        "y_pred_emotion": emo_pred[i],
                        "y_true_arousal2": None if ya2_np[i] < 0 else int(ya2_np[i]),
                        "y_pred_arousal2": None if ya2_np[i] < 0 else int(a2_pred_idx[i]),
                        "y_true_arousal3": None if ya3_np[i] < 0 else int(ya3_np[i]),
                        "y_pred_arousal3": None if ya3_np[i] < 0 else int(a3_pred_idx[i]),
                    }
                )

    summary = {
        "checkpoint": str(args.checkpoint),
        "index_csv": str(args.index_csv),
        "calib_dir": str(args.calib_dir),
        "n": int(len(clip_ids)),
        "emotion6": {"accuracy": emo_acc, "macro_f1": emo_f1},
        "arousal2": {"mae": a2_mae, "n": int(m2.sum())},
        "arousal3": {"mae": a3_mae, "n": int(m3.sum())},
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
