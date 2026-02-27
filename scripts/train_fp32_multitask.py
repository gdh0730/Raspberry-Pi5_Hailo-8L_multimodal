#!/usr/bin/env python3
"""
FP32 multitask training on cached audio/video features.

Tasks
- Emotion classification (common-6)
- Arousal2 classification (binary)
- Arousal3 classification (ternary, optional/masked)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import train_ml_baselines as tmb
from research_metrics import bootstrap_ci


@dataclass
class Sample:
    clip_id: str
    dataset: str
    actor_id: str
    audio: np.ndarray
    video: np.ndarray
    emotion: str
    arousal2: int
    arousal3: int  # -1 means missing


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FP32 multitask model")
    p.add_argument("--manifest", type=Path, default=Path("derived/manifests/manifest_multimodal_common6_av.jsonl"))
    p.add_argument("--fold-dir", type=Path, default=Path("derived/splits/groupkfold5_all"))
    p.add_argument("--out-dir", type=Path, default=Path("derived/results/fp32_multitask_main"))
    p.add_argument("--cache-dir", type=Path, default=Path("derived/features/cache_v1"))
    p.add_argument("--mode", type=str, default="fusion", choices=["audio", "video", "fusion"])
    p.add_argument("--fusion-type", type=str, default="concat", choices=["concat", "gated"])
    p.add_argument("--modality-dropout-p", type=float, default=0.0)
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--train-list", type=Path, default=None)
    p.add_argument("--val-list", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--emb-dim", type=int, default=96)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument(
        "--no-head-layernorm",
        action="store_true",
        help="Disable LayerNorm in fusion head for Hailo-friendly architecture.",
    )
    p.add_argument("--emotion-loss", type=str, default="ce", choices=["ce", "focal"])
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--weighted-sampler", action="store_true")
    p.add_argument("--lambda-a2", type=float, default=1.0)
    p.add_argument("--lambda-a3", type=float, default=1.0)
    p.add_argument("--n-bootstrap", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-per-fold", type=int, default=0)
    p.add_argument("--max-val-per-fold", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=500)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(req: str) -> torch.device:
    key = (req or "auto").strip().lower()
    if key == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if key == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device=cuda but CUDA is not available in current torch build.")
        return torch.device("cuda")
    if key == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported --device value: {req!r}. Use one of: auto, cpu, cuda")


def to_int_or_neg1(v: Optional[int]) -> int:
    return -1 if v is None else int(v)


def maybe_subsample(ids: List[str], max_n: int, seed: int) -> List[str]:
    if max_n <= 0 or len(ids) <= max_n:
        return ids
    rng = random.Random(seed)
    ids2 = ids[:]
    rng.shuffle(ids2)
    return sorted(ids2[:max_n])


def build_samples(
    rows: Sequence[tmb.Row],
    store: tmb.FeatureStore,
    progress_cb=None,
    progress_every: int = 500,
) -> List[Sample]:
    out: List[Sample] = []
    total = len(rows)
    for i, r in enumerate(rows, start=1):
        fa = store.get_audio(r)
        fv = store.get_video(r)
        if fa is None or fv is None:
            if progress_cb and (i % progress_every == 0 or i == total):
                progress_cb(i, total, len(out))
            continue
        if r.emotion6 is None:
            if progress_cb and (i % progress_every == 0 or i == total):
                progress_cb(i, total, len(out))
            continue
        out.append(
            Sample(
                clip_id=r.clip_id,
                dataset=r.dataset,
                actor_id=r.actor_id,
                audio=fa.astype(np.float32),
                video=fv.astype(np.float32),
                emotion=r.emotion6,
                arousal2=to_int_or_neg1(r.arousal2),
                arousal3=to_int_or_neg1(r.arousal3),
            )
        )
        if progress_cb and (i % progress_every == 0 or i == total):
            progress_cb(i, total, len(out))
    return out


class DualEncoderMultiTask(nn.Module):
    def __init__(
        self,
        in_audio: int,
        in_video: int,
        n_emotions: int,
        mode: str = "fusion",
        fusion_type: str = "concat",
        modality_dropout_p: float = 0.0,
        emb_dim: int = 96,
        hidden_dim: int = 192,
        dropout: float = 0.2,
        use_head_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.fusion_type = fusion_type
        self.modality_dropout_p = min(max(float(modality_dropout_p), 0.0), 0.9)
        self.audio_enc = nn.Sequential(
            nn.Linear(in_audio, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.ReLU(),
        )
        self.video_enc = nn.Sequential(
            nn.Linear(in_video, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.ReLU(),
        )

        fuse_in = emb_dim
        if mode == "fusion":
            if fusion_type == "gated":
                self.gate = nn.Sequential(
                    nn.Linear(emb_dim * 2, emb_dim),
                    nn.ReLU(),
                    nn.Linear(emb_dim, emb_dim),
                    nn.Sigmoid(),
                )
                fuse_in = emb_dim * 3
            else:
                fuse_in = emb_dim * 2
        fuse_layers: List[nn.Module] = [nn.Linear(fuse_in, hidden_dim)]
        if use_head_layernorm:
            fuse_layers.append(nn.LayerNorm(hidden_dim))
        fuse_layers.extend(
            [
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        self.fuse = nn.Sequential(*fuse_layers)
        self.head_emotion = nn.Linear(hidden_dim, n_emotions)
        self.head_a2 = nn.Linear(hidden_dim, 2)
        self.head_a3 = nn.Linear(hidden_dim, 3)

    def _apply_modality_dropout(self, za: torch.Tensor, zv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout_p <= 0.0:
            return za, zv
        keep_a = (torch.rand((za.shape[0], 1), device=za.device) > self.modality_dropout_p).float()
        keep_v = (torch.rand((zv.shape[0], 1), device=zv.device) > self.modality_dropout_p).float()
        both_dropped = (keep_a + keep_v) <= 0.0
        if both_dropped.any():
            keep_a[both_dropped] = 1.0
        return za * keep_a, zv * keep_v

    def forward(self, xa: torch.Tensor, xv: torch.Tensor) -> Dict[str, torch.Tensor]:
        za = self.audio_enc(xa)
        zv = self.video_enc(xv)
        if self.mode == "audio":
            z = za
        elif self.mode == "video":
            z = zv
        else:
            za, zv = self._apply_modality_dropout(za, zv)
            if self.fusion_type == "gated":
                g = self.gate(torch.cat([za, zv], dim=1))
                blend = g * za + (1.0 - g) * zv
                z = torch.cat([blend, torch.abs(za - zv), za * zv], dim=1)
            else:
                z = torch.cat([za, zv], dim=1)
        h = self.fuse(z)
        return {
            "emotion": self.head_emotion(h),
            "a2": self.head_a2(h),
            "a3": self.head_a3(h),
        }


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def fit_norm_stats(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_norm(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((x - mu) / sd).astype(np.float32)


def encode_emotions(samples: List[Sample], classes: List[str]) -> np.ndarray:
    m = {c: i for i, c in enumerate(classes)}
    return np.array([m[s.emotion] for s in samples], dtype=np.int64)


def class_weight_from_targets(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts = np.where(counts <= 0, 1.0, counts)
    w = counts.sum() / (counts * n_classes)
    return torch.tensor(w, dtype=torch.float32)


def create_dataloader(
    xa: np.ndarray,
    xv: np.ndarray,
    ye: np.ndarray,
    ya2: np.ndarray,
    ya3: np.ndarray,
    batch_size: int,
    shuffle: bool,
    sampler: Optional[WeightedRandomSampler] = None,
    pin_memory: bool = False,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(xa),
        torch.from_numpy(xv),
        torch.from_numpy(ye),
        torch.from_numpy(ya2),
        torch.from_numpy(ya3),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        pin_memory=pin_memory,
    )


def evaluate(
    model: nn.Module,
    xa: np.ndarray,
    xv: np.ndarray,
    ye: np.ndarray,
    ya2: np.ndarray,
    ya3: np.ndarray,
    emotion_classes: List[str],
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    with torch.no_grad():
        ta = torch.from_numpy(xa).to(device, non_blocking=True)
        tv = torch.from_numpy(xv).to(device, non_blocking=True)
        out = model(ta, tv)
        emo_logits = out["emotion"].cpu()
        a2_logits = out["a2"].cpu()
        a3_logits = out["a3"].cpu()

    emo_pred_idx = emo_logits.argmax(dim=1).numpy()
    emo_true_idx = ye
    emo_pred = [emotion_classes[i] for i in emo_pred_idx.tolist()]
    emo_true = [emotion_classes[i] for i in emo_true_idx.tolist()]

    emo_acc = float(accuracy_score(emo_true, emo_pred))
    emo_f1 = float(f1_score(emo_true, emo_pred, average="macro"))
    emo_prob = F.softmax(emo_logits, dim=1).numpy()
    try:
        emo_auc = float(
            roc_auc_score(
                emo_true_idx,
                emo_prob,
                multi_class="ovr",
                labels=np.arange(len(emotion_classes)),
            )
        )
    except Exception:
        emo_auc = None

    # Arousal2 (masked)
    m2 = ya2 >= 0
    a2_true = ya2[m2]
    a2_pred = a2_logits.argmax(dim=1).numpy()[m2]
    a2_mae = float(mean_absolute_error(a2_true, a2_pred)) if len(a2_true) else None

    # Arousal3 (masked)
    m3 = ya3 >= 0
    a3_true = ya3[m3]
    a3_pred = a3_logits.argmax(dim=1).numpy()[m3]
    a3_mae = float(mean_absolute_error(a3_true, a3_pred)) if len(a3_true) else None

    return {
        "emotion_true": emo_true,
        "emotion_pred": emo_pred,
        "emotion_acc": emo_acc,
        "emotion_f1": emo_f1,
        "emotion_auc": emo_auc,
        "a2_true": a2_true.tolist(),
        "a2_pred": a2_pred.tolist(),
        "a2_mae": a2_mae,
        "a3_true": a3_true.tolist(),
        "a3_pred": a3_pred.tolist(),
        "a3_mae": a3_mae,
    }


def update_progress(path: Path, payload: dict, print_line: bool = True) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    if print_line:
        stage = payload.get("stage", "")
        fold = payload.get("fold")
        epoch = payload.get("epoch")
        msg = payload.get("message", "")
        print(f"[{stage}] fold={fold} epoch={epoch} {msg}", flush=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    repo_root = Path(".").resolve()
    store = tmb.FeatureStore(repo_root=repo_root, cache_dir=args.cache_dir)
    manifest = tmb.load_manifest(args.manifest)

    if (args.train_list is None) ^ (args.val_list is None):
        raise ValueError("--train-list and --val-list must be provided together")

    if args.train_list and args.val_list:
        split_defs = [(0, args.train_list, args.val_list)]
    else:
        split_defs = []
        for fold in range(args.num_folds):
            split_defs.append(
                (fold, args.fold_dir / f"fold_{fold}_train.txt", args.fold_dir / f"fold_{fold}_val.txt")
            )

    all_pred_rows: List[dict] = []
    fold_summaries: List[dict] = []
    t_start = time.time()

    update_progress(
        progress_path,
        {
            "status": "running",
            "stage": "init",
            "fold": None,
            "epoch": None,
            "message": f"manifest rows={len(manifest)} mode={args.mode} device={device}",
            "elapsed_sec": 0.0,
        },
    )

    for fold, tr_file, va_file in split_defs:
        train_ids = [cid for cid in tmb.load_ids(tr_file) if cid in manifest]
        val_ids = [cid for cid in tmb.load_ids(va_file) if cid in manifest]
        train_ids = maybe_subsample(train_ids, args.max_train_per_fold, args.seed + fold * 11 + 1)
        val_ids = maybe_subsample(val_ids, args.max_val_per_fold, args.seed + fold * 11 + 2)
        train_rows = [manifest[cid] for cid in train_ids]
        val_rows = [manifest[cid] for cid in val_ids]

        update_progress(
            progress_path,
            {
                "status": "running",
                "stage": "features_train",
                "fold": fold,
                "epoch": 0,
                "message": f"extracting train features from {len(train_rows)} rows",
                "elapsed_sec": round(time.time() - t_start, 2),
            },
        )

        train_samples = build_samples(
            train_rows,
            store,
            progress_cb=lambda i, n, kept: update_progress(
                progress_path,
                {
                    "status": "running",
                    "stage": "features_train",
                    "fold": fold,
                    "epoch": 0,
                    "message": f"{i}/{n} processed, kept={kept}",
                    "elapsed_sec": round(time.time() - t_start, 2),
                },
                print_line=(i == n),
            ),
            progress_every=max(1, args.progress_every),
        )

        update_progress(
            progress_path,
            {
                "status": "running",
                "stage": "features_val",
                "fold": fold,
                "epoch": 0,
                "message": f"extracting val features from {len(val_rows)} rows",
                "elapsed_sec": round(time.time() - t_start, 2),
            },
        )
        val_samples = build_samples(
            val_rows,
            store,
            progress_cb=lambda i, n, kept: update_progress(
                progress_path,
                {
                    "status": "running",
                    "stage": "features_val",
                    "fold": fold,
                    "epoch": 0,
                    "message": f"{i}/{n} processed, kept={kept}",
                    "elapsed_sec": round(time.time() - t_start, 2),
                },
                print_line=(i == n),
            ),
            progress_every=max(1, args.progress_every),
        )

        if not train_samples or not val_samples:
            fold_summaries.append({"fold": fold, "error": "empty samples after feature extraction"})
            continue

        emo_classes = sorted(set(s.emotion for s in train_samples))
        emo_map = {c: i for i, c in enumerate(emo_classes)}
        # keep only val classes seen in train
        val_samples = [s for s in val_samples if s.emotion in emo_map]
        if not val_samples:
            fold_summaries.append({"fold": fold, "error": "no val samples with train emotion classes"})
            continue

        xa_tr = np.stack([s.audio for s in train_samples]).astype(np.float32)
        xv_tr = np.stack([s.video for s in train_samples]).astype(np.float32)
        xa_va = np.stack([s.audio for s in val_samples]).astype(np.float32)
        xv_va = np.stack([s.video for s in val_samples]).astype(np.float32)

        mu_a, sd_a = fit_norm_stats(xa_tr)
        mu_v, sd_v = fit_norm_stats(xv_tr)
        xa_tr = apply_norm(xa_tr, mu_a, sd_a)
        xv_tr = apply_norm(xv_tr, mu_v, sd_v)
        xa_va = apply_norm(xa_va, mu_a, sd_a)
        xv_va = apply_norm(xv_va, mu_v, sd_v)

        ye_tr = np.array([emo_map[s.emotion] for s in train_samples], dtype=np.int64)
        ya2_tr = np.array([s.arousal2 for s in train_samples], dtype=np.int64)
        ya3_tr = np.array([s.arousal3 for s in train_samples], dtype=np.int64)
        ye_va = np.array([emo_map[s.emotion] for s in val_samples], dtype=np.int64)
        ya2_va = np.array([s.arousal2 for s in val_samples], dtype=np.int64)
        ya3_va = np.array([s.arousal3 for s in val_samples], dtype=np.int64)

        sampler = None
        if args.weighted_sampler:
            cls_counts = np.bincount(ye_tr, minlength=len(emo_classes)).astype(np.float64)
            cls_counts = np.where(cls_counts <= 0.0, 1.0, cls_counts)
            cls_w = cls_counts.sum() / (cls_counts * float(len(emo_classes)))
            sample_w = cls_w[ye_tr]
            gen = torch.Generator()
            gen.manual_seed(args.seed + fold)
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_w).double(),
                num_samples=len(sample_w),
                replacement=True,
                generator=gen,
            )

        dl = create_dataloader(
            xa_tr,
            xv_tr,
            ye_tr,
            ya2_tr,
            ya3_tr,
            batch_size=args.batch_size,
            shuffle=True,
            sampler=sampler,
            pin_memory=(device.type == "cuda"),
        )

        model = DualEncoderMultiTask(
            in_audio=xa_tr.shape[1],
            in_video=xv_tr.shape[1],
            n_emotions=len(emo_classes),
            mode=args.mode,
            fusion_type=args.fusion_type,
            modality_dropout_p=args.modality_dropout_p,
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            use_head_layernorm=(not args.no_head_layernorm),
        ).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, args.epochs))

        emo_class_weight = class_weight_from_targets(ye_tr, len(emo_classes)).to(device)
        if args.emotion_loss == "focal":
            ce_emotion = FocalLoss(gamma=args.focal_gamma, weight=emo_class_weight)
        else:
            ls = min(max(float(args.label_smoothing), 0.0), 0.3)
            # Keep smoothing bounded for stability with class-weighted CE.
            ce_emotion = lambda logits, targets: F.cross_entropy(
                logits,
                targets,
                weight=emo_class_weight,
                label_smoothing=ls,
            )
        ce_a2 = nn.CrossEntropyLoss()
        ce_a3 = nn.CrossEntropyLoss()

        best_f1 = -1.0
        best_state = None
        best_eval = None

        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for xb_a, xb_v, yb_e, yb_a2, yb_a3 in dl:
                xb_a = xb_a.to(device, non_blocking=True)
                xb_v = xb_v.to(device, non_blocking=True)
                yb_e = yb_e.to(device, non_blocking=True)
                yb_a2 = yb_a2.to(device, non_blocking=True)
                yb_a3 = yb_a3.to(device, non_blocking=True)

                out = model(xb_a, xb_v)
                loss = ce_emotion(out["emotion"], yb_e)

                m2 = yb_a2 >= 0
                if m2.any():
                    loss = loss + args.lambda_a2 * ce_a2(out["a2"][m2], yb_a2[m2])
                m3 = yb_a3 >= 0
                if m3.any():
                    loss = loss + args.lambda_a3 * ce_a3(out["a3"][m3], yb_a3[m3])

                optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optim.step()
                losses.append(float(loss.item()))
            sched.step()

            ev = evaluate(model, xa_va, xv_va, ye_va, ya2_va, ya3_va, emo_classes, device)
            is_best = ev["emotion_f1"] > best_f1
            if is_best:
                best_f1 = ev["emotion_f1"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_eval = ev

            update_progress(
                progress_path,
                {
                    "status": "running",
                    "stage": "train",
                    "fold": fold,
                    "epoch": epoch,
                    "message": (
                        f"loss={np.mean(losses):.4f} "
                        f"val_acc={ev['emotion_acc']:.4f} "
                        f"val_f1={ev['emotion_f1']:.4f} "
                        f"best_f1={best_f1:.4f}"
                    ),
                    "elapsed_sec": round(time.time() - t_start, 2),
                },
            )

        if best_state is not None:
            model.load_state_dict(best_state)
            torch.save(
                {
                    "fold": fold,
                    "mode": args.mode,
                    "emotion_classes": emo_classes,
                    "audio_mu": mu_a,
                    "audio_sd": sd_a,
                    "video_mu": mu_v,
                    "video_sd": sd_v,
                    "use_head_layernorm": (not args.no_head_layernorm),
                    "state_dict": model.state_dict(),
                },
                ckpt_dir / f"best_fold_{fold}.pt",
            )
            ev = best_eval
        else:
            ev = evaluate(model, xa_va, xv_va, ye_va, ya2_va, ya3_va, emo_classes, device)

        # bootstrap CIs
        acc_lo, acc_hi = bootstrap_ci(
            ev["emotion_true"], ev["emotion_pred"], accuracy_score, n_boot=args.n_bootstrap, seed=args.seed + fold
        )
        f1_lo, f1_hi = bootstrap_ci(
            ev["emotion_true"],
            ev["emotion_pred"],
            lambda y1, y2: f1_score(y1, y2, average="macro"),
            n_boot=args.n_bootstrap,
            seed=args.seed + 100 + fold,
        )
        if ev["a2_true"]:
            a2_lo, a2_hi = bootstrap_ci(
                ev["a2_true"], ev["a2_pred"], mean_absolute_error, n_boot=args.n_bootstrap, seed=args.seed + 200 + fold
            )
        else:
            a2_lo, a2_hi = None, None
        if ev["a3_true"]:
            a3_lo, a3_hi = bootstrap_ci(
                ev["a3_true"], ev["a3_pred"], mean_absolute_error, n_boot=args.n_bootstrap, seed=args.seed + 300 + fold
            )
        else:
            a3_lo, a3_hi = None, None

        fold_summary = {
            "fold": fold,
            "n_train": len(train_samples),
            "n_val": len(val_samples),
            "emotion_classes": emo_classes,
            "emotion": {
                "accuracy": ev["emotion_acc"],
                "accuracy_ci95": [acc_lo, acc_hi],
                "macro_f1": ev["emotion_f1"],
                "macro_f1_ci95": [f1_lo, f1_hi],
                "ovr_auc": ev["emotion_auc"],
                "n": len(ev["emotion_true"]),
            },
            "arousal2": {
                "mae": ev["a2_mae"],
                "mae_ci95": [a2_lo, a2_hi],
                "n": len(ev["a2_true"]),
            },
            "arousal3": {
                "mae": ev["a3_mae"],
                "mae_ci95": [a3_lo, a3_hi],
                "n": len(ev["a3_true"]),
            },
        }
        fold_summaries.append(fold_summary)

        # Save per-sample predictions
        for s, yp in zip(val_samples, ev["emotion_pred"]):
            all_pred_rows.append(
                {
                    "model_type": f"fp32_{args.mode}",
                    "fold": fold,
                    "clip_id": s.clip_id,
                    "dataset": s.dataset,
                    "actor_id": s.actor_id,
                    "y_true_emotion": s.emotion,
                    "y_pred_emotion": yp,
                    "y_true_arousal2": s.arousal2,
                    "y_pred_arousal2": None,  # filled below by clip order mapping if needed
                    "y_true_arousal3": None if s.arousal3 < 0 else s.arousal3,
                    "y_pred_arousal3": None,
                }
            )

        # fill arousal predictions by index
        a2_pred_full = np.full(len(val_samples), -1, dtype=np.int64)
        if ev["a2_true"]:
            m2 = ya2_va >= 0
            a2_pred_full[m2] = np.array(ev["a2_pred"], dtype=np.int64)
        a3_pred_full = np.full(len(val_samples), -1, dtype=np.int64)
        if ev["a3_true"]:
            m3 = ya3_va >= 0
            a3_pred_full[m3] = np.array(ev["a3_pred"], dtype=np.int64)

        start = len(all_pred_rows) - len(val_samples)
        for i in range(len(val_samples)):
            all_pred_rows[start + i]["y_pred_arousal2"] = None if a2_pred_full[i] < 0 else int(a2_pred_full[i])
            all_pred_rows[start + i]["y_pred_arousal3"] = None if a3_pred_full[i] < 0 else int(a3_pred_full[i])

    # Global summary
    y_true_e = [r["y_true_emotion"] for r in all_pred_rows if r["y_true_emotion"] and r["y_pred_emotion"]]
    y_pred_e = [r["y_pred_emotion"] for r in all_pred_rows if r["y_true_emotion"] and r["y_pred_emotion"]]
    global_emotion = {
        "accuracy": float(accuracy_score(y_true_e, y_pred_e)) if y_true_e else None,
        "macro_f1": float(f1_score(y_true_e, y_pred_e, average="macro")) if y_true_e else None,
        "n": len(y_true_e),
    }
    y_true_a2 = [int(r["y_true_arousal2"]) for r in all_pred_rows if r["y_true_arousal2"] is not None and r["y_pred_arousal2"] is not None]
    y_pred_a2 = [int(r["y_pred_arousal2"]) for r in all_pred_rows if r["y_true_arousal2"] is not None and r["y_pred_arousal2"] is not None]
    global_a2 = {
        "mae": float(mean_absolute_error(y_true_a2, y_pred_a2)) if y_true_a2 else None,
        "n": len(y_true_a2),
    }
    y_true_a3 = [int(r["y_true_arousal3"]) for r in all_pred_rows if r["y_true_arousal3"] is not None and r["y_pred_arousal3"] is not None]
    y_pred_a3 = [int(r["y_pred_arousal3"]) for r in all_pred_rows if r["y_true_arousal3"] is not None and r["y_pred_arousal3"] is not None]
    global_a3 = {
        "mae": float(mean_absolute_error(y_true_a3, y_pred_a3)) if y_true_a3 else None,
        "n": len(y_true_a3),
    }

    pred_csv = out_dir / "predictions.csv"
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
        for r in all_pred_rows:
            w.writerow(r)

    summary = {
        "run": {
            "manifest": str(args.manifest),
            "fold_dir": str(args.fold_dir),
            "train_list": str(args.train_list) if args.train_list else None,
            "val_list": str(args.val_list) if args.val_list else None,
            "mode": args.mode,
            "num_folds": args.num_folds,
            "fusion_type": args.fusion_type,
            "modality_dropout_p": args.modality_dropout_p,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "emb_dim": args.emb_dim,
            "dropout": args.dropout,
            "use_head_layernorm": (not args.no_head_layernorm),
            "emotion_loss": args.emotion_loss,
            "focal_gamma": args.focal_gamma,
            "label_smoothing": args.label_smoothing,
            "weighted_sampler": args.weighted_sampler,
            "seed": args.seed,
            "device_requested": args.device,
            "device_resolved": str(device),
            "cuda_name": (torch.cuda.get_device_name(0) if device.type == "cuda" else None),
            "n_bootstrap": args.n_bootstrap,
            "max_train_per_fold": args.max_train_per_fold,
            "max_val_per_fold": args.max_val_per_fold,
        },
        "folds": fold_summaries,
        "global": {
            "emotion": global_emotion,
            "arousal2": global_a2,
            "arousal3": global_a3,
            "n_predictions": len(all_pred_rows),
        },
        "outputs": {"predictions_csv": str(pred_csv)},
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    update_progress(
        progress_path,
        {
            "status": "completed",
            "stage": "done",
            "fold": None,
            "epoch": None,
            "message": f"finished summary={summary_path}",
            "elapsed_sec": round(time.time() - t_start, 2),
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
