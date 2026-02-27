#!/usr/bin/env python3
"""
Train and evaluate lightweight ML baselines:
- B1: audio-only
- B2: video-only
- B3: audio+video late-fusion (feature concat)

Uses ffmpeg feature extraction + scikit-learn models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as sk_acc
from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from research_metrics import bootstrap_ci


@dataclass
class Row:
    clip_id: str
    dataset: str
    actor_id: str
    modality: str
    emotion6: Optional[str]
    arousal2: Optional[int]
    arousal3: Optional[int]
    path_audio: Optional[str]
    path_video: Optional[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train B1/B2/B3 sklearn baselines")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("derived/manifests/manifest_multimodal_common6_av.jsonl"),
    )
    p.add_argument(
        "--fold-dir",
        type=Path,
        default=Path("derived/splits/groupkfold5_all"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("derived/results/ml_baselines"))
    p.add_argument("--cache-dir", type=Path, default=Path("derived/features/cache_v1"))
    p.add_argument("--modalities", type=str, default="audio,video,fusion")
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument(
        "--classifier",
        type=str,
        default="logreg",
        choices=["logreg", "linear_svm", "rbf_svm", "random_forest"],
    )
    p.add_argument("--train-list", type=Path, default=None)
    p.add_argument("--val-list", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bootstrap", type=int, default=300)
    p.add_argument("--max-train-per-fold", type=int, default=0)
    p.add_argument("--max-val-per-fold", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=250)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument(
        "--domain-adapt",
        type=str,
        default="none",
        choices=["none", "coral"],
        help="Domain adaptation strategy for train->val feature shift.",
    )
    p.add_argument(
        "--coral-eps",
        type=float,
        default=1e-5,
        help="Numerical stability epsilon for CORAL covariance operations.",
    )
    return p.parse_args()


def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def safe_id(clip_id: str) -> str:
    key = clip_id.encode("utf-8")
    return hashlib.md5(key).hexdigest()


def load_manifest(path: Path) -> Dict[str, Row]:
    rows: Dict[str, Row] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            row = Row(
                clip_id=obj["clip_id"],
                dataset=obj.get("dataset", ""),
                actor_id=obj.get("actor_id", ""),
                modality=obj.get("modality", ""),
                emotion6=obj.get("emotion6"),
                arousal2=obj.get("arousal2"),
                arousal3=obj.get("arousal3"),
                path_audio=obj.get("path_audio"),
                path_video=obj.get("path_video"),
            )
            rows[row.clip_id] = row
    return rows


def load_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def maybe_subsample(ids: List[str], max_n: int, seed: int) -> List[str]:
    if max_n <= 0 or len(ids) <= max_n:
        return ids
    rng = random.Random(seed)
    ids2 = ids[:]
    rng.shuffle(ids2)
    return sorted(ids2[:max_n])


def _bar(curr: int, total: int, width: int = 26) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    frac = max(0.0, min(1.0, curr / total))
    fill = int(math.floor(frac * width))
    return "[" + ("#" * fill) + ("-" * (width - fill)) + "]"


class ProgressTracker:
    def __init__(self, out_dir: Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = out_dir / "progress.json"
        self.start_ts = time.time()
        self.last_state: Dict[str, object] = {
            "status": "starting",
            "stage": "init",
            "message": "initializing",
            "fold": None,
            "mode": None,
            "current": 0,
            "total": 0,
            "percent": 0.0,
            "elapsed_sec": 0.0,
        }
        self._flush()

    def _flush(self) -> None:
        self.last_state["elapsed_sec"] = round(time.time() - self.start_ts, 2)
        self.path.write_text(json.dumps(self.last_state, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    def update(
        self,
        *,
        status: str,
        stage: str,
        message: str,
        fold: Optional[int] = None,
        mode: Optional[str] = None,
        current: int = 0,
        total: int = 0,
        force_line: bool = False,
    ) -> None:
        pct = (100.0 * current / total) if total else 0.0
        self.last_state = {
            "status": status,
            "stage": stage,
            "message": message,
            "fold": fold,
            "mode": mode,
            "current": current,
            "total": total,
            "percent": round(pct, 2),
            "elapsed_sec": self.last_state.get("elapsed_sec", 0.0),
        }
        self._flush()
        if not self.enabled:
            return
        prefix = f"[{status}] {stage}"
        if fold is not None:
            prefix += f" fold={fold}"
        if mode is not None:
            prefix += f" mode={mode}"
        line = f"{prefix} {_bar(current, total)} {current}/{total} ({pct:5.1f}%) {message}"
        if sys.stdout.isatty():
            end = "\n" if force_line or (total and current >= total) else ""
            sys.stdout.write("\r" + line + (" " * 8))
            sys.stdout.flush()
            if end:
                sys.stdout.write("\n")
                sys.stdout.flush()
        else:
            # Non-interactive shell: print only major checkpoints.
            if force_line or (total and current >= total):
                print(line, flush=True)


def extract_audio_feature(media_path: Path) -> Optional[np.ndarray]:
    # 2s, mono 16k, float32 PCM
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(media_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-t",
        "2",
        "-f",
        "f32le",
        "-",
    ]
    cp = run_cmd(cmd)
    if cp.returncode != 0 or not cp.stdout:
        return None
    wav = np.frombuffer(cp.stdout, dtype=np.float32)
    if wav.size < 320:  # 20ms
        return None

    abs_w = np.abs(wav)
    diff = np.diff(wav)
    zcr = float(np.mean((wav[:-1] * wav[1:]) < 0)) if wav.size > 1 else 0.0
    rms = float(np.sqrt(np.mean(wav**2)))

    # frequency-domain summaries
    n_fft = min(4096, wav.size)
    if n_fft < 64:
        return None
    spec = np.fft.rfft(wav[:n_fft])
    mag = np.abs(spec) + 1e-8
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / 16000.0)
    centroid = float(np.sum(freqs * mag) / np.sum(mag))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / np.sum(mag)))
    cumsum = np.cumsum(mag)
    roll_idx = int(np.searchsorted(cumsum, 0.85 * cumsum[-1]))
    rolloff85 = float(freqs[min(roll_idx, len(freqs) - 1)])

    feat = np.array(
        [
            float(np.mean(wav)),
            float(np.std(wav)),
            float(np.min(wav)),
            float(np.max(wav)),
            float(np.mean(abs_w)),
            rms,
            float(np.std(diff)) if diff.size else 0.0,
            zcr,
            float(np.percentile(wav, 5)),
            float(np.percentile(wav, 25)),
            float(np.percentile(wav, 50)),
            float(np.percentile(wav, 75)),
            float(np.percentile(wav, 95)),
            centroid,
            bandwidth,
            rolloff85,
        ],
        dtype=np.float32,
    )
    return feat


def extract_video_feature(media_path: Path) -> Optional[np.ndarray]:
    # up to first 2s, 16 frames, grayscale 64x64
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(media_path),
        "-t",
        "2",
        "-vf",
        "fps=8,scale=64:64,format=gray",
        "-frames:v",
        "16",
        "-f",
        "rawvideo",
        "-",
    ]
    cp = run_cmd(cmd)
    if cp.returncode != 0 or not cp.stdout:
        return None
    raw = np.frombuffer(cp.stdout, dtype=np.uint8)
    frame_size = 64 * 64
    if raw.size < frame_size:
        return None
    n_frames = raw.size // frame_size
    frames = raw[: n_frames * frame_size].reshape(n_frames, frame_size).astype(np.float32) / 255.0

    fmean = frames.mean(axis=1)
    fstd = frames.std(axis=1)
    fdiff = np.diff(fmean) if n_frames > 1 else np.array([0.0], dtype=np.float32)

    feat = np.array(
        [
            float(np.mean(frames)),
            float(np.std(frames)),
            float(np.min(frames)),
            float(np.max(frames)),
            float(np.mean(fmean)),
            float(np.std(fmean)),
            float(np.percentile(fmean, 10)),
            float(np.percentile(fmean, 50)),
            float(np.percentile(fmean, 90)),
            float(np.mean(fstd)),
            float(np.std(fstd)),
            float(np.mean(np.abs(fdiff))),
            float(np.std(fdiff)),
            float(n_frames),
        ],
        dtype=np.float32,
    )
    return feat


class FeatureStore:
    def __init__(self, repo_root: Path, cache_dir: Path) -> None:
        self.repo_root = repo_root
        self.cache_dir = cache_dir
        (self.cache_dir / "audio").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "video").mkdir(parents=True, exist_ok=True)

    def _feature_path(self, clip_id: str, kind: str) -> Path:
        return self.cache_dir / kind / f"{safe_id(clip_id)}.npy"

    def _fail_path(self, clip_id: str, kind: str) -> Path:
        return self.cache_dir / kind / f"{safe_id(clip_id)}.fail"

    def get_audio(self, row: Row) -> Optional[np.ndarray]:
        return self._get(row, "audio")

    def get_video(self, row: Row) -> Optional[np.ndarray]:
        return self._get(row, "video")

    def _get(self, row: Row, kind: str) -> Optional[np.ndarray]:
        p = self._feature_path(row.clip_id, kind)
        pf = self._fail_path(row.clip_id, kind)
        if p.exists():
            return np.load(p)
        if pf.exists():
            return None

        if kind == "audio":
            if not row.path_audio:
                pf.write_text("missing path_audio\n", encoding="utf-8")
                return None
            media = self.repo_root / row.path_audio
            feat = extract_audio_feature(media)
        else:
            if not row.path_video:
                pf.write_text("missing path_video\n", encoding="utf-8")
                return None
            media = self.repo_root / row.path_video
            feat = extract_video_feature(media)

        if feat is None:
            pf.write_text("extract failed\n", encoding="utf-8")
            return None
        np.save(p, feat)
        return feat


def build_xy(
    rows: Sequence[Row],
    store: FeatureStore,
    mode: str,
    progress: Optional[ProgressTracker] = None,
    fold: Optional[int] = None,
    stage: str = "feature",
    progress_every: int = 250,
) -> Tuple[np.ndarray, List[Optional[str]], List[Optional[int]], List[Optional[int]], List[str]]:
    X: List[np.ndarray] = []
    y_emotion: List[Optional[str]] = []
    y_a2: List[Optional[int]] = []
    y_a3: List[Optional[int]] = []
    ids: List[str] = []

    total = len(rows)
    for idx, r in enumerate(rows, start=1):
        fa = store.get_audio(r)
        fv = store.get_video(r)

        if mode == "audio":
            if fa is None:
                continue
            feat = fa
        elif mode == "video":
            if fv is None:
                continue
            feat = fv
        elif mode == "fusion":
            if fa is None or fv is None:
                continue
            feat = np.concatenate([fa, fv], axis=0)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        X.append(feat)
        y_emotion.append(r.emotion6)
        y_a2.append(r.arousal2)
        y_a3.append(r.arousal3)
        ids.append(r.clip_id)
        if progress and (idx % progress_every == 0 or idx == total):
            progress.update(
                status="running",
                stage=stage,
                message="extracting features",
                fold=fold,
                mode=mode,
                current=idx,
                total=total,
            )

    if not X:
        return np.empty((0, 0), dtype=np.float32), y_emotion, y_a2, y_a3, ids
    return np.vstack(X), y_emotion, y_a2, y_a3, ids


def fit_classifier(X: np.ndarray, y: Sequence, classifier: str) -> object:
    if classifier == "logreg":
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        solver="lbfgs",
                        n_jobs=None,
                        random_state=42,
                    ),
                ),
            ]
        )
    elif classifier == "linear_svm":
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
    elif classifier == "rbf_svm":
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=2.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        )
    elif classifier == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    clf.fit(X, y)
    return clf


def _covariance(x: np.ndarray, eps: float) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape={x.shape}")
    n, d = x.shape
    if n <= 1:
        return np.eye(d, dtype=np.float64)
    c = (x.T @ x) / float(n - 1)
    c = c + (eps * np.eye(d, dtype=np.float64))
    return c


def _sqrtm_psd(mat: np.ndarray, eps: float, inverse: bool) -> np.ndarray:
    vals, vecs = np.linalg.eigh(mat)
    vals = np.maximum(vals, eps)
    if inverse:
        s = 1.0 / np.sqrt(vals)
    else:
        s = np.sqrt(vals)
    return (vecs * s) @ vecs.T


def coral_align_train_to_val(
    Xtr: np.ndarray, Xva: np.ndarray, eps: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CORAL (CORrelation ALignment):
    - Align source/train covariance to target/val covariance.
    - Uses target features only (no target labels).
    """
    if Xtr.ndim != 2 or Xva.ndim != 2 or Xtr.shape[1] != Xva.shape[1]:
        raise ValueError(
            f"CORAL shape mismatch: Xtr={Xtr.shape}, Xva={Xva.shape}"
        )
    if Xtr.shape[0] <= 1 or Xva.shape[0] <= 1:
        return Xtr, Xva

    Xs = Xtr.astype(np.float64, copy=False)
    Xt = Xva.astype(np.float64, copy=False)

    ms = Xs.mean(axis=0, keepdims=True)
    mt = Xt.mean(axis=0, keepdims=True)
    Xs0 = Xs - ms
    Xt0 = Xt - mt

    Cs = _covariance(Xs0, eps=eps)
    Ct = _covariance(Xt0, eps=eps)

    As = _sqrtm_psd(Cs, eps=eps, inverse=True)
    At = _sqrtm_psd(Ct, eps=eps, inverse=False)
    A = As @ At

    Xs_aligned = (Xs0 @ A) + mt
    return Xs_aligned.astype(np.float32), Xva.astype(np.float32, copy=False)


def unique_labels(y: Sequence) -> List:
    return sorted(set(y))


def metric_emotion(
    y_true: List[str],
    y_pred: List[str],
    y_proba: Optional[np.ndarray],
    labels: Optional[List[str]],
    n_bootstrap: int,
    seed: int,
) -> dict:
    acc = float(sk_acc(y_true, y_pred)) if y_true else None
    f1 = float(f1_score(y_true, y_pred, average="macro")) if y_true else None
    if y_true:
        acc_ci = bootstrap_ci(y_true, y_pred, sk_acc, n_boot=n_bootstrap, seed=seed)
        f1_ci = bootstrap_ci(
            y_true,
            y_pred,
            lambda a, b: f1_score(a, b, average="macro"),
            n_boot=n_bootstrap,
            seed=seed,
        )
    else:
        acc_ci = (None, None)
        f1_ci = (None, None)

    auc = None
    if y_true and y_proba is not None and labels is not None:
        try:
            if len(labels) == 2:
                # binary: pick the positive class column
                auc = float(roc_auc_score(y_true, y_proba[:, 1]))
            elif len(labels) > 2:
                auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", labels=labels))
        except Exception:
            auc = None

    return {
        "accuracy": acc,
        "accuracy_ci95": list(acc_ci),
        "macro_f1": f1,
        "macro_f1_ci95": list(f1_ci),
        "ovr_auc": auc,
        "n": len(y_true),
    }


def metric_arousal(
    y_true: List[int], y_pred: List[int], n_bootstrap: int, seed: int
) -> dict:
    if not y_true:
        return {"mae": None, "mae_ci95": [None, None], "n": 0}
    mae = float(mean_absolute_error(y_true, y_pred))
    lo, hi = bootstrap_ci(
        y_true, y_pred, mean_absolute_error, n_boot=n_bootstrap, seed=seed
    )
    return {"mae": mae, "mae_ci95": [lo, hi], "n": len(y_true)}


def main() -> None:
    args = parse_args()
    repo_root = Path(".").resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressTracker(out_dir=args.out_dir, enabled=not args.no_progress)
    store = FeatureStore(repo_root=repo_root, cache_dir=args.cache_dir)
    manifest = load_manifest(args.manifest)
    progress.update(
        status="running",
        stage="init",
        message=f"loaded manifest rows={len(manifest)}",
        current=1,
        total=1,
        force_line=True,
    )

    modes = [m.strip() for m in args.modalities.split(",") if m.strip()]
    valid_modes = {"audio", "video", "fusion"}
    if not set(modes).issubset(valid_modes):
        raise ValueError(f"--modalities must be subset of {sorted(valid_modes)}")

    if (args.train_list is None) ^ (args.val_list is None):
        raise ValueError("--train-list and --val-list must be provided together")

    predictions_csv = args.out_dir / "predictions.csv"
    pred_fp = predictions_csv.open("w", newline="", encoding="utf-8")
    pred_writer = csv.writer(pred_fp)
    pred_writer.writerow(
        [
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
        ]
    )

    all_metrics: Dict[str, List[dict]] = {m: [] for m in modes}
    global_rows: Dict[str, List[dict]] = {m: [] for m in modes}

    if args.train_list is not None and args.val_list is not None:
        split_defs: List[Tuple[int, Path, Path]] = [(0, args.train_list, args.val_list)]
    else:
        split_defs = []
        for fold in range(args.num_folds):
            train_file = args.fold_dir / f"fold_{fold}_train.txt"
            val_file = args.fold_dir / f"fold_{fold}_val.txt"
            split_defs.append((fold, train_file, val_file))

    total_stages = max(1, len(split_defs) * max(1, len(modes)))
    done_stages = 0

    for fold, train_file, val_file in split_defs:
        if not train_file.exists() or not val_file.exists():
            raise FileNotFoundError(
                f"Missing split files: train={train_file} val={val_file}"
            )

        train_ids = [cid for cid in load_ids(train_file) if cid in manifest]
        val_ids = [cid for cid in load_ids(val_file) if cid in manifest]

        train_ids = maybe_subsample(train_ids, args.max_train_per_fold, args.seed + fold * 17 + 1)
        val_ids = maybe_subsample(val_ids, args.max_val_per_fold, args.seed + fold * 17 + 2)

        train_rows = [manifest[cid] for cid in train_ids]
        val_rows = [manifest[cid] for cid in val_ids]
        progress.update(
            status="running",
            stage="split",
            message=f"loaded split train={len(train_rows)} val={len(val_rows)}",
            fold=fold,
            current=done_stages,
            total=total_stages,
            force_line=True,
        )

        for mode in modes:
            progress.update(
                status="running",
                stage="train_features",
                message="start train feature extraction",
                fold=fold,
                mode=mode,
                current=0,
                total=max(1, len(train_rows)),
                force_line=True,
            )
            Xtr, ytr_e, ytr_a2, ytr_a3, tr_clip_ids = build_xy(
                train_rows,
                store,
                mode,
                progress=progress,
                fold=fold,
                stage="train_features",
                progress_every=max(1, args.progress_every),
            )
            progress.update(
                status="running",
                stage="val_features",
                message="start val feature extraction",
                fold=fold,
                mode=mode,
                current=0,
                total=max(1, len(val_rows)),
                force_line=True,
            )
            Xva, yva_e, yva_a2, yva_a3, va_clip_ids = build_xy(
                val_rows,
                store,
                mode,
                progress=progress,
                fold=fold,
                stage="val_features",
                progress_every=max(1, args.progress_every),
            )

            fold_result = {
                "fold": fold,
                "mode": mode,
                "n_train": int(Xtr.shape[0]),
                "n_val": int(Xva.shape[0]),
            }

            if Xtr.size == 0 or Xva.size == 0:
                fold_result["error"] = "empty features"
                all_metrics[mode].append(fold_result)
                done_stages += 1
                progress.update(
                    status="running",
                    stage="model",
                    message="skipped (empty features)",
                    fold=fold,
                    mode=mode,
                    current=done_stages,
                    total=total_stages,
                    force_line=True,
                )
                continue

            if args.domain_adapt == "coral":
                progress.update(
                    status="running",
                    stage="domain_adapt",
                    message="applying CORAL(train->val) without target labels",
                    fold=fold,
                    mode=mode,
                    current=done_stages,
                    total=total_stages,
                    force_line=True,
                )
                Xtr, Xva = coral_align_train_to_val(
                    Xtr, Xva, eps=max(args.coral_eps, 1e-12)
                )

            progress.update(
                status="running",
                stage="model",
                message="training and evaluating",
                fold=fold,
                mode=mode,
                current=done_stages,
                total=total_stages,
                force_line=True,
            )

            # Emotion classifier
            tr_idx_e = [i for i, y in enumerate(ytr_e) if y is not None]
            va_idx_e = [i for i, y in enumerate(yva_e) if y is not None]
            y_pred_e: List[Optional[str]] = [None] * len(yva_e)
            y_prob_e = None
            labels_e = None
            if tr_idx_e and va_idx_e:
                Xtr_e = Xtr[tr_idx_e]
                ytr_e2 = [ytr_e[i] for i in tr_idx_e]
                Xva_e = Xva[va_idx_e]
                labels_train = unique_labels(ytr_e2)
                if len(labels_train) < 2:
                    const_label = str(labels_train[0])
                    for idx in va_idx_e:
                        y_pred_e[idx] = const_label
                    y_prob_e = None
                    labels_e = [const_label]
                else:
                    clf_e = fit_classifier(Xtr_e, ytr_e2, args.classifier)
                    pred_e = clf_e.predict(Xva_e)
                    for idx, yp in zip(va_idx_e, pred_e):
                        y_pred_e[idx] = str(yp)
                    try:
                        y_prob_e = clf_e.predict_proba(Xva_e)
                        if isinstance(clf_e, Pipeline):
                            labels_e = [str(c) for c in clf_e.named_steps["model"].classes_]
                        else:
                            labels_e = [str(c) for c in clf_e.classes_]
                    except Exception:
                        y_prob_e = None
                        labels_e = None

                metrics_e = metric_emotion(
                    y_true=[str(yva_e[i]) for i in va_idx_e if yva_e[i] is not None],
                    y_pred=[str(y_pred_e[i]) for i in va_idx_e if y_pred_e[i] is not None],
                    y_proba=y_prob_e,
                    labels=labels_e,
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed + fold,
                )
            else:
                metrics_e = {
                    "accuracy": None,
                    "accuracy_ci95": [None, None],
                    "macro_f1": None,
                    "macro_f1_ci95": [None, None],
                    "ovr_auc": None,
                    "n": 0,
                }

            # Arousal2 classifier
            tr_idx_a2 = [i for i, y in enumerate(ytr_a2) if y is not None]
            va_idx_a2 = [i for i, y in enumerate(yva_a2) if y is not None]
            y_pred_a2: List[Optional[int]] = [None] * len(yva_a2)
            if tr_idx_a2 and va_idx_a2:
                Xtr_a2 = Xtr[tr_idx_a2]
                ytr_a2_ = [int(ytr_a2[i]) for i in tr_idx_a2 if ytr_a2[i] is not None]
                Xva_a2 = Xva[va_idx_a2]
                labels_train = unique_labels(ytr_a2_)
                if len(labels_train) < 2:
                    const_label = int(labels_train[0])
                    for idx in va_idx_a2:
                        y_pred_a2[idx] = const_label
                else:
                    clf_a2 = fit_classifier(Xtr_a2, ytr_a2_, args.classifier)
                    pred_a2 = clf_a2.predict(Xva_a2)
                    for idx, yp in zip(va_idx_a2, pred_a2):
                        y_pred_a2[idx] = int(yp)
                metrics_a2 = metric_arousal(
                    y_true=[int(yva_a2[i]) for i in va_idx_a2 if yva_a2[i] is not None],
                    y_pred=[int(y_pred_a2[i]) for i in va_idx_a2 if y_pred_a2[i] is not None],
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed + 100 + fold,
                )
            else:
                metrics_a2 = {"mae": None, "mae_ci95": [None, None], "n": 0}

            # Arousal3 classifier
            tr_idx_a3 = [i for i, y in enumerate(ytr_a3) if y is not None]
            va_idx_a3 = [i for i, y in enumerate(yva_a3) if y is not None]
            y_pred_a3: List[Optional[int]] = [None] * len(yva_a3)
            if tr_idx_a3 and va_idx_a3:
                Xtr_a3 = Xtr[tr_idx_a3]
                ytr_a3_ = [int(ytr_a3[i]) for i in tr_idx_a3 if ytr_a3[i] is not None]
                Xva_a3 = Xva[va_idx_a3]
                labels_train = unique_labels(ytr_a3_)
                if len(labels_train) < 2:
                    const_label = int(labels_train[0])
                    for idx in va_idx_a3:
                        y_pred_a3[idx] = const_label
                else:
                    clf_a3 = fit_classifier(Xtr_a3, ytr_a3_, args.classifier)
                    pred_a3 = clf_a3.predict(Xva_a3)
                    for idx, yp in zip(va_idx_a3, pred_a3):
                        y_pred_a3[idx] = int(yp)
                metrics_a3 = metric_arousal(
                    y_true=[int(yva_a3[i]) for i in va_idx_a3 if yva_a3[i] is not None],
                    y_pred=[int(y_pred_a3[i]) for i in va_idx_a3 if y_pred_a3[i] is not None],
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed + 200 + fold,
                )
            else:
                metrics_a3 = {"mae": None, "mae_ci95": [None, None], "n": 0}

            fold_result["emotion6"] = metrics_e
            fold_result["arousal2"] = metrics_a2
            fold_result["arousal3"] = metrics_a3
            all_metrics[mode].append(fold_result)
            done_stages += 1
            progress.update(
                status="running",
                stage="model",
                message=(
                    f"done acc={metrics_e.get('accuracy')} "
                    f"f1={metrics_e.get('macro_f1')} "
                    f"a2_mae={metrics_a2.get('mae')}"
                ),
                fold=fold,
                mode=mode,
                current=done_stages,
                total=total_stages,
                force_line=True,
            )

            # store per-sample predictions
            va_map = {r.clip_id: r for r in val_rows}
            for i, clip_id in enumerate(va_clip_ids):
                vr = va_map[clip_id]
                row = {
                    "model_type": mode,
                    "fold": fold,
                    "clip_id": clip_id,
                    "dataset": vr.dataset,
                    "actor_id": vr.actor_id,
                    "y_true_emotion": yva_e[i],
                    "y_pred_emotion": y_pred_e[i],
                    "y_true_arousal2": yva_a2[i],
                    "y_pred_arousal2": y_pred_a2[i],
                    "y_true_arousal3": yva_a3[i],
                    "y_pred_arousal3": y_pred_a3[i],
                }
                global_rows[mode].append(row)
                pred_writer.writerow(
                    [
                        row["model_type"],
                        row["fold"],
                        row["clip_id"],
                        row["dataset"],
                        row["actor_id"],
                        row["y_true_emotion"],
                        row["y_pred_emotion"],
                        row["y_true_arousal2"],
                        row["y_pred_arousal2"],
                        row["y_true_arousal3"],
                        row["y_pred_arousal3"],
                    ]
                )

    pred_fp.close()

    # Global aggregate from per-sample predictions
    global_summary = {}
    for mode in modes:
        rows = global_rows[mode]
        yte = [r["y_true_emotion"] for r in rows if r["y_true_emotion"] is not None and r["y_pred_emotion"] is not None]
        ype = [r["y_pred_emotion"] for r in rows if r["y_true_emotion"] is not None and r["y_pred_emotion"] is not None]
        if yte:
            em = {
                "accuracy": float(sk_acc(yte, ype)),
                "macro_f1": float(f1_score(yte, ype, average="macro")),
                "n": len(yte),
            }
        else:
            em = {"accuracy": None, "macro_f1": None, "n": 0}

        yta2 = [int(r["y_true_arousal2"]) for r in rows if r["y_true_arousal2"] is not None and r["y_pred_arousal2"] is not None]
        ypa2 = [int(r["y_pred_arousal2"]) for r in rows if r["y_true_arousal2"] is not None and r["y_pred_arousal2"] is not None]
        if yta2:
            a2 = {"mae": float(mean_absolute_error(yta2, ypa2)), "n": len(yta2)}
        else:
            a2 = {"mae": None, "n": 0}

        yta3 = [int(r["y_true_arousal3"]) for r in rows if r["y_true_arousal3"] is not None and r["y_pred_arousal3"] is not None]
        ypa3 = [int(r["y_pred_arousal3"]) for r in rows if r["y_true_arousal3"] is not None and r["y_pred_arousal3"] is not None]
        if yta3:
            a3 = {"mae": float(mean_absolute_error(yta3, ypa3)), "n": len(yta3)}
        else:
            a3 = {"mae": None, "n": 0}

        global_summary[mode] = {"emotion6": em, "arousal2": a2, "arousal3": a3}

    summary = {
        "run": {
            "manifest": str(args.manifest),
            "fold_dir": str(args.fold_dir),
            "train_list": str(args.train_list) if args.train_list else None,
            "val_list": str(args.val_list) if args.val_list else None,
            "modalities": modes,
            "num_folds": args.num_folds,
            "classifier": args.classifier,
            "seed": args.seed,
            "n_bootstrap": args.n_bootstrap,
            "max_train_per_fold": args.max_train_per_fold,
            "max_val_per_fold": args.max_val_per_fold,
            "cache_dir": str(args.cache_dir),
            "domain_adapt": args.domain_adapt,
            "coral_eps": args.coral_eps,
        },
        "fold_metrics": all_metrics,
        "global_metrics": global_summary,
        "outputs": {"predictions_csv": str(predictions_csv)},
    }

    out_json = args.out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    progress.update(
        status="completed",
        stage="done",
        message=f"finished. summary={out_json}",
        current=total_stages,
        total=total_stages,
        force_line=True,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
