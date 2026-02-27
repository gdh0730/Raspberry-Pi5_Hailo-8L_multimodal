#!/usr/bin/env python3
"""
Prepare richer multimodal features (cache_v2) from raw media.

Outputs:
- <cache_dir>/audio/<hash>.npy
- <cache_dir>/video/<hash>.npy
- <cache_dir>/summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import train_ml_baselines as tmb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare advanced audio/video features")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("derived/manifests/manifest_multimodal_common6_av.jsonl"),
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("derived/features/cache_v2"),
    )
    p.add_argument(
        "--source-cache-dir",
        type=Path,
        default=Path("derived/features/cache_v1"),
        help="Existing cache to lift into richer v2 features (fast path).",
    )
    p.add_argument(
        "--prefer-source-cache",
        dest="prefer_source_cache",
        action="store_true",
        help="Prefer using source cache entries when present.",
    )
    p.add_argument(
        "--no-prefer-source-cache",
        dest="prefer_source_cache",
        action="store_false",
        help="Do not use source cache; force raw extraction path.",
    )
    p.add_argument(
        "--fallback-raw",
        action="store_true",
        help="If source cache entry is missing, decode raw media and extract v2 feature.",
    )
    p.add_argument(
        "--video-pretrained-backbone",
        type=str,
        default="none",
        choices=["none", "resnet18", "resnet34", "efficientnet_b0"],
        help="Append pretrained video embedding.",
    )
    p.add_argument(
        "--audio-pretrained-backbone",
        type=str,
        default="none",
        choices=["none", "wav2vec2_base", "hubert_base", "wavlm_base_plus"],
        help="Append pretrained audio embedding.",
    )
    p.add_argument(
        "--audio-pretrained-max-samples",
        type=int,
        default=32000,
        help="Max audio samples (at 16kHz) used for pretrained audio embedding.",
    )
    p.add_argument("--video-pretrained-frames", type=int, default=8)
    p.add_argument("--video-pretrained-size", type=int, default=224)
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--kind", type=str, default="both", choices=["audio", "video", "both"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=250)
    p.set_defaults(prefer_source_cache=True)
    return p.parse_args()


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


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


def safe_id(clip_id: str) -> str:
    return hashlib.md5(clip_id.encode("utf-8")).hexdigest()


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    mel_pts = np.linspace(hz_to_mel(np.array([fmin]))[0], hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        if c <= l or r <= c:
            continue
        left = np.logical_and(freqs >= l, freqs <= c)
        right = np.logical_and(freqs >= c, freqs <= r)
        fb[i, left] = (freqs[left] - l) / (c - l + 1e-12)
        fb[i, right] = (r - freqs[right]) / (r - c + 1e-12)
    # normalize each filter to unit area-ish
    s = fb.sum(axis=1, keepdims=True)
    s = np.where(s <= 1e-8, 1.0, s)
    return (fb / s).astype(np.float32)


def extract_audio_pcm(path_audio: Path) -> Optional[np.ndarray]:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path_audio),
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
    if wav.size < 640:
        return None
    return wav


def audio_feature_v2(wav: np.ndarray) -> Optional[np.ndarray]:
    sr = 16000
    n_fft = 512
    win = 400
    hop = 160
    n_mels = 80
    if wav.size < win:
        return None

    # simple peak normalization
    peak = float(np.max(np.abs(wav)) + 1e-8)
    wav = (wav / peak).astype(np.float32)

    n_frames = 1 + (wav.size - win) // hop
    if n_frames < 2:
        return None
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = wav[idx]
    window = np.hanning(win).astype(np.float32)[None, :]
    frames = frames * window
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (np.abs(spec) ** 2).astype(np.float32)

    fb = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=20.0, fmax=8000.0)
    mel = np.maximum(power @ fb.T, 1e-8)
    logmel = np.log(mel).astype(np.float32)
    d1 = np.diff(logmel, axis=0, prepend=logmel[:1])

    mel_mean = logmel.mean(axis=0)
    mel_std = logmel.std(axis=0)
    d1_mean = d1.mean(axis=0)
    d1_std = d1.std(axis=0)

    frame_energy = np.mean(frames**2, axis=1)
    e_mean = float(frame_energy.mean())
    e_std = float(frame_energy.std())
    e_p50 = float(np.percentile(frame_energy, 50))
    e_p90 = float(np.percentile(frame_energy, 90))
    thr = max(e_p50 * 0.5, 1e-8)
    voiced_ratio = float(np.mean(frame_energy > thr))

    zcr = float(np.mean((wav[:-1] * wav[1:]) < 0)) if wav.size > 1 else 0.0
    rms = float(np.sqrt(np.mean(wav**2)))

    extra = np.array(
        [e_mean, e_std, e_p50, e_p90, voiced_ratio, zcr, rms, float(n_frames)],
        dtype=np.float32,
    )
    feat = np.concatenate([mel_mean, mel_std, d1_mean, d1_std, extra], axis=0).astype(np.float32)
    return feat


def extract_video_rgb(path_video: Path) -> Optional[np.ndarray]:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path_video),
        "-t",
        "2",
        "-vf",
        "fps=8,scale=96:96:flags=bicubic,format=rgb24",
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
    frame_size = 96 * 96 * 3
    if raw.size < frame_size:
        return None
    n_frames = raw.size // frame_size
    rgb = raw[: n_frames * frame_size].reshape(n_frames, 96, 96, 3).astype(np.float32) / 255.0
    return rgb


def extract_video_rgb_for_pretrained(
    path_video: Path,
    frames: int = 8,
    size: int = 224,
) -> Optional[np.ndarray]:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path_video),
        "-t",
        "2",
        "-vf",
        f"fps=8,scale={size}:{size}:flags=bicubic,format=rgb24",
        "-frames:v",
        str(frames),
        "-f",
        "rawvideo",
        "-",
    ]
    cp = run_cmd(cmd)
    if cp.returncode != 0 or not cp.stdout:
        return None
    raw = np.frombuffer(cp.stdout, dtype=np.uint8)
    frame_size = size * size * 3
    if raw.size < frame_size:
        return None
    n_frames = raw.size // frame_size
    rgb = raw[: n_frames * frame_size].reshape(n_frames, size, size, 3).astype(np.float32) / 255.0
    return rgb


def video_feature_v2(rgb: np.ndarray) -> Optional[np.ndarray]:
    if rgb.size == 0 or rgb.shape[0] < 2:
        return None
    ch_mean = rgb.mean(axis=(0, 1, 2))
    ch_std = rgb.std(axis=(0, 1, 2))

    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    b_ts = gray.mean(axis=(1, 2))
    c_ts = gray.std(axis=(1, 2))

    diff = np.diff(gray, axis=0)
    motion = np.abs(diff).mean(axis=(1, 2))

    gx = np.diff(gray, axis=2)
    gy = np.diff(gray, axis=1)
    edge = np.sqrt(gx[:, :-1, :] ** 2 + gy[:, :, :-1] ** 2)
    edge_ts = edge.mean(axis=(1, 2))

    stats = np.array(
        [
            float(b_ts.mean()),
            float(b_ts.std()),
            float(np.percentile(b_ts, 10)),
            float(np.percentile(b_ts, 50)),
            float(np.percentile(b_ts, 90)),
            float(c_ts.mean()),
            float(c_ts.std()),
            float(motion.mean()),
            float(motion.std()),
            float(np.percentile(motion, 90)),
            float(edge_ts.mean()),
            float(edge_ts.std()),
            float(np.percentile(edge_ts, 90)),
            float(rgb.shape[0]),
        ],
        dtype=np.float32,
    )
    feat = np.concatenate([ch_mean.astype(np.float32), ch_std.astype(np.float32), stats], axis=0)
    return feat


class VideoPretrainedEmbedder:
    def __init__(self, backbone: str = "none", device: Optional[torch.device] = None) -> None:
        self.backbone = backbone
        self.device = torch.device("cpu") if device is None else device
        self.model: Optional[nn.Module] = None
        self.mean = None
        self.std = None
        if backbone == "none":
            return
        if backbone in {"resnet18", "resnet34"}:
            try:
                from torchvision.models import (
                    ResNet18_Weights,
                    ResNet34_Weights,
                    resnet18,
                    resnet34,
                )
            except Exception as e:
                raise RuntimeError(
                    "torchvision is required for pretrained video embedding. "
                    "Install torchvision in .venv."
                ) from e
            if backbone == "resnet18":
                weights = ResNet18_Weights.DEFAULT
                m = resnet18(weights=weights)
            else:
                weights = ResNet34_Weights.DEFAULT
                m = resnet34(weights=weights)
            m.fc = nn.Identity()
            m = m.to(self.device)
            m.eval()
            self.model = m
            self.mean = torch.tensor(weights.transforms().mean, dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
            self.std = torch.tensor(weights.transforms().std, dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
        elif backbone == "efficientnet_b0":
            try:
                from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
            except Exception as e:
                raise RuntimeError(
                    "torchvision is required for pretrained video embedding. "
                    "Install torchvision in .venv."
                ) from e
            weights = EfficientNet_B0_Weights.DEFAULT
            m = efficientnet_b0(weights=weights)
            m.classifier = nn.Identity()
            m = m.to(self.device)
            m.eval()
            self.model = m
            self.mean = torch.tensor(weights.transforms().mean, dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
            self.std = torch.tensor(weights.transforms().std, dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def embed(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        if rgb.size == 0:
            return None
        x = torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous().float().to(self.device, non_blocking=True)
        x = (x - self.mean) / self.std
        with torch.no_grad():
            y = self.model(x).cpu().numpy().astype(np.float32)
        if y.ndim != 2 or y.shape[0] == 0:
            return None
        return np.concatenate([y.mean(axis=0), y.std(axis=0)], axis=0).astype(np.float32)


class AudioPretrainedEmbedder:
    def __init__(self, backbone: str = "none", device: Optional[torch.device] = None) -> None:
        self.backbone = backbone
        self.device = torch.device("cpu") if device is None else device
        self.model = None
        self.sample_rate = 16000
        if backbone == "none":
            return
        if backbone in {"wav2vec2_base", "hubert_base", "wavlm_base_plus"}:
            try:
                import torchaudio
            except Exception as e:
                raise RuntimeError(
                    "torchaudio is required for pretrained audio embedding. "
                    "Install torchaudio in .venv."
                ) from e
            if backbone == "wav2vec2_base":
                bundle = torchaudio.pipelines.WAV2VEC2_BASE
            elif backbone == "hubert_base":
                bundle = torchaudio.pipelines.HUBERT_BASE
            else:
                bundle = torchaudio.pipelines.WAVLM_BASE_PLUS
            self.sample_rate = int(bundle.sample_rate)
            self.model = bundle.get_model().to(self.device)
            self.model.eval()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def embed(self, wav: np.ndarray, max_samples: int = 32000) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        x = np.asarray(wav, dtype=np.float32).reshape(-1).copy()
        if x.size == 0:
            return None
        max_n = max(1, int(max_samples))
        if x.size > max_n:
            x = x[:max_n]
        t = torch.from_numpy(x).view(1, -1).float().to(self.device, non_blocking=True)
        with torch.no_grad():
            # torchaudio API compatibility across versions.
            try:
                feats = self.model.extract_features(t)
                if isinstance(feats, tuple):
                    feats = feats[0]
                if isinstance(feats, list):
                    h = feats[-1]
                else:
                    h = feats
            except Exception:
                out = self.model(t)
                if isinstance(out, tuple):
                    h = out[0]
                else:
                    h = out
            if h.ndim == 3:
                y = h[0].cpu().numpy().astype(np.float32)
            elif h.ndim == 2:
                y = h.cpu().numpy().astype(np.float32)
            else:
                return None
        if y.size == 0:
            return None
        return np.concatenate([y.mean(axis=0), y.std(axis=0)], axis=0).astype(np.float32)


def lift_audio_from_v1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x2 = x * x
    xa = np.sqrt(np.abs(x) + 1e-8).astype(np.float32)
    inter = (x[::2] * x[1::2]).astype(np.float32)
    return np.concatenate([x, x2, xa, inter], axis=0).astype(np.float32)


def lift_video_from_v1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x2 = x * x
    xa = np.sqrt(np.abs(x) + 1e-8).astype(np.float32)
    inter = (x[::2] * x[1::2]).astype(np.float32)
    return np.concatenate([x, x2, xa, inter], axis=0).astype(np.float32)


def process_row(
    row: tmb.Row,
    repo_root: Path,
    cache_dir: Path,
    source_cache_dir: Path,
    prefer_source_cache: bool,
    fallback_raw: bool,
    audio_embedder: Optional[AudioPretrainedEmbedder],
    audio_pretrained_max_samples: int,
    video_embedder: Optional[VideoPretrainedEmbedder],
    video_pretrained_frames: int,
    video_pretrained_size: int,
    kind: str,
    overwrite: bool,
) -> Tuple[bool, str]:
    audio_dir = cache_dir / "audio"
    video_dir = cache_dir / "video"
    audio_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    sid = safe_id(row.clip_id)

    if kind in {"audio", "both"}:
        pa = audio_dir / f"{sid}.npy"
        pf = audio_dir / f"{sid}.fail"
        if overwrite or (not pa.exists() and not pf.exists()):
            src = source_cache_dir / "audio" / f"{sid}.npy"
            fa = None
            if prefer_source_cache and src.exists():
                fa = lift_audio_from_v1(np.load(src))
            elif fallback_raw:
                if not row.path_audio:
                    pf.write_text("missing path_audio\n", encoding="utf-8")
                    return False, "audio_missing_path"
                wav = extract_audio_pcm(repo_root / row.path_audio)
                if wav is None:
                    pf.write_text("decode_fail\n", encoding="utf-8")
                    return False, "audio_decode_fail"
                fa = audio_feature_v2(wav)
                if fa is not None and audio_embedder is not None and audio_embedder.model is not None:
                    emb = audio_embedder.embed(wav, max_samples=audio_pretrained_max_samples)
                    if emb is not None:
                        fa = np.concatenate([fa, emb], axis=0).astype(np.float32)
            else:
                pf.write_text("missing source cache\n", encoding="utf-8")
                return False, "audio_missing_source_cache"
            if fa is None:
                pf.write_text("feature_fail\n", encoding="utf-8")
                return False, "audio_feature_fail"
            np.save(pa, fa)
            if pf.exists():
                pf.unlink(missing_ok=True)

    if kind in {"video", "both"}:
        pv = video_dir / f"{sid}.npy"
        pf = video_dir / f"{sid}.fail"
        if overwrite or (not pv.exists() and not pf.exists()):
            src = source_cache_dir / "video" / f"{sid}.npy"
            fv = None
            if prefer_source_cache and src.exists():
                fv = lift_video_from_v1(np.load(src))
            elif fallback_raw:
                if not row.path_video:
                    pf.write_text("missing path_video\n", encoding="utf-8")
                    return False, "video_missing_path"
                rgb = extract_video_rgb(repo_root / row.path_video)
                if rgb is None:
                    pf.write_text("decode_fail\n", encoding="utf-8")
                    return False, "video_decode_fail"
                fv = video_feature_v2(rgb)
                if fv is not None and video_embedder is not None and video_embedder.model is not None:
                    rgbp = extract_video_rgb_for_pretrained(
                        repo_root / row.path_video,
                        frames=video_pretrained_frames,
                        size=video_pretrained_size,
                    )
                    if rgbp is not None:
                        emb = video_embedder.embed(rgbp)
                        if emb is not None:
                            fv = np.concatenate([fv, emb], axis=0).astype(np.float32)
            else:
                pf.write_text("missing source cache\n", encoding="utf-8")
                return False, "video_missing_source_cache"
            if fv is None:
                pf.write_text("feature_fail\n", encoding="utf-8")
                return False, "video_feature_fail"
            np.save(pv, fv)
            if pf.exists():
                pf.unlink(missing_ok=True)

    return True, "ok"


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    manifest = tmb.load_manifest(args.manifest)
    rows = list(manifest.values())
    if args.max_items > 0:
        rows = rows[: args.max_items]
    audio_embedder = AudioPretrainedEmbedder(args.audio_pretrained_backbone, device=device)
    video_embedder = VideoPretrainedEmbedder(args.video_pretrained_backbone, device=device)

    t0 = time.time()
    ok = 0
    fail = 0
    fail_reasons: Dict[str, int] = {}
    total = len(rows)

    for i, row in enumerate(rows, start=1):
        good, reason = process_row(
            row=row,
            repo_root=repo_root,
            cache_dir=cache_dir,
            source_cache_dir=args.source_cache_dir,
            prefer_source_cache=args.prefer_source_cache,
            fallback_raw=args.fallback_raw,
            audio_embedder=audio_embedder,
            audio_pretrained_max_samples=args.audio_pretrained_max_samples,
            video_embedder=video_embedder,
            video_pretrained_frames=args.video_pretrained_frames,
            video_pretrained_size=args.video_pretrained_size,
            kind=args.kind,
            overwrite=args.overwrite,
        )
        if good:
            ok += 1
        else:
            fail += 1
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

        if i % max(1, args.progress_every) == 0 or i == total:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            print(
                f"[features_v2] {i}/{total} ok={ok} fail={fail} "
                f"elapsed={elapsed:.1f}s rate={rate:.2f}/s device={device}",
                flush=True,
            )

    summary = {
        "manifest": str(args.manifest),
        "cache_dir": str(cache_dir),
        "kind": args.kind,
        "overwrite": args.overwrite,
        "source_cache_dir": str(args.source_cache_dir),
        "prefer_source_cache": args.prefer_source_cache,
        "fallback_raw": args.fallback_raw,
        "audio_pretrained_backbone": args.audio_pretrained_backbone,
        "audio_pretrained_max_samples": args.audio_pretrained_max_samples,
        "video_pretrained_backbone": args.video_pretrained_backbone,
        "video_pretrained_frames": args.video_pretrained_frames,
        "video_pretrained_size": args.video_pretrained_size,
        "device_requested": args.device,
        "device_resolved": str(device),
        "cuda_name": (torch.cuda.get_device_name(0) if device.type == "cuda" else None),
        "total_rows": total,
        "ok": ok,
        "fail": fail,
        "fail_reasons": fail_reasons,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    (cache_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
