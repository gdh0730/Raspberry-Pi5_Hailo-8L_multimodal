#!/usr/bin/env python3
"""
Export FP32 multitask checkpoints to ONNX for Hailo E4 preparation.

Outputs:
- full multitask ONNX (2-input, 3-output)
- audio encoder ONNX
- video encoder ONNX
- export metadata JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_fp32_multitask as fp32  # noqa: E402


class FullWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, xa: torch.Tensor, xv: torch.Tensor):
        out = self.model(xa, xv)
        return out["emotion"], out["a2"], out["a3"]


class AudioEncoderWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.enc = model.audio_enc

    def forward(self, xa: torch.Tensor):
        return self.enc(xa)


class VideoEncoderWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.enc = model.video_enc

    def forward(self, xv: torch.Tensor):
        return self.enc(xv)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export FP32 checkpoint to ONNX")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("derived/results/fp32_multitask_phase35_v8_hubert_gated_wide_tune4"),
    )
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--onnx-dir", type=Path, default=Path("derived/hailo/onnx"))
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--name", type=str, default="")
    return p.parse_args()


def infer_arch_from_state(
    state: Dict[str, torch.Tensor], run_cfg: Dict[str, object], use_head_layernorm: bool
) -> Dict[str, object]:
    in_audio = int(state["audio_enc.0.weight"].shape[1])
    in_video = int(state["video_enc.0.weight"].shape[1])
    hidden_dim = int(state["audio_enc.0.weight"].shape[0])
    emb_dim = int(state["audio_enc.3.weight"].shape[0])
    n_emotions = int(state["head_emotion.weight"].shape[0])

    return {
        "in_audio": in_audio,
        "in_video": in_video,
        "hidden_dim": hidden_dim,
        "emb_dim": emb_dim,
        "n_emotions": n_emotions,
        "mode": str(run_cfg.get("mode", "fusion")),
        "fusion_type": str(run_cfg.get("fusion_type", "concat")),
        "modality_dropout_p": float(run_cfg.get("modality_dropout_p", 0.0)),
        "dropout": float(run_cfg.get("dropout", 0.2)),
        "use_head_layernorm": bool(use_head_layernorm),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "summary.json"
    ckpt_path = run_dir / "checkpoints" / f"best_fold_{args.fold}.pt"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json: {summary_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_cfg = summary.get("run", {})
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Backward compatibility for older torch versions without weights_only arg.
        payload = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected checkpoint format: {type(payload)}")
    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        state = payload["state_dict"]
    else:
        state = payload
    use_head_layernorm = bool(
        run_cfg.get(
            "use_head_layernorm",
            payload.get("use_head_layernorm", ("fuse.1.weight" in state and "fuse.1.bias" in state)),
        )
    )

    arch = infer_arch_from_state(state, run_cfg, use_head_layernorm=use_head_layernorm)
    model = fp32.DualEncoderMultiTask(
        in_audio=arch["in_audio"],
        in_video=arch["in_video"],
        n_emotions=arch["n_emotions"],
        mode=arch["mode"],
        fusion_type=arch["fusion_type"],
        modality_dropout_p=arch["modality_dropout_p"],
        emb_dim=arch["emb_dim"],
        hidden_dim=arch["hidden_dim"],
        dropout=arch["dropout"],
        use_head_layernorm=arch["use_head_layernorm"],
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    name = args.name.strip() or f"{run_dir.name}_fold{args.fold}"
    out_dir = args.onnx_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dummy_a = torch.randn(1, arch["in_audio"], dtype=torch.float32)
    dummy_v = torch.randn(1, arch["in_video"], dtype=torch.float32)

    full_path = out_dir / f"{name}_full.onnx"
    audio_path = out_dir / f"{name}_audio_encoder.onnx"
    video_path = out_dir / f"{name}_video_encoder.onnx"
    meta_path = out_dir / f"{name}_export_meta.json"

    full_wrapper = FullWrapper(model).eval()
    audio_wrapper = AudioEncoderWrapper(model).eval()
    video_wrapper = VideoEncoderWrapper(model).eval()

    torch.onnx.export(
        full_wrapper,
        (dummy_a, dummy_v),
        full_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["xa", "xv"],
        output_names=["emotion_logits", "arousal2_logits", "arousal3_logits"],
        dynamic_axes={
            "xa": {0: "batch"},
            "xv": {0: "batch"},
            "emotion_logits": {0: "batch"},
            "arousal2_logits": {0: "batch"},
            "arousal3_logits": {0: "batch"},
        },
    )

    torch.onnx.export(
        audio_wrapper,
        (dummy_a,),
        audio_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["xa"],
        output_names=["za"],
        dynamic_axes={"xa": {0: "batch"}, "za": {0: "batch"}},
    )

    torch.onnx.export(
        video_wrapper,
        (dummy_v,),
        video_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["xv"],
        output_names=["zv"],
        dynamic_axes={"xv": {0: "batch"}, "zv": {0: "batch"}},
    )

    meta = {
        "run_dir": str(run_dir),
        "summary_json": str(summary_path),
        "checkpoint": str(ckpt_path),
        "onnx_full": str(full_path),
        "onnx_audio_encoder": str(audio_path),
        "onnx_video_encoder": str(video_path),
        "opset": args.opset,
        "arch": arch,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(meta, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
