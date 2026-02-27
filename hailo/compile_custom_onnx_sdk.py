#!/usr/bin/env python3
"""Compile custom ONNX to HAR/HEF with hailo_sdk_client."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnx
from onnx import helper

from hailo_sdk_client import ClientRunner
from hailo_sdk_client.exposed_definitions import CalibrationDataType


def parse_csv_names(v: str) -> List[str]:
    out = [x.strip() for x in v.split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("Expected at least one name")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compile custom ONNX with Hailo SDK")
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--calib-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--network-name", type=str, required=True)
    p.add_argument("--hw-arch", type=str, default="hailo8l", choices=["hailo8", "hailo8l"])
    p.add_argument("--input-names", type=parse_csv_names, default=parse_csv_names("xa,xv"))
    p.add_argument(
        "--output-names",
        type=parse_csv_names,
        default=parse_csv_names("emotion_logits,arousal2_logits,arousal3_logits"),
    )
    p.add_argument("--audio-subdir", type=str, default="audio")
    p.add_argument("--video-subdir", type=str, default="video")
    p.add_argument("--max-calib", type=int, default=1024)
    p.add_argument("--optimization-level", type=int, default=0)
    p.add_argument("--compression-level", type=int, default=0)
    p.add_argument("--strip-layernorm", action="store_true")
    p.add_argument("--no-strip-layernorm", dest="strip_layernorm", action="store_false")
    p.set_defaults(strip_layernorm=True)
    return p.parse_args()


def load_calib_pair(audio_dir: Path, video_dir: Path, max_calib: int) -> Tuple[np.ndarray, np.ndarray]:
    audio_files = sorted(audio_dir.glob("*.npy"))
    video_files = sorted(video_dir.glob("*.npy"))
    if not audio_files:
        raise FileNotFoundError(f"No calibration npy files in {audio_dir}")
    if not video_files:
        raise FileNotFoundError(f"No calibration npy files in {video_dir}")
    n = min(len(audio_files), len(video_files), max_calib if max_calib > 0 else min(len(audio_files), len(video_files)))
    if n <= 0:
        raise RuntimeError("Calibration sample count is zero")

    xa = np.stack([np.load(p).reshape(-1).astype(np.float32) for p in audio_files[:n]], axis=0)
    xv = np.stack([np.load(p).reshape(-1).astype(np.float32) for p in video_files[:n]], axis=0)
    return xa, xv


def strip_layernorm_to_identity(src: Path, dst: Path) -> int:
    model = onnx.load(str(src))
    new_nodes = []
    replaced = 0
    for n in model.graph.node:
        if n.op_type == "LayerNormalization":
            new_nodes.append(
                helper.make_node(
                    "Identity",
                    inputs=[n.input[0]],
                    outputs=list(n.output),
                    name=(n.name or "layernorm") + "_identity",
                )
            )
            replaced += 1
        else:
            new_nodes.append(n)
    model.graph.ClearField("node")
    model.graph.node.extend(new_nodes)
    onnx.save(model, str(dst))
    return replaced


def main() -> None:
    args = parse_args()
    onnx_path = args.onnx.resolve()
    calib_dir = args.calib_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX: {onnx_path}")
    if not calib_dir.exists():
        raise FileNotFoundError(f"Missing calib dir: {calib_dir}")
    if len(args.input_names) != 2:
        raise ValueError(f"Expected exactly 2 input names, got: {args.input_names}")
    requested_input_names = list(args.input_names)
    requested_output_names = list(args.output_names)

    audio_dir = calib_dir / args.audio_subdir
    video_dir = calib_dir / args.video_subdir
    xa, xv = load_calib_pair(audio_dir, video_dir, args.max_calib)
    print(f"[INFO] calibration audio={xa.shape}, video={xv.shape}")

    parse_onnx = onnx_path
    stripped_count = 0
    if args.strip_layernorm:
        parse_onnx = out_dir / f"{args.network_name}_parse_ready.onnx"
        stripped_count = strip_layernorm_to_identity(onnx_path, parse_onnx)
        print(f"[INFO] LayerNormalization -> Identity: {stripped_count} node(s)")
    else:
        print("[INFO] LayerNormalization strip disabled")

    hef_path = out_dir / f"{args.network_name}.hef"
    har_path = out_dir / f"{args.network_name}.har"

    runner = ClientRunner(hw_arch=args.hw_arch)
    runner.translate_onnx_model(
        str(parse_onnx),
        net_name=args.network_name,
        start_node_names=list(requested_input_names),
        end_node_names=list(requested_output_names),
    )

    input_layers = [l.name for l in runner.get_hn_model().get_input_layers()]
    if len(input_layers) != 2:
        raise RuntimeError(f"Expected 2 parsed input layers, got: {input_layers}")

    runner.load_model_script(
        f"model_optimization_flavor(optimization_level={args.optimization_level}, "
        f"compression_level={args.compression_level})"
    )
    runner.optimize(
        {input_layers[0]: xa, input_layers[1]: xv},
        data_type=CalibrationDataType.np_array,
    )
    hef_bytes = runner.compile()
    hef_path.write_bytes(hef_bytes)
    runner.save_har(str(har_path))

    meta = {
        "network_name": args.network_name,
        "onnx_input": str(onnx_path),
        "onnx_parse_ready": str(parse_onnx),
        "strip_layernorm": bool(args.strip_layernorm),
        "layernorm_nodes_replaced": int(stripped_count),
        "calib_audio_shape": list(xa.shape),
        "calib_video_shape": list(xv.shape),
        "parsed_input_layers": input_layers,
        "requested_input_names": requested_input_names,
        "requested_output_names": requested_output_names,
        "hw_arch": args.hw_arch,
        "optimization_level": args.optimization_level,
        "compression_level": args.compression_level,
        "har": str(har_path),
        "hef": str(hef_path),
    }
    (out_dir / "compile_meta.json").write_text(json.dumps(meta, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] HAR: {har_path}")
    print(f"[OK] HEF: {hef_path}")


if __name__ == "__main__":
    main()
