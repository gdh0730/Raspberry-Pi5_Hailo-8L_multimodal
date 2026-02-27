#!/usr/bin/env python3
"""Single-sample inference runner for Hailo HEF (2-input AV model)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from hailo_platform import (
    ConfigureParams,
    FormatType,
    HailoStreamInterface,
    HEF,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)


DEFAULT_EMOTION_CLASSES = ["angry", "disgust", "fearful", "happy", "neutral", "sad"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one AV inference on Hailo HEF")
    p.add_argument("--hef", type=Path, required=True)
    p.add_argument("--audio-npy", type=Path, required=True)
    p.add_argument("--video-npy", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--emotion-classes", type=str, default=",".join(DEFAULT_EMOTION_CLASSES))
    p.add_argument("--strict-shape", action="store_true")
    return p.parse_args()


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    s = np.sum(e)
    return (e / s).astype(np.float64) if s > 0 else np.zeros_like(e, dtype=np.float64)


def load_feature(path: Path) -> np.ndarray:
    arr = np.load(path)
    return arr.astype(np.float32).reshape(-1)


def pick_input_mapping(
    input_infos,
    audio_vec: np.ndarray,
    video_vec: np.ndarray,
    strict_shape: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    if len(input_infos) != 2:
        raise RuntimeError(f"Expected exactly 2 inputs in HEF, got {len(input_infos)}")

    i0, i1 = input_infos[0], input_infos[1]
    shp0 = tuple(i0.shape)
    shp1 = tuple(i1.shape)
    c0 = int(shp0[-1])
    c1 = int(shp1[-1])
    la = int(audio_vec.shape[0])
    lv = int(video_vec.shape[0])

    if strict_shape and (la not in (c0, c1) or lv not in (c0, c1)):
        raise RuntimeError(
            f"Strict mode: feature dims do not match HEF input dims. "
            f"audio={la}, video={lv}, hef_inputs=({c0},{c1})"
        )

    # Prefer channel-size matching when possible; fall back to input order.
    if la == c0 and lv == c1:
        mapping = {i0.name: audio_vec.reshape((1, *shp0)), i1.name: video_vec.reshape((1, *shp1))}
        note = "matched_by_channel_size"
    elif la == c1 and lv == c0:
        mapping = {i0.name: video_vec.reshape((1, *shp0)), i1.name: audio_vec.reshape((1, *shp1))}
        note = "matched_by_channel_size_swapped"
    else:
        mapping = {
            i0.name: audio_vec[:c0].reshape((1, *shp0)),
            i1.name: video_vec[:c1].reshape((1, *shp1)),
        }
        note = "fallback_by_order_clipped"

    meta = {
        "input0": {"name": i0.name, "shape": [1, *list(shp0)], "channels": c0},
        "input1": {"name": i1.name, "shape": [1, *list(shp1)], "channels": c1},
        "audio_len": la,
        "video_len": lv,
        "mapping_note": note,
    }
    return mapping, meta


def to_flat_logits(outputs: Dict[str, np.ndarray], expected_classes: List[str]) -> Dict[str, Dict[str, object]]:
    out_items = sorted(outputs.items(), key=lambda kv: int(np.prod(kv[1].shape)))
    parsed: Dict[str, Dict[str, object]] = {}

    for name, arr in out_items:
        vec = arr.reshape(-1).astype(np.float64)
        probs = softmax(vec)
        idx = int(np.argmax(vec)) if vec.size else -1
        parsed[name] = {
            "shape": list(arr.shape),
            "logits": vec.tolist(),
            "probs": probs.tolist(),
            "argmax_index": idx,
        }
        if vec.size == len(expected_classes):
            parsed[name]["argmax_label"] = expected_classes[idx]
    return parsed


def main() -> None:
    args = parse_args()
    classes = [x.strip() for x in args.emotion_classes.split(",") if x.strip()]
    if not classes:
        classes = DEFAULT_EMOTION_CLASSES

    if not args.hef.exists():
        raise FileNotFoundError(f"Missing HEF: {args.hef}")
    if not args.audio_npy.exists():
        raise FileNotFoundError(f"Missing audio npy: {args.audio_npy}")
    if not args.video_npy.exists():
        raise FileNotFoundError(f"Missing video npy: {args.video_npy}")

    audio_vec = load_feature(args.audio_npy)
    video_vec = load_feature(args.video_npy)

    hef = HEF(str(args.hef))
    in_infos = hef.get_input_vstream_infos()
    out_infos = hef.get_output_vstream_infos()
    infer_inputs, infer_meta = pick_input_mapping(in_infos, audio_vec, video_vec, args.strict_shape)

    with VDevice() as target:
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, cfg)[0]
        ng_params = network_group.create_params()
        in_params = InputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.FLOAT32
        )
        out_params = OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.FLOAT32
        )
        with InferVStreams(network_group, in_params, out_params) as pipe:
            with network_group.activate(ng_params):
                outputs = pipe.infer(infer_inputs)

    parsed_outputs = to_flat_logits(outputs, classes)

    result = {
        "hef": str(args.hef),
        "audio_npy": str(args.audio_npy),
        "video_npy": str(args.video_npy),
        "emotion_classes": classes,
        "inputs": infer_meta,
        "output_vstreams": [o.name for o in out_infos],
        "outputs": parsed_outputs,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "output_keys": list(parsed_outputs.keys())}, ensure_ascii=True))


if __name__ == "__main__":
    main()
