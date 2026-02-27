#!/usr/bin/env python3
"""Evaluate Pi-side Hailo inference JSON outputs against manifest labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from research_metrics import accuracy_score, macro_f1_score, mean_absolute_error


DEFAULT_EMOTION_CLASSES = ["angry", "disgust", "fearful", "happy", "neutral", "sad"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Hailo Pi inference JSON directory")
    p.add_argument("--index-csv", type=Path, default=Path("derived/hailo/calib/fold0_train_1024/index.csv"))
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("derived/manifests/manifest_multimodal_common6_av.jsonl"),
    )
    p.add_argument("--json-dir", type=Path, required=True)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=0, help="0 means all")
    p.add_argument("--emotion-classes", type=str, default=",".join(DEFAULT_EMOTION_CLASSES))
    p.add_argument("--out-pred-csv", type=Path, required=True)
    p.add_argument("--out-summary-json", type=Path, required=True)
    return p.parse_args()


def parse_optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "none" or s.lower() == "null":
        return None
    return int(s)


def load_manifest(path: Path) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row["clip_id"])] = row
    return out


def iter_index_rows(path: Path, start_idx: int, max_samples: int) -> Iterable[Dict[str, str]]:
    emitted = 0
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["idx"])
            if idx < start_idx:
                continue
            yield row
            emitted += 1
            if max_samples > 0 and emitted >= max_samples:
                break


def argmax(values: List[float]) -> Optional[int]:
    if not values:
        return None
    best_i = 0
    best_v = float(values[0])
    for i in range(1, len(values)):
        if float(values[i]) > best_v:
            best_v = float(values[i])
            best_i = i
    return best_i


def extract_preds(
    infer_obj: Dict[str, object],
    emotion_classes: List[str],
) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    outputs = infer_obj.get("outputs")
    if not isinstance(outputs, dict):
        return None, None, None

    emo_idx: Optional[int] = None
    emo_label: Optional[str] = None
    a2_idx: Optional[int] = None
    a3_idx: Optional[int] = None
    n_emo = len(emotion_classes)

    for _, out in outputs.items():
        if not isinstance(out, dict):
            continue
        logits = out.get("logits")
        if not isinstance(logits, list):
            continue
        n = len(logits)
        out_idx = out.get("argmax_index")
        if out_idx is None:
            out_idx = argmax([float(v) for v in logits])
        else:
            out_idx = int(out_idx)

        if n == n_emo and emo_idx is None:
            emo_idx = out_idx
            if isinstance(out.get("argmax_label"), str):
                emo_label = str(out["argmax_label"])
        elif n == 2 and a2_idx is None:
            a2_idx = out_idx
        elif n == 3 and a3_idx is None:
            a3_idx = out_idx

    if emo_label is None and emo_idx is not None and 0 <= emo_idx < n_emo:
        emo_label = emotion_classes[emo_idx]

    return emo_label, a2_idx, a3_idx


def safe_metric_emotion(rows: List[Dict[str, str]]) -> Dict[str, object]:
    pairs = [
        (r["y_true_emotion"], r["y_pred_emotion"])
        for r in rows
        if r.get("y_true_emotion") and r.get("y_pred_emotion")
    ]
    if not pairs:
        return {"accuracy": None, "macro_f1": None, "n": 0}
    yt = [p[0] for p in pairs]
    yp = [p[1] for p in pairs]
    return {
        "accuracy": accuracy_score(yt, yp),
        "macro_f1": macro_f1_score(yt, yp),
        "n": len(pairs),
    }


def safe_metric_mae(rows: List[Dict[str, str]], y_true_key: str, y_pred_key: str) -> Dict[str, object]:
    yt: List[Optional[int]] = [parse_optional_int(r.get(y_true_key)) for r in rows]
    yp: List[Optional[int]] = [parse_optional_int(r.get(y_pred_key)) for r in rows]
    valid = [(a, b) for a, b in zip(yt, yp) if a is not None and b is not None]
    if not valid:
        return {"mae": None, "n": 0}
    a = [x for x, _ in valid]
    b = [x for _, x in valid]
    return {"mae": mean_absolute_error(a, b), "n": len(valid)}


def main() -> None:
    args = parse_args()
    classes = [x.strip() for x in args.emotion_classes.split(",") if x.strip()]
    if not classes:
        classes = DEFAULT_EMOTION_CLASSES

    manifest = load_manifest(args.manifest)
    rows_out: List[Dict[str, str]] = []
    missing_json = 0
    parsed_json = 0
    pred_missing = 0

    for row in iter_index_rows(args.index_csv, args.start_idx, args.max_samples):
        idx = int(row["idx"])
        clip_id = row["clip_id"]
        dataset = row.get("dataset") or ""
        m = manifest.get(clip_id, {})

        json_path = args.json_dir / f"{idx:05d}.json"
        pred_emotion = ""
        pred_a2 = ""
        pred_a3 = ""

        if json_path.exists():
            parsed_json += 1
            infer_obj = json.loads(json_path.read_text(encoding="utf-8"))
            p_emo, p_a2, p_a3 = extract_preds(infer_obj, classes)
            if p_emo is not None:
                pred_emotion = p_emo
            if p_a2 is not None:
                pred_a2 = str(int(p_a2))
            if p_a3 is not None:
                pred_a3 = str(int(p_a3))
        else:
            missing_json += 1

        if not pred_emotion:
            pred_missing += 1

        rows_out.append(
            {
                "idx": str(idx),
                "clip_id": clip_id,
                "dataset": str(dataset),
                "pred_json": str(json_path),
                "y_true_emotion": str(m.get("emotion6", "")),
                "y_pred_emotion": pred_emotion,
                "y_true_arousal2": "" if m.get("arousal2") is None else str(m.get("arousal2")),
                "y_pred_arousal2": pred_a2,
                "y_true_arousal3": "" if m.get("arousal3") is None else str(m.get("arousal3")),
                "y_pred_arousal3": pred_a3,
            }
        )

    args.out_pred_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_pred_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "clip_id",
                "dataset",
                "pred_json",
                "y_true_emotion",
                "y_pred_emotion",
                "y_true_arousal2",
                "y_pred_arousal2",
                "y_true_arousal3",
                "y_pred_arousal3",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    global_emotion = safe_metric_emotion(rows_out)
    global_a2 = safe_metric_mae(rows_out, "y_true_arousal2", "y_pred_arousal2")
    global_a3 = safe_metric_mae(rows_out, "y_true_arousal3", "y_pred_arousal3")

    by_dataset: Dict[str, List[Dict[str, str]]] = {}
    for row in rows_out:
        key = row.get("dataset") or "unknown"
        by_dataset.setdefault(key, []).append(row)

    per_dataset = {}
    for ds, ds_rows in sorted(by_dataset.items()):
        per_dataset[ds] = {
            "emotion6": safe_metric_emotion(ds_rows),
            "arousal2": safe_metric_mae(ds_rows, "y_true_arousal2", "y_pred_arousal2"),
            "arousal3": safe_metric_mae(ds_rows, "y_true_arousal3", "y_pred_arousal3"),
        }

    summary = {
        "input": {
            "index_csv": str(args.index_csv),
            "manifest": str(args.manifest),
            "json_dir": str(args.json_dir),
            "start_idx": args.start_idx,
            "max_samples": args.max_samples,
            "emotion_classes": classes,
        },
        "counts": {
            "total_rows": len(rows_out),
            "json_found": parsed_json,
            "json_missing": missing_json,
            "emotion_pred_missing": pred_missing,
        },
        "emotion6": global_emotion,
        "arousal2": global_a2,
        "arousal3": global_a3,
        "per_dataset": per_dataset,
    }

    args.out_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
