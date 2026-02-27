#!/usr/bin/env python3
"""
Build unified manifests and actor-independent splits for CREMA-D and RAVDESS.

Outputs
- derived/manifests/manifest_all.jsonl
- derived/manifests/manifest_crema_d.jsonl
- derived/manifests/manifest_ravdess.jsonl
- derived/manifests/manifest_common6_all.jsonl
- derived/manifests/manifest_multimodal_common6_av.jsonl
- derived/manifests/manifest_audio_enabled_common6.jsonl
- derived/manifests/manifest_video_enabled_common6.jsonl
- derived/manifests/manifest_ravdess_audio_only_common6.jsonl
- derived/manifests/manifest_ravdess_av_common6.jsonl
- derived/manifests/summary.json
- derived/splits/groupkfold5_{all,crema_d,ravdess}.csv
- derived/splits/groupkfold5_{all,crema_d,ravdess}/fold_{0..4}_{train,val}.txt
- derived/splits/cross_dataset/{train_crema_test_ravdess_common6,test_crema_train_ravdess_common6}_{train,test}.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CREMA_EMO_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

RAVDESS_EMO_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

COMMON6 = {"neutral", "happy", "sad", "angry", "fearful", "disgust"}


@dataclass
class ManifestRow:
    clip_id: str
    dataset: str
    actor_id: str
    actor_numeric: int
    modality: str
    modality_code: str
    channel_code: str
    emotion_raw: str
    emotion6: Optional[str]
    emotion8: Optional[str]
    intensity_raw: str
    arousal2: Optional[int]
    arousal3: Optional[int]
    sentence_id: str
    repetition: Optional[int]
    path_audio: Optional[str]
    path_video: Optional[str]
    source_file: str
    is_common6: bool
    has_audio: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare manifests and splits")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root (default: current directory)",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def intensity_to_arousal_crema(level: str) -> Tuple[Optional[int], Optional[int]]:
    level = level.upper()
    if level == "LO":
        return 0, 0
    if level == "MD":
        return 0, 1
    if level == "HI":
        return 1, 2
    return None, None


def intensity_to_arousal_ravdess(intensity: str) -> Optional[int]:
    if intensity == "01":
        return 0
    if intensity == "02":
        return 1
    return None


def build_crema_manifest(repo_root: Path) -> Tuple[List[ManifestRow], Dict[str, int]]:
    crema_root = repo_root / "datasets" / "crema_d" / "crema-d-mirror"
    wav_dir = crema_root / "AudioWAV"
    video_dir = crema_root / "VideoFlash"

    wav_files = {p.stem: p for p in wav_dir.glob("*.wav")}
    flv_files = {p.stem: p for p in video_dir.glob("*.flv")}

    stems = sorted(set(wav_files.keys()) | set(flv_files.keys()))
    rows: List[ManifestRow] = []

    missing_wav = 0
    missing_flv = 0
    unparsable = 0

    for stem in stems:
        parts = stem.split("_")
        if len(parts) != 4:
            unparsable += 1
            continue
        actor, sentence, emo_raw, level = parts
        if not actor.isdigit():
            unparsable += 1
            continue

        emo6 = CREMA_EMO_MAP.get(emo_raw)
        arousal2, arousal3 = intensity_to_arousal_crema(level)

        wav = wav_files.get(stem)
        flv = flv_files.get(stem)
        if wav is None:
            missing_wav += 1
        if flv is None:
            missing_flv += 1

        rows.append(
            ManifestRow(
                clip_id=f"crema_d:{stem}",
                dataset="crema_d",
                actor_id=f"crema_d:{actor}",
                actor_numeric=int(actor),
                modality="av",
                modality_code="av",
                channel_code="speech",
                emotion_raw=emo_raw,
                emotion6=emo6,
                emotion8=emo6,
                intensity_raw=level,
                arousal2=arousal2,
                arousal3=arousal3,
                sentence_id=sentence,
                repetition=None,
                path_audio=str(wav.relative_to(repo_root)) if wav else None,
                path_video=str(flv.relative_to(repo_root)) if flv else None,
                source_file=stem,
                is_common6=emo6 in COMMON6 if emo6 else False,
                has_audio=wav is not None,
            )
        )

    stats = {
        "clips": len(rows),
        "missing_wav": missing_wav,
        "missing_flv": missing_flv,
        "unparsable_stems": unparsable,
    }
    return rows, stats


def build_ravdess_manifest(repo_root: Path) -> Tuple[List[ManifestRow], Dict[str, int]]:
    ravdess_video_root = repo_root / "datasets" / "ravdess" / "raw_video_speech"
    ravdess_audio_root = repo_root / "datasets" / "ravdess" / "raw_audio_speech"

    video_files = sorted(ravdess_video_root.rglob("*.mp4")) if ravdess_video_root.exists() else []
    audio_files = sorted(ravdess_audio_root.rglob("*.wav")) if ravdess_audio_root.exists() else []
    all_files = video_files + audio_files

    rows: List[ManifestRow] = []
    unparsable = 0

    modality_counts = Counter()
    source_counts = Counter()

    for media_file in all_files:
        stem = media_file.stem
        parts = stem.split("-")
        if len(parts) != 7:
            unparsable += 1
            continue

        modality_code, channel_code, emo_code, intensity, statement, repetition, actor = parts
        if not actor.isdigit():
            unparsable += 1
            continue

        if modality_code == "01":
            modality = "av"
            has_audio = True
            path_audio: Optional[str] = str(media_file.relative_to(repo_root))
            path_video: Optional[str] = str(media_file.relative_to(repo_root))
        elif modality_code == "02":
            modality = "video_only"
            has_audio = False
            path_audio = None
            path_video = str(media_file.relative_to(repo_root))
        elif modality_code == "03":
            modality = "audio_only"
            has_audio = True
            path_audio = str(media_file.relative_to(repo_root))
            path_video = None
        else:
            modality = "unknown"
            has_audio = False
            path_audio = None
            path_video = str(media_file.relative_to(repo_root))

        modality_counts[modality] += 1
        if media_file.suffix.lower() == ".wav":
            source_counts["raw_audio_speech"] += 1
        else:
            source_counts["raw_video_speech"] += 1

        emo8 = RAVDESS_EMO_MAP.get(emo_code)
        emo6 = emo8 if emo8 in COMMON6 else None
        arousal2 = intensity_to_arousal_ravdess(intensity)

        rows.append(
            ManifestRow(
                clip_id=f"ravdess:{stem}",
                dataset="ravdess",
                actor_id=f"ravdess:{actor}",
                actor_numeric=int(actor),
                modality=modality,
                modality_code=modality_code,
                channel_code=channel_code,
                emotion_raw=emo_code,
                emotion6=emo6,
                emotion8=emo8,
                intensity_raw=intensity,
                arousal2=arousal2,
                arousal3=None,
                sentence_id=statement,
                repetition=int(repetition),
                path_audio=path_audio,
                path_video=path_video,
                source_file=stem,
                is_common6=emo6 in COMMON6 if emo6 else False,
                has_audio=has_audio,
            )
        )

    stats = {
        "clips": len(rows),
        "unparsable_files": unparsable,
        "modality_counts": {k: int(v) for k, v in sorted(modality_counts.items())},
        "source_counts": {k: int(v) for k, v in sorted(source_counts.items())},
    }
    return rows, stats


def write_jsonl(path: Path, rows: Iterable[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=True))
            f.write("\n")


def assign_group_folds(rows: List[ManifestRow], k: int, seed: int) -> Dict[str, int]:
    group_counts = Counter(row.actor_id for row in rows)
    items = list(group_counts.items())
    rng = random.Random(seed)
    rng.shuffle(items)
    items.sort(key=lambda x: x[1], reverse=True)

    fold_sizes = [0] * k
    group_to_fold: Dict[str, int] = {}

    for group, count in items:
        min_fold = min(range(k), key=lambda idx: fold_sizes[idx])
        group_to_fold[group] = min_fold
        fold_sizes[min_fold] += count

    return group_to_fold


def write_split_files(
    out_dir: Path, split_name: str, rows: List[ManifestRow], k: int, seed: int
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    group_to_fold = assign_group_folds(rows, k=k, seed=seed)

    split_csv = out_dir / f"groupkfold{k}_{split_name}.csv"
    with split_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "dataset", "actor_id", "fold"])
        for row in rows:
            writer.writerow([row.clip_id, row.dataset, row.actor_id, group_to_fold[row.actor_id]])

    fold_dir = out_dir / f"groupkfold{k}_{split_name}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    rows_by_fold: Dict[int, List[ManifestRow]] = defaultdict(list)
    for row in rows:
        rows_by_fold[group_to_fold[row.actor_id]].append(row)

    for fold in range(k):
        train_ids: List[str] = []
        val_ids: List[str] = []
        for row_fold, fold_rows in rows_by_fold.items():
            target = val_ids if row_fold == fold else train_ids
            target.extend(r.clip_id for r in fold_rows)

        train_path = fold_dir / f"fold_{fold}_train.txt"
        val_path = fold_dir / f"fold_{fold}_val.txt"
        train_path.write_text("\n".join(sorted(train_ids)) + "\n", encoding="utf-8")
        val_path.write_text("\n".join(sorted(val_ids)) + "\n", encoding="utf-8")

    counts = Counter(group_to_fold[row.actor_id] for row in rows)
    return {
        "split_name": split_name,
        "fold_counts": {str(k_): int(v) for k_, v in sorted(counts.items())},
        "num_rows": len(rows),
        "num_groups": len(set(r.actor_id for r in rows)),
    }


def write_cross_dataset_splits(out_dir: Path, rows: List[ManifestRow]) -> Dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    crema_common6 = sorted(
        row.clip_id
        for row in rows
        if row.dataset == "crema_d" and row.is_common6 and row.emotion6 is not None
    )
    ravdess_common6 = sorted(
        row.clip_id
        for row in rows
        if row.dataset == "ravdess" and row.is_common6 and row.emotion6 is not None
    )
    ravdess_common6_av = sorted(
        row.clip_id
        for row in rows
        if row.dataset == "ravdess"
        and row.is_common6
        and row.emotion6 is not None
        and row.modality == "av"
    )

    (out_dir / "train_crema_test_ravdess_common6_train.txt").write_text(
        "\n".join(crema_common6) + "\n", encoding="utf-8"
    )
    (out_dir / "train_crema_test_ravdess_common6_test.txt").write_text(
        "\n".join(ravdess_common6) + "\n", encoding="utf-8"
    )
    (out_dir / "test_crema_train_ravdess_common6_train.txt").write_text(
        "\n".join(ravdess_common6) + "\n", encoding="utf-8"
    )
    (out_dir / "test_crema_train_ravdess_common6_test.txt").write_text(
        "\n".join(crema_common6) + "\n", encoding="utf-8"
    )
    (out_dir / "train_crema_test_ravdess_common6_av_train.txt").write_text(
        "\n".join(crema_common6) + "\n", encoding="utf-8"
    )
    (out_dir / "train_crema_test_ravdess_common6_av_test.txt").write_text(
        "\n".join(ravdess_common6_av) + "\n", encoding="utf-8"
    )
    (out_dir / "test_crema_train_ravdess_common6_av_train.txt").write_text(
        "\n".join(ravdess_common6_av) + "\n", encoding="utf-8"
    )
    (out_dir / "test_crema_train_ravdess_common6_av_test.txt").write_text(
        "\n".join(crema_common6) + "\n", encoding="utf-8"
    )

    return {
        "crema_common6": len(crema_common6),
        "ravdess_common6": len(ravdess_common6),
        "ravdess_common6_av": len(ravdess_common6_av),
    }


def emotion_counts(rows: List[ManifestRow], field: str) -> Dict[str, int]:
    counter = Counter(getattr(row, field) for row in rows if getattr(row, field) is not None)
    return {str(k): int(v) for k, v in sorted(counter.items())}


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    manifests_dir = repo_root / "derived" / "manifests"
    splits_dir = repo_root / "derived" / "splits"

    crema_rows, crema_stats = build_crema_manifest(repo_root)
    ravdess_rows, ravdess_stats = build_ravdess_manifest(repo_root)
    all_rows = sorted(crema_rows + ravdess_rows, key=lambda r: r.clip_id)
    common6_rows = [r for r in all_rows if r.is_common6 and r.emotion6 is not None]
    multimodal_common6_av_rows = [
        r
        for r in common6_rows
        if r.path_video is not None and r.path_audio is not None and r.has_audio
    ]
    audio_enabled_common6_rows = [r for r in common6_rows if r.path_audio is not None and r.has_audio]
    video_enabled_common6_rows = [r for r in common6_rows if r.path_video is not None]
    ravdess_audio_only_common6_rows = [
        r for r in common6_rows if r.dataset == "ravdess" and r.modality == "audio_only"
    ]
    ravdess_av_common6_rows = [
        r for r in common6_rows if r.dataset == "ravdess" and r.modality == "av"
    ]

    write_jsonl(manifests_dir / "manifest_crema_d.jsonl", crema_rows)
    write_jsonl(manifests_dir / "manifest_ravdess.jsonl", ravdess_rows)
    write_jsonl(manifests_dir / "manifest_all.jsonl", all_rows)
    write_jsonl(manifests_dir / "manifest_common6_all.jsonl", common6_rows)
    write_jsonl(
        manifests_dir / "manifest_multimodal_common6_av.jsonl", multimodal_common6_av_rows
    )
    write_jsonl(manifests_dir / "manifest_audio_enabled_common6.jsonl", audio_enabled_common6_rows)
    write_jsonl(manifests_dir / "manifest_video_enabled_common6.jsonl", video_enabled_common6_rows)
    write_jsonl(
        manifests_dir / "manifest_ravdess_audio_only_common6.jsonl",
        ravdess_audio_only_common6_rows,
    )
    write_jsonl(
        manifests_dir / "manifest_ravdess_av_common6.jsonl",
        ravdess_av_common6_rows,
    )

    split_all = write_split_files(splits_dir, "all", all_rows, k=args.folds, seed=args.seed)
    split_crema = write_split_files(
        splits_dir, "crema_d", crema_rows, k=args.folds, seed=args.seed
    )
    split_ravdess = write_split_files(
        splits_dir, "ravdess", ravdess_rows, k=args.folds, seed=args.seed
    )
    cross_stats = write_cross_dataset_splits(splits_dir / "cross_dataset", all_rows)

    summary = {
        "totals": {
            "all": len(all_rows),
            "crema_d": len(crema_rows),
            "ravdess": len(ravdess_rows),
            "common6_all": len(common6_rows),
            "multimodal_common6_av": len(multimodal_common6_av_rows),
            "audio_enabled_common6": len(audio_enabled_common6_rows),
            "video_enabled_common6": len(video_enabled_common6_rows),
            "ravdess_audio_only_common6": len(ravdess_audio_only_common6_rows),
            "ravdess_av_common6": len(ravdess_av_common6_rows),
        },
        "crema_d_stats": crema_stats,
        "ravdess_stats": ravdess_stats,
        "emotion6_counts": emotion_counts(all_rows, "emotion6"),
        "emotion8_counts_ravdess": emotion_counts(ravdess_rows, "emotion8"),
        "arousal2_counts": emotion_counts(all_rows, "arousal2"),
        "arousal3_counts": emotion_counts(all_rows, "arousal3"),
        "splits": {
            "all": split_all,
            "crema_d": split_crema,
            "ravdess": split_ravdess,
            "cross_dataset_common6": cross_stats,
        },
    }

    manifests_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
