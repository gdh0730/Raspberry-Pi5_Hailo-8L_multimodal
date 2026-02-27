#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import sys
import tarfile
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract Hailo-related wheel files from Hailo docker SW suite zip."
    )
    p.add_argument(
        "--suite-zip",
        type=Path,
        required=True,
        help="Path to hailo*_docker.zip",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("third_party/hailo_wheels"),
        help="Directory to write extracted wheels",
    )
    p.add_argument(
        "--max-wheels",
        type=int,
        default=0,
        help="Optional cap for extracted wheel count (0 means no limit)",
    )
    return p.parse_args()


def is_candidate_wheel(name: str) -> bool:
    base = Path(name).name.lower()
    if not base.endswith(".whl"):
        return False
    # Keep only Hailo-related wheels (ignore generic pip/setuptools wheels).
    tokens = ("hailo", "dataflow", "tappas")
    return any(t in base for t in tokens)


def unique_output_path(out_dir: Path, wheel_name: str) -> Path:
    p = out_dir / wheel_name
    if not p.exists():
        return p
    stem = p.stem
    suffix = p.suffix
    idx = 2
    while True:
        cand = out_dir / f"{stem}.{idx}{suffix}"
        if not cand.exists():
            return cand
        idx += 1


def main() -> int:
    args = parse_args()
    suite_zip = args.suite_zip
    out_dir = args.out_dir
    max_wheels = args.max_wheels

    if not suite_zip.is_file():
        print(f"[ERROR] Missing zip: {suite_zip}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    scanned_layers = 0

    with zipfile.ZipFile(suite_zip) as zf:
        tgz_candidates = [n for n in zf.namelist() if n.lower().endswith(".tar.gz")]
        if not tgz_candidates:
            print("[ERROR] No .tar.gz found inside suite zip.", file=sys.stderr)
            return 2
        inner_tgz = tgz_candidates[0]
        print(f"[INFO] Inner docker archive: {inner_tgz}")
        with zf.open(inner_tgz) as tgz_stream:
            with tarfile.open(fileobj=tgz_stream, mode="r|gz") as outer_tar:
                for member in outer_tar:
                    if not member.isfile() or not member.name.endswith("layer.tar"):
                        continue
                    scanned_layers += 1
                    print(f"[INFO] Scanning layer {scanned_layers}: {member.name}")
                    layer_file = outer_tar.extractfile(member)
                    if layer_file is None:
                        continue
                    try:
                        with tarfile.open(fileobj=layer_file, mode="r|") as layer_tar:
                            for inner in layer_tar:
                                if not inner.isfile():
                                    continue
                                if not is_candidate_wheel(inner.name):
                                    continue
                                src = layer_tar.extractfile(inner)
                                if src is None:
                                    continue
                                out_path = unique_output_path(out_dir, Path(inner.name).name)
                                with out_path.open("wb") as f:
                                    while True:
                                        chunk = src.read(1024 * 1024)
                                        if not chunk:
                                            break
                                        f.write(chunk)
                                extracted += 1
                                print(f"[FOUND] {inner.name} -> {out_path}")
                                if max_wheels > 0 and extracted >= max_wheels:
                                    print("[INFO] max-wheels reached.")
                                    print(f"[INFO] Extracted wheels: {extracted}")
                                    return 0
                    except tarfile.ReadError:
                        # Some layer members may be malformed for stream reads; skip safely.
                        continue

    print(f"[INFO] Scanned layers: {scanned_layers}")
    print(f"[INFO] Extracted wheels: {extracted}")
    if extracted == 0:
        print("[ERROR] No Hailo-related wheels were extracted.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
