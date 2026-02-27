#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Patch Hailo DFC wheel metadata to skip pygraphviz dependency."
    )
    p.add_argument("--in-wheel", type=Path, required=True)
    p.add_argument("--out-wheel", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_wheel = args.in_wheel
    out_wheel = args.out_wheel

    if not in_wheel.is_file():
        raise SystemExit(f"[ERROR] Missing input wheel: {in_wheel}")

    out_wheel.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(in_wheel, "r") as zin, zipfile.ZipFile(
        out_wheel, "w", compression=zipfile.ZIP_DEFLATED
    ) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith(".dist-info/METADATA"):
                text = data.decode("utf-8", errors="replace").splitlines()
                filtered = []
                removed = 0
                for line in text:
                    low = line.lower()
                    if low.startswith("requires-dist:") and "pygraphviz" in low:
                        removed += 1
                        continue
                    filtered.append(line)
                data = ("\n".join(filtered) + "\n").encode("utf-8")
                print(
                    f"[INFO] Patched METADATA: removed {removed} pygraphviz requirement(s)."
                )
            zout.writestr(item, data)

    print(f"[OK] Patched wheel created: {out_wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
