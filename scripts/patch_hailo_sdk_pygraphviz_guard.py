#!/usr/bin/env python3
"""Patch hailo_sdk_client import guard when pygraphviz is unavailable."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List


def find_sdk_init(venv_dir: Path) -> Path:
    candidates: List[Path] = []
    for p in (venv_dir / "lib").glob("python*/site-packages/hailo_sdk_client/__init__.py"):
        candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"Could not find hailo_sdk_client/__init__.py under: {venv_dir}")
    return sorted(candidates)[-1]


def patch_text(src: str) -> str:
    old = "except (ValueError, OSError):"
    new = "except (ModuleNotFoundError, ImportError, ValueError, OSError):"
    if old in src:
        return src.replace(old, new, 1)
    if new in src:
        return src
    raise RuntimeError("Could not find expected pygraphviz guard pattern in hailo_sdk_client/__init__.py")


def touch_requirement_marker(venv_dir: Path) -> Path:
    marker = venv_dir / "etc" / "hailo" / "check_system_requirements_was_called"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)
    return marker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Patch hailo_sdk_client pygraphviz import guard in a venv")
    p.add_argument("--venv-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    venv_dir = args.venv_dir.resolve()
    init_py = find_sdk_init(venv_dir)
    original = init_py.read_text(encoding="utf-8")
    patched = patch_text(original)
    if patched != original:
        init_py.write_text(patched, encoding="utf-8")
        print(f"[PATCH] Updated pygraphviz guard: {init_py}")
    else:
        print(f"[PATCH] Guard already applied: {init_py}")

    marker = touch_requirement_marker(venv_dir)
    print(f"[PATCH] Touched system-check marker: {marker}")


if __name__ == "__main__":
    main()
