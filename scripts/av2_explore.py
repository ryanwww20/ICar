#!/usr/bin/env python3
"""Explore Argoverse 2 Sensor Dataset layout and camera folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_AV2_ROOT = REPO_ROOT / "vggt/data/AV2"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from av2_utils import (  # noqa: E402
    detect_stereo_cameras,
    discover_cameras_in_log,
    find_log_dirs,
    resolve_split_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan Argoverse 2 Sensor Dataset and list camera folders."
    )
    parser.add_argument(
        "--av2-root",
        type=Path,
        default=DEFAULT_AV2_ROOT,
        help=f"Root path of Argoverse 2 Sensor Dataset (default: {DEFAULT_AV2_ROOT})",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split (default: val)",
    )
    parser.add_argument(
        "--max-logs",
        type=int,
        default=5,
        help="Maximum number of logs to scan (default: 5)",
    )
    return parser.parse_args()


def format_candidate_pairs(left: str | None, right: str | None) -> list[str]:
    if left and right:
        return [f"{left} <-> {right}"]
    return []


def explore_log(log_dir: Path) -> None:
    cameras = discover_cameras_in_log(log_dir)
    camera_names = sorted(cameras.keys())

    print(f"Log: {log_dir.name}")
    print(f"  Path: {log_dir}")
    print("  Cameras:")
    if not cameras:
        print("    (no camera folders with images found)")
    else:
        for name in camera_names:
            images = cameras[name]
            print(f"    {name}: {len(images)} images")
            for sample in images[:3]:
                print(f"      - {sample}")

    left, right = detect_stereo_cameras(camera_names)
    candidate_pairs = format_candidate_pairs(left, right)
    print("  Candidate stereo pairs:")
    if candidate_pairs:
        for pair in candidate_pairs:
            print(f"    {pair}")
    else:
        print("    (none detected automatically)")
        if camera_names:
            print("    Available camera folders:")
            for name in camera_names:
                print(f"      - {name}")
            print(
                "    Hint: re-run av2_make_pairs.py with "
                "--left-camera and --right-camera after confirming names."
            )
    print()


def main() -> int:
    args = parse_args()
    av2_root = args.av2_root.expanduser().resolve()

    if not av2_root.exists():
        print(f"ERROR: --av2-root does not exist: {av2_root}", file=sys.stderr)
        return 1

    split_root = resolve_split_root(av2_root, args.split)
    print(f"Using split root: {split_root}")
    print(f"Split: {args.split}")
    print(f"Max logs: {args.max_logs}")
    print()

    try:
        logs = find_log_dirs(split_root, max_logs=args.max_logs)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not logs:
        print(
            "No logs with camera images found. Check --av2-root and --split.",
            file=sys.stderr,
        )
        return 1

    for log_dir in logs:
        explore_log(log_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
