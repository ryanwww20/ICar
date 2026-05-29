#!/usr/bin/env python3
"""Export image pairs from Argoverse 2 Sensor Dataset for DUSt3R."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from av2_utils import (  # noqa: E402
    choose_default_camera,
    detect_stereo_cameras,
    discover_cameras_in_log,
    extract_timestamp,
    find_camera_by_name,
    find_log_dirs,
    match_image_pairs,
    make_temporal_pairs,
    resolve_ring_adjacent_pairs,
    resolve_split_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create DUSt3R image pair lists from Argoverse 2 Sensor Dataset."
    )
    parser.add_argument("--av2-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=Path, default=Path("data/av2_pairs"))
    parser.add_argument(
        "--pair-type",
        type=str,
        default="stereo",
        choices=["stereo", "temporal", "ring-adjacent"],
    )
    parser.add_argument("--left-camera", type=str, default=None)
    parser.add_argument("--right-camera", type=str, default=None)
    parser.add_argument("--max-logs", type=int, default=3)
    parser.add_argument("--max-pairs-per-log", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--copy-images", action="store_true", default=False)
    parser.add_argument("--resize-long-edge", type=int, default=None)
    parser.add_argument("--pair-list-name", type=str, default="pairs.json")
    return parser.parse_args()


def maybe_resize_and_copy(
    src: Path,
    dst: Path,
    resize_long_edge: int | None,
) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if resize_long_edge is None:
        shutil.copy2(src, dst)
        return dst.resolve()

    with Image.open(src) as img:
        img = img.convert("RGB")
        long_edge = max(img.size)
        if long_edge != resize_long_edge:
            scale = resize_long_edge / float(long_edge)
            new_size = (
                max(1, int(round(img.size[0] * scale))),
                max(1, int(round(img.size[1] * scale))),
            )
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        img.save(dst, quality=95)
    return dst.resolve()


def build_pair_record(
    log_id: str,
    pair_type: str,
    image1: Path,
    image2: Path,
    camera1: str,
    camera2: str,
    copied1: Path | None = None,
    copied2: Path | None = None,
) -> dict:
    record = {
        "log_id": log_id,
        "pair_type": pair_type,
        "image1": str(image1.resolve()),
        "image2": str(image2.resolve()),
        "camera1": camera1,
        "camera2": camera2,
        "timestamp1": str(extract_timestamp(image1)),
        "timestamp2": str(extract_timestamp(image2)),
    }
    if copied1 is not None:
        record["copied_image1"] = str(copied1.resolve())
    if copied2 is not None:
        record["copied_image2"] = str(copied2.resolve())
    return record


def export_stereo_pairs(
    log_id: str,
    cameras: dict[str, list[Path]],
    left_camera: str | None,
    right_camera: str | None,
    max_pairs: int,
    stride: int,
) -> list[tuple[Path, Path, str, str]]:
    camera_names = sorted(cameras.keys())
    if left_camera and right_camera:
        left = find_camera_by_name(camera_names, left_camera)
        right = find_camera_by_name(camera_names, right_camera)
        if not left or not right:
            raise ValueError(
                f"Could not resolve cameras for log {log_id}. "
                f"Requested left={left_camera}, right={right_camera}. "
                f"Available: {camera_names}"
            )
    else:
        left, right = detect_stereo_cameras(camera_names)
        if not left or not right:
            raise ValueError(
                f"No stereo pair detected for log {log_id}. "
                f"Available cameras: {camera_names}. "
                "Use --left-camera and --right-camera."
            )

    pairs = match_image_pairs(cameras[left], cameras[right])
    pairs = pairs[:: max(1, stride)][:max_pairs]
    return [(p1, p2, left, right) for p1, p2 in pairs]


def export_temporal_pairs(
    log_id: str,
    cameras: dict[str, list[Path]],
    camera_name: str | None,
    max_pairs: int,
    stride: int,
) -> list[tuple[Path, Path, str, str]]:
    camera_names = sorted(cameras.keys())
    if camera_name:
        chosen = find_camera_by_name(camera_names, camera_name)
        if not chosen:
            raise ValueError(
                f"Camera '{camera_name}' not found in log {log_id}. "
                f"Available: {camera_names}"
            )
    else:
        chosen = choose_default_camera(camera_names)
        if not chosen:
            raise ValueError(f"No cameras found in log {log_id}")

    pairs = make_temporal_pairs(cameras[chosen], stride=stride)[:max_pairs]
    return [(p1, p2, chosen, chosen) for p1, p2 in pairs]


def export_ring_adjacent_pairs(
    log_id: str,
    cameras: dict[str, list[Path]],
    left_camera: str | None,
    right_camera: str | None,
    max_pairs: int,
    stride: int,
) -> list[tuple[Path, Path, str, str]]:
    camera_names = sorted(cameras.keys())
    camera_pairs: list[tuple[str, str]] = []

    if left_camera and right_camera:
        left = find_camera_by_name(camera_names, left_camera)
        right = find_camera_by_name(camera_names, right_camera)
        if not left or not right:
            raise ValueError(
                f"Could not resolve ring cameras for log {log_id}. "
                f"Requested left={left_camera}, right={right_camera}. "
                f"Available: {camera_names}"
            )
        camera_pairs = [(left, right)]
    else:
        camera_pairs = resolve_ring_adjacent_pairs(camera_names)
        if not camera_pairs:
            raise ValueError(
                f"No ring-adjacent camera pairs found for log {log_id}. "
                f"Available cameras: {camera_names}. "
                "Use --left-camera and --right-camera."
            )

    exported: list[tuple[Path, Path, str, str]] = []
    per_pair_budget = max(1, max_pairs // max(1, len(camera_pairs)))
    for left, right in camera_pairs:
        pairs = match_image_pairs(cameras[left], cameras[right])
        pairs = pairs[:: max(1, stride)][:per_pair_budget]
        exported.extend((p1, p2, left, right) for p1, p2 in pairs)
        if len(exported) >= max_pairs:
            break
    return exported[:max_pairs]


def main() -> int:
    args = parse_args()
    av2_root = args.av2_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not av2_root.exists():
        print(f"ERROR: --av2-root does not exist: {av2_root}", file=sys.stderr)
        return 1

    split_root = resolve_split_root(av2_root, args.split)
    logs = find_log_dirs(split_root, max_logs=args.max_logs)
    if not logs:
        print(
            f"ERROR: No logs found under {split_root}. Check --av2-root and --split.",
            file=sys.stderr,
        )
        return 1

    all_pairs: list[dict] = []
    pair_idx = 0

    for log_dir in logs:
        log_id = log_dir.name
        cameras = discover_cameras_in_log(log_dir)
        if not cameras:
            print(f"WARNING: Skipping log {log_id}: no camera images found.", file=sys.stderr)
            continue

        try:
            if args.pair_type == "stereo":
                raw_pairs = export_stereo_pairs(
                    log_id,
                    cameras,
                    args.left_camera,
                    args.right_camera,
                    args.max_pairs_per_log,
                    args.stride,
                )
            elif args.pair_type == "temporal":
                camera_for_temporal = args.left_camera or args.right_camera
                raw_pairs = export_temporal_pairs(
                    log_id,
                    cameras,
                    camera_for_temporal,
                    args.max_pairs_per_log,
                    args.stride,
                )
            else:
                raw_pairs = export_ring_adjacent_pairs(
                    log_id,
                    cameras,
                    args.left_camera,
                    args.right_camera,
                    args.max_pairs_per_log,
                    args.stride,
                )
        except ValueError as exc:
            print(f"WARNING: {exc}", file=sys.stderr)
            continue

        for image1, image2, camera1, camera2 in raw_pairs:
            copied1 = copied2 = None
            if args.copy_images:
                images_dir = output_dir / "images"
                copied1 = maybe_resize_and_copy(
                    image1,
                    images_dir / f"pair_{pair_idx:06d}_0{image1.suffix.lower()}",
                    args.resize_long_edge,
                )
                copied2 = maybe_resize_and_copy(
                    image2,
                    images_dir / f"pair_{pair_idx:06d}_1{image2.suffix.lower()}",
                    args.resize_long_edge,
                )

            all_pairs.append(
                build_pair_record(
                    log_id=log_id,
                    pair_type=args.pair_type,
                    image1=image1,
                    image2=image2,
                    camera1=camera1,
                    camera2=camera2,
                    copied1=copied1,
                    copied2=copied2,
                )
            )
            pair_idx += 1

    if not all_pairs:
        print("ERROR: No image pairs were exported.", file=sys.stderr)
        return 1

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "av2_root": str(av2_root),
        "split": args.split,
        "pair_type": args.pair_type,
        "num_pairs": len(all_pairs),
        "pairs": all_pairs,
    }

    pairs_json_path = output_dir / args.pair_list_name
    pairs_txt_path = output_dir / "pairs.txt"

    with pairs_json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with pairs_txt_path.open("w", encoding="utf-8") as f:
        for pair in all_pairs:
            img1 = pair.get("copied_image1", pair["image1"])
            img2 = pair.get("copied_image2", pair["image2"])
            f.write(f"{img1} {img2}\n")

    print(f"Exported {len(all_pairs)} pairs")
    print(f"  JSON: {pairs_json_path.resolve()}")
    print(f"  TXT:  {pairs_txt_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
