#!/usr/bin/env python3
"""
Version A (relative / up-to-scale):
Run VGGT on 6 nuScenes cameras at the same sample (one forward pass).

Uses VGGT world_points directly. Geometry is consistent across views, but absolute
size in meters is not guaranteed (scale is arbitrary).

Output default: outputs/vggt_nuscenes_6cam_relative/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
VGGT_ROOT = REPO_ROOT / "vggt"
VERSION_LABEL = "relative_up_to_scale"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(VGGT_ROOT) not in sys.path:
    sys.path.insert(0, str(VGGT_ROOT))

from nuscenes_utils import (  # noqa: E402
    CAM_CHANNELS,
    build_scene_mapping,
    collect_six_cam_views,
    load_nuscenes,
    resolve_scene,
)
from vggt_nuscenes_common import (  # noqa: E402
    add_nuscenes_args,
    load_vggt_model,
    run_vggt_multi,
    save_depth_png,
    world_points_to_ply,
    write_pointcloud,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "[Version A: relative / up-to-scale] "
            "Run VGGT on 6 nuScenes cameras at the same sample."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vggt_nuscenes_6cam_relative"),
        help="Output root (default: outputs/vggt_nuscenes_6cam_relative).",
    )
    add_nuscenes_args(parser)
    return parser.parse_args()


def process_sample(
    nusc,
    scene: dict,
    scene_label: str,
    scene_index: int,
    sample_idx: int,
    out_dir: Path,
    args: argparse.Namespace,
    device: str,
    model,
) -> int:
    sample_token, views = collect_six_cam_views(nusc, scene, sample_idx)
    sample_dir = out_dir / f"sample_{sample_idx:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{scene_label} sample {sample_idx}] scene={scene['name']} token={sample_token}")
    for view in views:
        print(f"  {view.name}: {view.image_path.name} (ts={view.timestamp_ns})")

    meta = {
        "version_label": VERSION_LABEL,
        "scale_mode": "none",
        "description": "VGGT world_points fused directly; relative geometry only.",
        "scene_id": scene_index,
        "scene_label": scene_label,
        "scene_name": scene["name"],
        "scene_token": scene["token"],
        "sample_idx": sample_idx,
        "sample_token": sample_token,
        "timestamp_ns": {view.name: view.timestamp_ns for view in views},
        "cameras": list(CAM_CHANNELS),
        "image_paths": {view.name: str(view.image_path) for view in views},
        "no_cleanup": args.no_cleanup,
        "voxel_size": args.voxel_size,
        "conf_thresh": args.conf_thresh,
        "pixel_stride": args.pixel_stride,
    }
    with (sample_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if args.dry_run:
        print(f"[dry-run] would run VGGT on 6 images -> {sample_dir}")
        return 0

    image_paths = [view.image_path for view in views]
    pred = run_vggt_multi(model, image_paths, device)

    cameras_out = {
        "extrinsic": pred["extrinsic"].tolist(),
        "intrinsic": pred["intrinsic"].tolist(),
        "proc_hw": list(pred["proc_hw"]),
    }
    with (sample_dir / "cameras.json").open("w", encoding="utf-8") as f:
        json.dump(cameras_out, f, indent=2)

    for i, view in enumerate(views):
        suffix = view.image_path.suffix.lower()
        shutil.copy2(view.image_path, sample_dir / f"image_{i:02d}_{view.name}{suffix}")
        save_depth_png(sample_dir / f"depth_{i:02d}_{view.name}.png", pred["depth"][i])

    pts, cols = world_points_to_ply(
        pred["world_points"],
        pred["world_conf"],
        pred["images"],
        args.conf_thresh,
        args.pixel_stride,
    )
    write_pointcloud(
        sample_dir / "pointcloud.ply",
        pts,
        cols,
        voxel_size=args.voxel_size,
        no_cleanup=args.no_cleanup,
        label=f"sample {sample_idx}",
    )
    return 0


def main() -> int:
    args = parse_args()
    dataroot = args.dataroot.expanduser().resolve()
    base_out_dir = args.output_dir.expanduser().resolve()

    if not dataroot.is_dir():
        print(f"ERROR: --dataroot does not exist: {dataroot}", file=sys.stderr)
        return 1

    nusc = load_nuscenes(dataroot, args.version)

    if args.list_scenes:
        mapping = build_scene_mapping(nusc)
        print(f"Scenes under {dataroot} ({args.version}):")
        for entry in mapping:
            print(
                f"  {entry['scene_label']}: {entry['scene_name']} "
                f"({entry['num_samples']} samples)"
            )
        mapping_path = base_out_dir / "scene_mapping.json"
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with mapping_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "version_label": VERSION_LABEL,
                    "version": args.version,
                    "dataroot": str(dataroot),
                    "scenes": mapping,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nSaved mapping -> {mapping_path}")
        return 0

    try:
        scene, scene_label, scene_index = resolve_scene(
            nusc,
            scene_id=args.scene_id if not args.scene_name else None,
            scene_name=args.scene_name,
        )
    except (FileNotFoundError, IndexError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out_dir = base_out_dir / scene_label
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[init] version={VERSION_LABEL} dataroot={dataroot} "
        f"{scene_label} scene={scene['name']} device={device}"
    )

    mapping = build_scene_mapping(nusc)
    with (base_out_dir / "scene_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "version_label": VERSION_LABEL,
                "version": args.version,
                "dataroot": str(dataroot),
                "scenes": mapping,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    run_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version_label": VERSION_LABEL,
        "scale_mode": "none",
        "dataroot": str(dataroot),
        "version": args.version,
        "scene_id": scene_index,
        "scene_label": scene_label,
        "scene_name": scene["name"],
        "sample_idx_start": args.sample_idx,
        "num_samples": args.num_samples,
        "cameras": list(CAM_CHANNELS),
        "dry_run": args.dry_run,
        "no_cleanup": args.no_cleanup,
        "voxel_size": args.voxel_size,
        "conf_thresh": args.conf_thresh,
        "pixel_stride": args.pixel_stride,
    }
    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    model = None
    if not args.dry_run:
        model = load_vggt_model(args.model_id, device)

    status = 0
    for offset in range(args.num_samples):
        sample_idx = args.sample_idx + offset
        try:
            status |= process_sample(
                nusc,
                scene,
                scene_label,
                scene_index,
                sample_idx,
                out_dir,
                args,
                device,
                model,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"ERROR sample {sample_idx}: {exc}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            return 1

    print(f"\n[done] Version A (relative) outputs under {out_dir}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
