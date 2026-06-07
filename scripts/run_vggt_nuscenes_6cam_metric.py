#!/usr/bin/env python3
"""
Version B (metric):
Run VGGT on 6 nuScenes cameras at the same sample (one forward pass).

Estimates metric scale from nuScenes camera baselines, then back-projects VGGT depth
with nuScenes intrinsics/extrinsics into world coordinates (meters).

Output default: outputs/vggt_nuscenes_6cam_metric/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
VGGT_ROOT = REPO_ROOT / "vggt"
VERSION_LABEL = "metric"

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
from ply_postprocess_common import (  # noqa: E402
    add_postprocess_args,
    maybe_postprocess_pointcloud,
)
from vggt_nuscenes_common import (  # noqa: E402
    add_nuscenes_args,
    backproject_metric_points,
    estimate_scale_from_camera_baselines,
    load_vggt_model,
    pointcloud_output_path,
    run_vggt_multi,
    save_depth_png,
    write_pointcloud,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "[Version B: metric] "
            "Run VGGT on 6 nuScenes cameras at the same sample with metric back-projection."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vggt_nuscenes_6cam_metric"),
        help="Output root (default: outputs/vggt_nuscenes_6cam_metric).",
    )
    add_nuscenes_args(parser)
    add_postprocess_args(parser)
    return parser.parse_args()


def process_sample(
    nusc,
    scene: dict,
    scene_label: str,
    scene_index: int,
    sample_idx: int,
    base_out_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
    device: str,
    model,
) -> int:
    sample_token, views = collect_six_cam_views(nusc, scene, sample_idx)
    sample_dir = out_dir / f"sample_{sample_idx:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    ply_path = pointcloud_output_path(
        base_out_dir,
        mode="metric",
        dataset="nuscenes",
        scene_index=scene_index,
        frame_index=sample_idx,
    )

    print(f"\n[{scene_label} sample {sample_idx}] scene={scene['name']} token={sample_token}")
    for view in views:
        print(f"  {view.name}: {view.image_path.name} (ts={view.timestamp_ns})")

    meta = {
        "version_label": VERSION_LABEL,
        "scale_mode": "camera_baseline",
        "description": (
            "Metric scale from nuScenes camera baselines; depth back-projected with nuScenes K/pose."
        ),
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
        "pointcloud_path": str(ply_path),
    }
    with (sample_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if args.dry_run:
        print(f"[dry-run] would run VGGT on 6 images -> {sample_dir} ({ply_path.name})")
        maybe_postprocess_pointcloud(
            ply_path,
            args,
            label=f"sample {sample_idx}",
            dry_run=True,
        )
        return 0

    image_paths = [view.image_path for view in views]
    pred = run_vggt_multi(model, image_paths, device)

    metric_T_W_C = [view.T_W_C for view in views]
    scale = estimate_scale_from_camera_baselines(pred["extrinsic"], metric_T_W_C)
    print(f"[sample {sample_idx}] estimated metric scale={scale:.4f}")

    meta["metric_scale"] = scale
    with (sample_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    cameras_out = {
        "vggt_extrinsic": pred["extrinsic"].tolist(),
        "vggt_intrinsic": pred["intrinsic"].tolist(),
        "nuscenes_intrinsic": {view.name: view.intrinsic.tolist() for view in views},
        "nuscenes_T_W_C": {view.name: view.T_W_C.tolist() for view in views},
        "metric_scale": scale,
        "proc_hw": list(pred["proc_hw"]),
    }
    with (sample_dir / "cameras.json").open("w", encoding="utf-8") as f:
        json.dump(cameras_out, f, indent=2)

    pts_list: list[np.ndarray] = []
    col_list: list[np.ndarray] = []

    for i, view in enumerate(views):
        suffix = view.image_path.suffix.lower()
        shutil.copy2(view.image_path, sample_dir / f"image_{i:02d}_{view.name}{suffix}")
        save_depth_png(sample_dir / f"depth_{i:02d}_{view.name}.png", pred["depth"][i])

        native_bgr = cv2.imread(str(view.image_path))
        if native_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {view.image_path}")

        pts, cols = backproject_metric_points(
            pred["depth"][i],
            pred["depth_conf"][i],
            view.intrinsic,
            view.T_W_C,
            native_bgr,
            scale,
            args.conf_thresh,
            args.pixel_stride,
        )
        if len(pts):
            pts_list.append(pts)
            col_list.append(cols)

    if pts_list:
        pts = np.concatenate(pts_list, axis=0)
        cols = np.concatenate(col_list, axis=0)
    else:
        pts = np.zeros((0, 3))
        cols = np.zeros((0, 3))

    write_pointcloud(
        ply_path,
        pts,
        cols,
        voxel_size=args.voxel_size,
        no_cleanup=args.no_cleanup,
        label=f"sample {sample_idx}",
    )
    post_path = maybe_postprocess_pointcloud(
        ply_path,
        args,
        label=f"sample {sample_idx}",
    )
    if post_path is not None:
        meta["postprocessed_pointcloud_path"] = str(post_path)
        meta["post_process"] = {
            "task": args.post_process_task,
            "sky_axis": args.sky_axis,
            "sky_side": args.sky_side,
            "sky_keep_percentile": args.sky_keep_percentile,
            "sky_max": args.sky_max,
        }
        with (sample_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
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
        "scale_mode": "camera_baseline",
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
        "post_process": args.post_process,
        "post_process_task": args.post_process_task,
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
                base_out_dir,
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

    print(f"\n[done] Version B (metric) outputs under {out_dir}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
