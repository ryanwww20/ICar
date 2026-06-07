#!/usr/bin/env python3
"""
Version B (metric):
Run VGGT on 7 AV2 ring camera images at the same frame index (one forward pass).

Estimates metric scale from AV2 camera baselines, then back-projects VGGT depth
with AV2 intrinsics/extrinsics into city/world coordinates (meters).

Output default: outputs/vggt_av2_7ring_metric/
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
DEFAULT_AV2_ROOT = REPO_ROOT / "vggt/data/AV2"
VERSION_LABEL = "metric"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(VGGT_ROOT) not in sys.path:
    sys.path.insert(0, str(VGGT_ROOT))

from av2_calibration_utils import (  # noqa: E402
    RING_CAMERAS,
    collect_seven_ring_views_metric,
    load_log_calibration,
)
from av2_utils import (  # noqa: E402
    build_scene_mapping,
    resolve_log_dir,
    resolve_split_root,
)
from vggt_nuscenes_common import (  # noqa: E402
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
            "Run VGGT on 7 AV2 ring cameras at the same frame with metric back-projection."
        )
    )
    parser.add_argument(
        "--av2-root",
        type=Path,
        default=DEFAULT_AV2_ROOT,
        help=f"AV2 dataset root (default: {DEFAULT_AV2_ROOT})",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--scene-id",
        type=str,
        default="0",
        help="Scene index under the split, e.g. 0 or scene-0 (default: scene-0).",
    )
    parser.add_argument(
        "--log-id",
        type=str,
        default=None,
        help="AV2 log UUID. Overrides --scene-id when set.",
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List scene-N -> log UUID mapping and exit.",
    )
    parser.add_argument(
        "--frame-idx",
        type=int,
        default=0,
        help="Frame index (same index in each ring camera's sorted image list).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=1,
        help="Number of consecutive frame indices to process (starting at --frame-idx).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vggt_av2_7ring_metric"),
        help="Output root (default: outputs/vggt_av2_7ring_metric).",
    )
    parser.add_argument("--model-id", type=str, default="facebook/VGGT-1B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--conf-thresh", type=float, default=0.5)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.10,
        help="Voxel size (m) for downsampling. Use 0 to skip voxel step only.",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip voxel downsampling and outlier removal; save all filtered points.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def process_frame(
    log_dir: Path,
    scene_label: str,
    frame_idx: int,
    base_out_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
    device: str,
    scene_index: int,
    model,
    calibration,
) -> int:
    views = collect_seven_ring_views_metric(log_dir, frame_idx, calibration)
    frame_dir = out_dir / f"frame_{frame_idx:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    ply_path = pointcloud_output_path(
        base_out_dir,
        mode="metric",
        dataset="av2",
        scene_index=scene_index,
        frame_index=frame_idx,
    )

    print(f"\n[{scene_label} frame {frame_idx}] log_id={log_dir.name}")
    for view in views:
        print(f"  {view.name}: {view.image_path.name} (ts={view.timestamp_ns})")

    meta = {
        "version_label": VERSION_LABEL,
        "scale_mode": "camera_baseline",
        "description": (
            "Metric scale from AV2 camera baselines; depth back-projected with AV2 K/pose."
        ),
        "scene_id": scene_index,
        "scene_label": scene_label,
        "log_id": log_dir.name,
        "frame_idx": frame_idx,
        "timestamp_ns": {view.name: view.timestamp_ns for view in views},
        "cameras": list(RING_CAMERAS),
        "image_paths": {view.name: str(view.image_path) for view in views},
        "no_cleanup": args.no_cleanup,
        "voxel_size": args.voxel_size,
        "conf_thresh": args.conf_thresh,
        "pixel_stride": args.pixel_stride,
        "pointcloud_path": str(ply_path),
    }
    with (frame_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if args.dry_run:
        print(f"[dry-run] would run VGGT on 7 images -> {frame_dir} ({ply_path.name})")
        return 0

    image_paths = [view.image_path for view in views]
    pred = run_vggt_multi(model, image_paths, device)

    metric_T_W_C = [view.T_W_C for view in views]
    scale = estimate_scale_from_camera_baselines(pred["extrinsic"], metric_T_W_C)
    print(f"[frame {frame_idx}] estimated metric scale={scale:.4f}")

    meta["metric_scale"] = scale
    with (frame_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    cameras_out = {
        "vggt_extrinsic": pred["extrinsic"].tolist(),
        "vggt_intrinsic": pred["intrinsic"].tolist(),
        "av2_intrinsic": {view.name: view.intrinsic.tolist() for view in views},
        "av2_T_W_C": {view.name: view.T_W_C.tolist() for view in views},
        "metric_scale": scale,
        "proc_hw": list(pred["proc_hw"]),
    }
    with (frame_dir / "cameras.json").open("w", encoding="utf-8") as f:
        json.dump(cameras_out, f, indent=2)

    pts_list: list[np.ndarray] = []
    col_list: list[np.ndarray] = []

    for i, view in enumerate(views):
        suffix = view.image_path.suffix.lower()
        shutil.copy2(view.image_path, frame_dir / f"image_{i:02d}_{view.name}{suffix}")
        save_depth_png(frame_dir / f"depth_{i:02d}_{view.name}.png", pred["depth"][i])

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
        label=f"frame {frame_idx}",
    )
    return 0


def main() -> int:
    args = parse_args()
    av2_root = args.av2_root.expanduser().resolve()
    base_out_dir = args.output_dir.expanduser().resolve()

    if not av2_root.is_dir():
        print(f"ERROR: --av2-root does not exist: {av2_root}", file=sys.stderr)
        return 1

    split_root = resolve_split_root(av2_root, args.split)

    if args.list_scenes:
        mapping = build_scene_mapping(split_root)
        if not mapping:
            print(f"No scenes under {split_root}", file=sys.stderr)
            return 1
        print(f"Scenes under {split_root} ({args.split}):")
        for entry in mapping:
            print(f"  {entry['scene_label']}: {entry['log_id']}")
        mapping_path = base_out_dir / "scene_mapping.json"
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with mapping_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "version_label": VERSION_LABEL,
                    "split": args.split,
                    "av2_root": str(av2_root),
                    "scenes": mapping,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nSaved mapping -> {mapping_path}")
        return 0

    try:
        log_dir, scene_label, scene_index = resolve_log_dir(
            split_root,
            scene_id=args.scene_id if not args.log_id else None,
            log_id=args.log_id,
        )
    except (FileNotFoundError, IndexError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        calibration = load_log_calibration(str(log_dir))
    except (FileNotFoundError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out_dir = base_out_dir / scene_label
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[init] version={VERSION_LABEL} av2_root={av2_root} "
        f"{scene_label} log_id={log_dir.name} device={device}"
    )

    mapping = build_scene_mapping(split_root)
    with (base_out_dir / "scene_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "version_label": VERSION_LABEL,
                "split": args.split,
                "av2_root": str(av2_root),
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
        "av2_root": str(av2_root),
        "split": args.split,
        "scene_id": scene_index,
        "scene_label": scene_label,
        "log_id": log_dir.name,
        "frame_idx_start": args.frame_idx,
        "num_frames": args.num_frames,
        "ring_cameras": list(RING_CAMERAS),
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
    for offset in range(args.num_frames):
        frame_idx = args.frame_idx + offset
        try:
            status |= process_frame(
                log_dir,
                scene_label,
                frame_idx,
                base_out_dir,
                out_dir,
                args,
                device,
                scene_index,
                model,
                calibration,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"ERROR frame {frame_idx}: {exc}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            return 1

    print(f"\n[done] Version B (metric) outputs under {out_dir} ({scene_label}, log_id={log_dir.name})")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
