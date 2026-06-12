#!/usr/bin/env python3
"""Shared PLY post-processing helpers for run_vggt_* scripts."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
POST_PROCESS_DIR = SCRIPT_DIR / "post_process"
PLY_POST_PROCESS_PATH = POST_PROCESS_DIR / "ply_post-process.py"
VIEW_PLY_PATH = POST_PROCESS_DIR / "view_ply.py"
_PLY_POST_PROCESS_MODULE = "ply_post_process"
_VIEW_PLY_MODULE = "view_ply"


def _load_ply_postprocess_module():
    if _PLY_POST_PROCESS_MODULE in sys.modules:
        return sys.modules[_PLY_POST_PROCESS_MODULE]

    post_dir = str(POST_PROCESS_DIR)
    if post_dir not in sys.path:
        sys.path.insert(0, post_dir)

    spec = importlib.util.spec_from_file_location(
        _PLY_POST_PROCESS_MODULE,
        PLY_POST_PROCESS_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load {PLY_POST_PROCESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Required before exec_module so @dataclass can resolve cls.__module__.
    sys.modules[_PLY_POST_PROCESS_MODULE] = module
    spec.loader.exec_module(module)
    return module


def _load_view_ply_module():
    if _VIEW_PLY_MODULE in sys.modules:
        return sys.modules[_VIEW_PLY_MODULE]

    post_dir = str(POST_PROCESS_DIR)
    if post_dir not in sys.path:
        sys.path.insert(0, post_dir)

    spec = importlib.util.spec_from_file_location(_VIEW_PLY_MODULE, VIEW_PLY_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load {VIEW_PLY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_VIEW_PLY_MODULE] = module
    spec.loader.exec_module(module)
    return module


def postprocessed_output_path(ply_path: Path) -> Path:
    """Return ``{stem}_post.ply`` next to the raw VGGT output."""
    return ply_path.with_name(f"{ply_path.stem}_post{ply_path.suffix}")


def add_postprocess_args(parser: argparse.ArgumentParser) -> None:
    post_dir = str(POST_PROCESS_DIR)
    if post_dir not in sys.path:
        sys.path.insert(0, post_dir)
    from add_ego_car import (
        DEFAULT_CAR_LENGTH_M,
        DEFAULT_CAR_PITCH_DEG,
        DEFAULT_CAR_ROLL_DEG,
        DEFAULT_CAR_SAMPLE_SPACING,
        DEFAULT_CAR_YAW_DEG,
    )

    group = parser.add_argument_group("PLY post-processing")
    group.add_argument(
        "--post-process",
        dest="post_process",
        action="store_true",
        help="Run sky removal + hole fill after saving the raw .ply (default).",
    )
    group.add_argument(
        "--no-post-process",
        dest="post_process",
        action="store_false",
        help="Skip PLY post-processing.",
    )
    group.add_argument(
        "--post-process-task",
        choices=("full_pipeline", "fill_hole", "export_bev"),
        default="full_pipeline",
        help="Post-process task (default: full_pipeline).",
    )
    group.add_argument(
        "--sky-axis",
        choices=("x", "y", "z"),
        default="y",
        help="Axis for sky clipping in full_pipeline.",
    )
    group.add_argument(
        "--sky-side",
        choices=("high", "low"),
        default="low",
        help="Sky location on chosen axis.",
    )
    group.add_argument(
        "--sky-keep-percentile",
        type=float,
        default=0.9,
        help="Fraction of non-sky points to keep in full_pipeline.",
    )
    group.add_argument(
        "--sky-max",
        type=float,
        default=None,
        help="Absolute sky threshold; overrides percentile when set.",
    )
    group.add_argument(
        "--export-bev",
        dest="export_bev",
        action="store_true",
        help="Export rear-top BEV PNG with car.png overlay after post-processing (default: on).",
    )
    group.add_argument(
        "--no-export-bev",
        dest="export_bev",
        action="store_false",
        help="Skip the rear-top BEV PNG export.",
    )
    group.add_argument(
        "--bev-car-png",
        type=Path,
        default=None,
        help="Top-down car PNG (RGBA); default: scripts/car.png",
    )
    group.add_argument(
        "--bev-car-length-fraction",
        type=float,
        default=0.11,
        help="Car icon length as a fraction of BEV image size (default: 0.11).",
    )
    group.add_argument(
        "--add-car-glb",
        dest="add_car_glb",
        action="store_true",
        help="Merge car_glb.glb into post-processed PLY (default: on).",
    )
    group.add_argument(
        "--no-add-car-glb",
        dest="add_car_glb",
        action="store_false",
        help="Skip merging car GLB into PLY.",
    )
    group.add_argument(
        "--car-glb",
        type=Path,
        default=None,
        help="Ego car GLB path (default: scripts/post_process/car_glb.glb).",
    )
    group.add_argument(
        "--car-length-m",
        type=float,
        default=DEFAULT_CAR_LENGTH_M,
        help="Target car length (m); default from add_ego_car.DEFAULT_CAR_LENGTH_M.",
    )
    group.add_argument(
        "--car-scale",
        type=float,
        default=None,
        help="Extra uniform scale after --car-length-m.",
    )
    group.add_argument(
        "--car-yaw-deg",
        type=float,
        default=DEFAULT_CAR_YAW_DEG,
        help="Car yaw about world +Y (degrees).",
    )
    group.add_argument(
        "--car-pitch-deg",
        type=float,
        default=DEFAULT_CAR_PITCH_DEG,
        help="Car pitch about world +X (degrees).",
    )
    group.add_argument(
        "--car-roll-deg",
        type=float,
        default=DEFAULT_CAR_ROLL_DEG,
        help="Car roll about world +Z (degrees).",
    )
    group.add_argument(
        "--car-offset-x",
        type=float,
        default=0.0,
        help="Car world +X offset from rig center (meters).",
    )
    group.add_argument(
        "--car-offset-y",
        type=float,
        default=0.0,
        help="Car world +Y offset from ground (meters).",
    )
    group.add_argument(
        "--car-offset-z",
        type=float,
        default=0.0,
        help="Car world +Z offset from rig center (meters).",
    )
    group.add_argument(
        "--car-sample-spacing",
        type=float,
        default=DEFAULT_CAR_SAMPLE_SPACING,
        help="Base voxel spacing for car mesh (scales with --car-length-m).",
    )

    vis = parser.add_argument_group("Point cloud visualization")
    vis.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        help="Open an Open3D viewer after saving the output point cloud (default).",
    )
    vis.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="Skip interactive point cloud visualization.",
    )
    vis.add_argument(
        "--viz-axis-size",
        type=float,
        default=1.0,
        help="Size of XYZ axes in the viewer (0 to hide).",
    )

    parser.set_defaults(post_process=True, visualize=True, export_bev=True, add_car_glb=True)


def maybe_postprocess_pointcloud(
    ply_path: Path,
    args: argparse.Namespace,
    *,
    label: str,
    dry_run: bool = False,
) -> Path | None:
    """Run configured post-processing on *ply_path* when enabled."""
    if not args.post_process:
        return None

    ply_path = ply_path.expanduser().resolve()
    out_path = postprocessed_output_path(ply_path)

    if dry_run:
        print(
            f"[dry-run] would post-process {ply_path.name} -> {out_path.name} "
            f"(task={args.post_process_task})"
        )
        return out_path

    if not ply_path.is_file():
        print(f"[{label}] skip post-process: {ply_path} not found")
        return None

    print(
        f"[{label}] post-processing {ply_path.name} -> {out_path.name} "
        f"(task={args.post_process_task})"
    )
    module = _load_ply_postprocess_module()
    module.postprocess_ply(
        ply_path,
        out_path,
        task=args.post_process_task,
        sky_axis=args.sky_axis,
        sky_side=args.sky_side,
        sky_keep_percentile=args.sky_keep_percentile,
        sky_max=args.sky_max,
        export_bev=args.export_bev,
        bev_car_png=args.bev_car_png,
        bev_car_length_fraction=args.bev_car_length_fraction,
        add_car_glb=args.add_car_glb,
        car_glb=args.car_glb,
        car_length_m=args.car_length_m,
        car_scale=args.car_scale,
        car_yaw_deg=args.car_yaw_deg,
        car_pitch_deg=args.car_pitch_deg,
        car_roll_deg=args.car_roll_deg,
        car_offset_x=args.car_offset_x,
        car_offset_y=args.car_offset_y,
        car_offset_z=args.car_offset_z,
        car_sample_spacing=args.car_sample_spacing,
        visualize=False,
    )
    print(f"[{label}] post-processed point cloud -> {out_path}")
    return out_path


def maybe_show_pointcloud(
    ply_path: Path,
    post_path: Path | None,
    args: argparse.Namespace,
    *,
    label: str,
    dry_run: bool = False,
) -> None:
    """Open the final point cloud in an Open3D viewer when enabled."""
    if dry_run or not args.visualize:
        return

    show_path = post_path if post_path is not None else ply_path
    show_path = show_path.expanduser().resolve()
    if not show_path.is_file():
        print(f"[{label}] skip visualize: {show_path} not found")
        return

    print(f"[{label}] opening viewer for {show_path.name}")
    view_ply = _load_view_ply_module()
    try:
        view_ply.show_pointcloud(
            str(show_path),
            window_name=f"{label}: {show_path.name}",
            axis_size=args.viz_axis_size,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[{label}] visualize failed: {exc}")
