#!/usr/bin/env python3
"""
Simple post-process task runner for PLY workflows.

Usage examples:
  python ply_post-process.py --list
  python ply_post-process.py fill_hole
  python ply_post-process.py fill_hole another_task
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import open3d as o3d

TaskFunc = Callable[[argparse.Namespace], None]


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    func: TaskFunc


def remove_sky_points(
    pcd: "o3d.geometry.PointCloud",
    axis: str = "y",
    keep_percentile: float = 0.9,
    sky_max: float | None = None,
    sky_side: str = "high",
) -> tuple["o3d.geometry.PointCloud", float, int]:
    """
    Remove sky-like high points by clipping along one axis.

    Parameters
    ----------
    pcd
        Input point cloud.
    axis
        Axis used for clipping: "x", "y", or "z".
    keep_percentile
        Fraction of points to keep as non-sky, must be in (0, 1].
        Ignored when `sky_max` is provided.
    sky_max
        Absolute threshold on the chosen axis.
    sky_side
        Which side is sky on the chosen axis:
        "high" -> sky has larger axis values, "low" -> sky has smaller axis values.

    Returns
    -------
    filtered_pcd, threshold, removed_count
    """
    import numpy as np

    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be one of: x, y, z")
    if not (0.0 < keep_percentile <= 1.0):
        raise ValueError("keep_percentile must be in (0, 1].")
    if sky_side not in {"high", "low"}:
        raise ValueError("sky_side must be one of: high, low")

    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        raise ValueError("Input point cloud is empty.")

    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    axis_values = pts[:, axis_idx]

    if sky_max is None:
        q = keep_percentile if sky_side == "high" else (1.0 - keep_percentile)
        threshold = float(np.quantile(axis_values, q))
    else:
        threshold = sky_max

    if sky_side == "high":
        keep_idx = np.where(axis_values <= threshold)[0]
    else:
        keep_idx = np.where(axis_values >= threshold)[0]
    removed_count = len(pts) - len(keep_idx)
    filtered_pcd = pcd.select_by_index(keep_idx)
    return filtered_pcd, threshold, removed_count


def _apply_car_config_to_fill_hole(args: argparse.Namespace) -> None:
    """Copy car GLB placement flags from CLI into fill_hole module globals."""
    import fill_hole
    from add_ego_car import (
        DEFAULT_CAR_LENGTH_M,
        DEFAULT_CAR_PITCH_DEG,
        DEFAULT_CAR_ROLL_DEG,
        DEFAULT_CAR_SAMPLE_SPACING,
        DEFAULT_CAR_SCALE,
        DEFAULT_CAR_YAW_DEG,
    )

    fill_hole.add_car_glb = bool(getattr(args, "add_car_glb", True))
    car_glb = getattr(args, "car_glb", None)
    fill_hole.car_glb_path = str(car_glb) if car_glb else None
    fill_hole.car_length_m = float(getattr(args, "car_length_m", DEFAULT_CAR_LENGTH_M))
    fill_hole.car_scale = getattr(args, "car_scale", DEFAULT_CAR_SCALE)
    fill_hole.car_yaw_deg = float(getattr(args, "car_yaw_deg", DEFAULT_CAR_YAW_DEG))
    fill_hole.car_pitch_deg = float(getattr(args, "car_pitch_deg", DEFAULT_CAR_PITCH_DEG))
    fill_hole.car_roll_deg = float(getattr(args, "car_roll_deg", DEFAULT_CAR_ROLL_DEG))
    fill_hole.car_offset_x = float(getattr(args, "car_offset_x", 0.0))
    fill_hole.car_offset_y = float(getattr(args, "car_offset_y", 0.0))
    fill_hole.car_offset_z = float(getattr(args, "car_offset_z", 0.0))
    fill_hole.car_sample_spacing = float(
        getattr(args, "car_sample_spacing", DEFAULT_CAR_SAMPLE_SPACING)
    )


def run_fill_hole(args: argparse.Namespace) -> None:
    """Run fill_hole.py main pipeline."""
    import fill_hole

    fill_hole.input_path = args.input
    fill_hole.output_path = args.output
    fill_hole.visualize = bool(getattr(args, "visualize", False))
    _apply_car_config_to_fill_hole(args)
    fill_hole.main()


def _bev_kwargs_from_namespace(args: argparse.Namespace) -> dict:
    car_png = getattr(args, "bev_car_png", None)
    add_car = bool(getattr(args, "add_car_glb", True))
    overlay_png = bool(getattr(args, "bev_overlay_car_png", not add_car))
    return {
        "image_size": getattr(args, "bev_size", 1024),
        "extent_percentile": getattr(args, "bev_extent_percentile", 72.0),
        "car_length_fraction": getattr(args, "bev_car_length_fraction", 0.11),
        "car_png": Path(car_png) if car_png else None,
        "overlay_car_png": overlay_png,
    }


def _export_rear_top_bev_from_ply(ply_path: Path, args: argparse.Namespace) -> Path:
    import open3d as o3d

    from add_ego_car import save_bev_png

    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        raise RuntimeError(f"No points in {ply_path}")

    bev_path = ply_path.with_name(f"{ply_path.stem}_bev.png")
    save_bev_png(pcd, bev_path, **_bev_kwargs_from_namespace(args))
    return bev_path


def run_export_bev(args: argparse.Namespace) -> None:
    """Export rear-top BEV PNG with car.png overlay; PLY is passed through unchanged."""
    import open3d as o3d

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    pcd = o3d.io.read_point_cloud(str(input_path))
    if len(pcd.points) == 0:
        raise RuntimeError(f"No points in {input_path}")

    if input_path.resolve() != output_path.resolve():
        o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)
        print(f"Copied {len(pcd.points)} points to {output_path}")
    else:
        print(f"Using {len(pcd.points)} points from {input_path}")

    _export_rear_top_bev_from_ply(output_path, args)


def run_full_pipeline(args: argparse.Namespace) -> None:
    """Run sky removal first, then fill_hole pipeline."""
    import open3d as o3d
    import fill_hole

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    pcd = o3d.io.read_point_cloud(str(input_path))
    if len(pcd.points) == 0:
        raise RuntimeError(f"No points in {input_path}")

    filtered_pcd, threshold, removed = remove_sky_points(
        pcd,
        axis=args.sky_axis,
        keep_percentile=args.sky_keep_percentile,
        sky_max=args.sky_max,
        sky_side=args.sky_side,
    )
    print(
        f"Sky removal: axis={args.sky_axis}, side={args.sky_side}, threshold={threshold:.4f}, "
        f"removed={removed}, remaining={len(filtered_pcd.points)}"
    )
    if len(filtered_pcd.points) == 0:
        raise RuntimeError("All points were removed by sky filtering.")

    temp_dir = input_path.parent
    with tempfile.NamedTemporaryFile(
        suffix="_sky_filtered.ply",
        dir=temp_dir,
        delete=False,
    ) as fp:
        temp_input = Path(fp.name)
    try:
        o3d.io.write_point_cloud(str(temp_input), filtered_pcd, write_ascii=False)
        print(f"Temporary sky-filtered cloud: {temp_input.name}")

        fill_hole.input_path = str(temp_input)
        fill_hole.output_path = args.output
        fill_hole.visualize = bool(getattr(args, "visualize", False))
        _apply_car_config_to_fill_hole(args)
        fill_hole.main()

        if args.export_bev:
            _export_rear_top_bev_from_ply(Path(args.output), args)
    finally:
        if temp_input.exists():
            temp_input.unlink()


TASKS: dict[str, Task] = {
    "fill_hole": Task(
        name="fill_hole",
        description="Execute the hole-filling pipeline in fill_hole.py.",
        func=run_fill_hole,
    ),
    "full_pipeline": Task(
        name="full_pipeline",
        description="Remove sky points, then execute fill_hole pipeline.",
        func=run_full_pipeline,
    ),
    "export_bev": Task(
        name="export_bev",
        description="Export rear-top BEV PNG with car.png overlay (PLY unchanged).",
        func=run_export_bev,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    from add_ego_car import (
        DEFAULT_CAR_LENGTH_M,
        DEFAULT_CAR_PITCH_DEG,
        DEFAULT_CAR_ROLL_DEG,
        DEFAULT_CAR_SAMPLE_SPACING,
        DEFAULT_CAR_SCALE,
        DEFAULT_CAR_YAW_DEG,
    )

    parser = argparse.ArgumentParser(
        description="Run one or more PLY post-processing tasks.",
    )
    parser.add_argument(
        "tasks",
        nargs="*",
        help="Task names to execute in order. Use --list to see available tasks.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print all available tasks and exit.",
    )
    parser.add_argument(
        "--input",
        default="pointcloud.ply",
        help="Input PLY path for processing tasks.",
    )
    parser.add_argument(
        "--output",
        default="post_processed_output.ply",
        help="Output PLY path for processing tasks.",
    )
    parser.add_argument(
        "--sky-axis",
        choices=("x", "y", "z"),
        default="y",
        help="Axis used for sky clipping in full_pipeline.",
    )
    parser.add_argument(
        "--sky-side",
        choices=("high", "low"),
        default="low",
        help="Sky location on chosen axis: high (large values) or low (small values).",
    )
    parser.add_argument(
        "--sky-keep-percentile",
        type=float,
        default=0.9,
        help="Fraction of non-sky points to keep in full_pipeline.",
    )
    parser.add_argument(
        "--sky-max",
        type=float,
        default=None,
        help="Absolute sky threshold in full_pipeline; overrides percentile.",
    )
    parser.add_argument(
        "--export-bev",
        dest="export_bev",
        action="store_true",
        help="Export rear-top BEV PNG with car.png overlay after full_pipeline (default: on).",
    )
    parser.add_argument(
        "--no-export-bev",
        dest="export_bev",
        action="store_false",
        help="Skip the rear-top BEV PNG export.",
    )
    parser.add_argument(
        "--bev-size",
        type=int,
        default=1024,
        help="BEV image width/height in pixels.",
    )
    parser.add_argument(
        "--bev-extent-percentile",
        type=float,
        default=72.0,
        help="Radial distance percentile that sets the view zoom (sixview default: 72).",
    )
    parser.add_argument(
        "--bev-car-png",
        type=Path,
        default=None,
        help="Top-down car PNG (RGBA); default: scripts/car.png",
    )
    parser.add_argument(
        "--bev-car-length-fraction",
        type=float,
        default=0.11,
        help="Car icon length as a fraction of image size (sixview default: 0.11).",
    )
    car = parser.add_argument_group("Ego car GLB (merged into output PLY)")
    car.add_argument(
        "--add-car-glb",
        dest="add_car_glb",
        action="store_true",
        help="Merge car_glb.glb into the output PLY at the scene center (default: on).",
    )
    car.add_argument(
        "--no-add-car-glb",
        dest="add_car_glb",
        action="store_false",
        help="Skip merging the car GLB into the PLY.",
    )
    car.add_argument(
        "--car-glb",
        type=Path,
        default=None,
        help="Ego car GLB path (default: scripts/post_process/car_glb.glb).",
    )
    car.add_argument(
        "--car-length-m",
        type=float,
        default=DEFAULT_CAR_LENGTH_M,
        help="Target car length after normalization (meters); default from add_ego_car.DEFAULT_CAR_LENGTH_M.",
    )
    car.add_argument(
        "--car-scale",
        type=float,
        default=DEFAULT_CAR_SCALE,
        help="Extra uniform scale after --car-length-m; default from add_ego_car.DEFAULT_CAR_SCALE.",
    )
    car.add_argument(
        "--car-yaw-deg",
        type=float,
        default=DEFAULT_CAR_YAW_DEG,
        help="Yaw about world +Y (degrees); turn car left/right.",
    )
    car.add_argument(
        "--car-pitch-deg",
        type=float,
        default=DEFAULT_CAR_PITCH_DEG,
        help="Pitch about world +X (degrees).",
    )
    car.add_argument(
        "--car-roll-deg",
        type=float,
        default=DEFAULT_CAR_ROLL_DEG,
        help="Roll about world +Z (degrees).",
    )
    car.add_argument(
        "--car-offset-x",
        type=float,
        default=0.0,
        help="World +X offset from rig center (meters).",
    )
    car.add_argument(
        "--car-offset-y",
        type=float,
        default=0.0,
        help="World +Y offset from ground (meters; +Y is down).",
    )
    car.add_argument(
        "--car-offset-z",
        type=float,
        default=0.0,
        help="World +Z offset from rig center (meters).",
    )
    car.add_argument(
        "--car-sample-spacing",
        type=float,
        default=DEFAULT_CAR_SAMPLE_SPACING,
        help="Base voxel spacing for car mesh sampling (scales down with --car-length-m).",
    )
    car.add_argument(
        "--bev-overlay-car-png",
        dest="bev_overlay_car_png",
        action="store_true",
        help="Also paste car.png on the BEV image (default: off when --add-car-glb).",
    )
    car.add_argument(
        "--no-bev-overlay-car-png",
        dest="bev_overlay_car_png",
        action="store_false",
        help="Skip car.png overlay on BEV (default when car GLB is merged).",
    )
    parser.set_defaults(export_bev=True, add_car_glb=True, bev_overlay_car_png=None)
    return parser


def print_task_list() -> None:
    print("Available tasks:")
    for task in TASKS.values():
        print(f"  - {task.name}: {task.description}")


def execute_task(task_name: str, args: argparse.Namespace) -> None:
    task = TASKS.get(task_name)
    if task is None:
        raise KeyError(task_name)

    print(f"\n=== Running task: {task.name} ===")
    task.func(args)
    print(f"=== Finished task: {task.name} ===")


def postprocess_ply(
    input_path: Path | str,
    output_path: Path | str,
    *,
    task: str = "full_pipeline",
    sky_axis: str = "y",
    sky_side: str = "low",
    sky_keep_percentile: float = 0.9,
    sky_max: float | None = None,
    export_bev: bool = True,
    bev_size: int = 1024,
    bev_extent_percentile: float = 72.0,
    bev_car_png: Path | str | None = None,
    bev_car_length_fraction: float = 0.11,
    add_car_glb: bool = True,
    car_glb: Path | str | None = None,
    car_length_m: float | None = None,
    car_scale: float | None = None,
    car_yaw_deg: float | None = None,
    car_pitch_deg: float | None = None,
    car_roll_deg: float | None = None,
    car_offset_x: float = 0.0,
    car_offset_y: float = 0.0,
    car_offset_z: float = 0.0,
    car_sample_spacing: float | None = None,
    bev_overlay_car_png: bool | None = None,
    visualize: bool = False,
) -> Path:
    """Run one post-processing task programmatically (non-interactive by default)."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    post_dir = Path(__file__).resolve().parent
    if str(post_dir) not in sys.path:
        sys.path.insert(0, str(post_dir))

    import fill_hole

    fill_hole.visualize = visualize

    from add_ego_car import (
        DEFAULT_CAR_LENGTH_M,
        DEFAULT_CAR_PITCH_DEG,
        DEFAULT_CAR_ROLL_DEG,
        DEFAULT_CAR_SAMPLE_SPACING,
        DEFAULT_CAR_SCALE,
        DEFAULT_CAR_YAW_DEG,
    )

    if bev_overlay_car_png is None:
        bev_overlay_car_png = not add_car_glb
    if car_length_m is None:
        car_length_m = DEFAULT_CAR_LENGTH_M
    if car_scale is None:
        car_scale = DEFAULT_CAR_SCALE
    if car_yaw_deg is None:
        car_yaw_deg = DEFAULT_CAR_YAW_DEG
    if car_pitch_deg is None:
        car_pitch_deg = DEFAULT_CAR_PITCH_DEG
    if car_roll_deg is None:
        car_roll_deg = DEFAULT_CAR_ROLL_DEG
    if car_sample_spacing is None:
        car_sample_spacing = DEFAULT_CAR_SAMPLE_SPACING

    args = argparse.Namespace(
        input=str(input_path),
        output=str(output_path),
        sky_axis=sky_axis,
        sky_side=sky_side,
        sky_keep_percentile=sky_keep_percentile,
        sky_max=sky_max,
        export_bev=export_bev,
        bev_size=bev_size,
        bev_extent_percentile=bev_extent_percentile,
        bev_car_png=bev_car_png,
        bev_car_length_fraction=bev_car_length_fraction,
        add_car_glb=add_car_glb,
        car_glb=car_glb,
        car_length_m=car_length_m,
        car_scale=car_scale,
        car_yaw_deg=car_yaw_deg,
        car_pitch_deg=car_pitch_deg,
        car_roll_deg=car_roll_deg,
        car_offset_x=car_offset_x,
        car_offset_y=car_offset_y,
        car_offset_z=car_offset_z,
        car_sample_spacing=car_sample_spacing,
        bev_overlay_car_png=bev_overlay_car_png,
        visualize=visualize,
    )

    if task not in TASKS:
        raise ValueError(f"Unknown task: {task!r}. Available: {', '.join(TASKS)}")

    execute_task(task, args)

    if not output_path.exists():
        raise RuntimeError(f"Post-process did not create output: {output_path}")
    return output_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.bev_overlay_car_png is None:
        args.bev_overlay_car_png = not args.add_car_glb

    if args.list:
        print_task_list()
        return 0

    task_names = args.tasks if args.tasks else ["full_pipeline"]
    if not args.tasks:
        print("No task specified. Defaulting to: full_pipeline")

    for idx, name in enumerate(task_names):
        try:
            if idx > 0:
                args.input = args.output
            execute_task(name, args)
        except KeyError:
            print(f"Unknown task: {name}", file=sys.stderr)
            print("Use --list to view available tasks.", file=sys.stderr)
            return 2
        except Exception as exc:  # noqa: BLE001
            print(f"Task '{name}' failed: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
