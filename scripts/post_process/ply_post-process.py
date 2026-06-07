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


def run_fill_hole(args: argparse.Namespace) -> None:
    """Run fill_hole.py main pipeline."""
    import fill_hole

    fill_hole.input_path = args.input
    fill_hole.output_path = args.output
    fill_hole.main()


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
        fill_hole.main()
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
}


def build_parser() -> argparse.ArgumentParser:
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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        print_task_list()
        return 0

    task_names = args.tasks if args.tasks else ["full_pipeline"]
    if not args.tasks:
        print("No task specified. Defaulting to: full_pipeline")

    for name in task_names:
        try:
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
