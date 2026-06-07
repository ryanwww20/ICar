#!/usr/bin/env python3
"""Shared PLY post-processing helpers for run_vggt_* scripts."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
POST_PROCESS_DIR = SCRIPT_DIR / "post_process"
PLY_POST_PROCESS_PATH = POST_PROCESS_DIR / "ply_post-process.py"


def _load_ply_postprocess_module():
    spec = importlib.util.spec_from_file_location(
        "ply_post_process",
        PLY_POST_PROCESS_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load {PLY_POST_PROCESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def postprocessed_output_path(ply_path: Path) -> Path:
    """Return ``{stem}_post.ply`` next to the raw VGGT output."""
    return ply_path.with_name(f"{ply_path.stem}_post{ply_path.suffix}")


def add_postprocess_args(parser: argparse.ArgumentParser) -> None:
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
        choices=("full_pipeline", "fill_hole"),
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
    parser.set_defaults(post_process=True)


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
        visualize=False,
    )
    print(f"[{label}] post-processed point cloud -> {out_path}")
    return out_path
