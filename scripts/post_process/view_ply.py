"""Load a .ply point cloud and display it in an Open3D window."""

import argparse
import sys

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a .ply point cloud.")
    parser.add_argument("input", help="Path to the input .ply file")
    parser.add_argument(
        "--window-name",
        default="Point Cloud Viewer",
        help="Title of the visualization window",
    )
    parser.add_argument(
        "--axis-size",
        type=float,
        default=1.0,
        help="Size of XYZ axes (set to 0 or negative to hide axes)",
    )
    parser.add_argument(
        "--remove-sky",
        action="store_true",
        help="Remove sky points by clipping high points along a chosen axis",
    )
    parser.add_argument(
        "--sky-axis",
        choices=("x", "y", "z"),
        default="y",
        help="Axis used for sky clipping (default: y)",
    )
    parser.add_argument(
        "--sky-keep-percentile",
        type=float,
        default=0.98,
        help="Keep points below this height percentile (0~1, default: 0.98)",
    )
    parser.add_argument(
        "--sky-max",
        type=float,
        default=None,
        help="Absolute max coordinate on sky-axis; overrides percentile if set",
    )
    return parser.parse_args()


def clip_sky_points(
    pcd: o3d.geometry.PointCloud,
    axis: str,
    keep_percentile: float,
    sky_max: float | None,
) -> tuple[o3d.geometry.PointCloud, float, int]:
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_idx = axis_map[axis]
    pts = np.asarray(pcd.points)

    if not (0.0 < keep_percentile <= 1.0):
        raise ValueError("--sky-keep-percentile must be in (0, 1].")

    threshold = float(np.quantile(pts[:, axis_idx], keep_percentile)) if sky_max is None else sky_max
    keep_idx = np.where(pts[:, axis_idx] <= threshold)[0]
    removed = len(pts) - len(keep_idx)
    return pcd.select_by_index(keep_idx), threshold, removed


def main() -> None:
    args = parse_args()

    pcd = o3d.io.read_point_cloud(args.input)
    if len(pcd.points) == 0:
        print(f"Error: no points found in {args.input}", file=sys.stderr)
        sys.exit(1)

    n_pts = len(pcd.points)
    has_colors = pcd.has_colors()
    has_normals = pcd.has_normals()
    print(f"Loaded {n_pts} points from {args.input}")
    print(f"  colors: {'yes' if has_colors else 'no'}")
    print(f"  normals: {'yes' if has_normals else 'no'}")

    if not has_colors:
        pcd.paint_uniform_color([0.6, 0.6, 0.6])

    if args.remove_sky:
        pcd, threshold, removed = clip_sky_points(
            pcd,
            axis=args.sky_axis,
            keep_percentile=args.sky_keep_percentile,
            sky_max=args.sky_max,
        )
        print(
            f"Sky clipping on {args.sky_axis}-axis: threshold={threshold:.4f}, "
            f"removed={removed}, remaining={len(pcd.points)}"
        )
        if len(pcd.points) == 0:
            print("Error: all points removed by sky clipping", file=sys.stderr)
            sys.exit(1)

    geometries = [pcd]
    if args.axis_size > 0:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axis_size)
        geometries.append(axes)

    o3d.visualization.draw_geometries(
        geometries,
        window_name=args.window_name,
    )


if __name__ == "__main__":
    main()
