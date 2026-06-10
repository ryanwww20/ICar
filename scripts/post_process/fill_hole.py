import time

import numpy as np
import open3d as o3d
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay, cKDTree

# --- paths & parameters ---
input_path = "pointcloud.ply"
output_path = "7_ring_output.ply"

voxel_size = 0.05  # downsample for RANSAC
use_floor_crop = False
floor_height_ratio = 0.2
distance_threshold = 0.03
ransac_n = 3
num_iterations = 500
use_manual_plane = False  # True: skip RANSAC and use manual_plane_model directly
manual_plane_model = (0, 1, 0, 0)  # ax + by + cz + d = 0
manual_inlier_distance = 0.03  # |dist to plane| for inliers when use_manual_plane=True
grid_resolution = 0.003  # candidate spacing on the plane (m)
plane_band = 0.06  # |dist to plane| for classifying floor points (m)
max_fill_distance = 0.25  # must be within this distance of a floor inlier (m)
min_hole_radius = 0.0015  # gap must be at least this far from existing floor (m)
max_hole_radius = 0.07  # gap must be smaller than this — excludes large voids (m)
enable_large_hole_patch = True  # allow larger holes in a local ROI (e.g., near ego car)
large_hole_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # world-space ROI center
large_hole_roi_radius = 0.5  # meters
large_hole_max_radius = 5  # meters, used only inside ROI
large_hole_max_fill_distance = 0.5  # meters, used only inside ROI
force_fill_large_hole_in_roi = True  # if True, skip inlier-distance gate inside ROI
enable_large_hole_fine_grid = True  # use finer grid spacing inside ROI for denser fills
large_hole_fine_resolution = 0.003  # candidate spacing inside ROI (m); smaller = denser
hull_sample_max = 8_000  # subsample floor UV for convex hull
interp_method = "linear"  # linear | nearest | cubic
interp_max_samples = 50_000
reduce_white_fill = True  # replace overly bright interpolated colors with local robust median
fill_max_luma = 0.80  # grayscale luminance threshold for "too bright" fills
fill_local_k = 12  # neighbors in UV for robust local color estimate
remove_white_in_roi = True  # drop bright/white points inside ROI before hole filling
white_roi_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # ROI center for white removal
white_roi_radius = 2.0  # meters; only remove white points within this radius of center
white_luma_threshold = 0.80  # mean(R,G,B) above this is treated as "white"
use_green_fill = False  # True: solid green fill points; False: interpolate from floor colors
show_inlier_points = False  # True: overlay RANSAC floor inliers in red
visualize = True
save_output = True


class StepTimer:
    """Wall-clock timer for pipeline steps; prints each step as it finishes."""

    def __init__(self, *, live: bool = True) -> None:
        self.timings: list[tuple[str, float]] = []
        self.live = live
        self._t = time.perf_counter()

    def tick(self, name: str) -> None:
        now = time.perf_counter()
        ms = (now - self._t) * 1000
        self.timings.append((name, ms))
        if self.live:
            print(f"[timing] {name}: {ms:.1f} ms")
        self._t = now

    def report(self, *, n_pts: int | None = None) -> None:
        total = sum(ms for _, ms in self.timings)
        header = f"Timing summary ({n_pts:,} input pts)" if n_pts is not None else "Timing summary"
        print(f"{header}, total {total:.0f} ms:")
        for name, ms in self.timings:
            pct = 100.0 * ms / total if total > 0 else 0.0
            print(f"  {name}: {ms:.1f} ms ({pct:.1f}%)")


def crop_floor_candidates(pcd: o3d.geometry.PointCloud, height_ratio: float) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    y = pts[:, 1]
    y_min, y_max = float(y.min()), float(y.max())
    y_thresh = y_min + height_ratio * (y_max - y_min)
    indices = np.where(y <= y_thresh)[0]
    if len(indices) < ransac_n:
        raise RuntimeError(
            f"Floor crop too aggressive: {len(indices)} points (need >= {ransac_n}). "
            f"Increase floor_height_ratio (y in [{y_min:.3f}, {y_max:.3f}], thresh={y_thresh:.3f})."
        )
    cropped = pcd.select_by_index(indices)
    print(
        f"Floor crop (bottom {height_ratio * 100:.0f}% of y-range): "
        f"{len(pcd.points)} -> {len(cropped.points)} points "
        f"(y <= {y_thresh:.3f}, range [{y_min:.3f}, {y_max:.3f}])"
    )
    return cropped


def plane_signed_distance(pts: np.ndarray, plane_model: tuple[float, float, float, float]) -> np.ndarray:
    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        raise RuntimeError("Invalid plane normal.")
    return (pts @ normal + d) / norm


def points_near_plane(
    pts: np.ndarray, plane_model: tuple[float, float, float, float], band: float
) -> np.ndarray:
    mask = np.abs(plane_signed_distance(pts, plane_model)) <= band
    return pts[mask]


def plane_normal_unit(plane_model: tuple[float, float, float, float]) -> np.ndarray:
    a, b, c, _ = plane_model
    n = np.array([a, b, c], dtype=np.float64)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        raise RuntimeError("Invalid plane normal.")
    return n / norm


def plane_tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal tangents (u, v) spanning the plane; n = u x v."""
    ref = np.array([0.0, 1.0, 0.0]) if abs(normal[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    return u, v


def project_to_plane(pts: np.ndarray, plane_origin: np.ndarray, normal: np.ndarray) -> np.ndarray:
    rel = pts - plane_origin
    return pts - np.outer(rel @ normal, normal)


def to_plane_uv(pts: np.ndarray, plane_origin: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    rel = pts - plane_origin
    return np.column_stack([rel @ u, rel @ v])


def from_plane_uv(uv: np.ndarray, plane_origin: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return plane_origin + uv[:, 0:1] * u + uv[:, 1:2] * v


def merge_point_clouds(
    base: o3d.geometry.PointCloud, extra_pts: np.ndarray, extra_colors: np.ndarray
) -> o3d.geometry.PointCloud:
    base_pts = np.asarray(base.points)
    merged_pts = np.vstack([base_pts, extra_pts])
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(merged_pts)
    if base.has_colors():
        base_colors = np.asarray(base.colors)
    else:
        base_colors = np.full((len(base_pts), 3), 0.6)
    out.colors = o3d.utility.Vector3dVector(np.vstack([base_colors, extra_colors]))
    return out


def grid_candidates_in_hull(inlier_uv: np.ndarray, resolution: float) -> np.ndarray:
    """Regular grid in plane UV, keeping only cell centers inside the inlier convex hull."""
    if len(inlier_uv) < 3:
        raise RuntimeError("Need at least 3 floor inliers for a 2D convex hull.")

    hull = ConvexHull(inlier_uv)
    delaunay = Delaunay(inlier_uv[hull.vertices])

    uv_min = inlier_uv.min(axis=0)
    uv_max = inlier_uv.max(axis=0)
    us = np.arange(uv_min[0], uv_max[0] + resolution, resolution)
    vs = np.arange(uv_min[1], uv_max[1] + resolution, resolution)
    uu, vv = np.meshgrid(us, vs, indexing="xy")
    candidates = np.column_stack([uu.ravel(), vv.ravel()])

    inside = delaunay.find_simplex(candidates) >= 0
    return candidates[inside]


def subsample_uv(uv: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if len(uv) <= max_n:
        return uv
    idx = rng.choice(len(uv), size=max_n, replace=False)
    return uv[idx]


def filter_fill_candidates(
    candidate_pts: np.ndarray,
    candidate_uv: np.ndarray,
    inlier_pts: np.ndarray,
    floor_uv: np.ndarray,
    max_fill_dist: float,
    min_hole_dist: float,
    max_hole_dist: float,
) -> np.ndarray:
    """Keep only narrow gaps on the road, not large empty regions."""
    dist_inlier, _ = cKDTree(inlier_pts).query(candidate_pts, k=1, workers=-1)
    # 2D distance on the plane to existing floor — matches scan-line gap width
    dist_floor, _ = cKDTree(floor_uv).query(candidate_uv, k=1, workers=-1)

    near_road = dist_inlier < max_fill_dist
    is_small_gap = (dist_floor > min_hole_dist) & (dist_floor < max_hole_dist)
    keep = near_road & is_small_gap

    if enable_large_hole_patch:
        # Only relax criteria near a local ROI (default around origin), so we do
        # not accidentally fill large legitimate open regions far away.
        dist_roi = np.linalg.norm(candidate_pts - large_hole_center, axis=1)
        in_roi = dist_roi <= large_hole_roi_radius
        near_road_relaxed = dist_inlier < large_hole_max_fill_distance
        is_large_gap = (dist_floor > min_hole_dist) & (dist_floor < large_hole_max_radius)
        if force_fill_large_hole_in_roi:
            keep = keep | (in_roi & is_large_gap)
        else:
            keep = keep | (in_roi & near_road_relaxed & is_large_gap)

    return keep


def subsample_for_interp(
    known_uv: np.ndarray, known_colors: np.ndarray, max_samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n = len(known_uv)
    if n <= max_samples:
        return known_uv, known_colors
    idx = rng.choice(n, size=max_samples, replace=False)
    return known_uv[idx], known_colors[idx]


def local_median_colors_uv(
    query_uv: np.ndarray,
    known_uv: np.ndarray,
    known_colors: np.ndarray,
    k: int,
) -> np.ndarray:
    k_eff = max(1, min(k, len(known_uv)))
    dists, idx = cKDTree(known_uv).query(query_uv, k=k_eff, workers=-1)
    if k_eff == 1:
        idx = idx[:, None]
    return np.median(known_colors[idx], axis=1)


def interpolate_colors_uv(
    hole_uv: np.ndarray,
    inlier_uv: np.ndarray,
    inlier_colors: np.ndarray,
    method: str,
    rng: np.random.Generator,
) -> np.ndarray:
    inlier_uv, inlier_colors = subsample_for_interp(inlier_uv, inlier_colors, interp_max_samples, rng)
    colors = griddata(inlier_uv, inlier_colors, hole_uv, method=method, fill_value=np.nan)
    nan_mask = np.isnan(colors).any(axis=1)
    if nan_mask.any():
        colors[nan_mask] = griddata(inlier_uv, inlier_colors, hole_uv[nan_mask], method="nearest")

    if reduce_white_fill and len(colors) > 0:
        local_med = local_median_colors_uv(hole_uv, inlier_uv, inlier_colors, k=fill_local_k)
        luma = colors.mean(axis=1)
        bright_mask = luma > fill_max_luma
        if bright_mask.any():
            colors[bright_mask] = local_med[bright_mask]

    return np.clip(colors, 0.0, 1.0)


def nearest_colors(pcd: o3d.geometry.PointCloud, query_pts: np.ndarray) -> np.ndarray:
    if pcd.has_colors():
        tree = cKDTree(np.asarray(pcd.points))
        _, idx = tree.query(query_pts, k=1, workers=-1)
        return np.asarray(pcd.colors)[idx]
    return np.full((len(query_pts), 3), 0.6)


def fill_holes_on_plane(
    plane_model: tuple[float, float, float, float],
    inlier_pts: np.ndarray,
    original_pcd: o3d.geometry.PointCloud,
    resolution: float,
    plane_band: float,
    max_fill_dist: float,
    min_hole_dist: float,
    max_hole_dist: float,
    interp_method: str,
    rng: np.random.Generator,
    use_green: bool = False,
    timer: StepTimer | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, int]]:
    normal = plane_normal_unit(plane_model)
    u, v = plane_tangent_basis(normal)

    original_pts = np.asarray(original_pcd.points)
    floor_pts = points_near_plane(original_pts, plane_model, plane_band)
    if len(floor_pts) == 0:
        raise RuntimeError(f"No floor points within plane_band={plane_band} m.")
    if timer:
        timer.tick("fill: floor classify")

    inlier_on_plane = project_to_plane(inlier_pts, inlier_pts.mean(axis=0), normal)
    floor_on_plane = project_to_plane(floor_pts, inlier_on_plane.mean(axis=0), normal)
    plane_origin = inlier_on_plane.mean(axis=0)

    floor_uv = to_plane_uv(floor_on_plane, plane_origin, u, v)
    hull_uv = subsample_uv(floor_uv, hull_sample_max, rng)
    if timer:
        timer.tick("fill: plane projection")

    candidates_uv = grid_candidates_in_hull(hull_uv, resolution)
    candidates_3d = from_plane_uv(candidates_uv, plane_origin, u, v)

    if enable_large_hole_patch and enable_large_hole_fine_grid:
        dist_roi = np.linalg.norm(candidates_3d - large_hole_center, axis=1)
        outside_roi = dist_roi > large_hole_roi_radius
        candidates_uv = candidates_uv[outside_roi]
        candidates_3d = candidates_3d[outside_roi]

        fine_uv = grid_candidates_in_hull(hull_uv, large_hole_fine_resolution)
        fine_3d = from_plane_uv(fine_uv, plane_origin, u, v)
        fine_in_roi = np.linalg.norm(fine_3d - large_hole_center, axis=1) <= large_hole_roi_radius
        candidates_uv = np.vstack([candidates_uv, fine_uv[fine_in_roi]])
        candidates_3d = np.vstack([candidates_3d, fine_3d[fine_in_roi]])
        print(
            f"Fine grid in ROI (r={large_hole_roi_radius} m, "
            f"spacing={large_hole_fine_resolution} m): {int(fine_in_roi.sum())} candidates"
        )

    if timer:
        timer.tick("fill: hull grid")

    keep = filter_fill_candidates(
        candidates_3d,
        candidates_uv,
        inlier_on_plane,
        floor_uv,
        max_fill_dist,
        min_hole_dist,
        max_hole_dist,
    )
    filled_pts = candidates_3d[keep]
    filled_uv = candidates_uv[keep]
    if timer:
        timer.tick("fill: filter candidates")

    filled_colors = None
    if len(filled_pts) > 0:
        if use_green:
            filled_colors = np.tile([0.0, 1.0, 0.0], (len(filled_pts), 1))
            if timer:
                timer.tick("fill: green color")
        else:
            floor_colors = nearest_colors(original_pcd, floor_pts)
            filled_colors = interpolate_colors_uv(
                filled_uv, floor_uv, floor_colors, interp_method, rng
            )
            if timer:
                timer.tick("fill: color interpolation")

    stats = {
        "hull_candidates": len(candidates_uv),
        "hole_cells": int(keep.sum()),
    }
    return filled_pts, filled_colors, stats


def remove_white_points_in_roi(
    pcd: o3d.geometry.PointCloud,
    center: np.ndarray,
    radius: float,
    luma_threshold: float,
) -> tuple[o3d.geometry.PointCloud, int]:
    """Drop bright/white points within `radius` of `center`.

    These stray points sit inside large road holes and make the gap look
    "already filled", which blocks hole detection. Removing them turns the
    region back into a genuine hole that can be filled.
    """
    if not pcd.has_colors():
        return pcd, 0

    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    dist_roi = np.linalg.norm(pts - center, axis=1)
    luma = colors.mean(axis=1)
    drop_mask = (dist_roi <= radius) & (luma >= luma_threshold)

    keep_idx = np.where(~drop_mask)[0]
    removed = int(drop_mask.sum())
    return pcd.select_by_index(keep_idx), removed


def main():
    timer = StepTimer()

    pcd = o3d.io.read_point_cloud(input_path)
    if len(pcd.points) == 0:
        raise RuntimeError(f"No points in {input_path}")
    timer.tick("read ply")

    if remove_white_in_roi:
        pcd, n_removed = remove_white_points_in_roi(
            pcd, white_roi_center, white_roi_radius, white_luma_threshold
        )
        print(
            f"White removal in ROI (center={white_roi_center.tolist()}, "
            f"r={white_roi_radius} m, luma>={white_luma_threshold}): removed {n_removed} points"
        )
        timer.tick("remove white in ROI")

    n_raw = len(pcd.points)
    original_pts = np.asarray(pcd.points)

    pcd_ransac = pcd.voxel_down_sample(voxel_size)
    print(f"Voxel downsample ({voxel_size} m): {n_raw} -> {len(pcd_ransac.points)} points")
    timer.tick("voxel downsample")

    if use_floor_crop:
        pcd_ransac = crop_floor_candidates(pcd_ransac, floor_height_ratio)
        timer.tick("floor crop")

    if use_manual_plane:
        plane_model = tuple(manual_plane_model)
        ransac_pts = np.asarray(pcd_ransac.points)
        dists = np.abs(plane_signed_distance(ransac_pts, plane_model))
        inlier_idx = np.where(dists <= manual_inlier_distance)[0].tolist()
        timer.tick("manual plane")
        print("Using manual plane (RANSAC skipped)")
    else:
        plane_model, inlier_idx = pcd_ransac.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        timer.tick("ransac plane")

    a, b, c, d = plane_model
    print(f"Plane equation: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")

    inlier_pts = np.asarray(pcd_ransac.select_by_index(inlier_idx).points)
    print(f"Floor inliers: {len(inlier_pts)}")

    rng = np.random.default_rng(0)
    floor_n = len(points_near_plane(original_pts, plane_model, plane_band))
    print(f"Floor points (|dist| <= {plane_band} m): {floor_n}")
    timer.tick("count floor points")

    filled_pts, filled_colors, stats = fill_holes_on_plane(
        plane_model,
        inlier_pts,
        pcd,
        grid_resolution,
        plane_band,
        max_fill_distance,
        min_hole_radius,
        max_hole_radius,
        interp_method,
        rng,
        use_green=use_green_fill,
        timer=timer,
    )

    color_mode = "green" if use_green_fill else interp_method
    print(
        f"Hull grid candidates: {stats['hull_candidates']}, "
        f"accepted fills: {stats['hole_cells']} "
        f"(gap band: {min_hole_radius}–{max_hole_radius} m on floor, "
        f"inlier<{max_fill_distance} m, colors={color_mode})"
    )
    if enable_large_hole_patch:
        print(
            "Large-hole local patch enabled: "
            f"center={large_hole_center.tolist()}, roi={large_hole_roi_radius} m, "
            f"max_gap={large_hole_max_radius} m, inlier<{large_hole_max_fill_distance} m"
        )

    filled_pcd = o3d.geometry.PointCloud()
    filled_pcd.points = o3d.utility.Vector3dVector(filled_pts)
    if filled_colors is not None:
        filled_pcd.colors = o3d.utility.Vector3dVector(filled_colors)
    else:
        filled_pcd.paint_uniform_color([0.0, 1.0, 0.0])
    timer.tick("prepare visualization")

    if save_output:
        if filled_colors is None:
            raise RuntimeError("No fill points produced; nothing to save.")
        merged_pcd = merge_point_clouds(pcd, filled_pts, filled_colors)
        timer.tick("merge point clouds")
        o3d.io.write_point_cloud(output_path, merged_pcd, write_ascii=False)
        timer.tick("write ply")
        print(
            f"Saved {len(merged_pcd.points)} points ({len(filled_pts)} filled, "
            f"colors={color_mode}) to {output_path}"
        )

    if visualize:
        vis_geoms: list[o3d.geometry.Geometry] = [pcd, filled_pcd]
        if show_inlier_points:
            inlier_vis = pcd_ransac.select_by_index(inlier_idx)
            inlier_vis.paint_uniform_color([1.0, 0.0, 0.0])
            vis_geoms.insert(1, inlier_vis)
        o3d.visualization.draw_geometries(
            vis_geoms,
            window_name=f"Floor fill (filled={color_mode})",
        )
        timer.tick("visualize (interactive)")

    timer.report(n_pts=n_raw)


if __name__ == "__main__":
    main()
