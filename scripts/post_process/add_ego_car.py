"""Rear-top scene export with car.png overlay (vggt_sixview_scene style).

Coordinate conventions (VGGT / OpenCV native, no axis flip):
- +Z: forward (car front, image up)
- +X: right (image horizontal)
- -Y: camera side (camera sits at negative Y, looking toward +Y and +Z)
- ``car.png`` front points up in the image (+Z), pasted at the rig center.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CAR_PNG = SCRIPT_DIR / "car.png"

# Render defaults (edit here).
BEV_IMAGE_SIZE = 1024
BEV_EXTENT_PERCENTILE = 72.0
BEV_MARGIN_FRACTION = 0.08
CAR_LENGTH_FRACTION = 0.11

# Rear-top oblique weights (native Y-down frame).
# v = -rel_z + OBLIQUE_Y_WEIGHT * rel_y  -> +Z up, +Y (down) tilts scene.
REAR_TOP_OBLIQUE_Y_WEIGHT = 0.1


def _load_car_rgba(car_path: Path):
    """Load top-down car PNG; key neutral gray checkerboard if no real alpha."""
    from PIL import Image

    car = Image.open(car_path)
    if car.mode != "RGBA":
        car = car.convert("RGBA")
    arr = np.asarray(car)
    if arr[:, :, 3].min() == 255:
        rgb = arr[:, :, :3].astype(np.int16)
        spread = rgb.max(axis=2) - rgb.min(axis=2)
        mean = rgb.mean(axis=2)
        bg = (spread <= 18) & (mean >= 165)
        arr = arr.copy()
        arr[:, :, 3] = np.where(bg, 0, 255).astype(np.uint8)
        car = Image.fromarray(arr, "RGBA")

    alpha = np.asarray(car)[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if ys.size == 0:
        return car
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return car.crop((x0, y0, x1, y1))


def _overlay_car_icon(
    base_rgb: np.ndarray,
    car_path: Path,
    *,
    center_u: float,
    center_v: float,
    car_length_px: float,
) -> np.ndarray:
    """Paste RGBA car icon at rig center; PNG front should point up (+Z)."""
    from PIL import Image

    car = _load_car_rgba(car_path)
    car_h = max(int(round(float(car_length_px))), 1)
    car_w = max(int(round(car_h * (car.width / car.height))), 1)
    car = car.resize((car_w, car_h), Image.Resampling.LANCZOS)

    base = Image.fromarray((np.clip(base_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)).convert("RGBA")
    left = int(round(center_u - car_w / 2.0))
    top = int(round(center_v - car_h / 2.0))
    base.paste(car, (left, top), car)
    return np.asarray(base.convert("RGB"), dtype=np.float64) / 255.0


def estimate_scene_center_xz(pts: np.ndarray) -> tuple[float, float]:
    """Robust XZ scene center (median), analogous to sixview's rig center."""
    return float(np.median(pts[:, 0])), float(np.median(pts[:, 2]))


def estimate_ground_y(pts: np.ndarray, center_xz: tuple[float, float]) -> float:
    """Median Y of points near the rig center on the ground plane."""
    center_x, center_z = center_xz
    r = np.hypot(pts[:, 0] - center_x, pts[:, 2] - center_z)
    if r.size == 0:
        return 0.0
    r_thresh = float(np.percentile(r, 30.0))
    near = pts[r <= max(r_thresh, 1e-6)]
    return float(np.median(near[:, 1])) if near.size else 0.0


def _project_rear_top_native(
    pts: np.ndarray,
    *,
    center_xz: tuple[float, float],
    ground_y: float,
    oblique_y_weight: float = REAR_TOP_OBLIQUE_Y_WEIGHT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map native VGGT coords to rear-top image axes.

    Image horizontal (u) = world +X.
    Image vertical (v)     = -world Z (+Z forward points up) with Y oblique tilt.
    Depth (for draw order) = distance along camera ray from (-Y, -Z) toward (+Y, +Z).
    """
    center_x, center_z = center_xz
    rel_x = pts[:, 0] - center_x
    rel_z = pts[:, 2] - center_z
    rel_y = pts[:, 1] - ground_y

    u = rel_x
    v = -rel_z + oblique_y_weight * rel_y

    # Camera at -Y / -Z quadrant looking toward +Y / +Z.
    view = np.array([0.0, 1.0, 1.0], dtype=np.float64)
    view /= np.linalg.norm(view)
    depth = pts[:, 0] * 0.0 + rel_y * view[1] + rel_z * view[2]
    return u, v, depth


def render_rear_top_bev(
    pts: np.ndarray,
    cols: np.ndarray,
    *,
    center_xz: tuple[float, float] = (0.0, 0.0),
    ground_y: float | None = None,
    image_size: int = BEV_IMAGE_SIZE,
    extent_percentile: float = BEV_EXTENT_PERCENTILE,
    margin_fraction: float = BEV_MARGIN_FRACTION,
    oblique_y_weight: float = REAR_TOP_OBLIQUE_Y_WEIGHT,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """
    Rear-top view in native VGGT coords (+Z forward, -Y camera side).

    Uses the same X/Z image axes as sixview top-down BEV (X horizontal, +Z up),
    plus a Y oblique term for the behind-and-above perspective.
    """
    pts = np.asarray(pts, dtype=np.float64)
    cols = np.asarray(cols, dtype=np.float64)

    if ground_y is None:
        ground_y = estimate_ground_y(pts, center_xz)

    center_x, center_z = center_xz
    r_l2 = np.hypot(pts[:, 0] - center_x, pts[:, 2] - center_z)
    pct = float(np.clip(extent_percentile, 1.0, 100.0))
    radius = float(np.percentile(r_l2, pct)) if r_l2.size else 0.0
    keep = r_l2 <= radius if radius > 0.0 else np.ones(r_l2.shape, dtype=bool)
    pts = pts[keep]
    cols = cols[keep]
    if pts.shape[0] == 0:
        raise RuntimeError("No points left for rear-top view after radial crop")

    u, v, depth = _project_rear_top_native(
        pts,
        center_xz=center_xz,
        ground_y=ground_y,
        oblique_y_weight=oblique_y_weight,
    )

    half_u = max(float(np.percentile(np.abs(u), pct)), 1e-6) * (1.0 + float(margin_fraction))
    half_v = max(float(np.percentile(np.abs(v), pct)), 1e-6) * (1.0 + float(margin_fraction))
    size = int(image_size)

    # Farther points first; nearer points overwrite.
    order = np.argsort(depth, kind="stable")
    u, v, cols = u[order], v[order], cols[order]

    ui = np.clip((size - 1) * (0.5 + 0.5 * u / half_u), 0, size - 1).round().astype(np.int32)
    vi = np.clip((size - 1) * (0.5 - 0.5 * v / half_v), 0, size - 1).round().astype(np.int32)

    img = np.zeros((size, size, 3), dtype=np.float64)
    img[vi, ui] = cols

    gu, gv, _ = _project_rear_top_native(
        np.array([[center_x, ground_y, center_z]], dtype=np.float64),
        center_xz=center_xz,
        ground_y=ground_y,
        oblique_y_weight=oblique_y_weight,
    )
    rig_u = float(np.clip((size - 1) * (0.5 + 0.5 * gu[0] / half_u), 0, size - 1))
    rig_v = float(np.clip((size - 1) * (0.5 - 0.5 * gv[0] / half_v), 0, size - 1))

    return img, radius, (rig_u, rig_v)


def save_bev_png(
    pcd,
    out_path: Path | str,
    *,
    center_xz: tuple[float, float] | None = None,
    car_png: Path | str | None = None,
    image_size: int = BEV_IMAGE_SIZE,
    extent_percentile: float = BEV_EXTENT_PERCENTILE,
    margin_fraction: float = BEV_MARGIN_FRACTION,
    car_length_fraction: float = CAR_LENGTH_FRACTION,
    oblique_y_weight: float = REAR_TOP_OBLIQUE_Y_WEIGHT,
) -> Path:
    """Render rear-top view + car.png overlay and save as PNG. Works headless."""
    from PIL import Image

    pts = np.asarray(pcd.points)
    if pcd.has_colors():
        cols = np.asarray(pcd.colors)
    else:
        cols = np.full((len(pts), 3), 0.6)

    if center_xz is None:
        center_xz = estimate_scene_center_xz(pts)

    img, radius, (rig_u, rig_v) = render_rear_top_bev(
        pts,
        cols,
        center_xz=center_xz,
        image_size=image_size,
        extent_percentile=extent_percentile,
        margin_fraction=margin_fraction,
        oblique_y_weight=oblique_y_weight,
    )

    car_path = Path(car_png) if car_png is not None else DEFAULT_CAR_PNG
    if car_path.is_file():
        car_len = max(float(car_length_fraction), 0.01) * image_size
        img = _overlay_car_icon(
            img,
            car_path,
            center_u=rig_u,
            center_v=rig_v,
            car_length_px=car_len,
        )
        car_msg = f" + car {car_path.name}"
    else:
        car_msg = f" (car PNG not found: {car_path})"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)).save(out_path)
    print(
        f"Rear-top BEV (+Z up, -Y cam, r={radius:.3f}, center={center_xz}){car_msg} "
        f"-> {out_path}"
    )
    return out_path
