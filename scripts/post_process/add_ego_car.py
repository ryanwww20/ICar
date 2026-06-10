"""Rear-top perspective export with car.png overlay.

Coordinate conventions (VGGT / OpenCV native, no axis flip):
- +Z: forward (into the scene)
- +X: right
- +Y: down (sky is -Y); camera sits at (-Y, -Z) behind and above ego.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CAR_PNG = SCRIPT_DIR / "car.png"

# Render defaults (edit here).
BEV_IMAGE_SIZE = 1024
BEV_EXTENT_PERCENTILE = 92.0
BEV_MARGIN_FRACTION = 0.12
CAR_LENGTH_FRACTION = 0.09
CAR_OFFSET_X = 0.35           # shift car icon along world +X (meters)

# Perspective camera behind & above ego (native Y-down frame).
REAR_TOP_CAM_Y = 2          # height along -Y (larger = higher above scene)
REAR_TOP_CAM_BACK_Z = 15.0      # offset along -Z behind ego (larger = further back)
REAR_TOP_LOOK_AHEAD_Z = 7.0     # look-at distance along +Z
REAR_TOP_LOOK_DOWN_Y = 0.4      # tilt view down toward ground (+Y in Y-down)
REAR_TOP_VIEW_SCALE = 1.3      # zoom-out on projected scene (>1 = wider)


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
    """Paste RGBA car icon; PNG front points toward +Z (image up)."""
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


def _rear_top_camera_basis(
    center_xz: tuple[float, float],
    ground_y: float,
    *,
    cam_y: float = REAR_TOP_CAM_Y,
    cam_back_z: float = REAR_TOP_CAM_BACK_Z,
    look_ahead_z: float = REAR_TOP_LOOK_AHEAD_Z,
    look_down_y: float = REAR_TOP_LOOK_DOWN_Y,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Camera at (-Y, -Z) behind ego, looking toward road ahead (+Z, +Y down)."""
    center_x, center_z = center_xz
    eye = np.array([center_x, -abs(cam_y), center_z - abs(cam_back_z)], dtype=np.float64)
    target = np.array(
        [center_x, ground_y + abs(look_down_y), center_z + abs(look_ahead_z)],
        dtype=np.float64,
    )
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-12

    world_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)  # sky = -Y
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1e-12
    up_v = np.cross(right, forward)
    up_v /= np.linalg.norm(up_v) + 1e-12
    return eye, target, right, up_v, forward


def _project_perspective(
    pts: np.ndarray,
    *,
    center_xz: tuple[float, float],
    ground_y: float,
    cam_y: float = REAR_TOP_CAM_Y,
    cam_back_z: float = REAR_TOP_CAM_BACK_Z,
    look_ahead_z: float = REAR_TOP_LOOK_AHEAD_Z,
    look_down_y: float = REAR_TOP_LOOK_DOWN_Y,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perspective projection: x/z and y/z in camera space."""
    eye, _, right, up_v, forward = _rear_top_camera_basis(
        center_xz,
        ground_y,
        cam_y=cam_y,
        cam_back_z=cam_back_z,
        look_ahead_z=look_ahead_z,
        look_down_y=look_down_y,
    )
    rel = pts - eye
    x_cam = rel @ right
    y_cam = rel @ up_v
    z_cam = rel @ forward
    eps = 1e-4
    u = x_cam / np.maximum(z_cam, eps)
    v = y_cam / np.maximum(z_cam, eps)
    return u, v, z_cam


def render_rear_top_bev(
    pts: np.ndarray,
    cols: np.ndarray,
    *,
    center_xz: tuple[float, float] = (0.0, 0.0),
    ground_y: float | None = None,
    image_size: int = BEV_IMAGE_SIZE,
    extent_percentile: float = BEV_EXTENT_PERCENTILE,
    margin_fraction: float = BEV_MARGIN_FRACTION,
    cam_y: float = REAR_TOP_CAM_Y,
    cam_back_z: float = REAR_TOP_CAM_BACK_Z,
    look_ahead_z: float = REAR_TOP_LOOK_AHEAD_Z,
    look_down_y: float = REAR_TOP_LOOK_DOWN_Y,
    view_scale: float = REAR_TOP_VIEW_SCALE,
    car_offset_x: float = CAR_OFFSET_X,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """
    Perspective rear-above view (like Open3D behind-and-above the ego).

    Unlike orthographic BEV, distant points shrink with depth so the road,
    buildings, and sides appear with natural 3D layout.
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

    u, v, depth = _project_perspective(
        pts,
        center_xz=center_xz,
        ground_y=ground_y,
        cam_y=cam_y,
        cam_back_z=cam_back_z,
        look_ahead_z=look_ahead_z,
        look_down_y=look_down_y,
    )

    visible = depth > 0.05
    u, v, cols, depth = u[visible], v[visible], cols[visible], depth[visible]
    if u.size == 0:
        raise RuntimeError("No points in front of the rear-top camera")

    zoom = max(float(view_scale), 0.1)
    u_center = float(np.median(u))
    v_center = float(np.median(v))
    half_u = max(float(np.percentile(np.abs(u - u_center), pct)), 1e-6) * (1.0 + float(margin_fraction)) * zoom
    half_v = max(float(np.percentile(np.abs(v - v_center), pct)), 1e-6) * (1.0 + float(margin_fraction)) * zoom
    size = int(image_size)

    order = np.argsort(depth, kind="stable")
    u, v, cols = u[order], v[order], cols[order]

    ui = np.clip((size - 1) * (0.5 + 0.5 * (u - u_center) / half_u), 0, size - 1).round().astype(np.int32)
    vi = np.clip((size - 1) * (0.5 - 0.5 * (v - v_center) / half_v), 0, size - 1).round().astype(np.int32)

    img = np.zeros((size, size, 3), dtype=np.float64)
    img[vi, ui] = cols

    car_x = center_x + car_offset_x
    gu, gv, gz = _project_perspective(
        np.array([[car_x, ground_y, center_z]], dtype=np.float64),
        center_xz=center_xz,
        ground_y=ground_y,
        cam_y=cam_y,
        cam_back_z=cam_back_z,
        look_ahead_z=look_ahead_z,
        look_down_y=look_down_y,
    )
    if gz[0] > 0.05:
        rig_u = float(np.clip((size - 1) * (0.5 + 0.5 * (gu[0] - u_center) / half_u), 0, size - 1))
        rig_v = float(np.clip((size - 1) * (0.5 - 0.5 * (gv[0] - v_center) / half_v), 0, size - 1))
    else:
        rig_u = rig_v = (size - 1) * 0.5

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
    cam_y: float = REAR_TOP_CAM_Y,
    cam_back_z: float = REAR_TOP_CAM_BACK_Z,
    look_ahead_z: float = REAR_TOP_LOOK_AHEAD_Z,
    look_down_y: float = REAR_TOP_LOOK_DOWN_Y,
    view_scale: float = REAR_TOP_VIEW_SCALE,
    car_offset_x: float = CAR_OFFSET_X,
) -> Path:
    """Render perspective rear-top view + car.png overlay. Works headless."""
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
        cam_y=cam_y,
        cam_back_z=cam_back_z,
        look_ahead_z=look_ahead_z,
        look_down_y=look_down_y,
        view_scale=view_scale,
        car_offset_x=car_offset_x,
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
        f"Perspective rear-top (cam_y={cam_y}, cam_back_z={cam_back_z}, "
        f"look_z={look_ahead_z}, scale={view_scale}, r={radius:.3f}){car_msg} -> {out_path}"
    )
    return out_path
