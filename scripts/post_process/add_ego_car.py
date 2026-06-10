"""Open3D rear-top perspective render of a point cloud with a car.png overlay.

Instead of a hand-rolled projection, this uses Open3D's offscreen (EGL headless)
renderer so the result looks exactly like an interactive Open3D viewport placed
behind and above the ego, looking forward down the road.

Coordinate conventions (VGGT / OpenCV native, no axis flip):
- +Z: forward (into the scene)
- +X: right
- +Y: down (sky is -Y); the camera sits at (-Y, -Z), behind and above the ego.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CAR_PNG = SCRIPT_DIR / "car.png"

# Render defaults (edit here).
IMAGE_SIZE = 1024
POINT_SIZE = 2.0
BACKGROUND_RGBA = (0.0, 0.0, 0.0, 1.0)   # black
CROP_PERCENTILE = 99.5                    # drop far flyers before framing the view

# Perspective camera behind & above the ego, sized relative to the scene radius
# (so the framing adapts to small ROIs and large scenes alike).
SCENE_RADIUS_PERCENTILE = 98.0  # radius used as the framing reference
CAM_BACK_FRAC = 0.5             # camera offset behind ego (toward -Z), x scene radius
CAM_HEIGHT_FRAC = 0.20          # camera height above ground (toward -Y), x scene radius
LOOK_AHEAD_FRAC = 0.80          # look-at distance ahead (toward +Z), x scene radius
LOOK_DOWN = 0.0                 # extra downward offset of the look-at target (toward +Y)
FIELD_OF_VIEW = 40.0            # vertical field of view in degrees

# Car icon overlay.
CAR_LENGTH_FRACTION = 0.09
CAR_OFFSET_X = 0.35       # shift car icon along world +X (meters)


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


def _camera_eye_target(
    center_xz: tuple[float, float],
    ground_y: float,
    scene_radius: float,
    *,
    cam_height_frac: float,
    cam_back_frac: float,
    look_ahead_frac: float,
    look_down: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eye behind & above ego, target ahead on the ground; up = -Y (sky).

    Distances scale with ``scene_radius`` so the framing adapts to scene size.
    """
    center_x, center_z = center_xz
    r = max(float(scene_radius), 1e-3)
    eye = np.array(
        [center_x, ground_y - abs(cam_height_frac) * r, center_z - abs(cam_back_frac) * r],
        dtype=np.float64,
    )
    target = np.array(
        [center_x, ground_y + look_down, center_z + abs(look_ahead_frac) * r],
        dtype=np.float64,
    )
    up = np.array([0.0, -1.0, 0.0], dtype=np.float64)  # sky = -Y
    return eye, target, up


def _project_point(
    point: np.ndarray,
    view: np.ndarray,
    proj: np.ndarray,
    width: int,
    height: int,
) -> tuple[float, float, float]:
    """World point -> pixel (u, v) and clip-space depth via Open3D matrices."""
    hom = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
    clip = proj @ (view @ hom)
    w = clip[3]
    if abs(w) < 1e-9:
        return float("nan"), float("nan"), float("nan")
    ndc = clip[:3] / w
    u = (ndc[0] * 0.5 + 0.5) * width
    v = (1.0 - (ndc[1] * 0.5 + 0.5)) * height
    return float(u), float(v), float(w)


def render_open3d_rear_top(
    pts: np.ndarray,
    cols: np.ndarray,
    *,
    center_xz: tuple[float, float] | None = None,
    ground_y: float | None = None,
    image_size: int = IMAGE_SIZE,
    point_size: float = POINT_SIZE,
    crop_percentile: float = CROP_PERCENTILE,
    scene_radius_percentile: float = SCENE_RADIUS_PERCENTILE,
    cam_height_frac: float = CAM_HEIGHT_FRAC,
    cam_back_frac: float = CAM_BACK_FRAC,
    look_ahead_frac: float = LOOK_AHEAD_FRAC,
    look_down: float = LOOK_DOWN,
    field_of_view: float = FIELD_OF_VIEW,
    car_offset_x: float = CAR_OFFSET_X,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Offscreen Open3D render from behind & above the ego.

    Returns the rendered RGB image (float in [0, 1]) and the projected ego
    pixel (u, v) used to place the car icon.
    """
    import open3d as o3d
    from open3d.visualization import rendering

    pts = np.asarray(pts, dtype=np.float64)
    cols = np.asarray(cols, dtype=np.float64)

    if center_xz is None:
        center_xz = estimate_scene_center_xz(pts)
    if ground_y is None:
        ground_y = estimate_ground_y(pts, center_xz)

    center_x, center_z = center_xz

    # Drop far flyers so the framing stays tight on the scene.
    r = np.hypot(pts[:, 0] - center_x, pts[:, 2] - center_z)
    pct = float(np.clip(crop_percentile, 1.0, 100.0))
    if pct < 100.0 and pts.shape[0] > 0:
        radius = float(np.percentile(r, pct))
        if radius > 0.0:
            keep = r <= radius
            pts = pts[keep]
            cols = cols[keep]
            r = r[keep]
    if pts.shape[0] == 0:
        raise RuntimeError("No points left to render after radial crop")

    scene_radius = float(np.percentile(r, float(np.clip(scene_radius_percentile, 1.0, 100.0))))
    scene_radius = max(scene_radius, 1e-3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0.0, 1.0))

    size = int(image_size)
    renderer = rendering.OffscreenRenderer(size, size)
    renderer.scene.set_background(list(BACKGROUND_RGBA))

    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = float(point_size)
    renderer.scene.add_geometry("points", pcd, mat)

    eye, target, up = _camera_eye_target(
        center_xz,
        ground_y,
        scene_radius,
        cam_height_frac=cam_height_frac,
        cam_back_frac=cam_back_frac,
        look_ahead_frac=look_ahead_frac,
        look_down=look_down,
    )
    renderer.setup_camera(float(field_of_view), target, eye, up)

    img = np.asarray(renderer.render_to_image(), dtype=np.float64) / 255.0

    view = np.asarray(renderer.scene.camera.get_view_matrix(), dtype=np.float64)
    proj = np.asarray(renderer.scene.camera.get_projection_matrix(), dtype=np.float64)
    car_world = np.array([center_x + car_offset_x, ground_y, center_z], dtype=np.float64)
    rig_u, rig_v, _ = _project_point(car_world, view, proj, size, size)
    if not np.isfinite(rig_u):
        rig_u, rig_v = size * 0.5, size * 0.7

    return img, (rig_u, rig_v)


def save_bev_png(
    pcd,
    out_path: Path | str,
    *,
    center_xz: tuple[float, float] | None = None,
    car_png: Path | str | None = None,
    image_size: int = IMAGE_SIZE,
    point_size: float = POINT_SIZE,
    crop_percentile: float = CROP_PERCENTILE,
    scene_radius_percentile: float = SCENE_RADIUS_PERCENTILE,
    car_length_fraction: float = CAR_LENGTH_FRACTION,
    cam_height_frac: float = CAM_HEIGHT_FRAC,
    cam_back_frac: float = CAM_BACK_FRAC,
    look_ahead_frac: float = LOOK_AHEAD_FRAC,
    look_down: float = LOOK_DOWN,
    field_of_view: float = FIELD_OF_VIEW,
    car_offset_x: float = CAR_OFFSET_X,
    **_ignored,
) -> Path:
    """Render an Open3D rear-top view + car.png overlay (headless via EGL).

    Extra keyword arguments (e.g. legacy ``extent_percentile``) are accepted and
    ignored so existing callers keep working.
    """
    from PIL import Image

    pts = np.asarray(pcd.points)
    if pcd.has_colors():
        cols = np.asarray(pcd.colors)
    else:
        cols = np.full((len(pts), 3), 0.6)

    if center_xz is None:
        center_xz = estimate_scene_center_xz(pts)

    img, (rig_u, rig_v) = render_open3d_rear_top(
        pts,
        cols,
        center_xz=center_xz,
        image_size=image_size,
        point_size=point_size,
        crop_percentile=crop_percentile,
        scene_radius_percentile=scene_radius_percentile,
        cam_height_frac=cam_height_frac,
        cam_back_frac=cam_back_frac,
        look_ahead_frac=look_ahead_frac,
        look_down=look_down,
        field_of_view=field_of_view,
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
        f"Open3D rear-top (cam_back_frac={cam_back_frac}, cam_height_frac={cam_height_frac}, "
        f"look_ahead_frac={look_ahead_frac}, fov={field_of_view}){car_msg} -> {out_path}"
    )
    return out_path
