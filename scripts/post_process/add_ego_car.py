"""Open3D rear-top BEV render and ego-car placement (GLB mesh or PNG overlay).

Coordinate conventions (VGGT / OpenCV native, no axis flip):
- +Z: forward (into the scene)
- +X: right
- +Y: down (sky is -Y); the camera sits at (-Y, -Z), behind and above the ego.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

POST_PROCESS_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = POST_PROCESS_DIR.parent
DEFAULT_CAR_PNG = SCRIPT_DIR / "car.png"
DEFAULT_CAR_GLB = POST_PROCESS_DIR / "car_glb.glb"

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
LOOK_AHEAD_FRAC = 0.0           # look-at offset ahead (toward +Z), x scene radius; 0 = scene center
LOOK_DOWN = 0.0                 # extra downward offset of the look-at target (toward +Y)
FIELD_OF_VIEW = 40.0            # vertical field of view in degrees
CENTER_VIEW_ITERS = 10          # iterations to nudge look-at so points land in image center

# Car icon overlay (BEV PNG fallback when GLB is not merged into the PLY).
CAR_LENGTH_FRACTION = 0.09
CAR_OFFSET_X = 0.35       # shift car icon along world +X (meters)

# GLB ego car defaults (merged into output PLY).
DEFAULT_CAR_LENGTH_M = 5
DEFAULT_CAR_SCALE = 0.5          # extra uniform scale after length normalization
DEFAULT_CAR_YAW_DEG = 0.0
DEFAULT_CAR_PITCH_DEG = 180.0
DEFAULT_CAR_ROLL_DEG = 0.0
DEFAULT_CAR_SAMPLE_SPACING = 0.15


@dataclass
class CarGlbConfig:
    """Placement of ``car_glb.glb`` merged into a scene point cloud."""

    glb_path: Path = DEFAULT_CAR_GLB
    enabled: bool = True
    length_m: float = DEFAULT_CAR_LENGTH_M
    scale: float | None = DEFAULT_CAR_SCALE
    yaw_deg: float = DEFAULT_CAR_YAW_DEG
    pitch_deg: float = DEFAULT_CAR_PITCH_DEG
    roll_deg: float = DEFAULT_CAR_ROLL_DEG
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0
    sample_spacing: float = DEFAULT_CAR_SAMPLE_SPACING
    anchor_center_xz: tuple[float, float] | None = None
    anchor_ground_y: float | None = None


def _rot_ypr_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler(
        "YXZ",
        [float(yaw_deg), float(pitch_deg), float(roll_deg)],
        degrees=True,
    ).as_matrix()


def load_car_glb_mesh(glb_path: Path):
    """Load GLB and return one concatenated mesh in model space."""
    import trimesh

    glb_path = Path(glb_path)
    if not glb_path.is_file():
        raise FileNotFoundError(f"Car GLB not found: {glb_path}")

    loaded = trimesh.load(str(glb_path), force="scene")
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    if mesh is None or len(mesh.vertices) == 0:
        raise RuntimeError(f"No mesh geometry in {glb_path}")
    return mesh


def _mesh_vertex_colors(mesh) -> np.ndarray:
    visual = getattr(mesh, "visual", None)
    vcols = getattr(visual, "vertex_colors", None) if visual is not None else None
    if vcols is not None and len(vcols) == len(mesh.vertices):
        return np.clip(np.asarray(vcols, dtype=np.float64)[:, :3] / 255.0, 0.0, 1.0)
    return np.full((len(mesh.vertices), 3), 0.55, dtype=np.float64)


def normalize_car_mesh_to_ground(mesh, *, length_m: float, extra_scale: float | None = None):
    """Center XZ at origin, put tire contact at y=0, uniform scale to target length."""
    import trimesh

    mesh = mesh.copy()
    bounds = mesh.bounds.astype(np.float64)
    center_x = 0.5 * (bounds[0, 0] + bounds[1, 0])
    center_z = 0.5 * (bounds[0, 2] + bounds[1, 2])
    bottom_y = float(bounds[1, 1])

    mesh.vertices[:, 0] -= center_x
    mesh.vertices[:, 2] -= center_z
    mesh.vertices[:, 1] -= bottom_y

    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    # GLB forward axis may be X or Z; use longest ground-plane extent as "length".
    length_extent = max(float(extents[0]), float(extents[2]), 1e-6)
    scale = float(length_m) / length_extent
    if extra_scale is not None and float(extra_scale) > 0.0:
        scale *= float(extra_scale)
    mesh.apply_scale(scale)
    return mesh


def effective_car_sample_spacing(length_m: float, sample_spacing: float) -> float:
    """Scale voxel spacing with car length so small cars stay proportionally sampled."""
    ref_len = max(float(DEFAULT_CAR_LENGTH_M), 1e-3)
    scaled = float(sample_spacing) * (float(length_m) / ref_len)
    return float(np.clip(scaled, 0.002, float(sample_spacing)))


def sample_car_mesh_points(
    mesh,
    *,
    sample_spacing: float,
    length_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Voxel-downsample mesh vertices for a dense colored car point cloud."""
    import open3d as o3d

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    cols = _mesh_vertex_colors(mesh)
    if length_m is not None:
        sample_spacing = effective_car_sample_spacing(length_m, sample_spacing)
    spacing = max(float(sample_spacing), 1e-4)

    car_pcd = o3d.geometry.PointCloud()
    car_pcd.points = o3d.utility.Vector3dVector(verts)
    car_pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0.0, 1.0))
    car_pcd = car_pcd.voxel_down_sample(spacing)
    pts = np.asarray(car_pcd.points, dtype=np.float64)
    out_cols = np.asarray(car_pcd.colors, dtype=np.float64)
    if pts.shape[0] == 0:
        raise RuntimeError("Car mesh sampling produced zero points")
    return pts, out_cols


def car_glb_points_in_world(
    config: CarGlbConfig,
    *,
    center_xz: tuple[float, float],
    ground_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Load, normalize, rotate, and place car GLB points in world coordinates."""
    mesh = load_car_glb_mesh(config.glb_path)
    mesh = normalize_car_mesh_to_ground(
        mesh,
        length_m=config.length_m,
        extra_scale=config.scale,
    )
    # Voxel spacing must follow the *actual* scaled size, otherwise a small
    # ``scale`` collapses the car to a handful of points.
    scale_factor = config.scale if (config.scale and config.scale > 0.0) else 1.0
    effective_length_m = config.length_m * scale_factor
    pts, cols = sample_car_mesh_points(
        mesh,
        sample_spacing=config.sample_spacing,
        length_m=effective_length_m,
    )

    rot = _rot_ypr_matrix(config.yaw_deg, config.pitch_deg, config.roll_deg)
    pts = pts @ rot.T

    center_x, center_z = center_xz
    offset = np.array(
        [config.offset_x, config.offset_y, config.offset_z],
        dtype=np.float64,
    )
    anchor = np.array([center_x, float(ground_y), center_z], dtype=np.float64) + offset
    pts = pts + anchor
    return pts, cols


def merge_car_glb_into_pcd(pcd, config: CarGlbConfig | None = None):
    """Append transformed ``car_glb.glb`` points into an Open3D point cloud."""
    import open3d as o3d

    cfg = config or CarGlbConfig()
    if not cfg.enabled:
        return pcd

    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.shape[0] == 0:
        raise RuntimeError("Cannot place car on an empty point cloud")

    center_xz = cfg.anchor_center_xz or estimate_scene_center_xz(pts)
    ground_y = (
        float(cfg.anchor_ground_y)
        if cfg.anchor_ground_y is not None
        else estimate_ground_y(pts, center_xz)
    )

    car_pts, car_cols = car_glb_points_in_world(cfg, center_xz=center_xz, ground_y=ground_y)

    if pcd.has_colors():
        base_cols = np.asarray(pcd.colors, dtype=np.float64)
    else:
        base_cols = np.full((pts.shape[0], 3), 0.6, dtype=np.float64)

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(np.vstack([pts, car_pts]))
    out.colors = o3d.utility.Vector3dVector(np.vstack([base_cols, car_cols]))
    car_ext = car_pts.max(axis=0) - car_pts.min(axis=0)
    print(
        f"Merged car GLB ({cfg.glb_path.name}): +{car_pts.shape[0]} pts at "
        f"XZ=({center_xz[0]:.3f}, {center_xz[1]:.3f}), y={ground_y:.3f}, "
        f"len_target={cfg.length_m:.3f}m, scale={cfg.scale}, "
        f"bbox=({car_ext[0]:.3f}, {car_ext[1]:.3f}, {car_ext[2]:.3f})m, "
        f"yaw={cfg.yaw_deg:.1f}°, offset=({cfg.offset_x:.2f}, {cfg.offset_y:.2f}, {cfg.offset_z:.2f})"
    )
    return out


def car_glb_config_from_namespace(args) -> CarGlbConfig:
    """Build ``CarGlbConfig`` from argparse / post-process namespace."""
    glb = getattr(args, "car_glb", None) or getattr(args, "bev_car_glb", None)
    return CarGlbConfig(
        glb_path=Path(glb) if glb else DEFAULT_CAR_GLB,
        enabled=bool(getattr(args, "add_car_glb", True)),
        length_m=float(getattr(args, "car_length_m", DEFAULT_CAR_LENGTH_M)),
        scale=getattr(args, "car_scale", DEFAULT_CAR_SCALE),
        yaw_deg=float(getattr(args, "car_yaw_deg", DEFAULT_CAR_YAW_DEG)),
        pitch_deg=float(getattr(args, "car_pitch_deg", DEFAULT_CAR_PITCH_DEG)),
        roll_deg=float(getattr(args, "car_roll_deg", DEFAULT_CAR_ROLL_DEG)),
        offset_x=float(getattr(args, "car_offset_x", 0.0)),
        offset_y=float(getattr(args, "car_offset_y", 0.0)),
        offset_z=float(getattr(args, "car_offset_z", 0.0)),
        sample_spacing=float(getattr(args, "car_sample_spacing", DEFAULT_CAR_SAMPLE_SPACING)),
    )


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


def _project_points(
    points: np.ndarray,
    view: np.ndarray,
    proj: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch world points -> pixel (u, v) and clip w."""
    n = points.shape[0]
    hom = np.concatenate([points, np.ones((n, 1), dtype=np.float64)], axis=1)
    clip = (proj @ (view @ hom.T)).T
    w = clip[:, 3]
    valid = np.abs(w) > 1e-9
    ndc = np.empty((n, 3), dtype=np.float64)
    ndc[valid] = clip[valid, :3] / w[valid, None]
    u = (ndc[:, 0] * 0.5 + 0.5) * width
    v = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height
    return u, v, w


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


def _center_camera_target_on_points(
    pts: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    *,
    renderer,
    field_of_view: float,
    width: int,
    height: int,
    n_iters: int = CENTER_VIEW_ITERS,
) -> np.ndarray:
    """Nudge look-at so the projected point cloud sits near the image center."""
    target = target.copy()
    dist = float(np.linalg.norm(target - eye))
    step = max(dist * np.tan(np.radians(field_of_view) * 0.5) * 0.15, 1e-3)

    sample_n = min(int(pts.shape[0]), 12000)
    if pts.shape[0] > sample_n:
        idx = np.random.default_rng(0).choice(pts.shape[0], sample_n, replace=False)
        sample = pts[idx]
    else:
        sample = pts

    for _ in range(max(int(n_iters), 0)):
        renderer.setup_camera(float(field_of_view), target, eye, up)
        view = np.asarray(renderer.scene.camera.get_view_matrix(), dtype=np.float64)
        proj = np.asarray(renderer.scene.camera.get_projection_matrix(), dtype=np.float64)
        u, v, w = _project_points(sample, view, proj, width, height)
        on_screen = (w > 0.05) & (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
        if not np.any(on_screen):
            break
        u_lo, u_hi = np.percentile(u[on_screen], [10.0, 90.0])
        v_lo, v_hi = np.percentile(v[on_screen], [10.0, 90.0])
        u_ctr = 0.5 * (float(u_lo) + float(u_hi))
        v_ctr = 0.5 * (float(v_lo) + float(v_hi))
        du = u_ctr - width * 0.5
        dv = v_ctr - height * 0.5
        if abs(du) < 4.0 and abs(dv) < 4.0:
            break

        forward = target - eye
        forward /= np.linalg.norm(forward) + 1e-12
        right = np.cross(forward, up)
        right /= np.linalg.norm(right) + 1e-12
        up_cam = np.cross(right, forward)
        up_cam /= np.linalg.norm(up_cam) + 1e-12

        # Shift look-at so the on-screen bbox moves toward the image center.
        target = target + right * (du / width) * step * 4.0 - up_cam * (dv / height) * step * 4.0

    return target


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
    target = _center_camera_target_on_points(
        pts,
        eye,
        target,
        up,
        renderer=renderer,
        field_of_view=field_of_view,
        width=size,
        height=size,
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
    overlay_car_png: bool = True,
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

    car_msg = ""
    if overlay_car_png:
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
            car_msg = f" + car PNG {car_path.name}"
        else:
            car_msg = f" (car PNG not found: {car_path})"
    else:
        car_msg = " (car already in PLY; PNG overlay skipped)"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)).save(out_path)
    print(
        f"Open3D rear-top (cam_back_frac={cam_back_frac}, cam_height_frac={cam_height_frac}, "
        f"look_ahead_frac={look_ahead_frac}, fov={field_of_view}){car_msg} -> {out_path}"
    )
    return out_path
