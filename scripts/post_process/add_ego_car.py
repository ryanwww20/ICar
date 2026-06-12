"""Open3D rear-top BEV render and ego-car placement (GLB mesh or PNG overlay).

Coordinate conventions (VGGT / OpenCV native, no axis flip):
- +Z: forward (into the scene)
- +X: right
- +Y: down (sky is -Y); the camera sits at (-Y, -Z), behind and above the ego.
"""

from __future__ import annotations

import ctypes.util
import io
import json
import os
import struct
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

POST_PROCESS_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = POST_PROCESS_DIR.parent
DEFAULT_CAR_PNG = SCRIPT_DIR / "car.png"
DEFAULT_CAR_GLB = POST_PROCESS_DIR / "car_glb.glb"
DEFAULT_CAR_PLY = POST_PROCESS_DIR / "car_glb.ply"

# Render defaults (edit here).
IMAGE_SIZE = 1024
POINT_SIZE = 2.0
BACKGROUND_RGBA = (0.0, 0.0, 0.0, 1.0)   # black (Open3D EGL path)
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
DEFAULT_CAR_SCALE = 0.07          # extra uniform scale after length normalization
DEFAULT_CAR_YAW_DEG = 0.0
DEFAULT_CAR_PITCH_DEG = 0
DEFAULT_CAR_ROLL_DEG = 0.0
DEFAULT_CAR_MIRROR_XZ = True          # mirror across XZ plane (Y -> -Y)
DEFAULT_CAR_OFFSET_X = 0.2
DEFAULT_CAR_OFFSET_Y = -0.15
DEFAULT_CAR_OFFSET_Z = 0.0
DEFAULT_CAR_SAMPLE_SPACING = 0.03   # dense surface sampling (~50k–80k pts at 5 m)

# GLB sub-meshes to drop (showroom floor, etc.).
CAR_GEOM_EXCLUDE_SUBSTRINGS = ("Plane.035",)
CAR_MATERIAL_EXCLUDE_SUBSTRINGS = ("Plane.035",)
# Drop large showroom quads even when the node/material name differs.
CAR_SHOWROOM_FLOOR_MIN_AREA = 3.0

# Pure-numpy BEV fallback (no EGL / Open3D OffscreenRenderer).
NUMPY_BEV_BACKGROUND_RGBA = (1.0, 1.0, 1.0, 1.0)  # white
MAX_BEV_RENDER_POINTS = 1_500_000
NUMPY_BEV_POINT_SIZE = 3.0


@dataclass
class GlbTextureLibrary:
    """Embedded glTF images keyed by material name (works across trimesh versions)."""

    images: list[np.ndarray] = field(default_factory=list)
    material_to_image: dict[str, int] = field(default_factory=dict)

    @classmethod
    @lru_cache(maxsize=4)
    def from_glb(cls, glb_path: Path | str) -> GlbTextureLibrary:
        glb_path = Path(glb_path)
        gltf, bin_data = _parse_glb_chunks(glb_path)
        images = _decode_glb_images(gltf, bin_data)
        material_to_image: dict[str, int] = {}
        textures = gltf.get("textures", [])
        for mat in gltf.get("materials", []):
            mat_name = mat.get("name")
            if not mat_name:
                continue
            pbr = mat.get("pbrMetallicRoughness", {})
            bct = pbr.get("baseColorTexture")
            if not bct or "index" not in bct:
                continue
            tex_idx = int(bct["index"])
            img_idx = int(textures[tex_idx]["source"])
            material_to_image[str(mat_name)] = img_idx
        return cls(images=images, material_to_image=material_to_image)

    def texture_for_material(self, material_name: str | None) -> np.ndarray | None:
        if not material_name:
            return None
        img_idx = self.material_to_image.get(material_name)
        if img_idx is None or img_idx >= len(self.images):
            return None
        return self.images[img_idx]


def _parse_glb_chunks(glb_path: Path) -> tuple[dict, bytes]:
    data = glb_path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError(f"Not a GLB file: {glb_path}")
    json_chunk_len = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20 : 20 + json_chunk_len])
    offset = 20 + json_chunk_len
    while offset % 4 != 0:
        offset += 1
    if offset + 8 > len(data):
        raise ValueError(f"GLB binary chunk missing in {glb_path}")
    bin_chunk_len = struct.unpack("<I", data[offset : offset + 4])[0]
    bin_data = data[offset + 8 : offset + 8 + bin_chunk_len]
    return gltf, bin_data


def _decode_image_bytes(raw: bytes) -> np.ndarray:
    try:
        from PIL import Image

        return _texture_to_rgb_uint8(Image.open(io.BytesIO(raw)))
    except ImportError:
        pass

    import cv2

    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError("Failed to decode GLB embedded image")
    if arr.ndim == 2:
        rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.shape[2] == 4:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    elif arr.shape[2] == 3:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported decoded image shape: {arr.shape}")
    return rgb.astype(np.uint8)


def _decode_glb_images(gltf: dict, bin_data: bytes) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for img_info in gltf.get("images", []):
        bv = gltf["bufferViews"][img_info["bufferView"]]
        start = int(bv.get("byteOffset", 0))
        end = start + int(bv["byteLength"])
        images.append(_decode_image_bytes(bin_data[start:end]))
    return images


def _texture_to_rgb_uint8(source) -> np.ndarray:
    """Normalize PIL / numpy textures to HxWx3 uint8 RGB."""
    try:
        from PIL import Image

        if isinstance(source, Image.Image):
            return np.asarray(source.convert("RGB"), dtype=np.uint8)
    except ImportError:
        pass

    arr = np.asarray(source)
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1).astype(np.uint8)
    if arr.shape[-1] == 1:
        rgb = np.repeat(arr, 3, axis=-1)
        return rgb.astype(np.uint8)
    if arr.shape[-1] == 2:
        lum = arr[..., 0]
        return np.stack([lum, lum, lum], axis=-1).astype(np.uint8)
    if arr.shape[-1] >= 3:
        return arr[..., :3].astype(np.uint8)
    raise ValueError(f"Unsupported texture shape: {arr.shape}")


def _resolve_material_texture_rgb(material, texture_lib: GlbTextureLibrary | None) -> np.ndarray | None:
    if material is None:
        return None
    tex = getattr(material, "baseColorTexture", None)
    if tex is not None:
        try:
            return _texture_to_rgb_uint8(tex)
        except (TypeError, ValueError):
            pass
    if texture_lib is not None:
        mat_name = getattr(material, "name", None)
        return texture_lib.texture_for_material(mat_name)
    return None


def _is_showroom_floor_mesh(mesh) -> bool:
    verts = getattr(mesh, "vertices", None)
    faces = getattr(mesh, "faces", None)
    if verts is None or faces is None:
        return False
    if len(verts) > 4 or len(faces) > 2:
        return False
    area = float(getattr(mesh, "area", 0.0) or 0.0)
    return area >= CAR_SHOWROOM_FLOOR_MIN_AREA


def _should_exclude_car_geometry(geom_name: str, mesh=None) -> bool:
    if any(token in geom_name for token in CAR_GEOM_EXCLUDE_SUBSTRINGS):
        return True
    if mesh is not None:
        visual = getattr(mesh, "visual", None)
        mat = getattr(visual, "material", None) if visual is not None else None
        mat_name = getattr(mat, "name", "") or ""
        if any(token in mat_name for token in CAR_MATERIAL_EXCLUDE_SUBSTRINGS):
            return True
        if _is_showroom_floor_mesh(mesh):
            return True
    return False


def _pbr_base_color_rgb(material) -> np.ndarray:
    """Read diffuse/base color from a glTF PBR material (0–1 RGB)."""
    if material is None:
        return np.array([0.55, 0.55, 0.55], dtype=np.float64)
    factor = getattr(material, "baseColorFactor", None)
    if factor is not None:
        rgb = np.asarray(factor, dtype=np.float64).reshape(-1)[:3]
        if rgb.max() > 1.0:
            rgb /= 255.0
        return np.clip(rgb, 0.0, 1.0)
    return np.array([0.55, 0.55, 0.55], dtype=np.float64)


def _name_color_hint(name: str) -> np.ndarray | None:
    """Fallback tint from mesh/material name when PBR factor is neutral grey."""
    hints: tuple[tuple[str, tuple[float, float, float]], ...] = (
        ("red_glass", (0.85, 0.12, 0.06)),
        ("orange_glass", (0.95, 0.45, 0.05)),
        ("headlights", (0.92, 0.92, 0.85)),
        ("license", (0.92, 0.92, 0.88)),
        ("glass", (0.55, 0.72, 0.82)),
        ("chrome", (0.78, 0.78, 0.82)),
        ("alloy", (0.58, 0.58, 0.62)),
        ("tire", (0.07, 0.07, 0.07)),
        ("black_paint", (0.06, 0.06, 0.06)),
        ("black_matte", (0.14, 0.14, 0.14)),
        ("paint", (0.20, 0.11, 0.03)),
        ("coat", (0.07, 0.05, 0.03)),
    )
    lower = name.lower()
    for key, rgb in hints:
        if key in lower:
            return np.array(rgb, dtype=np.float64)
    return None


def _sample_texture_colors(
    mesh,
    pts: np.ndarray,
    face_idx: np.ndarray,
    *,
    texture_lib: GlbTextureLibrary | None = None,
) -> np.ndarray | None:
    """Sample RGB at surface points via UV texture or PBR baseColorFactor."""
    import trimesh

    visual = getattr(mesh, "visual", None)
    if visual is None:
        return None

    if isinstance(visual, trimesh.visual.texture.TextureVisuals):
        mat = visual.material
        img = _resolve_material_texture_rgb(mat, texture_lib)

        if img is not None and visual.uv is not None and len(visual.uv) == len(mesh.vertices):
            faces = mesh.faces[face_idx]
            tri_uv = visual.uv[faces]
            tris = mesh.vertices[faces]
            bc = trimesh.triangles.points_to_barycentric(tris, pts)
            pt_uv = (tri_uv * bc[:, :, None]).sum(axis=1)
            h, w = img.shape[:2]
            px = np.clip((pt_uv[:, 0] * (w - 1)).astype(np.int64), 0, w - 1)
            py = np.clip(((1.0 - pt_uv[:, 1]) * (h - 1)).astype(np.int64), 0, h - 1)
            return np.clip(img[py, px].astype(np.float64) / 255.0, 0.0, 1.0)

        base = _pbr_base_color_rgb(mat)
        return np.tile(base, (len(pts), 1))

    vcols = getattr(visual, "vertex_colors", None)
    if vcols is not None and len(vcols) == len(mesh.vertices):
        faces = mesh.faces[face_idx]
        tris = mesh.vertices[faces]
        bc = trimesh.triangles.points_to_barycentric(tris, pts)
        rgb = np.asarray(vcols[:, :3], dtype=np.float64)
        if rgb.max() > 1.0:
            rgb /= 255.0
        return np.clip((rgb[faces] * bc[:, :, None]).sum(axis=1), 0.0, 1.0)

    return None


def _first_color_hint(*names: str) -> np.ndarray | None:
    for name in names:
        hint = _name_color_hint(name)
        if hint is not None:
            return hint
    return None


def _resolve_mesh_colors(
    mesh,
    geom_name: str,
    pts: np.ndarray,
    face_idx: np.ndarray,
    *,
    texture_lib: GlbTextureLibrary | None = None,
) -> np.ndarray:
    cols = _sample_texture_colors(mesh, pts, face_idx, texture_lib=texture_lib)
    if cols is None:
        mat = getattr(getattr(mesh, "visual", None), "material", None)
        cols = np.tile(_pbr_base_color_rgb(mat), (len(pts), 1))
    mat = getattr(getattr(mesh, "visual", None), "material", None)
    mat_name = getattr(mat, "name", "") or ""
    hint = _first_color_hint(geom_name, mat_name)
    if hint is not None:
        neutral = np.all(np.abs(cols - cols.mean(axis=0, keepdims=True)) < 0.04, axis=1)
        cols[neutral] = hint
    return np.clip(cols, 0.0, 1.0)


@dataclass
class CarGlbConfig:
    """Placement of ``car_glb.glb`` merged into a scene point cloud."""

    glb_path: Path = DEFAULT_CAR_GLB
    ply_path: Path | None = None  # default: DEFAULT_CAR_PLY when present
    enabled: bool = True
    length_m: float = DEFAULT_CAR_LENGTH_M
    scale: float | None = DEFAULT_CAR_SCALE
    yaw_deg: float = DEFAULT_CAR_YAW_DEG
    pitch_deg: float = DEFAULT_CAR_PITCH_DEG
    roll_deg: float = DEFAULT_CAR_ROLL_DEG
    offset_x: float = DEFAULT_CAR_OFFSET_X
    offset_y: float = DEFAULT_CAR_OFFSET_Y
    offset_z: float = DEFAULT_CAR_OFFSET_Z
    mirror_xz: bool = DEFAULT_CAR_MIRROR_XZ
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


def load_car_glb_meshes(
    glb_path: Path,
    *,
    texture_lib: GlbTextureLibrary | None = None,
) -> list[tuple[str, object]]:
    """Load GLB scene graph; return ``[(geom_name, mesh), ...]`` with transforms applied."""
    import trimesh

    glb_path = Path(glb_path)
    if not glb_path.is_file():
        raise FileNotFoundError(f"Car GLB not found: {glb_path}")

    if texture_lib is None:
        texture_lib = GlbTextureLibrary.from_glb(glb_path)

    loaded = trimesh.load(str(glb_path), force="scene")
    if not isinstance(loaded, trimesh.Scene):
        if loaded is None or len(loaded.vertices) == 0:
            raise RuntimeError(f"No mesh geometry in {glb_path}")
        return [("mesh", loaded)]

    meshes: list[tuple[str, object]] = []
    for node_name in loaded.graph.nodes_geometry:
        transform, geom_name = loaded.graph[node_name]
        geom = loaded.geometry[geom_name].copy()
        geom.apply_transform(transform)
        if _should_exclude_car_geometry(geom_name, geom):
            continue
        if len(geom.vertices) == 0 or len(getattr(geom, "faces", [])) == 0:
            continue
        meshes.append((geom_name, geom))

    if not meshes:
        raise RuntimeError(f"No car body geometry left in {glb_path} after exclusions")
    _ = texture_lib  # ensure library is built before sampling colors
    return meshes


def load_car_glb_mesh(glb_path: Path):
    """Load GLB and return one concatenated mesh (floor geometry excluded)."""
    import trimesh

    parts = [mesh for _, mesh in load_car_glb_meshes(glb_path)]
    mesh = trimesh.util.concatenate(parts)
    if mesh is None or len(mesh.vertices) == 0:
        raise RuntimeError(f"No mesh geometry in {glb_path}")
    return mesh


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
    geom_name: str = "mesh",
    texture_lib: GlbTextureLibrary | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample colored points on one mesh surface."""
    import trimesh

    if length_m is not None:
        sample_spacing = effective_car_sample_spacing(length_m, sample_spacing)
    spacing = max(float(sample_spacing), 1e-4)

    area = float(getattr(mesh, "area", 0.0) or 0.0)
    faces = getattr(mesh, "faces", None)
    if area > 0.0 and faces is not None and len(faces) > 0:
        n_points = int(area / (spacing**2))
        n_points = int(np.clip(n_points, 10, 500_000))
        pts, face_idx = trimesh.sample.sample_surface(mesh, n_points)
        pts = np.asarray(pts, dtype=np.float64)
        out_cols = _resolve_mesh_colors(
            mesh,
            geom_name,
            pts,
            face_idx,
            texture_lib=texture_lib,
        )
    else:
        import open3d as o3d

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        mat = getattr(getattr(mesh, "visual", None), "material", None)
        mat_name = getattr(mat, "name", "") or geom_name
        col = _first_color_hint(geom_name, mat_name) or _pbr_base_color_rgb(mat)
        car_pcd = o3d.geometry.PointCloud()
        car_pcd.points = o3d.utility.Vector3dVector(verts)
        car_pcd.colors = o3d.utility.Vector3dVector(np.tile(col, (len(verts), 1)))
        car_pcd = car_pcd.voxel_down_sample(spacing)
        pts = np.asarray(car_pcd.points, dtype=np.float64)
        out_cols = np.asarray(car_pcd.colors, dtype=np.float64)

    if pts.shape[0] == 0:
        raise RuntimeError("Car mesh sampling produced zero points")
    return pts, out_cols


def sample_car_meshes_points(
    meshes: list[tuple[str, object]],
    *,
    sample_spacing: float,
    length_m: float,
    texture_lib: GlbTextureLibrary | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize, densely sample, and merge all car body sub-meshes."""
    import trimesh

    combined_raw = trimesh.util.concatenate([m for _, m in meshes])
    bounds = combined_raw.bounds.astype(np.float64)
    center_x = 0.5 * (bounds[0, 0] + bounds[1, 0])
    center_z = 0.5 * (bounds[0, 2] + bounds[1, 2])
    bottom_y = float(bounds[1, 1])
    extents = bounds[1] - bounds[0]
    length_extent = max(float(extents[0]), float(extents[2]), 1e-6)
    uniform_scale = float(length_m) / length_extent

    all_pts: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    for geom_name, mesh in meshes:
        part = mesh.copy()
        part.vertices[:, 0] -= center_x
        part.vertices[:, 2] -= center_z
        part.vertices[:, 1] -= bottom_y
        part.apply_scale(uniform_scale)
        pts, cols = sample_car_mesh_points(
            part,
            sample_spacing=sample_spacing,
            length_m=length_m,
            geom_name=geom_name,
            texture_lib=texture_lib,
        )
        all_pts.append(pts)
        all_cols.append(cols)

    pts = np.vstack(all_pts)
    cols = np.vstack(all_cols)
    if pts.shape[0] == 0:
        raise RuntimeError("Car mesh sampling produced zero points")
    return pts, cols


def export_car_glb_to_ply(
    out_path: Path | str | None = None,
    *,
    glb_path: Path | str = DEFAULT_CAR_GLB,
    length_m: float = DEFAULT_CAR_LENGTH_M,
    sample_spacing: float = DEFAULT_CAR_SAMPLE_SPACING,
) -> Path:
    """Bake ``car_glb.glb`` into a canonical dense ``car_glb.ply`` (model space)."""
    import open3d as o3d

    out_path = Path(out_path or DEFAULT_CAR_PLY)
    glb_path = Path(glb_path)
    texture_lib = GlbTextureLibrary.from_glb(glb_path)
    meshes = load_car_glb_meshes(glb_path, texture_lib=texture_lib)
    pts, cols = sample_car_meshes_points(
        meshes,
        sample_spacing=sample_spacing,
        length_m=length_m,
        texture_lib=texture_lib,
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0.0, 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False)
    ext = pts.max(axis=0) - pts.min(axis=0)
    print(
        f"Exported car PLY ({len(pts)} pts, spacing={sample_spacing:.3f} m, "
        f"len={length_m:.1f} m, bbox=({ext[0]:.2f}, {ext[1]:.2f}, {ext[2]:.2f}) m) -> {out_path}"
    )
    return out_path


def load_car_ply_canonical(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load canonical car PLY (centered XZ, ground at y=0, DEFAULT_CAR_LENGTH_M)."""
    import open3d as o3d

    ply_path = Path(ply_path)
    if not ply_path.is_file():
        raise FileNotFoundError(f"Car PLY not found: {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.shape[0] == 0:
        raise RuntimeError(f"No points in {ply_path}")
    if pcd.has_colors():
        cols = np.asarray(pcd.colors, dtype=np.float64)
    else:
        cols = np.full((pts.shape[0], 3), 0.55, dtype=np.float64)
    return pts, cols


def _car_size_scale(config: CarGlbConfig) -> float:
    """Scale canonical PLY points to match requested length_m and scale."""
    scale_factor = config.scale if (config.scale and config.scale > 0.0) else 1.0
    return (float(config.length_m) / float(DEFAULT_CAR_LENGTH_M)) * scale_factor


def _resolve_car_ply_path(config: CarGlbConfig) -> Path:
    if config.ply_path is not None:
        return Path(config.ply_path)
    return DEFAULT_CAR_PLY


def _canonical_car_points(config: CarGlbConfig) -> tuple[np.ndarray, np.ndarray, str]:
    """Load dense car points; prefer baked PLY, else export once from GLB."""
    ply_path = _resolve_car_ply_path(config)
    if ply_path.is_file():
        pts, cols = load_car_ply_canonical(ply_path)
        return pts, cols, ply_path.name

    print(f"Car PLY not found ({ply_path}); exporting from {config.glb_path.name} ...")
    export_car_glb_to_ply(
        ply_path,
        glb_path=config.glb_path,
        length_m=DEFAULT_CAR_LENGTH_M,
        sample_spacing=config.sample_spacing,
    )
    pts, cols = load_car_ply_canonical(ply_path)
    return pts, cols, ply_path.name


def car_glb_points_in_world(
    config: CarGlbConfig,
    *,
    center_xz: tuple[float, float],
    ground_y: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load canonical car PLY, scale/rotate/place in world coordinates."""
    pts, cols, source = _canonical_car_points(config)

    size_scale = _car_size_scale(config)
    if abs(size_scale - 1.0) > 1e-9:
        pts = pts * size_scale

    rot = _rot_ypr_matrix(config.yaw_deg, config.pitch_deg, config.roll_deg)
    pts = pts @ rot.T

    if config.mirror_xz:
        pts[:, 1] *= -1.0  # mirror across world XZ plane (Y -> -Y)

    center_x, center_z = center_xz
    offset = np.array(
        [config.offset_x, config.offset_y, config.offset_z],
        dtype=np.float64,
    )
    anchor = np.array([center_x, float(ground_y), center_z], dtype=np.float64) + offset
    pts = pts + anchor
    return pts, cols, source


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

    car_pts, car_cols, car_source = car_glb_points_in_world(cfg, center_xz=center_xz, ground_y=ground_y)

    if pcd.has_colors():
        base_cols = np.asarray(pcd.colors, dtype=np.float64)
    else:
        base_cols = np.full((pts.shape[0], 3), 0.6, dtype=np.float64)

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(np.vstack([pts, car_pts]))
    out.colors = o3d.utility.Vector3dVector(np.vstack([base_cols, car_cols]))
    car_ext = car_pts.max(axis=0) - car_pts.min(axis=0)
    print(
        f"Merged car ({car_source}): +{car_pts.shape[0]} pts at "
        f"XZ=({center_xz[0]:.3f}, {center_xz[1]:.3f}), y={ground_y:.3f}, "
        f"len_target={cfg.length_m:.3f}m, scale={cfg.scale}, "
        f"bbox=({car_ext[0]:.3f}, {car_ext[1]:.3f}, {car_ext[2]:.3f})m, "
        f"yaw={cfg.yaw_deg:.1f}°, offset=({cfg.offset_x:.2f}, {cfg.offset_y:.2f}, {cfg.offset_z:.2f})"
    )
    return out


def car_glb_config_from_namespace(args) -> CarGlbConfig:
    """Build ``CarGlbConfig`` from argparse / post-process namespace."""
    glb = getattr(args, "car_glb", None) or getattr(args, "bev_car_glb", None)
    car_ply = getattr(args, "car_ply", None)
    return CarGlbConfig(
        glb_path=Path(glb) if glb else DEFAULT_CAR_GLB,
        ply_path=Path(car_ply) if car_ply else None,
        enabled=bool(getattr(args, "add_car_glb", True)),
        length_m=float(getattr(args, "car_length_m", DEFAULT_CAR_LENGTH_M)),
        scale=getattr(args, "car_scale", DEFAULT_CAR_SCALE),
        yaw_deg=float(getattr(args, "car_yaw_deg", DEFAULT_CAR_YAW_DEG)),
        pitch_deg=float(getattr(args, "car_pitch_deg", DEFAULT_CAR_PITCH_DEG)),
        roll_deg=float(getattr(args, "car_roll_deg", DEFAULT_CAR_ROLL_DEG)),
        offset_x=float(getattr(args, "car_offset_x", DEFAULT_CAR_OFFSET_X)),
        offset_y=float(getattr(args, "car_offset_y", DEFAULT_CAR_OFFSET_Y)),
        offset_z=float(getattr(args, "car_offset_z", DEFAULT_CAR_OFFSET_Z)),
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


def _open3d_offscreen_available() -> bool:
    """Return True when libEGL is present (OffscreenRenderer is unsafe without it)."""
    if os.environ.get("ICAR_BEV_BACKEND", "auto").lower() == "numpy":
        return False
    return ctypes.util.find_library("EGL") is not None


def _look_at_view_matrix(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    """World-to-camera matrix (OpenGL / Filament convention)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-12
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-9:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right /= right_norm
    up_cam = np.cross(right, forward)
    up_cam /= np.linalg.norm(up_cam) + 1e-12

    view = np.eye(4, dtype=np.float64)
    view[0, :3] = right
    view[1, :3] = up_cam
    view[2, :3] = -forward
    view[0, 3] = -float(np.dot(right, eye))
    view[1, 3] = -float(np.dot(up_cam, eye))
    view[2, 3] = float(np.dot(forward, eye))
    return view


def _perspective_projection_matrix(
    fov_y_deg: float,
    aspect: float,
    near: float,
    far: float,
) -> np.ndarray:
    """Vertical-FOV perspective matrix matching Open3D Filament NDC."""
    fov = np.radians(float(fov_y_deg))
    f = 1.0 / max(np.tan(fov * 0.5), 1e-9)
    near = max(float(near), 1e-4)
    far = max(float(far), near + 1e-3)
    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = f / max(float(aspect), 1e-9)
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2.0 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj


def _estimate_near_far_planes(eye: np.ndarray, pts: np.ndarray) -> tuple[float, float]:
    dist = np.linalg.norm(np.asarray(pts, dtype=np.float64) - eye, axis=1)
    if dist.size == 0:
        return 0.1, 1000.0
    near = max(float(np.percentile(dist, 1.0)) * 0.25, 0.05)
    far = max(float(np.percentile(dist, 99.5)) * 2.5, near + 1.0)
    return near, far


def _view_proj_from_camera(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    *,
    field_of_view: float,
    width: int,
    height: int,
    near_far_pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    view = _look_at_view_matrix(eye, target, up)
    near, far = _estimate_near_far_planes(eye, near_far_pts)
    proj = _perspective_projection_matrix(
        field_of_view,
        width / max(height, 1),
        near,
        far,
    )
    return view, proj


def _center_camera_target_on_points(
    pts: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    *,
    view_proj_fn,
    width: int,
    height: int,
    n_iters: int = CENTER_VIEW_ITERS,
    field_of_view: float = FIELD_OF_VIEW,
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
        view, proj = view_proj_fn(eye, target, up)
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


def _prepare_rear_top_points(
    pts: np.ndarray,
    cols: np.ndarray,
    *,
    center_xz: tuple[float, float] | None,
    ground_y: float | None,
    crop_percentile: float,
    scene_radius_percentile: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float], float, float]:
    pts = np.asarray(pts, dtype=np.float64)
    cols = np.asarray(cols, dtype=np.float64)

    if center_xz is None:
        center_xz = estimate_scene_center_xz(pts)
    if ground_y is None:
        ground_y = estimate_ground_y(pts, center_xz)

    center_x, center_z = center_xz
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
    return pts, cols, center_xz, float(ground_y), scene_radius


def _subsample_points_for_bev(
    pts: np.ndarray,
    cols: np.ndarray,
    *,
    max_points: int = MAX_BEV_RENDER_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(pts.shape[0])
    cap = max(int(max_points), 1)
    if n <= cap:
        return pts, cols
    idx = np.random.default_rng(0).choice(n, cap, replace=False)
    return pts[idx], cols[idx]


def _splat_points_to_image(
    pts: np.ndarray,
    cols: np.ndarray,
    view: np.ndarray,
    proj: np.ndarray,
    *,
    width: int,
    height: int,
    point_size: float,
    background_rgb: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Paint points into an RGB image with depth ordering (numpy, no EGL)."""
    bg = background_rgb if background_rgb is not None else BACKGROUND_RGBA[:3]
    u, v, w = _project_points(pts, view, proj, width, height)
    ndc_z = np.empty(pts.shape[0], dtype=np.float64)
    valid_w = np.abs(w) > 1e-9
    ndc_z[valid_w] = (proj @ (view @ np.concatenate([pts[valid_w], np.ones((valid_w.sum(), 1))], axis=1).T)).T[valid_w, 2] / w[valid_w]

    valid = (
        valid_w
        & (w > 0.05)
        & (u >= 0.0)
        & (u < width)
        & (v >= 0.0)
        & (v < height)
    )
    if not np.any(valid):
        img = np.zeros((height, width, 3), dtype=np.float64)
        img[:] = bg
        return img

    u = u[valid]
    v = v[valid]
    w = w[valid]
    ndc_z = ndc_z[valid]
    cols = np.clip(cols[valid], 0.0, 1.0)

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    ui = np.clip(ui, 0, width - 1)
    vi = np.clip(vi, 0, height - 1)
    order = np.lexsort((ndc_z, vi, ui))
    ui = ui[order]
    vi = vi[order]
    cols = cols[order]

    img = np.zeros((height, width, 3), dtype=np.float64)
    img[:] = bg

    radius = max(int(round(float(point_size) * 0.5)), 1)
    if radius <= 1:
        img[vi, ui] = cols
        return img

    offsets = [(du, dv) for du in range(-radius, radius + 1) for dv in range(-radius, radius + 1)]
    for du, dv in offsets:
        uu = np.clip(ui + du, 0, width - 1)
        vv = np.clip(vi + dv, 0, height - 1)
        img[vv, uu] = cols
    return img


def render_numpy_rear_top(
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
    """Headless rear-top BEV via numpy projection (no Open3D OffscreenRenderer)."""
    pts, cols, center_xz, ground_y, scene_radius = _prepare_rear_top_points(
        pts,
        cols,
        center_xz=center_xz,
        ground_y=ground_y,
        crop_percentile=crop_percentile,
        scene_radius_percentile=scene_radius_percentile,
    )
    center_x, center_z = center_xz
    pts, cols = _subsample_points_for_bev(pts, cols)

    size = int(image_size)
    eye, target, up = _camera_eye_target(
        center_xz,
        ground_y,
        scene_radius,
        cam_height_frac=cam_height_frac,
        cam_back_frac=cam_back_frac,
        look_ahead_frac=look_ahead_frac,
        look_down=look_down,
    )

    def view_proj_fn(eye_, target_, up_):
        return _view_proj_from_camera(
            eye_,
            target_,
            up_,
            field_of_view=field_of_view,
            width=size,
            height=size,
            near_far_pts=pts,
        )

    target = _center_camera_target_on_points(
        pts,
        eye,
        target,
        up,
        view_proj_fn=view_proj_fn,
        width=size,
        height=size,
        field_of_view=field_of_view,
    )
    view, proj = view_proj_fn(eye, target, up)
    effective_point_size = max(float(point_size), NUMPY_BEV_POINT_SIZE)
    img = _splat_points_to_image(
        pts,
        cols,
        view,
        proj,
        width=size,
        height=size,
        point_size=effective_point_size,
        background_rgb=NUMPY_BEV_BACKGROUND_RGBA[:3],
    )

    car_world = np.array([center_x + car_offset_x, ground_y, center_z], dtype=np.float64)
    rig_u, rig_v, _ = _project_point(car_world, view, proj, size, size)
    if not np.isfinite(rig_u):
        rig_u, rig_v = size * 0.5, size * 0.7
    return img, (rig_u, rig_v)


def render_rear_top(
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
) -> tuple[np.ndarray, tuple[float, float], str]:
    """Render rear-top BEV; auto-select Open3D (EGL) or numpy fallback."""
    backend = os.environ.get("ICAR_BEV_BACKEND", "auto").lower()
    kwargs = dict(
        center_xz=center_xz,
        ground_y=ground_y,
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
    if backend == "numpy" or (backend == "auto" and not _open3d_offscreen_available()):
        return (*render_numpy_rear_top(pts, cols, **kwargs), "numpy")
    try:
        return (*render_open3d_rear_top(pts, cols, **kwargs), "open3d")
    except Exception as exc:  # noqa: BLE001
        print(f"[bev] Open3D render failed ({exc}); falling back to numpy")
        return (*render_numpy_rear_top(pts, cols, **kwargs), "numpy")


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

    pts, cols, center_xz, ground_y, scene_radius = _prepare_rear_top_points(
        pts,
        cols,
        center_xz=center_xz,
        ground_y=ground_y,
        crop_percentile=crop_percentile,
        scene_radius_percentile=scene_radius_percentile,
    )
    center_x, center_z = center_xz

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

    def view_proj_fn(eye_, target_, up_):
        renderer.setup_camera(float(field_of_view), target_, eye_, up_)
        view = np.asarray(renderer.scene.camera.get_view_matrix(), dtype=np.float64)
        proj = np.asarray(renderer.scene.camera.get_projection_matrix(), dtype=np.float64)
        return view, proj

    target = _center_camera_target_on_points(
        pts,
        eye,
        target,
        up,
        view_proj_fn=view_proj_fn,
        width=size,
        height=size,
        field_of_view=field_of_view,
    )
    renderer.setup_camera(float(field_of_view), target, eye, up)

    img = np.asarray(renderer.render_to_image(), dtype=np.float64) / 255.0

    view, proj = view_proj_fn(eye, target, up)
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
    """Render a rear-top view + car.png overlay (Open3D EGL or numpy fallback).

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

    img, (rig_u, rig_v), backend = render_rear_top(
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
        f"BEV rear-top ({backend}, cam_back_frac={cam_back_frac}, cam_height_frac={cam_height_frac}, "
        f"look_ahead_frac={look_ahead_frac}, fov={field_of_view}){car_msg} -> {out_path}"
    )
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export car_glb.glb to a dense canonical car_glb.ply.",
    )
    parser.add_argument("--glb", type=Path, default=DEFAULT_CAR_GLB, help="Input GLB path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CAR_PLY,
        help="Output PLY path (default: scripts/post_process/car_glb.ply).",
    )
    parser.add_argument(
        "--length-m",
        type=float,
        default=DEFAULT_CAR_LENGTH_M,
        help="Canonical car length baked into the PLY.",
    )
    parser.add_argument(
        "--sample-spacing",
        type=float,
        default=DEFAULT_CAR_SAMPLE_SPACING,
        help="Surface sample spacing in meters (smaller = denser).",
    )
    export_args = parser.parse_args()
    export_car_glb_to_ply(
        export_args.output,
        glb_path=export_args.glb,
        length_m=export_args.length_m,
        sample_spacing=export_args.sample_spacing,
    )
