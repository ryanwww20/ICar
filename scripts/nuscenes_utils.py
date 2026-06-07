#!/usr/bin/env python3
"""Shared nuScenes helpers for VGGT sample export scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from av2_utils import format_scene_id, parse_scene_id

CAM_CHANNELS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

DEFAULT_DATAROOT = Path(__file__).resolve().parent.parent / "v1.0-mini"
DEFAULT_VERSION = "v1.0-mini"


@dataclass(frozen=True)
class CameraView:
    name: str
    image_path: Path
    timestamp_ns: int
    sample_data_token: str
    intrinsic: np.ndarray
    T_W_C: np.ndarray


def quat_wxyz_to_matrix(rotation_wxyz: list[float]) -> np.ndarray:
    w, x, y, z = rotation_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_T(translation: list[float], rotation_wxyz: list[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_wxyz_to_matrix(rotation_wxyz)
    T[:3, 3] = translation
    return T


class NuScenesLite:
    """Minimal nuScenes reader backed by JSON metadata (no devkit required)."""

    def __init__(self, dataroot: Path, version: str = DEFAULT_VERSION) -> None:
        self.dataroot = dataroot.resolve()
        meta_dir = self.dataroot / version
        if not meta_dir.is_dir():
            raise FileNotFoundError(f"Metadata directory not found: {meta_dir}")

        self.version = version
        self._tables: dict[str, dict[str, dict[str, Any]]] = {}
        for name in (
            "scene",
            "sample",
            "sample_data",
            "calibrated_sensor",
            "ego_pose",
        ):
            path = meta_dir / f"{name}.json"
            with path.open(encoding="utf-8") as f:
                rows = json.load(f)
            self._tables[name] = {row["token"]: row for row in rows}

        self.scene = list(self._tables["scene"].values())
        self._sample_data_by_sample: dict[str, dict[str, str]] = {}
        for sd in self._tables["sample_data"].values():
            if not sd.get("is_key_frame") or not sd.get("sample_token"):
                continue
            channel = _channel_from_filename(sd["filename"])
            if channel is None:
                continue
            self._sample_data_by_sample.setdefault(sd["sample_token"], {})[channel] = sd[
                "token"
            ]

    def get(self, table: str, token: str) -> dict[str, Any]:
        try:
            return self._tables[table][token]
        except KeyError as exc:
            raise KeyError(f"{table} token not found: {token}") from exc

    def get_sample_data_path(self, sample_data_token: str) -> str:
        sd = self.get("sample_data", sample_data_token)
        return str(self.dataroot / sd["filename"])

    def get_sample_with_data(self, sample_token: str) -> dict[str, Any]:
        sample = dict(self.get("sample", sample_token))
        sample["data"] = dict(self._sample_data_by_sample.get(sample_token, {}))
        return sample


def _channel_from_filename(filename: str) -> str | None:
    if "__" not in filename:
        return None
    channel = filename.split("__")[1]
    return channel if channel in CAM_CHANNELS else None


def load_nuscenes(dataroot: Path, version: str = DEFAULT_VERSION) -> NuScenesLite:
    try:
        from nuscenes.nuscenes import NuScenes

        return NuScenes(version=version, dataroot=str(dataroot.resolve()), verbose=False)
    except ImportError:
        print("[nuscenes] devkit not installed; using JSON metadata reader")
        return NuScenesLite(dataroot, version)


def get_camera_geometry(nusc: Any, sample_data_token: str) -> tuple[str, np.ndarray, np.ndarray]:
    sd = nusc.get("sample_data", sample_data_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    intrinsic = np.array(cs["camera_intrinsic"], dtype=np.float64)
    T_E_C = make_T(cs["translation"], cs["rotation"])
    T_W_E = make_T(ep["translation"], ep["rotation"])
    T_W_C = T_W_E @ T_E_C
    path = Path(nusc.get_sample_data_path(sample_data_token))
    return str(path), intrinsic, T_W_C


def build_scene_mapping(nusc: Any) -> list[dict[str, str | int]]:
    mapping: list[dict[str, str | int]] = []
    for index, scene in enumerate(nusc.scene):
        sample_count = count_samples_in_scene(nusc, scene["token"])
        mapping.append(
            {
                "scene_id": index,
                "scene_label": format_scene_id(index),
                "scene_name": scene["name"],
                "scene_token": scene["token"],
                "description": scene["description"],
                "num_samples": sample_count,
            }
        )
    return mapping


def count_samples_in_scene(nusc: Any, scene_token: str) -> int:
    scene = nusc.get("scene", scene_token)
    count = 0
    sample = nusc.get("sample", scene["first_sample_token"])
    while True:
        count += 1
        if not sample["next"]:
            break
        sample = nusc.get("sample", sample["next"])
    return count


def resolve_scene(
    nusc: Any,
    *,
    scene_id: int | str | None = None,
    scene_name: str | None = None,
) -> tuple[dict, str, int]:
    if scene_name:
        for index, scene in enumerate(nusc.scene):
            if scene["name"] == scene_name:
                return scene, format_scene_id(index), index
        raise FileNotFoundError(f"scene-name not found: {scene_name}")

    index = 0 if scene_id is None else parse_scene_id(scene_id)
    if index >= len(nusc.scene):
        raise IndexError(
            f"scene-{index} out of range: dataset has {len(nusc.scene)} scene(s) "
            f"(valid: scene-0 .. scene-{len(nusc.scene) - 1})"
        )
    scene = nusc.scene[index]
    return scene, format_scene_id(index), index


def iter_sample_tokens(nusc: Any, scene: dict) -> list[str]:
    tokens: list[str] = []
    sample = nusc.get("sample", scene["first_sample_token"])
    while True:
        tokens.append(sample["token"])
        if not sample["next"]:
            break
        sample = nusc.get("sample", sample["next"])
    return tokens


def _get_sample(nusc: Any, sample_token: str) -> dict[str, Any]:
    if isinstance(nusc, NuScenesLite):
        return nusc.get_sample_with_data(sample_token)
    return nusc.get("sample", sample_token)


def collect_six_cam_views(
    nusc: Any,
    scene: dict,
    sample_idx: int,
) -> tuple[str, list[CameraView]]:
    sample_tokens = iter_sample_tokens(nusc, scene)
    if sample_idx < 0 or sample_idx >= len(sample_tokens):
        raise ValueError(
            f"sample-idx={sample_idx} out of range [0, {len(sample_tokens) - 1}] "
            f"for scene {scene['name']}"
        )

    sample = _get_sample(nusc, sample_tokens[sample_idx])
    views: list[CameraView] = []
    missing: list[str] = []

    for channel in CAM_CHANNELS:
        if channel not in sample["data"]:
            missing.append(channel)
            continue
        token = sample["data"][channel]
        path_str, intrinsic, T_W_C = get_camera_geometry(nusc, token)
        sd = nusc.get("sample_data", token)
        views.append(
            CameraView(
                name=channel,
                image_path=Path(path_str),
                timestamp_ns=int(sd["timestamp"]),
                sample_data_token=token,
                intrinsic=intrinsic,
                T_W_C=T_W_C,
            )
        )

    if missing:
        raise ValueError(
            f"Sample {sample_idx} in scene {scene['name']} missing cameras: {missing}"
        )
    return sample["token"], views
