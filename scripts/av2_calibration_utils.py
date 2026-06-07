#!/usr/bin/env python3
"""AV2 calibration and per-frame camera geometry for metric VGGT export."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from av2_utils import discover_cameras_in_log, extract_timestamp

RING_CAMERAS = (
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_side_left",
    "ring_side_right",
    "ring_rear_left",
    "ring_rear_right",
)


@dataclass(frozen=True)
class Av2CameraView:
    name: str
    image_path: Path
    timestamp_ns: int
    intrinsic: np.ndarray
    T_W_C: np.ndarray


@dataclass
class LogCalibration:
    log_dir: Path
    intrinsics: Any
    extrinsics: Any
    ego_poses: Any


def quat_wxyz_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def se3_from_quat_translation(row: Any) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_wxyz_to_matrix(row["qw"], row["qx"], row["qy"], row["qz"])
    T[:3, 3] = [row["tx_m"], row["ty_m"], row["tz_m"]]
    return T


def intrinsic_matrix_from_row(row: Any) -> np.ndarray:
    return np.array(
        [
            [row["fx_px"], 0.0, row["cx_px"]],
            [0.0, row["fy_px"], row["cy_px"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def read_feather_table(path: Path) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to read AV2 feather calibration files") from exc
    try:
        return pd.read_feather(path)
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required to read AV2 feather calibration files. "
            "Install with: pip install pyarrow"
        ) from exc


@lru_cache(maxsize=32)
def load_log_calibration(log_dir: str) -> LogCalibration:
    root = Path(log_dir).resolve()
    calib_dir = root / "calibration"
    intrinsics_path = calib_dir / "intrinsics.feather"
    extrinsics_path = calib_dir / "egovehicle_SE3_sensor.feather"
    ego_path = root / "city_SE3_egovehicle.feather"

    missing = [
        str(path)
        for path in (intrinsics_path, extrinsics_path, ego_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Log {root.name} is missing metric calibration files: {missing}. "
            "Download calibration/ and city_SE3_egovehicle.feather for the log."
        )

    return LogCalibration(
        log_dir=root,
        intrinsics=read_feather_table(intrinsics_path),
        extrinsics=read_feather_table(extrinsics_path),
        ego_poses=read_feather_table(ego_path),
    )


def _camera_rows(calibration: LogCalibration, camera_name: str) -> tuple[Any, Any]:
    intr_rows = calibration.intrinsics
    ext_rows = calibration.extrinsics
    intr = intr_rows[intr_rows["sensor_name"] == camera_name]
    ext = ext_rows[ext_rows["sensor_name"] == camera_name]
    if intr.empty or ext.empty:
        raise ValueError(
            f"Camera {camera_name} not found in calibration for log {calibration.log_dir.name}"
        )
    return intr.iloc[0], ext.iloc[0]


def ego_pose_at_timestamp(calibration: LogCalibration, timestamp_ns: int) -> np.ndarray:
    ego_df = calibration.ego_poses
    idx = int((ego_df["timestamp_ns"] - timestamp_ns).abs().argmin())
    return se3_from_quat_translation(ego_df.iloc[idx])


def camera_to_world_at_timestamp(
    calibration: LogCalibration,
    camera_name: str,
    timestamp_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    intr_row, ext_row = _camera_rows(calibration, camera_name)
    T_city_ego = ego_pose_at_timestamp(calibration, timestamp_ns)
    T_ego_cam = se3_from_quat_translation(ext_row)
    T_W_C = T_city_ego @ T_ego_cam
    return intrinsic_matrix_from_row(intr_row), T_W_C


def collect_seven_ring_views_metric(
    log_dir: Path,
    frame_idx: int,
    calibration: LogCalibration | None = None,
) -> list[Av2CameraView]:
    calibration = calibration or load_log_calibration(str(log_dir.resolve()))
    cameras = discover_cameras_in_log(log_dir)
    missing = [name for name in RING_CAMERAS if name not in cameras]
    if missing:
        raise ValueError(
            f"Log {log_dir.name} is missing ring cameras: {missing}. "
            f"Available: {sorted(cameras.keys())}"
        )

    min_count = min(len(cameras[name]) for name in RING_CAMERAS)
    if frame_idx < 0 or frame_idx >= min_count:
        raise ValueError(
            f"frame-idx={frame_idx} out of range [0, {min_count - 1}] for log {log_dir.name}"
        )

    views: list[Av2CameraView] = []
    for name in RING_CAMERAS:
        path = cameras[name][frame_idx]
        timestamp_ns = extract_timestamp(path)
        intrinsic, T_W_C = camera_to_world_at_timestamp(calibration, name, timestamp_ns)
        views.append(
            Av2CameraView(
                name=name,
                image_path=path,
                timestamp_ns=timestamp_ns,
                intrinsic=intrinsic,
                T_W_C=T_W_C,
            )
        )
    return views
