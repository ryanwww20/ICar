#!/usr/bin/env python3
"""Shared utilities for Argoverse 2 Sensor Dataset exploration and pair export."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

STEREO_LEFT_KEYWORDS = (
    "stereo",
    "front_left",
    "front-left",
    "ring_front_left",
    "stereo_front_left",
)

STEREO_RIGHT_KEYWORDS = (
    "stereo",
    "front_right",
    "front-right",
    "ring_front_right",
    "stereo_front_right",
)

RING_ADJACENT_PAIRS = (
    ("ring_front_left", "ring_front_center"),
    ("ring_front_center", "ring_front_right"),
    ("ring_front_right", "ring_rear_right"),
    ("ring_rear_right", "ring_rear_left"),
    ("ring_rear_left", "ring_side_left"),
    ("ring_side_left", "ring_front_left"),
)

CAMERA_DIR_PATTERNS = (
    "sensors/cameras/*",
    "sensor/cameras/*",
    "cameras/*",
    "sensors/camera/*",
)


def normalize_camera_name(name: str) -> str:
    return re.sub(r"[-\s]+", "_", name.strip().lower())


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    images = [
        p.resolve()
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda p: (extract_timestamp(p), p.name))


def extract_timestamp(path: Path) -> int:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return int(digits)
    return 0


def resolve_split_root(av2_root: Path, split: str | None) -> Path:
    av2_root = av2_root.resolve()
    candidates: list[Path] = []
    if split:
        candidates.extend(
            [
                av2_root / split,
                av2_root / "sensor" / split,
                av2_root / "Sensor" / split,
                av2_root / "sensors" / split,
            ]
        )
    candidates.extend([av2_root, av2_root / "sensor", av2_root / "Sensor"])

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_dir() and _contains_log_dirs(candidate):
            return candidate

    if split and (av2_root / split).is_dir():
        return (av2_root / split).resolve()
    return av2_root.resolve()


def _contains_log_dirs(directory: Path) -> bool:
    count = 0
    for child in directory.iterdir():
        if not child.is_dir():
            continue
        if discover_cameras_in_log(child):
            count += 1
            if count >= 1:
                return True
    return False


def find_log_dirs(split_root: Path, max_logs: int | None = None) -> list[Path]:
    split_root = split_root.resolve()
    if not split_root.is_dir():
        raise FileNotFoundError(f"Split root does not exist: {split_root}")

    logs: list[Path] = []
    for child in sorted(split_root.iterdir()):
        if not child.is_dir():
            continue
        if discover_cameras_in_log(child):
            logs.append(child.resolve())
            if max_logs is not None and len(logs) >= max_logs:
                break
    return logs


def discover_cameras_in_log(log_dir: Path, max_images_per_camera: int | None = None) -> dict[str, list[Path]]:
    log_dir = log_dir.resolve()
    cameras: dict[str, list[Path]] = {}

    for pattern in CAMERA_DIR_PATTERNS:
        for cam_dir in log_dir.glob(pattern):
            if not cam_dir.is_dir():
                continue
            images = list_images(cam_dir)
            if images:
                cameras[cam_dir.name] = images

    if cameras:
        return _maybe_trim_cameras(cameras, max_images_per_camera)

    grouped: dict[str, list[Path]] = defaultdict(list)
    max_depth = 6
    for path in log_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            rel = path.relative_to(log_dir)
        except ValueError:
            continue
        if len(rel.parts) > max_depth:
            continue
        camera_name = rel.parts[-2] if len(rel.parts) >= 2 else rel.parts[0]
        grouped[camera_name].append(path.resolve())

    for name, images in grouped.items():
        if len(images) >= 3:
            cameras[name] = sorted(images, key=lambda p: (extract_timestamp(p), p.name))

    return _maybe_trim_cameras(cameras, max_images_per_camera)


def _maybe_trim_cameras(
    cameras: dict[str, list[Path]], max_images_per_camera: int | None
) -> dict[str, list[Path]]:
    if max_images_per_camera is None:
        return cameras
    return {
        name: images[:max_images_per_camera]
        for name, images in cameras.items()
    }


def _name_matches(name: str, keywords: Iterable[str], required_any: tuple[str, ...] | None = None) -> bool:
    normalized = normalize_camera_name(name)
    if required_any and not any(token in normalized for token in required_any):
        return False
    return any(token in normalized for token in keywords)


def detect_stereo_cameras(camera_names: list[str]) -> tuple[str | None, str | None]:
    left_candidates = [
        name
        for name in camera_names
        if _name_matches(name, STEREO_LEFT_KEYWORDS, required_any=("front", "stereo"))
        or normalize_camera_name(name) in {"stereo_front_left", "ring_front_left"}
    ]
    right_candidates = [
        name
        for name in camera_names
        if _name_matches(name, STEREO_RIGHT_KEYWORDS, required_any=("front", "stereo"))
        or normalize_camera_name(name) in {"stereo_front_right", "ring_front_right"}
    ]

    if not left_candidates or not right_candidates:
        return None, None

    def score(name: str, side: str) -> tuple[int, str]:
        normalized = normalize_camera_name(name)
        stereo_bonus = 2 if "stereo" in normalized else 0
        side_bonus = 3 if side in normalized else 0
        return (stereo_bonus + side_bonus, normalized)

    left = max(left_candidates, key=lambda n: score(n, "left"))
    right = max(right_candidates, key=lambda n: score(n, "right"))
    return left, right


def find_camera_by_name(camera_names: list[str], requested: str) -> str | None:
    requested_norm = normalize_camera_name(requested)
    for name in camera_names:
        if normalize_camera_name(name) == requested_norm:
            return name
    for name in camera_names:
        if requested_norm in normalize_camera_name(name):
            return name
    return None


def match_image_pairs(
    left_images: list[Path],
    right_images: list[Path],
) -> list[tuple[Path, Path]]:
    left_sorted = sorted(left_images, key=lambda p: (extract_timestamp(p), p.name))
    right_sorted = sorted(right_images, key=lambda p: (extract_timestamp(p), p.name))

    if not left_sorted or not right_sorted:
        return []

    if len(left_sorted) == len(right_sorted):
        left_ts = [extract_timestamp(p) for p in left_sorted]
        right_ts = [extract_timestamp(p) for p in right_sorted]
        if left_ts == right_ts or all(abs(a - b) <= 1 for a, b in zip(left_ts, right_ts)):
            return list(zip(left_sorted, right_sorted))

    right_ts_values = [extract_timestamp(p) for p in right_sorted]
    pairs: list[tuple[Path, Path]] = []
    for left_path in left_sorted:
        left_ts = extract_timestamp(left_path)
        idx = min(range(len(right_ts_values)), key=lambda i: abs(right_ts_values[i] - left_ts))
        pairs.append((left_path, right_sorted[idx]))
    return pairs


def make_temporal_pairs(
    images: list[Path], stride: int = 1
) -> list[tuple[Path, Path]]:
    images = sorted(images, key=lambda p: (extract_timestamp(p), p.name))
    pairs: list[tuple[Path, Path]] = []
    for i in range(0, len(images) - stride, stride):
        pairs.append((images[i], images[i + stride]))
    return pairs


def resolve_ring_adjacent_pairs(camera_names: list[str]) -> list[tuple[str, str]]:
    normalized_to_actual = {normalize_camera_name(name): name for name in camera_names}
    resolved: list[tuple[str, str]] = []
    for left, right in RING_ADJACENT_PAIRS:
        left_actual = normalized_to_actual.get(normalize_camera_name(left))
        right_actual = normalized_to_actual.get(normalize_camera_name(right))
        if left_actual and right_actual:
            resolved.append((left_actual, right_actual))
    return resolved


def choose_default_camera(camera_names: list[str]) -> str | None:
    if not camera_names:
        return None
    priority = (
        "stereo_front_left",
        "ring_front_center",
        "ring_front_left",
        "front",
    )
    normalized = {normalize_camera_name(name): name for name in camera_names}
    for token in priority:
        for norm, actual in normalized.items():
            if token in norm:
                return actual
    return sorted(camera_names)[0]
