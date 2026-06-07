#!/usr/bin/env python3
"""Shared VGGT inference and point-cloud helpers for nuScenes scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from nuscenes_utils import DEFAULT_DATAROOT, DEFAULT_VERSION


def add_nuscenes_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataroot",
        type=Path,
        default=DEFAULT_DATAROOT,
        help=f"nuScenes dataroot (default: {DEFAULT_DATAROOT})",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=DEFAULT_VERSION,
        help=f"nuScenes version (default: {DEFAULT_VERSION})",
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        default="0",
        help="Scene index in the mini split, e.g. 0 or scene-0.",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default=None,
        help="nuScenes scene name, e.g. scene-0061. Overrides --scene-id.",
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List scene-N -> scene name mapping and exit.",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Sample index within the scene (keyframe chain).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of consecutive samples to process (starting at --sample-idx).",
    )
    parser.add_argument("--model-id", type=str, default="facebook/VGGT-1B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--conf-thresh", type=float, default=0.5)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.10,
        help="Voxel size (m) for downsampling. Use 0 to skip voxel step only.",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip voxel downsampling and outlier removal; save all filtered points.",
    )
    parser.add_argument("--dry-run", action="store_true")


def save_depth_png(path: Path, depth: np.ndarray) -> None:
    d = depth.astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if not valid.any():
        return
    d_norm = np.zeros_like(d)
    lo, hi = np.percentile(d[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    d_norm[valid] = np.clip((d[valid] - lo) / (hi - lo), 0, 1)
    cv2.imwrite(str(path), (d_norm * 255).astype(np.uint8))


def world_points_to_ply(
    world_points: np.ndarray,
    conf: np.ndarray,
    images_rgb: np.ndarray,
    conf_thresh: float,
    pixel_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    s, h, w, _ = world_points.shape
    pts_list: list[np.ndarray] = []
    col_list: list[np.ndarray] = []

    for i in range(s):
        wp = world_points[i]
        cf = conf[i]
        img = images_rgb[i].transpose(1, 2, 0)
        vs = np.arange(0, h, pixel_stride)
        us = np.arange(0, w, pixel_stride)
        vv, uu = np.meshgrid(vs, us, indexing="ij")
        pts = wp[vv, uu].reshape(-1, 3)
        cc = cf[vv, uu].reshape(-1)
        cols = img[vv, uu].reshape(-1, 3)
        mask = (cc >= conf_thresh) & np.isfinite(pts).all(axis=1)
        pts_list.append(pts[mask])
        col_list.append(cols[mask])

    if not pts_list:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(pts_list, axis=0), np.concatenate(col_list, axis=0)


def estimate_scale_from_camera_baselines(
    vggt_extrinsic: np.ndarray,
    metric_T_W_C: list[np.ndarray],
) -> float:
    """Estimate metric scale by comparing pairwise camera baselines."""
    vggt_centers = np.array([-e[:3, :3].T @ e[:3, 3] for e in vggt_extrinsic])
    metric_centers = np.array([T[:3, 3] for T in metric_T_W_C])

    scales: list[float] = []
    for i in range(len(vggt_centers)):
        for j in range(i + 1, len(vggt_centers)):
            vggt_dist = float(np.linalg.norm(vggt_centers[i] - vggt_centers[j]))
            metric_dist = float(np.linalg.norm(metric_centers[i] - metric_centers[j]))
            if vggt_dist > 1e-6 and metric_dist > 1e-6:
                scales.append(metric_dist / vggt_dist)

    if not scales:
        return 1.0
    return float(np.median(scales))


def backproject_metric_points(
    depth: np.ndarray,
    conf: np.ndarray,
    intrinsic: np.ndarray,
    T_W_C: np.ndarray,
    native_bgr: np.ndarray,
    scale: float,
    conf_thresh: float,
    pixel_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    native_h, native_w = native_bgr.shape[:2]
    depth_native = cv2.resize(depth, (native_w, native_h), interpolation=cv2.INTER_LINEAR)
    conf_native = cv2.resize(conf, (native_w, native_h), interpolation=cv2.INTER_NEAREST)

    vs = np.arange(0, native_h, pixel_stride)
    us = np.arange(0, native_w, pixel_stride)
    vv, uu = np.meshgrid(vs, us, indexing="ij")

    z = depth_native[vv, uu].reshape(-1) * scale
    cc = conf_native[vv, uu].reshape(-1)
    u = uu.reshape(-1).astype(np.float64)
    v = vv.reshape(-1).astype(np.float64)
    mask = (z > 0) & (cc >= conf_thresh) & np.isfinite(z)
    u, v, z = u[mask], v[mask], z[mask]

    x = (u - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (v - intrinsic[1, 2]) * z / intrinsic[1, 1]
    pts_c = np.stack([x, y, z, np.ones_like(z)], axis=0)
    pts_w = (T_W_C @ pts_c)[:3].T

    cols = native_bgr[vv, uu][..., ::-1].reshape(-1, 3)[mask] / 255.0
    return pts_w, cols


def load_vggt_model(model_id: str, device: str):
    import torch
    from vggt.models.vggt import VGGT

    print(f"[vggt] loading model {model_id} on {device} ...")
    return VGGT.from_pretrained(model_id).to(device).eval()


def run_vggt_multi(model, image_paths: list[Path], device: str) -> dict:
    import torch
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )

    paths = [str(p) for p in image_paths]
    images = load_and_preprocess_images(paths).to(device)
    print(f"[vggt] input shape: {tuple(images.shape)} ({len(paths)} views)")

    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                pred = model(images)
        else:
            pred = model(images.float())

    proc_hw = (images.shape[-2], images.shape[-1])
    extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], proc_hw)
    extr = extr[0].cpu().numpy()
    intr = intr[0].cpu().numpy()

    return {
        "depth": pred["depth"].squeeze(0).squeeze(-1).float().cpu().numpy(),
        "depth_conf": pred["depth_conf"].squeeze(0).float().cpu().numpy(),
        "world_points": pred["world_points"].squeeze(0).float().cpu().numpy(),
        "world_conf": pred["world_points_conf"].squeeze(0).float().cpu().numpy(),
        "images": pred["images"].squeeze(0).float().cpu().numpy(),
        "extrinsic": extr,
        "intrinsic": intr,
        "proc_hw": proc_hw,
    }


def write_pointcloud(
    path: Path,
    pts: np.ndarray,
    cols: np.ndarray,
    *,
    voxel_size: float,
    no_cleanup: bool,
    label: str,
) -> None:
    import open3d as o3d

    if len(pts) == 0:
        print(f"[{label}] WARNING: no points passed confidence filter")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0, 1))
    print(f"[{label}] points: {len(pts)}")

    if no_cleanup:
        print(f"[{label}] cleanup disabled (--no-cleanup)")
    else:
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"[{label}] points after cleanup: {len(pcd.points)}")

    o3d.io.write_point_cloud(str(path), pcd)
