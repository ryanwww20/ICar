#!/usr/bin/env python3
"""
Version B (multi-frame, one forward pass):
Run VGGT on 7 Argoverse 2 ring cameras across MULTIPLE consecutive frames
in a single forward pass.

跟原版 (run_vggt_av2_7ring.py) 的差異:
  原版: 每個 frame 各跑一次 VGGT (single frame, 7 views)。
  本版: 收集 frame_idx ~ frame_idx+num_frames-1 的全部 7xN 張影像，
        一次餵進 VGGT。連續 frame 同相機間有 overlap，VGGT 能利用
        multi-view 一致性，scale 自洽、可補洞。

注意:
  - 7xN 張一次餵，記憶體隨 N 線性增加。先用 --num-frames 2 (14 張) 試。
  - 一定要在 GPU 上跑 (CPU 會慢到不可用)。

Uses VGGT world_points directly (relative / up-to-scale)。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
VGGT_ROOT = REPO_ROOT / "vggt"
DEFAULT_AV2_ROOT = REPO_ROOT / "vggt/data/AV2"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(VGGT_ROOT) not in sys.path:
    sys.path.insert(0, str(VGGT_ROOT))

from av2_utils import (  # noqa: E402
    build_scene_mapping,
    discover_cameras_in_log,
    extract_timestamp,
    resolve_log_dir,
    resolve_split_root,
)
from ply_postprocess_common import (  # noqa: E402
    add_postprocess_args,
    maybe_postprocess_pointcloud,
    maybe_show_pointcloud,
)
from vggt_nuscenes_common import (  # noqa: E402
    pointcloud_output_path,
    add_cleanup_args,
    write_pointcloud,
)

RING_CAMERAS = (
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_side_left",
    "ring_side_right",
    "ring_rear_left",
    "ring_rear_right",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VGGT on 7 AV2 ring cameras across multiple frames (one forward pass)."
    )
    parser.add_argument("--av2-root", type=Path, default=DEFAULT_AV2_ROOT,
                        help=f"AV2 dataset root (default: {DEFAULT_AV2_ROOT})")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--scene-id", type=str, default="0",
                        help="Scene index under the split (default: 0).")
    parser.add_argument("--log-id", type=str, default=None,
                        help="AV2 log UUID. Overrides --scene-id when set.")
    parser.add_argument("--list-scenes", action="store_true",
                        help="List scene-N -> log UUID mapping and exit.")
    parser.add_argument("--frame-idx", type=int, default=0,
                        help="Start frame index.")
    parser.add_argument("--num-frames", type=int, default=2,
                        help="Number of consecutive frames to fuse in ONE forward pass.")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="Stride between fused frames (1=consecutive).")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/vggt_av2_7ring_multiframe"))
    parser.add_argument("--model-id", type=str, default="facebook/VGGT-1B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--conf-thresh", type=float, default=0.5)
    parser.add_argument("--pixel-stride", type=int, default=2)
    add_cleanup_args(parser)
    add_postprocess_args(parser)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def collect_multiframe_views(
    log_dir: Path, frame_idx: int, num_frames: int, frame_step: int
) -> list[tuple[str, Path, int, int]]:
    """
    收集多個 frame 的七路影像。
    回傳 [(camera_name, image_path, timestamp_ns, frame_idx), ...]
    順序: 先 frame，再相機 (frame0 的七路, frame1 的七路, ...)。
    """
    cameras = discover_cameras_in_log(log_dir)
    missing = [name for name in RING_CAMERAS if name not in cameras]
    if missing:
        raise ValueError(
            f"Log {log_dir.name} is missing ring cameras: {missing}. "
            f"Available: {sorted(cameras.keys())}"
        )

    min_count = min(len(cameras[name]) for name in RING_CAMERAS)
    frame_indices = [frame_idx + i * frame_step for i in range(num_frames)]
    for fi in frame_indices:
        if fi < 0 or fi >= min_count:
            raise ValueError(
                f"frame-idx={fi} out of range [0, {min_count - 1}] for log {log_dir.name}"
            )

    views: list[tuple[str, Path, int, int]] = []
    for fi in frame_indices:
        for name in RING_CAMERAS:
            path = cameras[name][fi]
            views.append((name, path, extract_timestamp(path), fi))
    return views


def save_depth_png(path: Path, depth: np.ndarray) -> None:
    import cv2
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
    """Subsample fused world points and colors. world_points: (S,H,W,3) S=7*num_frames."""
    s, h, w, _ = world_points.shape
    pts_list, col_list = [], []
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


def run_vggt(image_paths: list[Path], device: str, model_id: str) -> dict:
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )

    print(f"[vggt] loading model {model_id} on {device} ...")
    model = VGGT.from_pretrained(model_id).to(device).eval()

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


def process_multiframe(
    log_dir: Path, scene_label: str, base_out_dir: Path, out_dir: Path,
    args: argparse.Namespace, device: str, scene_index: int,
) -> int:
    views = collect_multiframe_views(
        log_dir, args.frame_idx, args.num_frames, args.frame_step)
    n_frames = args.num_frames
    tag = f"f{args.frame_idx:06d}_n{n_frames}"
    frame_dir = out_dir / tag
    frame_dir.mkdir(parents=True, exist_ok=True)
    ply_path = pointcloud_output_path(
        base_out_dir, mode="rel", dataset="av2",
        scene_index=scene_index, frame_index=args.frame_idx,
    )

    print(f"\n[{scene_label}] fusing {n_frames} frames x 7 cams = {len(views)} views")
    for cam, path, ts, fi in views:
        print(f"  [frame {fi}] {cam}: {path.name} (ts={ts})")

    meta = {
        "scene_id": scene_index, "scene_label": scene_label, "log_id": log_dir.name,
        "frame_idx_start": args.frame_idx, "num_frames": n_frames,
        "frame_step": args.frame_step,
        "fused_views": len(views),
        "timestamp_ns": [{"cam": c, "frame": fi, "ts": ts} for c, _, ts, fi in views],
        "image_paths": [str(p) for _, p, _, _ in views],
        "conf_thresh": args.conf_thresh, "pixel_stride": args.pixel_stride,
        "voxel_size": args.voxel_size, "no_cleanup": args.no_cleanup,
        "pointcloud_path": str(ply_path),
    }
    with (frame_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if args.dry_run:
        print(f"[dry-run] would run VGGT on {len(views)} images -> {ply_path.name}")
        maybe_postprocess_pointcloud(ply_path, args, label=tag, dry_run=True)
        return 0

    image_paths = [path for _, path, _, _ in views]
    pred = run_vggt(image_paths, device, args.model_id)

    with (frame_dir / "cameras.json").open("w", encoding="utf-8") as f:
        json.dump({
            "extrinsic": pred["extrinsic"].tolist(),
            "intrinsic": pred["intrinsic"].tolist(),
            "proc_hw": list(pred["proc_hw"]),
        }, f, indent=2)

    import shutil
    for i, (cam, src, _, fi) in enumerate(views):
        shutil.copy2(src, frame_dir / f"image_{i:02d}_frame{fi}_{cam}{src.suffix.lower()}")
        save_depth_png(frame_dir / f"depth_{i:02d}_frame{fi}_{cam}.png", pred["depth"][i])

    pts, cols = world_points_to_ply(
        pred["world_points"], pred["world_conf"], pred["images"],
        args.conf_thresh, args.pixel_stride,
    )
    write_pointcloud(
        ply_path, pts, cols, voxel_size=args.voxel_size,
        no_cleanup=args.no_cleanup, label=tag,
    )
    post_path = maybe_postprocess_pointcloud(ply_path, args, label=tag)
    maybe_show_pointcloud(ply_path, post_path, args, label=tag)
    if post_path is not None:
        meta["postprocessed_pointcloud_path"] = str(post_path)
        with (frame_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    return 0


def main() -> int:
    args = parse_args()
    av2_root = args.av2_root.expanduser().resolve()
    base_out_dir = args.output_dir.expanduser().resolve()

    if not av2_root.is_dir():
        print(f"ERROR: --av2-root does not exist: {av2_root}", file=sys.stderr)
        return 1

    split_root = resolve_split_root(av2_root, args.split)

    if args.list_scenes:
        mapping = build_scene_mapping(split_root)
        if not mapping:
            print(f"No scenes under {split_root}", file=sys.stderr)
            return 1
        print(f"Scenes under {split_root} ({args.split}):")
        for entry in mapping:
            print(f"  {entry['scene_label']}: {entry['log_id']}")
        return 0

    try:
        log_dir, scene_label, scene_index = resolve_log_dir(
            split_root,
            scene_id=args.scene_id if not args.log_id else None,
            log_id=args.log_id,
        )
    except (FileNotFoundError, IndexError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out_dir = base_out_dir / scene_label
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] av2_root={av2_root} {scene_label} log_id={log_dir.name} device={device}")
    if device == "cpu":
        print("[WARN] device=cpu! VGGT 在 CPU 上跑多 frame 會非常慢，建議改 GPU。")

    mapping = build_scene_mapping(split_root)
    with (base_out_dir / "scene_mapping.json").open("w", encoding="utf-8") as f:
        json.dump({"split": args.split, "av2_root": str(av2_root), "scenes": mapping},
                  f, indent=2, ensure_ascii=False)

    run_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "av2_root": str(av2_root), "split": args.split,
        "scene_id": scene_index, "scene_label": scene_label, "log_id": log_dir.name,
        "frame_idx_start": args.frame_idx, "num_frames": args.num_frames,
        "frame_step": args.frame_step, "mode": "multiframe_single_forward",
        "ring_cameras": list(RING_CAMERAS), "dry_run": args.dry_run,
        "voxel_size": args.voxel_size, "conf_thresh": args.conf_thresh,
        "pixel_stride": args.pixel_stride,
    }
    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    try:
        status = process_multiframe(
            log_dir, scene_label, base_out_dir, out_dir, args, device, scene_index)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    print(f"\n[done] outputs under {out_dir} ({scene_label}, log_id={log_dir.name})")
    return status


if __name__ == "__main__":
    raise SystemExit(main())