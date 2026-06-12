#!/usr/bin/env python3
"""
AV2 LiDAR -> .ply (standalone，零專案依賴)，附 XYZ 座標軸 (X=紅 Y=綠 Z=藍)。

LiDAR 在 AV2 已 egomotion-compensated 到 ego frame，本身 metric。
單 frame 直接輸出 ego 座標；多 frame 累積用 city_SE3_egovehicle 投到 city 座標。

依賴: pip install pandas pyarrow numpy   (--coord city 需 scipy)

跑:
  python av2_lidar_to_ply.py --av2-root /path/AV2 --split val --log-id <UUID> --frame-idx 0
  python av2_lidar_to_ply.py --av2-root /path/AV2 --split val --list
  justin@please-3 ICar % python scripts/lidar_to_ply.py \                         
    --av2-root /Users/justin/Desktop/ICar/vggt/data/AV2 \
    --split val \
    --log-id 02a00399-3857-444e-8db3-a8f58489c394 \
    --frame-idx 0 \
    --no-axes
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="AV2 LiDAR -> metric .ply (+XYZ axes)")
    p.add_argument("--av2-root", type=Path, required=True)
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--log-id", type=str, default=None)
    p.add_argument("--scene-id", type=int, default=0)
    p.add_argument("--list", action="store_true")
    p.add_argument("--frame-idx", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=1)
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--coord", choices=["ego", "city"], default="ego")
    p.add_argument("--max-range", type=float, default=80.0)
    p.add_argument("--min-range", type=float, default=0.5)
    p.add_argument("--axes", action="store_true", default=True,
                   help="加上 XYZ 座標軸 (預設開)")
    p.add_argument("--no-axes", dest="axes", action="store_false")
    p.add_argument("--axis-length", type=float, default=5.0,
                   help="座標軸長度 (m)")
    p.add_argument("--axis-density", type=int, default=500,
                   help="每條軸用幾個點畫")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def resolve_log_dir(av2_root, split, log_id, scene_id):
    split_root = av2_root / split
    if not split_root.is_dir():
        split_root = av2_root
    logs = sorted(d for d in split_root.iterdir() if d.is_dir() and (d / "sensors").is_dir())
    if not logs:
        raise FileNotFoundError(f"{split_root} 下找不到含 sensors/ 的 log")
    if log_id:
        for d in logs:
            if d.name == log_id:
                return d
        raise FileNotFoundError(f"找不到 log_id={log_id}")
    if scene_id < 0 or scene_id >= len(logs):
        raise IndexError(f"scene-id={scene_id} 超出範圍 [0,{len(logs)-1}]")
    return logs[scene_id]


def list_lidar_sweeps(log_dir):
    d = log_dir / "sensors" / "lidar"
    files = sorted(d.glob("*.feather"), key=lambda p: int(p.stem))
    if not files:
        raise FileNotFoundError(f"{d} 內沒有 .feather")
    return files


def read_sweep(path):
    df = pd.read_feather(path)
    return df[["x", "y", "z"]].to_numpy(dtype=np.float64)


def ego_to_city_T(log_dir, ts_ns):
    pose_file = log_dir / "city_SE3_egovehicle.feather"
    if not pose_file.is_file():
        return None
    df = pd.read_feather(pose_file)
    row = df.iloc[(df["timestamp_ns"] - ts_ns).abs().argmin()]
    from scipy.spatial.transform import Rotation
    q_xyzw = [row["qx"], row["qy"], row["qz"], row["qw"]]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
    T[:3, 3] = [row["tx_m"], row["ty_m"], row["tz_m"]]
    return T


def range_filter(xyz, lo, hi):
    d = np.linalg.norm(xyz, axis=1)
    return xyz[(d > lo) & (d < hi)]


def height_color(xyz):
    z = xyz[:, 2]
    lo, hi = np.percentile(z, [2, 98])
    t = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.stack([t, np.zeros_like(t), 1 - t], axis=1)


def make_axes(length, density):
    """產生 XYZ 三軸的點: X=紅, Y=綠, Z=藍。回傳 (pts, cols)。"""
    t = np.linspace(0, length, density).reshape(-1, 1)
    zero = np.zeros_like(t)
    x_axis = np.hstack([t, zero, zero]);  x_col = np.tile([1, 0, 0], (density, 1))
    y_axis = np.hstack([zero, t, zero]);  y_col = np.tile([0, 1, 0], (density, 1))
    z_axis = np.hstack([zero, zero, t]);  z_col = np.tile([0, 0, 1], (density, 1))
    pts  = np.vstack([x_axis, y_axis, z_axis])
    cols = np.vstack([x_col, y_col, z_col]).astype(np.float64)
    return pts, cols


def write_ply(path, xyz, rgb):
    rgb8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(xyz)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        verts = np.empty(len(xyz), dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ])
        verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        verts["red"], verts["green"], verts["blue"] = rgb8[:, 0], rgb8[:, 1], rgb8[:, 2]
        f.write(verts.tobytes())


def main():
    a = parse_args()
    av2_root = a.av2_root.expanduser().resolve()

    if a.list:
        split_root = av2_root / a.split
        if not split_root.is_dir():
            split_root = av2_root
        logs = sorted(d.name for d in split_root.iterdir()
                      if d.is_dir() and (d / "sensors").is_dir())
        print(f"{a.split} 下 {len(logs)} 個 log:")
        for i, name in enumerate(logs):
            print(f"  [{i}] {name}")
        return 0

    log_dir = resolve_log_dir(av2_root, a.split, a.log_id, a.scene_id)
    print(f"[log] {log_dir.name}")
    sweeps = list_lidar_sweeps(log_dir)
    print(f"[lidar] {len(sweeps)} sweeps")

    if a.num_frames > 1 and a.coord == "ego":
        print("[warn] 多 frame 用 ego 座標會錯位，建議 --coord city")

    all_xyz, all_col = [], []
    for off in range(0, a.num_frames * a.frame_step, a.frame_step):
        idx = a.frame_idx + off
        if idx >= len(sweeps):
            break
        sweep = sweeps[idx]
        ts_ns = int(sweep.stem)
        xyz = range_filter(read_sweep(sweep), a.min_range, a.max_range)
        print(f"  [sweep {idx}] {sweep.name}  {len(xyz)} pts")

        if a.coord == "city":
            T = ego_to_city_T(log_dir, ts_ns)
            if T is None:
                print("  [warn] 無 city pose，留在 ego 座標")
            else:
                h = np.hstack([xyz, np.ones((len(xyz), 1))])
                xyz = (T @ h.T)[:3].T

        all_xyz.append(xyz)
        all_col.append(height_color(xyz))

    xyz = np.concatenate(all_xyz)
    col = np.concatenate(all_col)

    if a.axes:
        ax_pts, ax_col = make_axes(a.axis_length, a.axis_density)
        xyz = np.vstack([xyz, ax_pts])
        col = np.vstack([col, ax_col])
        print(f"[axes] 加上 XYZ 軸 (X=紅 Y=綠 Z=藍), 長度 {a.axis_length}m")

    print(f"[total] {len(xyz)} pts")
    out = a.out or Path(f"av2_lidar_{log_dir.name[:8]}_f{a.frame_idx}_{a.coord}.ply")
    write_ply(out, xyz, col)
    print(f"[done] saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())