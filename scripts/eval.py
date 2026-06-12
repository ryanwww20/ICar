#!/usr/bin/env python3
"""
比較 VGGT 重建點雲 vs LiDAR ground-truth，輸出幾何 evaluation 指標。

流程:
  1. 讀兩個 ply、downsample、去離群
  2. (relative 版) 把 VGGT 相機座標 (X右Y下Z前) 轉到 ego 座標 (X前Y左Z上)
  3. 用 bbox 對角線估初始 scale
  4. ICP 精對齊
  5. 算 point-to-point 距離 / Chamfer / 覆蓋率
  6. 輸出疊合 ply (pred=紅, gt=灰) 供 viewer 檢查

依賴: pip install open3d numpy
跑:
  python eval_ply.py --pred vggt.ply --gt lidar.ply
  python eval_ply.py --pred vggt.ply --gt lidar.ply --no-cam-to-ego   # metric 版(已同座標)用這個
"""

import argparse
import numpy as np
import open3d as o3d


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True, help="VGGT 重建 ply")
    p.add_argument("--gt", required=True, help="LiDAR ground-truth ply")
    p.add_argument("--pred-voxel", type=float, default=0.05,
                   help="pred downsample (relative 版尺度小，用小值)")
    p.add_argument("--gt-voxel", type=float, default=0.3)
    p.add_argument("--cam-to-ego", dest="cam_to_ego", action="store_true", default=True,
                   help="把 VGGT 相機座標轉到 ego 座標 (relative 版需要，預設開)")
    p.add_argument("--no-cam-to-ego", dest="cam_to_ego", action="store_false",
                   help="metric 版已在同座標系，關掉座標轉換")
    p.add_argument("--auto-scale", dest="auto_scale", action="store_true", default=True,
                   help="用 bbox 對角線估 scale (relative 版需要，預設開)")
    p.add_argument("--no-auto-scale", dest="auto_scale", action="store_false")
    p.add_argument("--icp-thresh", type=float, default=5.0,
                   help="ICP 對應點最大距離 (m)")
    p.add_argument("--out", type=str, default="overlay.ply",
                   help="疊合輸出 (pred=紅, gt=灰)")
    return p.parse_args()


def load(path, voxel):
    pcd = o3d.io.read_point_cloud(path)
    pcd = pcd.voxel_down_sample(voxel)
    pcd, _ = pcd.remove_statistical_outlier(20, 2.0)
    return pcd


def diag(pcd):
    e = np.asarray(pcd.points)
    return np.linalg.norm(e.max(0) - e.min(0))


def cam_to_ego(pcd):
    """VGGT 相機座標 (X右 Y下 Z前) -> ego (X前 Y左 Z上)."""
    P = np.asarray(pcd.points)
    P2 = np.stack([P[:, 2], -P[:, 0], -P[:, 1]], axis=1)
    pcd.points = o3d.utility.Vector3dVector(P2)
    return pcd


def main():
    a = parse_args()
    pred = load(a.pred, a.pred_voxel)
    gt = load(a.gt, a.gt_voxel)
    print(f"[load] pred={len(pred.points)}  gt={len(gt.points)}")

    if a.cam_to_ego:
        pred = cam_to_ego(pred)
        print("[coord] VGGT 相機座標 -> ego 座標")

    if a.auto_scale:
        s = diag(gt) / max(diag(pred), 1e-9)
        pred.scale(s, center=pred.get_center())
        print(f"[scale] init scale = {s:.3f}")

    pred.translate(-pred.get_center())
    gt.translate(-gt.get_center())

    reg = o3d.pipelines.registration.registration_icp(
        pred, gt, a.icp_thresh, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    pred.transform(reg.transformation)
    print(f"[icp] fitness={reg.fitness:.3f}  inlier_rmse={reg.inlier_rmse:.3f} m")

    d_pg = np.asarray(pred.compute_point_cloud_distance(gt))
    d_gp = np.asarray(gt.compute_point_cloud_distance(pred))

    print("\n=== 幾何誤差 (對齊後, 單位 m) ===")
    print(f"  pred->gt  mean={d_pg.mean():.3f}  median={np.median(d_pg):.3f}  "
          f"rmse={np.sqrt((d_pg**2).mean()):.3f}")
    print(f"  gt->pred  mean={d_gp.mean():.3f}  median={np.median(d_gp):.3f}")
    print(f"  Chamfer   = {d_pg.mean() + d_gp.mean():.3f} m")
    print("\n=== 覆蓋率 (pred 點落在 gt 附近的比例) ===")
    for t in [0.5, 1.0, 2.0, 5.0]:
        print(f"  {t:>4.1f} m 內: {(d_pg < t).mean() * 100:5.1f}%")

    pred.paint_uniform_color([1, 0, 0])      # 紅 = 重建
    gt.paint_uniform_color([0.6, 0.6, 0.6])  # 灰 = LiDAR
    o3d.io.write_point_cloud(a.out, pred + gt)
    print(f"\n[done] 疊合 -> {a.out} (紅=VGGT, 灰=LiDAR)")


if __name__ == "__main__":
    main()