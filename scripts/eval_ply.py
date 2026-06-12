#!/usr/bin/env python3
"""
用 VGGT 重建的顏色染色 LiDAR 點雲。

原理:
  1. 跟 eval_ply.py 一樣，用座標轉換 + scale + ICP 把 VGGT 對齊到 LiDAR。
  2. ICP 算出的 transform 把兩者接到同座標。
  3. 對每個 LiDAR 點，找對齊後最近的 VGGT 點，抄它的顏色。
     超過距離閾值 (預設 2m) 的點 → 標灰 (代表無顏色資訊)。
輸出: 點分布 = LiDAR (位置準)，顏色 = VGGT 投影過來。

依賴: pip install open3d numpy
跑:
  python colorize_lidar.py --pred vggt.ply --gt lidar.ply --out colored.ply
"""

import argparse
import numpy as np
import open3d as o3d


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True, help="VGGT 重建 ply (提供顏色)")
    p.add_argument("--gt", required=True, help="LiDAR ply (提供點分布)")
    p.add_argument("--pred-voxel", type=float, default=0.05)
    p.add_argument("--gt-voxel", type=float, default=0.0,
                   help="LiDAR downsample；0=不抽稀(保留完整點分布)")
    p.add_argument("--cam-to-ego", dest="cam_to_ego", action="store_true", default=True)
    p.add_argument("--no-cam-to-ego", dest="cam_to_ego", action="store_false")
    p.add_argument("--auto-scale", dest="auto_scale", action="store_true", default=True)
    p.add_argument("--no-auto-scale", dest="auto_scale", action="store_false")
    p.add_argument("--icp-thresh", type=float, default=5.0)
    p.add_argument("--color-thresh", type=float, default=2.0,
                   help="LiDAR 點離最近 VGGT 點超過此距離 → 不染色(標灰)")
    p.add_argument("--gray", type=float, default=0.5, help="無顏色點的灰階值")
    p.add_argument("--out", type=str, default="colored_lidar.ply")
    return p.parse_args()


def load(path, voxel):
    pcd = o3d.io.read_point_cloud(path)
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    return pcd


def diag(pcd):
    e = np.asarray(pcd.points)
    return np.linalg.norm(e.max(0) - e.min(0))


def cam_to_ego(pcd):
    P = np.asarray(pcd.points)
    pcd.points = o3d.utility.Vector3dVector(
        np.stack([P[:, 2], -P[:, 0], -P[:, 1]], axis=1))
    return pcd


def main():
    a = parse_args()

    # pred 清理離群(顏色來源要乾淨)；gt 保留完整點分布
    pred = load(a.pred, a.pred_voxel)
    pred, _ = pred.remove_statistical_outlier(20, 2.0)
    gt = load(a.gt, a.gt_voxel)
    print(f"[load] pred={len(pred.points)}  gt(LiDAR)={len(gt.points)}")

    if not pred.has_colors():
        raise SystemExit("ERROR: pred 沒有顏色，無法染色")

    # --- 對齊 (跟 eval_ply 相同流程) ---
    if a.cam_to_ego:
        pred = cam_to_ego(pred)
    if a.auto_scale:
        s = diag(gt) / max(diag(pred), 1e-9)
        pred.scale(s, center=pred.get_center())
        print(f"[scale] {s:.3f}")

    # 注意：gt 不平移，保留原始 LiDAR 座標當輸出骨架。
    #       只把 pred 對齊到 gt 的原始位置。
    pred_c = pred.get_center()
    gt_c = np.asarray(gt.points).mean(0)
    pred.translate(gt_c - pred_c)   # 先粗對到 gt 中心

    reg = o3d.pipelines.registration.registration_icp(
        pred, gt, a.icp_thresh, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    pred.transform(reg.transformation)
    print(f"[icp] fitness={reg.fitness:.3f}  inlier_rmse={reg.inlier_rmse:.3f} m")

    # --- 染色: 每個 LiDAR 點找最近 VGGT 點抄色 ---
    pred_pts = np.asarray(pred.points)
    pred_cols = np.asarray(pred.colors)
    gt_pts = np.asarray(gt.points)

    kdt = o3d.geometry.KDTreeFlann(pred)
    out_cols = np.full((len(gt_pts), 3), a.gray, dtype=np.float64)
    colored = 0
    for i, p in enumerate(gt_pts):
        _, idx, dist2 = kdt.search_knn_vector_3d(p, 1)
        if dist2[0] <= a.color_thresh ** 2:
            out_cols[i] = pred_cols[idx[0]]
            colored += 1

    pct = colored / len(gt_pts) * 100
    print(f"[colorize] {colored}/{len(gt_pts)} ({pct:.1f}%) 點染到色 "
          f"(閾值 {a.color_thresh}m)，其餘標灰")

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(gt_pts)
    out.colors = o3d.utility.Vector3dVector(out_cols)
    o3d.io.write_point_cloud(a.out, out)
    print(f"[done] -> {a.out}")


if __name__ == "__main__":
    main()