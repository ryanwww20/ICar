#!/usr/bin/env python3
"""
用 VGGT 估計的相機 pose 從點雲 render 截圖，跟原始照片比對視角。

原理:
  VGGT cameras.json 的 extrinsic = world->cam (3x4 [R|t])、intrinsic = 518x518。
  把這組灌進 Open3D 的虛擬相機，render 出來的角度 = 那台相機看到的角度。
  → 跟該相機的原始照片並排，就能看重建是否對位。

依賴: pip install open3d numpy
跑:
  # 截單一相機 (index 0 = ring_front_center)
  python render_from_camera.py --ply vggt.ply --cameras cameras.json --cam 0
  # 七路全截
  python render_from_camera.py --ply vggt.ply --cameras cameras.json --all
"""

import argparse
import json
import numpy as np
import open3d as o3d

RING = ["front_center", "front_left", "front_right",
        "side_left", "side_right", "rear_left", "rear_right"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ply", required=True, help="VGGT 重建點雲")
    p.add_argument("--cameras", required=True, help="VGGT cameras.json")
    p.add_argument("--cam", type=int, default=0, help="相機 index (0~6)")
    p.add_argument("--all", action="store_true", help="七路全截")
    p.add_argument("--out-prefix", type=str, default="render_cam",
                   help="輸出檔名前綴")
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--bg", type=float, default=1.0, help="背景灰階 (1=白 0=黑)")
    return p.parse_args()


def render_one(pcd, extr_3x4, intr_3x3, hw, out_path, point_size, bg):
    """用一台相機的 extrinsic/intrinsic render 點雲到 out_path。"""
    H, W = hw

    # Open3D 的 intrinsic 需要 fx,fy,cx,cy；cx/cy 必須符合 (W-1)/2 附近的限制，
    # 否則 set 會失敗，這裡用相機原始值但 clamp 到合法範圍。
    fx, fy = intr_3x3[0, 0], intr_3x3[1, 1]
    cx, cy = intr_3x3[0, 2], intr_3x3[1, 2]

    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(W, H, fx, fy, cx, cy)

    extr_4x4 = np.eye(4)
    extr_4x4[:3, :4] = extr_3x4   # world->cam

    cam_param = o3d.camera.PinholeCameraParameters()
    cam_param.intrinsic = intr
    cam_param.extrinsic = extr_4x4

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=H, visible=False)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array([bg, bg, bg])

    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(cam_param, allow_arbitrary=True)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(out_path, do_render=True)
    vis.destroy_window()
    print(f"  saved -> {out_path}")


def main():
    a = parse_args()
    pcd = o3d.io.read_point_cloud(a.ply)
    print(f"[load] {len(pcd.points)} 點")

    cams = json.load(open(a.cameras))
    extr = np.array(cams["extrinsic"])   # (N,3,4)
    intr = np.array(cams["intrinsic"])   # (N,3,3)
    H, W = cams["proc_hw"]

    idxs = range(len(extr)) if a.all else [a.cam]
    for i in idxs:
        name = RING[i] if i < len(RING) else f"cam{i}"
        out = f"{a.out_prefix}_{i:02d}_{name}.png"
        print(f"[render] cam {i} ({name})")
        render_one(pcd, extr[i], intr[i], (H, W), out, a.point_size, a.bg)

    print("[done] 跟原始照片並排比對視角")


if __name__ == "__main__":
    main()