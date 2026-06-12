#!/usr/bin/env python3
"""
用 Open3D 看 ply。
跑: python view_ply.py 檔案.ply
"""

import sys
import open3d as o3d


def main():
    if len(sys.argv) < 2:
        print("用法: python view_ply.py <檔案.ply>")
        sys.exit(1)

    path = sys.argv[1]
    pcd = o3d.io.read_point_cloud(path)
    n = len(pcd.points)
    if n == 0:
        print(f"讀不到點: {path}")
        sys.exit(1)
    print(f"{path}: {n} 點")
    print("操作: 左鍵拖曳=轉  滾輪=縮放  右鍵拖曳=平移  Q=關閉")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=path,
        width=1280,
        height=800,
        point_show_normal=False,
    )


if __name__ == "__main__":
    main()