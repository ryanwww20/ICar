#!/usr/bin/env python3
"""Run DUSt3R inference on AV2 image pairs exported by av2_make_pairs.py."""

from __future__ import annotations

import os

# Headless servers often lack libGL.so.1; use non-interactive backends before
# importing OpenCV/matplotlib-dependent DUSt3R modules.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "0")

import argparse
import json
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DUST3R_ROOT = REPO_ROOT / "dust3r"

if str(DUST3R_ROOT) not in sys.path:
    sys.path.insert(0, str(DUST3R_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DUSt3R on AV2 image pairs listed in pairs.json."
    )
    parser.add_argument("--pairs-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/av2_dust3r"))
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--save-pointcloud", action="store_true", default=True)
    parser.add_argument("--no-save-pointcloud", action="store_false", dest="save_pointcloud")
    parser.add_argument("--save-depth", action="store_true", default=True)
    parser.add_argument("--no-save-depth", action="store_false", dest="save_depth")
    parser.add_argument("--save-camera-json", action="store_true", default=True)
    parser.add_argument("--no-save-camera-json", action="store_false", dest="save_camera_json")
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()


DEFAULT_HF_MODEL = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
DEFAULT_LOCAL_CHECKPOINT = DUST3R_ROOT / "checkpoints" / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
CHECKPOINT_URL = (
    "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/"
    "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
)


def resolve_model_path(model_path: str | None) -> str:
    if model_path is None:
        if DEFAULT_LOCAL_CHECKPOINT.is_file() and DEFAULT_LOCAL_CHECKPOINT.stat().st_size > 0:
            return str(DEFAULT_LOCAL_CHECKPOINT.resolve())
        return DEFAULT_HF_MODEL

    path = Path(model_path).expanduser()
    candidates: list[Path] = []

    def add(candidate: Path) -> None:
        resolved = candidate.expanduser()
        if resolved not in candidates:
            candidates.append(resolved)

    if path.is_absolute():
        add(path)
    else:
        add(Path.cwd() / path)
        add(REPO_ROOT / path)
        if path.parts and path.parts[0] == "dust3r" and len(path.parts) > 1:
            add(DUST3R_ROOT / Path(*path.parts[1:]))
        add(DUST3R_ROOT / path)
        add(DUST3R_ROOT / "checkpoints" / path.name)
        add(path)

    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return str(candidate.resolve())

    if path.suffix == ".pth":
        searched = ", ".join(str(c.resolve()) for c in candidates)
        raise FileNotFoundError(
            "Local checkpoint not found.\n"
            f"  Requested: {model_path}\n"
            f"  Searched: {searched}\n"
            "Download it with:\n"
            f"  mkdir -p {DEFAULT_LOCAL_CHECKPOINT.parent}\n"
            f"  wget {CHECKPOINT_URL} -O {DEFAULT_LOCAL_CHECKPOINT}"
        )

    return model_path


def load_pairs_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "pairs" in payload:
        return payload["pairs"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unexpected pairs.json format: {path}")


def resolve_pair_images(pair: dict) -> tuple[Path, Path]:
    image1 = Path(pair.get("copied_image1") or pair["image1"]).expanduser().resolve()
    image2 = Path(pair.get("copied_image2") or pair["image2"]).expanduser().resolve()
    return image1, image2


def save_depth_png(depth: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth)
    if valid.any():
        vmin = float(np.min(depth[valid]))
        vmax = float(np.max(depth[valid]))
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    plt.imsave(path, depth, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.close("all")


def save_pointcloud_ply(scene, path: Path, min_conf_thr: float = 3.0) -> None:
    import torch
    import trimesh
    from dust3r.utils.device import to_numpy

    rgbimg = scene.imgs
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    mask = to_numpy(scene.get_masks())

    pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
    col = np.concatenate([p[m] for p, m in zip(rgbimg, mask)])
    if pts.size == 0:
        raise RuntimeError("Empty point cloud after confidence filtering")

    pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
    pct.export(path)


def save_cameras_json(scene, path: Path) -> None:
    from dust3r.utils.device import to_numpy

    focals = to_numpy(scene.get_focals())
    poses = to_numpy(scene.get_im_poses())
    principal_points = to_numpy(scene.get_principal_points())

    cameras = []
    for idx, (focal, pose, pp) in enumerate(zip(focals, poses, principal_points)):
        cameras.append(
            {
                "index": idx,
                "focal": float(focal),
                "principal_point": [float(pp[0]), float(pp[1])],
                "cam_to_world": pose.tolist(),
            }
        )

    with path.open("w", encoding="utf-8") as f:
        json.dump({"cameras": cameras}, f, indent=2)


def run_dust3r_pair(
    image1: Path,
    image2: Path,
    model,
    device: str,
    batch_size: int,
    image_size: int,
):
    import copy

    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images

    try:
        square_ok = model.square_ok
    except Exception:
        square_ok = False

    imgs = load_images(
        [str(image1), str(image2)],
        size=image_size,
        verbose=False,
        patch_size=model.patch_size,
        square_ok=square_ok,
    )
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]["idx"] = 1

    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size, verbose=False)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer, verbose=False)
    return scene


def process_pair(
    pair_idx: int,
    pair: dict,
    args: argparse.Namespace,
    model,
) -> bool:
    image1, image2 = resolve_pair_images(pair)
    pair_dir = args.output_dir / f"pair_{pair_idx:06d}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    for src, name in ((image1, "image_0.jpg"), (image2, "image_1.jpg")):
        if not src.exists():
            raise FileNotFoundError(f"Missing image: {src}")
        shutil.copy2(src, pair_dir / name)

    dust3r_config = {
        "model_path": args.model_path,
        "device": args.device,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "save_pointcloud": args.save_pointcloud,
        "save_depth": args.save_depth,
        "save_camera_json": args.save_camera_json,
    }

    metadata = {
        "pair_index": pair_idx,
        "log_id": pair.get("log_id"),
        "pair_type": pair.get("pair_type"),
        "camera1": pair.get("camera1"),
        "camera2": pair.get("camera2"),
        "image1": str(image1),
        "image2": str(image2),
        "timestamp1": pair.get("timestamp1"),
        "timestamp2": pair.get("timestamp2"),
        "dust3r_config": dust3r_config,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "status": "success",
    }

    if args.dry_run:
        metadata["status"] = "dry_run"
        with (pair_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"[dry-run] pair_{pair_idx:06d}: {image1.name} + {image2.name}")
        return True

    scene = run_dust3r_pair(
        image1=image1,
        image2=image2,
        model=model,
        device=args.device,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    if args.save_depth:
        from dust3r.utils.device import to_numpy

        depths = to_numpy(scene.get_depthmaps())
        for idx, depth in enumerate(depths):
            save_depth_png(depth, pair_dir / f"depth_{idx}.png")

    if args.save_pointcloud:
        save_pointcloud_ply(scene, pair_dir / "pointcloud.ply")

    if args.save_camera_json:
        save_cameras_json(scene, pair_dir / "cameras.json")

    with (pair_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"OK pair_{pair_idx:06d}: {image1.name} + {image2.name}")
    return True


def main() -> int:
    args = parse_args()
    args.pairs_json = args.pairs_json.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.pairs_json.exists():
        print(f"ERROR: pairs.json not found: {args.pairs_json}", file=sys.stderr)
        return 1

    pairs = load_pairs_json(args.pairs_json)[: args.max_pairs]
    if not pairs:
        print("ERROR: No pairs found in pairs.json", file=sys.stderr)
        return 1

    model = None
    if not args.dry_run:
        import torch
        from dust3r.model import AsymmetricCroCo3DStereo

        model_path = resolve_model_path(args.model_path)
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            print(
                "WARNING: CUDA not available, falling back to CPU.",
                file=sys.stderr,
            )
            args.device = "cpu"

        print(f"Loading DUSt3R model: {model_path}")
        model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(args.device)
        model.eval()

    success = 0
    failed = 0

    for pair_idx, pair in enumerate(pairs):
        pair_dir = args.output_dir / f"pair_{pair_idx:06d}"
        try:
            process_pair(pair_idx, pair, args, model)
            success += 1
        except Exception as exc:
            failed += 1
            pair_dir.mkdir(parents=True, exist_ok=True)
            error_path = pair_dir / "error.txt"
            error_path.write_text(
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
                encoding="utf-8",
            )
            print(f"FAIL pair_{pair_idx:06d}: {exc}", file=sys.stderr)

    print(f"Finished. success={success}, failed={failed}, output={args.output_dir}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
