from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def scale_center_crop_pil(
    image: Image.Image,
    target_h: int,
    target_w: int,
) -> tuple[Image.Image, dict]:
    image = image.convert("RGB")
    src_w, src_h = image.size
    scale = max(float(target_w) / float(src_w), float(target_h) / float(src_h))
    rs_w = max(int(round(src_w * scale)), target_w)
    rs_h = max(int(round(src_h * scale)), target_h)
    resized = image.resize((rs_w, rs_h), Image.Resampling.LANCZOS)

    left = max((rs_w - target_w) // 2, 0)
    top = max((rs_h - target_h) // 2, 0)
    crop = resized.crop((left, top, left + target_w, top + target_h))
    meta = {
        "orig_hw": [src_h, src_w],
        "resized_hw": [rs_h, rs_w],
        "crop_xy": [left, top],
        "target_hw": [target_h, target_w],
        "scale": float(scale),
    }
    return crop, meta


def _resize_paths_to_target(
    image_paths: list[Path],
    out_dir: Path,
    target_hw: tuple[int, int],
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"target_hw": [int(target_hw[0]), int(target_hw[1])], "frames": []}
    for idx, path in enumerate(image_paths):
        img = Image.open(path).convert("RGB")
        crop, frame_meta = scale_center_crop_pil(img, int(target_hw[0]), int(target_hw[1]))
        frame_out = out_dir / f"frame_{idx:02d}.png"
        crop.save(frame_out)
        frame_meta["input_path"] = str(path)
        frame_meta["output_path"] = str(frame_out)
        meta["frames"].append(frame_meta)
    return meta


def write_resized_inputs(
    image_paths: list[Path],
    out_dir: Path,
    target_hw: tuple[int, int],
) -> dict:
    return _resize_paths_to_target(image_paths, out_dir, target_hw)


def load_images_scale_center_crop(
    image_paths: list[Path],
    target_hw: tuple[int, int],
):
    from vggt.utils.load_fn import load_and_preprocess_images

    prep_dir = Path(".cache") / "vggt_encode_inputs"
    preprocess_meta = _resize_paths_to_target(image_paths, prep_dir, target_hw)
    proc_paths = [frame["output_path"] for frame in preprocess_meta["frames"]]
    images_t = load_and_preprocess_images(proc_paths)
    vggt_hw = (int(images_t.shape[-2]), int(images_t.shape[-1]))
    preprocess_meta["vggt_hw"] = [int(vggt_hw[0]), int(vggt_hw[1])]
    preprocess_meta["num_frames"] = len(proc_paths)
    return images_t, vggt_hw, preprocess_meta


def load_vggt(checkpoint: Path, device: torch.device):
    from vggt.models.vggt import VGGT

    ckpt = str(checkpoint)
    if checkpoint.exists():
        model = VGGT.from_pretrained(ckpt)
    else:
        # Fallback to HF id expected by VGGT.
        model = VGGT.from_pretrained("facebook/VGGT-1B")
    return model.to(device).eval()


def _to_cam2world(extrinsic: np.ndarray) -> np.ndarray:
    # VGGT extrinsic is world-to-camera; invert to camera-to-world.
    extr = extrinsic.astype(np.float64)
    if extr.ndim == 3 and extr.shape[-2:] == (3, 4):
        n = extr.shape[0]
        hom = np.repeat(np.eye(4, dtype=np.float64)[None, ...], n, axis=0)
        hom[:, :3, :4] = extr
        extr = hom
    elif extr.ndim == 2 and extr.shape == (3, 4):
        hom = np.eye(4, dtype=np.float64)
        hom[:3, :4] = extr
        extr = hom
    return np.linalg.inv(extr).astype(np.float32)


def run_vggt_forward(model, images_t: torch.Tensor, device: torch.device) -> dict:
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    images = images_t.to(device)
    if device.type == "cuda":
        major = torch.cuda.get_device_capability()[0]
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        amp_dtype = torch.float32

    with torch.no_grad():
        if device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                pred = model(images)
        else:
            pred = model(images.float())

    proc_hw = (int(images.shape[-2]), int(images.shape[-1]))
    extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], proc_hw)
    extr = extr[0].float().cpu().numpy()
    intr = intr[0].float().cpu().numpy()
    cam2world = _to_cam2world(extr)

    return {
        "proc_hw": proc_hw,
        "extrinsic": extr,
        "intrinsic": intr,
        "cam2world": cam2world,
        "depth": pred["depth"].squeeze(0).squeeze(-1).float().cpu().numpy(),
        "depth_conf": pred["depth_conf"].squeeze(0).float().cpu().numpy(),
        "world_points": pred["world_points"].squeeze(0).float().cpu().numpy(),
        "world_points_conf": pred["world_points_conf"].squeeze(0).float().cpu().numpy(),
        "images": pred["images"].squeeze(0).float().cpu().numpy(),
    }


def _resize_world_points(world_points: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    # (N, H, W, 3) -> resize each xyz channel bilinearly
    wp = torch.from_numpy(world_points).permute(0, 3, 1, 2).float()
    wp = F.interpolate(wp, size=(int(target_hw[0]), int(target_hw[1])), mode="bilinear", align_corners=False)
    return wp.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)


def _resize_conf(conf: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    c = torch.from_numpy(conf).float()
    if c.ndim == 4 and c.shape[-1] == 1:
        c = c[..., 0]
    c = c.unsqueeze(1)
    c = F.interpolate(c, size=(int(target_hw[0]), int(target_hw[1])), mode="bilinear", align_corners=False)
    return c.squeeze(1).cpu().numpy().astype(np.float32)


def save_encode_bundle(
    raw_dir: Path,
    bundle: dict,
    target_hw: tuple[int, int],
    image_paths: list[Path],
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    world_points = np.asarray(bundle["world_points"], dtype=np.float32)
    world_conf = np.asarray(bundle["world_points_conf"], dtype=np.float32)

    wp_at = _resize_world_points(world_points, target_hw)
    wc_at = _resize_conf(world_conf, target_hw)

    np.save(raw_dir / "world_points_at_target_hw.npy", wp_at)
    np.save(raw_dir / "world_points_conf_at_target_hw.npy", wc_at)
    np.save(raw_dir / "cam2world.npy", np.asarray(bundle["cam2world"], dtype=np.float32))
    np.save(raw_dir / "intrinsic.npy", np.asarray(bundle["intrinsic"], dtype=np.float32))
    np.save(raw_dir / "extrinsic.npy", np.asarray(bundle["extrinsic"], dtype=np.float32))
    np.save(raw_dir / "depth.npy", np.asarray(bundle["depth"], dtype=np.float32))
    np.save(raw_dir / "depth_conf.npy", np.asarray(bundle["depth_conf"], dtype=np.float32))

    meta = {
        "target_hw": [int(target_hw[0]), int(target_hw[1])],
        "proc_hw": [int(bundle["proc_hw"][0]), int(bundle["proc_hw"][1])],
        "num_views": int(len(image_paths)),
        "image_paths": [str(p) for p in image_paths],
    }
    with (raw_dir / "bundle_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
