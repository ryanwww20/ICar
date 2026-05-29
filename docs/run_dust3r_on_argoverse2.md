# Running DUSt3R on Argoverse 2 Sensor Dataset

This guide describes how to use the AV2 → DUSt3R pipeline in this repo (`ICar` integration with official DUSt3R under `dust3r/`).

## 1. Goal

Use **high-overlap camera pairs** from the **Argoverse 2 Sensor Dataset** as input to **DUSt3R** for 3D reconstruction.

Priority order:

1. **Stereo front-left / front-right** pairs (best overlap for DUSt3R)
2. **Temporal** pairs from the same camera (controlled baseline via `--stride`)
3. **Ring-adjacent** pairs or manually specified cameras

The workflow is intentionally staged:

| Stage | Script | Purpose |
|-------|--------|---------|
| 1 | `av2_explore.py` | Inspect dataset layout and camera names |
| 2 | `av2_make_pairs.py` | Export `pairs.json` / `pairs.txt` |
| 3 | `run_dust3r_on_av2_pairs.py` | Run reconstruction on a small batch |

Start with a **small partial download** of AV2 — do not download the full dataset initially.

---

## 2. Environment setup

### DUSt3R dependencies

```bash
cd dust3r
pip install -r requirements.txt

# Optional but recommended for faster inference:
cd croco/models/curope
python setup.py build_ext --inplace
cd ../../../
```

### Download DUSt3R weights

```bash
mkdir -p dust3r/checkpoints
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  -P dust3r/checkpoints/
```

Or let HuggingFace Hub download automatically on first run (`naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt`).

### Optional dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `av2` (av2-api) | Structured calibration access | `pip install av2` |
| `s5cmd` | Fast S3 download for AV2 | [s5cmd releases](https://github.com/peak/s5cmd) |

---

## 3. Dataset layout

Place your manually downloaded AV2 Sensor Dataset under a local root, for example:

```
data/av2/sensor/
  val/
    <log_id>/
      sensors/
        cameras/
          ring_front_center/
            *.jpg
          ring_front_left/
            *.jpg
          stereo_front_left/
            *.jpg
          stereo_front_right/
            *.jpg
      calibration/
        intrinsics.feather
        extrinsics.feather
  train/
    ...
  test/
    ...
```

The scripts tolerate variations (`cameras/` vs `sensors/cameras/`, `.jpg` vs `.png`). Always run the explorer first to confirm your layout.

### Downloading AV2 (partial)

Refer to the [Argoverse 2 download page](https://www.argoverse.org/av2.html#download-link). For Sensor Dataset, you typically download specific log ids for a split rather than the full corpus.

With `s5cmd` (example — adjust to your target logs):

```bash
# Example pattern; see official AV2 docs for current S3 paths
s5cmd cp "s3://argoverse/datasets/av2/sensor/val/<log_id>/*" data/av2/sensor/val/<log_id>/
```

---

## 4. Explore the dataset (Stage 1)

From repo root:

```bash
python scripts/av2_explore.py \
  --av2-root data/av2/sensor \
  --split val \
  --max-logs 3
```

Example output:

```
Log: abc123...
  Cameras:
    stereo_front_left: 315 images
    stereo_front_right: 315 images
    ring_front_center: 315 images
  Candidate stereo pairs:
    stereo_front_left <-> stereo_front_right
```

If no stereo pair is detected, the script lists all camera folders — use those names in Stage 2.

---

## 5. Create stereo pairs (Stage 2)

```bash
python scripts/av2_make_pairs.py \
  --av2-root data/av2/sensor \
  --split val \
  --pair-type stereo \
  --output-dir data/av2_pairs \
  --max-logs 2 \
  --max-pairs-per-log 10
```

Outputs:

- `data/av2_pairs/pairs.json` — full metadata (absolute paths)
- `data/av2_pairs/pairs.txt` — `image1 image2` per line

### Manual camera override

If auto-detection fails:

```bash
python scripts/av2_make_pairs.py \
  --av2-root data/av2/sensor \
  --split val \
  --pair-type stereo \
  --left-camera stereo_front_left \
  --right-camera stereo_front_right \
  --output-dir data/av2_pairs
```

### Temporal pairs (fallback)

Useful when stereo naming is unclear or overlap is still high within one camera:

```bash
python scripts/av2_make_pairs.py \
  --av2-root data/av2/sensor \
  --split val \
  --pair-type temporal \
  --stride 1 \
  --max-pairs-per-log 20 \
  --output-dir data/av2_pairs_temporal
```

### Optional: copy and resize images

```bash
python scripts/av2_make_pairs.py \
  --av2-root data/av2/sensor \
  --split val \
  --pair-type stereo \
  --copy-images \
  --resize-long-edge 512 \
  --output-dir data/av2_pairs_512
```

---

## 6. Run DUSt3R (Stage 3)

Dry-run first (no model load):

```bash
python scripts/run_dust3r_on_av2_pairs.py \
  --pairs-json data/av2_pairs/pairs.json \
  --output-dir outputs/av2_dust3r \
  --dry-run \
  --max-pairs 3
```

Full inference:

```bash
python scripts/run_dust3r_on_av2_pairs.py \
  --pairs-json data/av2_pairs/pairs.json \
  --output-dir outputs/av2_dust3r \
  --model-path dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --device cuda \
  --max-pairs 10 \
  --image-size 512
```

Per-pair outputs:

```
outputs/av2_dust3r/pair_000000/
  image_0.jpg
  image_1.jpg
  metadata.json
  pointcloud.ply
  depth_0.png
  depth_1.png
  cameras.json
```

Use `--device cpu` if CUDA is unavailable.

---

## 7. Optional: extract AV2 calibration

For comparing DUSt3R poses against dataset ground truth:

```bash
python scripts/av2_extract_calibration.py \
  --av2-root data/av2/sensor \
  --split val \
  --log-id <LOG_ID> \
  --output-json outputs/av2_calibration_summary.json
```

Requires `calibration/` in the downloaded log, or `av2-api` installed.

---

## 8. Debug checklist

- [ ] Do all paths in `pairs.json` exist? (`python -c "import json; ..."`)
- [ ] Did `av2_explore.py` show the expected stereo camera names?
- [ ] Is overlap sufficient? (stereo > temporal stride=1 > ring-adjacent)
- [ ] Are images very high resolution? Try `--resize-long-edge 512` or lower `--image-size`
- [ ] Is `--model-path` correct and readable?
- [ ] Is CUDA available? (`python -c "import torch; print(torch.cuda.is_available())"`)
- [ ] Did a pair fail? Check `outputs/av2_dust3r/pair_XXXXXX/error.txt`
- [ ] If stereo quality is poor, try `--pair-type temporal --stride 1`

---

## 9. Related docs

- Internal inspection notes: [av2_dust3r_notes.md](./av2_dust3r_notes.md)
- Upstream DUSt3R README: [../dust3r/README.md](../dust3r/README.md)
