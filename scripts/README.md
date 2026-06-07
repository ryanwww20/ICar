# VGGT Point Cloud Scripts

Run [VGGT](https://github.com/facebookresearch/vggt) on multi-camera frames and export fused point clouds (`.ply`), depth maps, and metadata.

All scripts live under `scripts/`. Run from the **repo root** (`/work/b12901015/car`).

---

## Relative vs Metric

| Version | Scale | Point cloud source | Use when |
|---------|-------|-------------------|----------|
| **Relative (A)** | Up-to-scale (arbitrary units) | VGGT `world_points` fused directly | Visualizing shape / relative geometry |
| **Metric (B)** | Meters | VGGT depth × estimated scale, back-projected with dataset K / pose | Aligning with real-world / map coordinates |

Each output folder includes `metadata.json` with `version_label`:
- Relative: `relative_up_to_scale`
- Metric: `metric` (also writes `metric_scale`)

---

## Prerequisites

```bash
# Example: AV2 conda env (adjust to your setup)
conda activate AV2   # or your env with torch + vggt deps

pip install pyarrow open3d opencv-python-headless
# nuScenes scripts work without nuscenes-devkit (JSON fallback),
# but you can optionally install: pip install nuscenes-devkit
```

GPU recommended. Default model: `facebook/VGGT-1B`.

---

## Argoverse 2 — 7 ring cameras

Data default: `vggt/data/AV2`  
Ring download must include `calibration/` and `city_SE3_egovehicle.feather` (required for **metric** only).

### List scenes

```bash
python scripts/run_vggt_av2_7ring.py --list-scenes
python scripts/run_vggt_av2_7ring_metric.py --list-scenes
```

### Relative (Version A)

```bash
# Dry-run: print 7 image paths only
python scripts/run_vggt_av2_7ring.py \
  --split val \
  --scene-id 0 \
  --frame-idx 0 \
  --dry-run

# Run VGGT on scene-0, frame 0
python scripts/run_vggt_av2_7ring.py \
  --split val \
  --scene-id 0 \
  --frame-idx 0

# Run 3 consecutive frames
python scripts/run_vggt_av2_7ring.py \
  --split val \
  --scene-id 0 \
  --frame-idx 0 \
  --num-frames 3

# Pick log by UUID instead of scene index
python scripts/run_vggt_av2_7ring.py \
  --split val \
  --log-id 02678d04-cc9f-3148-9f95-1ba66347dff9 \
  --frame-idx 0
```

**Output:** `outputs/vggt_av2_7ring/scene-N/frame_XXXXXX/`

### Metric (Version B)

Same CLI as relative; different script and output directory.

```bash
# Dry-run
python scripts/run_vggt_av2_7ring_metric.py \
  --split val \
  --scene-id 0 \
  --frame-idx 0 \
  --dry-run

# Run VGGT + metric back-projection
python scripts/run_vggt_av2_7ring_metric.py \
  --split val \
  --scene-id 0 \
  --frame-idx 0

# Custom AV2 root / output
python scripts/run_vggt_av2_7ring_metric.py \
  --av2-root vggt/data/AV2 \
  --split val \
  --scene-id 0 \
  --frame-idx 0 \
  --output-dir outputs/vggt_av2_7ring_metric
```

**Output:** `outputs/vggt_av2_7ring_metric/scene-N/frame_XXXXXX/`

---

## nuScenes v1.0-mini — 6 cameras

Data default: `v1.0-mini/` (repo root)

### List scenes

```bash
python scripts/run_vggt_nuscenes_6cam.py --list-scenes
python scripts/run_vggt_nuscenes_6cam_metric.py --list-scenes
```

### Relative (Version A)

```bash
# Dry-run
python scripts/run_vggt_nuscenes_6cam.py \
  --scene-id 0 \
  --sample-idx 0 \
  --dry-run

# Run VGGT on scene-0, sample 0
python scripts/run_vggt_nuscenes_6cam.py \
  --scene-id 0 \
  --sample-idx 0

# Run 5 consecutive samples
python scripts/run_vggt_nuscenes_6cam.py \
  --scene-id 0 \
  --sample-idx 0 \
  --num-samples 5

# Pick scene by nuScenes name
python scripts/run_vggt_nuscenes_6cam.py \
  --scene-name scene-0061 \
  --sample-idx 0
```

**Output:** `outputs/vggt_nuscenes_6cam_relative/scene-N/sample_XXXXXX/`

### Metric (Version B)

```bash
# Dry-run
python scripts/run_vggt_nuscenes_6cam_metric.py \
  --scene-id 0 \
  --sample-idx 0 \
  --dry-run

# Run VGGT + metric back-projection
python scripts/run_vggt_nuscenes_6cam_metric.py \
  --scene-id 0 \
  --sample-idx 0

# Custom dataroot
python scripts/run_vggt_nuscenes_6cam_metric.py \
  --dataroot /work/b12901015/car/v1.0-mini \
  --version v1.0-mini \
  --scene-id 0 \
  --sample-idx 0
```

**Output:** `outputs/vggt_nuscenes_6cam_metric/scene-N/sample_XXXXXX/`

---

## Common options (all scripts)

| Flag | Default | Description |
|------|---------|-------------|
| `--model-id` | `facebook/VGGT-1B` | Hugging Face model id |
| `--device` | auto (`cuda` / `cpu`) | Force device, e.g. `cuda:0` |
| `--conf-thresh` | `0.5` | Confidence threshold for points |
| `--pixel-stride` | `2` | Subsample stride (larger = fewer points) |
| `--voxel-size` | `0.10` | Voxel downsample size in meters (`0` to skip) |
| `--no-cleanup` | off | Skip voxel downsample + outlier removal |
| `--dry-run` | off | List inputs only; do not run VGGT |

### PLY post-processing (all `run_vggt_*` scripts)

After each raw `.ply` is saved, the scripts run `scripts/post_process/ply_post-process.py` by default (`full_pipeline`: sky removal → floor hole fill).

| Flag | Default | Description |
|------|---------|-------------|
| `--post-process` | on | Enable post-processing after saving raw `.ply` |
| `--no-post-process` | — | Skip post-processing |
| `--post-process-task` | `full_pipeline` | `full_pipeline` or `fill_hole` only |
| `--sky-axis` | `y` | Axis for sky clipping (`x` / `y` / `z`) |
| `--sky-side` | `low` | Sky on the high or low side of the axis |
| `--sky-keep-percentile` | `0.9` | Fraction of non-sky points to keep |
| `--sky-max` | — | Absolute sky threshold (overrides percentile) |

```bash
# Default: VGGT → raw .ply → post-processed .ply
python scripts/run_vggt_av2_7ring.py \
  --split val \
  --scene-id 0 \
  --frame-idx 0

# Skip post-processing
python scripts/run_vggt_av2_7ring.py \
  --split val \
  --scene-id 0 \
  --frame-idx 0 \
  --no-post-process

# Only fill holes (no sky removal)
python scripts/run_vggt_av2_7ring.py \
  --split val \
  --scene-id 0 \
  --frame-idx 0 \
  --post-process-task fill_hole
```

To run post-processing standalone on an existing `.ply`:

```bash
python scripts/post_process/ply_post-process.py \
  --input outputs/vggt_av2_7ring/rel_av2_scene0_frame0.ply \
  --output outputs/vggt_av2_7ring/rel_av2_scene0_frame0_post.ply

python scripts/post_process/ply_post-process.py --list
```

---

## Output layout (per frame / sample)

Point clouds are written under the output root (not inside each frame/sample folder):

```
outputs/vggt_av2_7ring/
├── rel_av2_scene0_frame0.ply          # raw VGGT output
├── rel_av2_scene0_frame0_post.ply     # post-processed (default)
├── scene_mapping.json
└── scene-0/
    ├── run_summary.json
    └── frame_000000/
        ├── metadata.json              # pointcloud_path, postprocessed_pointcloud_path
        ├── cameras.json               # predicted (and metric: dataset) calibration
        ├── image_00_*.jpg
        └── depth_00_*.png
```

nuScenes uses the same pattern with `sample_XXXXXX/` folders and names like `rel_nuscenes_scene0_frame0.ply`.

---

## Script reference

| Dataset | Relative (A) | Metric (B) |
|---------|--------------|------------|
| AV2 7-ring | `run_vggt_av2_7ring.py` | `run_vggt_av2_7ring_metric.py` |
| nuScenes 6-cam | `run_vggt_nuscenes_6cam.py` | `run_vggt_nuscenes_6cam_metric.py` |

Shared helpers: `av2_utils.py`, `av2_calibration_utils.py`, `nuscenes_utils.py`, `vggt_nuscenes_common.py`, `ply_postprocess_common.py`

Post-processing: `post_process/ply_post-process.py`, `post_process/fill_hole.py`, `post_process/view_ply.py`
