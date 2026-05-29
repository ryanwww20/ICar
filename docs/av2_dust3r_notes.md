# AV2 + DUSt3R Integration Notes

Last updated: 2026-05-29

## Task 0: Repo Inspection

### Repo identity

| Item | Finding |
|------|---------|
| Top-level repo | `ICar` (`git@github.com:ryanwww20/ICar.git`) — user integration repo |
| DUSt3R location | `dust3r/` subdirectory |
| DUSt3R origin | Official Naver DUSt3R codebase (README, LICENSE, package layout match upstream) |
| Submodule | `croco/` is a git submodule (`https://github.com/naver/croco`) |
| AV2 integration | **Not present yet** — added as standalone scripts under `scripts/` |

This is **not** a bare clone of only DUSt3R. It is a **user integration repo** that vendors the official DUSt3R tree under `dust3r/`, with new AV2 pipeline scripts added at repo root.

### 1. DUSt3R inference entry points

| Entry | Path | Role |
|-------|------|------|
| **Primary Python API** | `dust3r/dust3r/inference.py` → `inference()` | Core batch inference on image pairs |
| **Gradio demo (CLI)** | `dust3r/demo.py` | Launches web UI; loads model and calls `dust3r/dust3r/demo.py` helpers |
| **Demo helpers** | `dust3r/dust3r/demo.py` → `get_reconstructed_scene()` | End-to-end: `load_images` → `make_pairs` → `inference` → `global_aligner` → export GLB |
| **Visloc script** | `dust3r/visloc.py` | Visual localization (separate task) |

Recommended programmatic flow (from official README):

```python
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
```

For **two-image pairs**, use `GlobalAlignerMode.PairViewer` (no iterative global alignment).

### 2. Existing demo / inference / package layout

| Component | Present? | Path |
|-----------|----------|------|
| `demo.py` (Gradio CLI) | Yes | `dust3r/demo.py` |
| `dust3r/demo.py` (Gradio logic) | Yes | `dust3r/dust3r/demo.py` |
| `inference.py` | Yes | `dust3r/dust3r/inference.py` |
| `dust3r` package | Yes | `dust3r/dust3r/` |
| Docker demo | Yes | `dust3r/docker/` |
| Notebooks | Yes | `dust3r/croco/interactive_demo.ipynb` |

### 3. Model weights — expected paths

| Method | Path / identifier |
|--------|-------------------|
| HuggingFace Hub (default) | `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt` |
| Local checkpoint (recommended for batch runs) | `dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` |
| CLI flag | `--weights checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` |
| Alternative models | `DUSt3R_ViTLarge_BaseDecoder_512_linear.pth`, `DUSt3R_ViTLarge_BaseDecoder_224_linear.pth` |

Download example (from upstream README):

```bash
mkdir -p dust3r/checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  -P dust3r/checkpoints/
```

`AsymmetricCroCo3DStereo.from_pretrained()` accepts either a HuggingFace repo id or a **local `.pth` file path**.

### 4. Input image pair methods

| Method | Supported? | Notes |
|--------|------------|-------|
| Python API | **Yes (primary)** | `load_images([path1, path2], size=512)` |
| Gradio demo | Yes | `python demo.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt` |
| Command-line batch | No native CLI | Use `scripts/run_dust3r_on_av2_pairs.py` (added in this integration) |

Official demo accepts a list of image paths via Gradio file upload; there is no standalone `inference.py` CLI for arbitrary pairs.

### 5. Dependencies

| File | Path |
|------|------|
| `requirements.txt` | `dust3r/requirements.txt` |
| `requirements_optional.txt` | `dust3r/requirements_optional.txt` |
| `environment.yml` | **Not present** |

Core deps: `torch`, `torchvision`, `roma`, `gradio`, `matplotlib`, `tqdm`, `opencv-python`, `scipy`, `einops`, `trimesh`, `huggingface-hub[torch]`.

### 6. Environment check (this machine)

| Package | Status |
|---------|--------|
| `torch` | Installed — `2.11.0a0+eb65b36914.nv26.02` |
| `torchvision` | Installed — `0.25.0a0+...` |
| `dust3r` (import from `dust3r/`) | **Works** when `dust3r/` is on `PYTHONPATH` |
| CUDA | **Not available** — driver too old for installed PyTorch build |
| RoPE CUDA kernels | Not compiled — falls back to slow PyTorch RoPE |
| `av2` / `av2-api` | **Not installed** |

To use DUSt3R from repo root scripts:

```bash
cd dust3r && pip install -r requirements.txt
# optional RoPE compile:
cd croco/models/curope && python setup.py build_ext --inplace
```

---

## Stage 1: Dataset Exploration

**Script:** `scripts/av2_explore.py`

Scans up to `--max-logs` log directories under the AV2 split root, lists camera folders, image counts, sample paths, and auto-detects stereo front-left / front-right candidates.

**Flexible path handling** (via `scripts/av2_utils.py`):

- Resolves split root from `--av2-root` + `--split`
- Tries `sensors/cameras/*`, `cameras/*`, etc.
- Falls back to limited-depth recursive image grouping

**Stereo detection keywords:** `stereo`, `front_left`, `front_right`, `front-left`, `front-right`, `ring_front_left`, `ring_front_right`

If no stereo pair is found, all camera folder names are printed for manual confirmation.

---

## Stage 2: Pair Export

**Script:** `scripts/av2_make_pairs.py`

Exports `pairs.json` and `pairs.txt` under `--output-dir`.

| `--pair-type` | Behavior |
|---------------|----------|
| `stereo` | Front stereo left/right; timestamp-aligned pairing |
| `temporal` | Same-camera consecutive frames with `--stride` |
| `ring-adjacent` | Adjacent ring camera pairs (e.g. `ring_front_left` ↔ `ring_front_center`) |

All paths in JSON are **absolute**. Optional `--copy-images` and `--resize-long-edge`.

---

## Stage 1 + Stage2: AV2 calibration (optional)

**Script:** `scripts/av2_extract_calibration.py`

| Priority | Method |
|----------|--------|
| 1 | `av2-api` + feather files (`calibration/intrinsics.feather`, `extrinsics.feather`) |
| 2 | File-search fallback for json/feather under each log |

**Uncertainties / manual verification needed:**

- Exact AV2 download layout on your machine (full log vs. camera-only subset)
- Whether your partial download includes `calibration/` feather files
- AV2 camera naming in your split may differ slightly from docs — always run `av2_explore.py` first
- `av2-api` is optional; install with `pip install av2` if you want structured calibration access

---

## Stage 3: DUSt3R Inference

**Script:** `scripts/run_dust3r_on_av2_pairs.py`

Wraps official DUSt3R API:

1. `AsymmetricCroCo3DStereo.from_pretrained(model_path)`
2. `load_images([img1, img2], size=image_size)`
3. `make_pairs(..., scene_graph='complete', symmetrize=True)`
4. `inference(pairs, model, device)`
5. `global_aligner(..., mode=GlobalAlignerMode.PairViewer)` for 2-view pairs

**Per-pair output** under `outputs/av2_dust3r/pair_XXXXXX/`:

- `image_0.jpg`, `image_1.jpg`
- `metadata.json`
- `pointcloud.ply` (if enabled)
- `depth_0.png`, `depth_1.png` (if enabled)
- `cameras.json` (if enabled)
- `error.txt` on failure (batch continues)

**Manual confirmation checklist before first real run:**

1. Download DUSt3R weights to `dust3r/checkpoints/` or pass `--model-path`
2. Verify CUDA if using `--device cuda` (currently unavailable on this machine)
3. Run `av2_explore.py` on your partial AV2 download to confirm camera names
4. Run `av2_make_pairs.py` with small `--max-logs` / `--max-pairs-per-log`
5. Dry-run inference: `run_dust3r_on_av2_pairs.py --dry-run ...`
6. Run 1–2 pairs before scaling batch size

---

## Files added (integration layer)

```
scripts/
  av2_utils.py
  av2_explore.py
  av2_make_pairs.py
  run_dust3r_on_av2_pairs.py
  av2_extract_calibration.py
docs/
  av2_dust3r_notes.md
  run_dust3r_on_argoverse2.md
```

No changes to `dust3r/dust3r/*` core code.
