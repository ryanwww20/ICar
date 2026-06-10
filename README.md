# ICar

多相機自駕場景的 3D 重建與點雲處理工具集。以 [VGGT](https://github.com/facebookresearch/vggt) 為主，從 **Argoverse 2** 與 **nuScenes** 多視角影像推論深度與點雲，並支援後處理（去天空、填地面洞）與量化評估規劃。

## 功能概覽

| 模組 | 說明 |
|------|------|
| **VGGT 多視角重建** | 7 環視（AV2）或 6 相機（nuScenes）一次 forward，輸出 `.ply`、深度圖、相機參數 |
| **Relative / Metric** | Relative：VGGT 原生 up-to-scale 幾何；Metric：用資料集相機 baseline 估 scale，反投影至公尺座標 |
| **點雲後處理** | 天空裁切、地面洞填補、可視化 |
| **資料下載** | AV2 部分下載腳本（ring / stereo / LiDAR） |
| **Baseline** | DUSt3R（AV2 pairs）、Depth-Anything-V2（`depth_recon.py`） |

詳細腳本用法見 [`scripts/README.md`](scripts/README.md)；量化評估計畫見 [`docs/Evaluation.md`](docs/Evaluation.md)。

---

## 環境需求

- **OS**：Linux（建議）
- **Python**：3.10+
- **GPU**：建議 NVIDIA GPU（VGGT-1B 推理）；CPU 可跑但很慢
- **磁碟**：模型權重約數 GB；資料集另計（nuScenes mini ~4 GB；AV2 可部分下載）

---

## 安裝步驟

### 1. 克隆本 repo

```bash
git clone git@github.com:ryanwww20/ICar.git
cd ICar
```

### 2. 建立 Python 環境

**Conda（建議）：**

```bash
conda create -n icar python=3.10 -y
conda activate icar
```

**或 venv：**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 安裝 PyTorch

依你的 CUDA 版本從 [PyTorch 官網](https://pytorch.org/get-started/locally/) 安裝。範例（CUDA 12.1）：

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

### 4. 安裝本 repo 依賴

```bash
pip install -r requirements.txt
```

### 5. 取得 VGGT 原始碼

`vggt/` 目錄未納入 git（見 `.gitignore`），需自行克隆到 repo 根目錄：

```bash
git clone https://github.com/facebookresearch/vggt.git vggt
```

腳本會自動將 `vggt/` 加入 `sys.path`。首次執行時會從 Hugging Face 下載模型權重（預設 `facebook/VGGT-1B`）。

### 6.（可選）額外依賴

| 用途 | 安裝指令 |
|------|----------|
| nuScenes 完整 API | `pip install nuscenes-devkit` |
| AV2 結構化 API | `pip install av2` |
| Depth-Anything baseline | `pip install transformers pyquaternion` |
| AV2 快速下載 | `conda install -c conda-forge s5cmd -y` |
| DUSt3R baseline | 見 [`docs/run_dust3r_on_argoverse2.md`](docs/run_dust3r_on_argoverse2.md) |

---

## 資料集準備

### nuScenes v1.0-mini

1. 至 [nuScenes 官網](https://www.nuscenes.org/nuscenes) 下載 `v1.0-mini.tgz`
2. 解壓到 repo 根目錄，結構應為：

```
v1.0-mini/
├── maps/
├── samples/
├── sweeps/
├── v1.0-mini/
└── ...
```

預設 dataroot：`v1.0-mini/`（可在腳本用 `--dataroot` 覆寫）。

### Argoverse 2 Sensor（部分下載）

使用內建下載腳本（需 `s5cmd`）：

```bash
# 列出可用 log
bash scripts/av2_download.sh --list-only

# 下載 2 個 log 的 7 環視相機（metric 還需 calibration）
bash scripts/av2_download.sh \
  --num-logs 2 \
  --mode ring \
  --out-dir vggt/data/AV2

# 含 LiDAR（評估用）
bash scripts/av2_download.sh \
  --num-logs 2 \
  --mode ring_lidar \
  --out-dir vggt/data/AV2
```

預設 AV2 路徑：`vggt/data/AV2/`。**Metric 模式**需包含 `calibration/` 與 `city_SE3_egovehicle.feather`。

---

## 快速開始

所有腳本請在 **repo 根目錄**執行。

### 列出場景

```bash
# AV2
python scripts/run_vggt_av2_7ring.py --list-scenes

# nuScenes
python scripts/run_vggt_nuscenes_6cam.py --list-scenes
```

### AV2 — 7 環視 Relative 重建

```bash
# 僅列出輸入影像路徑（不跑模型）
python scripts/run_vggt_av2_7ring.py \
  --split val \
  --scene-id 0 \
  --frame-idx 0 \
  --dry-run

# 執行 VGGT
python scripts/run_vggt_av2_7ring.py \
  --split val \
  --scene-id 0 \
  --frame-idx 0
```

輸出：`outputs/vggt_av2_7ring/`

### nuScenes — 6 相機 Metric 重建

```bash
python scripts/run_vggt_nuscenes_6cam_metric.py \
  --scene-id 0 \
  --sample-idx 0
```

輸出：`outputs/vggt_nuscenes_6cam_metric/`

### 點雲後處理（單獨執行）

```bash
python scripts/post_process/ply_post-process.py \
  --input outputs/vggt_av2_7ring/rel_av2_scene0_frame0.ply \
  --output outputs/vggt_av2_7ring/rel_av2_scene0_frame0_post.ply
```

---

## 專案結構

```
ICar/
├── README.md                 # 本文件
├── requirements.txt          # Python 依賴
├── scripts/
│   ├── README.md             # VGGT 腳本詳細說明
│   ├── run_vggt_av2_7ring*.py
│   ├── run_vggt_nuscenes_6cam*.py
│   ├── av2_download.sh       # AV2 部分下載
│   ├── av2_utils.py
│   ├── nuscenes_utils.py
│   ├── vggt_nuscenes_common.py
│   └── post_process/         # 點雲後處理
├── docs/
│   ├── Evaluation.md         # 量化評估計畫
│   └── run_dust3r_on_argoverse2.md
├── vggt/                     # 需自行 clone（見上方安裝步驟）
├── v1.0-mini/                # nuScenes mini（需自行下載）
├── data/                     # AV2 pairs 等中間產物
└── outputs/                  # 推理輸出（gitignore）
```

---

## Relative vs Metric

| 版本 | Scale | 點雲來源 | 適用情境 |
|------|-------|----------|----------|
| **Relative (A)** | 任意尺度 | VGGT `world_points` 直接融合 | 視覺化形狀、相對幾何 |
| **Metric (B)** | 公尺 | VGGT depth × 估計 scale，用資料集 K / pose 反投影 | 與真實座標 / BEV 對齊 |

| 資料集 | Relative | Metric |
|--------|----------|--------|
| AV2 7-ring | `run_vggt_av2_7ring.py` | `run_vggt_av2_7ring_metric.py` |
| nuScenes 6-cam | `run_vggt_nuscenes_6cam.py` | `run_vggt_nuscenes_6cam_metric.py` |

---

## 輸出說明

每次推理會產生：

- `{rel|metric}_{dataset}_scene{n}_frame{m}.ply` — 原始點雲
- `{...}_post.ply` — 後處理點雲（預設開啟）
- `scene-N/frame_XXXXXX/` 或 `sample_XXXXXX/` — 深度圖、影像、`metadata.json`、`cameras.json`

**量化評估請使用 raw `.ply`**，不要用 `_post.ply`（後處理會填補地面，影響指標公平性）。詳見 [`docs/Evaluation.md`](docs/Evaluation.md)。

---

## 常見問題

**`ModuleNotFoundError: No module named 'vggt'`**  
確認已將 [facebookresearch/vggt](https://github.com/facebookresearch/vggt) clone 到 `vggt/` 目錄。

**AV2 metric 報 calibration 錯誤**  
下載時需包含 `calibration/`；可用 `--mode ring` 並確認 log 內有 feather 檔案，或改用 full 模式。

**CUDA out of memory**  
VGGT-1B 需要較大顯存；可嘗試 `--device cpu`（很慢）或減少 `--num-frames` / `--num-samples`。

**nuScenes 沒裝 devkit**  
腳本會自動 fallback 讀 JSON metadata；安裝 `nuscenes-devkit` 可獲得完整 API。

---

## 授權

本 repo 腳本與文件依專案 LICENSE 授權。VGGT、nuScenes、Argoverse 2 等第三方資源請遵循各自授權條款。
