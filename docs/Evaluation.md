# 量化評估計畫（Evaluation Plan）

本文件整理期末 project「VGGT → 點雲 → BEV」的量化評估流程，供團隊依序實作與撰寫 report 使用。

**相關程式與文件：**

| 項目 | 路徑 |
|------|------|
| nuScenes metric 重建 | `scripts/run_vggt_nuscenes_6cam_metric.py` |
| nuScenes relative 重建 | `scripts/run_vggt_nuscenes_6cam.py` |
| AV2 metric 重建 | `scripts/run_vggt_av2_7ring_metric.py` |
| AV2 relative 重建 | `scripts/run_vggt_av2_7ring.py` |
| 共用 helper | `scripts/vggt_nuscenes_common.py`, `scripts/nuscenes_utils.py` |
| DUSt3R baseline | `scripts/run_dust3r_on_av2_pairs.py` |
| 腳本使用說明 | `scripts/README.md` |

---

## 1. 評估總原則

### 1.1 先定座標系，再算數字

所有指標都必須明確寫出 **在哪個座標系下比較**：

| 座標系 | 適合評估的項目 |
|--------|----------------|
| **Camera frame** | 單幀 depth 精度（最乾淨、最優先） |
| **Ego / LiDAR frame** | BEV occupancy、跟車相關的指標 |
| **World frame** | 多幀融合點雲、Chamfer、時序漂移 |

建議：**depth 在 camera 評；BEV 在 ego 評；點雲融合在 world 評。**

### 1.2 兩種 scale 設定都要報告

VGGT 輸出為 **up-to-scale** depth。我們用 dataset 相機 baseline 估 metric scale（見 `estimate_scale_from_camera_baselines`）。

評估時建議同時報告：

| 模式 | 說明 | 用途 |
|------|------|------|
| **Pipeline scale** | 使用系統估出的 `metric_scale` | 反映真實 pipeline 表現 |
| **Oracle / median scale** | 在 valid pixels 上取 `median(D_gt / D_pred)` | 只評 depth「形狀」，隔離 scale 誤差 |

Report 中兩者都寫，便於分析 scale 估計對下游的影響。

### 1.3 評估用 raw 輸出，report 用 post-processed

`run_vggt_*` 預設會跑 `post_process/ply_post-process.py`（去天空、填地面洞）。  
**量化指標應在 post-process 之前計算**，否則 BEV / Chamfer 會混入人工填補的點，無法公平比較。

---

## 2. 建議實作順序（Roadmap）

```
Phase A（必做，1–2 天）
  └─ Layer 1：單幀 depth vs LiDAR sparse GT

Phase B（加分，2–3 天）
  ├─ Layer 2：scale 估計誤差
  ├─ Layer 3：3D 點雲 Chamfer / F-score
  └─ Layer 4：BEV occupancy IoU

Phase C（有餘力）
  ├─ Layer 5：Ablation（相機數、window 長度、有無 scale）
  ├─ 方法比較（VGGT vs DUSt3R vs Depth-Anything）
  └─ AV2 val 上重複 Phase A–B
```

---

## 3. Phase A — 第一步：單幀 Depth 評估

> **這是整份評估的起點。** 完成後即可在 report 寫出第一組數字。

### 3.1 目標

比較 **VGGT 預測深度** 與 **nuScenes LiDAR 投影到相機的 sparse depth GT**。

### 3.2 資料選擇（最低可行）

| 項目 | 建議值 |
|------|--------|
| Dataset | `v1.0-mini`（`v1.0-mini/`） |
| Scene | `scene-id 0` |
| Sample | `sample-idx 0`（第一個 keyframe） |
| Camera | `CAM_FRONT`（先單路，通過後再擴到 6 路） |

### 3.3 操作步驟

#### Step 1：產生 VGGT depth

```bash
# 從 repo root 執行
python scripts/run_vggt_nuscenes_6cam_metric.py \
  --scene-id 0 \
  --sample-idx 0 \
  --num-samples 1
```

輸出目錄範例：

```
outputs/vggt_nuscenes_6cam_metric/scene-0/sample_000000/
├── metadata.json          # 含 metric_scale
├── cameras.json           # 含 K、T_W_C
├── depth_00_CAM_FRONT.png # 可視化用；實際 depth 需從 inference 讀取
└── ...
```

> **待實作：** `scripts/eval_depth_nuscenes.py` 應直接呼叫 `vggt_nuscenes_common.run_vggt_multi`，避免重複 inference 邏輯。

#### Step 2：建立 LiDAR sparse depth GT

對同一 keyframe 的 `LIDAR_TOP`：

1. 讀取 LiDAR 點雲（ego / lidar frame）
2. 用 extrinsic 變換到 `CAM_FRONT` camera frame：`T_C_L = inv(T_W_C) @ T_W_L`
3. 保留相機前方點：`z_cam > 0.5 m`
4. 用內參 `K` 投影：  
   `u = fx * x/z + cx`，`v = fy * y/z + cy`
5. 保留落在影像範圍內的 pixel
6. 同一 `(u, v)` 有多個 LiDAR 點 → 取 **最小 z**（最近深度）

得到 sparse depth map `D_gt`，大小與 native 影像相同。

#### Step 3：對齊 pred 與 GT

```python
D_pred_scaled = D_pred * metric_scale   # pipeline scale
# 或
D_pred_oracle = D_pred * median(D_gt[mask] / D_pred[mask])
```

Valid mask 條件（同時滿足）：

- `D_gt > 0`（LiDAR 有量測）
- `conf >= conf_thresh`（預設 0.5）
- `1 m <= D_gt <= 50 m`（合理深度範圍，可調整）

Depth 需 resize 回 **native 解析度**（與 `backproject_metric_points` 相同做法）。

#### Step 4：計算指標

在 valid mask 上計算：

| 指標 | 公式 | 單位 |
|------|------|------|
| **Abs Rel** | `mean(|d_pred - d_gt| / d_gt)` | 無 |
| **RMSE** | `sqrt(mean((d_pred - d_gt)²))` | 公尺 |
| **log RMSE** | `sqrt(mean((log d_pred - log d_gt)²))` | 無 |
| **δ < 1.25** | `mean(max(d_pred/d_gt, d_gt/d_pred) < 1.25)` | 比例 |

#### Step 5：產出圖表（report 用）

- [ ] 原圖 + LiDAR sparse depth + VGGT depth 並排
- [ ] Error map：`|D_pred - D_gt|`
- [ ] Error histogram

### 3.4 Phase A 完成標準

```
1 scene × 1 sample × 1 camera
→ Abs Rel, RMSE, δ<1.25 三個數字
→ 至少 1 張對比圖
```

通過後擴展至：

- 同一 scene 多個 sample（例如 5 個 keyframe）
- 6 個 camera 各自評估 + 平均
- 2–3 個 scene 取 mean ± std

### 3.4 進階：按距離分段

將 valid pixels 依 `D_gt` 分 bin，分別報告指標：

| Bin | 意義 |
|-----|------|
| 0–10 m | 近距離（行人、路緣） |
| 10–30 m | 一般行車 |
| 30 m+ | 遠距離（通常誤差較大） |

此表格非常適合寫 limitation（「遠距離深度退化」）。

---

## 4. Layer 2 — Scale 估計評估

### 4.1 目標

評估 `estimate_scale_from_camera_baselines` 的準確度。這是我們 pipeline 相對於「直接 metric depth model」的特色環節。

### 4.2 Ground Truth

同一 forward pass 內，多相機兩兩 baseline：

```python
# VGGT 相機中心（up-to-scale）
vggt_centers = [-R.T @ t for each extrinsic]

# Dataset 相機中心（公尺）
metric_centers = [T_W_C[:3, 3] for each camera]

# 每一對 (i, j) 的真 scale
scale_gt_ij = ||metric_centers[i] - metric_centers[j]||
              / ||vggt_centers[i] - vggt_centers[j]||
```

### 4.3 指標

| 指標 | 說明 |
|------|------|
| **Scale relative error** | `|s_pred - median(scale_gt_ij)| / median(scale_gt_ij)` |
| **Scale vs num-samples** | 固定 scene，改 `--num-samples`（1 / 3 / 5），畫誤差曲線 |
| **Depth RMSE w/ vs w/o scale** | Ablation：`metric_scale=1.0` vs pipeline scale |

### 4.4 建議實驗

```bash
# 1 sample（只有空間 baseline，無時序）
python scripts/run_vggt_nuscenes_6cam_metric.py --scene-id 0 --sample-idx 0 --num-samples 1

# 5 samples（含時序位移，scale 來源更豐富）
python scripts/run_vggt_nuscenes_6cam_metric.py --scene-id 0 --sample-idx 0 --num-samples 5
```

---

## 5. Layer 3 — 3D 點雲幾何評估

### 5.1 目標

評估 metric back-projection 融合後的整體幾何品質。

### 5.2 輸入

| 來源 | 取得方式 |
|------|----------|
| **Pred 點雲** | `run_vggt_nuscenes_6cam_metric.py` 輸出的 raw `.ply`（post-process 前） |
| **GT 點雲** | 同一 sample 的 `LIDAR_TOP`，變換到 world frame |

### 5.3 前處理

1. **ROI 裁剪**：ego 周圍 ±25 m（水平）、高度 −2 m ~ +3 m
2. **Voxel downsample**：0.1 m（與 `--voxel-size 0.10` 一致）
3. **可選**：用 nuScenes instance annotation 移除 dynamic objects（車、行人）

### 5.4 指標

| 指標 | 說明 | 工具 |
|------|------|------|
| **Chamfer Distance** | pred→GT 與 GT→pred 最近鄰距離的平均 | Open3D |
| **F-score @ τ** | 距離 < τ（0.2 m / 0.5 m）的點比例 | 自行實作 |
| **Precision / Recall** | 單向最近鄰命中率 | 3D detection 常用 |

### 5.5 注意事項

- Metric pipeline 已用 dataset pose，**原則上不需 ICP**。若整體偏移，先檢查 scale 與 extrinsic。
- 動態物體會造成「拖影」，Chamfer 偏高屬正常；report 中應說明或 mask 掉。

---

## 6. Layer 4 — BEV 評估

### 6.1 目標

評估 bird's-eye view 佔據格網品質，最貼近「智慧汽車」應用。

### 6.2 BEV 定義

在 **ego / LiDAR frame** 下建立格網：

| 參數 | 建議值 |
|------|--------|
| 範圍 | x, y ∈ [−25 m, +25 m] |
| 解析度 | 0.2 m → 250 × 250 grid |
| 高度過濾 | z ∈ [−0.5 m, +2.0 m]（地面至車高） |

每個 cell 記錄：

- **Binary occupancy**：該 cell 內是否有點（0 / 1）
- **Max height**（可選）：該 cell 內最高 z 值

### 6.3 輸入

| 來源 | 說明 |
|------|------|
| **Pred BEV** | VGGT metric 點雲投影 |
| **GT BEV** | LiDAR 點雲投影（同一 ego pose、同一 ROI） |

### 6.4 指標

**Binary occupancy：**

| 指標 | 公式 |
|------|------|
| **IoU** | `TP / (TP + FP + FN)` |
| **Precision** | `TP / (TP + FP)` |
| **Recall** | `TP / (TP + FN)` |
| **F1** | Precision 與 Recall 的調和平均 |

**Height map（若使用 max height）：**

| 指標 | 說明 |
|------|------|
| **RMSE** | occupied cells 上的高度 RMSE（公尺） |
| **Mean Abs Error** | occupied cells 上的平均絕對誤差 |

### 6.5 Report 圖表

- [ ] GT LiDAR BEV | VGGT pred BEV | 差異 map 三聯圖
- [ ] Occupancy IoU 隨相機數量變化的 bar chart

---

## 7. Layer 5 — Ablation 與方法比較

### 7.1 相機數量 Ablation

| 配置 | 預期觀察 |
|------|----------|
| 1 cam（`CAM_FRONT`） | BEV coverage 最低 |
| 3 cams（front + left/right） | 覆蓋率提升 |
| 6 cams | 完整環景 |
| 7 ring（AV2） | 後方覆蓋更好 |

Coverage 定義：GT occupied cells 中被 pred 覆蓋的比例。

### 7.2 Scale Ablation

| 設定 | Depth RMSE | BEV IoU |
|------|------------|---------|
| VGGT + metric scale（pipeline） | | |
| VGGT，scale = 1.0 | | |
| Oracle median scale | | |

### 7.3 方法比較表（建議 report 主表）

在同一 scene / sample / camera 下比較：

| 方法 | Abs Rel ↓ | RMSE ↓ | Chamfer ↓ | BEV IoU ↑ | 推理時間 |
|------|-----------|--------|-----------|-----------|----------|
| VGGT + ego/camera scale | | | | | |
| VGGT relative（Version A） | N/A | N/A | | | |
| DUSt3R | | | | | |
| Depth-Anything-V2 metric（`depth_recon.py`） | | | | | |

### 7.4 時序長度 Ablation

| `--num-samples` | Scale error | Chamfer | 備註 |
|-----------------|-------------|---------|------|
| 1 | | | 僅空間 baseline |
| 3 | | | |
| 5 | | | |
| 10 | | | 可能累積漂移 |

---

## 8. 建議新增的評估腳本

以下腳本尚未實作，建議依 Phase 順序新增：

| 優先級 | 腳本 | 功能 |
|--------|------|------|
| P0 | `scripts/eval_depth_nuscenes.py` | Layer 1：LiDAR → sparse depth GT，算 Abs Rel / RMSE / δ |
| P1 | `scripts/eval_scale_nuscenes.py` | Layer 2：scale GT vs pred |
| P1 | `scripts/eval_pointcloud_nuscenes.py` | Layer 3：Chamfer / F-score |
| P2 | `scripts/eval_bev_nuscenes.py` | Layer 4：BEV occupancy IoU |
| P3 | `scripts/eval_run_all.py` | 批次跑多 scene，輸出 CSV / JSON summary |

### 8.1 `eval_depth_nuscenes.py` 介面草案

```bash
python scripts/eval_depth_nuscenes.py \
  --scene-id 0 \
  --sample-idx 0 \
  --camera CAM_FRONT \
  --scale-mode pipeline        # pipeline | oracle | both
  --output-dir outputs/eval/depth
```

輸出：

```
outputs/eval/depth/
├── metrics.json               # Abs Rel, RMSE, delta_1.25, ...
├── depth_compare.png          # 原圖 | GT | Pred | Error
└── error_hist.png
```

### 8.2 共用模組建議

將 LiDAR 投影邏輯放在 `scripts/nuscenes_utils.py` 或新建 `scripts/eval_utils.py`：

```python
def lidar_to_sparse_depth(nusc, sample, camera_channel) -> np.ndarray:
    """Return sparse depth map (H, W), 0 = no measurement."""

def compute_depth_metrics(d_pred, d_gt, mask) -> dict:
    """Return abs_rel, rmse, log_rmse, delta_1.25, ..."""

def points_to_bev(points, x_range, y_range, resolution) -> np.ndarray:
    """Project 3D points to binary occupancy grid."""
```

---

## 9. Report 交付清單

### 9.1 必備表格

- [ ] Depth 指標表（Abs Rel, RMSE, δ<1.25），至少 1 scene × 1 camera
- [ ] Scale ablation 表（有 / 無 scale）
- [ ] 方法比較表（VGGT vs baseline）

### 9.2 必備圖表

- [ ] Depth 對比圖（原圖 / LiDAR GT / VGGT pred / error map）
- [ ] BEV 三聯圖（GT / pred / diff）
- [ ] 至少 2 個 failure case（動態物體拖影、低紋理路面、遠距離）

### 9.3 建議補充

- [ ] 按距離分段的 depth 表
- [ ] Scale error vs `--num-samples` 折線圖
- [ ] 相機數量 vs BEV coverage bar chart
- [ ] 推理時間與點雲大小（工程面）

---

## 10. 常見問題

| 問題 | 影響 | 處理方式 |
|------|------|----------|
| LiDAR 與 camera 不同步 | depth GT 錯位 | 使用同一 keyframe 的 paired `sample_data` token |
| VGGT resize 後內參不一致 | 投影偏移 | depth resize 回 native 解析度再比較 |
| Scale 估計不穩定 | 整體深度偏差 | 報告 scale error；必要時 clip 異常值 |
| 動態物體 | Chamfer / BEV IoU 偏高 | 寫入 limitation，或 mask dynamic class |
| Post-process 填補地面 | BEV 假陽性 | **評估用 raw `.ply`，不用 `_post.ply`** |
| v1.0-mini 樣本少 | 數字波動大 | 多 scene 平均 + 報告 std |

---

## 11. 資料集與 Ground Truth 對照

| Dataset | LiDAR GT | Camera GT (K, pose) | 適用 Layer |
|---------|----------|---------------------|------------|
| nuScenes v1.0-mini | `LIDAR_TOP` | `calibrated_sensor` + `ego_pose` | 1–5（優先） |
| Argoverse 2 val | `sensors/lidar/` | `calibration/` + `city_SE3_egovehicle.feather` | 1–5（Phase C） |

nuScenes mini 共 10 scenes、404 samples，不必全部跑完。建議：

- **Phase A–B：** 2–3 scenes × 5 samples × 6 cameras
- **Report 主表：** 取 mean ± std

---

## 12. 檢核表（Checklist）

### Phase A

- [ ] 跑通 `run_vggt_nuscenes_6cam_metric.py` 並讀取 `metric_scale`
- [ ] 實作 LiDAR → sparse depth 投影
- [ ] 算出 CAM_FRONT 的 Abs Rel / RMSE / δ<1.25
- [ ] 產生至少 1 張 depth 對比圖

### Phase B

- [ ] Scale relative error
- [ ] 6 cam 融合點雲 Chamfer
- [ ] BEV occupancy IoU
- [ ] 有 / 無 scale ablation

### Phase C

- [ ] VGGT vs DUSt3R 比較
- [ ] 相機數量 ablation
- [ ] AV2 val 評估
- [ ] Failure case 整理

---

*Last updated: 2026-06-07*
