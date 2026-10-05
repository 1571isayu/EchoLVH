# 第 2 週：訓練資料擷取

- 程式：`ml/03_extract_frames.py`（可中斷重跑，已存在的圖會跳過）
- 抽樣：train 2,400 · val 300 · test 325（全部官方 test），失敗 0 支
- 處理：取標註那一格 → 灰階 → 右下補黑邊成正方形（不變形）→ 縮成 256×256
- 輸出：`data/echolvh_256.zip`（39 MB），內含 `images/*.png` 與 `labels.csv`
- `labels.csv` 欄位：4 個點在 256×256 上的 x,y；IVSd/LVIDd/LVPWd（cm）；`cm_per_px_256`（每個像素幾 cm）
- 驗證：用座標反算三個量測值，與醫師數值平均差 < 0.05 mm
- **重要尺度**：256×256 圖上 1 px ≈ 0.96 mm → 目標誤差 ≤ 3 mm ≈ 模型點位誤差約 3 px 以內
