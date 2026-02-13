import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
import os  # 新增: 用於處理路徑

# 導入我們自己寫的模組
from features.alignment import synthesize_mtf_data
from features.labeling import apply_triple_barrier
from features.preprocess import prepare_features
from models.tcn_core import ParallelTCNAlphaHunter
from utils.loss import FocalLoss, calculate_mcc
from data.dataset import CryptoTimeSeriesDataset

# --- 設定參數 (Hyperparameters) ---
CONFIG = {
    'seq_len': 60,  # Benchmark: 30, 40, 60, 80
    'norm_method': 'z_score',  # Benchmark: 'z_score' vs 'log_return'
    'batch_size': 64,
    'epochs': 20,
    'learning_rate': 1e-3,
    'atr_period': 14,
    'horizon': 60,  # Triple Barrier 的時間牆
    'pt_mul': 2.0,  # 止盈寬度
    'sl_mul': 2.0,  # 止損寬度
}


def load_and_clean_data(filepath):
    """
    專門讀取用戶格式的 CSV 檔案
    格式: Open Time, Open, High, Low, Close, Volume
    """
    print(f"📄 讀取檔案: {filepath}")
    # 讀取 CSV
    df = pd.read_csv(filepath)

    # 1. 重新命名欄位 (轉為小寫以符合系統變數)
    rename_map = {
        'Open Time': 'datetime',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }
    # 容錯處理：有些數據可能已經是小寫，這裡做個檢查
    current_cols = df.columns
    actual_rename = {k: v for k, v in rename_map.items() if k in current_cols}
    df = df.rename(columns=actual_rename)

    # 2. 處理時間索引
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    elif df.index.name != 'datetime':
        # 嘗試將 index 轉為 datetime (如果原本沒有 Open Time 欄位)
        df.index = pd.to_datetime(df.index)

    df.sort_index(inplace=True)

    # 3. 確保數值型態 (移除可能的字串)
    cols = ['open', 'high', 'low', 'close', 'volume']
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(float)

    return df


def train_model():
    print(f"🚀 啟動 Alpha Hunter 訓練程序...")
    print(f"⚙️  設定: Seq_Len={CONFIG['seq_len']}, Norm={CONFIG['norm_method']}")

    # 1. 載入數據
    # Windows 路徑處理: 使用 os.path.join 確保相容性
    # 注意：我們只需要讀取 1H 數據，features/alignment.py 會幫我們「零延遲」合成 4H 和 1D
    # 這樣可以避免直接讀取 1D 檔案造成的「偷看未來」風險
    csv_path = os.path.join('data', 'raw', 'BTCUSDT_1H.csv')

    if os.path.exists(csv_path):
        df = load_and_clean_data(csv_path)
        print(f"✅ 成功載入 {len(df)} 筆數據")
    else:
        print(f"⚠️  警告: 找不到檔案 {csv_path}")
        print("⚠️  切換至: 使用隨機假數據進行測試模式...")
        dates = pd.date_range(start='2023-01-01', periods=2000, freq='1h')
        df = pd.DataFrame(np.random.random((2000, 5)) * 1000 + 20000,
                          index=dates,
                          columns=['open', 'high', 'low', 'close', 'volume'])

    # 2. 特徵工程管線
    print("🔄 執行 Point-in-Time 數據對齊 (合成 4H/1D)...")
    df_aligned = synthesize_mtf_data(df)

    print("🏷️  生成 Triple Barrier 標籤...")
    df_labeled = apply_triple_barrier(df_aligned,
                                      horizon=CONFIG['horizon'],
                                      atr_period=CONFIG['atr_period'],
                                      pt_mul=CONFIG['pt_mul'],
                                      sl_mul=CONFIG['sl_mul'])

    print("⚖️  執行數據標準化...")
    df_final = prepare_features(df_labeled,
                                method=CONFIG['norm_method'],
                                window=30)

    # 檢查是否有標籤 (因為 Triple Barrier 在最後 horizon 根 K 棒會是 NaN 或 0)
    # 我們移除無法標記的尾部數據
    df_final = df_final.dropna()

    # 3. 建立資料集與 DataLoader
    # 簡單的時間序列切分: 前 80% 訓練, 後 20% 驗證
    split_idx = int(len(df_final) * 0.8)
    train_df = df_final.iloc[:split_idx]
    val_df = df_final.iloc[split_idx:]

    print(f"📊 訓練集樣本數: {len(train_df)}, 驗證集樣本數: {len(val_df)}")

    train_dataset = CryptoTimeSeriesDataset(train_df,
                                            seq_len=CONFIG['seq_len'])
    val_dataset = CryptoTimeSeriesDataset(val_df, seq_len=CONFIG['seq_len'])

    # 如果數據量太少導致 dataset 為空，做個保護
    if len(train_dataset) == 0:
        print("❌ 錯誤: 有效數據量不足以建立序列，請檢查 seq_len 或數據長度。")
        return

    train_loader = DataLoader(train_dataset,
                              batch_size=CONFIG['batch_size'],
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=CONFIG['batch_size'],
                            shuffle=False)

    # 4. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 使用運算裝置: {device}")

    model = ParallelTCNAlphaHunter(input_features=5, num_classes=3).to(device)

    # 5. 設定 Loss 與 Optimizer
    # 根據你的數據，這裡的權重可能需要根據實際 Label 分佈調整
    # 你可以先跑一次 features/labeling.py 看一下分佈
    focal_loss = FocalLoss(alpha=torch.tensor([0.5, 1.0, 1.0]).to(device),
                           gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # 6. 訓練迴圈
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        train_preds, train_targets = [], []

        for batch in train_loader:
            x_1h = batch['1h'].to(device)
            x_4h = batch['4h'].to(device)
            x_1d = batch['1d'].to(device)
            y = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(x_1h, x_4h, x_1d)
            loss = focal_loss(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.append(logits.detach())
            train_targets.append(y)

        # 計算訓練集 MCC
        if len(train_preds) > 0:
            train_preds = torch.cat(train_preds)
            train_targets = torch.cat(train_targets)
            train_mcc = calculate_mcc(train_preds, train_targets)
        else:
            train_mcc = 0

        # 驗證
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                x_1h = batch['1h'].to(device)
                x_4h = batch['4h'].to(device)
                x_1d = batch['1d'].to(device)
                y = batch['label'].to(device)

                logits = model(x_1h, x_4h, x_1d)
                val_preds.append(logits)
                val_targets.append(y)

        if len(val_preds) > 0:
            val_preds = torch.cat(val_preds)
            val_targets = torch.cat(val_targets)
            val_mcc = calculate_mcc(val_preds, val_targets)
        else:
            val_mcc = 0

        print(
            f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {train_loss/len(train_loader):.4f} | Train MCC: {train_mcc:.3f} | Val MCC: {val_mcc:.3f}"
        )


if __name__ == "__main__":
    train_model()
