import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np

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


def train_model():
    print(f"🚀 啟動 Alpha Hunter 訓練程序...")
    print(f"⚙️  設定: Seq_Len={CONFIG['seq_len']}, Norm={CONFIG['norm_method']}")

    # 1. 載入數據 (這裡假設你有一個 csv，請替換成你的真實路徑)
    # df = pd.read_csv('data/raw/BTCUSDT_1H.csv', index_col='datetime', parse_dates=True)
    # 為了演示，我們生成假數據
    print("⚠️  使用隨機假數據進行測試 (請替換為真實數據)...")
    dates = pd.date_range(start='2023-01-01', periods=2000, freq='1h')
    df = pd.DataFrame(np.random.random((2000, 5)) * 1000 + 20000,
                      index=dates,
                      columns=['open', 'high', 'low', 'close', 'volume'])

    # 2. 特徵工程管線
    print("🔄 執行 Point-in-Time 數據對齊...")
    df_aligned = synthesize_mtf_data(df)

    print("🏷️  生成 Triple Barrier 標籤...")
    df_labeled = apply_triple_barrier(df_aligned,
                                      horizon=CONFIG['horizon'],
                                      atr_period=CONFIG['atr_period'],
                                      pt_mul=CONFIG['pt_mul'],
                                      sl_mul=CONFIG['sl_mul'])

    print("Scale  執行數據標準化...")
    df_final = prepare_features(df_labeled,
                                method=CONFIG['norm_method'],
                                window=30)

    # 3. 建立資料集與 DataLoader
    # 簡單的時間序列切分: 前 80% 訓練, 後 20% 驗證 (不使用隨機切分以防漏題)
    split_idx = int(len(df_final) * 0.8)
    train_df = df_final.iloc[:split_idx]
    val_df = df_final.iloc[split_idx:]

    train_dataset = CryptoTimeSeriesDataset(train_df,
                                            seq_len=CONFIG['seq_len'])
    val_dataset = CryptoTimeSeriesDataset(val_df, seq_len=CONFIG['seq_len'])

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'],
        shuffle=True)  # 訓練集可以 Shuffle batch，因為順序在 Dataset 內部已經保留
    val_loader = DataLoader(val_dataset,
                            batch_size=CONFIG['batch_size'],
                            shuffle=False)

    # 4. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ParallelTCNAlphaHunter(input_features=5, num_classes=3).to(device)

    # 5. 設定 Loss 與 Optimizer
    # alpha: 設定類別權重 (解決不平衡), gamma: 專注難樣本
    # 假設類別分佈: Hold(0): 70%, Buy(1): 15%, Sell(2): 15% -> 權重設為 [0.3, 1.0, 1.0]
    focal_loss = FocalLoss(alpha=torch.tensor([0.3, 1.0, 1.0]).to(device),
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
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_mcc = calculate_mcc(train_preds, train_targets)

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

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_mcc = calculate_mcc(val_preds, val_targets)

        print(
            f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {train_loss/len(train_loader):.4f} | Train MCC: {train_mcc:.3f} | Val MCC: {val_mcc:.3f}"
        )


if __name__ == "__main__":
    train_model()
