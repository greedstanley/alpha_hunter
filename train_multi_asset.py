import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset # 引入 ConcatDataset
import pandas as pd
import numpy as np
import os

# 導入模組
from features.alignment import synthesize_mtf_data
from features.labeling import apply_triple_barrier
from features.preprocess import prepare_features
from models.tcn_core import ParallelTCNAlphaHunter
from utils.loss import FocalLoss, calculate_mcc
from data.dataset import CryptoTimeSeriesDataset
from train import load_and_clean_data, CONFIG # 複用設定

def process_single_asset(filepath):
    """
    對單一幣種進行完整的特徵工程流程
    """
    if not os.path.exists(filepath):
        print(f"⚠️ 找不到檔案: {filepath}，跳過。")
        return None, None

    df = load_and_clean_data(filepath)
    
    # 1. 對齊 (合成 4H/1D)
    df_aligned = synthesize_mtf_data(df)
    
    # 2. 標籤 (Triple Barrier)
    # 注意：ATR 會自動適應不同幣種的價格 scale，所以這裡參數不用改
    df_labeled = apply_triple_barrier(df_aligned, horizon=CONFIG['horizon'], atr_period=CONFIG['atr_period'])
    
    # 3. 標準化 (Z-Score/LogReturn)
    # 這是關鍵！因為做了標準化，BTC 和 ETH 的數值分佈會變得一樣，可以混合訓練
    df_final = prepare_features(df_labeled, method=CONFIG['norm_method'], window=30)
    df_final = df_final.dropna()
    
    # 4. 切分訓練/驗證
    split_idx = int(len(df_final) * 0.8)
    train_df = df_final.iloc[:split_idx]
    val_df = df_final.iloc[split_idx:]
    
    return train_df, val_df

def train_multi_asset_model():
    print(f"🚀 啟動 Alpha Hunter [多幣種] 訓練程序...")
    print(f"⚙️  Epochs={CONFIG['epochs']}, Norm={CONFIG['norm_method']}")
    
    # --- 定義要訓練的幣種清單 ---
    # 請確保 data/raw/ 資料夾下有這些檔案
    asset_files = [
        os.path.join('data', 'raw', 'BTCUSDT_1H.csv'),
        os.path.join('data', 'raw', 'ETHUSDT_1H.csv'),
        os.path.join('data', 'raw', 'BNBUSDT_1H.csv'),
        os.path.join('data', 'raw', 'SOLUSDT_1H.csv'),
        # 你之後可以下載 SOLUSDT, BNBUSDT 等加進來
    ]
    
    train_datasets = []
    val_datasets = []
    
    for filepath in asset_files:
        print(f"\n🔄 處理資產: {os.path.basename(filepath)} ...")
        t_df, v_df = process_single_asset(filepath)
        
        if t_df is not None:
            # 為每個幣種建立獨立的 Dataset (確保時間序列不中斷)
            train_datasets.append(CryptoTimeSeriesDataset(t_df, seq_len=CONFIG['seq_len']))
            val_datasets.append(CryptoTimeSeriesDataset(v_df, seq_len=CONFIG['seq_len']))
            print(f"   Samples -> Train: {len(t_df)}, Val: {len(v_df)}")
            
    if not train_datasets:
        print("❌ 沒有有效的訓練數據，程式終止。")
        return

    # --- 關鍵一步：合併數據集 ---
    # ConcatDataset 會把多個 Dataset 虛擬地接在一起，讓 DataLoader 以為這是一個大資料庫
    combined_train_dataset = ConcatDataset(train_datasets)
    combined_val_dataset = ConcatDataset(val_datasets)
    
    print(f"\n📊 [總計] 訓練樣本數: {len(combined_train_dataset)}, 驗證樣本數: {len(combined_val_dataset)}")
    
    # 建立 DataLoader (混合了所有幣種的數據)
    train_loader = DataLoader(combined_train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(combined_val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # --- 以下模型初始化與訓練邏輯與原本相同 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 使用裝置: {device}")
    
    model = ParallelTCNAlphaHunter(input_features=5, num_classes=3).to(device)
    focal_loss = FocalLoss(alpha=torch.tensor([0.5, 1.0, 1.0]).to(device), gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    checkpoint_dir = os.path.join('models', 'checkpoints')
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
    best_val_mcc = -float('inf')

    # ... (這裡省略重複的 Resume 邏輯，與 train.py 相同) ...
    # 為了簡潔，這裡直接開始訓練 loop

    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        train_preds, train_targets = [], []
        
        for batch in train_loader:
            x_1h, x_4h, x_1d, y = batch['1h'].to(device), batch['4h'].to(device), batch['1d'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(x_1h, x_4h, x_1d)
            loss = focal_loss(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.append(logits.detach())
            train_targets.append(y)
            
        train_mcc = calculate_mcc(torch.cat(train_preds), torch.cat(train_targets)) if train_preds else 0
        
        # 驗證
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                x_1h, x_4h, x_1d, y = batch['1h'].to(device), batch['4h'].to(device), batch['1d'].to(device), batch['label'].to(device)
                logits = model(x_1h, x_4h, x_1d)
                val_preds.append(logits)
                val_targets.append(y)
        
        val_mcc = calculate_mcc(torch.cat(val_preds), torch.cat(val_targets)) if val_preds else 0
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {train_loss/len(train_loader):.4f} | Train MCC: {train_mcc:.3f} | Val MCC: {val_mcc:.3f}")
        
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"    🔥 新高點 (MCC: {val_mcc:.3f}) -> 模型已更新")

if __name__ == "__main__":
    train_multi_asset_model()