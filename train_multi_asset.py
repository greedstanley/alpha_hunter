import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import pandas as pd
import numpy as np
import os
import glob

# 導入模組
from features.alignment import synthesize_mtf_data
from features.labeling import apply_triple_barrier
from features.preprocess import prepare_features
from models.tcn_core import ParallelTCNAlphaHunter
from utils.loss import FocalLoss, calculate_mcc
from data.dataset import CryptoTimeSeriesDataset
from train import load_and_clean_data  # 複用函數

# 重新定義 CONFIG 以便於此檔案獨立運行
CONFIG = {
    'seq_len': 60,
    'norm_method': 'z_score',
    'batch_size': 64,
    'epochs': 20,          # 這是預設的「單次」訓練目標
    'learning_rate': 1e-3,
    'atr_period': 14,
    'horizon': 60,
    'pt_mul': 2.0,
    'sl_mul': 2.0,
}

def process_single_asset(filepath):
    """處理單一幣種數據"""
    if not os.path.exists(filepath):
        print(f"⚠️ 找不到檔案: {filepath}，跳過。")
        return None, None

    df = load_and_clean_data(filepath)
    df_aligned = synthesize_mtf_data(df)
    # 這裡的 horizon 用於標籤生成，跟回測無關
    df_labeled = apply_triple_barrier(df_aligned, horizon=CONFIG['horizon'], atr_period=CONFIG['atr_period'])
    df_final = prepare_features(df_labeled, method=CONFIG['norm_method'], window=30)
    df_final = df_final.dropna()
    
    # 簡單的時間切分
    split_idx = int(len(df_final) * 0.8)
    train_df = df_final.iloc[:split_idx]
    val_df = df_final.iloc[split_idx:]
    
    return train_df, val_df

def save_checkpoint(model, optimizer, epoch, val_mcc, filename):
    """保存完整的訓練狀態"""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_mcc': val_mcc,
        'config': CONFIG
    }
    torch.save(state, filename)
    print(f"    💾 Checkpoint saved: {filename} (MCC: {val_mcc:.4f})")

def train_multi_asset_model(resume=False, additional_epochs=0):
    """
    Args:
        resume (bool): 是否從 best_model.pth 恢復訓練
        additional_epochs (int): 如果是 resume，要額外再訓練多少個 epochs
    """
    print(f"🚀 啟動 Alpha Hunter [多幣種] 訓練程序...")
    
    # 1. 準備數據
    # 搜尋 data/raw 下所有的 _1H.csv 檔案
    asset_files = glob.glob(os.path.join('data', 'raw', '*_1H.csv'))
    if not asset_files:
        # Fallback for explicit list if glob fails or folder structure differs
        asset_files = [
            os.path.join('data', 'raw', 'BTCUSDT_1H.csv'),
            os.path.join('data', 'raw', 'ETHUSDT_1H.csv'),
        ]
    
    print(f"📋 偵測到資產檔案: {[os.path.basename(f) for f in asset_files]}")

    train_datasets = []
    val_datasets = []
    
    for filepath in asset_files:
        t_df, v_df = process_single_asset(filepath)
        if t_df is not None and len(t_df) > CONFIG['seq_len']:
            train_datasets.append(CryptoTimeSeriesDataset(t_df, seq_len=CONFIG['seq_len']))
            val_datasets.append(CryptoTimeSeriesDataset(v_df, seq_len=CONFIG['seq_len']))
            
    if not train_datasets:
        print("❌ 無有效數據，終止。")
        return

    combined_train = ConcatDataset(train_datasets)
    combined_val = ConcatDataset(val_datasets)
    
    train_loader = DataLoader(combined_train, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0) # Windows/Colab有时设 num_workers=0 更稳
    val_loader = DataLoader(combined_val, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    print(f"📊 總訓練樣本: {len(combined_train)} | 總驗證樣本: {len(combined_val)}")

    # 2. 初始化模型與環境
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    
    model = ParallelTCNAlphaHunter(input_features=5, num_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    focal_loss = FocalLoss(alpha=torch.tensor([0.5, 1.0, 1.0]).to(device), gamma=2.0)
    
    checkpoint_dir = os.path.join('models', 'checkpoints')
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
    
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    # 3. 斷點續訓邏輯
    start_epoch = 0
    best_val_mcc = -1.0 # Initialize low
    
    if resume and os.path.exists(best_model_path):
        print(f"🔄 載入 Checkpoint: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        
        # 兼容性檢查：確認 checkpoint 格式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_mcc = checkpoint.get('val_mcc', 0.0)
            print(f"   ✅ 成功恢復狀態。上次停止於 Epoch {checkpoint['epoch']}, Best MCC: {best_val_mcc:.4f}")
        else:
            # 舊版只有 state_dict 的情況
            model.load_state_dict(checkpoint)
            print("   ⚠️ 僅載入權重 (舊版格式)，Optimizer 狀態已重置。")
            
    # 設定總目標 Epochs
    total_epochs = CONFIG['epochs']
    if resume:
        total_epochs = start_epoch + additional_epochs
        print(f"🎯 續訓模式: 目標從 Epoch {start_epoch} 練到 {total_epochs}")
    else:
        print(f"🎯 全新訓練: 目標 {total_epochs} Epochs")

    # 4. 訓練迴圈
    for epoch in range(start_epoch, total_epochs):
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
            
        # 快速計算 Train MCC (使用 GPU tensor 運算避免 CPU copy 開銷)
        train_loss_avg = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                x_1h, x_4h, x_1d, y = batch['1h'].to(device), batch['4h'].to(device), batch['1d'].to(device), batch['label'].to(device)
                logits = model(x_1h, x_4h, x_1d)
                val_preds.append(logits)
                val_targets.append(y)
        
        if val_preds:
            val_all_preds = torch.cat(val_preds)
            val_all_targets = torch.cat(val_targets)
            val_mcc = calculate_mcc(val_all_preds, val_all_targets)
        else:
            val_mcc = 0.0

        # 因為 calculate_mcc 可能需要 CPU，這裡簡化 log
        print(f"Epoch {epoch+1}/{total_epochs} | Loss: {train_loss_avg:.4f} | Val MCC: {val_mcc:.4f}")
        
        # 保存最佳模型
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            save_checkpoint(model, optimizer, epoch, val_mcc, best_model_path)

if __name__ == "__main__":
    # 使用範例：
    # 1. 全新訓練 20 epochs
    # train_multi_asset_model(resume=False)
    
    # 2. 續訓：假設之前跑了 20，現在想再加 50 (總共到 70)
    train_multi_asset_model(resume=True, additional_epochs=50)