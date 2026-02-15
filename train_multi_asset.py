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
from train import load_and_clean_data

CONFIG = {
    'seq_len': 60,
    'norm_method': 'z_score',
    'batch_size': 64,
    'epochs': 50,
    'learning_rate': 1e-3,
    'atr_period': 14,
    'horizon': 60,
    'pt_mul': 2.0,
    'sl_mul': 2.0,
}

def process_single_asset(filepath):
    if not os.path.exists(filepath):
        return None, None

    df = load_and_clean_data(filepath)
    df_aligned = synthesize_mtf_data(df)
    df_labeled = apply_triple_barrier(df_aligned, horizon=CONFIG['horizon'], atr_period=CONFIG['atr_period'])
    df_final = prepare_features(df_labeled, method=CONFIG['norm_method'], window=30)
    df_final = df_final.dropna()
    
    split_idx = int(len(df_final) * 0.8)
    train_df = df_final.iloc[:split_idx]
    val_df = df_final.iloc[split_idx:]
    
    return train_df, val_df

def save_checkpoint(model, optimizer, epoch, val_mcc, filename):
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
    print(f"🚀 啟動 Alpha Hunter [多幣種] 訓練程序...")
    
    asset_files = glob.glob(os.path.join('data', 'raw', '*_1H.csv'))
    if not asset_files:
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
            ds_train = CryptoTimeSeriesDataset(t_df, seq_len=CONFIG['seq_len'])
            ds_val = CryptoTimeSeriesDataset(v_df, seq_len=CONFIG['seq_len'])
            train_datasets.append(ds_train)
            val_datasets.append(ds_val)
            
    if not train_datasets:
        print("❌ 無有效數據，終止。")
        return

    # [關鍵修復] 1. 掃描所有 Dataset 找出「全域最大特徵數」
    all_dims = [ds.get_input_dim() for ds in train_datasets]
    global_max_dim = max(all_dims)
    print(f"🧠 全域特徵維度對齊 (Global Feature Align): {all_dims} -> 統一為 {global_max_dim}")

    # [關鍵修復] 2. 強制所有 Dataset 更新為全域維度 (Padding 不足的部分)
    for ds in train_datasets:
        ds.set_target_dim(global_max_dim)
    for ds in val_datasets:
        ds.set_target_dim(global_max_dim)

    # 合併 Dataset
    combined_train = ConcatDataset(train_datasets)
    combined_val = ConcatDataset(val_datasets)
    
    train_loader = DataLoader(combined_train, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(combined_val, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    print(f"📊 總訓練樣本: {len(combined_train)} | 總驗證樣本: {len(combined_val)}")

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    
    # 使用全域統一的維度
    model = ParallelTCNAlphaHunter(input_features=global_max_dim, num_classes=3).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    focal_loss = FocalLoss(alpha=torch.tensor([0.5, 1.0, 1.0]).to(device), gamma=2.0)
    
    checkpoint_dir = os.path.join('models', 'checkpoints')
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
    
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    start_epoch = 0
    best_val_mcc = -1.0
    
    if resume and os.path.exists(best_model_path):
        print(f"🔄 載入 Checkpoint: {best_model_path}")
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            current_model_dict = model.state_dict()
            loaded_state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            
            first_layer_key = 'tcn_1h.network.0.conv1.weight'
            if first_layer_key in loaded_state_dict and \
               loaded_state_dict[first_layer_key].shape != current_model_dict[first_layer_key].shape:
                print("⚠️  偵測到特徵維度改變。將捨棄舊權重，重新開始訓練。")
            else:
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    best_val_mcc = checkpoint.get('val_mcc', 0.0)
                    print(f"   ✅ 成功恢復狀態。上次停止於 Epoch {checkpoint['epoch']}, Best MCC: {best_val_mcc:.4f}")
                else:
                    model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"❌ 載入失敗 ({e})，將重新開始訓練。")
            
    total_epochs = CONFIG['epochs']
    if resume:
        total_epochs = start_epoch + additional_epochs

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
            
        train_loss_avg = train_loss / len(train_loader)
        
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

        print(f"Epoch {epoch+1}/{total_epochs} | Loss: {train_loss_avg:.4f} | Val MCC: {val_mcc:.4f}")
        
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            save_checkpoint(model, optimizer, epoch, val_mcc, best_model_path)

if __name__ == "__main__":
    train_multi_asset_model(resume=False)


