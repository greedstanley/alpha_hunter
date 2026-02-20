import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import pandas as pd
import os
import glob
import argparse

# --- 模組匯入 ---
from features.pipeline import process_single_asset, PIPELINE_CONFIG
from models.tcn_core import ParallelTCNAlphaHunter
from utils.loss import FocalLoss, calculate_mcc
from data.dataset import CryptoTimeSeriesDataset

# --- 訓練專用設定 ---
TRAIN_CONFIG = {
    'batch_size': 64,
    'epochs': 150,
    'learning_rate': 1e-3,
}

def save_checkpoint(model, optimizer, scheduler, epoch, val_mcc, filename, quiet=False):
    """儲存模型檢查點，包含訓練狀態和設定"""
    full_config = {**PIPELINE_CONFIG, **TRAIN_CONFIG}
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_mcc': val_mcc,
        'config': full_config
    }
    torch.save(state, filename)
    if not quiet:
        print(f"    💾 Checkpoint saved: {filename} (MCC: {val_mcc:.4f})")

# [重構] 函式簽名新增 use_processed_data 參數
def train_multi_asset_model(data_directory, model_save_directory, resume=False, additional_epochs=0, save_csv=False, use_processed_data=False):
    """多資產模型訓練的主函式"""
    print(f"🚀 啟動 Alpha Hunter [多幣種/Colab版] 訓練程序...")
    
    train_datasets, val_datasets = [], []

    # [新增] 根據 use_processed_data 旗標決定數據載入策略
    if use_processed_data:
        print("💡 模式: 從 'data/processed/' 目錄直接載入預處理數據...")
        processed_train_path = os.path.join(data_directory.replace('raw', 'processed'), 'train', '*_train_processed.csv')
        train_files = glob.glob(processed_train_path)

        if not train_files:
            print(f"❌ 錯誤: 在 '{processed_train_path}' 中找不到任何預處理過的訓練檔案。")
            return

        print(f"📋 偵測到已處理的檔案: {[os.path.basename(f) for f in train_files]}")

        for train_filepath in train_files:
            symbol = os.path.basename(train_filepath).split('_')[0]
            print(f"    - 正在載入 {symbol}...")
            
            val_filepath = os.path.join(data_directory.replace('raw', 'processed'), 'validation', f"{symbol}_validation_processed.csv")

            if not os.path.exists(val_filepath):
                print(f"   ⚠️ 警告: 找不到對應的驗證檔案: {val_filepath}，跳過此資產。")
                continue

            try:
                t_df = pd.read_csv(train_filepath, index_col=0, parse_dates=True)
                v_df = pd.read_csv(val_filepath, index_col=0, parse_dates=True)

                if t_df is not None and len(t_df) > PIPELINE_CONFIG['seq_len']:
                    train_datasets.append(CryptoTimeSeriesDataset(t_df, seq_len=PIPELINE_CONFIG['seq_len']))
                    val_datasets.append(CryptoTimeSeriesDataset(v_df, seq_len=PIPELINE_CONFIG['seq_len']))
            except Exception as e:
                print(f"   ❌ 載入預處理檔案時出錯 ({symbol}): {e}")

    else:
        print("💡 模式: 從 'data/raw/' 目錄讀取原始數據並即時處理...")
        asset_files = glob.glob(os.path.join(data_directory, '*_1H.csv'))
        if not asset_files:
            print(f"❌ 錯誤: 在 '{data_directory}' 中找不到任何 *_1H.csv 檔案。請檢查路徑。")
            return
        
        print(f"📋 偵測到原始檔案: {[os.path.basename(f) for f in asset_files]}")

        for filepath in asset_files:
            t_df, v_df = process_single_asset(filepath, config=PIPELINE_CONFIG, save_csv=save_csv)
            if t_df is not None and len(t_df) > PIPELINE_CONFIG['seq_len']:
                train_datasets.append(CryptoTimeSeriesDataset(t_df, seq_len=PIPELINE_CONFIG['seq_len']))
                val_datasets.append(CryptoTimeSeriesDataset(v_df, seq_len=PIPELINE_CONFIG['seq_len']))

    # --- 後續的訓練邏輯保持不變 ---
            
    if not train_datasets:
        print("❌ 無有效數據可供訓練，終止。")
        return

    all_dims = [ds.get_input_dim() for ds in train_datasets]
    global_max_dim = max(all_dims)
    print(f"🧠 全域特徵維度對齊: {all_dims} -> 統一為 {global_max_dim}")

    for ds in train_datasets: ds.set_target_dim(global_max_dim)
    for ds in val_datasets: ds.set_target_dim(global_max_dim)

    combined_train, combined_val = ConcatDataset(train_datasets), ConcatDataset(val_datasets)
    train_loader = DataLoader(combined_train, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(combined_val, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False, num_workers=2)
    
    print(f"📊 總訓練樣本: {len(combined_train)} | 總驗證樣本: {len(combined_val)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    
    model = ParallelTCNAlphaHunter(input_features=global_max_dim, num_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)
    focal_loss = FocalLoss(alpha=torch.tensor([0.5, 1.0, 1.0]).to(device), gamma=2.0)
    
    checkpoint_dir = model_save_directory
    if not os.path.exists(checkpoint_dir): 
        print(f"📂 正在建立模型儲存目錄: {checkpoint_dir}")
        os.makedirs(checkpoint_dir)
    
    best_model_path = os.path.join(checkpoint_dir, 'best_model_multi_asset.pth')
    latest_model_path = os.path.join(checkpoint_dir, 'latest_model_multi_asset.pth')
    
    start_epoch = 0
    best_val_mcc = -1.0
    
    if resume and os.path.exists(best_model_path):
        print(f"🔄 載入 Checkpoint: {best_model_path}")
        try:
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                try: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except: print("⚠️ Scheduler 狀態載入失敗，將使用新設定。")
            start_epoch = checkpoint['epoch'] + 1
            best_val_mcc = checkpoint.get('val_mcc', 0.0)
            print(f"   ✅ 恢復狀態。上次停止於 Epoch {checkpoint['epoch']}, Best MCC: {best_val_mcc:.4f}")
        except Exception as e:
            print(f"❌ 載入失敗 ({e})，重新開始訓練。")
            
    total_epochs = TRAIN_CONFIG['epochs']
    if resume:
        if additional_epochs > 0: total_epochs = start_epoch + additional_epochs
        else:
            if start_epoch >= TRAIN_CONFIG['epochs']: total_epochs = start_epoch + 20 
            else: total_epochs = TRAIN_CONFIG['epochs']
        print(f"🎯 續訓模式: {start_epoch} -> {total_epochs}")
    else:
        print(f"🎯 全新訓練: 目標 {total_epochs} Epochs")

    for epoch in range(start_epoch, total_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x_1h, x_4h, x_1d, y = batch['1h'].to(device), batch['4h'].to(device), batch['1d'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(x_1h, x_4h, x_1d)
            loss = focal_loss(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss_avg = train_loss / len(train_loader)
        
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                x_1h, x_4h, x_1d, y = batch['1h'].to(device), batch['4h'].to(device), batch['1d'].to(device), batch['label'].to(device)
                logits = model(x_1h, x_4h, x_1d)
                val_preds.append(logits)
                val_targets.append(y)
        
        val_mcc = calculate_mcc(torch.cat(val_preds), torch.cat(val_targets)) if val_preds else 0.0
        scheduler.step(val_mcc)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{total_epochs} | Loss: {train_loss_avg:.4f} | Val MCC: {val_mcc:.4f} (Best: {best_val_mcc:.4f}) | LR: {current_lr:.1e}")
        
        save_checkpoint(model, optimizer, scheduler, epoch, val_mcc, latest_model_path, quiet=True)
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            save_checkpoint(model, optimizer, scheduler, epoch, val_mcc, best_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha Hunter Multi-Asset Training Script for Colab", formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='[必須] 包含數據檔案的目錄路徑。\n'
                             '預設模式下，這是 data/raw 的路徑。\n'
                             '若使用 --use_processed_data，這仍是 data/raw 的路徑，腳本會自動推導出 data/processed。')
    
    parser.add_argument('--model_dir', type=str, required=True, 
                        help='[必須] 用於儲存訓練好的模型和檢查點的目錄路徑。')

    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='從現有的最佳模型檢查點恢復訓練。(預設)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='強制從頭開始一個全新的訓練。')
    parser.set_defaults(resume=True)

    parser.add_argument('--save_csv', action='store_true',
                        help='(僅在即時處理模式下有效) 將處理後的數據集儲存為CSV檔案。')
    
    # [新增] --use_processed_data 旗標
    parser.add_argument('--use_processed_data', action='store_true',
                        help='啟用此旗標以直接從 data/processed/ 載入數據，跳過即時處理。')

    args = parser.parse_args()

    train_multi_asset_model(
        data_directory=args.data_dir,
        model_save_directory=args.model_dir,
        resume=args.resume,
        save_csv=args.save_csv,
        use_processed_data=args.use_processed_data # [新增] 傳遞新參數
    )
