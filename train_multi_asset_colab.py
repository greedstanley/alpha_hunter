import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import os
import glob
import argparse

# --- 模組匯入 ---
# [重構] 從新的管線檔案匯入核心處理函式和其設定
from features.pipeline import process_single_asset, PIPELINE_CONFIG
from models.tcn_core import ParallelTCNAlphaHunter
from utils.loss import FocalLoss, calculate_mcc
from data.dataset import CryptoTimeSeriesDataset

# --- 訓練專用設定 ---
# 這些設定僅與模型訓練過程相關
TRAIN_CONFIG = {
    'batch_size': 64,
    'epochs': 150,
    'learning_rate': 1e-3,
}

# [重構] process_single_asset 函式已從此檔案移除，移至 features/pipeline.py

def save_checkpoint(model, optimizer, scheduler, epoch, val_mcc, filename, quiet=False):
    """儲存模型檢查點，包含訓練狀態和設定"""
    # [重構] 將管線和訓練設定合併後存檔
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

def train_multi_asset_model(data_directory, model_save_directory, resume=False, additional_epochs=0, save_csv=False):
    """多資產模型訓練的主函式"""
    print(f"🚀 啟動 Alpha Hunter [多幣種/Colab版] 訓練程序...")
    
    asset_files = glob.glob(os.path.join(data_directory, '*_1H.csv'))
    if not asset_files:
        print(f"❌ 錯誤: 在 '{data_directory}' 中找不到任何 *_1H.csv 檔案。請檢查路徑。")
        return
    
    print(f"📋 偵測到資產檔案: {[os.path.basename(f) for f in asset_files]}")

    train_datasets, val_datasets = [], []
    for filepath in asset_files:
        # [重構] 呼叫從外部匯入的中央處理函式
        t_df, v_df = process_single_asset(filepath, config=PIPELINE_CONFIG, save_csv=save_csv)
        
        # [重構] 使用從管線匯入的設定檔
        if t_df is not None and len(t_df) > PIPELINE_CONFIG['seq_len']:
            train_datasets.append(CryptoTimeSeriesDataset(t_df, seq_len=PIPELINE_CONFIG['seq_len']))
            val_datasets.append(CryptoTimeSeriesDataset(v_df, seq_len=PIPELINE_CONFIG['seq_len']))
            
    if not train_datasets:
        print("❌ 無有效數據可供訓練，終止。")
        return

    all_dims = [ds.get_input_dim() for ds in train_datasets]
    global_max_dim = max(all_dims)
    print(f"🧠 全域特徵維度對齊: {all_dims} -> 統一為 {global_max_dim}")

    for ds in train_datasets: ds.set_target_dim(global_max_dim)
    for ds in val_datasets: ds.set_target_dim(global_max_dim)

    combined_train = ConcatDataset(train_datasets)
    combined_val = ConcatDataset(val_datasets)
    
    # [重構] 使用訓練專用設定
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
            
    # [重構] 使用訓練專用設定
    total_epochs = TRAIN_CONFIG['epochs']
    if resume:
        if additional_epochs > 0:
            total_epochs = start_epoch + additional_epochs
        else:
            if start_epoch >= TRAIN_CONFIG['epochs']:
                total_epochs = start_epoch + 20 
            else:
                total_epochs = TRAIN_CONFIG['epochs']
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
    parser = argparse.ArgumentParser(description="Alpha Hunter Multi-Asset Training Script for Colab")
    
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='[必須] 包含原始CSV數據檔案的目錄路徑。')
    
    parser.add_argument('--model_dir', type=str, required=True, 
                        help='[必須] 用於儲存訓練好的模型和檢查點的目錄路徑。')

    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='從現有的最佳模型檢查點恢復訓練。(預設)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='強制從頭開始一個全新的訓練。')
    parser.set_defaults(resume=True)

    parser.add_argument('--save_csv', action='store_true',
                        help='將處理後的訓練數據集儲存為CSV檔案以供檢查。')

    args = parser.parse_args()

    train_multi_asset_model(
        data_directory=args.data_dir,
        model_save_directory=args.model_dir,
        resume=args.resume,
        save_csv=args.save_csv
    )
