import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F  # 新增

# 確保路徑正確
from features.alignment import synthesize_mtf_data
from features.preprocess import prepare_features
from models.tcn_core import ParallelTCNAlphaHunter
from data.dataset import CryptoTimeSeriesDataset
from train import load_and_clean_data, CONFIG

def run_vectorized_backtest(asset_name='BTCUSDT', fee_rate=0.001, threshold=0.0):
    """
    Args:
        threshold: 信心門檻 (0.0 代表不設限)。如果模型最大機率 < threshold，則強制 Hold。
    """
    print(f"🧪 開始回測: {asset_name} | 手續費: {fee_rate*100:.2f}% | 信心門檻: {threshold}")
    
    # 1. 載入與處理數據
    filepath = os.path.join('data', 'raw', f'{asset_name}_1H.csv')
    if not os.path.exists(filepath):
        # 嘗試直接讀取代碼所在目錄
        filepath = f'{asset_name}_1H.csv'
        if not os.path.exists(filepath):
            print(f"❌ 找不到數據: {filepath}")
            return

    df = load_and_clean_data(filepath)
    print("🔄 處理特徵...")
    df_aligned = synthesize_mtf_data(df)
    
    # 保留原始數據供後續分析
    raw_df = df_aligned[['open', 'high', 'low', 'close']].copy()
    
    df_features = prepare_features(df_aligned, method=CONFIG['norm_method'], window=30)
    df_features = df_features.dropna()
    
    # 注入 Dummy Label
    if 'label' not in df_features.columns:
        df_features['label'] = 0
    
    # 對齊原始價格
    raw_df = raw_df.loc[df_features.index]
    
    # 2. 載入模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 使用裝置: {device}")
    
    model = ParallelTCNAlphaHunter(input_features=5, num_classes=3).to(device)
    
    # 尋找模型路徑
    possible_paths = [
        os.path.join('models', 'checkpoints', 'best_model.pth'),
        'best_model.pth'
    ]
    checkpoint_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if checkpoint_path:
        print(f"🔄 載入模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    else:
        print("❌ 找不到 best_model.pth")
        return

    model.eval()
    
    # 3. 推論 (含信心度)
    dataset = CryptoTimeSeriesDataset(df_features, seq_len=CONFIG['seq_len'])
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    all_preds = []
    all_probs = [] # 儲存信心度
    
    print("🔮 執行推論...")
    with torch.no_grad():
        for batch in loader:
            x_1h = batch['1h'].to(device)
            x_4h = batch['4h'].to(device)
            x_1d = batch['1d'].to(device)
            
            logits = model(x_1h, x_4h, x_1d)
            probs = F.softmax(logits, dim=1) # 轉成機率
            
            # 取得最大機率與對應類別
            max_probs, preds = torch.max(probs, dim=1)
            
            # 如果信心不足，強制轉為 Hold (0)
            if threshold > 0:
                mask = max_probs < threshold
                preds[mask] = 0
                
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(max_probs.cpu().numpy())
            
    # 4. 構建詳細日誌 (Trade Log)
    valid_len = len(all_preds)
    # 時間索引從 seq_len 之後開始
    log_index = df_features.index[CONFIG['seq_len'] : CONFIG['seq_len']+valid_len]
    
    log_df = pd.DataFrame(index=log_index)
    log_df['Close'] = raw_df['close'].iloc[CONFIG['seq_len'] : CONFIG['seq_len']+valid_len].values
    log_df['Signal'] = all_preds
    log_df['Confidence'] = all_probs
    
    # 映射訊號: 0->0 (Hold), 1->1 (Long), 2->-1 (Short)
    log_df['Position'] = log_df['Signal'].map({0: 0, 1: 1, 2: -1})
    
    # 計算回報
    log_df['Market_Ret'] = np.log(log_df['Close'] / log_df['Close'].shift(1)).fillna(0)
    # 策略回報 = 昨天的部位 * 今天的漲跌
    log_df['Strategy_Ret'] = log_df['Position'].shift(1) * log_df['Market_Ret']
    
    # 計算手續費
    log_df['Pos_Change'] = log_df['Position'].diff().abs().fillna(0)
    log_df['Fees'] = log_df['Pos_Change'] * fee_rate
    log_df['Net_Ret'] = log_df['Strategy_Ret'] - log_df['Fees']
    
    # 累計淨值
    log_df['Equity'] = (1 + log_df['Net_Ret']).cumprod()
    log_df['Market_Equity'] = (1 + log_df['Market_Ret']).cumprod()
    
    # 5. 輸出報告與檔案
    total_ret = log_df['Equity'].iloc[-1] - 1
    mkt_ret = log_df['Market_Equity'].iloc[-1] - 1
    
    print("\n" + "="*30)
    print(f"📊 詳細回測報告: {asset_name}")
    print(f"   總回報: {total_ret*100:.2f}% (基準: {mkt_ret*100:.2f}%)")
    print(f"   總交易次數: {log_df['Pos_Change'].sum()/2:.0f}")
    print(f"   平均信心度: {np.mean(all_probs):.4f}")
    print("="*30)

    # 儲存詳細日誌 CSV
    csv_filename = f'backtest_log_{asset_name}.csv'
    log_df.to_csv(csv_filename)
    print(f"💾 交易日誌已儲存: {csv_filename} (請下載並用 Excel 打開分析)")
    
    # 繪圖
    plt.figure(figsize=(12, 6))
    plt.plot(log_df.index, log_df['Market_Equity'], label='Market', alpha=0.5, color='gray')
    plt.plot(log_df.index, log_df['Equity'], label='Strategy', linewidth=1.5, color='blue')
    plt.title(f'Equity Curve: {asset_name} (Thresh={threshold})')
    plt.yscale('log') # 使用對數坐標看清楚虧損
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'backtest_{asset_name}.png')

if __name__ == "__main__":
    # 嘗試提高門檻，減少隨機交易
    run_vectorized_backtest('BTCUSDT', fee_rate=0.001, threshold=0.0)
    run_vectorized_backtest('ETHUSDT', fee_rate=0.001, threshold=0.0)