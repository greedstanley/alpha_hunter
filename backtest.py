import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from features.alignment import synthesize_mtf_data
from features.preprocess import prepare_features
from models.tcn_core import ParallelTCNAlphaHunter
from data.dataset import CryptoTimeSeriesDataset
from train import load_and_clean_data, CONFIG

def run_vectorized_backtest(asset_name='BTCUSDT', fee_rate=0.001):
    """
    向量化回測：快速驗證模型在單一資產上的績效
    Args:
        asset_name: 資產名稱
        fee_rate: 手續費率 (0.001 = 0.1%)
    """
    print(f"🧪 開始回測: {asset_name} | 手續費: {fee_rate*100}%")
    
    # 1. 載入數據
    filepath = os.path.join('data', 'raw', f'{asset_name}_1H.csv')
    if not os.path.exists(filepath):
        print(f"❌ 找不到數據: {filepath}")
        return

    df = load_and_clean_data(filepath)
    
    # 2. 特徵工程 (必須與訓練時完全一致)
    print("🔄 處理特徵...")
    df_aligned = synthesize_mtf_data(df)
    # 注意：回測不需要 Triple Barrier Label，但為了 reuse code，我們直接做 prepare_features
    # 我們需要保留 close price 來計算損益
    raw_close = df_aligned['close'].copy()
    
    df_features = prepare_features(df_aligned, method=CONFIG['norm_method'], window=30)
    df_features = df_features.dropna()
    
    # 對齊 raw_close (因為 prepare_features 可能會因為 window 而 drop 前面的數據)
    raw_close = raw_close.loc[df_features.index]
    
    # 3. 載入模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ParallelTCNAlphaHunter(input_features=5, num_classes=3).to(device)
    
    checkpoint_path = os.path.join('models', 'checkpoints', 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ 模型載入成功")
    else:
        print("❌ 找不到模型 Checkpoint")
        return

    model.eval()
    
    # 4. 批量預測 (Batch Inference)
    # 為了節省記憶體，我們還是用 DataLoader，但不用 shuffle
    dataset = CryptoTimeSeriesDataset(df_features, seq_len=CONFIG['seq_len'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    
    all_preds = []
    
    print("🔮 執行推論...")
    with torch.no_grad():
        for batch in loader:
            x_1h = batch['1h'].to(device)
            x_4h = batch['4h'].to(device)
            x_1d = batch['1d'].to(device)
            
            logits = model(x_1h, x_4h, x_1d)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            
    # 5. 計算回測邏輯
    # dataset[i] 對應的是 df_features[i + seq_len] 的時間點 (因為 seq_len 窗口)
    # 預測的是 "下一根 K 棒" 的動作? 
    # 根據 labeling.py，Triple Barrier 是標記 "未來"。
    # 所以當我們在 T 時刻預測出訊號，我們是在 T+1 開盤執行。
    
    # 對齊長度
    valid_len = len(all_preds)
    # 取得對應的收盤價 (用於計算 return)
    # CryptoTimeSeriesDataset 的 index 0 對應原始 df 的 seq_len 處
    # 我們預測的訊號是用於 "未來" 的
    
    analysis_df = pd.DataFrame(index=df_features.index[CONFIG['seq_len']:])
    # 確保長度一致 (有些微差距需裁切)
    analysis_df = analysis_df.iloc[:valid_len]
    analysis_df['close'] = raw_close.iloc[CONFIG['seq_len']:].iloc[:valid_len].values
    analysis_df['signal_idx'] = all_preds # 0: Hold, 1: Buy, 2: Sell
    
    # 映射訊號: 0->0, 1->1, 2->-1
    signal_map = {0: 0, 1: 1, 2: -1}
    analysis_df['position'] = analysis_df['signal_idx'].map(signal_map)
    
    # 計算市場回報 (Log Return)
    analysis_df['market_return'] = np.log(analysis_df['close'] / analysis_df['close'].shift(1)).fillna(0)
    
    # 策略回報 = 持倉 * 市場回報
    # 注意：今日的訊號 (Position) 是基於昨日數據預測的，所以是用來吃今日的 Market Return
    # 但代碼中 output 是對應當下 window 的預測。如果是預測未來，我們應該 shift(1) position?
    # 假設模型是預測 "下一根":
    analysis_df['strategy_return'] = analysis_df['position'].shift(1) * analysis_df['market_return']
    
    # 計算手續費 (只有當持倉改變時才扣費)
    analysis_df['position_change'] = analysis_df['position'].diff().abs().fillna(0)
    # 簡化：每次變動都視為開倉或平倉，扣手續費
    analysis_df['fees'] = analysis_df['position_change'] * fee_rate
    
    analysis_df['net_return'] = analysis_df['strategy_return'] - analysis_df['fees']
    
    # 累計回報
    analysis_df['cum_market_return'] = analysis_df['market_return'].cumsum().apply(np.exp)
    analysis_df['cum_strategy_return'] = analysis_df['net_return'].cumsum().apply(np.exp)
    
    # 6. 績效指標
    total_ret = analysis_df['cum_strategy_return'].iloc[-1] - 1
    sharpe = analysis_df['net_return'].mean() / (analysis_df['net_return'].std() + 1e-9) * np.sqrt(24*365) # 年化
    win_rate = (analysis_df['net_return'] > 0).mean()
    
    print("\n" + "="*30)
    print(f"📊 回測結果: {asset_name}")
    print(f"   總回報: {total_ret*100:.2f}%")
    print(f"   夏普率: {sharpe:.2f}")
    print(f"   交易勝率: {win_rate*100:.2f}% (含 Hold)")
    print(f"   Buy & Hold 回報: {(analysis_df['cum_market_return'].iloc[-1]-1)*100:.2f}%")
    print("="*30 + "\n")

    # 繪圖
    plt.figure(figsize=(12, 6))
    plt.plot(analysis_df.index, analysis_df['cum_market_return'], label='Buy & Hold', alpha=0.5)
    plt.plot(analysis_df.index, analysis_df['cum_strategy_return'], label='Alpha Hunter', linewidth=2)
    plt.title(f'Alpha Hunter Strategy Equity Curve ({asset_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'backtest_{asset_name}.png')
    print(f"📈 權益曲線已保存至 backtest_{asset_name}.png")

if __name__ == "__main__":
    run_vectorized_backtest('BTCUSDT')
    # run_vectorized_backtest('ETHUSDT')


### 📅 PM 與 QA 規劃 (Management)

#### 1. Colab 部署指南 (Production Deployment)
# * **上傳方式：** 將 `train_multi_asset.py` 與整個 `alpha_hunter` 資料夾直接拖入 Colab 左側的檔案區，或者掛載 Google Drive。
# * **執行指令：**
#     ```python
#     !pip install ta-lib # 如果有用到 ta-lib
#     !python train_multi_asset.py