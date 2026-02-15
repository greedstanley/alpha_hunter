import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# 確保路徑正確，根據你的環境可能需要調整 import
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
    print(f"🧪 開始回測: {asset_name} | 手續費: {fee_rate*100:.2f}%")
    
    # 1. 載入數據
    filepath = os.path.join('data', 'raw', f'{asset_name}_1H.csv')
    if not os.path.exists(filepath):
        print(f"❌ 找不到數據: {filepath}")
        # Colab 路徑容錯 (有時候用戶會放在 content 根目錄)
        filepath = f'{asset_name}_1H.csv'
        if not os.path.exists(filepath):
            print(f"❌ 也找不到根目錄數據: {filepath}，終止。")
            return
        else:
            print(f"✅ 在根目錄找到數據: {filepath}")

    df = load_and_clean_data(filepath)
    
    # 2. 特徵工程 (必須與訓練時完全一致)
    print("🔄 處理特徵...")
    df_aligned = synthesize_mtf_data(df)
    
    # 保留 Close 用於計算損益 (需與 Feature 對齊)
    raw_close = df_aligned['close'].copy()
    
    # 生成特徵 (注意：回測時通常沒有 label，prepare_features 會處理特徵部分)
    df_features = prepare_features(df_aligned, method=CONFIG['norm_method'], window=30)
    df_features = df_features.dropna()
    
    # --- 關鍵修復：注入 Dummy Label ---
    # CryptoTimeSeriesDataset 預設需要 'label' 欄位，否則會報 KeyError
    if 'label' not in df_features.columns:
        # 填入 0 (Hold) 作為佔位符，這不會影響模型推論(Inference)
        df_features['label'] = 0
        print("🔧 已注入 Dummy Label 以符合 Dataset 格式要求")
    
    # 對齊 raw_close (因為 dropna 移除了部分數據)
    raw_close = raw_close.loc[df_features.index]
    
    # 3. 載入模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 使用裝置: {device}")
    
    model = ParallelTCNAlphaHunter(input_features=5, num_classes=3).to(device)
    
    # 支援多種路徑檢查
    possible_paths = [
        os.path.join('models', 'checkpoints', 'best_model.pth'),
        'best_model.pth', # Colab 根目錄
        '/content/models/checkpoints/best_model.pth'
    ]
    
    checkpoint_path = None
    for p in possible_paths:
        if os.path.exists(p):
            checkpoint_path = p
            break
            
    if checkpoint_path:
        print(f"🔄 載入模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ 模型載入成功")
    else:
        print("❌ 找不到模型 Checkpoint (best_model.pth)")
        return

    model.eval()
    
    # 4. 批量預測 (Batch Inference)
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
    # 注意: dataset[i] 的數據時間點是 T，標籤是對應 T+1 之後的未來
    # 我們的模型在 T 時刻給出預測，我們在 T+1 開盤執行
    
    valid_len = len(all_preds)
    
    # 分析用的 DataFrame (從 seq_len 之後開始)
    analysis_df = pd.DataFrame(index=df_features.index[CONFIG['seq_len']:])
    
    # 裁切長度以匹配預測結果
    analysis_df = analysis_df.iloc[:valid_len].copy()
    analysis_df['close'] = raw_close.iloc[CONFIG['seq_len']:].iloc[:valid_len].values
    analysis_df['signal_idx'] = all_preds 
    
    # 映射訊號: 0->0, 1->1 (Buy), 2->-1 (Sell)
    # 假設 dataset.py 裡的轉換邏輯是: -1 -> 2, 0 -> 0, 1 -> 1
    # 所以這裡要轉回來: 2 -> -1
    signal_map = {0: 0, 1: 1, 2: -1}
    analysis_df['position'] = analysis_df['signal_idx'].map(signal_map)
    
    # 計算市場回報 (Log Return)
    analysis_df['market_return'] = np.log(analysis_df['close'] / analysis_df['close'].shift(1)).fillna(0)
    
    # 策略回報
    # 關鍵：今天的 Position 是由昨天的數據預測出來的 (shift(1))
    # 這樣我們才能吃到今天的 market_return
    analysis_df['strategy_return'] = analysis_df['position'].shift(1) * analysis_df['market_return']
    
    # 計算手續費 (只有當持倉改變時才扣費)
    analysis_df['position_change'] = analysis_df['position'].diff().abs().fillna(0)
    analysis_df['fees'] = analysis_df['position_change'] * fee_rate
    
    analysis_df['net_return'] = analysis_df['strategy_return'] - analysis_df['fees']
    
    # 累計回報 (權益曲線)
    analysis_df['cum_market_return'] = analysis_df['market_return'].cumsum().apply(np.exp)
    analysis_df['cum_strategy_return'] = analysis_df['net_return'].cumsum().apply(np.exp)
    
    # 6. 績效指標
    total_ret = analysis_df['cum_strategy_return'].iloc[-1] - 1
    # 夏普率 (假設無風險利率為 0，按小時數據年化)
    sharpe = analysis_df['net_return'].mean() / (analysis_df['net_return'].std() + 1e-9) * np.sqrt(365*24)
    
    # 勝率 (不含 Hold)
    trade_returns = analysis_df[analysis_df['position'].shift(1) != 0]['net_return']
    win_rate = (trade_returns > 0).mean() if len(trade_returns) > 0 else 0
    
    print("\n" + "="*30)
    print(f"📊 回測結果: {asset_name}")
    print(f"   總回報: {total_ret*100:.2f}%")
    print(f"   夏普率: {sharpe:.2f}")
    print(f"   交易勝率: {win_rate*100:.2f}% (有開倉的時刻)")
    print(f"   Buy & Hold: {(analysis_df['cum_market_return'].iloc[-1]-1)*100:.2f}%")
    print("="*30 + "\n")

    # 繪圖
    plt.figure(figsize=(12, 6))
    plt.plot(analysis_df.index, analysis_df['cum_market_return'], label='Buy & Hold', alpha=0.5)
    plt.plot(analysis_df.index, analysis_df['cum_strategy_return'], label='Alpha Hunter', linewidth=2)
    plt.title(f'Alpha Hunter Strategy Equity Curve ({asset_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_img = f'backtest_{asset_name}.png'
    plt.savefig(output_img)
    print(f"📈 權益曲線已保存至 {output_img}")

if __name__ == "__main__":
    # run_vectorized_backtest('BTCUSDT')
    run_vectorized_backtest('ETHUSDT')