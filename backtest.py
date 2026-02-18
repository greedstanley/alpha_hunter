import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 導入專案模組
from features.alignment import synthesize_mtf_data
from features.labeling import apply_triple_barrier # 雖然回測不看 Label，但 dataset 需要結構一致
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
    # 嘗試多種路徑讀取
    possible_files = [
        os.path.join('data', 'raw', f'{asset_name}_1H.csv'),
        f'{asset_name}_1H.csv'
    ]
    filepath = next((p for p in possible_files if os.path.exists(p)), None)
    
    if not filepath:
        print(f"❌ 找不到數據: {asset_name}_1H.csv")
        return

    df = load_and_clean_data(filepath)
    print("🔄 處理特徵 (Point-in-Time)...")
    df_aligned = synthesize_mtf_data(df)
    
    # 保留原始數據供後續分析 (Close Price)
    raw_close = df_aligned['close'].copy()
    raw_open = df_aligned['open'].copy()
    
    # 這裡的 apply_triple_barrier 主要是為了產生 'label' 欄位讓 dataset.py 不會報錯
    # 回測本身的損益計算不依賴這個 label
    df_labeled = apply_triple_barrier(df_aligned, horizon=CONFIG['horizon'], atr_period=CONFIG['atr_period'])
    
    # [關鍵] 使用與訓練一致的特徵工程
    df_features = prepare_features(df_labeled, method=CONFIG['norm_method'], window=30)
    df_features = df_features.dropna()
    
    # 對齊原始價格索引
    raw_close = raw_close.loc[df_features.index]
    
    # 2. 初始化 Dataset 以獲取正確的特徵維度
    dataset = CryptoTimeSeriesDataset(df_features, seq_len=CONFIG['seq_len'])
    
    # [修正點] 動態獲取特徵維度
    input_dims = dataset.get_input_dim()
    print(f"🧠 模型輸入特徵維度: {input_dims}")
    
    # 3. 載入模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 使用裝置: {device}")
    
    # 使用動態維度初始化
    model = ParallelTCNAlphaHunter(input_features=input_dims, num_classes=3).to(device)
    
    # 尋找模型路徑
    possible_paths = [
        os.path.join('models', 'checkpoints', 'best_model.pth'),
        'best_model.pth'
    ]
    checkpoint_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if checkpoint_path:
        print(f"🔄 載入權重: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            print("✅ 模型載入成功")
        except Exception as e:
            print(f"❌ 模型載入失敗 (可能是特徵維度不匹配): {e}")
            return
    else:
        print("❌ 找不到 best_model.pth")
        return

    model.eval()
    
    # 4. 推論
    # 為了回測，我們不打亂順序 (shuffle=False)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    all_preds = []
    all_probs = []
    
    print("🔮 執行推論...")
    with torch.no_grad():
        for batch in loader:
            x_1h = batch['1h'].to(device)
            x_4h = batch['4h'].to(device)
            x_1d = batch['1d'].to(device)
            
            logits = model(x_1h, x_4h, x_1d)
            probs = F.softmax(logits, dim=1)
            
            max_probs, preds = torch.max(probs, dim=1)
            
            # 信心門檻過濾
            if threshold > 0:
                mask = max_probs < threshold
                preds[mask] = 0
                
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(max_probs.cpu().numpy())
            
    # 5. 構建回測日誌
    # 因為 Dataset 會吃掉前 seq_len 筆資料，所以預測結果長度 = len(df) - seq_len
    valid_len = len(all_preds)
    # 預測的是 window_end 的時間點
    # log_index 是從第 60 根 K 棒開始的
    log_index = df_features.index[CONFIG['seq_len']:]
    
    if len(log_index) != valid_len:
        # 再次確認長度對齊，避免 index error
        min_len = min(len(log_index), valid_len)
        log_index = log_index[:min_len]
        all_preds = all_preds[:min_len]
        all_probs = all_probs[:min_len]
    
    log_df = pd.DataFrame(index=log_index)
    log_df['Close'] = raw_close.loc[log_index]
    log_df['Signal'] = all_preds
    log_df['Confidence'] = all_probs
    
    # 映射訊號: 0->0 (Hold), 1->1 (Long), 2->-1 (Short)
    log_df['Position'] = log_df['Signal'].map({0: 0, 1: 1, 2: -1})
    
    # 計算回報
    # Market Return: 今天的收盤 / 昨天的收盤
    log_df['Market_Ret'] = np.log(log_df['Close'] / log_df['Close'].shift(1)).fillna(0)
    
    # Strategy Return: 昨天的部位 * 今天的漲跌 (Open-to-Close or Close-to-Close assumption)
    # 這裡使用 Close-to-Close 邏輯，假設訊號產生後在收盤前執行，或者下一根開盤立刻執行
    # 為了保守，我們使用 shift(1)，代表「看到訊號後，下一根K棒才持有部位」
    log_df['Strategy_Ret'] = log_df['Position'].shift(1) * log_df['Market_Ret']
    
    # 計算手續費 (當 Position 改變時扣除)
    log_df['Pos_Change'] = log_df['Position'].diff().abs().fillna(0)
    log_df['Fees'] = log_df['Pos_Change'] * fee_rate
    
    log_df['Net_Ret'] = log_df['Strategy_Ret'] - log_df['Fees']
    
    # 累計淨值
    log_df['Equity'] = (1 + log_df['Net_Ret']).cumprod()
    log_df['Market_Equity'] = (1 + log_df['Market_Ret']).cumprod()
    
    # 6. 輸出報告
    total_ret = log_df['Equity'].iloc[-1] - 1
    mkt_ret = log_df['Market_Equity'].iloc[-1] - 1
    
    print("\n" + "="*30)
    print(f"📊 詳細回測報告: {asset_name}")
    print(f"   總回報: {total_ret*100:.2f}% (基準: {mkt_ret*100:.2f}%)")
    print(f"   總交易次數: {log_df['Pos_Change'].sum()/2:.0f}")
    print(f"   平均信心度: {np.mean(all_probs):.4f}")
    
    # 計算簡單夏普率 (假設無風險利率=0，年化)
    # 1H 資料，一年約 24*365 = 8760 根
    std_dev = log_df['Net_Ret'].std()
    if std_dev > 0:
        sharpe = (log_df['Net_Ret'].mean() / std_dev) * np.sqrt(8760)
        print(f"   夏普率 (Sharpe): {sharpe:.2f}")
    
    print("="*30)

    # 儲存與繪圖
    csv_filename = f'backtest_log_{asset_name}.csv'
    log_df.to_csv(csv_filename)
    print(f"💾 交易日誌已儲存: {csv_filename}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(log_df.index, log_df['Market_Equity'], label='Market (Buy&Hold)', alpha=0.5, color='gray')
    plt.plot(log_df.index, log_df['Equity'], label='Alpha Hunter', linewidth=1.5, color='blue')
    plt.title(f'Equity Curve: {asset_name} (Thresh={threshold})')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_img = f'backtest_{asset_name}.png'
    plt.savefig(save_img)
    print(f"📈 圖片已儲存: {save_img}")

if __name__ == "__main__":
    print("🔬 正在執行高門檻壓力測試...")
    
    # [修正] 將信心門檻從 0.0 提升到 0.55 或 0.60
    # 意義：只有當模型預測某個方向的機率超過 55% 時才交易，否則空手 (Hold)
    
    # 測試 BTC
    run_vectorized_backtest('BTCUSDT', fee_rate=0.001, threshold=0.6) 
    
    # 測試 ETH
    run_vectorized_backtest('ETHUSDT', fee_rate=0.001, threshold=0.6)