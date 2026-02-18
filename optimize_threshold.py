import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

from features.alignment import synthesize_mtf_data
from features.labeling import apply_triple_barrier
from features.preprocess import prepare_features
from models.tcn_core import ParallelTCNAlphaHunter
from data.dataset import CryptoTimeSeriesDataset
from train import load_and_clean_data, CONFIG

def backtest_with_threshold(probs, raw_returns, threshold, fee_rate=0.001):
    """
    快速向量化回測 (純數值計算，不涉及 IO)
    probs: 模型輸出的機率矩陣 (N, 3)
    raw_returns: 市場回報序列 (N,)
    """
    # 取得最大機率與預測類別
    max_probs, preds = torch.max(probs, dim=1)
    
    # 應用濾網
    mask = max_probs < threshold
    preds[mask] = 0
    
    # 轉為 numpy
    signals = preds.cpu().numpy()
    
    # 映射部位: 0->0, 1->1, 2->-1
    position = np.zeros_like(signals)
    position[signals == 1] = 1
    position[signals == 2] = -1
    
    # 計算策略回報 (Shift 1)
    # position[:-1] * raw_returns[1:]
    strategy_ret = position[:-1] * raw_returns[1:]
    
    # 計算換手率與手續費
    pos_diff = np.abs(np.diff(position))
    fees = pos_diff * fee_rate
    
    net_ret = strategy_ret - fees
    
    # 計算績效指標
    total_ret = np.prod(1 + net_ret) - 1
    
    # 夏普率 (年化)
    if np.std(net_ret) == 0:
        sharpe = 0
    else:
        sharpe = (np.mean(net_ret) / np.std(net_ret)) * np.sqrt(24*365)
        
    trade_count = np.sum(pos_diff) / 2
    
    return total_ret, sharpe, trade_count

def scan_thresholds(asset_name='BTCUSDT'):
    print(f"\n🔍 正在掃描最佳門檻: {asset_name} ...")
    
    # 1. 準備數據 (與 backtest.py 相同)
    filepath = os.path.join('data', 'raw', f'{asset_name}_1H.csv')
    if not os.path.exists(filepath):
        print("❌ 找不到數據")
        return

    df = load_and_clean_data(filepath)
    df_aligned = synthesize_mtf_data(df)
    df_labeled = apply_triple_barrier(df_aligned, horizon=CONFIG['horizon'], atr_period=CONFIG['atr_period'])
    df_features = prepare_features(df_labeled, method=CONFIG['norm_method'], window=30).dropna()
    
    # 原始回報 (用於回測)
    raw_close = df_aligned['close'].loc[df_features.index]
    market_returns = np.log(raw_close / raw_close.shift(1)).fillna(0).values
    # 截掉前 seq_len 個 (因為 dataset 會吃掉)
    market_returns = market_returns[CONFIG['seq_len']:]

    # 2. 模型推論 (只做一次)
    dataset = CryptoTimeSeriesDataset(df_features, seq_len=CONFIG['seq_len'])
    input_dims = dataset.get_input_dim()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ParallelTCNAlphaHunter(input_features=input_dims, num_classes=3).to(device)
    
    checkpoint_path = os.path.join('models', 'checkpoints', 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print("❌ 沒訓練好模型")
        return
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    model.eval()
    
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            x_1h = batch['1h'].to(device)
            x_4h = batch['4h'].to(device)
            x_1d = batch['1d'].to(device)
            logits = model(x_1h, x_4h, x_1d)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)
            
    all_probs = torch.cat(all_probs)
    
    # 3. 掃描參數
    thresholds = np.arange(0.35, 0.95, 0.05)
    results = []
    
    print(f"{'Threshold':<10} | {'Return':<10} | {'Sharpe':<10} | {'Trades':<10}")
    print("-" * 50)
    
    best_sharpe = -999
    best_thresh = 0
    
    for th in thresholds:
        ret, sharpe, trades = backtest_with_threshold(all_probs, market_returns, th, fee_rate=0.001)
        print(f"{th:.2f}       | {ret*100:>6.2f}%    | {sharpe:>6.2f}     | {int(trades)}")
        
        results.append({'threshold': th, 'sharpe': sharpe, 'return': ret})
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_thresh = th
            
    print("-" * 50)
    print(f"🏆 最佳門檻建議: {best_thresh:.2f} (Sharpe: {best_sharpe:.2f})")
    return best_thresh

if __name__ == "__main__":
    scan_thresholds('BTCUSDT')
    scan_thresholds('ETHUSDT')