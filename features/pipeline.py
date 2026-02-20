import os
import pandas as pd
import sys

# 處理 Python 的模組搜尋路徑，確保可以從根目錄匯入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 匯入特徵工程和標籤相關的函式
from features.alignment import synthesize_mtf_data
from features.labeling import apply_triple_barrier
from features.preprocess import prepare_features
from features.binance_features import integrate_binance_features
from train import load_and_clean_data

# 數據處理管線專用的設定
PIPELINE_CONFIG = {
    'seq_len': 60,
    'norm_method': 'z_score',
    'atr_period': 14,
    'horizon': 60,
}

def process_single_asset(filepath, config=PIPELINE_CONFIG, save_csv=False):
    """
    對單一資產的原始數據檔案執行完整的預處理、特徵工程和標籤化。

    Args:
        filepath (str): 原始數據CSV檔案的路徑。
        config (dict): 處理管線的設定。
        save_csv (bool): 是否將處理完成的訓練集和驗證集儲存為CSV。

    Returns:
        tuple: 一個包含 (train_df, val_df) 的元組，如果處理失敗則為 (None, None)。
    """
    if not os.path.exists(filepath):
        print(f"⚠️  警告: 在 process_single_asset 中找不到檔案 {filepath}，跳過。")
        return None, None

    filename = os.path.basename(filepath)
    symbol = filename.split('_')[0]

    print(f"🔄 開始處理資產: {symbol}...")
    df = load_and_clean_data(filepath)
    
    if df is None or df.empty:
        print(f"    - 讀取或清理數據失敗，終止處理 {symbol}。")
        return None, None

    print(f"    - 正在抓取 Binance 外部數據...")
    df = integrate_binance_features(df, symbol)
    
    print(f"    - 正在合成多時間框架特徵...")
    df_aligned = synthesize_mtf_data(df)
    
    print(f"    - 正在應用三元標籤法...")
    df_labeled = apply_triple_barrier(df_aligned, horizon=config['horizon'], atr_period=config['atr_period'])
    
    print(f"    - 正在準備最終特徵 (正規化)...")
    df_final = prepare_features(df_labeled, method=config['norm_method'], window=30)
    df_final = df_final.dropna()
    
    if len(df_final) < config['seq_len'] * 2:
        print(f"    - 數據不足 ({len(df_final)} 行)，無法用於訓練。")
        return None, None

    split_idx = int(len(df_final) * 0.8)
    train_df = df_final.iloc[:split_idx]
    val_df = df_final.iloc[split_idx:]
    
    print(f"    - ✅ {symbol} 處理完成。訓練集: {len(train_df)} | 驗證集: {len(val_df)}")

    if save_csv:
        try:
            # --- 儲存訓練集 ---
            base_data_dir = os.path.dirname(os.path.dirname(filepath)) 
            train_save_dir = os.path.join(base_data_dir, 'processed', 'train')
            os.makedirs(train_save_dir, exist_ok=True)
            
            train_save_path = os.path.join(train_save_dir, f"{symbol}_train_processed.csv")
            print(f"    💾 正在儲存處理後的訓練數據至: {train_save_path}")
            train_df.to_csv(train_save_path)

            # --- [新增] 儲存驗證集 ---
            val_save_dir = os.path.join(base_data_dir, 'processed', 'validation')
            os.makedirs(val_save_dir, exist_ok=True)
            
            val_save_path = os.path.join(val_save_dir, f"{symbol}_validation_processed.csv")
            print(f"    💾 正在儲存處理後的驗證數據至: {val_save_path}")
            val_df.to_csv(val_save_path)

        except Exception as e:
            print(f"    ❌ 儲存CSV時發生錯誤: {e}")

    return train_df, val_df
