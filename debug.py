import torch
import pandas as pd
import numpy as np
import os
from features.alignment import synthesize_mtf_data
from features.labeling import apply_triple_barrier
from features.preprocess import prepare_features
from train import load_and_clean_data, CONFIG

def debug_data_pipeline(asset_name='BTCUSDT'):
    print(f"🕵️‍♂️ Debugging Data Pipeline for {asset_name}...")
    
    filepath = os.path.join('data', 'raw', f'{asset_name}_1H.csv')
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return

    # 1. Load Data
    df = load_and_clean_data(filepath)
    print(f"   Raw Data Rows: {len(df)}")

    # 2. Alignment
    df_aligned = synthesize_mtf_data(df)
    
    # 3. Labeling
    df_labeled = apply_triple_barrier(df_aligned, horizon=CONFIG['horizon'], atr_period=CONFIG['atr_period'])
    label_counts = df_labeled['label'].value_counts()
    print(f"   Label Distribution (After Labeling): {label_counts.to_dict()}")
    
    # 4. Preprocessing (Normalization + Indicators)
    df_final = prepare_features(df_labeled, method=CONFIG['norm_method'], window=30)
    
    # Check for NaNs
    nan_counts = df_final.isna().sum().sum()
    print(f"   NaNs before drop: {nan_counts}")
    
    df_final = df_final.dropna()
    print(f"   Rows after dropna: {len(df_final)}")
    
    # 5. Check Leakage (Correlation)
    print("   Checking for Data Leakage (Correlation > 0.9)...")
    target = df_final['label']
    features = df_final.drop(columns=['label'])
    
    leaks_found = False
    for col in features.columns:
        # Skip string columns if any
        if not np.issubdtype(features[col].dtype, np.number): continue
        
        corr = features[col].corr(target)
        if abs(corr) > 0.9:
            print(f"   🚨 LEAK ALERT: Feature '{col}' has correlation {corr:.4f} with label!")
            leaks_found = True
            
    if not leaks_found:
        print("   ✅ No obvious linear leakage found.")

    # 6. Check Dataset Split
    split_idx = int(len(df_final) * 0.8)
    train_labels = df_final['label'].iloc[:split_idx]
    val_labels = df_final['label'].iloc[split_idx:]
    
    print(f"   Train Label Dist: {train_labels.value_counts().to_dict()}")
    print(f"   Val Label Dist: {val_labels.value_counts().to_dict()}")

if __name__ == "__main__":
    debug_data_pipeline('BTCUSDT')