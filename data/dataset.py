import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class CryptoTimeSeriesDataset(Dataset):
    def __init__(self, features_df: pd.DataFrame, target_col: str = 'label', seq_len: int = 60):
        """
        Args:
            features_df: 已經標準化後的 DataFrame (包含 RSI, MACD 等所有特徵)。
            target_col: 標籤欄位名稱。
            seq_len: TCN 的輸入序列長度。
        """
        if target_col in features_df.columns:
            self.labels = features_df[target_col].values
            self.features = features_df.drop(columns=[target_col])
        else:
            self.labels = np.zeros(len(features_df))
            self.features = features_df
            
        self.seq_len = seq_len
        self.feature_values = self.features.values
        
        all_cols = self.features.columns.tolist()
        
        self.idx_1d = [i for i, c in enumerate(all_cols) if '1d_' in c]
        self.idx_4h = [i for i, c in enumerate(all_cols) if '4h_' in c]
        self.idx_1h = [i for i, c in enumerate(all_cols) if i not in self.idx_1d and i not in self.idx_4h]
        
        self.n_feats_1h = len(self.idx_1h)
        self.n_feats_4h = len(self.idx_4h)
        self.n_feats_1d = len(self.idx_1d)
        
        # 計算自身的最大特徵數
        self.calculated_max = max(self.n_feats_1h, self.n_feats_4h, self.n_feats_1d)
        # [關鍵新增] 預設目標維度等於自身最大值，但允許外部覆寫
        self.target_dim = self.calculated_max

    def set_target_dim(self, global_dim):
        """[新增] 允許外部強制設定統一的特徵維度 (用於多幣種對齊)"""
        # 取自身計算與全局要求的最大值，確保不會切掉數據
        self.target_dim = max(self.calculated_max, global_dim)

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        window_start = idx
        window_end = idx + self.seq_len
        
        X = self.feature_values[window_start:window_end]
        y = self.labels[window_end - 1]
        
        # 輔助函數：提取並 Padding 到 target_dim
        def extract_and_pad(indices, required_dim):
            data = X[:, indices].transpose(1, 0) # (Features, Length)
            current_dim = data.shape[0]
            
            if current_dim < required_dim:
                # 補 0 直到達到 required_dim
                pad_size = required_dim - current_dim
                data = np.pad(data, ((0, pad_size), (0, 0)), 'constant')
            
            return torch.tensor(data, dtype=torch.float32)

        # 使用 self.target_dim 而不是 self.calculated_max
        x_1h = extract_and_pad(self.idx_1h, self.target_dim)
        x_4h = extract_and_pad(self.idx_4h, self.target_dim)
        x_1d = extract_and_pad(self.idx_1d, self.target_dim)
        
        if y == -1: y_mapped = 2
        else: y_mapped = int(y)
            
        return {
            '1h': x_1h,
            '4h': x_4h,
            '1d': x_1d,
            'label': torch.tensor(y_mapped, dtype=torch.long)
        }
    
    def get_input_dim(self):
        return self.target_dim


