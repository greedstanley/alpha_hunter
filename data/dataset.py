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
        # 1. 分離特徵與標籤
        if target_col in features_df.columns:
            self.labels = features_df[target_col].values
            self.features = features_df.drop(columns=[target_col])
        else:
            # 回測模式可能沒有 label，補 0
            self.labels = np.zeros(len(features_df))
            self.features = features_df
            
        self.seq_len = seq_len
        self.feature_values = self.features.values
        
        # 2. 自動偵測特徵分配 (Auto-Discovery)
        # 根據 preprocess.py 的命名規則：
        # - 1D 特徵包含 '1d_'
        # - 4H 特徵包含 '4h_'
        # - 1H 特徵是剩下的 (open, close_rsi 等)
        
        all_cols = self.features.columns.tolist()
        
        # 找出欄位索引
        self.idx_1d = [i for i, c in enumerate(all_cols) if '1d_' in c]
        self.idx_4h = [i for i, c in enumerate(all_cols) if '4h_' in c]
        # 1H 是排除掉 1d 和 4h 之後的所有欄位
        self.idx_1h = [i for i, c in enumerate(all_cols) if i not in self.idx_1d and i not in self.idx_4h]
        
        # 計算特徵數量
        self.n_feats_1h = len(self.idx_1h)
        self.n_feats_4h = len(self.idx_4h)
        self.n_feats_1d = len(self.idx_1d)
        
        # 驗證: 確保所有時間框架都有特徵，且最好維度一致 (雖然 TCNBranch 可以處理不同維度，但通常建議一致)
        # 如果 preprocess.py 運作正常，這些應該都要是 8 (OHLCV + RSI + MACD + BBW)
        # print(f"Features Debug: 1H={self.n_feats_1h}, 4H={self.n_feats_4h}, 1D={self.n_feats_1d}")

    def __len__(self):
        # 因為需要回看 seq_len，所以前 seq_len 筆資料無法當作 sample
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        # 取出一段長度為 seq_len 的視窗
        window_start = idx
        window_end = idx + self.seq_len
        
        # X shape: (seq_len, total_features)
        X = self.feature_values[window_start:window_end]
        
        # 標籤取視窗最後一個時間點的標籤
        y = self.labels[window_end - 1]
        
        # 根據索引提取不同時間框架的數據
        # transpose(1, 0) 將 (Length, Feats) 轉為 (Feats, Length) 符合 PyTorch Conv1d
        x_1h = X[:, self.idx_1h].transpose(1, 0)
        x_4h = X[:, self.idx_4h].transpose(1, 0)
        x_1d = X[:, self.idx_1d].transpose(1, 0)
        
        # 將 Label 轉換為 0, 1, 2
        # -1 (Sell) -> 2, 0 (Hold) -> 0, 1 (Buy) -> 1
        if y == -1:
            y_mapped = 2
        else:
            y_mapped = int(y)
            
        return {
            '1h': torch.tensor(x_1h, dtype=torch.float32),
            '4h': torch.tensor(x_4h, dtype=torch.float32),
            '1d': torch.tensor(x_1d, dtype=torch.float32),
            'label': torch.tensor(y_mapped, dtype=torch.long)
        }
    
    def get_input_dim(self):
        """返回 1H 通道的特徵數量，供模型初始化使用"""
        # 假設所有通道特徵數相近，以 1H 為主
        return self.n_feats_1h