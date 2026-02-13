import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class CryptoTimeSeriesDataset(Dataset):

    def __init__(self,
                 features_df: pd.DataFrame,
                 target_col: str = 'label',
                 seq_len: int = 60):
        """
        Args:
            features_df: 已經標準化後的 DataFrame。
            target_col: 標籤欄位名稱。
            seq_len: TCN 的輸入序列長度 (Benchmark 變數: 30/40/60/80)。
        """
        # 分離特徵與標籤
        self.labels = features_df[target_col].values
        self.features = features_df.drop(columns=[target_col]).values
        self.seq_len = seq_len

        # 區分 1H, 4H, 1D 的欄位索引 (假設 preprocess 保持了順序)
        # 這裡需要根據你的 synthesize_mtf_data 輸出的欄位順序做動態調整
        # 為了簡化，我們假設 features 的欄位順序是:
        # [1H_O, 1H_H..., 4H_O, 4H_H..., 1D_O, 1D_H...]
        self.n_features_per_tf = 5  # OHLCV

    def __len__(self):
        # 因為需要回看 seq_len，所以前 seq_len 筆資料無法當作 sample
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        # 取出一段長度為 seq_len 的視窗
        # X shape: (seq_len, total_features)
        window_start = idx
        window_end = idx + self.seq_len

        X = self.features[window_start:window_end]

        # 標籤取視窗最後一個時間點的標籤 (或是你定義的預測目標)
        # 注意：這裡的 label 必須對應 window_end (當下時刻) 或是 window_end 的未來
        # 根據我們的 labeling.py，label 是標記在 entry bar 的，所以取 window_end - 1
        y = self.labels[window_end - 1]

        # 將 Label 轉換為 0, 1, 2 (因為原始是 -1, 0, 1)
        # -1 (Sell) -> 2
        # 0 (Hold) -> 0
        # 1 (Buy) -> 1
        if y == -1:
            y_mapped = 2
        else:
            y_mapped = int(y)

        # 將 X 拆解為三個通道 (1H, 4H, 1D)
        # 假設欄位順序是: 1h(5) + 4h(5) + 1d(5) = 15
        # 轉置為 (Channels, Length) 符合 PyTorch Conv1d 格式
        x_1h = X[:, 0:5].transpose()
        x_4h = X[:, 5:10].transpose()
        x_1d = X[:, 10:15].transpose()

        return {
            '1h': torch.tensor(x_1h, dtype=torch.float32),
            '4h': torch.tensor(x_4h, dtype=torch.float32),
            '1d': torch.tensor(x_1d, dtype=torch.float32),
            'label': torch.tensor(y_mapped, dtype=torch.long)
        }
