import pandas as pd
import numpy as np


def calculate_log_return(series: pd.Series) -> pd.Series:
    """計算對數收益率 (Log Return) - 動能派"""
    return np.log(series / series.shift(1)).fillna(0)


def calculate_z_score(series: pd.Series, window: int = 30) -> pd.Series:
    """計算滾動 Z-Score - 型態派"""
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    # 避免除以零
    return ((series - mean) / (std + 1e-8)).fillna(0)


def prepare_features(df: pd.DataFrame,
                     method: str = 'z_score',
                     window: int = 30) -> pd.DataFrame:
    """
    根據 Benchmark 選擇的方法，對 OHLCV 數據進行標準化。
    
    Args:
        df: 經過 alignment 合成後的 DataFrame，包含 1h, 4h, 1d 的欄位。
        method: 'z_score' 或 'log_return'。
        window: Z-Score 的滾動視窗大小。
    
    Returns:
        標準化後的特徵矩陣 (所有欄位數值約在 -3 ~ 3 之間)。
    """
    df_norm = pd.DataFrame(index=df.index)

    # 需要處理的特徵欄位 (Open, High, Low, Close, Volume)
    # 注意：Volume 通常建議用 Log 處理
    feature_cols = [c for c in df.columns if 'label' not in c]

    for col in feature_cols:
        if method == 'log_return':
            if 'volume' in col:
                # Volume 不適合直接算 return，改用 log change 或 relative volume
                df_norm[col] = np.log(df[col] + 1) - np.log(df[col].shift(1) +
                                                            1)
            else:
                df_norm[col] = calculate_log_return(df[col])
        elif method == 'z_score':
            # Volume 可以用 log 後再做 z-score
            if 'volume' in col:
                df_norm[col] = calculate_z_score(np.log(df[col] + 1), window)
            else:
                df_norm[col] = calculate_z_score(df[col], window)

    # 補回 Label (如果有)
    if 'label' in df.columns:
        df_norm['label'] = df['label']

    return df_norm.fillna(0)
