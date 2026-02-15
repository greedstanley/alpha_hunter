import pandas as pd
import numpy as np

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line # Histogram

def calculate_bollinger_width(series, window=20, num_std=2):
    ma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = ma + (std * num_std)
    lower = ma - (std * num_std)
    return (upper - lower) / ma

def calculate_log_return(series: pd.Series) -> pd.Series:
    return np.log(series / series.shift(1)).fillna(0)

def calculate_z_score(series: pd.Series, window: int = 30) -> pd.Series:
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return ((series - mean) / (std + 1e-8)).fillna(0)

def prepare_features(df: pd.DataFrame, method: str = 'z_score', window: int = 30) -> pd.DataFrame:
    """
    增強版：加入 RSI, MACD, BB Width
    """
    df_norm = pd.DataFrame(index=df.index)
    
    # 1. 處理基礎價格特徵
    feature_cols = [c for c in df.columns if 'label' not in c and 'datetime' not in c]
    
    for col in feature_cols:
        if 'volume' in col:
            series = np.log(df[col] + 1)
        else:
            series = df[col]
        
        # 基礎標準化
        if method == 'log_return' and 'volume' not in col:
            df_norm[col] = calculate_log_return(series)
        else:
            df_norm[col] = calculate_z_score(series, window)
            
    # 2. [新增] 針對各時間框架的 Close 加入技術指標
    # 這些指標能捕捉「動能」、「趨勢」與「波動率」
    target_closes = ['close', '4h_close', '1d_close']
    
    for col in target_closes:
        if col in df.columns:
            # RSI (標準化到 0~1 或 Z-score)
            rsi = calculate_rsi(df[col])
            df_norm[f'{col}_rsi'] = (rsi - 50) / 50 # Centered around 0
            
            # MACD (Histogram)
            macd = calculate_macd(df[col])
            df_norm[f'{col}_macd'] = calculate_z_score(macd, window=window)
            
            # Bollinger Width (波動率特徵)
            bbw = calculate_bollinger_width(df[col])
            df_norm[f'{col}_bbw'] = calculate_z_score(bbw, window=window)

    return df_norm.fillna(0)