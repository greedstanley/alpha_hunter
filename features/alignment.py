import pandas as pd


def synthesize_mtf_data(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    將原始的 1H OHLCV 數據，即時(零延遲)合成出 4H 與 1D 的當下狀態。
    
    Args:
        df_1h: 必須包含 ['open', 'high', 'low', 'close', 'volume']，且 index 為 DatetimeIndex。
    
    Returns:
        包含 '1d_open', '1d_high', ... '4h_open', '4h_high' ... 等合成特徵的 DataFrame。
    """
    df = df_1h.copy()

    # 確保 index 是 datetime 格式
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # --- 合成 1D (日線) 零延遲特徵 ---
    # 利用 date 分組，使用 cummax/cummin 確保 t 時刻只能看到 00:00 到 t 的極值
    df['date'] = df.index.date
    df['1d_open'] = df.groupby('date')['open'].transform('first')
    df['1d_high'] = df.groupby('date')['high'].cummax()
    df['1d_low'] = df.groupby('date')['low'].cummin()
    df['1d_close'] = df['close']  # 當下 1H 的收盤價即為合成 1D 的當下收盤價
    df['1d_volume'] = df.groupby('date')['volume'].cumsum()

    # --- 合成 4H (四小時線) 零延遲特徵 ---
    # 每 4 小時一個區間 (例如 00:00-03:59, 04:00-07:59)
    # 我們自訂一個 4H session ID 確保不會跨區間
    # floor('4h') 需要 pandas 版本支援，若報錯可用自定義函數替代
    try:
        df['4h_session'] = df.index.floor('4h')
    except ValueError:
        # 相容性處理：如果 '4h' floor 不支援
        df['4h_session'] = (df.index.hour // 4) + (df.index.day * 6
                                                   )  # 簡易 session id

    df['4h_open'] = df.groupby('4h_session')['open'].transform('first')
    df['4h_high'] = df.groupby('4h_session')['high'].cummax()
    df['4h_low'] = df.groupby('4h_session')['low'].cummin()
    df['4h_close'] = df['close']
    df['4h_volume'] = df.groupby('4h_session')['volume'].cumsum()

    # 清理輔助欄位
    df = df.drop(columns=['date', '4h_session'])
    return df
