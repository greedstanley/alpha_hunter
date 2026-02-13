import pandas as pd
import numpy as np


def apply_triple_barrier(df: pd.DataFrame,
                         horizon: int = 60,
                         atr_period: int = 14,
                         pt_mul: float = 2.0,
                         sl_mul: float = 2.0) -> pd.DataFrame:
    """
    計算三重障礙標籤 (Triple Barrier Method)。
    
    Args:
        df: 包含 'close', 'high', 'low' 的 DataFrame。
        horizon: 最大持有 K 棒數 (時間障礙)。
        atr_period: ATR 計算週期。
        pt_mul: 止盈 ATR 倍數 (Profit Taking)。
        sl_mul: 止損 ATR 倍數 (Stop Loss)。
    
    Returns:
        新增 'label' 欄位的 DataFrame: 1 (Buy), -1 (Sell), 0 (Hold)。
    """
    data = df.copy()

    # 簡單計算 ATR (此處簡化為 TR 的移動平均)
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['tr'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr'] = data['tr'].rolling(window=atr_period).mean()

    labels = pd.Series(0, index=data.index)  # 預設為 0 (Hold)

    # 轉換為 numpy array 以加速運算
    prices = data['close'].values
    atrs = data['atr'].values
    highs = data['high'].values
    lows = data['low'].values
    n = len(data)

    # 模擬路徑 (Path Dependency)
    for i in range(n - horizon):
        if np.isnan(atrs[i]):
            continue

        entry_price = prices[i]
        upper_barrier = entry_price + (atrs[i] * pt_mul)
        lower_barrier = entry_price - (atrs[i] * sl_mul)

        # 往未來尋找哪一道牆先被碰到
        for j in range(1, horizon + 1):
            future_idx = i + j

            # 邊界檢查
            if future_idx >= n:
                break

            if highs[future_idx] >= upper_barrier:
                labels.iloc[i] = 1  # 碰上軌，標記 Buy
                break
            elif lows[future_idx] <= lower_barrier:
                labels.iloc[i] = -1  # 碰下軌，標記 Sell
                break
            # 如果跑完 horizon 都沒碰到，保持預設 0 (Hold)

    data['label'] = labels
    # 清理中間計算過程
    return data.drop(columns=['prev_close', 'tr1', 'tr2', 'tr3', 'tr'])
