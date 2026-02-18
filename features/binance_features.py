import requests
import pandas as pd
import numpy as np
import os
import time

def fetch_funding_rate(symbol, limit=1000):
    """
    獲取幣安歷史資金費率 (Funding Rate)
    Endpoint: GET /fapi/v1/fundingRate
    """
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {'symbol': symbol, 'limit': limit}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        df = pd.DataFrame(data)
        df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df = df[['fundingTime', 'fundingRate']].rename(columns={'fundingTime': 'datetime', 'fundingRate': 'funding_rate'})
        df['funding_rate'] = df['funding_rate'].astype(float)
        df.set_index('datetime', inplace=True)
        return df
    except Exception as e:
        print(f"⚠️ 抓取 Funding Rate 失敗 ({symbol}): {e}")
        return pd.DataFrame()

def fetch_long_short_ratio(symbol, period='1h', limit=500):
    """
    獲取幣安多空比 (Global Long/Short Account Ratio)
    Endpoint: GET /futures/data/globalLongShortAccountRatio
    """
    url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
    params = {'symbol': symbol, 'period': period, 'limit': limit}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['timestamp', 'longShortRatio']].rename(columns={'timestamp': 'datetime', 'longShortRatio': 'ls_ratio'})
        df['ls_ratio'] = df['ls_ratio'].astype(float)
        df.set_index('datetime', inplace=True)
        return df
    except Exception as e:
        print(f"⚠️ 抓取 Long/Short Ratio 失敗 ({symbol}): {e}")
        return pd.DataFrame()

def integrate_binance_features(df, symbol):
    """
    將外部特徵對齊到主 DataFrame (1H)
    """
    # 1. 抓取資料
    df_funding = fetch_funding_rate(symbol)
    df_ls = fetch_long_short_ratio(symbol)
    
    # 2. 合併 (使用 ffill 處理資金費率，因為它通常每 8 小時更新一次)
    if not df_funding.empty:
        df = df.join(df_funding, how='left')
        df['funding_rate'] = df['funding_rate'].ffill().fillna(0)
        
    if not df_ls.empty:
        df = df.join(df_ls, how='left')
        df['ls_ratio'] = df['ls_ratio'].ffill().fillna(1.0) # 預設多空平衡
        
    return df
