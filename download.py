import pandas as pd
from binance.client import Client
import os
from datetime import datetime

# 初始化 Client (下載公開數據無需 API Key)
client = Client()

# 設定目標幣種與時間框架
target_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
interval = Client.KLINE_INTERVAL_1HOUR
start_date = "1 Jan, 2017"  # 設定一個夠早的時間，API 會自動對齊上市日

def download_1h_data(symbol):
    print(f"⏳ [{symbol}] 正在下載 1H 全歷史數據，請稍候...")
    
    try:
        # 抓取 K 線數據
        klines = client.get_historical_klines(symbol, interval, start_date)
        
        if not klines:
            print(f"❌ [{symbol}] 未抓取到數據，請檢查代號是否正確。")
            return

        # 轉換 DataFrame
        df = pd.DataFrame(klines, columns=[
            'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close Time', 'Quote Asset Volume', 'Number of Trades',
            'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
        ])
        
        # 格式化處理
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
        df = df[['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # 轉換數值型別 (API 回傳預設是字串)
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

        # 取得實際數據的起訖日期
        first_date = df['Open Time'].iloc[0].strftime('%Y%m%d')
        last_date = df['Open Time'].iloc[-1].strftime('%Y%m%d')
        
        # 存檔
        filename = f"{symbol}_1H_{first_date}_{last_date}.csv"
        df.to_csv(filename, index=False)
        
        print(f"✅ [{symbol}] 下載完成！")
        print(f"   - 區間: {first_date} 到 {last_date}")
        print(f"   - 筆數: {len(df)} 筆")
        print(f"   - 檔名: {filename}\n")
        
    except Exception as e:
        print(f"❌ [{symbol}] 下載發生錯誤: {e}")

# 執行批量下載
print("=== 開始批量下載任務 ===\n")
for symbol in target_symbols:
    download_1h_data(symbol)
print("=== 所有任務結束 ===")