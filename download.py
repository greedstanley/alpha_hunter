import pandas as pd
from binance.client import Client
import os
from datetime import datetime
import subprocess

# 初始化 Client (下載公開數據無需 API Key)
client = Client()

# 設定目標幣種與時間框架
target_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
interval = Client.KLINE_INTERVAL_1HOUR
start_date = "1 Jan, 2017"  # 設定一個夠早的時間，API 會自動對齊上市日

def download_1h_data(symbol):
    """下載指定幣種的1小時K線，並觸發後續處理"""
    print(f"⏳ [{symbol}] 正在下載 1H 全歷史數據...")
    
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
        
        # 轉換數值型別
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

        # [重構] 建立標準化的儲存路徑和檔名
        save_dir = os.path.join('data', 'raw')
        os.makedirs(save_dir, exist_ok=True)
        
        # 使用固定的檔名，方便後續腳本讀取
        filename = f"{symbol}_1H.csv"
        filepath = os.path.join(save_dir, filename)
        
        df.to_csv(filepath, index=False)
        
        print(f"✅ [{symbol}] 原始數據下載完成！")
        print(f"   - 筆數: {len(df)} 筆")
        print(f"   - 已儲存至: {filepath}")
        
        # [新增] 自動呼叫 process_data.py 進行數據處理
        print(f"   - 🚀 正在觸發數據處理腳本...")
        try:
            # 使用 subprocess 執行外部 Python 腳本
            result = subprocess.run(
                ['python', 'process_data.py', '--filepath', filepath],
                check=True,  # 如果 process_data.py 執行失敗，會拋出例外
                capture_output=True, # 捕捉子程序的標準輸出和錯誤
                text=True # 以文字模式解碼輸出
            )
            # 印出子腳本的輸出，方便追蹤
            print(result.stdout)
            if result.stderr:
                print("--- 處理腳本錯誤輸出 ---")
                print(result.stderr)
        except FileNotFoundError:
            print("   ❌ 錯誤: 'python' 命令找不到。請確保 Python 已安裝並在系統路徑中。")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ 錯誤: 數據處理腳本 process_data.py 執行失敗。返回碼: {e.returncode}")
            print("--- 處理腳本輸出 ---")
            print(e.stdout)
            print("--- 處理腳本錯誤輸出 ---")
            print(e.stderr)
        
        print("-" * 30 + "\n")
        
    except Exception as e:
        print(f"❌ [{symbol}] 下載過程發生預期外錯誤: {e}\n")

# --- 主執行區 ---
if __name__ == "__main__":
    print("=== 開始批量下載與處理任務 ===\n")
    for symbol in target_symbols:
        download_1h_data(symbol)
    print("=== 所有任務結束 ===")
