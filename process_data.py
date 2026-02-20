import argparse
import os
import sys

# 確保在任何地方執行此腳本時，都能找到 features 模組
# 將專案根目錄加入到 Python 的模組搜尋路徑中
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from features.pipeline import process_single_asset

def main():
    """
    數據處理腳本的主函數。
    接收一個原始數據檔案路徑，執行完整的處理流程，並儲存結果。
    """
    parser = argparse.ArgumentParser(
        description="單一資產數據處理與儲存腳本。"
                    "讀取一個原始CSV，處理後將訓練集儲存到 'processed/train' 資料夾下。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--filepath', type=str, required=True,
                        help='[必須] 原始數據CSV檔案的路徑 (例如: data/raw/BTCUSDT_1H.csv)。')
                        
    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        print(f"❌ 錯誤: 找不到指定的檔案: {args.filepath}")
        return

    print(f"=== 開始處理檔案: {os.path.basename(args.filepath)} ===")
    
    # 調用核心處理管線，並強制儲存CSV結果
    # 處理後的檔案會被存在
    process_single_asset(args.filepath, save_csv=True)
    
    print(f"=== 處理完畢 ===")

if __name__ == "__main__":
    main()
