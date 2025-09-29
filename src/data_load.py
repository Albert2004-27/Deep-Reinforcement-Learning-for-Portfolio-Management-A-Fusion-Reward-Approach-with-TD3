# -----------------------------------------------------------
# 下載 AAPL、MSFT 歷史日線 (2018-01-01 ~ 2024-01-01)
# -----------------------------------------------------------
import yfinance as yf
import pandas as pd

# 1. 基本參數
TICKERS = ["AAPL", "AMZN", "BA", "BAC", "JNJ", "MSFT", "PFE", "WMT", "XOM"]
START_DATE  = "2010-01-01"
END_DATE    = "2024-12-31"

# 2. 下載 (auto_adjust=True 會自動調整拆股＆股利)
raw = yf.download(
    tickers     = TICKERS,
    start       = START_DATE,
    end         = END_DATE,
    interval    = "1d",
    group_by    = "ticker",   # 方便一次取多檔
    auto_adjust = True,
    progress    = False       # 關閉進度列
)

# 3. 取得 Close 收盤價，整理成單層欄位
close_px = pd.concat({t: raw[t]["Close"] for t in TICKERS}, axis=1)
close_px.columns = TICKERS        # 兩欄：AAPL, MSFT

# 4. 儲存為 CSV 供後續模型使用
close_px.to_csv("nine_stock_prices.csv")

print("下載完成！前 5 列資料：")
print(close_px.head())
