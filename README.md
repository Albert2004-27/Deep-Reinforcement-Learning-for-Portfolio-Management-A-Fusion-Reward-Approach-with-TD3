# RL x 資產配置專案紀錄

本專案系統性整理了利用強化學習（TD3）於多資產配置（股票組合）的完整實作、回測與分析流程，並對比多種經典量化投資基準策略。

---

## 目標與特色

- 利用 RL (TD3) 日頻調整投資組合權重（含現金部位）
- 支援多股票資料集（如 `nine_stock_prices.csv`, `eight_stock_prices.csv`）
- 提供自動化 benchmark 績效指標評比與專業級視覺化
- 完整紀錄訓練→策略轉換→策略評估→回測流程

---

## 專案結構

```
RL-Asset-Allocation/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── nine_stock_prices.csv
│   └── eight_stock_prices.csv
├── results/
│   └── （自動產生圖表、報表）
├── src/
│   ├── TD3_best_SandA.py
│   ├── TD3_best_SandA2.py
│   ├── td3_strategy_converter.py
│   ├── benchmark_suite.py
│   ├── main_analysis.py
│   ├── data_load.py
│   └── main.py
├── notebooks/
│   └── （可放你的實驗紀錄或心得）
```

---

## 使用流程

1. **資料準備**
    - 執行 `src/data_load.py` 下載或整理資料至 `data/`

2. **RL (TD3) 強化學習訓練**
    - 執行 `src/TD3_best_SandA.py` 或 `src/TD3_best_SandA2.py`
    - 產生權重檔於 `results/`

3. **策略收益轉換**
    - 用 `src/td3_strategy_converter.py` 將權重轉成日收益序列

4. **績效與 benchmark 分析**
    - 執行 `src/main_analysis.py` 進行完整回測、產生圖表與指標

5. **成果展示**
    - 主要圖表、報表皆儲存於 `results/`，可直接用於論文或報告

---

## 主要檔案說明

- `TD3_best_SandA.py` / `TD3_best_SandA2.py`：TD3 RL 策略訓練流程（含 Early Stopping 版本）
- `td3_strategy_converter.py`：權重→日報酬序列轉換器（適用不同股票組合）
- `benchmark_suite.py`：增強型多策略績效回測模組（內建各類 benchmark）
- `main_analysis.py`：一鍵比較回測與視覺化主控腳本
- `data_load.py`：股票資料下載/前處理
- `main.py`：輔助測試與數據抽取

---

## 依賴套件

- numpy, pandas, matplotlib, tensorflow, yfinance, scipy, openpyxl, etc.

安裝指令：

```bash
pip install -r requirements.txt
```

---

## 實驗成果與心得

- 可以於 `notebooks/` 資料夾新建 Jupyter Notebook，記錄每次嘗試的策略、超參數、遇到的問題與心得。
- 也可將部分主要圖表（如權重堆疊圖、累積報酬對比圖）放入此區展示。

---

> _本專案為個人學習與研究用途，歡迎討論與交流。_
