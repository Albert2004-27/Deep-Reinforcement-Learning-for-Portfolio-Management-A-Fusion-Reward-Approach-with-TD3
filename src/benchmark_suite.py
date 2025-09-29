"""
Enhanced Quantitative Benchmark Suite (完整增強版)
一個完整的量化策略benchmark比較模組 - 包含專業績效指標

支援的benchmark策略：
1. 等權重（月再平衡）
2. 等權重買入持有  
3. 逆波動（1/σ）
4. 最小方差（長-only）
5. 動能（Winners-Losers, k=2）
6. BLSW（Losers-Winners, k=2）

增強功能：
✅ Sortino Ratio - 下行風險調整報酬
✅ Calmar Ratio - 回撤調整報酬  
✅ VaR/CVaR - 尾端風險指標
✅ Omega Ratio - 機率加權收益
✅ Rolling Metrics - 動態穩健性分析
✅ Statistical Testing - 顯著性檢驗
✅ Professional Visualization - 機構級圖表
✅ Individual Chart Export - 11張個別JPG圖表輸出

使用方法：
```python
import pandas as pd
from enhanced_benchmark_suite import BenchmarkSuite

# 載入你的股票數據
data = pd.read_csv('your_stock_data.csv', index_col=0, parse_dates=True)

# 創建增強版benchmark suite
suite = BenchmarkSuite(data)

# 一鍵生成完整報告（包含你的RL策略）
results = suite.generate_performance_report(
    start_date='2022-01-01',
    end_date='2024-12-31',
    custom_strategies={'TD3_Strategy': your_td3_returns},
    save_dir='./results'
)
```
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings
import os
warnings.filterwarnings('ignore')

class BenchmarkSuite:
    """
    增強版量化策略Benchmark比較套件
    包含專業級績效指標和分析功能
    """
    
    def __init__(self, price_data, transaction_cost=0.0015):
        """
        初始化BenchmarkSuite
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            股票價格數據，index為日期，columns為股票代碼
        transaction_cost : float
            交易成本率，預設0.15%
        """
        self.price_data = price_data.copy()
        self.returns_data = price_data.pct_change().dropna()
        self.transaction_cost = transaction_cost
        
    def equal_weight_rebalance(self, start_date, end_date, rebal_freq='M'):
        """
        等權重策略（定期再平衡，成本直接扣在再平衡當日的日報酬）
        事件順序：
        1) 以今日開盤前（或昨收後）的權重 w 吃到今日市場報酬 → day_r
        2) 權重「自然漂移」到今日收盤前狀態 w_eod
        3) 若為再平衡日：計 turnover，day_r -= turnover * cost，並把明日 w 設成 target
            否則：明日 w = w_eod
        """
        data = self.returns_data.loc[start_date:end_date]
        n = data.shape[1]
        idx = data.index

        # 再平衡日期
        if rebal_freq == 'M':
            rebal_dates = idx.to_series().resample('M').last().dropna().index
            rebal_dates = rebal_dates.intersection(idx)  # 保險
        elif isinstance(rebal_freq, int):
            rebal_dates = idx[::rebal_freq]  # 每 N 個交易日
        else:
            rebal_dates = idx  # 每日

        # 今日開盤前權重（第一天等權）
        w = np.ones(n) / n
        rets = []

        for t in idx:
            r_t = data.loc[t].values.astype(float)

            # 1) 以今日 SOD 權重吃今日報酬
            day_r = float(np.dot(w, r_t))

            # 2) 權重自然漂移至今日收盤（尚未交易）
            w_eod = w * (1.0 + r_t)
            s = w_eod.sum()
            if s > 0:
                w_eod = w_eod / s

            # 3) 收盤再平衡（若當日為再平衡日）：成本扣到「今天」的 day_r
            if t in rebal_dates:
                target = np.ones(n) / n
                turnover = float(np.abs(w_eod - target).sum())
                day_r -= turnover * self.transaction_cost     # ← 成本進入報酬
                w = target                                    # ← 明日 SOD 權重
            else:
                w = w_eod                                     # ← 明日 SOD 權重

            rets.append(day_r)

        return pd.Series(rets, index=idx, name='EQ_Rebal')

    
    def equal_weight_buy_hold(self, start_date, end_date):
        """等權重買入持有（不再平衡）
        - 只用 start_date 當日「有有效價格」的資產做等權建倉
        - 中途不納新股
        - 對 NaN/Inf 報酬做 0 處理以保證健壯
        """
        # 取區間資料（價格 & 報酬）
        px = self.price_data.loc[start_date:end_date]
        rt = self.returns_data.loc[start_date:end_date]

        if len(rt.index) == 0:
            return pd.Series(dtype=float, name='EQ_BuyHold')

        # 1) 初始可交易成分：start_date 當天有有效價格的標的
        first_row = px.iloc[0]
        init_universe = first_row[first_row.notna()].index.tolist()
        if len(init_universe) == 0:
            return pd.Series(dtype=float, name='EQ_BuyHold')

        # 固定成分（不納新股）
        rt = rt[init_universe]

        n = len(init_universe)
        idx = rt.index

        # 2) t0 等權建倉
        w = np.ones(n, dtype=float) / n
        out = []

        for t in idx:
            daily = rt.loc[t].values.astype(float)
            # 數值健壯處理
            daily = np.nan_to_num(daily, nan=0.0, posinf=0.0, neginf=0.0)

            # 以今日 SOD 權重吃今日報酬
            day_r = float(np.dot(w, daily))
            out.append(day_r)

            # 權重自然漂移到 EOD
            w = w * (1.0 + daily)
            s = w.sum()
            if s > 1e-12:
                w = w / s
            else:
                # 極端防呆：全歸零時回到等權
                w = np.ones(n, dtype=float) / n

        return pd.Series(out, index=idx, name='EQ_BuyHold')

    
    def inverse_volatility(self, start_date, end_date, lookback=60, rebal_freq='M'):
        """
        逆波動率（1/σ）長-only 基準，含：
        - 再平衡頻率（預設月頻）
        - 交易成本（扣在再平衡當日的日報酬）
        - IPO/缺值健壯處理（lookback 視窗內沒完整價格者當期不分配）
        - 無前視（權重於收盤調整，明日生效）
        事件順序（每日）：
        1) 以今天開盤前權重 w 吃到今日報酬 → day_r
        2) 權重自然漂移到 w_eod
        3) 若今天是再平衡日：用近端資料算 target；day_r -= turnover*cost；明日 w = target
            否則：明日 w = w_eod
        """
        # 區間資料
        rt = self.returns_data.loc[start_date:end_date]
        px = self.price_data.loc[start_date:end_date]
        idx = rt.index
        cols = rt.columns
        n = len(cols)

        # 再平衡日期
        if rebal_freq == 'M':
            rebal_dates = idx.to_series().resample('M').last().dropna().index.intersection(idx)
        elif isinstance(rebal_freq, int):
            rebal_dates = idx[::rebal_freq]  # 每 N 交易日
        else:
            rebal_dates = idx               # 每日

        # 初始等權
        w = np.ones(n, dtype=float) / n
        out = []

        for i, t in enumerate(idx):
            r_t = rt.loc[t].values.astype(float)
            # 報酬健壯處理
            r_t = np.nan_to_num(r_t, nan=0.0, posinf=0.0, neginf=0.0)

            # 1) 今日以 SOD 權重吃到市場 → 當日報酬
            day_r = float(np.dot(w, r_t))

            # 2) 權重自然漂移至 EOD（尚未交易）
            w_eod = w * (1.0 + r_t)
            s = w_eod.sum()
            if s > 1e-12:
                w_eod = w_eod / s
            else:
                w_eod = np.ones(n, dtype=float) / n  # 防呆

            # 3) 收盤再平衡
            if (t in rebal_dates) and (i >= lookback):
                # 用價格資料確認 lookback 視窗可用性（包含今日）
                px_win = px.iloc[i - lookback + 1 : i + 1]
                available = px_win.notna().all(axis=0)  # 每檔在這窗內是否都有價格

                # 僅對可用資產估算波動
                avail_cols = cols[available.values]
                if len(avail_cols) > 0:
                    rt_win = rt.iloc[i - lookback + 1 : i + 1][avail_cols]
                    vol = rt_win.std()  # pandas 對每檔 std（年化不需要，比例而已）
                    vol = vol.replace(0.0, np.nan)        # 0 波動視為不可用，免得無限加碼
                    inv = 1.0 / vol
                    inv = inv.replace([np.inf, -np.inf], np.nan).fillna(0.0)

                    if inv.sum() > 0:
                        target_series = inv / inv.sum()
                    else:
                        target_series = pd.Series(1.0 / len(avail_cols), index=avail_cols)

                    # 對齊到全體資產
                    target = pd.Series(0.0, index=cols, dtype=float)
                    target.loc[target_series.index] = target_series.values
                    target = target.values
                else:
                    target = np.ones(n, dtype=float) / n

                # turnover 以 EOD 漂移後的權重與 target 的差計算
                turnover = float(np.abs(w_eod - target).sum())
                # 成本扣在「今天」的日報酬
                day_r -= turnover * self.transaction_cost
                # 明日開盤權重
                w = target
            else:
                # 非再平衡日，明日權重 = 今日漂移後權重
                w = w_eod

            out.append(day_r)

        return pd.Series(out, index=idx, name='InvVol')

    
    def minimum_variance(self, start_date, end_date, lookback=60, rebal_freq='M'):
        """
        最小方差（長-only）基準，含：
        - 再平衡頻率：'M'（月末）或整數天數（例如 21）
        - 交易成本：扣在再平衡當日的日報酬
        - 無前視：協方差用 [t-lookback, t) 的報酬
        - IPO/缺值：lookback 期不完整的資產當期不分配
        日內事件順序：
        1) 以今日開盤前權重 w 吃今日報酬 → day_r
        2) 權重自然漂移到收盤 → w_eod
        3) 若為再平衡日：用至 t-1 的窗口求 target；day_r -= turnover*cost；明日 w = target
            否則：明日 w = w_eod
        """
        import numpy as np
        import pandas as pd
        from scipy.optimize import minimize

        # 取區間資料
        rt = self.returns_data.loc[start_date:end_date]
        px = self.price_data.loc[start_date:end_date]
        idx = rt.index
        cols = rt.columns
        n = len(cols)

        if len(idx) == 0 or n == 0:
            return pd.Series(dtype=float, name='MinVar')

        # 再平衡日期
        if isinstance(rebal_freq, str) and rebal_freq.upper() == 'M':
            rebal_dates = idx.to_series().resample('M').last().dropna().index.intersection(idx)
        elif isinstance(rebal_freq, int) and rebal_freq > 0:
            rebal_dates = idx[::rebal_freq]
        else:
            rebal_dates = idx  # 每日

        # 初始化：等權
        w = np.ones(n, dtype=float) / n
        out = []

        ridge = 1e-6  # 協方差正則
        bounds = [(0.0, 1.0)] * n
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},)

        for i, t in enumerate(idx):
            r_t = rt.loc[t].values.astype(float)
            r_t = np.nan_to_num(r_t, nan=0.0, posinf=0.0, neginf=0.0)

            # 1) 以今日 SOD 權重吃到市場 → 當日報酬
            day_r = float(np.dot(w, r_t))

            # 2) 權重自然漂移至收盤（尚未交易）
            w_eod = w * (1.0 + r_t)
            s = w_eod.sum()
            if s > 1e-12:
                w_eod = w_eod / s
            else:
                w_eod = np.ones(n, dtype=float) / n  # 防呆

            # 3) 收盤再平衡（無前視：用 [i-lookback, i)）
            if (t in rebal_dates) and (i >= lookback):
                # 僅用 lookback 期「價格皆有值」的資產
                px_win = px.iloc[i - lookback : i]  # 不含今天
                available = px_win.notna().all(axis=0)
                avail_cols = cols[available.values]

                if len(avail_cols) >= 1:
                    rt_win = rt.iloc[i - lookback : i][avail_cols]
                    # 協方差 + ridge
                    cov = rt_win.cov().values + np.eye(len(avail_cols)) * ridge

                    # 以等權作初值
                    w0 = np.ones(len(avail_cols), dtype=float) / len(avail_cols)

                    def obj(x):
                        return float(x @ cov @ x)

                    # 在可用子集合上解最小方差
                    try:
                        res = minimize(obj, w0, method='SLSQP',
                                    bounds=[(0.0, 1.0)] * len(avail_cols),
                                    constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},),
                                    options={'maxiter': 300, 'ftol': 1e-9})
                        sub_target = res.x if res.success else w0
                    except Exception:
                        sub_target = w0

                    # 映射回全體資產（不可用者 0）
                    target = pd.Series(0.0, index=cols, dtype=float)
                    target.loc[avail_cols] = sub_target
                    target = target.values
                else:
                    # 若沒有可用資產，退回等權
                    target = np.ones(n, dtype=float) / n

                # turnover：以收盤漂移後權重與 target 的差
                turnover = float(np.abs(w_eod - target).sum())
                # 成本扣在「今天」的報酬
                day_r -= turnover * self.transaction_cost
                # 明日開盤權重
                w = target
            else:
                # 非再平衡日 → 明日權重 = 今日漂移後
                w = w_eod

            out.append(day_r)

        return pd.Series(out, index=idx, name='MinVar')

    
    def momentum_strategy(self, start_date, end_date, formation_period=60, holding_period=20, k=2):
        """
        動能 (Winners − Losers) 長短倉（市值中性），無前視、含交易成本、對 IPO/缺值健壯。
        事件順序（每日）：
        1) 以今日開盤前權重 w 吃到今日報酬 → day_r
        2) 權重自然漂移至收盤 → w_eod
        3) 若 holding 到期要換籃：用 [t-L, t) 期間的價格計形成期，
            過濾缺值資產，選 winners/losers，建 w_target；
            day_r -= turnover * transaction_cost；明日 w = w_target
            否則：明日 w = w_eod
        """
        import numpy as np
        import pandas as pd

        rt = self.returns_data.loc[start_date:end_date]
        px = self.price_data.loc[start_date:end_date]
        idx = rt.index
        cols = rt.columns
        n = len(cols)

        if len(idx) == 0 or n == 0:
            return pd.Series(dtype=float, name='Momentum (W-L)')

        # 初始：空倉（第一次換籃時會產生建倉成本）
        w = np.zeros(n, dtype=float)
        out = []
        hold = 0  # 持有期倒數

        for i, t in enumerate(idx):
            r_t = rt.loc[t].values.astype(float)
            r_t = np.nan_to_num(r_t, nan=0.0, posinf=0.0, neginf=0.0)

            # 1) 今日以 SOD 權重吃市場 → 當日報酬
            day_r = float(np.dot(w, r_t))

            # 2) 權重自然漂移到收盤（尚未交易）
            w_eod = w * (1.0 + r_t)
            s = w_eod.sum()
            # 對長短倉，w 的正負相抵，總和可能接近 0；此處用 L1 正規化做防呆
            if abs(s) < 1e-12:
                # 用 L1 總量避免全 0
                gross = np.sum(np.abs(w_eod))
                if gross > 1e-12:
                    w_eod = w_eod / gross  # 保持方向，總絕對權重歸一
                else:
                    w_eod = np.zeros(n, dtype=float)

            # 3) 是否該換籃（holding 期結束，且形成期足夠）
            if (hold == 0) and (i >= formation_period):
                # 形成期價格窗：不含今天，內部 ffill，過濾缺值
                px_win = px.iloc[i - formation_period : i].ffill()
                available = px_win.notna().all(axis=0)
                avail_cols = cols[available.values]

                if len(avail_cols) >= 2:
                    # 計形成期累積報酬（以價格比值，比用報酬累乘更穩）
                    p0 = px_win[avail_cols].iloc[0].values
                    p1 = px_win[avail_cols].iloc[-1].values
                    # 防零價
                    p0 = np.maximum(p0, 1e-12)
                    form = p1 / p0 - 1.0

                    # 動態 k（至少 1，至多 avail//2）
                    kk = max(1, min(int(k), len(avail_cols) // 2))
                    order = np.argsort(form)           # 由小到大
                    losers_idx_sub  = order[:kk]       # 形成期報酬最差
                    winners_idx_sub = order[-kk:]      # 形成期報酬最好

                    # 映回全體資產 index
                    target = pd.Series(0.0, index=cols, dtype=float)
                    target.loc[avail_cols[ winners_idx_sub ]] =  1.0 / kk
                    target.loc[avail_cols[ losers_idx_sub  ]] = -1.0 / kk
                    target = target.values
                else:
                    # 可用資產不足，維持原權重（不換籃）
                    target = w_eod.copy()

                # turnover：以收盤漂移後權重與目標的差（含多空兩邊）
                turnover = float(np.abs(w_eod - target).sum())
                # 成本扣在「今天」的報酬
                day_r -= turnover * self.transaction_cost

                # 明日開盤權重 = 目標；重置持有期
                w = target
                hold = holding_period
            else:
                # 不換籃：明日權重 = 今日漂移後
                w = w_eod
                hold = max(0, hold - 1)

            out.append(day_r)

        return pd.Series(out, index=idx, name='Momentum (W-L)')

    
    def blsw_strategy(self, start_date, end_date, formation_period=60, holding_period=20, k=2):
        """
        BLSW (Losers − Winners) 長短倉：無前視、含交易成本、對 IPO/缺值健壯。
        日序：
        1) 以今日 SOD 權重 w 吃到今日報酬 → day_r
        2) 權重自然漂移至 EOD → w_eod
        3) 若 holding 到期：用 [t-L, t) 價格求籃子，建 target；
            day_r -= turnover*transaction_cost；明日 w = target
            否則：明日 w = w_eod
        """
        import numpy as np
        import pandas as pd

        rt = self.returns_data.loc[start_date:end_date]
        px = self.price_data.loc[start_date:end_date]
        idx = rt.index
        cols = rt.columns
        n = len(cols)

        if len(idx) == 0 or n == 0:
            return pd.Series(dtype=float, name='BLSW (L-W)')

        # 初始：空倉（第一次換籃會計入建倉成本）
        w = np.zeros(n, dtype=float)
        out = []
        hold = 0

        for i, t in enumerate(idx):
            # 今日資產日報酬（健壯處理）
            r_t = rt.loc[t].values.astype(float)
            r_t = np.nan_to_num(r_t, nan=0.0, posinf=0.0, neginf=0.0)

            # 1) 以今日 SOD 權重吃市場 → 當日報酬
            day_r = float(np.dot(w, r_t))

            # 2) 權重自然漂移至 EOD（尚未交易）
            w_eod = w * (1.0 + r_t)
            s = w_eod.sum()
            if abs(s) < 1e-12:
                # 對長短倉用 L1 正規化，維持毛曝險尺度
                gross = float(np.sum(np.abs(w_eod)))
                if gross > 1e-12:
                    w_eod = w_eod / gross
                else:
                    w_eod = np.zeros(n, dtype=float)

            # 3) 是否到期換籃（且形成期足夠）
            if (hold == 0) and (i >= formation_period):
                # 形成期價格窗：[i-L, i)；先 ffill，再過濾不完整的資產
                px_win = px.iloc[i - formation_period : i].ffill()
                available = px_win.notna().all(axis=0)
                avail_cols = cols[available.values]

                if len(avail_cols) >= 2:
                    p0 = px_win[avail_cols].iloc[0].values
                    p1 = px_win[avail_cols].iloc[-1].values
                    p0 = np.maximum(p0, 1e-12)            # 防零價
                    form = p1 / p0 - 1.0                  # 形成期累積報酬

                    kk = max(1, min(int(k), len(avail_cols) // 2))
                    order = np.argsort(form)              # 小→大
                    losers_sub  = order[:kk]              # 最差 k 檔
                    winners_sub = order[-kk:]             # 最好 k 檔

                    target = pd.Series(0.0, index=cols, dtype=float)
                    # BLSW：做多 losers、做空 winners（各腿等權）
                    target.loc[avail_cols[losers_sub]]  =  1.0 / kk
                    target.loc[avail_cols[winners_sub]] = -1.0 / kk
                    target = target.values
                else:
                    # 可用資產不足：不換籃
                    target = w_eod.copy()

                # 轉倉率以 EOD 漂移後權重與目標的差計
                turnover = float(np.abs(w_eod - target).sum())
                # 成本扣在「今天」的報酬
                day_r -= turnover * self.transaction_cost

                # 明日 SOD 權重 = 目標；重置持有期
                w = target
                hold = holding_period
            else:
                # 不換籃：明日權重 = 今日漂移後
                w = w_eod
                hold = max(0, hold - 1)

            out.append(day_r)

        return pd.Series(out, index=idx, name='BLSW (L-W)')

    
    # ==================== 增強績效指標 ====================
    
    def calculate_sortino_ratio(self, returns, cash_rate=0.02):
        """計算Sortino比率 - 只考慮下行風險"""
        returns = pd.Series(returns).dropna()
        if len(returns) == 0:
            return np.nan
            
        rf_daily = cash_rate / 252.0
        excess_returns = returns - rf_daily
        
        # 只計算負報酬的標準差
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf if excess_returns.mean() > 0 else np.nan
        
        downside_std = downside_returns.std() * np.sqrt(252)
        annual_excess = excess_returns.mean() * 252
        
        return annual_excess / downside_std if downside_std > 0 else np.nan
    
    def calculate_calmar_ratio(self, returns):
        """計算Calmar比率 = 年化報酬 / 最大回撤"""
        returns = pd.Series(returns).dropna()
        if len(returns) == 0:
            return np.nan
            
        # 計算年化報酬
        cumulative = (1 + returns).prod()
        annual_return = cumulative ** (252 / len(returns)) - 1
        
        # 計算最大回撤
        nav = (1 + returns).cumprod()
        running_max = nav.cummax()
        drawdown = (nav / running_max) - 1
        max_drawdown = abs(drawdown.min())
        
        return annual_return / max_drawdown if max_drawdown > 0 else np.inf
    
    def calculate_var_cvar(self, returns, confidence=0.05):
        """計算VaR和CVaR (條件風險值)"""
        returns = pd.Series(returns).dropna()
        if len(returns) == 0:
            return np.nan, np.nan
            
        # VaR: 分位數
        var = np.percentile(returns, confidence * 100)
        
        # CVaR: 超過VaR的平均損失
        tail_losses = returns[returns <= var]
        cvar = tail_losses.mean() if len(tail_losses) > 0 else var
        
        return var, cvar
    
    def calculate_omega_ratio(self, returns, threshold=0.0):
        """計算Omega比率 = 收益機率加權 / 損失機率加權"""
        returns = pd.Series(returns).dropna()
        if len(returns) == 0:
            return np.nan
            
        excess = returns - threshold
        gains = excess[excess > 0].sum()
        losses = abs(excess[excess < 0].sum())
        
        return gains / losses if losses > 0 else np.inf
    
    def calculate_max_consecutive_losses(self, returns):
        """計算最大連續虧損天數和最大連續虧損幅度"""
        returns = pd.Series(returns).dropna()
        if len(returns) == 0:
            return {'max_loss_streak': 0, 'max_loss_magnitude': 0.0}
            
        # 識別虧損期
        is_loss = returns < 0
        
        # 計算連續虧損streak
        streaks = []
        current_streak = 0
        
        for loss in is_loss:
            if loss:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        # 最後一個streak
        if current_streak > 0:
            streaks.append(current_streak)
        
        max_streak = max(streaks) if streaks else 0
        
        # 計算最大連續虧損幅度
        max_loss_magnitude = 0.0
        current_loss = 0.0
        
        for ret in returns:
            if ret < 0:
                current_loss += ret
                max_loss_magnitude = min(max_loss_magnitude, current_loss)
            else:
                current_loss = 0.0
        
        return {
            'max_loss_streak': max_streak,
            'max_loss_magnitude': abs(max_loss_magnitude)
        }

    def calculate_rolling_metrics(self, returns, window=252, cash_rate=0.02):
        """
        計算滾動績效指標
        
        Parameters:
        -----------
        returns : pd.Series
            策略日報酬率 (帶日期index)
        window : int
            滾動視窗大小 (默認252個交易日)
        cash_rate : float
            無風險利率
            
        Returns:
        --------
        pd.DataFrame : 包含滾動Sharpe、波動率、回撤等指標
        """
        returns = pd.Series(returns).dropna()
        if len(returns) < window:
            return pd.DataFrame()
            
        results = pd.DataFrame(index=returns.index[window-1:])
        
        # 滾動Sharpe
        rf_daily = cash_rate / 252
        excess = returns - rf_daily
        results['Rolling_Sharpe'] = (
            excess.rolling(window).mean() / excess.rolling(window).std() * np.sqrt(252)
        )
        
        # 滾動波動率 (年化)
        results['Rolling_Volatility'] = returns.rolling(window).std() * np.sqrt(252)
        
        # 滾動最大回撤
        def rolling_max_dd(x):
            nav = (1 + x).cumprod()
            running_max = nav.cummax()
            dd = (nav / running_max - 1).min()
            return abs(dd)
            
        results['Rolling_MaxDD'] = returns.rolling(window).apply(rolling_max_dd, raw=False)
        
        # 滾動Sortino
        def rolling_sortino(x):
            excess_ret = x - rf_daily
            downside = excess_ret[excess_ret < 0]
            if len(downside) == 0:
                return np.inf if excess_ret.mean() > 0 else np.nan
            downside_std = downside.std() * np.sqrt(252)
            annual_excess = excess_ret.mean() * 252
            return annual_excess / downside_std if downside_std > 0 else np.nan
            
        results['Rolling_Sortino'] = returns.rolling(window).apply(rolling_sortino, raw=False)
        
        return results.round(4)

    def calculate_performance_metrics(self, returns, cash_rate=0.02, final_cash_weight=0.0):
        """計算增強版績效指標（包含Sortino, Calmar, VaR/CVaR等）"""
        r = pd.Series(returns).dropna()
        
        if r.empty:
            return {metric: np.nan for metric in [
                'Total Return (%)', 'Max Drawdown (%)', 'Sharpe Ratio', 'Sortino Ratio',
                'Calmar Ratio', 'Volatility (%)', 'VaR_95 (%)', 'CVaR_95 (%)', 
                'Omega Ratio', 'Max Loss Streak', 'Max Loss Magnitude (%)',
                'Final Cash (%)', 'Win Rate (%)'
            ]}

        # 基礎指標計算
        rf = cash_rate / 252.0
        er = r - rf

        nav = (1 + r).cumprod()
        peak = nav.cummax()
        drawdown = (nav / peak) - 1.0
        mdd = float(abs(drawdown.min()))

        vol = float(er.std() * np.sqrt(252))
        sharpe = float((er.mean() / (er.std() + 1e-12)) * np.sqrt(252))
        win_rate = float((r > 0).mean())
        total_ret = float(nav.iloc[-1] - 1.0)
        
        # 增強指標計算
        sortino = self.calculate_sortino_ratio(r, cash_rate)
        calmar = self.calculate_calmar_ratio(r)
        var_95, cvar_95 = self.calculate_var_cvar(r, 0.05)
        omega = self.calculate_omega_ratio(r, rf)
        loss_stats = self.calculate_max_consecutive_losses(r)

        return {
            'Total Return (%)': total_ret * 100,
            'Max Drawdown (%)': mdd * 100,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Calmar Ratio': calmar,
            'Volatility (%)': vol * 100,
            'VaR_95 (%)': var_95 * 100,
            'CVaR_95 (%)': cvar_95 * 100,
            'Omega Ratio': omega,
            'Max Loss Streak': loss_stats['max_loss_streak'],
            'Max Loss Magnitude (%)': loss_stats['max_loss_magnitude'] * 100,
            'Final Cash (%)': final_cash_weight * 100,
            'Win Rate (%)': win_rate * 100
        }

    def test_strategy_significance(self, strategy_returns, benchmark_returns, test_type='ttest'):
        """
        測試策略是否統計顯著優於基準
        
        Parameters:
        -----------
        strategy_returns : pd.Series
            策略報酬率
        benchmark_returns : pd.Series  
            基準報酬率
        test_type : str
            檢驗類型 ('ttest', 'wilcoxon')
            
        Returns:
        --------
        dict : 包含檢驗統計量和p值
        """
        try:
            from scipy import stats
            
            s_ret = pd.Series(strategy_returns).dropna()
            b_ret = pd.Series(benchmark_returns).dropna()
            
            if len(s_ret) == 0 or len(b_ret) == 0:
                return {'test_stat': np.nan, 'p_value': np.nan, 'is_significant': False}
            
            # 對齊時間序列
            common_idx = s_ret.index.intersection(b_ret.index)
            if len(common_idx) < 30:  # 至少需要30個觀測值
                return {'test_stat': np.nan, 'p_value': np.nan, 'is_significant': False}
                
            s_aligned = s_ret.loc[common_idx]
            b_aligned = b_ret.loc[common_idx]
            
            if test_type == 'ttest':
                # 雙樣本t檢驗
                t_stat, p_value = stats.ttest_ind(s_aligned, b_aligned)
                test_name = 't-test'
            elif test_type == 'wilcoxon':
                # Wilcoxon符號秩檢驗 (適用於非常態分布)
                try:
                    t_stat, p_value = stats.wilcoxon(s_aligned, b_aligned)
                    test_name = 'Wilcoxon'
                except ValueError:
                    # 如果數據相同，改用t檢驗
                    t_stat, p_value = stats.ttest_ind(s_aligned, b_aligned)
                    test_name = 't-test (fallback)'
            
            is_significant = p_value < 0.05
            
            return {
                'test_name': test_name,
                'test_stat': t_stat,
                'p_value': p_value,
                'is_significant': is_significant,
                'n_observations': len(common_idx)
            }
            
        except ImportError:
            print("Need scipy for statistical tests: pip install scipy")
            return {'test_stat': np.nan, 'p_value': np.nan, 'is_significant': False}

    def run_benchmarks(self, start_date, end_date, cash_rate=0.02):
        """運行六種基準並回傳增強績效表"""
        print("Running benchmark strategies with enhanced metrics...")

        strategies = {
            'Equal Weight (Monthly Rebal)': self.equal_weight_rebalance(start_date, end_date, rebal_freq='M'),
            'Equal Weight (Buy & Hold)':    self.equal_weight_buy_hold(start_date, end_date),
            'Inverse Volatility':           self.inverse_volatility(start_date, end_date, lookback=60, rebal_freq='M'),
            'Minimum Variance':             self.minimum_variance(start_date, end_date, lookback=60, rebal_freq='M'),
            'Momentum (W-L)':               self.momentum_strategy(start_date, end_date, formation_period=60, holding_period=20, k=2),
            'BLSW (L-W)':                   self.blsw_strategy(start_date, end_date, formation_period=60, holding_period=20, k=2),
        }

        results = {}
        for name, ret_series in strategies.items():
            print(f"  Calculating {name}...")
            if len(ret_series) > 0:
                results[name] = self.calculate_performance_metrics(
                    ret_series, cash_rate=cash_rate, final_cash_weight=0.0
                )
            else:
                print(f"  Warning: {name} has no valid data")

        results_df = pd.DataFrame(results).T
        
        # 重新排序列，突出重要指標
        preferred_order = [
            'Total Return (%)', 'Volatility (%)', 'Sharpe Ratio', 'Sortino Ratio', 
            'Calmar Ratio', 'Max Drawdown (%)', 'VaR_95 (%)', 'CVaR_95 (%)',
            'Omega Ratio', 'Max Loss Streak', 'Max Loss Magnitude (%)', 
            'Win Rate (%)', 'Final Cash (%)'
        ]
        
        # 重新排序（保留所有列）
        available_cols = [col for col in preferred_order if col in results_df.columns]
        other_cols = [col for col in results_df.columns if col not in preferred_order]
        results_df = results_df[available_cols + other_cols].round(3)
        
        print("\n" + "="*80)
        print("ENHANCED BENCHMARK PERFORMANCE SUMMARY")
        print("="*80)
        print(results_df)
        print("="*80)
        
        return results_df

    
    def plot_cumulative_returns(self, start_date, end_date, figsize=(12, 8), save_dir=None, custom_strategies=None, save_individual=True):
        """Plot cumulative returns chart (including custom strategies)"""
        try:
            import matplotlib.pyplot as plt
            import os
            
            # Basic benchmark strategies
            strategies = {
                'Equal Weight (Monthly)': self.equal_weight_rebalance(start_date, end_date),
                'Equal Weight (B&H)': self.equal_weight_buy_hold(start_date, end_date),
                'Inverse Vol': self.inverse_volatility(start_date, end_date),
                'Min Variance': self.minimum_variance(start_date, end_date),
                'Momentum': self.momentum_strategy(start_date, end_date),
                'BLSW': self.blsw_strategy(start_date, end_date)
            }
            
            # Add custom strategies (like TD3)
            if custom_strategies:
                strategies.update(custom_strategies)
            
            plt.figure(figsize=figsize)
            
            # Color palette
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            for i, (name, returns) in enumerate(strategies.items()):
                if len(returns) > 0:
                    cumulative = (1 + returns).cumprod()
                    
                    # Special style for TD3 strategy
                    if 'TD3' in name:
                        plt.plot(cumulative.index, cumulative.values, 
                                label=name, linewidth=3, color='red', linestyle='-', alpha=1.0, zorder=10)
                    else:
                        plt.plot(cumulative.index, cumulative.values, 
                                label=name, linewidth=2, color=colors[i % len(colors)], alpha=0.8)
            
            plt.title('Strategy Cumulative Returns Comparison', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Cumulative Return', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            
            # Save chart to specified folder
            if save_dir:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    
                save_path = os.path.join(save_dir, 'benchmark_cumulative_returns.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"Chart saved to: {save_path}")
                
                # Also save as PDF format
                pdf_path = os.path.join(save_dir, 'benchmark_cumulative_returns.pdf')
                plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"PDF chart saved to: {pdf_path}")
                
                # Save individual cumulative returns chart
                if save_individual:
                    individual_dir = os.path.join(save_dir, 'individual_charts')
                    if not os.path.exists(individual_dir):
                        os.makedirs(individual_dir)
                    
                    # Save as JPG in individual charts folder
                    individual_jpg_path = os.path.join(individual_dir, '11_cumulative_returns.jpg')
                    plt.savefig(individual_jpg_path, dpi=300, bbox_inches='tight', facecolor='white', format='jpeg')
                    print(f"  Individual cumulative returns chart saved: {individual_jpg_path}")
            
            plt.show()
            
        except ImportError:
            print("Need to install matplotlib to plot charts: pip install matplotlib")
    
    def plot_performance_comparison(self, results_df, save_dir=None, save_individual=True):
        """Plot enhanced performance comparison charts"""
        try:
            import matplotlib.pyplot as plt
            import os
            
            # Create subplots - 2x3 for more comprehensive analysis
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Enhanced Strategy Performance Comparison', 
                        fontsize=16, fontweight='bold')
            
            colors = ['red' if 'TD3' in idx else 'blue' for idx in results_df.index]
            sizes = [150 if 'TD3' in idx else 100 for idx in results_df.index]
            
            # Individual charts data for separate saving
            individual_charts = []
            
            # 1. Risk-Adjusted Returns: Sharpe vs Sortino
            ax1 = axes[0, 0]
            x = results_df['Sharpe Ratio']
            y = results_df['Sortino Ratio']
            
            ax1.scatter(x, y, c=colors, s=sizes, alpha=0.7)
            
            # Annotate strategy names
            for i, txt in enumerate(results_df.index):
                ax1.annotate(txt, (x.iloc[i], y.iloc[i]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.8)
            
            ax1.set_xlabel('Sharpe Ratio')
            ax1.set_ylabel('Sortino Ratio')
            ax1.set_title('Risk-Adjusted Returns')
            ax1.grid(True, alpha=0.3)
            
            individual_charts.append(('01_risk_adjusted_returns', ax1, x, y, colors, sizes, 'Sharpe vs Sortino Ratio'))
            
            # 2. Return vs Risk scatter
            ax2 = axes[0, 1]
            x2 = results_df['Volatility (%)']
            y2 = results_df['Total Return (%)']
            ax2.scatter(x2, y2, c=colors, s=sizes, alpha=0.7)
            
            for i, txt in enumerate(results_df.index):
                ax2.annotate(txt, (x2.iloc[i], y2.iloc[i]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.8)
            
            ax2.set_xlabel('Volatility (%)')
            ax2.set_ylabel('Total Return (%)')
            ax2.set_title('Return vs Risk')
            ax2.grid(True, alpha=0.3)
            
            individual_charts.append(('02_return_vs_risk', ax2, x2, y2, colors, sizes, 'Return vs Risk'))
            
            # 3. Calmar Ratio comparison
            ax3 = axes[0, 2]
            calmar_values = results_df['Calmar Ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
            bars = ax3.bar(range(len(results_df)), calmar_values, 
                          color=['red' if 'TD3' in idx else 'orange' for idx in results_df.index])
            ax3.set_xlabel('Strategy')
            ax3.set_ylabel('Calmar Ratio')
            ax3.set_title('Risk-Adjusted Return (Calmar)')
            ax3.set_xticks(range(len(results_df)))
            ax3.set_xticklabels(results_df.index, rotation=45, ha='right')
            
            individual_charts.append(('03_calmar_ratio', ax3, None, calmar_values, None, None, 'Calmar Ratio Comparison'))
            
            # 4. Tail Risk: VaR vs CVaR
            ax4 = axes[1, 0]
            x4 = results_df['VaR_95 (%)']
            y4 = results_df['CVaR_95 (%)']
            ax4.scatter(x4, y4, c=colors, s=sizes, alpha=0.7)
            
            for i, txt in enumerate(results_df.index):
                ax4.annotate(txt, (x4.iloc[i], y4.iloc[i]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.8)
            
            ax4.set_xlabel('VaR 95% (%)')
            ax4.set_ylabel('CVaR 95% (%)')
            ax4.set_title('Tail Risk Analysis')
            ax4.grid(True, alpha=0.3)
            
            individual_charts.append(('04_tail_risk', ax4, x4, y4, colors, sizes, 'VaR vs CVaR Analysis'))
            
            # 5. Maximum consecutive losses
            ax5 = axes[1, 1]
            bars = ax5.bar(range(len(results_df)), results_df['Max Loss Streak'], 
                          color=['red' if 'TD3' in idx else 'purple' for idx in results_df.index])
            ax5.set_xlabel('Strategy')
            ax5.set_ylabel('Max Consecutive Loss Days')
            ax5.set_title('Resilience Analysis')
            ax5.set_xticks(range(len(results_df)))
            ax5.set_xticklabels(results_df.index, rotation=45, ha='right')
            
            individual_charts.append(('05_resilience_analysis', ax5, None, results_df['Max Loss Streak'], None, None, 'Maximum Consecutive Losses'))
            
            # 6. Omega ratio comparison
            ax6 = axes[1, 2]
            omega_values = results_df['Omega Ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
            bars = ax6.bar(range(len(results_df)), omega_values, 
                          color=['red' if 'TD3' in idx else 'green' for idx in results_df.index])
            ax6.set_xlabel('Strategy')
            ax6.set_ylabel('Omega Ratio')
            ax6.set_title('Probability-Weighted Returns')
            ax6.set_xticks(range(len(results_df)))
            ax6.set_xticklabels(results_df.index, rotation=45, ha='right')
            
            individual_charts.append(('06_omega_ratio', ax6, None, omega_values, None, None, 'Omega Ratio Comparison'))
            
            plt.tight_layout()
            
            # Save charts
            if save_dir:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    
                # Save integrated chart
                save_path = os.path.join(save_dir, 'enhanced_performance_comparison.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"Enhanced performance comparison chart saved to: {save_path}")
                
                pdf_path = os.path.join(save_dir, 'enhanced_performance_comparison.pdf')
                plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"PDF enhanced performance comparison chart saved to: {pdf_path}")
                
                # Save individual charts
                if save_individual:
                    individual_dir = os.path.join(save_dir, 'individual_charts')
                    if not os.path.exists(individual_dir):
                        os.makedirs(individual_dir)
                    
                    self._save_individual_performance_charts(individual_charts, results_df, individual_dir)
            
            plt.show()
            
        except ImportError:
            print("Need to install matplotlib to plot charts: pip install matplotlib")
    
    def _save_individual_performance_charts(self, chart_data, results_df, save_dir):
        """Save individual performance charts as separate JPG files"""
        try:
            import matplotlib.pyplot as plt
            import os
            
            colors = ['red' if 'TD3' in idx else 'blue' for idx in results_df.index]
            sizes = [150 if 'TD3' in idx else 100 for idx in results_df.index]
            
            for filename, ax, x_data, y_data, scatter_colors, scatter_sizes, title in chart_data:
                fig, new_ax = plt.subplots(1, 1, figsize=(10, 8))
                
                if x_data is not None and y_data is not None and scatter_colors is not None:
                    # Scatter plot
                    new_ax.scatter(x_data, y_data, c=scatter_colors, s=scatter_sizes, alpha=0.7)
                    
                    # Annotate strategy names
                    for i, txt in enumerate(results_df.index):
                        new_ax.annotate(txt, (x_data.iloc[i], y_data.iloc[i]), 
                                       xytext=(5, 5), textcoords='offset points', 
                                       fontsize=10, alpha=0.8)
                    
                    new_ax.set_xlabel(ax.get_xlabel())
                    new_ax.set_ylabel(ax.get_ylabel())
                    new_ax.grid(True, alpha=0.3)
                    
                elif 'calmar' in filename.lower():
                    # Calmar ratio bar chart
                    calmar_values = results_df['Calmar Ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
                    bars = new_ax.bar(range(len(results_df)), calmar_values, 
                                     color=['red' if 'TD3' in idx else 'orange' for idx in results_df.index])
                    new_ax.set_xlabel('Strategy')
                    new_ax.set_ylabel('Calmar Ratio')
                    new_ax.set_xticks(range(len(results_df)))
                    new_ax.set_xticklabels(results_df.index, rotation=45, ha='right')
                    
                elif 'resilience' in filename.lower():
                    # Max loss streak bar chart
                    bars = new_ax.bar(range(len(results_df)), results_df['Max Loss Streak'], 
                                     color=['red' if 'TD3' in idx else 'purple' for idx in results_df.index])
                    new_ax.set_xlabel('Strategy')
                    new_ax.set_ylabel('Max Consecutive Loss Days')
                    new_ax.set_xticks(range(len(results_df)))
                    new_ax.set_xticklabels(results_df.index, rotation=45, ha='right')
                    
                elif 'omega' in filename.lower():
                    # Omega ratio bar chart
                    omega_values = results_df['Omega Ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
                    bars = new_ax.bar(range(len(results_df)), omega_values, 
                                     color=['red' if 'TD3' in idx else 'green' for idx in results_df.index])
                    new_ax.set_xlabel('Strategy')
                    new_ax.set_ylabel('Omega Ratio')
                    new_ax.set_xticks(range(len(results_df)))
                    new_ax.set_xticklabels(results_df.index, rotation=45, ha='right')
                
                new_ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
                plt.tight_layout()
                
                # Save as JPG
                jpg_path = os.path.join(save_dir, f'{filename}.jpg')
                plt.savefig(jpg_path, dpi=300, bbox_inches='tight', facecolor='white', format='jpeg')
                plt.close()
                
                print(f"  Individual chart saved: {jpg_path}")
                
        except Exception as e:
            print(f"Error saving individual performance charts: {e}")
    
    def plot_rolling_metrics(self, start_date, end_date, save_dir=None, custom_strategies=None, window=252, save_individual=True):
        """Plot rolling performance metrics - 展示策略全天候穩健性"""
        try:
            import matplotlib.pyplot as plt
            import os
            
            # Get strategy returns
            strategies = {
                'Equal Weight (Monthly)': self.equal_weight_rebalance(start_date, end_date),
                'Inverse Vol': self.inverse_volatility(start_date, end_date),
                'Min Variance': self.minimum_variance(start_date, end_date),
                'Momentum': self.momentum_strategy(start_date, end_date),
            }
            
            if custom_strategies:
                strategies.update(custom_strategies)
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Rolling Performance Analysis - Strategy Robustness Over Time', 
                        fontsize=16, fontweight='bold')
            
            colors = ['blue', 'green', 'orange', 'purple', 'red', 'brown', 'pink']
            
            # 1. Rolling Sharpe ratio
            ax1 = axes[0, 0]
            for i, (name, returns) in enumerate(strategies.items()):
                if len(returns) > window:
                    rolling_metrics = self.calculate_rolling_metrics(returns, window)
                    if not rolling_metrics.empty:
                        style = {'linewidth': 3, 'alpha': 0.9} if 'TD3' in name else {'linewidth': 2, 'alpha': 0.7}
                        ax1.plot(rolling_metrics.index, rolling_metrics['Rolling_Sharpe'], 
                                label=name, color=colors[i % len(colors)], **style)
            
            ax1.set_title(f'Rolling Sharpe Ratio ({window} days)')
            ax1.set_ylabel('Sharpe Ratio')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # 2. Rolling volatility
            ax2 = axes[0, 1]
            for i, (name, returns) in enumerate(strategies.items()):
                if len(returns) > window:
                    rolling_metrics = self.calculate_rolling_metrics(returns, window)
                    if not rolling_metrics.empty:
                        style = {'linewidth': 3, 'alpha': 0.9} if 'TD3' in name else {'linewidth': 2, 'alpha': 0.7}
                        ax2.plot(rolling_metrics.index, rolling_metrics['Rolling_Volatility'] * 100, 
                                label=name, color=colors[i % len(colors)], **style)
            
            ax2.set_title(f'Rolling Volatility ({window} days)')
            ax2.set_ylabel('Annualized Volatility (%)')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            
            # 3. Rolling Maximum Drawdown
            ax3 = axes[1, 0]
            for i, (name, returns) in enumerate(strategies.items()):
                if len(returns) > window:
                    rolling_metrics = self.calculate_rolling_metrics(returns, window)
                    if not rolling_metrics.empty:
                        style = {'linewidth': 3, 'alpha': 0.9} if 'TD3' in name else {'linewidth': 2, 'alpha': 0.7}
                        ax3.plot(rolling_metrics.index, rolling_metrics['Rolling_MaxDD'] * 100, 
                                label=name, color=colors[i % len(colors)], **style)
            
            ax3.set_title(f'Rolling Maximum Drawdown ({window} days)')
            ax3.set_ylabel('Max Drawdown (%)')
            ax3.set_xlabel('Date')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
            
            # 4. Rolling Sortino Ratio
            ax4 = axes[1, 1]
            for i, (name, returns) in enumerate(strategies.items()):
                if len(returns) > window:
                    rolling_metrics = self.calculate_rolling_metrics(returns, window)
                    if not rolling_metrics.empty:
                        style = {'linewidth': 3, 'alpha': 0.9} if 'TD3' in name else {'linewidth': 2, 'alpha': 0.7}
                        # 處理無限值
                        sortino_clean = rolling_metrics['Rolling_Sortino'].replace([np.inf, -np.inf], np.nan)
                        ax4.plot(rolling_metrics.index, sortino_clean, 
                                label=name, color=colors[i % len(colors)], **style)
            
            ax4.set_title(f'Rolling Sortino Ratio ({window} days)')
            ax4.set_ylabel('Sortino Ratio')
            ax4.set_xlabel('Date')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            
            # Save charts
            if save_dir:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    
                # Save integrated chart
                save_path = os.path.join(save_dir, 'rolling_performance_metrics.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"Rolling metrics chart saved to: {save_path}")
                
                pdf_path = os.path.join(save_dir, 'rolling_performance_metrics.pdf')
                plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"PDF rolling metrics chart saved to: {pdf_path}")
                
                # Save individual rolling charts
                if save_individual:
                    individual_dir = os.path.join(save_dir, 'individual_charts')
                    if not os.path.exists(individual_dir):
                        os.makedirs(individual_dir)
                    
                    self._save_individual_rolling_charts(strategies, window, individual_dir)
            
            plt.show()
            
        except ImportError:
            print("Need to install matplotlib to plot charts: pip install matplotlib")
    
    def _save_individual_rolling_charts(self, strategies, window, save_dir):
        """Save individual rolling charts as separate JPG files"""
        try:
            import matplotlib.pyplot as plt
            import os
            
            colors = ['blue', 'green', 'orange', 'purple', 'red', 'brown', 'pink']
            chart_configs = [
                ('07_rolling_sharpe_ratio', 'Rolling_Sharpe', 'Rolling Sharpe Ratio', 'Sharpe Ratio'),
                ('08_rolling_volatility', 'Rolling_Volatility', 'Rolling Volatility', 'Annualized Volatility (%)'),
                ('09_rolling_max_drawdown', 'Rolling_MaxDD', 'Rolling Maximum Drawdown', 'Max Drawdown (%)'),
                ('10_rolling_sortino_ratio', 'Rolling_Sortino', 'Rolling Sortino Ratio', 'Sortino Ratio')
            ]
            
            for filename, metric, title, ylabel in chart_configs:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                
                has_data = False
                for i, (name, returns) in enumerate(strategies.items()):
                    if len(returns) > window:
                        rolling_metrics = self.calculate_rolling_metrics(returns, window)
                        if not rolling_metrics.empty and metric in rolling_metrics.columns:
                            style = {'linewidth': 3, 'alpha': 0.9} if 'TD3' in name else {'linewidth': 2, 'alpha': 0.7}
                            
                            if metric == 'Rolling_Volatility':
                                # Convert to percentage
                                data = rolling_metrics[metric] * 100
                            elif metric == 'Rolling_MaxDD':
                                # Convert to percentage
                                data = rolling_metrics[metric] * 100
                            elif metric == 'Rolling_Sortino':
                                # Handle infinite values
                                data = rolling_metrics[metric].replace([np.inf, -np.inf], np.nan)
                            else:
                                data = rolling_metrics[metric]
                            
                            ax.plot(rolling_metrics.index, data, 
                                   label=name, color=colors[i % len(colors)], **style)
                            has_data = True
                
                if has_data:
                    ax.set_title(f'{title} ({window} days)', fontsize=14, fontweight='bold', pad=15)
                    ax.set_ylabel(ylabel)
                    ax.set_xlabel('Date')
                    ax.legend(fontsize=10)
                    ax.grid(True, alpha=0.3)
                    
                    if 'sharpe' in filename.lower() or 'sortino' in filename.lower():
                        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    
                    plt.tight_layout()
                    
                    # Save as JPG
                    jpg_path = os.path.join(save_dir, f'{filename}.jpg')
                    plt.savefig(jpg_path, dpi=300, bbox_inches='tight', facecolor='white', format='jpeg')
                    plt.close()
                    
                    print(f"  Individual rolling chart saved: {jpg_path}")
                else:
                    plt.close()
                    
        except Exception as e:
            print(f"Error saving individual rolling charts: {e}")

    def generate_performance_report(self, start_date, end_date, custom_strategies=None, save_dir=None):
        """
        生成完整的績效分析報告
        
        Parameters:
        -----------
        start_date, end_date : str
            分析期間
        custom_strategies : dict
            自定義策略 {'策略名': pd.Series(報酬率)}
        save_dir : str
            儲存路徑
        """
        print("Generating Comprehensive Performance Report...")
        print("="*60)
        
        # 1. 運行基準策略
        benchmark_results = self.run_benchmarks(start_date, end_date)
        
        # 2. 如果有自定義策略，加入比較
        if custom_strategies:
            print(f"\nAdding {len(custom_strategies)} custom strategies...")
            for name, returns in custom_strategies.items():
                if len(returns) > 0:
                    custom_metrics = self.calculate_performance_metrics(returns)
                    benchmark_results.loc[name] = custom_metrics
                    print(f"  Added {name}")
        
        # 3. 生成所有圖表
        if save_dir:
            print(f"\nGenerating charts and saving to: {save_dir}")
            
            # 累積報酬圖
            self.plot_cumulative_returns(start_date, end_date, save_dir=save_dir, 
                                       custom_strategies=custom_strategies)
            
            # 增強績效比較圖
            self.plot_performance_comparison(benchmark_results, save_dir=save_dir)
            
            # 滾動指標圖
            self.plot_rolling_metrics(start_date, end_date, save_dir=save_dir, 
                                    custom_strategies=custom_strategies)
            
            # 儲存績效表格
            excel_path = os.path.join(save_dir, 'performance_metrics.xlsx') if save_dir else None
            if excel_path:
                benchmark_results.to_excel(excel_path)
                print(f"Performance metrics saved to: {excel_path}")
        
        print("\nFINAL PERFORMANCE RANKING")
        print("="*60)
        
        # 依據 Sharpe 排序
        ranked = benchmark_results.sort_values('Sharpe Ratio', ascending=False)
        print("Ranked by Sharpe Ratio:")
        print(ranked[['Total Return (%)', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown (%)']].round(3))
        
        print("\nSUMMARY INSIGHTS:")
        best_sharpe = ranked.index[0]
        best_return = benchmark_results.sort_values('Total Return (%)', ascending=False).index[0]
        lowest_dd = benchmark_results.sort_values('Max Drawdown (%)', ascending=True).index[0]
        
        print(f"  Best Risk-Adjusted Return: {best_sharpe}")
        print(f"  Highest Total Return: {best_return}")  
        print(f"  Lowest Drawdown: {lowest_dd}")
        print("="*60)
        
        return benchmark_results


# 使用範例 - 展示完整功能
if __name__ == "__main__":
    print("Enhanced Quantitative Benchmark Suite Demo")
    print("="*60)
    
    # Create sample data
    dates = pd.date_range('2022-06-24', '2024-12-31', freq='D')
    n_stocks = 10
    stock_names = [f'Stock_{i+1}' for i in range(n_stocks)]
    
    # Generate more realistic price data
    np.random.seed(42)
    returns = np.random.normal(0.0008, 0.02, (len(dates), n_stocks))
    
    # Add sector effects and momentum
    for i in range(n_stocks):
        # 添加個股特性
        if i < 3:  # Tech stocks - higher vol, higher return
            returns[:, i] += np.random.normal(0.0003, 0.005, len(dates))
        elif i < 6:  # Value stocks - lower vol, steady return  
            returns[:, i] = np.random.normal(0.0005, 0.015, len(dates))
        else:  # Defensive stocks - low vol, low return
            returns[:, i] = np.random.normal(0.0002, 0.012, len(dates))
    
    # Convert to prices
    prices = pd.DataFrame(index=dates, columns=stock_names)
    prices.iloc[0] = 100
    
    for i in range(1, len(dates)):
        prices.iloc[i] = prices.iloc[i-1] * (1 + returns[i])
    
    # Create enhanced benchmark suite
    suite = BenchmarkSuite(prices, transaction_cost=0.0015)
    
    print("Running enhanced benchmark analysis...")
    
    # 模擬一個自定義策略 (比如你的TD3 RL策略)
    np.random.seed(123)
    custom_returns = pd.Series(
        np.random.normal(0.0012, 0.018, len(dates)), 
        index=dates, 
        name='TD3_RL_Strategy'
    )
    
    custom_strategies = {
        'TD3_RL_Strategy': custom_returns
    }
    
    # 生成完整報告
    results = suite.generate_performance_report(
        start_date='2022-06-24',
        end_date='2024-12-31',
        custom_strategies=custom_strategies,
        save_dir='./benchmark_results'  # 會創建資料夾並儲存所有圖表
    )
    
    print("\nStatistical Significance Testing:")
    print("-" * 40)
    
    # 測試自定義策略 vs 基準
    benchmark_strategy = suite.equal_weight_rebalance('2022-06-24', '2024-12-31')
    sig_test = suite.test_strategy_significance(
        custom_returns, 
        benchmark_strategy,
        test_type='ttest'
    )
    
    print(f"TD3 vs Equal Weight:")
    print(f"  Test: {sig_test.get('test_name', 'N/A')}")
    print(f"  p-value: {sig_test['p_value']:.4f}")
    print(f"  Significant at 5%: {'YES' if sig_test['is_significant'] else 'NO'}")
    
    print("\nAnalysis Complete!")
    print("="*60)
    print("Check './benchmark_results' folder for:")
    print("  • benchmark_cumulative_returns.png")
    print("  • enhanced_performance_comparison.png") 
    print("  • rolling_performance_metrics.png")
    print("  • performance_metrics.xlsx")
    print("  • All charts also saved as PDF")
    print("  • individual_charts/ folder with 11 JPG files:")
    print("    - 01_risk_adjusted_returns.jpg")
    print("    - 02_return_vs_risk.jpg")
    print("    - 03_calmar_ratio.jpg")
    print("    - 04_tail_risk.jpg")
    print("    - 05_resilience_analysis.jpg")
    print("    - 06_omega_ratio.jpg")
    print("    - 07_rolling_sharpe_ratio.jpg")
    print("    - 08_rolling_volatility.jpg")
    print("    - 09_rolling_max_drawdown.jpg")
    print("    - 10_rolling_sortino_ratio.jpg")
    print("    - 11_cumulative_returns.jpg")
    
    print(f"\nKey Findings:")
    best_strategy = results.sort_values('Sharpe Ratio', ascending=False).index[0]
    print(f"  Best Overall Strategy: {best_strategy}")
    print(f"     Sharpe: {results.loc[best_strategy, 'Sharpe Ratio']:.3f}")
    print(f"     Sortino: {results.loc[best_strategy, 'Sortino Ratio']:.3f}")
    print(f"     Max DD: {results.loc[best_strategy, 'Max Drawdown (%)']:.2f}%")
