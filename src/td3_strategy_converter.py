# td3_strategy_converter_universal.py
# 通用版TD3策略转换器 - 基於原版修改，支援任意股票組合

import pandas as pd
import numpy as np
import os
from datetime import datetime

def convert_td3_weights_to_returns_universal(weights_file_path=None, price_file_path='nine_stock_prices.csv'): #改這裡
    """
    通用版：支援任意股票組合的TD3策略轉換器
    基於原版td3_strategy_converter_corrected.py修改
    
    Parameters:
    -----------
    weights_file_path : str
        權重檔案路徑，如果為None則使用預設路徑
    price_file_path : str
        價格數據檔案路徑
    """
    
    print("TD3策略轉換器 (通用版 - 支援任意股票組合)")
    print("="*60)
    
    # ==========================================
    # 1. 載入TD3權重數據 (通用化)
    # ==========================================
    
    # 如果沒有指定權重檔案，使用預設路徑
    if weights_file_path is None:
        weights_file_path = '/storage/ssd1/Albertchen2004/ISS/td3_fusion_early_stop_20250902_081813/data/test_weights_by_days_early_stop.csv' #改這裡
    
    try:
        weights_data = pd.read_csv(weights_file_path)
        print(f"載入TD3權重數據: {len(weights_data)} 個交易日")
        print(f"權重數據列: {list(weights_data.columns)}")
        
        # 通用化：自動識別股票欄位
        exclude_cols = ['CASH', 'trading_day', 'Unnamed: 0', 'index']  # 常見的非股票欄位
        detected_stocks = [col for col in weights_data.columns if col not in exclude_cols]
        
        print(f"自動識別到的股票: {detected_stocks}")
        print(f"股票數量: {len(detected_stocks)}")
        
        # 檢查是否有CASH欄位
        has_cash = 'CASH' in weights_data.columns
        print(f"包含現金部位: {'是' if has_cash else '否'}")
        
        # 驗證權重數據合理性
        if has_cash:
            weight_cols = detected_stocks + ['CASH']
        else:
            weight_cols = detected_stocks
            
        sample_sums = weights_data[weight_cols].head().sum(axis=1)
        print(f"前5行權重總和: {sample_sums.values}")
        
        if not np.allclose(sample_sums, 1.0, atol=0.02):
            print(f"警告：權重總和偏離1.0較多，可能需要檢查數據格式")
        
    except FileNotFoundError:
        print("找不到權重文件，請檢查路徑是否正確")
        return None
    except Exception as e:
        print(f"載入權重數據失敗: {e}")
        return None
    
    # ==========================================
    # 2. 載入並重現TD3的數據預處理流程 (通用化)
    # ==========================================
    
    try:
        print(f"\n載入價格數據: {price_file_path}")
        # 載入原始價格數據
        df = pd.read_csv(price_file_path, index_col=0, parse_dates=True)
        df = df.ffill().dropna()
        print(f"載入原始價格數據: {df.shape}")
        print(f"價格數據股票: {list(df.columns)}")
        
        # 關鍵：檢查股票匹配情況
        price_stocks = set(df.columns)
        weight_stocks = set(detected_stocks)
        
        matched_stocks = weight_stocks.intersection(price_stocks)
        missing_in_price = weight_stocks - price_stocks
        extra_in_price = price_stocks - weight_stocks
        
        print(f"\n股票匹配分析:")
        print(f"  匹配成功 ({len(matched_stocks)}): {sorted(matched_stocks)}")
        if missing_in_price:
            print(f"  權重中有但價格中缺少 ({len(missing_in_price)}): {sorted(missing_in_price)}")
        if extra_in_price:
            print(f"  價格中有但權重中不需要 ({len(extra_in_price)}): {sorted(extra_in_price)}")
        
        if len(missing_in_price) > 0:
            print(f"錯誤：無法找到 {len(missing_in_price)} 支股票的價格數據")
            print("請確保價格檔案包含權重檔案中的所有股票")
            return None
        
        # 只使用匹配的股票，並確保順序一致
        td3_stocks = sorted(matched_stocks)  # 使用排序確保一致性
        df = df[td3_stocks]
        print(f"最終使用的股票 ({len(td3_stocks)}): {td3_stocks}")
        
        # 重現TD3的預處理流程
        returns = df.pct_change().fillna(0)
        
        # 技術特徵計算 (適用於任意數量股票)
        momentum_5d = returns.rolling(window=5).mean().fillna(0)
        momentum_20d = returns.rolling(window=20).mean().fillna(0)
        volatility_20d = returns.rolling(window=20).std().fillna(0)
        
        # RSI
        price_delta = df.diff()
        gain = price_delta.clip(lower=0).rolling(window=14).mean()
        loss = (-price_delta.clip(upper=0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50) / 100.0
        
        # 布林带
        ma_20 = df.rolling(window=20).mean()
        std_20 = df.rolling(window=20).std()
        upper_band = ma_20 + (2 * std_20)
        lower_band = ma_20 - (2 * std_20)
        bollinger_b = (df - lower_band) / (upper_band - lower_band + 1e-8)
        bollinger_b = bollinger_b.fillna(0.5)
        
        # 合并特征
        feature_df = pd.concat([
            momentum_5d.add_suffix('_mom5d'),
            momentum_20d.add_suffix('_mom20d'),
            volatility_20d.add_suffix('_vol20d'),
            rsi.add_suffix('_rsi'),
            bollinger_b.add_suffix('_bollinger_b'),
        ], axis=1)
        
        # 关键：避免前视偏差的shift
        feature_df = feature_df.shift(1)
        shifted_returns = returns.copy().shift(-1)  # t动作对t->t+1的报酬
        
        # 对齐并去除NaN
        combined = pd.concat([feature_df, shifted_returns], axis=1).dropna()
        aligned_feature_df = combined[feature_df.columns]
        aligned_returns = combined[shifted_returns.columns]
        
        # 数据集分割 (重现TD3的分割逻辑)
        total = len(aligned_feature_df)
        train_end = int(0.7 * total)
        val_end = int(0.8 * total)
        
        # 测试集数据 (对应TD3的test_env)
        test_features = aligned_feature_df.iloc[val_end:].values
        test_returns = aligned_returns.iloc[val_end:].values
        test_dates = aligned_feature_df.iloc[val_end:].index
        
        print(f"\n重現TD3數據分割:")
        print(f"  測試期間: {test_dates[0]} 到 {test_dates[-1]}")
        print(f"  測試天數: {len(test_returns)}")
        print(f"  權重數據天數: {len(weights_data)}")
        
    except Exception as e:
        print(f"數據預處理失敗: {e}")
        return None
    
    # ==========================================
    # 3. 验证数据长度匹配
    # ==========================================
    
    if len(weights_data) != len(test_returns):
        print(f"警告：數據長度不匹配!")
        print(f"  權重數據: {len(weights_data)} 天")
        print(f"  測試收益: {len(test_returns)} 天")
        
        # 取較短的長度
        min_length = min(len(weights_data), len(test_returns))
        weights_data = weights_data.iloc[:min_length]
        test_returns = test_returns[:min_length]
        test_dates = test_dates[:min_length]
        print(f"  調整為: {min_length} 天")
    
    # ==========================================
    # 4. 通用化的收益計算 (保持TD3訓練邏輯)
    # ==========================================
    
    print(f"\n計算策略收益 (通用化但保持TD3邏輯)...")
    
    strategy_returns = []
    
    # 关键参数 (与TD3环境一致)
    risk_free_rate = 0.02 / 252  # 日化现金收益率
    
    for i in range(len(test_returns)):
        # 当日股票收益 (按td3_stocks順序)
        daily_stock_returns = test_returns[i]
        daily_stock_returns = np.nan_to_num(daily_stock_returns, nan=0.0)
        
        # 关键修正：與TD3邏輯一致 - 当天权重计算当天收益
        if i < len(weights_data):
            # 从权重数据中获取当天的权重配置 (通用化提取)
            weight_row = weights_data.iloc[i]
            
            # 通用化：按td3_stocks順序提取股票權重
            stock_weights = []
            for stock in td3_stocks:
                if stock in weight_row:
                    stock_weights.append(weight_row[stock])
                else:
                    print(f"警告：第{i+1}天找不到股票 {stock} 的權重，使用0")
                    stock_weights.append(0.0)
            
            current_weights = np.array(stock_weights)
            
            # 現金權重處理
            if has_cash and 'CASH' in weight_row:
                cash_weight = weight_row['CASH']
            else:
                # 如果沒有現金權重，計算剩餘權重
                cash_weight = 1.0 - np.sum(current_weights)
                cash_weight = max(0.0, cash_weight)  # 確保非負
                
            # 合併權重 (為了與原版邏輯一致)
            full_weights = np.append(current_weights, cash_weight)
        else:
            # 如果权重数据不够，使用等权重作为默认
            n_stocks = len(td3_stocks)
            stock_weight_each = 0.8 / n_stocks
            current_weights = np.array([stock_weight_each] * n_stocks)
            cash_weight = 0.2
            full_weights = np.append(current_weights, cash_weight)
        
        # 確保權重有效性 (與原版邏輯一致)
        total_weight = np.sum(full_weights)
        if abs(total_weight - 1.0) > 0.01 or np.any(np.isnan(full_weights)):
            # 使用等權重作為默認
            n_stocks = len(td3_stocks)
            current_weights = np.array([0.8/n_stocks] * n_stocks)
            cash_weight = 0.2
            full_weights = np.append(current_weights, cash_weight)
        else:
            current_weights = full_weights[:-1]
            cash_weight = full_weights[-1]
        
        # 计算投资组合收益 (完全按照TD3 step函数逻辑)
        stock_return = np.dot(current_weights, daily_stock_returns)
        cash_return = cash_weight * risk_free_rate
        portfolio_return = stock_return + cash_return
        
        # 应用TD3的收益限制
        portfolio_return = np.clip(portfolio_return, -0.5, 0.5)
        
        strategy_returns.append(portfolio_return)
        
        # 调试信息 (前5天)
        if i < 5:
            print(f"  第{i+1}天:")
            print(f"    股票權重: {current_weights}")
            print(f"    現金權重: {cash_weight:.3f}")
            print(f"    股票收益: {stock_return:.6f}")
            print(f"    現金收益: {cash_return:.6f}")
            print(f"    總收益: {portfolio_return:.6f}")
    
    # ==========================================
    # 5. 创建时间序列
    # ==========================================
    
    td3_returns_series = pd.Series(strategy_returns, index=test_dates)
    
    # ==========================================
    # 6. 保存结果 (通用化檔名)
    # ==========================================
    
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 使用時間戳和股票數量創建唯一檔名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_stocks = len(td3_stocks)
    output_file = f'results/td3_strategy_returns_{n_stocks}stocks_{timestamp}.csv'
    
    td3_returns_df = pd.DataFrame({
        'Strategy_Return': td3_returns_series
    })
    td3_returns_df.to_csv(output_file)
    
    print(f"通用版TD3策略報酬率已保存至: {output_file}")
    
    # ==========================================
    # 7. 详细性能分析 (保持原版分析邏輯)
    # ==========================================
    
    # 累积收益
    cumulative_returns = (1 + pd.Series(strategy_returns)).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    
    # 基础绩效指标
    volatility = pd.Series(strategy_returns).std() * np.sqrt(252)
    sharpe_ratio = (pd.Series(strategy_returns).mean() - risk_free_rate) / pd.Series(strategy_returns).std() * np.sqrt(252)
    
    # 最大回撤
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = abs(drawdown.min())
    
    # 其他指标
    win_rate = (pd.Series(strategy_returns) > 0).mean()
    avg_return = pd.Series(strategy_returns).mean()
    
    print(f"\nTD3策略绩效分析 (通用版):")
    print(f"股票組合: {td3_stocks}")
    print(f"測試期間: {test_dates[0].strftime('%Y-%m-%d')} 到 {test_dates[-1].strftime('%Y-%m-%d')}")
    print(f"測試天數: {len(strategy_returns)} 個交易日")
    print(f"總收益率: {total_return*100:.2f}%")
    print(f"年化收益率: {((1 + total_return)**(252/len(strategy_returns)) - 1)*100:.2f}%")
    print(f"年化波動率: {volatility*100:.2f}%")
    print(f"夏普比率: {sharpe_ratio:.3f}")
    print(f"最大回撤: {max_drawdown*100:.2f}%")
    print(f"胜率: {win_rate*100:.1f}%")
    print(f"平均日收益: {avg_return*10000:.2f} bps")
    
    # 权重分析 (通用化)
    print(f"\n平均權重分配:")
    if has_cash:
        avg_weights = weights_data[td3_stocks + ['CASH']].mean()
        for stock in td3_stocks:
            print(f"  {stock}: {avg_weights[stock]*100:.2f}%")
        print(f"  現金: {avg_weights['CASH']*100:.2f}%")
    else:
        avg_weights = weights_data[td3_stocks].mean()
        for stock in td3_stocks:
            print(f"  {stock}: {avg_weights[stock]*100:.2f}%")
        print(f"  現金: (計算得出)")
    
    # 权重统计
    if has_cash:
        weight_stats = weights_data[td3_stocks + ['CASH']].describe()
        print(f"\n權重統計 (最小值/最大值):")
        for stock in td3_stocks:
            print(f"  {stock}: {weight_stats.loc['min', stock]*100:.1f}% - {weight_stats.loc['max', stock]*100:.1f}%")
        print(f"  現金: {weight_stats.loc['min', 'CASH']*100:.1f}% - {weight_stats.loc['max', 'CASH']*100:.1f}%")
    
    # ==========================================
    # 8. 保存详细分析报告 (通用化)
    # ==========================================
    
    report_file = f'results/td3_strategy_analysis_{n_stocks}stocks_{timestamp}.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== TD3策略分析報告 (通用版 - 支援任意股票組合) ===\n\n")
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"權重檔案: {weights_file_path}\n")
        f.write(f"價格檔案: {price_file_path}\n")
        f.write(f"測試期間: {test_dates[0].strftime('%Y-%m-%d')} 到 {test_dates[-1].strftime('%Y-%m-%d')}\n")
        f.write(f"測試天數: {len(strategy_returns)} 個交易日\n\n")
        
        f.write("通用化改進:\n")
        f.write("1. 自動識別股票組合: 不再硬編碼特定股票\n")
        f.write("2. 智能匹配價格數據: 確保所有股票都有對應價格\n")
        f.write("3. 動態權重提取: 適應任意數量和組合的股票\n")
        f.write("4. 錯誤處理強化: 更好的異常情況處理\n")
        f.write("5. 保持TD3邏輯: 完全遵循原始訓練邏輯\n\n")
        
        f.write(f"股票組合 ({len(td3_stocks)}支):\n")
        for i, stock in enumerate(td3_stocks, 1):
            f.write(f"  {i}. {stock}\n")
        f.write(f"包含現金部位: {'是' if has_cash else '否'}\n\n")
        
        f.write("绩效指标:\n")
        f.write(f"  總收益率: {total_return*100:.2f}%\n")
        f.write(f"  年化收益率: {((1 + total_return)**(252/len(strategy_returns)) - 1)*100:.2f}%\n")
        f.write(f"  年化波動率: {volatility*100:.2f}%\n")
        f.write(f"  夏普比率: {sharpe_ratio:.3f}\n")
        f.write(f"  最大回撤: {max_drawdown*100:.2f}%\n")
        f.write(f"  胜率: {win_rate*100:.1f}%\n")
        f.write(f"  平均日收益: {avg_return*10000:.2f} bps\n\n")
        
        f.write("權重分析:\n")
        if has_cash:
            for stock in td3_stocks:
                f.write(f"  {stock}: 平均 {avg_weights[stock]*100:.2f}%, 範圍 {weight_stats.loc['min', stock]*100:.1f}%-{weight_stats.loc['max', stock]*100:.1f}%\n")
            f.write(f"  現金: 平均 {avg_weights['CASH']*100:.2f}%, 範圍 {weight_stats.loc['min', 'CASH']*100:.1f}%-{weight_stats.loc['max', 'CASH']*100:.1f}%\n")
        else:
            for stock in td3_stocks:
                f.write(f"  {stock}: 平均 {avg_weights[stock]*100:.2f}%\n")
    
    print(f"完整分析報告已保存至: {report_file}")
    
    # ==========================================
    # 9. 保存累积收益曲线数据
    # ==========================================
    
    cumulative_file = f'results/td3_cumulative_returns_{n_stocks}stocks_{timestamp}.csv'
    cumulative_df = pd.DataFrame({
        'Date': test_dates,
        'Daily_Return': strategy_returns,
        'Cumulative_Return': cumulative_returns.values,
        'Cumulative_Return_Pct': (cumulative_returns.values - 1) * 100
    })
    cumulative_df.to_csv(cumulative_file, index=False)
    print(f"累積收益曲線數據已保存至: {cumulative_file}")
    
    return td3_returns_series, output_file

# 使用範例和測試函數
def main():
    """主函數 - 提供使用指南"""
    print("通用版TD3策略轉換器使用指南")
    print("="*50)
    
    print("使用方法:")
    print("1. 準備權重檔案 (CSV格式，包含股票權重和CASH)")
    print("2. 準備價格檔案 (CSV格式，包含所有股票的歷史價格)")
    print("3. 執行轉換")
    print()
    
    # 互動式使用
    weights_file = input("請輸入權重檔案路徑 (按Enter使用預設): ").strip()
    if not weights_file:
        weights_file = None  # 使用預設路徑
    
    price_file = input("請輸入價格檔案路徑 (按Enter使用預設): ").strip()
    if not price_file:
        price_file = 'eight_stock_prices.csv'
    
    print(f"\n開始轉換...")
    result = convert_td3_weights_to_returns_universal(weights_file, price_file)
    
    if result:
        returns_series, output_file = result
        print(f"\n轉換成功!")
        print(f"輸出檔案: {output_file}")
        print(f"\n在benchmark分析中使用:")
        print(f"custom_strategies = {{'TD3_Universal': pd.read_csv('{output_file}', index_col=0, parse_dates=True).iloc[:, 0]}}")
    else:
        print("轉換失敗，請檢查錯誤訊息")

if __name__ == "__main__":
    # 可以直接呼叫main()進行互動式使用
    # 或者直接呼叫轉換函數
    
    # 範例：直接轉換 (使用你的9支股票)
    # result = convert_td3_weights_to_returns_universal(
    #     weights_file_path='your_9_stocks_weights.csv',
    #     price_file_path='nine_stock_prices.csv'
    # )
    
    main()
