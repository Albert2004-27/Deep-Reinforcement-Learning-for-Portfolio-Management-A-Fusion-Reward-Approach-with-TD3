import pandas as pd
import numpy as np
import re
import os

# 修正的portfolio_value提取
test_feedback_file = '/storage/ssd1/Albertchen2004/ISS/td3_fusion_early_stop_20250802_184453/data/enhanced_td3_test_feedback_by_days_early_stop.csv'

print("修正的portfolio_value提取...")

test_df = pd.read_csv(test_feedback_file)

def extract_portfolio_value_fixed(env_info_str):
    """修正版：专门处理np.float64格式"""
    try:
        # 匹配 np.float64(数字) 格式
        pattern = r"'portfolio_value':\\s*np\\.float64\\(([0-9.]+)\\)"
        match = re.search(pattern, str(env_info_str))
        if match:
            return float(match.group(1))
        return None
    except:
        return None

# 测试前几行
print("测试提取:")
for i in range(3):
    test_str = test_df['env_info'].iloc[i]
    pv = extract_portfolio_value_fixed(test_str)
    print(f"第{i}行: {pv}")

# 提取所有数据
portfolio_values = []
failed_count = 0

for i, row in test_df.iterrows():
    pv = extract_portfolio_value_fixed(row['env_info'])
    if pv is not None:
        portfolio_values.append(pv)
    else:
        failed_count += 1

print(f"提取结果: {len(portfolio_values)} 成功, {failed_count} 失败")

if len(portfolio_values) >= 600:
    start_value = portfolio_values[0]
    end_value = portfolio_values[-1]
    total_return = (end_value / start_value - 1) * 100
    
    print(f"\\nTD3真实收益数据:")
    print(f"  起始: ${start_value:,.0f}")
    print(f"  最终: ${end_value:,.0f}")
    print(f"  总收益: {total_return:.2f}%")
    print(f"  目标: 463.41%")
    
    # 计算日收益率
    daily_returns = []
    for i in range(1, len(portfolio_values)):
        daily_ret = portfolio_values[i] / portfolio_values[i-1] - 1
        daily_returns.append(daily_ret)
    
    print(f"日收益率: {len(daily_returns)} 天")
    print(f"累积验证: {((1 + pd.Series(daily_returns)).prod() - 1) * 100:.2f}%")
    
    # 生成正确的策略文件
    if abs(total_return - 463.41) < 20:  # 允许一些误差
        # 获取测试期间的日期
        price_df = pd.read_csv('eight_stock_prices.csv', index_col=0, parse_dates=True)
        returns = price_df.pct_change().fillna(0)
        
        # 重现TD3数据分割
        feature_df = returns.rolling(20).mean().fillna(0).shift(1)
        combined = pd.concat([feature_df, returns.shift(-1)], axis=1).dropna()
        
        total_len = len(combined)
        val_end = int(0.8 * total_len)
        
        # 确保日期数量匹配
        available_dates = combined.index[val_end:]
        if len(available_dates) >= len(daily_returns):
            test_dates = available_dates[:len(daily_returns)]
            
            strategy_returns_series = pd.Series(daily_returns, index=test_dates)
            
            os.makedirs('results', exist_ok=True)
            output_file = 'results/td3_strategy_returns_true.csv'
            strategy_df = pd.DataFrame({'Strategy_Return': strategy_returns_series})
            strategy_df.to_csv(output_file)
            
            print(f"\\n真实TD3策略收益已保存: {output_file}")
            print(f"期间: {test_dates[0]} 到 {test_dates[-1]}")
            print(f"总收益: {((strategy_returns_series + 1).prod() - 1) * 100:.2f}%")
            
            # 策略统计
            vol = strategy_returns_series.std() * np.sqrt(252)
            sharpe = (strategy_returns_series.mean() - 0.02/252) / strategy_returns_series.std() * np.sqrt(252)
            
            print(f"年化波动: {vol:.1%}")
            print(f"夏普比率: {sharpe:.2f}")
            
        else:
            print(f"日期不足: 需要{len(daily_returns)}, 可用{len(available_dates)}")
    else:
        print(f"收益不匹配: {total_return:.2f}% vs 463.41%")
        
else:
    print("提取失败，检查数据格式")
    # 显示一些调试信息
    sample = test_df['env_info'].iloc[0]
    print(f"样本数据: {sample[:200]}...")