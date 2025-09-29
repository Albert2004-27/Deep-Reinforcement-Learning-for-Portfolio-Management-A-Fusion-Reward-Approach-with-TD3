# main_analysis.py
# Updated to work with Enhanced BenchmarkSuite

import pandas as pd
import numpy as np
import os
from datetime import datetime
from benchmark_suite import BenchmarkSuite  # Import enhanced version

def main():
    # ===========================================
    # 0. Create Results Folder
    # ===========================================
    
    # Create results folder (if not exists)
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created folder: {results_dir}/")
    
    # Create timestamped subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, f"benchmark_run_{timestamp}")
    os.makedirs(run_dir)
    print(f"Created run folder: {run_dir}/")
    
    # ===========================================
    # 1. Load Stock Data
    # ===========================================
    
    # Method A: Load from CSV
    stock_data = pd.read_csv('nine_stock_prices.csv', 
                            index_col=0, 
                            parse_dates=True)
    
    # Check data format
    print("Data overview:")
    print(f"Time range: {stock_data.index.min()} to {stock_data.index.max()}")
    print(f"Number of stocks: {len(stock_data.columns)}")
    print(f"Data length: {len(stock_data)} trading days")
    print("\nFirst 5 rows:")
    print(stock_data.head())
    
    # Save data summary to text file
    summary_file = os.path.join(run_dir, "data_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=== DATA OVERVIEW ===\n")
        f.write(f"Time range: {stock_data.index.min()} to {stock_data.index.max()}\n")
        f.write(f"Number of stocks: {len(stock_data.columns)}\n")
        f.write(f"Data length: {len(stock_data)} trading days\n")
        f.write(f"Stock symbols: {list(stock_data.columns)}\n")
        f.write("\n=== FIRST 5 ROWS ===\n")
        f.write(stock_data.head().to_string())
    
    # ===========================================
    # 2. Create Enhanced Benchmark Suite
    # ===========================================
    
    suite = BenchmarkSuite(
        price_data=stock_data,
        transaction_cost=0.0000  # 0.15% transaction cost (changed from 0.0000)
    )
    
    # ===========================================
    # 3. Set Test Parameters
    # ===========================================

    # Test period for another time period
    #start_date = '2014-11-30'
    #end_date = '2015-10-04'   
    
    # Test period for eight stocks
    start_date = '2022-06-26'
    end_date = '2024-12-29'
    
        
    # Test period for ten stocks
    # start_date = '2016-01-03'
    # end_date = '2019-12-29'
    
    # Risk-free rate
    cash_rate = 0.02  # 2%
    
    # ===========================================
    # 4. Load TD3 Strategy (BEFORE running benchmarks)
    # ===========================================

    print("Loading TD3 reinforcement learning strategy...")
    
    # TD3 strategy files
    strategy_files = {
        'TD3_Strategy': '/storage/ssd1/Albertchen2004/ISS/results/td3_cumulative_returns_9stocks_20250902_123547.csv',
    }

    custom_strategies = {}  # For enhanced suite
    
    for strategy_name, file_path in strategy_files.items():
        if os.path.exists(file_path):
            print(f"Loading strategy: {strategy_name}")
            
            # Read strategy data
            strategy_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            strategy_returns = strategy_data.iloc[:, 0]
            
            # Align test period
            strategy_returns_aligned = strategy_returns[start_date:end_date]
            
            if len(strategy_returns_aligned) > 0:
                # Add to custom strategies for enhanced analysis
                custom_strategies[strategy_name] = strategy_returns_aligned
                
                print(f"  {strategy_name} loaded successfully")
                print(f"  Test period: {strategy_returns_aligned.index.min()} to {strategy_returns_aligned.index.max()}")
                print(f"  Data length: {len(strategy_returns_aligned)} trading days")
            else:
                print(f"  {strategy_name} has no valid data in test period")
        else:
            print(f"  Strategy file not found: {file_path}")
            print("  Please run first: python td3_strategy_converter.py")

    # ===========================================
    # 5. OPTION A: Use Enhanced One-Click Analysis
    # ===========================================
    
    print(f"\nStarting ENHANCED Benchmark analysis ({start_date} to {end_date})...")
    print("=" * 70)
    
    # One-click enhanced analysis (recommended for new features)
    results = suite.generate_performance_report(
        start_date=start_date,
        end_date=end_date,
        custom_strategies=custom_strategies,  # Include TD3 automatically
        save_dir=run_dir
    )
    
    # ===========================================
    # 6. Enhanced Results Analysis
    # ===========================================
    
    # Display enhanced metrics
    print("\n" + "=" * 70)
    print("ENHANCED BENCHMARK PERFORMANCE SUMMARY")
    print("=" * 70)
    print(results)
    
    # Save enhanced results
    try:
        excel_file = os.path.join(run_dir, 'enhanced_benchmark_results.xlsx')
        results.to_excel(excel_file)
        print(f"\nEnhanced results saved to: {excel_file}")
    except ImportError:
        csv_file = os.path.join(run_dir, 'enhanced_benchmark_results.csv')
        results.to_csv(csv_file)
        print(f"\nEnhanced results saved to: {csv_file}")
    
    # ===========================================
    # 7. Statistical Significance Testing (NEW)
    # ===========================================
    
    if 'TD3_Strategy' in results.index:
        print("\n" + "=" * 70)
        print("STATISTICAL SIGNIFICANCE TESTING")
        print("=" * 70)
        
        # Test TD3 vs key benchmarks
        td3_returns = custom_strategies['TD3_Strategy']
        
        benchmarks_to_test = {
            'Equal Weight (Monthly Rebal)': suite.equal_weight_rebalance(start_date, end_date),
            'Minimum Variance': suite.minimum_variance(start_date, end_date),
            'Momentum (W-L)': suite.momentum_strategy(start_date, end_date)
        }
        
        significance_results = {}
        
        for bench_name, bench_returns in benchmarks_to_test.items():
            sig_test = suite.test_strategy_significance(
                td3_returns, 
                bench_returns,
                test_type='ttest'
            )
            significance_results[bench_name] = sig_test
            
            print(f"\nTD3 vs {bench_name}:")
            print(f"  Test: {sig_test.get('test_name', 'N/A')}")
            print(f"  p-value: {sig_test['p_value']:.4f}")
            print(f"  Significant at 5%: {'YES' if sig_test['is_significant'] else 'NO'}")
        
        # Save significance test results
        sig_file = os.path.join(run_dir, "significance_tests.txt")
        with open(sig_file, 'w', encoding='utf-8') as f:
            f.write("=== STATISTICAL SIGNIFICANCE TESTS ===\n\n")
            for bench_name, result in significance_results.items():
                f.write(f"TD3 vs {bench_name}:\n")
                f.write(f"  Test: {result.get('test_name', 'N/A')}\n")
                f.write(f"  Test statistic: {result['test_stat']:.4f}\n")
                f.write(f"  p-value: {result['p_value']:.4f}\n")
                f.write(f"  Significant at 5%: {'YES' if result['is_significant'] else 'NO'}\n")
                f.write(f"  Sample size: {result.get('n_observations', 'N/A')}\n\n")

    # ===========================================
    # 8. Enhanced Performance Analysis
    # ===========================================
    
    # Find best strategies across all enhanced metrics
    best_sharpe = results['Sharpe Ratio'].idxmax()
    best_sortino = results['Sortino Ratio'].idxmax()
    best_calmar = results['Calmar Ratio'].replace([np.inf, -np.inf], np.nan).idxmax()
    best_return = results['Total Return (%)'].idxmax()
    lowest_dd = results['Max Drawdown (%)'].idxmin()
    lowest_var = results['VaR_95 (%)'].idxmax()  # Higher VaR is better (less negative)
    
    print(f"\n" + "="*70)
    print("TOP PERFORMERS ACROSS ALL METRICS")
    print("="*70)
    print(f"Best Sharpe Ratio: {best_sharpe} ({results.loc[best_sharpe, 'Sharpe Ratio']:.3f})")
    print(f"Best Sortino Ratio: {best_sortino} ({results.loc[best_sortino, 'Sortino Ratio']:.3f})")
    if pd.notna(results.loc[best_calmar, 'Calmar Ratio']):
        print(f"Best Calmar Ratio: {best_calmar} ({results.loc[best_calmar, 'Calmar Ratio']:.3f})")
    print(f"Highest Return: {best_return} ({results.loc[best_return, 'Total Return (%)']:.1f}%)")
    print(f"Lowest Drawdown: {lowest_dd} ({results.loc[lowest_dd, 'Max Drawdown (%)']:.1f}%)")
    print(f"Best VaR (95%): {lowest_var} ({results.loc[lowest_var, 'VaR_95 (%)']:.2f}%)")
    
    # TD3 specific analysis
    if 'TD3_Strategy' in results.index:
        td3_metrics = results.loc['TD3_Strategy']
        
        # Calculate rankings
        rankings = {}
        for metric in ['Sharpe Ratio', 'Sortino Ratio', 'Total Return (%)', 'Calmar Ratio']:
            if metric in results.columns:
                if metric == 'Total Return (%)':
                    rank = (results[metric] >= td3_metrics[metric]).sum()
                else:
                    rank = (results[metric].replace([np.inf, -np.inf], np.nan) >= td3_metrics[metric]).sum()
                rankings[metric] = rank
        
        print(f"\nTD3 STRATEGY DETAILED ANALYSIS")
        print("="*40)
        for metric, rank in rankings.items():
            print(f"{metric}: {rank}/{len(results)} (Top {rank/len(results)*100:.1f}%)")
        
        # Performance assessment
        if rankings.get('Sharpe Ratio', 999) == 1:
            print("\nCongratulations! TD3 achieved the highest risk-adjusted return!")
        elif rankings.get('Sharpe Ratio', 999) <= 3:
            print("\nExcellent! TD3 performed very well, ranking in the top 3!")

    # ===========================================
    # 9. OPTION B: Manual Step-by-Step (if you prefer the old way)
    # ===========================================
    
    # Uncomment this section if you want to use the old manual approach instead
    """
    print(f"\nStarting Manual Benchmark analysis ({start_date} to {end_date})...")
    
    # Run just the benchmarks first
    benchmark_results = suite.run_benchmarks(
        start_date=start_date,
        end_date=end_date,
        cash_rate=cash_rate
    )
    
    # Manually add TD3 if available
    if custom_strategies:
        for name, returns in custom_strategies.items():
            custom_metrics = suite.calculate_performance_metrics(returns, cash_rate)
            benchmark_results.loc[name] = custom_metrics
    
    # Manual plotting
    suite.plot_cumulative_returns(start_date, end_date, save_dir=run_dir, 
                                 custom_strategies=custom_strategies)
    suite.plot_performance_comparison(benchmark_results, save_dir=run_dir)
    suite.plot_rolling_metrics(start_date, end_date, save_dir=run_dir,
                              custom_strategies=custom_strategies)
    
    results = benchmark_results
    """

    # ===========================================
    # 10. Create Enhanced Run Summary  
    # ===========================================

    print(f"\nAll results saved to: {run_dir}/")
    print("File list:")
    for file in sorted(os.listdir(run_dir)):
        file_path = os.path.join(run_dir, file)
        size_kb = os.path.getsize(file_path) // 1024
        print(f"   {file} ({size_kb} KB)")

    # Create enhanced README file
    readme_file = os.path.join(run_dir, "README.txt")
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write("=== ENHANCED BENCHMARK ANALYSIS RESULTS ===\n\n")
        f.write("File descriptions:\n")
        f.write("├── enhanced_benchmark_results.xlsx      # Complete enhanced results table\n")
        f.write("├── significance_tests.txt               # Statistical significance tests\n")
        f.write("├── data_summary.txt                    # Original data overview\n")
        f.write("├── performance_metrics.xlsx            # Detailed metrics export\n")
        f.write("├── benchmark_cumulative_returns.png/pdf # Cumulative returns chart\n")
        f.write("├── enhanced_performance_comparison.png/pdf # 6-panel enhanced comparison\n")
        f.write("├── rolling_performance_metrics.png/pdf # 4-panel rolling analysis\n")
        f.write("└── README.txt                          # This description file\n\n")
        
        f.write("ENHANCED METRICS INCLUDED:\n")
        f.write("• Traditional: Total Return, Volatility, Sharpe Ratio, Max Drawdown, Win Rate\n")
        f.write("• Risk-Adjusted: Sortino Ratio, Calmar Ratio, Omega Ratio\n")
        f.write("• Tail Risk: VaR (95%), CVaR (95%)\n")
        f.write("• Resilience: Max Consecutive Losses, Max Loss Magnitude\n")
        f.write("• Rolling Analysis: Dynamic Sharpe, Sortino, Volatility, Drawdown\n")
        f.write("• Statistical Testing: t-tests for significance\n\n")
        
        f.write("BENCHMARK STRATEGIES:\n")
        f.write("1. Equal Weight (Monthly Rebal) - Equal weight monthly rebalancing\n")
        f.write("2. Equal Weight (Buy & Hold)    - Equal weight buy and hold\n")
        f.write("3. Inverse Volatility           - Inverse volatility weighting\n")
        f.write("4. Minimum Variance             - Minimum variance optimization\n")
        f.write("5. Momentum (W-L)               - Momentum strategy (winners-losers)\n")
        f.write("6. BLSW (L-W)                   - Contrarian momentum (losers-winners)\n")
        if 'TD3_Strategy' in results.index:
            f.write("7. TD3_Strategy                 - Reinforcement learning strategy\n")
        
        f.write(f"\nAnalysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test period: {start_date} to {end_date}\n")
        f.write(f"Transaction cost: 0.15%\n")
        f.write(f"Risk-free rate: {cash_rate*100}%\n")
    
    return results, run_dir

def load_sample_data():
    """Generate sample data for testing if needed"""
    print("Generating sample data for testing...")
    
    dates = pd.date_range('2021-01-01', '2024-12-31', freq='D')
    stock_names = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']
    
    np.random.seed(42)
    trends = np.random.normal(0.0008, 0.0003, len(stock_names))
    volatilities = np.random.uniform(0.15, 0.35, len(stock_names))
    
    prices = pd.DataFrame(index=dates, columns=stock_names)
    prices.iloc[0] = 100
    
    for i, stock in enumerate(stock_names):
        for t in range(1, len(dates)):
            daily_return = np.random.normal(trends[i]/252, volatilities[i]/np.sqrt(252))
            prices.iloc[t, i] = prices.iloc[t-1, i] * (1 + daily_return)
    
    return prices

if __name__ == "__main__":
    print("Starting ENHANCED Benchmark Analysis...")
    print("Checking environment...")
    
    # Check necessary packages
    try:
        import matplotlib.pyplot as plt
        print("matplotlib installed")
    except ImportError:
        print("Recommend installing matplotlib: pip install matplotlib")
    
    try:
        import openpyxl
        print("openpyxl installed")
    except ImportError:
        print("Recommend installing openpyxl: pip install openpyxl")
    
    try:
        from scipy import stats
        print("scipy installed (for significance tests)")
    except ImportError:
        print("Recommend installing scipy: pip install scipy")
    
    # Check data file
    if os.path.exists('nine_stock_prices.csv'):
        print("Found data file: nine_stock_prices.csv")
    else:
        print("Data file not found: nine_stock_prices.csv")
        print("Please ensure the file path is correct")
        exit(1)
    
    print("\n" + "="*50)
    
    # Run enhanced analysis
    try:
        results, run_dir = main()
        print(f"\nAnalysis completed! All results saved to: {run_dir}/")
        print("\nYour analysis now includes:")
        print("• 13 enhanced performance metrics")
        print("• Professional-grade visualizations") 
        print("• Statistical significance testing")
        print("• Rolling performance analysis")
        print("• Comprehensive risk assessment")
    except Exception as e:
        print(f"\nError occurred during analysis: {e}")
        print("Please check data format and dependencies are correctly installed")
