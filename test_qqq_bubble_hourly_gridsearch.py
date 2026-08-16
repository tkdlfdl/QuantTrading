"""
QQQ BUBBLE + BUY MOMENTUM STRATEGY - HOURLY GRIDSEARCH (2019-2026)
==================================================================
Strategy:
1. Calculate QQQ Bubble Score (daily MA window, daily Z-score window)
2. When bubble is LOW (undervalued), BUY top momentum stocks (hourly lookback)
3. HOLD for couple of days (grid search different periods)
4. Test all parameter combinations

Grid Search Parameters:
- QQQ Bubble Thresholds: -0.7, -0.6, -0.5
- Momentum Lookback: 10h, 20h, 30h hours
- Hold Periods: 8h (1 day), 16h (2 days), 24h (3 days), 40h (5 days)
- Top N Stocks: 5, 10
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os
import warnings
warnings.filterwarnings('ignore')
from itertools import product

print("="*140)
print("QQQ BUBBLE + BUY MOMENTUM - HOURLY GRIDSEARCH (2019-2026)")
print("="*140)

# Load hourly data
print("\nLoading hourly data...")
close_data = pd.read_parquet("data/cache/merged_hourly_close.parquet")
ret_hourly = close_data.pct_change().fillna(0)

print(f"Data: {close_data.shape}")
print(f"Period: {close_data.index[0]} to {close_data.index[-1]}")
years = (close_data.index[-1] - close_data.index[0]).days / 365.25
print(f"Years: {years:.2f}")

# ============================================================================
# FUNCTION: Calculate QQQ Bubble Score
# ============================================================================

def calculate_bubble_score_proxy(price, ma_window=252, z_window=252):
    """
    Calculate bubble score using QQQ price
    ma_window: moving average window (default 252 = 1 year of daily trading)
    z_window: z-score normalization window
    """
    log_price = np.log(np.maximum(price, 0.0001))

    fair_value = price.rolling(ma_window, min_periods=1).mean()
    fair_value = fair_value.clip(lower=0.0001)
    log_fair_value = np.log(fair_value)

    residual = log_price - log_fair_value

    residual_mean = residual.rolling(z_window, min_periods=1).mean()
    residual_std = residual.rolling(z_window, min_periods=1).std()
    residual_std = residual_std.clip(lower=1e-6)

    z = (residual - residual_mean) / residual_std

    return np.tanh(z / 2)

# Load QQQ data from qqq_hourly_close
print("\nLoading QQQ hourly data...")
try:
    qqq_data = pd.read_parquet("data/cache/qqq_hourly_close.parquet")
    qqq_close = qqq_data.iloc[:, 0]  # Get first column (QQQ)
    print(f"QQQ Data: {len(qqq_close)} hours ({qqq_close.index[0]} to {qqq_close.index[-1]})")

    # Calculate QQQ bubble on hourly data
    # Use smaller windows for hourly (6h MA ~ 1 day trading, 48h Z ~ 1 week)
    bubble_qqq = calculate_bubble_score_proxy(qqq_close, ma_window=24, z_window=120).fillna(0)
    print(f"QQQ Bubble calculated: min={bubble_qqq.min():.3f}, max={bubble_qqq.max():.3f}")

except Exception as e:
    print(f"Error loading QQQ data: {e}")
    print("Using market proxy instead...")
    market_proxy = close_data.mean(axis=1)
    bubble_qqq = calculate_bubble_score_proxy(market_proxy, ma_window=24, z_window=120).fillna(0)

# Align data
min_len = min(len(bubble_qqq), len(close_data))
bubble_qqq = bubble_qqq.iloc[:min_len]
close_data_aligned = close_data.iloc[:min_len]
ret_hourly_aligned = ret_hourly.iloc[:min_len]

print(f"Aligned data length: {min_len} hours")

# ============================================================================
# GRID SEARCH
# ============================================================================

print("\n" + "="*140)
print("GRID SEARCH PARAMETERS")
print("="*140)

bubble_thresholds = [-0.7, -0.6, -0.5]  # When to buy
momentum_lookbacks = [10, 20, 30]  # Hours
hold_periods = [8, 16, 24, 40]  # Hours (1, 2, 3, 5 days)
top_n_list = [5, 10]  # Top N stocks by momentum

total_combos = len(bubble_thresholds) * len(momentum_lookbacks) * len(hold_periods) * len(top_n_list)
print(f"\nTotal combinations: {total_combos}")
print(f"Bubble Thresholds: {bubble_thresholds}")
print(f"Momentum Lookbacks (h): {momentum_lookbacks}")
print(f"Hold Periods (h): {hold_periods}")
print(f"Top N Stocks: {top_n_list}\n")

results = []
best_sharpe = -np.inf
best_params = None

combo_num = 0

for bubble_threshold, lookback, hold_period, top_n in product(
    bubble_thresholds, momentum_lookbacks, hold_periods, top_n_list
):
    combo_num += 1

    if combo_num % 10 == 0:
        print(f"Testing {combo_num}/{total_combos}...")

    # Calculate momentum
    ret_mom = close_data_aligned.pct_change(lookback).fillna(0)

    strategy_returns = []

    for i in range(lookback + 1, len(bubble_qqq) - hold_period):
        bs = bubble_qqq.iloc[i]

        # Check if bubble is LOW enough to buy
        if bs < bubble_threshold:
            # Rank all stocks by momentum
            mom = ret_mom.iloc[i]
            top_n_idx = mom.nlargest(top_n).index

            # Calculate next hold_period returns
            future_rets = ret_hourly_aligned.iloc[i+1:i+1+hold_period, :][top_n_idx].values.mean()
            strat_ret = future_rets - 0.001
        else:
            strat_ret = 0

        strategy_returns.append(np.clip(strat_ret, -0.05, 0.05))

    if len(strategy_returns) == 0:
        continue

    # Calculate metrics
    s_series = pd.Series(strategy_returns)
    s_wealth = (1 + s_series).cumprod()

    total_ret = s_wealth.iloc[-1] - 1
    annual_ret = (1 + total_ret) ** (1/years) - 1 if total_ret > -1 else -1
    sharpe = s_series.mean() / s_series.std() * np.sqrt(252*6.5) if s_series.std() > 0 else 0
    dd = (s_wealth / s_wealth.cummax() - 1).min()
    win_rate = (s_series > 0).sum() / len(s_series) * 100

    results.append({
        'Bubble_Threshold': bubble_threshold,
        'Momentum_Lookback': lookback,
        'Hold_Period': hold_period,
        'Top_N': top_n,
        'Total_Return': total_ret,
        'Annual_Return': annual_ret,
        'Sharpe_Ratio': sharpe,
        'Max_Drawdown': dd,
        'Win_Rate': win_rate,
        'Data_Points': len(strategy_returns)
    })

    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_params = {
            'Bubble_Threshold': bubble_threshold,
            'Momentum_Lookback': lookback,
            'Hold_Period': hold_period,
            'Top_N': top_n
        }

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "="*140)
print("GRID SEARCH RESULTS")
print("="*140)

results_df = pd.DataFrame(results).sort_values('Sharpe_Ratio', ascending=False)

print("\nTop 10 Combinations (by Sharpe Ratio):")
print(results_df.head(10)[['Bubble_Threshold', 'Momentum_Lookback', 'Hold_Period', 'Top_N',
                            'Annual_Return', 'Sharpe_Ratio', 'Max_Drawdown', 'Win_Rate']].to_string(index=False))

print("\n" + "="*140)
print("BEST PARAMETERS")
print("="*140)

if best_params:
    print(f"\nBest Sharpe Ratio: {best_sharpe:.4f}")
    print(f"Bubble Threshold: {best_params['Bubble_Threshold']}")
    print(f"Momentum Lookback: {best_params['Momentum_Lookback']}h")
    print(f"Hold Period: {best_params['Hold_Period']}h (~{best_params['Hold_Period']/6.5:.1f} days)")
    print(f"Top N Stocks: {best_params['Top_N']}")

    best_result = results_df.iloc[0]
    print(f"\nPerformance with Best Params:")
    print(f"  Annual Return: {best_result['Annual_Return']:.2%}")
    print(f"  Total Return: {best_result['Total_Return']:.2%}")
    print(f"  Sharpe Ratio: {best_result['Sharpe_Ratio']:.4f}")
    print(f"  Max Drawdown: {best_result['Max_Drawdown']:.2%}")
    print(f"  Win Rate: {best_result['Win_Rate']:.1f}%")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Sharpe by Bubble Threshold
sharpe_by_bubble = results_df.groupby('Bubble_Threshold')['Sharpe_Ratio'].mean()
axes[0, 0].plot(sharpe_by_bubble.index, sharpe_by_bubble.values, marker='o', linewidth=2, markersize=8)
axes[0, 0].set_title('Average Sharpe by Bubble Threshold', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Bubble Threshold')
axes[0, 0].set_ylabel('Sharpe Ratio')
axes[0, 0].grid(True, alpha=0.3)

# Sharpe by Momentum Lookback
sharpe_by_lookback = results_df.groupby('Momentum_Lookback')['Sharpe_Ratio'].mean()
axes[0, 1].plot(sharpe_by_lookback.index, sharpe_by_lookback.values, marker='o', linewidth=2, markersize=8, color='orange')
axes[0, 1].set_title('Average Sharpe by Momentum Lookback', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Momentum Lookback (hours)')
axes[0, 1].set_ylabel('Sharpe Ratio')
axes[0, 1].grid(True, alpha=0.3)

# Sharpe by Hold Period
sharpe_by_hold = results_df.groupby('Hold_Period')['Sharpe_Ratio'].mean()
axes[1, 0].plot(sharpe_by_hold.index, sharpe_by_hold.values, marker='o', linewidth=2, markersize=8, color='green')
axes[1, 0].set_title('Average Sharpe by Hold Period', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Hold Period (hours)')
axes[1, 0].set_ylabel('Sharpe Ratio')
axes[1, 0].grid(True, alpha=0.3)

# Sharpe by Top N
sharpe_by_topn = results_df.groupby('Top_N')['Sharpe_Ratio'].mean()
axes[1, 1].bar(sharpe_by_topn.index, sharpe_by_topn.values, color='steelblue', alpha=0.7)
axes[1, 1].set_title('Average Sharpe by Top N Stocks', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Top N Stocks')
axes[1, 1].set_ylabel('Sharpe Ratio')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/qqq_bubble_momentum_hourly_gridsearch.png', dpi=150, bbox_inches='tight')
print("Saved: results/qqq_bubble_momentum_hourly_gridsearch.png")

# Save results
results_df.to_csv('results/qqq_bubble_momentum_hourly_results.csv', index=False)
print("Saved: results/qqq_bubble_momentum_hourly_results.csv")

print("\n" + "="*140)
print("GRID SEARCH COMPLETE")
print("="*140)

# Summary stats
print(f"\nSummary Statistics:")
print(f"  Average Sharpe: {results_df['Sharpe_Ratio'].mean():.4f}")
print(f"  Median Sharpe: {results_df['Sharpe_Ratio'].median():.4f}")
print(f"  Best Sharpe: {results_df['Sharpe_Ratio'].max():.4f}")
print(f"  Worst Sharpe: {results_df['Sharpe_Ratio'].min():.4f}")
print(f"  Positive Sharpe Combos: {(results_df['Sharpe_Ratio'] > 0).sum()}/{len(results_df)}")
