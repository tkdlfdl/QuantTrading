"""
STRATEGY D: Comprehensive Grid Search
Testing: Bubble Threshold × Hold Period × Top-N Stocks

This is a 3D grid search covering:
- Thresholds: -0.95 to -0.05 (fine granularity)
- Hold Periods: 1h to 104h
- Top-N: 5 to 30 stocks
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

print("=" * 130)
print("STRATEGY D: COMPREHENSIVE GRID SEARCH - THRESHOLD × HOLD PERIOD × TOP-N (2024-2026)")
print("=" * 130)

# Load data
print("\n[Loading Data]")
hourly_close = pd.read_parquet('data/cache/merged_hourly_close.parquet')
hourly_ret = hourly_close.pct_change().fillna(0)

print(f"  Period: {hourly_close.index[0]} to {hourly_close.index[-1]}")
print(f"  Bars: {len(hourly_close):,}, Stocks: {len(hourly_close.columns)}")

# Calculate bubble scores once (vectorized)
print(f"\n[Computing Bubble Scores (Once)]")

MA_WINDOW = 104
TRANSACTION_COST = 0.001

log_p = np.log(hourly_close.replace(0, np.nan).ffill().bfill())
fair = hourly_close.rolling(MA_WINDOW, min_periods=MA_WINDOW//2).mean()
residual = log_p - np.log(fair.replace(0, np.nan).bfill())

z_scores = (residual - residual.rolling(MA_WINDOW, min_periods=MA_WINDOW//2).mean()) / \
           residual.rolling(MA_WINDOW, min_periods=MA_WINDOW//2).std()

bubble_scores = np.tanh(z_scores / 2).fillna(0)

print(f"  Bubble scores computed: {bubble_scores.shape}")

# Comprehensive grid search parameters
THRESHOLDS = np.arange(-0.95, -0.04, 0.1).round(2).tolist()  # -0.95, -0.85, -0.75, ..., -0.15, -0.05
HOLD_PERIODS = [1, 2, 4, 6, 8, 13, 26, 39, 52, 65, 78, 104]
TOP_N_VALUES = [5, 10, 15, 20, 25, 30]

print(f"\n[Grid Search Setup]")
print(f"  Thresholds: {len(THRESHOLDS)} values from {min(THRESHOLDS)} to {max(THRESHOLDS)}")
print(f"  Hold periods: {len(HOLD_PERIODS)} values from {min(HOLD_PERIODS)}h to {max(HOLD_PERIODS)}h")
print(f"  Top-N values: {TOP_N_VALUES}")
print(f"  Total combinations: {len(THRESHOLDS) * len(HOLD_PERIODS) * len(TOP_N_VALUES)}")

# Run grid search
results = []
best_sharpe = -999
best_params = None

print(f"\n[Running Grid Search]")
print(f"{'Threshold':<12} {'Hold':<6} {'Top-N':<6} {'Annual%':<10} {'Sharpe':<8} {'MaxDD%':<10} {'WinRate%':<10} {'Status'}")
print("-" * 100)

combo_count = 0
total_combos = len(THRESHOLDS) * len(HOLD_PERIODS) * len(TOP_N_VALUES)

for threshold_idx, threshold in enumerate(THRESHOLDS):
    for hold_idx, hold_period in enumerate(HOLD_PERIODS):
        for top_n_idx, top_n in enumerate(TOP_N_VALUES):
            combo_count += 1

            # Backtest with these parameters
            hourly_pnl = []

            for i in range(MA_WINDOW + MA_WINDOW, len(hourly_close) - hold_period - 1):
                # Find signals
                signals = bubble_scores.iloc[i]
                depressed = signals[signals < threshold].nsmallest(top_n)

                if len(depressed) > 0:
                    # Get returns from next hold_period bars
                    position_returns = []

                    for j in range(i + 1, min(i + 1 + hold_period, len(hourly_ret))):
                        ret = hourly_ret.iloc[j][depressed.index].mean()
                        position_returns.append(ret)

                    if position_returns:
                        daily_pnl = np.mean(position_returns) - TRANSACTION_COST / hold_period
                    else:
                        daily_pnl = 0
                else:
                    daily_pnl = 0

                hourly_pnl.append(daily_pnl)

            # Convert to daily and calculate metrics
            if len(hourly_pnl) == 0:
                continue

            hourly_returns = pd.Series(hourly_pnl, index=hourly_close.index[MA_WINDOW + MA_WINDOW:-hold_period-1])
            daily_returns = hourly_returns.groupby(hourly_returns.index.date).apply(lambda x: (1 + x).prod() - 1)
            daily_returns.index = pd.to_datetime(daily_returns.index)

            if len(daily_returns) == 0:
                continue

            # Metrics
            wealth = (1 + daily_returns).cumprod()
            total_ret = wealth.iloc[-1] / wealth.iloc[0] - 1
            annual_ret = (wealth.iloc[-1] / wealth.iloc[0]) ** (252 / len(daily_returns)) - 1
            sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
            max_dd = ((wealth / wealth.cummax()) - 1).min()
            win_rate = (daily_returns > 0).sum() / len(daily_returns) * 100

            results.append({
                'threshold': threshold,
                'hold_period': hold_period,
                'top_n': top_n,
                'annual_return': annual_ret * 100,
                'sharpe': sharpe,
                'max_dd': max_dd * 100,
                'win_rate': win_rate,
                'total_return': total_ret * 100,
            })

            print(f"{threshold:<12.2f} {hold_period:<6} {top_n:<6} {annual_ret*100:<10.2f} {sharpe:<8.2f} {max_dd*100:<10.2f} {win_rate:<10.1f} [{combo_count}/{total_combos}]")

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (threshold, hold_period, top_n)

# Create results dataframe
results_df = pd.DataFrame(results)

print(f"\n{'='*130}")
print(f"[BEST PARAMETERS]")
print(f"  Threshold: {best_params[0]}")
print(f"  Hold Period: {best_params[1]}h")
print(f"  Top-N: {best_params[2]}")
print(f"  Sharpe: {best_sharpe:.2f}")

best_row = results_df[
    (results_df['threshold'] == best_params[0]) &
    (results_df['hold_period'] == best_params[1]) &
    (results_df['top_n'] == best_params[2])
].iloc[0]
print(f"  Annual Return: {best_row['annual_return']:+.2f}%")
print(f"  Max DD: {best_row['max_dd']:.2f}%")
print(f"  Win Rate: {best_row['win_rate']:.1f}%")

# Top 20 by Sharpe
print(f"\n[TOP 20 BY SHARPE]")
top20 = results_df.nlargest(20, 'sharpe')
print(top20[['threshold', 'hold_period', 'top_n', 'annual_return', 'sharpe', 'max_dd', 'win_rate']].to_string(index=False))

# Save full results
results_df.to_csv('results/Strategy_D_Comprehensive_GridSearch_2024_2026.csv', index=False)
print(f"\n[OK] Full results saved to Strategy_D_Comprehensive_GridSearch_2024_2026.csv")
print(f"     Total combinations tested: {len(results_df)}")

# Comparison with baseline and first grid search
baseline_sharpe = 0.32
baseline_annual = 4.10
first_gs_sharpe = 1.99
first_gs_annual = 13.62

print(f"\n[COMPARISON]")
print(f"  Baseline (threshold=-0.8, hold=13h, top=20): Sharpe {baseline_sharpe}, Annual {baseline_annual:+.2f}%")
print(f"  First Grid Search (threshold=-0.8, hold=52h, top=20): Sharpe {first_gs_sharpe}, Annual {first_gs_annual:+.2f}%")
print(f"  Best Found: Sharpe {best_sharpe:.2f}, Annual {best_row['annual_return']:+.2f}%")
print(f"  Improvement over baseline: {(best_sharpe / baseline_sharpe - 1) * 100:+.1f}% Sharpe, {(best_row['annual_return'] / baseline_annual - 1) * 100:+.1f}% Annual")

print(f"\n{'='*130}")
print(f"COMPREHENSIVE GRID SEARCH COMPLETE")
print(f"{'='*130}")

