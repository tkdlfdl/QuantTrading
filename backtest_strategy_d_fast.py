"""
STRATEGY D: Contrarian Bubble Score (FAST VECTORIZED VERSION)
Period: 2024-2026 (Recent hourly data with good coverage)
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

print("=" * 100)
print("STRATEGY D: CONTRARIAN BUBBLE SCORE - FAST BACKTEST (2024-2026)")
print("=" * 100)

# Load data
print("\n[Loading Data]")
hourly_close = pd.read_parquet('data/cache/merged_hourly_close.parquet')
hourly_ret = hourly_close.pct_change().fillna(0)

print(f"  Period: {hourly_close.index[0]} to {hourly_close.index[-1]}")
print(f"  Bars: {len(hourly_close):,}, Stocks: {len(hourly_close.columns)}")

# Parameters
MA_WINDOW = 104
THRESHOLD = -0.8
HOLD_PERIOD = 13
TOP_N = 20
TRANSACTION_COST = 0.001

print(f"\n[Parameters]")
print(f"  MA Window: {MA_WINDOW}h, Threshold: {THRESHOLD}, Hold: {HOLD_PERIOD}h, Top-N: {TOP_N}")

# Calculate bubble scores for all stocks at once (vectorized)
print(f"\n[Computing Bubble Scores (Vectorized)]")

log_p = np.log(hourly_close.replace(0, np.nan).ffill().bfill())
fair = hourly_close.rolling(MA_WINDOW, min_periods=MA_WINDOW//2).mean()
residual = log_p - np.log(fair.replace(0, np.nan).bfill())

z_scores = (residual - residual.rolling(MA_WINDOW, min_periods=MA_WINDOW//2).mean()) / \
           residual.rolling(MA_WINDOW, min_periods=MA_WINDOW//2).std()

bubble_scores = np.tanh(z_scores / 2).fillna(0)

print(f"  Bubble scores computed: {bubble_scores.shape}")

# Find entry signals
print(f"\n[Finding Signals]")

signals = (bubble_scores < THRESHOLD).astype(int)
signal_counts = signals.sum(axis=1)
signal_days = (signal_counts > 0).sum()

print(f"  Days with at least 1 signal: {signal_days:,}/{len(signals):,} ({signal_days/len(signals)*100:.1f}%)")
print(f"  Avg signals per day: {signal_counts[signal_counts > 0].mean():.1f}")

# Simple strategy: On days with signals, take top-20 most depressed (lowest bubble score)
# Hold for HOLD_PERIOD days then exit
print(f"\n[Backtesting]")

returns = []

for i in range(MA_WINDOW + MA_WINDOW, len(bubble_scores)):
    # Find stocks with signal at this bar
    signal_stocks = bubble_scores.iloc[i]
    depressed = signal_stocks[signal_stocks < THRESHOLD].nsmallest(TOP_N)

    if len(depressed) > 0:
        # Return = average return of depressed stocks
        day_ret = hourly_ret.iloc[i][depressed.index].mean()
    else:
        day_ret = 0

    returns.append(day_ret)

    if (i - MA_WINDOW - MA_WINDOW) % 500 == 0:
        print(f"  Processed {i - MA_WINDOW - MA_WINDOW:,}/{len(bubble_scores) - MA_WINDOW - MA_WINDOW:,} bars...")

hourly_returns = pd.Series(returns, index=hourly_close.index[MA_WINDOW + MA_WINDOW:])

# Convert to daily
daily_returns = hourly_returns.groupby(hourly_returns.index.date).apply(lambda x: (1 + x).prod() - 1)
daily_returns.index = pd.to_datetime(daily_returns.index)

print(f"  Generated {len(daily_returns)} daily returns")

# Metrics
print(f"\n[RESULTS] 2024-2026")
wealth = (1 + daily_returns).cumprod()
total_ret = wealth.iloc[-1] / wealth.iloc[0] - 1
annual_ret = (wealth.iloc[-1] / wealth.iloc[0]) ** (252 / len(daily_returns)) - 1
sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
max_dd = ((wealth / wealth.cummax()) - 1).min()
win_rate = (daily_returns > 0).sum() / len(daily_returns)

print(f"  Total Return: {total_ret:+.2%}")
print(f"  Annual Return: {annual_ret:+.2%}")
print(f"  Sharpe: {sharpe:.2f}")
print(f"  Max DD: {max_dd:.2%}")
print(f"  Win Rate: {win_rate:.1%}")

# Yearly
print(f"\n[YEARLY]")
for year in [2024, 2025, 2026]:
    yr = daily_returns[daily_returns.index.year == year]
    if len(yr) > 0:
        w = (1 + yr).cumprod()
        print(f"  {year}: {(w.iloc[-1]/w.iloc[0]-1):+.2%}")

# Save
hourly_returns.to_csv('results/Strategy_D_Fast_Hourly_2024_2026.csv', header=['Return'])
daily_returns.to_csv('results/Strategy_D_Fast_Daily_2024_2026.csv', header=['Return'])

summary = pd.DataFrame({
    'Metric': ['Total_Return_%', 'Annual_Return_%', 'Sharpe', 'Max_DD_%', 'Win_Rate_%'],
    'Value': [f"{total_ret*100:+.2f}", f"{annual_ret*100:+.2f}", f"{sharpe:.2f}", f"{max_dd*100:.2f}", f"{win_rate*100:.1f}"]
})
summary.to_csv('results/Strategy_D_Fast_Summary_2024_2026.csv', index=False)

print(f"\n[OK] Results saved")
print("=" * 100)

