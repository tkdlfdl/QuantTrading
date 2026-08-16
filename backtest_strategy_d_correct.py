"""
STRATEGY D: Contrarian Bubble Score (CORRECT BACKTEST)
Period: 2024-2026 (Recent hourly data)

KEY FIX: Returns are calculated on FUTURE bars (after entry), not same-bar
Buy at bar i, hold until bar i+13
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

print("=" * 100)
print("STRATEGY D: CONTRARIAN BUBBLE SCORE - CORRECT BACKTEST (2024-2026)")
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

# Calculate bubble scores (vectorized)
print(f"\n[Computing Bubble Scores]")

log_p = np.log(hourly_close.replace(0, np.nan).ffill().bfill())
fair = hourly_close.rolling(MA_WINDOW, min_periods=MA_WINDOW//2).mean()
residual = log_p - np.log(fair.replace(0, np.nan).bfill())

z_scores = (residual - residual.rolling(MA_WINDOW, min_periods=MA_WINDOW//2).mean()) / \
           residual.rolling(MA_WINDOW, min_periods=MA_WINDOW//2).std()

bubble_scores = np.tanh(z_scores / 2).fillna(0)

print(f"  Bubble scores computed: {bubble_scores.shape}")

# Backtest with proper entry/exit and holding periods
print(f"\n[Backtesting]")

hourly_pnl = []
position_tracker = {}  # {stock: (entry_price, exit_bar)}

for i in range(MA_WINDOW + MA_WINDOW, len(hourly_close) - HOLD_PERIOD - 1):
    # Check which positions should exit
    positions_to_remove = []
    for stock, (entry_bar, exit_bar) in list(position_tracker.items()):
        if i >= exit_bar:
            positions_to_remove.append(stock)

    for stock in positions_to_remove:
        del position_tracker[stock]

    # Find new signals
    signals = bubble_scores.iloc[i]
    depressed = signals[signals < THRESHOLD].nsmallest(TOP_N - len(position_tracker))

    # Entry: buy at open of next bar (i+1)
    entry_bar = i + 1
    exit_bar = min(entry_bar + HOLD_PERIOD, len(hourly_close) - 1)

    for stock in depressed.index:
        if stock not in position_tracker and entry_bar < len(hourly_close):
            position_tracker[stock] = (entry_bar, exit_bar)

    # Calculate P&L for active positions
    # P&L = average return of positions held at bar i+1 to i+HOLD_PERIOD
    active_positions = list(position_tracker.keys())

    if len(active_positions) > 0:
        # Get the return FROM bar i+1 to i+HOLD_PERIOD for each position
        position_returns = []

        for stock, (ent_bar, ext_bar) in position_tracker.items():
            if i >= ent_bar - 1:  # Position has been entered
                # Calculate cumulative return from entry to now (or exit)
                current_exit_bar = min(i + 1, ext_bar)
                if ent_bar <= i and current_exit_bar > i:
                    ret = hourly_ret.iloc[i + 1][stock]
                    position_returns.append(ret)

        if position_returns:
            daily_pnl = np.mean(position_returns) - TRANSACTION_COST / HOLD_PERIOD
        else:
            daily_pnl = 0
    else:
        daily_pnl = 0

    hourly_pnl.append(daily_pnl)

    if i % 500 == 0:
        print(f"  Bar {i:,}/{len(hourly_close) - HOLD_PERIOD - 1:,}")

hourly_returns = pd.Series(hourly_pnl, index=hourly_close.index[MA_WINDOW + MA_WINDOW:-HOLD_PERIOD-1])

# Convert to daily
daily_returns = hourly_returns.groupby(hourly_returns.index.date).apply(lambda x: (1 + x).prod() - 1)
daily_returns.index = pd.to_datetime(daily_returns.index)

print(f"  Generated {len(daily_returns)} daily returns")

# Metrics
print(f"\n[RESULTS] 2024-2026")
wealth = (1 + daily_returns).cumprod()
total_ret = wealth.iloc[-1] / wealth.iloc[0] - 1
annual_ret = (wealth.iloc[-1] / wealth.iloc[0]) ** (252 / len(daily_returns)) - 1 if len(daily_returns) > 0 else 0
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
        y_sharpe = yr.mean() / yr.std() * np.sqrt(252) if yr.std() > 0 else 0
        print(f"  {year}: {(w.iloc[-1]/w.iloc[0]-1):+.2%} (Sharpe: {y_sharpe:.2f})")

# Save
print(f"\n[Saving Results]")
hourly_returns.to_csv('results/Strategy_D_Correct_Hourly_2024_2026.csv', header=['Return'])
daily_returns.to_csv('results/Strategy_D_Correct_Daily_2024_2026.csv', header=['Return'])

summary = pd.DataFrame({
    'Metric': ['Total_Return_%', 'Annual_Return_%', 'Sharpe', 'Max_DD_%', 'Win_Rate_%'],
    'Value': [f"{total_ret*100:+.2f}", f"{annual_ret*100:+.2f}", f"{sharpe:.2f}", f"{max_dd*100:.2f}", f"{win_rate*100:.1f}"]
})
summary.to_csv('results/Strategy_D_Correct_Summary_2024_2026.csv', index=False)

print(f"[OK] Results saved")
print("=" * 100)

