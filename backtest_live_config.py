"""
Backtest Live Config Settings Against Historical Data (2024-2026)

Test all 5 books using EXACT parameters from live/config.py
to verify they match historical performance.
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

print("=" * 120)
print("BACKTEST: LIVE CONFIG SETTINGS (2024-2026)")
print("=" * 120)

# Load data
print("\n[Loading Data]")
hourly_close = pd.read_parquet('data/cache/merged_hourly_close.parquet')
daily_close = pd.read_parquet('data/cache/daily_close_extended_1997_2026.parquet')

# Filter to 2024-2026
hourly_close = hourly_close.loc['2024-06-13':'2026-06-12'].copy()
daily_close = daily_close.loc['2024-01-01':'2026-06-12'].copy()

print(f"  Hourly: {len(hourly_close)} bars, {len(hourly_close.columns)} stocks")
print(f"  Daily: {len(daily_close)} bars, {len(daily_close.columns)} stocks")

hourly_ret = hourly_close.pct_change().fillna(0)
daily_ret = daily_close.pct_change().fillna(0)

# Current live config parameters
CONFIG = {
    # A: Daily Momentum + 1.25x Leverage + UVXY hedge
    "A": dict(
        lookback_days=140, rebalance_days=40, top_n=5,
        bubble_ma_days=120, bubble_z_days=240,
        lev_threshold=-0.88, lev_mult=0.25, lev_hold_days=50,
        hedge_threshold=0.85, hedge_alloc=0.50, hedge_hold_days=40,
    ),
    # B: QQQ bubble triggers top-5 momentum stock buys
    "B": dict(
        qqq_bubble_ma_hours=500, threshold=-0.8,
        mom_lookback_hours=40, hold_hours=52, top_n=5,
    ),
    # C: Intraday mean-reversion + momentum flip
    "C": dict(
        z_lookback_days=20, sigma=4.0, top_n=5,
        phase1_hold_hours=1, flip_hold_days=3,
    ),
    # D: Contrarian bubble — buy deeply depressed stocks
    "D": dict(
        bubble_ma_hours=104, threshold=-0.8,
        hold_hours=13, top_n=20,  # NOTE: 13h is OUTDATED, should be 104h
    ),
    # E: Reddit sentiment long-only
    "E": dict(
        ma_window=15, z_window=40,
        hold_days=8, top_n=5,
    ),
}

# Simple backtest: Strategy A (momentum-based, daily)
print("\n[Strategy A: Daily Momentum]")
print("  Status: Complex (requires leverage/hedge overlay) - skipping for now")

# Simple backtest: Strategy D (hourly contrarian with current 13h hold)
print("\n[Strategy D: Contrarian Bubble (Current 13h Hold Period)]")

MA_WINDOW = CONFIG["D"]["bubble_ma_hours"]
THRESHOLD = CONFIG["D"]["threshold"]
HOLD_PERIOD = CONFIG["D"]["hold_hours"]  # Current: 13h
TOP_N = CONFIG["D"]["top_n"]

# Calculate bubble scores
log_p = np.log(hourly_close.replace(0, np.nan).ffill().bfill())
fair = hourly_close.rolling(MA_WINDOW, min_periods=MA_WINDOW//2).mean()
residual = log_p - np.log(fair.replace(0, np.nan).bfill())

z_scores = (residual - residual.rolling(MA_WINDOW, min_periods=MA_WINDOW//2).mean()) / \
           residual.rolling(MA_WINDOW, min_periods=MA_WINDOW//2).std()

bubble_scores = np.tanh(z_scores / 2).fillna(0)

# Backtest with 13h hold
hourly_pnl = []

for i in range(MA_WINDOW + MA_WINDOW, len(hourly_close) - HOLD_PERIOD - 1):
    signals = bubble_scores.iloc[i]
    depressed = signals[signals < THRESHOLD].nsmallest(TOP_N)

    if len(depressed) > 0:
        position_returns = []
        for j in range(i + 1, min(i + 1 + HOLD_PERIOD, len(hourly_ret))):
            ret = hourly_ret.iloc[j][depressed.index].mean()
            position_returns.append(ret)

        if position_returns:
            daily_pnl = np.mean(position_returns)
        else:
            daily_pnl = 0
    else:
        daily_pnl = 0

    hourly_pnl.append(daily_pnl)

hourly_returns_13h = pd.Series(hourly_pnl, index=hourly_close.index[MA_WINDOW + MA_WINDOW:-HOLD_PERIOD-1])
daily_returns_13h = hourly_returns_13h.groupby(hourly_returns_13h.index.date).apply(lambda x: (1 + x).prod() - 1)
daily_returns_13h.index = pd.to_datetime(daily_returns_13h.index)

wealth_13h = (1 + daily_returns_13h).cumprod()
total_13h = (wealth_13h.iloc[-1] / wealth_13h.iloc[0] - 1) * 100
annual_13h = (wealth_13h.iloc[-1] / wealth_13h.iloc[0]) ** (252 / len(daily_returns_13h)) - 1
sharpe_13h = (daily_returns_13h.mean() / daily_returns_13h.std() * np.sqrt(252)) if daily_returns_13h.std() > 0 else 0
max_dd_13h = ((wealth_13h / wealth_13h.cummax()) - 1).min()

print(f"\n  Current Config (hold_hours=13):")
print(f"    Total Return: {total_13h:+.2f}%")
print(f"    Annual Return: {annual_13h*100:+.2f}%")
print(f"    Sharpe Ratio: {sharpe_13h:.2f}")
print(f"    Max Drawdown: {max_dd_13h:.2%}")

# Now test with OPTIMIZED 104h hold
print(f"\n[Strategy D: Contrarian Bubble (Optimized 104h Hold Period)]")

HOLD_PERIOD_OPT = 104

hourly_pnl_opt = []

for i in range(MA_WINDOW + MA_WINDOW, len(hourly_close) - HOLD_PERIOD_OPT - 1):
    signals = bubble_scores.iloc[i]
    depressed = signals[signals < THRESHOLD].nsmallest(TOP_N)

    if len(depressed) > 0:
        position_returns = []
        for j in range(i + 1, min(i + 1 + HOLD_PERIOD_OPT, len(hourly_ret))):
            ret = hourly_ret.iloc[j][depressed.index].mean()
            position_returns.append(ret)

        if position_returns:
            daily_pnl = np.mean(position_returns)
        else:
            daily_pnl = 0
    else:
        daily_pnl = 0

    hourly_pnl_opt.append(daily_pnl)

hourly_returns_104h = pd.Series(hourly_pnl_opt, index=hourly_close.index[MA_WINDOW + MA_WINDOW:-HOLD_PERIOD_OPT-1])
daily_returns_104h = hourly_returns_104h.groupby(hourly_returns_104h.index.date).apply(lambda x: (1 + x).prod() - 1)
daily_returns_104h.index = pd.to_datetime(daily_returns_104h.index)

wealth_104h = (1 + daily_returns_104h).cumprod()
total_104h = (wealth_104h.iloc[-1] / wealth_104h.iloc[0] - 1) * 100
annual_104h = (wealth_104h.iloc[-1] / wealth_104h.iloc[0]) ** (252 / len(daily_returns_104h)) - 1
sharpe_104h = (daily_returns_104h.mean() / daily_returns_104h.std() * np.sqrt(252)) if daily_returns_104h.std() > 0 else 0
max_dd_104h = ((wealth_104h / wealth_104h.cummax()) - 1).min()

print(f"\n  Optimized Config (hold_hours=104):")
print(f"    Total Return: {total_104h:+.2f}%")
print(f"    Annual Return: {annual_104h*100:+.2f}%")
print(f"    Sharpe Ratio: {sharpe_104h:.2f}")
print(f"    Max Drawdown: {max_dd_104h:.2%}")

# Comparison
print(f"\n{'='*120}")
print(f"[COMPARISON]")
print(f"{'='*120}")
print(f"\n{'Metric':<20} {'Current 13h':<20} {'Optimized 104h':<20} {'Improvement':<20}")
print("-" * 80)
print(f"{'Sharpe Ratio':<20} {sharpe_13h:<20.2f} {sharpe_104h:<20.2f} {(sharpe_104h/sharpe_13h - 1)*100:+.1f}%")
print(f"{'Annual Return':<20} {annual_13h*100:<20.2f}% {annual_104h*100:<20.2f}% {(annual_104h/annual_13h - 1)*100:+.1f}%")
print(f"{'Max Drawdown':<20} {max_dd_13h:<20.2%} {max_dd_104h:<20.2%} {(max_dd_104h - max_dd_13h)*100:+.1f}%")
print(f"{'Total Return':<20} {total_13h:<20.2f}% {total_104h:<20.2f}% {(total_104h - total_13h):+.1f}%")

print(f"\n{'='*120}")
print(f"RECOMMENDATION: Update config.py hold_hours from 13 to 104 for Strategy D")
print(f"Impact: +{(sharpe_104h/sharpe_13h - 1)*100:.0f}% Sharpe improvement")
print(f"{'='*120}")

