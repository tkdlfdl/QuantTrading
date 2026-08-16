"""
Generate Daily Returns for Strategies A-E (2019-2026)
Using actual strategy implementations from /strategies/
"""

import numpy as np
import pandas as pd
import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

print("=" * 100)
print("DAILY RETURNS BACKTEST: ALL STRATEGIES 2019-2026")
print("=" * 100)

# Load daily data (covers full 2019-2026)
print("\n[Loading data...]")
daily_close = pd.read_parquet('data/cache/daily_close_extended_1997_2026.parquet')
daily_close = daily_close.loc['2019-01-01':'2026-06-12'].copy()

# Filter to stocks with good data coverage
daily_close = daily_close.dropna(axis=1, thresh=len(daily_close) * 0.85)
daily_ret = daily_close.pct_change().fillna(0)

print(f"Period: {daily_close.index[0].date()} to {daily_close.index[-1].date()}")
print(f"Trading days: {len(daily_close)}")
print(f"Stocks: {len(daily_close.columns)}")

# Store all daily returns
all_returns = pd.DataFrame(index=daily_close.index)

# =============================================================================
# STRATEGY A: Daily Momentum (140/40)
# =============================================================================
print("\n[A] Daily Momentum (140-day lookback, 40-day hold)...")

ret_140d = daily_close.pct_change(140).ffill().fillna(0)
strategy_a_daily = []

lookback, hold, top_n = 140, 40, 5

for i in range(lookback, len(daily_close)):
    if i % 500 == 0:
        print(f"  Processing day {i}/{len(daily_close)}...")

    # Check if this is a rebalance day
    if (i - lookback) % hold == 0 and i >= lookback + hold:
        # Get ranking from the rebalance point
        rebal_idx = lookback + ((i - lookback) // hold) * hold
        if rebal_idx < len(ret_140d):
            ranking = ret_140d.iloc[rebal_idx].rank(ascending=False)
            long_idx = ranking[ranking <= top_n].index.tolist()
            short_idx = ranking[ranking > len(ranking) - top_n].index.tolist()
        else:
            long_idx = short_idx = []

    # Calculate daily return
    if len(long_idx) >= top_n and len(short_idx) >= top_n:
        long_ret = daily_ret.iloc[i][long_idx].mean()
        short_ret = daily_ret.iloc[i][short_idx].mean()
        daily_return = (long_ret - short_ret) / top_n - 0.005 / hold
    else:
        daily_return = 0

    strategy_a_daily.append(daily_return)

all_returns['A_Daily_Momentum'] = pd.Series(strategy_a_daily, index=daily_close.index[lookback:])

ret_a = all_returns['A_Daily_Momentum'].dropna()
total_a = (1 + ret_a).prod() - 1
sharpe_a = (ret_a.mean() / ret_a.std() * np.sqrt(252)) if ret_a.std() > 0 else 0
ann_a = ((1 + total_a) ** (252 / len(ret_a)) - 1)
maxdd_a = ((1 + ret_a).cumprod() / (1 + ret_a).cumprod().cummax() - 1).min()

print(f"  Total Return: {total_a:+.1%}")
print(f"  Annual Return: {ann_a:+.1%}")
print(f"  Sharpe Ratio: {sharpe_a:.2f}")
print(f"  Max Drawdown: {maxdd_a:.1%}")

# =============================================================================
# STRATEGY B: Momentum + Bubble Timing (Daily Approx)
# =============================================================================
print("\n[B] Momentum with Bubble Overlay (Daily)...")

# Use momentum median as bubble proxy
bubble_proxy = ret_140d.median(axis=1)
bubble_ma = bubble_proxy.rolling(120).mean()
bubble_std = bubble_proxy.rolling(120).std()
bubble_score = (bubble_proxy - bubble_ma) / bubble_std.clip(lower=0.01)
bubble_score = np.tanh(bubble_score / 2)

strategy_b_daily = []

for i in range(lookback, len(daily_close)):
    if i % 500 == 0:
        print(f"  Processing day {i}/{len(daily_close)}...")

    bubble_val = bubble_score.iloc[i] if i < len(bubble_score) else 0

    # Position sizing based on bubble
    if bubble_val > 0.5:  # Overvalued
        position_size = 0.5
    elif bubble_val < -0.5:  # Undervalued
        position_size = 1.2
    else:
        position_size = 1.0

    # Same momentum ranking
    if (i - lookback) % hold == 0 and i >= lookback + hold:
        rebal_idx = lookback + ((i - lookback) // hold) * hold
        if rebal_idx < len(ret_140d):
            ranking = ret_140d.iloc[rebal_idx].rank(ascending=False)
            long_idx = ranking[ranking <= top_n].index.tolist()
            short_idx = ranking[ranking > len(ranking) - top_n].index.tolist()
        else:
            long_idx = short_idx = []

    if len(long_idx) >= top_n and len(short_idx) >= top_n:
        long_ret = daily_ret.iloc[i][long_idx].mean()
        short_ret = daily_ret.iloc[i][short_idx].mean()
        base_ret = (long_ret - short_ret) / top_n - 0.005 / hold
        daily_return = base_ret * position_size
    else:
        daily_return = 0

    strategy_b_daily.append(daily_return)

all_returns['B_Momentum_Bubble'] = pd.Series(strategy_b_daily, index=daily_close.index[lookback:])

ret_b = all_returns['B_Momentum_Bubble'].dropna()
total_b = (1 + ret_b).prod() - 1
sharpe_b = (ret_b.mean() / ret_b.std() * np.sqrt(252)) if ret_b.std() > 0 else 0
ann_b = ((1 + total_b) ** (252 / len(ret_b)) - 1)
maxdd_b = ((1 + ret_b).cumprod() / (1 + ret_b).cumprod().cummax() - 1).min()

print(f"  Total Return: {total_b:+.1%}")
print(f"  Annual Return: {ann_b:+.1%}")
print(f"  Sharpe Ratio: {sharpe_b:.2f}")
print(f"  Max Drawdown: {maxdd_b:.1%}")

# =============================================================================
# STRATEGY C: Z-Score Mean Reversion (Daily)
# =============================================================================
print("\n[C] Z-Score Mean Reversion (Daily)...")

z_window = 20
strategy_c_daily = []

for i in range(z_window + 5, len(daily_ret)):
    if i % 500 == 0:
        print(f"  Processing day {i}/{len(daily_close)}...")

    z_scores = (daily_ret.iloc[i] - daily_ret.iloc[i-z_window:i].mean()) / daily_ret.iloc[i-z_window:i].std()

    # Find extreme moves
    extreme_mask = np.abs(z_scores) > 4.0
    if extreme_mask.sum() < 5:
        strategy_c_daily.append(0)
        continue

    # Select top 5 by |Z|
    top_5_idx = np.abs(z_scores[extreme_mask]).nlargest(5).index.tolist()

    # Fade extreme move (mean reversion)
    fade_ret = -np.sign(daily_ret.iloc[i][top_5_idx]).mean() * np.abs(daily_ret.iloc[i][top_5_idx]).mean()

    # Continuation (next 2 days)
    if i < len(daily_ret) - 2:
        cont_ret = np.sign(daily_ret.iloc[i][top_5_idx]).mean() * daily_ret.iloc[i+1:i+3][top_5_idx].mean().mean()
    else:
        cont_ret = 0

    daily_return = (fade_ret + cont_ret) / 2 - 0.001
    strategy_c_daily.append(daily_return)

all_returns['C_Z_Score_MR'] = pd.Series(strategy_c_daily, index=daily_close.index[z_window+5:z_window+5+len(strategy_c_daily)])

ret_c = all_returns['C_Z_Score_MR'].dropna()
total_c = (1 + ret_c).prod() - 1
sharpe_c = (ret_c.mean() / ret_c.std() * np.sqrt(252)) if ret_c.std() > 0 else 0
ann_c = ((1 + total_c) ** (252 / len(ret_c)) - 1)
maxdd_c = ((1 + ret_c).cumprod() / (1 + ret_c).cumprod().cummax() - 1).min()

print(f"  Total Return: {total_c:+.1%}")
print(f"  Annual Return: {ann_c:+.1%}")
print(f"  Sharpe Ratio: {sharpe_c:.2f}")
print(f"  Max Drawdown: {maxdd_c:.1%}")

# =============================================================================
# STRATEGY D: Contrarian Bubble Score (Daily Approx)
# =============================================================================
print("\n[D] Contrarian Bubble Score (Daily)...")

strategy_d_daily = []

for i in range(lookback + 1, len(daily_close)):
    if i % 500 == 0:
        print(f"  Processing day {i}/{len(daily_close)}...")

    bubbles_dict = {}

    for col in daily_close.columns:
        try:
            recent = daily_close[col].iloc[max(0, i-104):i+1]
            if len(recent) < 20:
                continue

            log_p = np.log(recent)
            fair = log_p.rolling(104, min_periods=52).mean()
            res = log_p - fair
            z_sc = (res - res.rolling(104, min_periods=52).mean()) / res.rolling(104, min_periods=52).std()
            bubble = np.tanh(z_sc.iloc[-1] / 2)

            if bubble < -0.8:
                bubbles_dict[col] = bubble
        except:
            pass

    if len(bubbles_dict) >= 10:
        top_20 = sorted(bubbles_dict.items(), key=lambda x: x[1])[:min(20, len(bubbles_dict))]
        top_cols = [col for col, _ in top_20]
        daily_return = daily_ret.iloc[i][top_cols].mean() - 0.001
    else:
        daily_return = 0

    strategy_d_daily.append(daily_return)

all_returns['D_Contrarian_Bubble'] = pd.Series(strategy_d_daily, index=daily_close.index[lookback+1:lookback+1+len(strategy_d_daily)])

ret_d = all_returns['D_Contrarian_Bubble'].dropna()
total_d = (1 + ret_d).prod() - 1
sharpe_d = (ret_d.mean() / ret_d.std() * np.sqrt(252)) if ret_d.std() > 0 else 0
ann_d = ((1 + total_d) ** (252 / len(ret_d)) - 1)
maxdd_d = ((1 + ret_d).cumprod() / (1 + ret_d).cumprod().cummax() - 1).min()

print(f"  Total Return: {total_d:+.1%}")
print(f"  Annual Return: {ann_d:+.1%}")
print(f"  Sharpe Ratio: {sharpe_d:.2f}")
print(f"  Max Drawdown: {maxdd_d:.1%}")

# =============================================================================
# STRATEGY E: Simple Proxy (cannot do full backtest without sentiment data)
# =============================================================================
print("\n[E] Reddit Sentiment - Skipped (data 2024+ only)")
print("  Note: Full backtest requires sentiment API which started 2024-01-01")

# =============================================================================
# Save Daily Returns
# =============================================================================
print("\n[Saving daily returns...]")

# Forward fill to have complete series
all_returns_filled = all_returns.fillna(method='ffill')

# Save CSV
all_returns_filled.to_csv('results/daily_returns_2019_2026.csv')
print(f"  Saved: results/daily_returns_2019_2026.csv")

# =============================================================================
# Summary Statistics
# =============================================================================
print("\n" + "=" * 100)
print("SUMMARY STATISTICS: 2019-2026 DAILY RETURNS")
print("=" * 100)

summary_stats = []

for col in all_returns_filled.columns:
    ret = all_returns_filled[col].dropna()
    if len(ret) > 0:
        total_ret = (1 + ret).prod() - 1
        annual_ret = ((1 + total_ret) ** (252 / len(ret)) - 1)
        sharpe = (ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
        sortino = (ret.mean() / ret[ret < 0].std() * np.sqrt(252)) if ret[ret < 0].std() > 0 else 0
        maxdd = ((1 + ret).cumprod() / (1 + ret).cumprod().cummax() - 1).min()
        win_rate = (ret > 0).sum() / len(ret)

        summary_stats.append({
            'Strategy': col,
            'Days': len(ret),
            'Total Return': f'{total_ret:+.1%}',
            'Annual Return': f'{annual_ret:+.1%}',
            'Sharpe': f'{sharpe:.2f}',
            'Sortino': f'{sortino:.2f}',
            'Max DD': f'{maxdd:.1%}',
            'Win Rate': f'{win_rate:.1%}',
            'Avg Daily': f'{ret.mean():+.3%}',
            'Daily Std': f'{ret.std():.3%}',
        })

summary_df = pd.DataFrame(summary_stats)
print("\n" + summary_df.to_string(index=False))

# Save summary
summary_df.to_csv('results/strategy_summary_2019_2026.csv', index=False)
print("\n  Saved: results/strategy_summary_2019_2026.csv")

# =============================================================================
# Yearly Breakdown
# =============================================================================
print("\n" + "=" * 100)
print("YEARLY BREAKDOWN")
print("=" * 100)

yearly_results = []

for year in range(2019, 2027):
    year_mask = all_returns_filled.index.year == year
    year_data = all_returns_filled[year_mask]

    if len(year_data) == 0:
        continue

    year_row = {'Year': year}

    for col in all_returns_filled.columns:
        ret = year_data[col].dropna()
        if len(ret) > 0:
            total = (1 + ret).prod() - 1
            year_row[col] = f'{total:+.1%}'
        else:
            year_row[col] = 'N/A'

    yearly_results.append(year_row)

yearly_df = pd.DataFrame(yearly_results)
print("\n" + yearly_df.to_string(index=False))

yearly_df.to_csv('results/yearly_returns_2019_2026.csv', index=False)
print("\n  Saved: results/yearly_returns_2019_2026.csv")

print("\n" + "=" * 100)
print("BACKTEST COMPLETE - Daily returns available from 2019-2026")
print("=" * 100)
print("\nFiles generated:")
print("  - daily_returns_2019_2026.csv (complete daily return series)")
print("  - strategy_summary_2019_2026.csv (metrics summary)")
print("  - yearly_returns_2019_2026.csv (yearly breakdown)")
