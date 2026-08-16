"""
STRATEGY D: Contrarian Bubble Score
Backtest Period: 2019-2026 (7.4 years)
Based on: CONTRARIAN_BUBBLE_STRATEGY.md (Exact Parameters)

Parameters:
- Bubble MA Window: 104 hours (~21 trading days)
- Threshold: -0.8 (extreme undervaluation)
- Hold Period: 13 hours (~2-3 trading days)
- Top-N Stocks: 20 (equal-weight)
- Transaction Cost: 0.1% per entry
- Universe: 515 tickers (S&P500 + NASDAQ100)
"""

import numpy as np
import pandas as pd
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

print("=" * 120)
print("STRATEGY D: CONTRARIAN BUBBLE SCORE (2019-2026)")
print("=" * 120)
print("\nParameters from CONTRARIAN_BUBBLE_STRATEGY.md:")
print("  Bubble MA Window: 104h (~21 trading days)")
print("  Threshold: -0.8")
print("  Hold Period: 13h (~2 trading days)")
print("  Top-N Stocks: 20")
print("  Transaction Cost: 0.1% per entry")
print("  Universe: 515 stocks (S&P500 + NASDAQ100)")

# Load daily data
print("\n[Loading Data]")
daily_close = pd.read_parquet('data/cache/daily_close_extended_1997_2026.parquet')
daily_close = daily_close.loc['2019-01-01':'2026-06-12'].copy()
daily_close = daily_close.dropna(axis=1, thresh=len(daily_close) * 0.85)

print(f"  Period: {daily_close.index[0].date()} to {daily_close.index[-1].date()}")
print(f"  Stocks: {len(daily_close.columns)}")

daily_ret = daily_close.pct_change().fillna(0)

# PARAMETER CONVERSION
# 104 hours ≈ 16 trading sessions ≈ 21 calendar days
ma_window = 21  # days
hold_period = 2  # days (13 hours ≈ 1-2 trading days)
top_n = 20
threshold = -0.8
transaction_cost = 0.001  # 0.1% per entry
z_score_window = 21  # same as MA window for z-score normalization

print(f"\n[Converting to Daily]")
print(f"  MA Window: {ma_window} days")
print(f"  Hold Period: {hold_period} days")
print(f"  Z-Score Window: {z_score_window} days")

# GENERATE DAILY RETURNS
print(f"\n[Generating Daily Returns]")

strategy_d_daily = []
trade_entry_dates = []

for i in range(ma_window + z_score_window, len(daily_close)):
    bubbles_dict = {}

    # Calculate bubble score for each stock
    for col in daily_close.columns:
        try:
            # Get recent price window
            recent_prices = daily_close[col].iloc[max(0, i-ma_window):i+1]

            if len(recent_prices) < ma_window:
                continue

            # Bubble Score Formula (from .md)
            # score = tanh((log(price) - log(MA)) / z_score / 2)

            log_p = np.log(recent_prices)
            ma_val = log_p.rolling(ma_window, min_periods=ma_window//2).mean()
            residual = log_p - ma_val

            z_val = (residual - residual.rolling(z_score_window, min_periods=z_score_window//2).mean()) / \
                    residual.rolling(z_score_window, min_periods=z_score_window//2).std()

            bubble = np.tanh(z_val.iloc[-1] / 2)

            # Signal: bubble < -0.8
            if bubble < threshold:
                bubbles_dict[col] = bubble
        except:
            pass

    # Select top-20 with lowest (most extreme) bubble scores
    if len(bubbles_dict) >= top_n:
        # Sort by bubble score (most negative = most depressed)
        sorted_bubbles = sorted(bubbles_dict.items(), key=lambda x: x[1])
        top_20_cols = [col for col, score in sorted_bubbles[:top_n]]

        # Calculate equal-weight return
        daily_return = daily_ret.iloc[i][top_20_cols].mean() - transaction_cost

        strategy_d_daily.append(daily_return)
        trade_entry_dates.append(daily_close.index[i])
    else:
        strategy_d_daily.append(0)
        trade_entry_dates.append(daily_close.index[i])

# Create daily returns series
daily_returns = pd.Series(strategy_d_daily, index=daily_close.index[ma_window+z_score_window:], dtype=float)

print(f"  Generated: {len(daily_returns):,} daily returns")
print(f"  Active trading days (bubble < -0.8): {(np.array(strategy_d_daily) != 0).sum():,}")

# CALCULATE OVERALL METRICS
print(f"\n[Overall Metrics (2019-2026)]")

wealth = (1 + daily_returns).cumprod()
total_ret = (wealth.iloc[-1] / wealth.iloc[0]) - 1
years = len(daily_returns) / 252
annual_ret = ((wealth.iloc[-1] / wealth.iloc[0]) ** (1 / years)) - 1
sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
sortino = (daily_returns.mean() / daily_returns[daily_returns < 0].std() * np.sqrt(252)) if daily_returns[daily_returns < 0].std() > 0 else 0
max_dd = ((wealth / wealth.cummax()) - 1).min()
win_rate = (daily_returns > 0).sum() / len(daily_returns)

print(f"  Total Return: {total_ret:+.2%}")
print(f"  Annual Return: {annual_ret:+.2%}")
print(f"  Sharpe Ratio: {sharpe:.2f}")
print(f"  Sortino Ratio: {sortino:.2f}")
print(f"  Max Drawdown: {max_dd:.2%}")
print(f"  Win Rate: {win_rate:.1%}")
print(f"  Trading Days: {len(daily_returns):,}")

# YEARLY BREAKDOWN
print(f"\n[Yearly Performance (2019-2026)]")
print(f"{'Year':<8} {'Days':<8} {'Return':<12} {'Sharpe':<10} {'Sortino':<10} {'MaxDD':<10} {'Win%':<8}")
print("-" * 80)

yearly_results = []

for year in range(2019, 2027):
    year_mask = daily_returns.index.year == year
    year_ret = daily_returns[year_mask]

    if len(year_ret) == 0:
        continue

    year_wealth = (1 + year_ret).cumprod()
    year_total = year_wealth.iloc[-1] / year_wealth.iloc[0] - 1
    year_sharpe = (year_ret.mean() / year_ret.std() * np.sqrt(252)) if year_ret.std() > 0 else 0
    year_sortino = (year_ret.mean() / year_ret[year_ret < 0].std() * np.sqrt(252)) if year_ret[year_ret < 0].std() > 0 else 0
    year_maxdd = ((year_wealth / year_wealth.cummax()) - 1).min()
    year_win = (year_ret > 0).sum() / len(year_ret) * 100

    print(f"{year:<8} {len(year_ret):<8} {year_total:+9.2%}    {year_sharpe:6.2f}     {year_sortino:6.2f}     {year_maxdd:7.2%}    {year_win:5.1f}%")

    yearly_results.append({
        'Year': year,
        'Days': len(year_ret),
        'Return_%': year_total * 100,
        'Sharpe': year_sharpe,
        'Sortino': year_sortino,
        'Max_DD_%': year_maxdd * 100,
        'Win_Rate_%': year_win,
    })

# SAVE RESULTS
print(f"\n[Saving Results]")

# Daily returns CSV
daily_returns.to_csv('results/Strategy_D_Daily_Returns_2019_2026.csv', header=['Daily_Return'])
print(f"  [OK] Strategy_D_Daily_Returns_2019_2026.csv")

# Yearly metrics CSV
yearly_df = pd.DataFrame(yearly_results)
yearly_df.to_csv('results/Strategy_D_Yearly_Metrics_2019_2026.csv', index=False)
print(f"  [OK] Strategy_D_Yearly_Metrics_2019_2026.csv")

# Summary CSV
summary = pd.DataFrame({
    'Metric': ['Total_Return_%', 'Annual_Return_%', 'Sharpe', 'Sortino', 'Max_DD_%', 'Win_Rate_%', 'Trading_Days', 'Years'],
    'Value': [f"{total_ret*100:+.2f}", f"{annual_ret*100:+.2f}", f"{sharpe:.2f}", f"{sortino:.2f}", f"{max_dd*100:.2f}", f"{win_rate*100:.1f}", f"{len(daily_returns)}", f"{years:.1f}"]
})
summary.to_csv('results/Strategy_D_Summary_2019_2026.csv', index=False)
print(f"  [OK] Strategy_D_Summary_2019_2026.csv")

# Wealth curve CSV
wealth_df = pd.DataFrame({'Date': daily_returns.index, 'Wealth': wealth.values})
wealth_df.to_csv('results/Strategy_D_Wealth_2019_2026.csv', index=False)
print(f"  [OK] Strategy_D_Wealth_2019_2026.csv")

print(f"\n{'='*120}")
print(f"STRATEGY D BACKTEST COMPLETE")
print(f"{'='*120}")

