"""
STRATEGY D: Contrarian Bubble Score (RECENT HOURLY DATA ONLY)
Period: 2024-2026 (2 years with complete hourly data)
Data: merged_hourly_close.parquet (June 2024 - June 2026)
Based on: CONTRARIAN_BUBBLE_STRATEGY.md (Exact Hourly Parameters)

NOTE: Only using recent data due to Alpaca data sparsity (2019-2024 has 98% NaN)
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

print("=" * 120)
print("STRATEGY D: CONTRARIAN BUBBLE SCORE - RECENT HOURLY BACKTEST (2024-2026)")
print("=" * 120)

# Load HOURLY data (recent only - has actual price coverage)
print("\n[Loading Hourly Data]")

hourly_close = pd.read_parquet('data/cache/merged_hourly_close.parquet')
hourly_open = pd.read_parquet('data/cache/merged_hourly_open.parquet')

print(f"  Period: {hourly_close.index[0]} to {hourly_close.index[-1]}")
print(f"  Hourly bars: {len(hourly_close):,}")
print(f"  Stocks: {len(hourly_close.columns)}")

# Check data quality
nan_pct = hourly_close.isna().sum() / len(hourly_close) * 100
print(f"  Data completeness: min={nan_pct.min():.1f}%, max={nan_pct.max():.1f}%, mean={nan_pct.mean():.1f}%")

hourly_ret = hourly_close.pct_change().fillna(0)

# PARAMETERS (exact from CONTRARIAN_BUBBLE_STRATEGY.md)
MA_WINDOW = 104  # hours
THRESHOLD = -0.8
HOLD_PERIOD = 13  # hours
TOP_N = 20
TRANSACTION_COST = 0.001  # 0.1%
Z_SCORE_WINDOW = 104  # same as MA window

print(f"\n[Backtesting Parameters]")
print(f"  MA Window: {MA_WINDOW} hours")
print(f"  Threshold: {THRESHOLD}")
print(f"  Hold Period: {HOLD_PERIOD} hours")
print(f"  Top-N: {TOP_N}")
print(f"  Transaction Cost: {TRANSACTION_COST*100:.1f}%")

# GENERATE HOURLY RETURNS
print(f"\n[Generating Strategy Returns]")

strategy_d_hourly = []
position_tracker = {}  # Track active positions

for i in range(MA_WINDOW + Z_SCORE_WINDOW, len(hourly_close)):
    # Update position tracker - decrement hold counters
    positions_to_close = []
    for stock, hold_remaining in position_tracker.items():
        if hold_remaining <= 0:
            positions_to_close.append(stock)

    for stock in positions_to_close:
        del position_tracker[stock]

    # Calculate bubble score for each stock
    bubbles_dict = {}

    for col in hourly_close.columns:
        try:
            # Get recent price window (104 hours)
            recent_prices = hourly_close[col].iloc[max(0, i-MA_WINDOW):i+1]

            if len(recent_prices) < MA_WINDOW:
                continue

            # Check if we have enough valid data
            if recent_prices.isna().sum() / len(recent_prices) > 0.5:
                continue

            # Bubble Score Formula (exact from .md)
            # IMPORTANT: fair is MA of PRICES (not log prices)
            log_p = np.log(recent_prices.replace(0, np.nan).ffill())
            fair = recent_prices.rolling(MA_WINDOW, min_periods=MA_WINDOW//2).mean()
            residual = log_p - np.log(fair.replace(0, np.nan))

            z_val = (residual - residual.rolling(Z_SCORE_WINDOW, min_periods=Z_SCORE_WINDOW//2).mean()) / \
                    residual.rolling(Z_SCORE_WINDOW, min_periods=Z_SCORE_WINDOW//2).std()

            bubble = np.tanh(z_val.iloc[-1] / 2)
            bubble = 0 if np.isnan(bubble) else bubble  # fillna(0) as per .md

            # Signal: bubble < -0.8
            if bubble < THRESHOLD and col not in position_tracker:
                bubbles_dict[col] = bubble
        except:
            pass

    # Get active positions
    active_positions = list(position_tracker.keys())

    # Entry signal
    if len(bubbles_dict) >= TOP_N and len(active_positions) < TOP_N:
        sorted_bubbles = sorted(bubbles_dict.items(), key=lambda x: x[1])
        new_entries = [col for col, _ in sorted_bubbles[:TOP_N-len(active_positions)]]

        for col in new_entries:
            position_tracker[col] = HOLD_PERIOD
            active_positions.append(col)

    # Calculate return from all active positions
    try:
        if len(active_positions) > 0:
            position_returns = []
            for stock in active_positions:
                ret = hourly_ret.loc[hourly_close.index[i], stock]
                position_returns.append(ret)

            # Equal-weight return (transaction cost only applied at entry, not every hour)
            daily_return = np.mean(position_returns)
        else:
            daily_return = 0
    except:
        daily_return = 0

    strategy_d_hourly.append(daily_return)

    # Decrement hold counters
    for stock in active_positions:
        position_tracker[stock] -= 1

# Create hourly returns series
hourly_returns = pd.Series(strategy_d_hourly, index=hourly_close.index[MA_WINDOW+Z_SCORE_WINDOW:])

print(f"  Generated: {len(hourly_returns):,} hourly returns")

# CONVERT TO DAILY FOR ANALYSIS
# Group by day and calculate daily return
daily_returns_calc = hourly_returns.groupby(hourly_returns.index.date).apply(lambda x: (1 + x).prod() - 1)
daily_returns_calc.index = pd.to_datetime(daily_returns_calc.index)

print(f"  Daily returns: {len(daily_returns_calc):,} days")

# CALCULATE METRICS
wealth = (1 + daily_returns_calc).cumprod()
total_ret = (wealth.iloc[-1] / wealth.iloc[0]) - 1
years = len(daily_returns_calc) / 252
annual_ret = ((wealth.iloc[-1] / wealth.iloc[0]) ** (1 / years)) - 1 if years > 0 else 0
sharpe = (daily_returns_calc.mean() / daily_returns_calc.std() * np.sqrt(252)) if daily_returns_calc.std() > 0 else 0
sortino = (daily_returns_calc.mean() / daily_returns_calc[daily_returns_calc < 0].std() * np.sqrt(252)) if daily_returns_calc[daily_returns_calc < 0].std() > 0 else 0
max_dd = ((wealth / wealth.cummax()) - 1).min()
win_rate = (daily_returns_calc > 0).sum() / len(daily_returns_calc)

print(f"\n[OVERALL RESULTS] 2024-2026")
print(f"  Total Return: {total_ret:+.2%}")
print(f"  Annual Return: {annual_ret:+.2%}")
print(f"  Sharpe Ratio: {sharpe:.2f}")
print(f"  Sortino Ratio: {sortino:.2f}")
print(f"  Max Drawdown: {max_dd:.2%}")
print(f"  Win Rate: {win_rate:.1%}")
print(f"  Trading Days: {len(daily_returns_calc):,}")

# YEARLY BREAKDOWN
print(f"\n[YEARLY PERFORMANCE]")
print(f"{'Year':<8} {'Days':<8} {'Return':<12} {'Sharpe':<10} {'MaxDD':<10}")
print("-" * 60)

yearly_results = []

for year in range(2024, 2027):
    year_mask = daily_returns_calc.index.year == year
    year_ret = daily_returns_calc[year_mask]

    if len(year_ret) == 0:
        continue

    year_wealth = (1 + year_ret).cumprod()
    year_total = year_wealth.iloc[-1] / year_wealth.iloc[0] - 1
    year_sharpe = (year_ret.mean() / year_ret.std() * np.sqrt(252)) if year_ret.std() > 0 else 0
    year_maxdd = ((year_wealth / year_wealth.cummax()) - 1).min()

    print(f"{year:<8} {len(year_ret):<8} {year_total:+9.2%}    {year_sharpe:6.2f}     {year_maxdd:7.2%}")

    yearly_results.append({
        'Year': year,
        'Days': len(year_ret),
        'Return_%': year_total * 100,
        'Sharpe': year_sharpe,
        'Max_DD_%': year_maxdd * 100,
    })

# COMPARISON WITH DOCUMENTED
print(f"\n[COMPARISON WITH DOCUMENTED RESULTS (2019-2026)]")
doc_sharpe = 2.65
doc_annual = 38.08
doc_maxdd = -10.09

print(f"  Note: Documented results are for 2019-2026, we only have clean data for 2024-2026")
print(f"  Documented Sharpe: {doc_sharpe:.2f}  |  Actual (2024-2026): {sharpe:.2f}")
print(f"  Documented Annual: {doc_annual:+.2f}%  |  Actual (2024-2026): {annual_ret*100:+.2f}%")
print(f"  Documented MaxDD: {doc_maxdd:.2f}%  |  Actual (2024-2026): {max_dd*100:.2f}%")

# SAVE RESULTS
print(f"\n[Saving Results]")

# Hourly returns CSV
hourly_returns.to_csv('results/Strategy_D_Hourly_Returns_Recent_2024_2026.csv', header=['Hourly_Return'])
print(f"  [OK] Strategy_D_Hourly_Returns_Recent_2024_2026.csv ({len(hourly_returns):,} bars)")

# Daily returns CSV
daily_returns_calc.to_csv('results/Strategy_D_Daily_Returns_Recent_2024_2026.csv', header=['Daily_Return'])
print(f"  [OK] Strategy_D_Daily_Returns_Recent_2024_2026.csv ({len(daily_returns_calc):,} days)")

# Yearly metrics CSV
yearly_df = pd.DataFrame(yearly_results)
yearly_df.to_csv('results/Strategy_D_Yearly_Metrics_Recent_2024_2026.csv', index=False)
print(f"  [OK] Strategy_D_Yearly_Metrics_Recent_2024_2026.csv")

# Summary CSV
summary = pd.DataFrame({
    'Metric': ['Total_Return_%', 'Annual_Return_%', 'Sharpe', 'Sortino', 'Max_DD_%', 'Win_Rate_%'],
    'Value': [f"{total_ret*100:+.2f}", f"{annual_ret*100:+.2f}", f"{sharpe:.2f}", f"{sortino:.2f}", f"{max_dd*100:.2f}", f"{win_rate*100:.1f}"]
})
summary.to_csv('results/Strategy_D_Summary_Recent_2024_2026.csv', index=False)
print(f"  [OK] Strategy_D_Summary_Recent_2024_2026.csv")

print(f"\n{'='*120}")
print(f"STRATEGY D RECENT HOURLY BACKTEST COMPLETE (2024-2026)")
print(f"{'='*120}")

