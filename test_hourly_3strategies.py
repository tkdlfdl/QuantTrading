"""
HOURLY DATA STRATEGIES (2019-2026)
==================================
Test 3 strategies on hourly merged data (7.41 years):
1. QQQ Bubble + Buy Momentum
2. Hourly Momentum (20h lookback, 5h hold)
3. Daily Mean Reversion (DailyMR)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os

print("="*120)
print("HOURLY STRATEGIES TEST (2019-2026)")
print("="*120)

# Load data
print("\nLoading hourly data...")
close_data = pd.read_parquet("data/cache/merged_hourly_close.parquet")
ret_hourly = close_data.pct_change().fillna(0)

print(f"Data shape: {close_data.shape}")
print(f"Period: {close_data.index[0]} to {close_data.index[-1]}")
years = (close_data.index[-1] - close_data.index[0]).days / 365.25
print(f"Years: {years:.2f}")

trading_hours_per_year = 252 * 6.5  # 6.5 hours/day, 252 days/year

# ============================================================================
# STRATEGY 1: QQQ BUBBLE + BUY MOMENTUM
# ============================================================================

print("\n" + "="*120)
print("STRATEGY 1: QQQ BUBBLE + BUY MOMENTUM")
print("="*120)

# Calculate market bubble (from average of top stocks)
# Use market-wide momentum as proxy
market_proxy = close_data.iloc[:, :100].mean(axis=1)  # Average of first 100 stocks

def calc_bubble(price, ma_w=24, z_w=48):
    log_p = np.log(price + 1)
    fair = log_p.rolling(ma_w).mean()
    res = log_p - fair
    z = (res - res.rolling(z_w).mean()) / res.rolling(z_w).std()
    return np.tanh(z / 2)

bubble = calc_bubble(market_proxy, ma_w=24, z_w=48)
print(f"Using market proxy bubble (avg of first 100 stocks)")

# Get momentum of all stocks
ret_mom_20h = close_data.pct_change(20).fillna(0)

strategy1_returns = []

for i in range(50, len(close_data) - 5):
    bs = bubble.iloc[i]

    if pd.isna(bs):
        continue

    # QQQ Bubble logic
    if bs < -0.7:  # Undervalued - buy momentum stocks
        # Rank all stocks by 20h momentum
        mom_ranking = ret_mom_20h.iloc[i].rank(ascending=False)
        top_5 = mom_ranking.nsmallest(5).index  # Top 5 by momentum

        # Buy top 5 for next 5 hours
        future_rets = ret_hourly.iloc[i+1:i+6, :][top_5]
        avg_ret = future_rets.mean().mean()
        strat_ret = avg_ret - 0.001

    elif bs > 0.8:  # Overvalued - short momentum stocks
        # Rank all stocks by 20h momentum
        mom_ranking = ret_mom_20h.iloc[i].rank(ascending=False)
        top_5 = mom_ranking.nsmallest(5).index  # Top 5 by momentum

        # Short top 5 for next 5 hours
        future_rets = ret_hourly.iloc[i+1:i+6, :][top_5]
        avg_ret = future_rets.mean().mean()
        strat_ret = -avg_ret - 0.001  # Short position

    else:
        strat_ret = 0  # Neutral

    strategy1_returns.append(strat_ret)

s1_series = pd.Series(strategy1_returns, index=close_data.index[50:50+len(strategy1_returns)])
s1_wealth = (1 + s1_series).cumprod()

s1_total = s1_wealth.iloc[-1] - 1
s1_annual = (1 + s1_total) ** (1/years) - 1
s1_sharpe = s1_series.mean() / s1_series.std() * np.sqrt(trading_hours_per_year) if s1_series.std() > 0 else 0
s1_dd = (s1_wealth / s1_wealth.cummax() - 1).min()
s1_win = (s1_series > 0).sum() / len(s1_series) * 100

print(f"\nQQQ Bubble + Buy Momentum Results:")
print(f"  Total Return: {s1_total:.2%}")
print(f"  Annual Return: {s1_annual:.2%}")
print(f"  Sharpe Ratio: {s1_sharpe:.4f}")
print(f"  Max Drawdown: {s1_dd:.2%}")
print(f"  Win Rate: {s1_win:.1f}%")
print(f"  Data Points: {len(s1_series)}")

# ============================================================================
# STRATEGY 2: HOURLY MOMENTUM
# ============================================================================

print("\n" + "="*120)
print("STRATEGY 2: HOURLY MOMENTUM (20h lookback, 5h hold)")
print("="*120)

lookback = 20
hold = 5

strategy2_returns = []

for i in range(lookback + 1, len(close_data) - hold, hold):
    # Rank by 20h momentum
    mom = ret_mom_20h.iloc[i]
    ranking = mom.rank(ascending=False)

    # Top 5 and bottom 5
    top_5 = ranking.nsmallest(5).index
    bot_5 = ranking.nlargest(5).index

    # Calculate 5h returns
    for j in range(i + 1, min(i + 1 + hold, len(close_data))):
        ret_row = ret_hourly.iloc[j]

        long_ret = ret_row[top_5].mean()
        short_ret = ret_row[bot_5].mean()

        strat_ret = (long_ret - short_ret) / 10 - 0.001 / hold
        strategy2_returns.append(strat_ret)

s2_series = pd.Series(strategy2_returns, index=close_data.index[lookback+2:lookback+2+len(strategy2_returns)])
s2_wealth = (1 + s2_series).cumprod()

s2_total = s2_wealth.iloc[-1] - 1
s2_annual = (1 + s2_total) ** (1/years) - 1
s2_sharpe = s2_series.mean() / s2_series.std() * np.sqrt(trading_hours_per_year) if s2_series.std() > 0 else 0
s2_dd = (s2_wealth / s2_wealth.cummax() - 1).min()
s2_win = (s2_series > 0).sum() / len(s2_series) * 100

print(f"\nHourly Momentum Results:")
print(f"  Total Return: {s2_total:.2%}")
print(f"  Annual Return: {s2_annual:.2%}")
print(f"  Sharpe Ratio: {s2_sharpe:.4f}")
print(f"  Max Drawdown: {s2_dd:.2%}")
print(f"  Win Rate: {s2_win:.1f}%")
print(f"  Data Points: {len(s2_series)}")

# ============================================================================
# STRATEGY 3: DAILY MEAN REVERSION (DailyMR)
# ============================================================================

print("\n" + "="*120)
print("STRATEGY 3: DAILY MEAN REVERSION (DailyMR)")
print("="*120)

strategy3_returns = []

for i in range(1, len(close_data) - 1):
    # Current hour returns
    ret_current = ret_hourly.iloc[i]

    # Mean reversion: position opposite to momentum
    # If stock up a lot, short it; if down a lot, long it
    signal = -ret_current  # Opposite direction

    # Normalize positions
    signal = (signal - signal.mean()) / signal.std()
    signal = signal.clip(-0.1, 0.1)  # Limit exposure

    # Next hour return
    ret_next = ret_hourly.iloc[i + 1]

    # PnL
    pnl = (signal * ret_next).sum()
    strat_ret = pnl - 0.0005  # Tiny cost

    strategy3_returns.append(strat_ret)

s3_series = pd.Series(strategy3_returns, index=close_data.index[2:2+len(strategy3_returns)])
s3_wealth = (1 + s3_series).cumprod()

s3_total = s3_wealth.iloc[-1] - 1
s3_annual = (1 + s3_total) ** (1/years) - 1
s3_sharpe = s3_series.mean() / s3_series.std() * np.sqrt(trading_hours_per_year) if s3_series.std() > 0 else 0
s3_dd = (s3_wealth / s3_wealth.cummax() - 1).min()
s3_win = (s3_series > 0).sum() / len(s3_series) * 100

print(f"\nDaily Mean Reversion Results:")
print(f"  Total Return: {s3_total:.2%}")
print(f"  Annual Return: {s3_annual:.2%}")
print(f"  Sharpe Ratio: {s3_sharpe:.4f}")
print(f"  Max Drawdown: {s3_dd:.2%}")
print(f"  Win Rate: {s3_win:.1f}%")
print(f"  Data Points: {len(s3_series)}")

# ============================================================================
# SUMMARY TABLE
# ============================================================================

print("\n" + "="*120)
print("STRATEGY COMPARISON - HOURLY DATA (2019-2026)")
print("="*120)

results = {
    'Strategy': ['QQQ Bubble + Buy Mom', 'Hourly Momentum', 'Daily MR'],
    'Total Return': [f'{s1_total:.2%}', f'{s2_total:.2%}', f'{s3_total:.2%}'],
    'Annual Return': [f'{s1_annual:.2%}', f'{s2_annual:.2%}', f'{s3_annual:.2%}'],
    'Sharpe Ratio': [f'{s1_sharpe:.4f}', f'{s2_sharpe:.4f}', f'{s3_sharpe:.4f}'],
    'Max Drawdown': [f'{s1_dd:.2%}', f'{s2_dd:.2%}', f'{s3_dd:.2%}'],
    'Win Rate': [f'{s1_win:.1f}%', f'{s2_win:.1f}%', f'{s3_win:.1f}%'],
}

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Wealth curves
axes[0, 0].plot(s1_wealth.index, s1_wealth.values, label='QQQ Bubble + Buy Mom', linewidth=2, alpha=0.8)
axes[0, 0].plot(s2_wealth.index, s2_wealth.values, label='Hourly Momentum', linewidth=2, alpha=0.8)
axes[0, 0].plot(s3_wealth.index, s3_wealth.values, label='Daily MR', linewidth=2, alpha=0.8)
axes[0, 0].set_title('Wealth Growth - All Strategies', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Wealth Multiple')
axes[0, 0].set_yscale('log')
axes[0, 0].legend(loc='best', fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# Annual returns
names = ['QQQ Bubble\n+ Buy Mom', 'Hourly\nMomentum', 'Daily MR']
annual_rets = [s1_annual * 100, s2_annual * 100, s3_annual * 100]
colors = ['green' if r > 0 else 'red' for r in annual_rets]
axes[0, 1].bar(names, annual_rets, color=colors, alpha=0.7)
axes[0, 1].set_title('Annual Returns', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Annual Return (%)')
axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Sharpe ratios
sharpes = [s1_sharpe, s2_sharpe, s3_sharpe]
axes[1, 0].bar(names, sharpes, color='steelblue', alpha=0.7)
axes[1, 0].set_title('Sharpe Ratios', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Sharpe Ratio')
axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Drawdowns
dds = [s1_dd * 100, s2_dd * 100, s3_dd * 100]
axes[1, 1].bar(names, dds, color='darkred', alpha=0.7)
axes[1, 1].set_title('Max Drawdowns', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Max Drawdown (%)')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/hourly_3strategies_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: results/hourly_3strategies_comparison.png")

# Save CSV
results_df.to_csv('results/hourly_3strategies_results.csv', index=False)
print("Saved: results/hourly_3strategies_results.csv")

print("\n" + "="*120)
print("ANALYSIS COMPLETE")
print("="*120)

# Ranking
print("\nBest by Metric:")
print(f"  Highest Annual Return: QQQ Bubble + Buy Mom ({s1_annual:.2%})" if s1_annual > max(s2_annual, s3_annual) else f"  Highest Annual Return: Other")
print(f"  Best Sharpe Ratio: QQQ Bubble + Buy Mom ({s1_sharpe:.4f})" if s1_sharpe > max(s2_sharpe, s3_sharpe) else f"  Best Sharpe: Other")
print(f"  Smallest Drawdown: QQQ Bubble + Buy Mom ({s1_dd:.2%})" if abs(s1_dd) < min(abs(s2_dd), abs(s3_dd)) else f"  Smallest DD: Other")
