"""
HOURLY STRATEGIES - SIMPLIFIED (2019-2026)
===========================================
Test 3 strategies on hourly data:
1. Market Bubble + Buy Momentum
2. Hourly Momentum
3. Daily Mean Reversion
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os

print("="*120)
print("HOURLY STRATEGIES TEST (2019-2026) - SIMPLIFIED")
print("="*120)

# Load data
print("\nLoading hourly data...")
close_data = pd.read_parquet("data/cache/merged_hourly_close.parquet")
ret_hourly = close_data.pct_change().fillna(0)

print(f"Data shape: {close_data.shape}")
print(f"Period: {close_data.index[0]} to {close_data.index[-1]}")
years = (close_data.index[-1] - close_data.index[0]).days / 365.25
print(f"Years: {years:.2f}\n")

# Helper function
def calc_bubble(price, ma_w=24, z_w=48):
    """Calculate bubble score bounded [-1, 1]"""
    log_p = np.log(np.maximum(price, 0.0001))  # Avoid log(0)
    fair = log_p.rolling(ma_w, min_periods=1).mean()
    res = log_p - fair
    std = res.rolling(z_w, min_periods=1).std()
    std = std.replace(0, 1)  # Avoid division by zero
    z = (res - res.rolling(z_w, min_periods=1).mean()) / std
    return np.tanh(z / 2)

# Calculate market bubble
market_avg = close_data.mean(axis=1)
bubble = calc_bubble(market_avg, ma_w=24, z_w=48).fillna(0)

# Calculate momentum
ret_mom = close_data.pct_change(20).fillna(0)

print("="*120)
print("STRATEGY 1: MARKET BUBBLE + BUY MOMENTUM")
print("="*120)

s1_ret = []
for i in range(50, len(bubble) - 5):
    bs = bubble.iloc[i]

    if abs(bs) < 0.5:  # Neutral
        s1_ret.append(0)
        continue

    # Get top 5 momentum stocks
    mom = ret_mom.iloc[i]
    top5_idx = mom.nlargest(5).index

    # Get future 5h average return
    future = ret_hourly.iloc[i+1:i+6, :][top5_idx].values.mean()

    if bs < -0.7:  # Undervalued - BUY
        ret = future - 0.001
    elif bs > 0.7:  # Overvalued - SHORT
        ret = -future - 0.001
    else:
        ret = 0

    s1_ret.append(ret)

s1_series = pd.Series(s1_ret, index=bubble.index[50:50+len(s1_ret)])
s1_wealth = (1 + s1_series).cumprod()

s1_total = s1_wealth.iloc[-1] - 1
s1_annual = (1 + s1_total) ** (1/years) - 1 if s1_total > -1 else -1
s1_sharpe = s1_series.mean() / s1_series.std() * np.sqrt(252*6.5) if s1_series.std() > 0 else 0
s1_dd = (s1_wealth / s1_wealth.cummax() - 1).min()
s1_win = (s1_series > 0).sum() / len(s1_series) * 100

print(f"Total Return: {s1_total:.2%}")
print(f"Annual Return: {s1_annual:.2%}")
print(f"Sharpe Ratio: {s1_sharpe:.4f}")
print(f"Max Drawdown: {s1_dd:.2%}")
print(f"Win Rate: {s1_win:.1f}%\n")

# ============================================================================
# STRATEGY 2: HOURLY MOMENTUM (20h lookback, 5h hold)
# ============================================================================

print("="*120)
print("STRATEGY 2: HOURLY MOMENTUM (20h lookback, 5h hold)")
print("="*120)

s2_ret = []

for i in range(20, len(ret_mom) - 5, 5):
    # Rank by momentum
    mom = ret_mom.iloc[i]

    # Top 5 and bottom 5
    top5_idx = mom.nlargest(5).index
    bot5_idx = mom.nsmallest(5).index

    # 5-hour forward return
    for j in range(i+1, min(i+6, len(ret_hourly))):
        ret_now = ret_hourly.iloc[j]

        long_ret = ret_now[top5_idx].mean()
        short_ret = ret_now[bot5_idx].mean()

        strat_ret = (long_ret - short_ret) / 10 - 0.001 / 5
        s2_ret.append(strat_ret)

s2_series = pd.Series(s2_ret, index=ret_hourly.index[21:21+len(s2_ret)])
s2_wealth = (1 + s2_series).cumprod()

s2_total = s2_wealth.iloc[-1] - 1
s2_annual = (1 + s2_total) ** (1/years) - 1 if s2_total > -1 else -1
s2_sharpe = s2_series.mean() / s2_series.std() * np.sqrt(252*6.5) if s2_series.std() > 0 else 0
s2_dd = (s2_wealth / s2_wealth.cummax() - 1).min()
s2_win = (s2_series > 0).sum() / len(s2_series) * 100

print(f"Total Return: {s2_total:.2%}")
print(f"Annual Return: {s2_annual:.2%}")
print(f"Sharpe Ratio: {s2_sharpe:.4f}")
print(f"Max Drawdown: {s2_dd:.2%}")
print(f"Win Rate: {s2_win:.1f}%\n")

# ============================================================================
# STRATEGY 3: DAILY MEAN REVERSION
# ============================================================================

print("="*120)
print("STRATEGY 3: DAILY MEAN REVERSION (DailyMR)")
print("="*120)

s3_ret = []

for i in range(1, len(ret_hourly) - 1):
    # Current momentum
    ret_current = ret_hourly.iloc[i]

    # Mean reversion signal: opposite of momentum
    signal = -ret_current / (ret_current.std() + 1e-6)
    signal = signal.clip(-0.05, 0.05)  # Limit leverage

    # Next hour return
    ret_next = ret_hourly.iloc[i+1]

    # PnL
    pnl = (signal * ret_next).mean()
    strat_ret = pnl - 0.0002

    s3_ret.append(strat_ret)

s3_series = pd.Series(s3_ret, index=ret_hourly.index[2:2+len(s3_ret)])
s3_wealth = (1 + s3_series).cumprod()

s3_total = s3_wealth.iloc[-1] - 1
s3_annual = (1 + s3_total) ** (1/years) - 1 if s3_total > -1 else -1
s3_sharpe = s3_series.mean() / s3_series.std() * np.sqrt(252*6.5) if s3_series.std() > 0 else 0
s3_dd = (s3_wealth / s3_wealth.cummax() - 1).min()
s3_win = (s3_series > 0).sum() / len(s3_series) * 100

print(f"Total Return: {s3_total:.2%}")
print(f"Annual Return: {s3_annual:.2%}")
print(f"Sharpe Ratio: {s3_sharpe:.4f}")
print(f"Max Drawdown: {s3_dd:.2%}")
print(f"Win Rate: {s3_win:.1f}%\n")

# ============================================================================
# SUMMARY
# ============================================================================

print("="*120)
print("HOURLY STRATEGIES SUMMARY (2019-2026)")
print("="*120)

summary_df = pd.DataFrame({
    'Strategy': ['Market Bubble + Buy Mom', 'Hourly Momentum', 'Daily MR'],
    'Total Return': [f'{s1_total:.2%}', f'{s2_total:.2%}', f'{s3_total:.2%}'],
    'Annual Return': [f'{s1_annual:.2%}', f'{s2_annual:.2%}', f'{s3_annual:.2%}'],
    'Sharpe Ratio': [f'{s1_sharpe:.4f}', f'{s2_sharpe:.4f}', f'{s3_sharpe:.4f}'],
    'Max Drawdown': [f'{s1_dd:.2%}', f'{s2_dd:.2%}', f'{s3_dd:.2%}'],
    'Win Rate': [f'{s1_win:.1f}%', f'{s2_win:.1f}%', f'{s3_win:.1f}%']
})

print("\n" + summary_df.to_string(index=False))

# ============================================================================
# VISUALIZE
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Wealth
axes[0, 0].plot(s1_wealth.index, s1_wealth.values, label='Market Bubble + Buy Mom', linewidth=2, alpha=0.8)
axes[0, 0].plot(s2_wealth.index, s2_wealth.values, label='Hourly Momentum', linewidth=2, alpha=0.8)
axes[0, 0].plot(s3_wealth.index, s3_wealth.values, label='Daily MR', linewidth=2, alpha=0.8)
axes[0, 0].set_title('Wealth Curves', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('Wealth (log scale)')
axes[0, 0].set_yscale('log')
axes[0, 0].legend(loc='best')
axes[0, 0].grid(True, alpha=0.3)

# Annual returns
axes[0, 1].bar(['Bubble+Mom', 'Hourly Mom', 'Daily MR'],
               [s1_annual*100, s2_annual*100, s3_annual*100],
               color=['green' if x > 0 else 'red' for x in [s1_annual, s2_annual, s3_annual]], alpha=0.7)
axes[0, 1].set_title('Annual Returns (%)', fontsize=13, fontweight='bold')
axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Sharpe
axes[1, 0].bar(['Bubble+Mom', 'Hourly Mom', 'Daily MR'],
               [s1_sharpe, s2_sharpe, s3_sharpe],
               color='steelblue', alpha=0.7)
axes[1, 0].set_title('Sharpe Ratios', fontsize=13, fontweight='bold')
axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Drawdown
axes[1, 1].bar(['Bubble+Mom', 'Hourly Mom', 'Daily MR'],
               [s1_dd*100, s2_dd*100, s3_dd*100],
               color='darkred', alpha=0.7)
axes[1, 1].set_title('Max Drawdowns (%)', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/hourly_3strategies_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved: results/hourly_3strategies_comparison.png")

summary_df.to_csv('results/hourly_3strategies_results.csv', index=False)
print("Saved: results/hourly_3strategies_results.csv")

print("\n" + "="*120)
print("COMPLETE")
print("="*120)
