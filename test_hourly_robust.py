"""
HOURLY STRATEGIES - ROBUST VERSION (2019-2026)
==============================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os
import warnings
warnings.filterwarnings('ignore')

print("="*120)
print("HOURLY STRATEGIES - ROBUST BACKTEST (2019-2026)")
print("="*120)

close_data = pd.read_parquet("data/cache/merged_hourly_close.parquet")
ret_hourly = close_data.pct_change().fillna(0)

print(f"\nData: {close_data.shape}")
print(f"Period: {close_data.index[0]} to {close_data.index[-1]}")
years = (close_data.index[-1] - close_data.index[0]).days / 365.25

# ============================================================================
# STRATEGY 1: MARKET BUBBLE + BUY MOMENTUM
# ============================================================================

print("\n" + "="*120)
print("STRATEGY 1: MARKET BUBBLE + BUY MOMENTUM")
print("="*120)

# Market bubble
market_px = close_data.mean(axis=1).ffill().bfill()
log_p = np.log(market_px.clip(lower=0.001))
ma = log_p.rolling(24, min_periods=1).mean()
res = log_p - ma
z = (res - res.rolling(48, min_periods=1).mean()) / (res.rolling(48, min_periods=1).std() + 1e-6)
bubble = np.tanh(z / 2).fillna(0)

# 20-hour momentum
ret_mom_20 = close_data.pct_change(20).fillna(0)

s1_returns = []
for i in range(50, len(bubble) - 5):
    bs = bubble.iloc[i]

    if bs < -0.7:  # Undervalued - BUY top momentum
        mom_rank = ret_mom_20.iloc[i].rank(ascending=False)
        top5_cols = mom_rank.nsmallest(5).index  # Top 5
        future_ret = ret_hourly.iloc[i+1:i+6, :][top5_cols].values.mean()
        s1_ret = future_ret - 0.001
    elif bs > 0.7:  # Overvalued - SHORT top momentum
        mom_rank = ret_mom_20.iloc[i].rank(ascending=False)
        top5_cols = mom_rank.nsmallest(5).index  # Top 5
        future_ret = ret_hourly.iloc[i+1:i+6, :][top5_cols].values.mean()
        s1_ret = -future_ret - 0.001
    else:
        s1_ret = 0

    s1_returns.append(np.clip(s1_ret, -0.1, 0.1))  # Clip large returns

s1_series = pd.Series(s1_returns, index=bubble.index[50:50+len(s1_returns)])
s1_wealth = (1 + s1_series).cumprod()

s1_total = s1_wealth.iloc[-1] - 1
s1_annual = (1 + s1_total) ** (1/years) - 1
s1_sharpe = s1_series.mean() / s1_series.std() * np.sqrt(252*6.5) if s1_series.std() > 0 else 0
s1_dd = (s1_wealth / s1_wealth.cummax() - 1).min()
s1_win = (s1_series > 0).sum() / len(s1_series) * 100

print(f"Total Return: {s1_total:.2%}")
print(f"Annual Return: {s1_annual:.2%}")
print(f"Sharpe Ratio: {s1_sharpe:.4f}")
print(f"Max Drawdown: {s1_dd:.2%}")
print(f"Win Rate: {s1_win:.1f}%")

# ============================================================================
# STRATEGY 2: HOURLY MOMENTUM (Conservative)
# ============================================================================

print("\n" + "="*120)
print("STRATEGY 2: HOURLY MOMENTUM - Conservative (20h lookback, 5h hold)")
print("="*120)

s2_returns = []

for i in range(20, len(ret_mom_20) - 5, 5):
    mom = ret_mom_20.iloc[i]

    # Top 5 long
    top5_idx = mom.nlargest(5).index
    # Bottom 5 short
    bot5_idx = mom.nsmallest(5).index

    # 5-hour forward
    for j in range(i+1, min(i+6, len(ret_hourly))):
        ret = ret_hourly.iloc[j]
        long_ret = ret[top5_idx].mean()
        short_ret = ret[bot5_idx].mean()

        # Market neutral 10% allocation
        s2_ret = (long_ret - short_ret) * 0.1 - 0.0005
        s2_returns.append(np.clip(s2_ret, -0.05, 0.05))

s2_series = pd.Series(s2_returns, index=ret_hourly.index[21:21+len(s2_returns)])
s2_wealth = (1 + s2_series).cumprod()

s2_total = s2_wealth.iloc[-1] - 1
s2_annual = (1 + s2_total) ** (1/years) - 1
s2_sharpe = s2_series.mean() / s2_series.std() * np.sqrt(252*6.5) if s2_series.std() > 0 else 0
s2_dd = (s2_wealth / s2_wealth.cummax() - 1).min()
s2_win = (s2_series > 0).sum() / len(s2_series) * 100

print(f"Total Return: {s2_total:.2%}")
print(f"Annual Return: {s2_annual:.2%}")
print(f"Sharpe Ratio: {s2_sharpe:.4f}")
print(f"Max Drawdown: {s2_dd:.2%}")
print(f"Win Rate: {s2_win:.1f}%")

# ============================================================================
# STRATEGY 3: DAILY MEAN REVERSION
# ============================================================================

print("\n" + "="*120)
print("STRATEGY 3: DAILY MEAN REVERSION (Conservative)")
print("="*120)

s3_returns = []

for i in range(1, len(ret_hourly) - 1):
    ret = ret_hourly.iloc[i]

    # Mean reversion: opposite direction
    signal = -ret / (ret.std() + 1e-6)
    signal = signal.clip(-0.02, 0.02)  # Light positions

    ret_next = ret_hourly.iloc[i+1]
    pnl = (signal * ret_next).mean()
    s3_ret = pnl - 0.0001

    s3_returns.append(np.clip(s3_ret, -0.02, 0.02))

s3_series = pd.Series(s3_returns, index=ret_hourly.index[2:2+len(s3_returns)])
s3_wealth = (1 + s3_series).cumprod()

s3_total = s3_wealth.iloc[-1] - 1
s3_annual = (1 + s3_total) ** (1/years) - 1
s3_sharpe = s3_series.mean() / s3_series.std() * np.sqrt(252*6.5) if s3_series.std() > 0 else 0
s3_dd = (s3_wealth / s3_wealth.cummax() - 1).min()
s3_win = (s3_series > 0).sum() / len(s3_series) * 100

print(f"Total Return: {s3_total:.2%}")
print(f"Annual Return: {s3_annual:.2%}")
print(f"Sharpe Ratio: {s3_sharpe:.4f}")
print(f"Max Drawdown: {s3_dd:.2%}")
print(f"Win Rate: {s3_win:.1f}%")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*120)
print("HOURLY STRATEGIES COMPARISON (2019-2026)")
print("="*120)

summary = pd.DataFrame({
    'Strategy': ['Market Bubble + Buy Mom', 'Hourly Momentum (LongShort)', 'Daily Mean Reversion'],
    'Total Return': [f'{s1_total:.2%}', f'{s2_total:.2%}', f'{s3_total:.2%}'],
    'Annual Return': [f'{s1_annual:.2%}', f'{s2_annual:.2%}', f'{s3_annual:.2%}'],
    'Sharpe Ratio': [f'{s1_sharpe:.4f}', f'{s2_sharpe:.4f}', f'{s3_sharpe:.4f}'],
    'Max Drawdown': [f'{s1_dd:.2%}', f'{s2_dd:.2%}', f'{s3_dd:.2%}'],
    'Win Rate': [f'{s1_win:.1f}%', f'{s2_win:.1f}%', f'{s3_win:.1f}%']
})

print("\n" + summary.to_string(index=False))

# ============================================================================
# PLOTS
# ============================================================================

fig, ax = plt.subplots(2, 2, figsize=(18, 12))

# Wealth
ax[0, 0].plot(s1_wealth.index, s1_wealth.values, label='Market Bubble + Buy Mom', linewidth=2, alpha=0.8)
ax[0, 0].plot(s2_wealth.index, s2_wealth.values, label='Hourly Momentum', linewidth=2, alpha=0.8)
ax[0, 0].plot(s3_wealth.index, s3_wealth.values, label='Daily MR', linewidth=2, alpha=0.8)
ax[0, 0].set_title('Wealth Curves (All Strategies)', fontsize=13, fontweight='bold')
ax[0, 0].set_ylabel('Wealth Multiple')
ax[0, 0].set_yscale('log')
ax[0, 0].legend(fontsize=11)
ax[0, 0].grid(True, alpha=0.3)

# Annual returns
names = ['Bubble+Mom', 'H.Mom', 'DailyMR']
rets = [s1_annual, s2_annual, s3_annual]
colors = ['green' if r > 0 else 'red' for r in rets]
ax[0, 1].bar(names, [r*100 for r in rets], color=colors, alpha=0.7)
ax[0, 1].set_title('Annual Returns (%)', fontsize=13, fontweight='bold')
ax[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax[0, 1].grid(True, alpha=0.3, axis='y')

# Sharpe
sharpes = [s1_sharpe, s2_sharpe, s3_sharpe]
ax[1, 0].bar(names, sharpes, color='steelblue', alpha=0.7)
ax[1, 0].set_title('Sharpe Ratios', fontsize=13, fontweight='bold')
ax[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax[1, 0].grid(True, alpha=0.3, axis='y')

# Drawdowns
dds = [s1_dd, s2_dd, s3_dd]
ax[1, 1].bar(names, [d*100 for d in dds], color='darkred', alpha=0.7)
ax[1, 1].set_title('Max Drawdowns (%)', fontsize=13, fontweight='bold')
ax[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/hourly_3strategies_comparison.png', dpi=150, bbox_inches='tight')
print("\n\nSaved: results/hourly_3strategies_comparison.png")

summary.to_csv('results/hourly_3strategies_results.csv', index=False)
print("Saved: results/hourly_3strategies_results.csv")

print("\n" + "="*120)
print("ANALYSIS COMPLETE")
print("="*120)

print("\nKEY OBSERVATIONS:")
best_annual = max(rets)
best_sharpe = max(sharpes)
best_dd = min([abs(d) for d in dds])

print(f"[OK] Best Annual Return: {names[rets.index(best_annual)]} ({best_annual:.2%})")
print(f"[OK] Best Sharpe Ratio: {names[sharpes.index(best_sharpe)]} ({best_sharpe:.4f})")
print(f"[OK] Smallest Drawdown: {names[[abs(d) for d in dds].index(best_dd)]} ({min(dds):.2%})")
print(f"\nNote: Hourly data (2019-2026) is shorter than daily data (1997-2026)")
print(f"Cannot test on major crises (2008, 2000) with hourly data")
print(f"\nIMPORTANT: All strategies show negative returns on hourly data")
print(f"Suggests these strategies work better on daily/longer timeframes")
