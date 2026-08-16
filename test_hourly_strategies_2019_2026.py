"""
HOURLY STRATEGIES BACKTEST (2019-2026)
=====================================
Test 4 hourly strategies on merged hourly data:
1. Hourly Momentum Strategy (20-hour lookback, 5-hour hold)
2. Daily Mean Reversion (DailyMR) - Revert to hourly open
3. BubbleQQQ Strategy - Buy QQQ when bubble low
4. Buy Momentum (Simple momentum-based buys)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os

print("="*120)
print("HOURLY STRATEGIES BACKTEST (2019-2026)")
print("="*120)

# Load hourly data
print("\nLoading hourly data...")
close_data = pd.read_parquet("data/cache/merged_hourly_close.parquet")
print(f"Loaded: {close_data.shape}")
print(f"Period: {close_data.index[0]} to {close_data.index[-1]}")
print(f"Years: {(close_data.index[-1] - close_data.index[0]).days / 365.25:.2f}")

# Calculate returns
ret_hourly = close_data.pct_change().fillna(0)

# ============================================================================
# STRATEGY 1: HOURLY MOMENTUM (20-hour lookback, 5-hour hold)
# ============================================================================

print("\n" + "="*120)
print("STRATEGY 1: HOURLY MOMENTUM")
print("="*120)

lookback_hours = 20
hold_hours = 5
top_n = 5

hourly_mom_returns = []
hourly_mom_signals = []

ret_mom_hourly = close_data.pct_change(lookback_hours).fillna(0)

for i in range(lookback_hours, len(ret_mom_hourly) - hold_hours, hold_hours):
    # Rank all stocks by 20-hour momentum
    ranking = ret_mom_hourly.iloc[i:i+1].rank(axis=1, ascending=False)
    ranked_idx = np.argsort(ranking.values[0])

    # Top 5 long, Bottom 5 short
    long_idx = ranked_idx[:top_n]
    short_idx = ranked_idx[-top_n:]

    # Calculate returns for 5-hour hold
    for j in range(i+1, min(i+1+hold_hours, len(ret_hourly))):
        date = ret_hourly.index[j]

        # Long returns
        long_ret = ret_hourly.iloc[j][long_idx].mean()

        # Short returns
        short_ret = ret_hourly.iloc[j][short_idx].mean()

        # Strategy return
        strat_ret = (long_ret - short_ret) / 10 - 0.001 / hold_hours  # Transaction cost

        hourly_mom_returns.append(strat_ret)
        hourly_mom_signals.append(date)

hourly_mom_series = pd.Series(hourly_mom_returns, index=hourly_mom_signals)
hourly_mom_wealth = (1 + hourly_mom_series).cumprod()

total_ret_hm = hourly_mom_wealth.iloc[-1] - 1
annual_ret_hm = (1 + total_ret_hm) ** (1/7.41) - 1
sharpe_hm = hourly_mom_series.mean() / hourly_mom_series.std() * np.sqrt(252*6.5) if hourly_mom_series.std() != 0 else 0
max_dd_hm = (hourly_mom_wealth / hourly_mom_wealth.cummax() - 1).min()

print(f"\nHourly Momentum Results:")
print(f"  Total Return: {total_ret_hm:.2%}")
print(f"  Annual Return: {annual_ret_hm:.2%}")
print(f"  Sharpe Ratio: {sharpe_hm:.4f}")
print(f"  Max Drawdown: {max_dd_hm:.2%}")
print(f"  Win Rate: {(hourly_mom_series > 0).sum() / len(hourly_mom_series) * 100:.1f}%")

# ============================================================================
# STRATEGY 2: DAILY MEAN REVERSION (DailyMR)
# ============================================================================

print("\n" + "="*120)
print("STRATEGY 2: DAILY MEAN REVERSION (DailyMR)")
print("="*120)

# Identify daily opens (9:30 AM)
hourly_data_with_date = ret_hourly.copy()
hourly_data_with_date['date'] = hourly_data_with_date.index.date
hourly_data_with_date['hour'] = hourly_data_with_date.index.hour
hourly_data_with_date['minute'] = hourly_data_with_date.index.minute

# Find open price (9:30) and close price (16:00)
open_rows = hourly_data_with_date[(hourly_data_with_date['hour'] == 9) |
                                   (hourly_data_with_date['hour'] == 14)]

daily_mr_returns = []
daily_mr_signals = []

for i in range(len(close_data) - 1):
    current_ret = ret_hourly.iloc[i]

    # Mean reversion: if up a lot, short; if down a lot, long
    momentum_signal = current_ret

    # Position: opposite of momentum (mean reversion)
    position = -momentum_signal.rank(ascending=False) / len(momentum_signal)
    position = position.clip(-1/10, 1/10)  # Limit exposure

    # Next hour return
    next_ret = ret_hourly.iloc[i+1]
    strategy_ret = (position * next_ret).sum() - 0.0005  # Tiny transaction cost

    daily_mr_returns.append(strategy_ret)
    daily_mr_signals.append(ret_hourly.index[i+1])

daily_mr_series = pd.Series(daily_mr_returns, index=daily_mr_signals)
daily_mr_wealth = (1 + daily_mr_series).cumprod()

total_ret_dmr = daily_mr_wealth.iloc[-1] - 1
annual_ret_dmr = (1 + total_ret_dmr) ** (1/7.41) - 1
sharpe_dmr = daily_mr_series.mean() / daily_mr_series.std() * np.sqrt(252*6.5) if daily_mr_series.std() != 0 else 0
max_dd_dmr = (daily_mr_wealth / daily_mr_wealth.cummax() - 1).min()

print(f"\nDaily Mean Reversion Results:")
print(f"  Total Return: {total_ret_dmr:.2%}")
print(f"  Annual Return: {annual_ret_dmr:.2%}")
print(f"  Sharpe Ratio: {sharpe_dmr:.4f}")
print(f"  Max Drawdown: {max_dd_dmr:.2%}")
print(f"  Win Rate: {(daily_mr_series > 0).sum() / len(daily_mr_series) * 100:.1f}%")

# ============================================================================
# STRATEGY 3: BUBBLE QQQ
# ============================================================================

print("\n" + "="*120)
print("STRATEGY 3: BUBBLE QQQ STRATEGY")
print("="*120)

# Get QQQ data
if "QQQ" in close_data.columns:
    qqq_close = close_data["QQQ"]
    qqq_ret = ret_hourly["QQQ"]

    # Calculate bubble score
    def calc_bubble_hourly(price, ma_w=24, z_w=48):  # 24 hours = 1 day, 48 hours = 2 days
        log_p = np.log(price + 1)
        fair = log_p.rolling(ma_w).mean()
        res = log_p - fair
        z = (res - res.rolling(z_w).mean()) / res.rolling(z_w).std()
        return np.tanh(z / 2)

    bubble_score = calc_bubble_hourly(qqq_close, 24, 48)

    bubble_qqq_returns = []
    bubble_qqq_signals = []

    for i in range(48, len(qqq_ret) - 1):
        # Buy when bubble < -0.7 (undervalued)
        if bubble_score.iloc[i] < -0.7:
            # Hold for 5 hours
            future_ret = qqq_ret.iloc[i+1:i+6].mean()
            strat_ret = future_ret - 0.001
        # Sell when bubble > 0.8 (overvalued)
        elif bubble_score.iloc[i] > 0.8:
            # Short for 5 hours
            future_ret = qqq_ret.iloc[i+1:i+6].mean()
            strat_ret = -future_ret - 0.001
        else:
            strat_ret = 0

        bubble_qqq_returns.append(strat_ret)
        bubble_qqq_signals.append(qqq_ret.index[i+1])

    bubble_qqq_series = pd.Series(bubble_qqq_returns, index=bubble_qqq_signals)
    bubble_qqq_wealth = (1 + bubble_qqq_series).cumprod()

    total_ret_bq = bubble_qqq_wealth.iloc[-1] - 1
    annual_ret_bq = (1 + total_ret_bq) ** (1/7.41) - 1
    sharpe_bq = bubble_qqq_series.mean() / bubble_qqq_series.std() * np.sqrt(252*6.5) if bubble_qqq_series.std() != 0 else 0
    max_dd_bq = (bubble_qqq_wealth / bubble_qqq_wealth.cummax() - 1).min()

    print(f"\nBubble QQQ Results:")
    print(f"  Total Return: {total_ret_bq:.2%}")
    print(f"  Annual Return: {annual_ret_bq:.2%}")
    print(f"  Sharpe Ratio: {sharpe_bq:.4f}")
    print(f"  Max Drawdown: {max_dd_bq:.2%}")
    print(f"  Win Rate: {(bubble_qqq_series > 0).sum() / len(bubble_qqq_series) * 100:.1f}%")
else:
    print("QQQ not in data")
    total_ret_bq = annual_ret_bq = sharpe_bq = max_dd_bq = 0

# ============================================================================
# STRATEGY 4: BUY MOMENTUM (Simple momentum-based buys)
# ============================================================================

print("\n" + "="*120)
print("STRATEGY 4: BUY MOMENTUM")
print("="*120)

buy_mom_returns = []
buy_mom_signals = []

ret_momentum_10h = close_data.pct_change(10).fillna(0)  # 10-hour momentum

for i in range(10, len(ret_momentum_10h) - 1):
    # Rank by 10-hour momentum
    ranking = ret_momentum_10h.iloc[i:i+1].rank(axis=1, ascending=False)
    ranked_idx = np.argsort(ranking.values[0])

    # Buy top 10 stocks
    top_10_idx = ranked_idx[:10]

    # Equal weight long only
    next_ret = ret_hourly.iloc[i+1][top_10_idx].mean()
    strat_ret = next_ret - 0.0005  # Small transaction cost

    buy_mom_returns.append(strat_ret)
    buy_mom_signals.append(ret_hourly.index[i+1])

buy_mom_series = pd.Series(buy_mom_returns, index=buy_mom_signals)
buy_mom_wealth = (1 + buy_mom_series).cumprod()

total_ret_bm = buy_mom_wealth.iloc[-1] - 1
annual_ret_bm = (1 + total_ret_bm) ** (1/7.41) - 1
sharpe_bm = buy_mom_series.mean() / buy_mom_series.std() * np.sqrt(252*6.5) if buy_mom_series.std() != 0 else 0
max_dd_bm = (buy_mom_wealth / buy_mom_wealth.cummax() - 1).min()

print(f"\nBuy Momentum Results:")
print(f"  Total Return: {total_ret_bm:.2%}")
print(f"  Annual Return: {annual_ret_bm:.2%}")
print(f"  Sharpe Ratio: {sharpe_bm:.4f}")
print(f"  Max Drawdown: {max_dd_bm:.2%}")
print(f"  Win Rate: {(buy_mom_series > 0).sum() / len(buy_mom_series) * 100:.1f}%")

# ============================================================================
# COMPARISON & SUMMARY
# ============================================================================

print("\n" + "="*120)
print("STRATEGY COMPARISON (2019-2026 Hourly Data)")
print("="*120)

summary_data = {
    'Strategy': ['Hourly Momentum', 'Daily Mean Reversion', 'Bubble QQQ', 'Buy Momentum'],
    'Total Return': [f'{total_ret_hm:.2%}', f'{total_ret_dmr:.2%}', f'{total_ret_bq:.2%}', f'{total_ret_bm:.2%}'],
    'Annual Return': [f'{annual_ret_hm:.2%}', f'{annual_ret_dmr:.2%}', f'{annual_ret_bq:.2%}', f'{annual_ret_bm:.2%}'],
    'Sharpe Ratio': [f'{sharpe_hm:.4f}', f'{sharpe_dmr:.4f}', f'{sharpe_bq:.4f}', f'{sharpe_bm:.4f}'],
    'Max Drawdown': [f'{max_dd_hm:.2%}', f'{max_dd_dmr:.2%}', f'{max_dd_bq:.2%}', f'{max_dd_bm:.2%}']
}

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: Wealth curves
axes[0, 0].plot(hourly_mom_wealth.index, hourly_mom_wealth.values, label='Hourly Momentum', linewidth=2)
axes[0, 0].plot(daily_mr_wealth.index, daily_mr_wealth.values, label='Daily Mean Reversion', linewidth=2)
axes[0, 0].plot(bubble_qqq_wealth.index, bubble_qqq_wealth.values, label='Bubble QQQ', linewidth=2)
axes[0, 0].plot(buy_mom_wealth.index, buy_mom_wealth.values, label='Buy Momentum', linewidth=2)
axes[0, 0].set_title('Wealth Curves - All Strategies', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Wealth Multiple')
axes[0, 0].set_yscale('log')
axes[0, 0].legend(loc='best')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Annual returns
strategies = ['Hourly Mom', 'Daily MR', 'Bubble QQQ', 'Buy Mom']
annual_returns = [annual_ret_hm, annual_ret_dmr, annual_ret_bq, annual_ret_bm]
colors = ['green' if r > 0 else 'red' for r in annual_returns]
axes[0, 1].bar(strategies, [r*100 for r in annual_returns], color=colors, alpha=0.7)
axes[0, 1].set_title('Annual Returns Comparison', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Annual Return (%)')
axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Sharpe ratios
sharpe_ratios = [sharpe_hm, sharpe_dmr, sharpe_bq, sharpe_bm]
axes[1, 0].bar(strategies, sharpe_ratios, color='steelblue', alpha=0.7)
axes[1, 0].set_title('Sharpe Ratios Comparison', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Sharpe Ratio')
axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Drawdowns
max_dds = [max_dd_hm, max_dd_dmr, max_dd_bq, max_dd_bm]
axes[1, 1].bar(strategies, [d*100 for d in max_dds], color='darkred', alpha=0.7)
axes[1, 1].set_title('Max Drawdowns Comparison', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Max Drawdown (%)')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/hourly_strategies_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: results/hourly_strategies_comparison.png")

# Save results to CSV
results_df = pd.DataFrame({
    'Strategy': strategies,
    'Total_Return': annual_returns,
    'Annual_Return': annual_returns,
    'Sharpe_Ratio': sharpe_ratios,
    'Max_Drawdown': max_dds
})

results_df.to_csv('results/hourly_strategies_results.csv', index=False)
print("Saved: results/hourly_strategies_results.csv")

print("\n" + "="*120)
print("HOURLY STRATEGIES BACKTEST COMPLETE")
print("="*120)

print("\nKEY FINDINGS:")
print(f"  Best Strategy: {strategies[np.argmax(sharpe_ratios)]} (Sharpe: {max(sharpe_ratios):.4f})")
print(f"  Most Profitable: {strategies[np.argmax(annual_returns)]} (Annual: {max(annual_returns):.2%})")
print(f"  Most Consistent: {strategies[np.argmin([abs(d) for d in max_dds])]} (Max DD: {min([abs(d) for d in max_dds]):.2%})")
