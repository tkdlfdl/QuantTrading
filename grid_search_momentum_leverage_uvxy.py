"""
GRID SEARCH: Daily Momentum + Low Bubble Leverage + High Bubble UVXY
===================================================================
Test all parameter combinations across 1997-2026 data
Find optimal thresholds and allocations
"""

import sys, warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product

print("="*100)
print("GRID SEARCH: DAILY MOMENTUM + LOW BUBBLE LEVERAGE + HIGH BUBBLE UVXY")
print("="*100)

# Load extended data
print("\nLoading data...")
close_data = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
print(f"Loaded: {close_data.shape}")
print(f"Period: {close_data.index[0].date()} to {close_data.index[-1].date()}")

# Parameters
lookback = 140
holding_period = 40
top = 5

# Calculate returns
ret_daily = close_data.pct_change().ffill().fillna(0)
ret_mom = close_data.pct_change(lookback).ffill().fillna(0)

print(f"\nGenerating base momentum signals...")
strategy_returns = []
signal_count = 0

# Generate daily momentum returns
for i in range(lookback + 1, len(ret_mom), holding_period):
    ranking = ret_mom.iloc[i - 1:i].rank(axis=1, ascending=False)
    ranked_idx = np.argsort(ranking.values[0])

    short_num = ret_mom.iloc[:, ranked_idx[:top]].iloc[i - 1:i].lt(0.0).sum().sum()
    long_num = top - short_num

    if long_num <= 0:
        continue

    hold_end = min(i + holding_period, len(ret_daily))

    for j in range(i, hold_end):
        date = ret_daily.index[j]

        long_rets = ret_daily.iloc[j, ranked_idx[:long_num]].values if long_num > 0 else []
        short_rets = ret_daily.iloc[j, ranked_idx[-short_num:]].values if short_num > 0 else []

        long_r = np.nanmean(long_rets) if len(long_rets) > 0 else 0.0
        short_r = np.nanmean(short_rets) if len(short_rets) > 0 else 0.0

        mom_r = (long_r - short_r) / 10.0 - 0.005 / holding_period

        # Get hedge assets
        uvxy_r = 0.0
        if 'UVXY' in close_data.columns and pd.notna(close_data.loc[date, 'UVXY']):
            uvxy_r = ret_daily.loc[date, 'UVXY']

        strategy_returns.append({
            'Date': date,
            'Momentum': mom_r,
            'UVXY': uvxy_r,
        })
        signal_count += 1

ret_df = pd.DataFrame(strategy_returns).set_index('Date')
print(f"Generated {signal_count} daily returns")

# Calculate bubble score
print("Calculating bubble score...")
def calc_bubble(price, ma_w=252, z_w=252):
    log_p = np.log(price)
    fair = price.rolling(ma_w).mean()
    log_fair = np.log(fair)
    res = log_p - log_fair
    z = (res - res.rolling(z_w).mean()) / res.rolling(z_w).std()
    return np.tanh(z / 2)

if 'QQQ' in close_data.columns:
    bubble_score = calc_bubble(close_data.loc[ret_df.index, 'QQQ'])
else:
    print("WARNING: QQQ not found, using momentum-based bubble")
    bubble_score = calc_bubble(calc_bubble((1 + ret_df['Momentum']).cumprod()))

print(f"Bubble score range: {bubble_score.min():.3f} to {bubble_score.max():.3f}")

# Grid search parameters
print(f"\n{'='*100}")
print("GRID SEARCH PARAMETERS")
print(f"{'='*100}")

grid_params = {
    'low_bubble_threshold': [-0.95, -0.92, -0.90, -0.88],  # When to activate leverage
    'leverage_multiplier': [1.1, 1.2, 1.3, 1.4],            # Leverage amount
    'leverage_hold_days': [40, 50, 60],                     # Hold period for leverage
    'high_bubble_threshold': [0.80, 0.85, 0.90],            # When to activate UVXY
    'uvxy_allocation': [0.3, 0.4, 0.5],                     # % allocated to UVXY
    'uvxy_hold_days': [30, 40, 50],                         # Hold period for UVXY
}

total_combos = np.prod([len(v) for v in grid_params.values()])
print(f"\nTotal combinations to test: {total_combos}")
print("\nParameters:")
for param, values in grid_params.items():
    print(f"  {param}: {values}")

# Grid search
print(f"\n{'='*100}")
print("RUNNING GRID SEARCH")
print(f"{'='*100}")

grid_results = []
combo_count = 0

for (low_bubble, leverage, lev_hold, high_bubble, uvxy_alloc, uvxy_hold) in product(
    grid_params['low_bubble_threshold'],
    grid_params['leverage_multiplier'],
    grid_params['leverage_hold_days'],
    grid_params['high_bubble_threshold'],
    grid_params['uvxy_allocation'],
    grid_params['uvxy_hold_days'],
):
    combo_count += 1

    if combo_count % 100 == 0:
        print(f"  Testing {combo_count}/{total_combos}...", flush=True)

    try:
        # Initialize tracking
        lev_remaining = 0
        uvxy_remaining = 0
        strategy_ret = []

        for date in ret_df.index:
            bubble = bubble_score.loc[date]
            mom_base = ret_df.loc[date, 'Momentum']
            uvxy_base = ret_df.loc[date, 'UVXY']

            # Check signals
            if lev_remaining == 0 and bubble < low_bubble:
                lev_remaining = lev_hold

            if uvxy_remaining == 0 and bubble > high_bubble:
                uvxy_remaining = uvxy_hold

            # Calculate return
            if lev_remaining > 0:
                # Leverage regime
                lev_cost = -(leverage - 1.0) * (0.10 / 252)
                daily_ret = leverage * mom_base + lev_cost
                lev_remaining -= 1

            elif uvxy_remaining > 0:
                # UVXY hedge regime
                daily_ret = (1 - uvxy_alloc) * mom_base + uvxy_alloc * uvxy_base
                uvxy_remaining -= 1

            else:
                # Normal momentum
                daily_ret = mom_base

            strategy_ret.append(daily_ret)

        # Calculate metrics
        strategy_series = pd.Series(strategy_ret, index=ret_df.index)
        wealth = (1 + strategy_series).cumprod() / ((1 + strategy_series).cumprod().iloc[0])

        total_ret = wealth.iloc[-1] - 1
        annual_ret = (1 + total_ret) ** (1 / 29.42) - 1 if total_ret > -1 else 0

        daily_std = strategy_series.std()
        annual_vol = daily_std * np.sqrt(252)
        sharpe = annual_ret / annual_vol if annual_vol != 0 else 0

        downside = strategy_series[strategy_series < 0]
        downside_std = downside.std() if len(downside) > 0 else 0
        sortino = annual_ret / downside_std if downside_std != 0 else 0

        max_dd = (wealth / wealth.cummax() - 1).min()

        grid_results.append({
            'low_bubble_threshold': low_bubble,
            'leverage_multiplier': leverage,
            'leverage_hold_days': lev_hold,
            'high_bubble_threshold': high_bubble,
            'uvxy_allocation': uvxy_alloc,
            'uvxy_hold_days': uvxy_hold,
            'Total Return': total_ret,
            'Annual Return': annual_ret,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': max_dd,
            'Annual Vol': annual_vol,
        })

    except Exception as e:
        continue

grid_df = pd.DataFrame(grid_results)
grid_df = grid_df.sort_values('Sharpe Ratio', ascending=False)

print(f"\nCompleted {combo_count} combinations")

# Display results
print(f"\n{'='*100}")
print("TOP 20 RESULTS (Sorted by Sharpe Ratio)")
print(f"{'='*100}")

display_cols = ['low_bubble_threshold', 'leverage_multiplier', 'high_bubble_threshold',
                'uvxy_allocation', 'Total Return', 'Annual Return', 'Sharpe Ratio', 'Max Drawdown']

print(grid_df[display_cols].head(20).to_string())

# Best result
best = grid_df.iloc[0]

print(f"\n{'='*100}")
print("OPTIMAL PARAMETERS")
print(f"{'='*100}")
print(f"""
Low Bubble Leverage:
  Threshold: {best['low_bubble_threshold']}
  Leverage: {best['leverage_multiplier']:.1f}x
  Hold Days: {int(best['leverage_hold_days'])}

High Bubble UVXY:
  Threshold: {best['high_bubble_threshold']}
  UVXY Allocation: {best['uvxy_allocation']:.1%}
  Hold Days: {int(best['uvxy_hold_days'])}

Performance:
  Total Return: {best['Total Return']:.2%}
  Annual Return: {best['Annual Return']:.2%}
  Sharpe Ratio: {best['Sharpe Ratio']:.4f}
  Sortino Ratio: {best['Sortino Ratio']:.4f}
  Max Drawdown: {best['Max Drawdown']:.2%}
  Annual Vol: {best['Annual Vol']:.2%}
""")

# Save results
os.makedirs('results', exist_ok=True)

grid_df.to_csv('results/grid_search_momentum_leverage_uvxy.csv', index=False)
print(f"\nSaved results: results/grid_search_momentum_leverage_uvxy.csv")

# Create comparison chart
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Sharpe by leverage
lev_sharpe = grid_df.groupby('leverage_multiplier')['Sharpe Ratio'].mean().sort_index()
axes[0, 0].plot(lev_sharpe.index, lev_sharpe.values, marker='o', linewidth=2)
axes[0, 0].set_title('Average Sharpe Ratio by Leverage Multiplier')
axes[0, 0].set_xlabel('Leverage Multiplier')
axes[0, 0].set_ylabel('Sharpe Ratio')
axes[0, 0].grid(True, alpha=0.3)

# Sharpe by UVXY allocation
uvxy_sharpe = grid_df.groupby('uvxy_allocation')['Sharpe Ratio'].mean().sort_index()
axes[0, 1].plot(uvxy_sharpe.index, uvxy_sharpe.values, marker='s', linewidth=2, color='orange')
axes[0, 1].set_title('Average Sharpe Ratio by UVXY Allocation')
axes[0, 1].set_xlabel('UVXY Allocation')
axes[0, 1].set_ylabel('Sharpe Ratio')
axes[0, 1].grid(True, alpha=0.3)

# Return by leverage threshold
lev_ret = grid_df.groupby('low_bubble_threshold')['Annual Return'].mean().sort_index()
axes[1, 0].bar(range(len(lev_ret)), lev_ret.values, alpha=0.7)
axes[1, 0].set_xticks(range(len(lev_ret)))
axes[1, 0].set_xticklabels([f"{x:.2f}" for x in lev_ret.index])
axes[1, 0].set_title('Average Annual Return by Low Bubble Threshold')
axes[1, 0].set_xlabel('Low Bubble Threshold')
axes[1, 0].set_ylabel('Annual Return')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Return by high bubble threshold
high_ret = grid_df.groupby('high_bubble_threshold')['Annual Return'].mean().sort_index()
axes[1, 1].bar(range(len(high_ret)), high_ret.values, alpha=0.7, color='orange')
axes[1, 1].set_xticks(range(len(high_ret)))
axes[1, 1].set_xticklabels([f"{x:.2f}" for x in high_ret.index])
axes[1, 1].set_title('Average Annual Return by High Bubble Threshold')
axes[1, 1].set_xlabel('High Bubble Threshold')
axes[1, 1].set_ylabel('Annual Return')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/grid_search_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved chart: results/grid_search_analysis.png")

print(f"\n{'='*100}")
print("GRID SEARCH COMPLETE")
print(f"{'='*100}")
print(f"""
Results Summary:
  Total combinations tested: {combo_count}
  Best Sharpe Ratio: {best['Sharpe Ratio']:.4f}
  Best Annual Return: {best['Annual Return']:.2%}
  Best Max Drawdown: {best['Max Drawdown']:.2%}

Files saved:
  - results/grid_search_momentum_leverage_uvxy.csv (all results)
  - results/grid_search_analysis.png (parameter analysis)

Recommendation:
  Use optimal parameters above for best risk-adjusted returns
  Test on live paper trading for 2-3 months
  Monitor leverage and UVXY allocation quarterly
""")
