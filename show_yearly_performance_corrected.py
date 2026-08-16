"""
YEARLY PERFORMANCE ANALYSIS - CORRECTED SHARPE RATIO
====================================================
Sharpe Ratio = (Daily Mean Return / Daily Std Dev) * sqrt(252)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Load extended data
close_data = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")

lookback = 140
holding_period = 40
top = 5
trading_days = 252

close = close_data.copy()
ret_daily_df = close.pct_change().ffill().fillna(0)
ret_df_mom = close.pct_change(lookback).ffill().fillna(0)

print("="*140)
print("CALCULATING CORRECTED YEARLY SHARPE RATIOS")
print("="*140)

strategy_returns = []
for i in range(lookback + 1, len(ret_df_mom), holding_period):
    ranking = ret_df_mom.iloc[i - 1:i].rank(axis=1, ascending=False)
    ranked_idx = np.argsort(ranking.values[0])

    short_num = ret_df_mom.iloc[:, ranked_idx[:top]].iloc[i - 1:i].lt(0.0).sum().sum()
    long_num = top - short_num

    if long_num <= 0:
        continue

    idx = min(i + holding_period, len(ret_df_mom))

    for j in range(i, idx):
        date = ret_daily_df.index[j]
        long_part = 0
        short_part = 0

        if long_num > 0:
            long_signal = np.sign(ret_df_mom.iloc[:, ranked_idx[:long_num]].iloc[i - 1:i]).abs()
            long_ret = long_signal.mul(np.array(ret_daily_df.iloc[:, ranked_idx[:long_num]].iloc[j:j + 1])[0])
            long_part = long_ret.values.mean() * long_num

        if short_num > 0:
            short_signal = np.sign(ret_df_mom.iloc[:, ranked_idx[-short_num:]].iloc[i - 1:i]).abs() * -1
            short_ret = short_signal.mul(np.array(ret_daily_df.iloc[:, ranked_idx[-short_num:]].iloc[j:j + 1])[0])
            short_part = short_ret.values.mean() * short_num

        mom_daily_ret = (long_part + short_part) / top - 0.005 / holding_period

        hedge_ret = 0.0
        if "UVXY" in close.columns and pd.notna(close.loc[date, "UVXY"]):
            hedge_ret = ret_daily_df.loc[date, "UVXY"]
        elif "^VIX" in close.columns and pd.notna(close.loc[date, "^VIX"]):
            vix_ret = ret_daily_df.loc[date, "^VIX"]
            if date < pd.Timestamp("2018-02-28"):
                hedge_ret = 2.0 * vix_ret - 0.0020 - 0.25 * (vix_ret ** 2)
            else:
                hedge_ret = 1.5 * vix_ret - 0.0015 - 0.25 * (vix_ret ** 2)

        strategy_returns.append({"Date": date, "Momentum": mom_daily_ret, "Hedge_Return": hedge_ret})

ret_df = pd.DataFrame(strategy_returns).set_index("Date")
ret_df = ret_df.dropna()

# Calculate bubble score
base_wealth = (1 + ret_df).cumprod()
base_wealth = base_wealth / base_wealth.iloc[0]

def calc_bubble(price, ma_w=120, z_w=240):
    log_p = np.log(price + 1)
    fair = log_p.rolling(ma_w).mean()
    res = log_p - fair
    z = (res - res.rolling(z_w).mean()) / res.rolling(z_w).std()
    return np.tanh(z / 2)

bubble_score = calc_bubble(base_wealth["Momentum"], 120, 240)

# Strategy parameters
hedge_bubble_entry = 0.85
hedge_alloc = 0.5
hedge_hold_days = 40
low_bubble_entry = -0.88
momentum_extra_leverage = 0.25
leverage_hold_days = 50
leverage_cost_annual = 0.1

raw_hedge_signal = bubble_score > hedge_bubble_entry
hedge_trade_signal = raw_hedge_signal.shift(1).fillna(False)

raw_leverage_signal = bubble_score < low_bubble_entry
leverage_trade_signal = raw_leverage_signal.shift(1).fillna(False)

daily_leverage_cost = leverage_cost_annual / trading_days
strategy_ret = []
hedge_remaining_days = 0
leverage_remaining_days = 0

for date in ret_df.index:
    if hedge_remaining_days == 0 and hedge_trade_signal.loc[date]:
        hedge_remaining_days = hedge_hold_days

    if leverage_remaining_days == 0 and leverage_trade_signal.loc[date]:
        leverage_remaining_days = leverage_hold_days

    base_momentum = ret_df.loc[date, "Momentum"]
    momentum_component = base_momentum

    if hedge_remaining_days > 0:
        momentum_component = (1.0 - hedge_alloc) * base_momentum
        uvxy_component = hedge_alloc * ret_df.loc[date, "Hedge_Return"]
        daily_ret = momentum_component + uvxy_component
        hedge_remaining_days -= 1

    elif leverage_remaining_days > 0:
        leveraged = momentum_extra_leverage * base_momentum
        cost = -momentum_extra_leverage * daily_leverage_cost
        daily_ret = momentum_component + leveraged + cost
        leverage_remaining_days -= 1

    else:
        daily_ret = momentum_component

    strategy_ret.append(daily_ret)

strategy_series = pd.Series(strategy_ret, index=ret_df.index)
wealth = (1 + strategy_series).cumprod() / ((1 + strategy_series).cumprod().iloc[0])

# Calculate yearly metrics with CORRECTED Sharpe
yearly_data = []

for year in sorted(strategy_series.index.year.unique()):
    year_mask = strategy_series.index.year == year
    year_ret = strategy_series[year_mask]
    year_wealth = wealth[year_mask]

    if len(year_ret) == 0:
        continue

    # Total return (geometric)
    total_ret = (1 + year_ret).prod() - 1

    # Sharpe: CORRECT formula = (Daily Mean / Daily Std) * sqrt(252)
    daily_mean = year_ret.mean()
    daily_std = year_ret.std()
    sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std != 0 else np.nan

    # Sortino: (Daily Mean / Downside Std) * sqrt(252)
    downside = year_ret[year_ret < 0]
    downside_std = downside.std() if len(downside) > 0 else 0
    sortino = (daily_mean / downside_std * np.sqrt(252)) if downside_std != 0 else np.nan

    # Max drawdown
    max_dd = (year_wealth / year_wealth.cummax() - 1).min()

    yearly_data.append({
        'Year': year,
        'Return': total_ret,
        'Daily_Mean': daily_mean,
        'Daily_Std': daily_std,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Max_Drawdown': max_dd,
        'Days': len(year_ret),
    })

yearly_df = pd.DataFrame(yearly_data)

print("\n" + "="*140)
print("CORRECTED YEARLY PERFORMANCE METRICS (1997-2026)")
print("="*140)

display_df = yearly_df.copy()
display_df['Return'] = display_df['Return'].apply(lambda x: f"{x:.2%}")
display_df['Daily_Mean'] = display_df['Daily_Mean'].apply(lambda x: f"{x:.6f}")
display_df['Daily_Std'] = display_df['Daily_Std'].apply(lambda x: f"{x:.6f}")
display_df['Sharpe'] = display_df['Sharpe'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "NaN")
display_df['Sortino'] = display_df['Sortino'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "NaN")
display_df['Max_Drawdown'] = display_df['Max_Drawdown'].apply(lambda x: f"{x:.2%}")

print(display_df[['Year', 'Return', 'Daily_Mean', 'Daily_Std', 'Sharpe', 'Max_Drawdown']].to_string(index=False))

# Summary statistics
yearly_numeric = yearly_df.copy()

print("\n" + "="*140)
print("CORRECTED SUMMARY STATISTICS")
print("="*140)

print(f"""
Period: 1997-2026 (30 years)
Total Years: {len(yearly_numeric)}
Positive Years: {(yearly_numeric['Return'] > 0).sum()}/{len(yearly_numeric)}

RETURNS:
  Best Year: {yearly_numeric['Return'].max():.2%} ({yearly_numeric[yearly_numeric['Return'] == yearly_numeric['Return'].max()]['Year'].values[0]})
  Worst Year: {yearly_numeric['Return'].min():.2%} ({yearly_numeric[yearly_numeric['Return'] == yearly_numeric['Return'].min()]['Year'].values[0]})
  Average: {yearly_numeric['Return'].mean():.2%}

SHARPE RATIOS (CORRECTED):
  Best: {yearly_numeric['Sharpe'].max():.4f}
  Worst: {yearly_numeric['Sharpe'].min():.4f}
  Average: {yearly_numeric['Sharpe'].mean():.4f}
  Median: {yearly_numeric['Sharpe'].median():.4f}

DRAWDOWNS:
  Worst: {yearly_numeric['Max_Drawdown'].min():.2%}
  Average: {yearly_numeric['Max_Drawdown'].mean():.2%}
""")

# Create visualization
fig, axes = plt.subplots(3, 1, figsize=(18, 12))

# Yearly returns
colors = ['green' if r > 0 else 'red' for r in yearly_numeric['Return']]
axes[0].bar(yearly_numeric['Year'], yearly_numeric['Return'] * 100, color=colors, alpha=0.7)
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[0].set_title('Yearly Returns (%)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Return (%)')
axes[0].grid(True, alpha=0.3, axis='y')

# Yearly Sharpe (CORRECTED)
axes[1].bar(yearly_numeric['Year'], yearly_numeric['Sharpe'], color='steelblue', alpha=0.7)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[1].set_title('Yearly Sharpe Ratios (CORRECTED)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Sharpe Ratio')
axes[1].grid(True, alpha=0.3, axis='y')

# Yearly Drawdowns
axes[2].bar(yearly_numeric['Year'], yearly_numeric['Max_Drawdown'] * 100, color='darkred', alpha=0.7)
axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[2].set_title('Yearly Maximum Drawdowns (%)', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Max Drawdown (%)')
axes[2].set_xlabel('Year')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/yearly_performance_metrics_corrected.png', dpi=150, bbox_inches='tight')
print(f"\nChart saved: results/yearly_performance_metrics_corrected.png")

yearly_df.to_csv('results/yearly_performance_corrected.csv', index=False)
print(f"Data saved: results/yearly_performance_corrected.csv")

print("\n" + "="*140)
print("CORRECTED ANALYSIS COMPLETE")
print("="*140)
