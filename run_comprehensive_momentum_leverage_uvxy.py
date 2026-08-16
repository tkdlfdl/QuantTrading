"""
RUN COMPREHENSIVE STRATEGY: Daily Momentum + Low Bubble Leverage + High Bubble UVXY
===================================================================================
Using the provided function with extended 1997-2026 data
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
print("COMPREHENSIVE STRATEGY: Daily Momentum + Low Bubble Leverage + High Bubble UVXY")
print("="*100)

# Load data
print("\nLoading extended data (1997-2026)...")
close_data = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
print(f"Loaded: {close_data.shape}")

# Performance stats function
def performance_stats(ret_df, wealth_df, trading_days=252):
    def sharpe_ratio(r):
        return np.sqrt(trading_days) * r.mean() / r.std() if r.std() != 0 else np.nan

    def sortino_ratio(r):
        downside = r[r < 0]
        downside_std = downside.std()
        return np.sqrt(trading_days) * r.mean() / downside_std if downside_std != 0 else np.nan

    def max_drawdown(w):
        return (w / w.cummax() - 1).min()

    analysis = pd.DataFrame(index=ret_df.columns)
    analysis["Total Return"] = wealth_df.iloc[-1] - 1
    analysis["Final Wealth"] = wealth_df.iloc[-1]
    analysis["Sharpe Ratio"] = ret_df.apply(sharpe_ratio)
    analysis["Sortino Ratio"] = ret_df.apply(sortino_ratio)
    analysis["Max Drawdown"] = wealth_df.apply(max_drawdown)
    return analysis


def yearly_performance_stats(ret_df, wealth_df, trading_days=252):
    def sortino_ratio(r):
        downside = r[r < 0]
        downside_std = downside.std()
        return np.sqrt(trading_days) * r.mean() / downside_std if downside_std != 0 else np.nan

    rows = []

    for year, yearly_ret in ret_df.groupby(ret_df.index.year):
        yearly_wealth = wealth_df.loc[yearly_ret.index]
        row = {"Year": year}

        for col in ret_df.columns:
            r = yearly_ret[col]
            w = yearly_wealth[col]

            row[f"{col}_Return"] = w.iloc[-1] / w.iloc[0] - 1
            row[f"{col}_Sharpe"] = (
                np.sqrt(trading_days) * r.mean() / r.std()
                if r.std() != 0 else np.nan
            )
            row[f"{col}_Sortino"] = sortino_ratio(r)
            row[f"{col}_MaxDrawdown"] = (w / w.cummax() - 1).min()

        rows.append(row)

    return pd.DataFrame(rows).set_index("Year")


def calculate_bubble_score_proxy(price, ma_window=252, z_window=252):
    log_price = np.log(price)
    fair_value = price.rolling(ma_window).mean()
    log_fair_value = np.log(fair_value)
    residual = log_price - log_fair_value
    z = (residual - residual.rolling(z_window).mean()) / residual.rolling(z_window).std()
    return np.tanh(z / 2)


def get_vix_col(close):
    if "^VIX" in close.columns:
        return "^VIX"
    if "VIX" in close.columns:
        return "VIX"
    return None


def synthetic_uvxy_return_from_vix(vix_ret, date):
    if date < pd.Timestamp("2018-02-28"):
        leverage = 2.0
        decay = 0.0020
    else:
        leverage = 1.5
        decay = 0.0015
    return leverage * vix_ret - decay - 0.25 * (vix_ret ** 2)


# Wrapper class
class DataContainer:
    def __init__(self, close_data):
        self.Close = close_data

    def __getitem__(self, key):
        if key == "Close":
            return self.Close
        raise KeyError(f"Key {key} not found")


# Prepare data
df_data = DataContainer(close_data.copy())

print("\nRunning comprehensive strategy with grid search...")

# Strategy parameters
lookback = 140
holding_period = 40
top = 5
trading_days = 252

close = df_data["Close"].copy()

# Calculate returns
ret_daily_df = close.pct_change().ffill().fillna(0)
ret_df_mom = close.pct_change(lookback).ffill().fillna(0)

print(f"Generating momentum signals ({lookback}-day lookback, {holding_period}-day hold)...")

strategy_returns = []
for i in range(lookback + 1, len(ret_df_mom), holding_period):
    ranking = ret_df_mom.iloc[i - 1:i].rank(axis=1, ascending=False)
    ranked_idx = np.argsort(ranking.values[0])

    short_num = (
        ret_df_mom.iloc[:, ranked_idx[:top]]
        .iloc[i - 1:i]
        .lt(0.0)
        .sum()
        .sum()
    )
    long_num = top - short_num

    if long_num <= 0:
        continue

    idx = min(i + holding_period, len(ret_df_mom))

    for j in range(i, idx):
        date = ret_daily_df.index[j]

        long_part = 0
        short_part = 0

        if long_num > 0:
            long_signal = (
                np.sign(ret_df_mom.iloc[:, ranked_idx[:long_num]].iloc[i - 1:i]).abs()
            )
            long_ret = long_signal.mul(
                np.array(ret_daily_df.iloc[:, ranked_idx[:long_num]].iloc[j:j + 1])[0]
            )
            long_part = long_ret.values.mean() * long_num

        if short_num > 0:
            short_signal = (
                np.sign(ret_df_mom.iloc[:, ranked_idx[-short_num:]].iloc[i - 1:i]).abs() * -1
            )
            short_ret = short_signal.mul(
                np.array(ret_daily_df.iloc[:, ranked_idx[-short_num:]].iloc[j:j + 1])[0]
            )
            short_part = short_ret.values.mean() * short_num

        mom_daily_ret = (long_part + short_part) / top
        mom_daily_ret = mom_daily_ret - 0.005 / holding_period

        hedge_ret = 0.0
        hedge_source = "NONE"

        if "UVXY" in close.columns and pd.notna(close.loc[date, "UVXY"]):
            hedge_ret = ret_daily_df.loc[date, "UVXY"]
            hedge_source = "UVXY"

        vix_col = get_vix_col(close)
        if vix_col and pd.notna(close.loc[date, vix_col]):
            vix_ret = ret_daily_df.loc[date, vix_col]
            hedge_ret = synthetic_uvxy_return_from_vix(vix_ret, date)
            hedge_source = "SYNTHETIC_VIX"

        strategy_returns.append({
            "Date": date,
            "Momentum": mom_daily_ret,
            "QQQ_BuyHold": ret_daily_df.loc[date, "QQQ"] if "QQQ" in close.columns else 0,
            "SPY_BuyHold": ret_daily_df.loc[date, "SPY"] if "SPY" in close.columns else 0,
            "Hedge_Return": hedge_ret,
            "Hedge_Source": hedge_source,
        })

ret_df = pd.DataFrame(strategy_returns).set_index("Date")
hedge_source_series = ret_df["Hedge_Source"].copy()
ret_df = ret_df.drop(columns=["Hedge_Source"]).dropna()

print(f"Generated {len(ret_df)} daily signals")

# Create bubble score
print("Calculating bubble score...")
base_wealth = (1 + ret_df).cumprod()
base_wealth = base_wealth / base_wealth.iloc[0]

if "QQQ" in close.columns:
    indicator_price = close.loc[ret_df.index, "QQQ"]
else:
    indicator_price = base_wealth["Momentum"]

bubble_score = calculate_bubble_score_proxy(indicator_price, ma_window=252, z_window=252)

print("\n" + "="*100)
print("GRID SEARCH: Testing Parameter Combinations")
print("="*100)

# Grid parameters
low_bubble_entry_grid = [-0.90]
momentum_extra_leverage_grid = [0.25, 0.30, 0.35]
leverage_hold_days_grid = [40, 50, 60]
hedge_bubble_entry_grid = [0.85]
hedge_alloc_grid = [0.30, 0.40, 0.50]
hedge_hold_days_grid = [30, 40, 50]

total_combos = (len(low_bubble_entry_grid) * len(momentum_extra_leverage_grid) *
                len(leverage_hold_days_grid) * len(hedge_bubble_entry_grid) *
                len(hedge_alloc_grid) * len(hedge_hold_days_grid))

print(f"Total combinations: {total_combos}")

grid_results = []
daily_leverage_cost = 0.10 / trading_days
best_sharpe = -np.inf
best_params = None

combo_count = 0

for (low_bubble_entry, momentum_extra_leverage, leverage_hold_days,
     hedge_bubble_entry, hedge_alloc, hedge_hold_days) in product(
    low_bubble_entry_grid,
    momentum_extra_leverage_grid,
    leverage_hold_days_grid,
    hedge_bubble_entry_grid,
    hedge_alloc_grid,
    hedge_hold_days_grid,
):
    combo_count += 1

    if combo_count % 10 == 0:
        print(f"  Testing {combo_count}/{total_combos}...", flush=True)

    # Signals
    raw_hedge_signal = bubble_score > hedge_bubble_entry
    hedge_trade_signal = raw_hedge_signal.shift(1).fillna(False)

    raw_leverage_signal = bubble_score < low_bubble_entry
    leverage_trade_signal = raw_leverage_signal.shift(1).fillna(False)

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
        uvxy_hedge_component = 0.0
        leveraged_momentum_component = 0.0
        leverage_cost_component = 0.0

        if hedge_remaining_days > 0:
            momentum_component = (1.0 - hedge_alloc) * base_momentum
            uvxy_hedge_component = hedge_alloc * ret_df.loc[date, "Hedge_Return"]
            daily_ret = momentum_component + uvxy_hedge_component
            hedge_remaining_days -= 1

        elif leverage_remaining_days > 0:
            leveraged_momentum_component = momentum_extra_leverage * base_momentum
            leverage_cost_component = -momentum_extra_leverage * daily_leverage_cost
            daily_ret = momentum_component + leveraged_momentum_component + leverage_cost_component
            leverage_remaining_days -= 1

        else:
            daily_ret = momentum_component

        strategy_ret.append(daily_ret)

    temp_ret = pd.Series(strategy_ret, index=ret_df.index)
    temp_wealth = (1 + temp_ret).cumprod() / ((1 + temp_ret).cumprod().iloc[0])

    temp_std = temp_ret.std()
    temp_sharpe = np.sqrt(trading_days) * temp_ret.mean() / temp_std if temp_std != 0 else np.nan

    downside = temp_ret[temp_ret < 0]
    downside_std = downside.std()
    temp_sortino = np.sqrt(trading_days) * temp_ret.mean() / downside_std if downside_std != 0 else np.nan

    temp_return = temp_wealth.iloc[-1] - 1
    temp_mdd = (temp_wealth / temp_wealth.cummax() - 1).min()

    grid_results.append({
        'Low_Bubble_Entry': low_bubble_entry,
        'Leverage_Multiplier': 1.0 + momentum_extra_leverage,
        'Leverage_Hold_Days': leverage_hold_days,
        'High_Bubble_Entry': hedge_bubble_entry,
        'UVXY_Allocation': hedge_alloc,
        'UVXY_Hold_Days': hedge_hold_days,
        'Total Return': temp_return,
        'Annual Return': (1 + temp_return) ** (1/29.42) - 1,
        'Sharpe Ratio': temp_sharpe,
        'Sortino Ratio': temp_sortino,
        'Max Drawdown': temp_mdd,
    })

    if pd.notna(temp_sharpe) and temp_sharpe > best_sharpe:
        best_sharpe = temp_sharpe
        best_params = {
            'Low_Bubble_Entry': low_bubble_entry,
            'Leverage_Multiplier': 1.0 + momentum_extra_leverage,
            'Leverage_Hold_Days': leverage_hold_days,
            'High_Bubble_Entry': hedge_bubble_entry,
            'UVXY_Allocation': hedge_alloc,
            'UVXY_Hold_Days': hedge_hold_days,
        }

grid_df = pd.DataFrame(grid_results)
grid_df = grid_df.sort_values('Sharpe Ratio', ascending=False)

print(f"\nCompleted {combo_count} combinations")

print("\n" + "="*100)
print("TOP RESULTS (Sorted by Sharpe Ratio)")
print("="*100)

print("\nTop 10 by Sharpe Ratio:")
print(grid_df.head(10)[['Leverage_Multiplier', 'Low_Bubble_Entry', 'UVXY_Allocation',
                         'High_Bubble_Entry', 'Total Return', 'Annual Return',
                         'Sharpe Ratio', 'Max Drawdown']].to_string(index=False))

print("\n" + "="*100)
print("OPTIMAL PARAMETERS (Best Sharpe Ratio)")
print("="*100)

best = grid_df.iloc[0]

print(f"""
OPTIMAL STRATEGY:

LOW BUBBLE LEVERAGE:
  Entry Threshold: {best['Low_Bubble_Entry']}
  Leverage Multiplier: {best['Leverage_Multiplier']:.2f}x
  Hold Days: {int(best['Leverage_Hold_Days'])}

HIGH BUBBLE UVXY HEDGE:
  Entry Threshold: {best['High_Bubble_Entry']}
  UVXY Allocation: {best['UVXY_Allocation']:.0%}
  Hold Days: {int(best['UVXY_Hold_Days'])}

PERFORMANCE (1997-2026):
  Total Return: {best['Total Return']:.2%}
  Annual Return: {best['Annual Return']:.2%}
  Sharpe Ratio: {best['Sharpe Ratio']:.4f}
  Sortino Ratio: {best['Sortino Ratio']:.4f}
  Max Drawdown: {best['Max Drawdown']:.2%}

INTERPRETATION:
  - Combines daily momentum with tactical overlays
  - Low bubble leverage amplifies recovery periods
  - High bubble UVXY hedges tail risk
  - Conservative parameters for stable returns
""")

# Save results
os.makedirs('results', exist_ok=True)
grid_df.to_csv('results/comprehensive_momentum_leverage_uvxy_results.csv', index=False)
print(f"\nSaved results: results/comprehensive_momentum_leverage_uvxy_results.csv")

print("\n" + "="*100)
print("STRATEGY COMPLETE - OPTIMAL PARAMETERS IDENTIFIED")
print("="*100)
