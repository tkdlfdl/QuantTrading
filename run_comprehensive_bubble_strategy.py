"""
COMPREHENSIVE BUBBLE-BASED TRADING STRATEGY
=============================================
Run the full momentum + hedge + leverage strategy with grid search

Strategy:
- Base: Daily momentum (140-day lookback, 40-day hold)
- High Bubble (>0.85): Add UVXY/VIX hedge (50% allocation, 40 days)
- Low Bubble (<-0.9): Add momentum leverage (+30%, 50 days)
- Grid search: 3 x 1 x 1 x 1 x 1 x 1 x 3 x 3 x 5 = 135 combinations
"""

import sys, warnings, os
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

# ============================================================================
# HELPER FUNCTIONS (from provided code)
# ============================================================================

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


# ============================================================================
# MAIN STRATEGY FUNCTION
# ============================================================================

def run_strategy(
    df,
    lookback=140,
    holding_period=40,
    LongShort_flag=True,
    top=5,
    bar=0.0,
    trading_days=252,
    bubble_indicator_grid=["QQQ", "SPY", "Momentum"],
    ma_window_grid=[120],
    z_window_grid=[240],
    hedge_bubble_entry_grid=[0.85],
    hedge_alloc_grid=[0.5],
    hedge_hold_days_grid=[40],
    low_bubble_entry_grid=[-0.9, -0.89, -0.88],
    momentum_extra_leverage_grid=[0.25, 0.3, 0.35],
    leverage_hold_days_grid=[30, 40, 50, 55, 60],
    leverage_cost_annual=0.10,
):
    print("\n" + "="*100)
    print("LOADING DATA & CALCULATING MOMENTUM")
    print("="*100)

    close = df["Close"].copy()
    print(f"Data shape: {close.shape}")
    print(f"Columns: {close.columns.tolist()[:10]}...")

    vix_col = get_vix_col(close)
    has_uvxy = "UVXY" in close.columns
    has_vix = vix_col is not None

    if has_uvxy and has_vix:
        hedge_mode = "HYBRID"
    elif has_uvxy:
        hedge_mode = "UVXY_ONLY"
    elif has_vix:
        hedge_mode = "SYNTHETIC_VIX"
    else:
        hedge_mode = "NO_HEDGE"

    print(f"Hedge mode: {hedge_mode}")
    print(f"Has UVXY: {has_uvxy}, Has VIX: {has_vix}")

    ret_daily_df = close.pct_change().ffill().fillna(0)
    ret_df_mom = close.pct_change(lookback).ffill().fillna(0)

    print(f"Daily returns: {ret_daily_df.shape}")
    print(f"Momentum returns ({lookback}-day): {ret_df_mom.shape}")

    strategy_returns = []

    print("\nGenerating momentum signals...")
    for i in range(lookback + 1, len(ret_df_mom), holding_period):
        ranking = ret_df_mom.iloc[i - 1:i].rank(axis=1, ascending=False)
        ranked_idx = np.argsort(ranking.values[0])

        if LongShort_flag:
            short_num = (
                ret_df_mom.iloc[:, ranked_idx[:top]]
                .iloc[i - 1:i]
                .lt(bar)
                .sum()
                .sum()
            )
            long_num = top - short_num
        else:
            short_num = 0
            long_num = top

        idx = min(i + holding_period, len(ret_df_mom))

        for j in range(i, idx):
            date = ret_daily_df.index[j]

            long_part = 0
            short_part = 0

            if long_num > 0:
                long_signal = np.sign(ret_df_mom.iloc[:, ranked_idx[:long_num]].iloc[i - 1:i]).abs()
                long_ret = long_signal.mul(np.array(ret_daily_df.iloc[:, ranked_idx[:long_num]].iloc[j:j + 1])[0])
                long_part = np.mean(long_ret, axis=1)[0] * long_num

            if LongShort_flag and short_num > 0:
                short_signal = np.sign(ret_df_mom.iloc[:, ranked_idx[-short_num:]].iloc[i - 1:i]).abs() * -1
                short_ret = short_signal.mul(np.array(ret_daily_df.iloc[:, ranked_idx[-short_num:]].iloc[j:j + 1])[0])
                short_part = np.mean(short_ret, axis=1)[0] * short_num

            mom_daily_ret = (long_part + short_part) / top
            mom_daily_ret = mom_daily_ret - 0.005 / holding_period

            hedge_ret = 0.0
            hedge_source = "NONE"

            if has_uvxy and pd.notna(close.loc[date, "UVXY"]):
                hedge_ret = ret_daily_df.loc[date, "UVXY"]
                hedge_source = "UVXY"
            elif has_vix and pd.notna(close.loc[date, vix_col]):
                vix_ret = ret_daily_df.loc[date, vix_col]
                hedge_ret = synthetic_uvxy_return_from_vix(vix_ret, date)
                hedge_source = "SYNTHETIC_VIX"

            strategy_returns.append({
                "Date": date,
                "Momentum": mom_daily_ret,
                "QQQ_BuyHold": ret_daily_df.loc[date, "QQQ"],
                "SPY_BuyHold": ret_daily_df.loc[date, "SPY"],
                "Hedge_Return": hedge_ret,
                "Hedge_Source": hedge_source,
            })

    ret_df = pd.DataFrame(strategy_returns).set_index("Date")
    hedge_source_series = ret_df["Hedge_Source"].copy()
    ret_df = ret_df.drop(columns=["Hedge_Source"]).dropna()

    base_wealth = (1 + ret_df).cumprod()
    base_wealth = base_wealth / base_wealth.iloc[0]

    daily_leverage_cost = leverage_cost_annual / trading_days

    grid_results = []

    print("\n" + "="*100)
    print("GRID SEARCH: TESTING ALL PARAMETER COMBINATIONS")
    print("="*100)

    total_combos = (
        len(bubble_indicator_grid) * len(ma_window_grid) * len(z_window_grid) *
        len(hedge_bubble_entry_grid) * len(hedge_alloc_grid) * len(hedge_hold_days_grid) *
        len(low_bubble_entry_grid) * len(momentum_extra_leverage_grid) * len(leverage_hold_days_grid)
    )
    print(f"Total combinations to test: {total_combos}")

    best_sharpe = -np.inf
    best_ret_series = None
    best_params = None
    combo_count = 0

    for (
        bubble_indicator,
        ma_window,
        z_window,
        hedge_bubble_entry,
        hedge_alloc,
        hedge_hold_days,
        low_bubble_entry,
        momentum_extra_leverage,
        leverage_hold_days,
    ) in product(
        bubble_indicator_grid,
        ma_window_grid,
        z_window_grid,
        hedge_bubble_entry_grid,
        hedge_alloc_grid,
        hedge_hold_days_grid,
        low_bubble_entry_grid,
        momentum_extra_leverage_grid,
        leverage_hold_days_grid,
    ):
        combo_count += 1
        if combo_count % 25 == 0:
            print(f"  Testing {combo_count}/{total_combos}...", flush=True)

        if bubble_indicator == "QQQ":
            indicator_price = close.loc[ret_df.index, "QQQ"]
        elif bubble_indicator == "SPY":
            indicator_price = close.loc[ret_df.index, "SPY"]
        elif bubble_indicator == "Momentum":
            indicator_price = base_wealth["Momentum"]

        bubble_score = calculate_bubble_score_proxy(indicator_price, ma_window=ma_window, z_window=z_window)

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
        temp_wealth = (1 + temp_ret).cumprod()
        temp_wealth = temp_wealth / temp_wealth.iloc[0]

        temp_std = temp_ret.std()
        temp_sharpe = (np.sqrt(trading_days) * temp_ret.mean() / temp_std) if temp_std != 0 else np.nan

        downside = temp_ret[temp_ret < 0]
        downside_std = downside.std()
        temp_sortino = (np.sqrt(trading_days) * temp_ret.mean() / downside_std) if downside_std != 0 else np.nan

        temp_return = temp_wealth.iloc[-1] - 1
        temp_mdd = (temp_wealth / temp_wealth.cummax() - 1).min()

        row = {
            "bubble_indicator": bubble_indicator,
            "hedge_mode": hedge_mode,
            "ma_window": ma_window,
            "z_window": z_window,
            "hedge_bubble_entry": hedge_bubble_entry,
            "hedge_alloc": hedge_alloc,
            "hedge_hold_days": hedge_hold_days,
            "low_bubble_entry": low_bubble_entry,
            "momentum_extra_leverage": momentum_extra_leverage,
            "leverage_hold_days": leverage_hold_days,
            "Total Return": temp_return,
            "Sharpe Ratio": temp_sharpe,
            "Sortino Ratio": temp_sortino,
            "Max Drawdown": temp_mdd,
        }

        grid_results.append(row)

        if pd.notna(temp_sharpe) and temp_sharpe > best_sharpe:
            best_sharpe = temp_sharpe
            best_ret_series = temp_ret
            best_params = row

    grid_result_df = pd.DataFrame(grid_results)
    grid_result_df = grid_result_df.sort_values("Sharpe Ratio", ascending=False)

    print(f"\nCompleted {combo_count} combinations")

    print("\n" + "="*100)
    print("TOP 20 RESULTS")
    print("="*100)
    print(grid_result_df.head(20).to_string())

    print("\n" + "="*100)
    print("BEST PARAMETERS")
    print("="*100)
    print(pd.Series(best_params))

    # Create wealth dataframe with best strategy
    ret_df["Best_Strategy"] = best_ret_series
    wealth_df = (1 + ret_df).cumprod()
    wealth_df = wealth_df / wealth_df.iloc[0]

    analysis = performance_stats(ret_df, wealth_df, trading_days)

    print("\n" + "="*100)
    print("PERFORMANCE ANALYSIS")
    print("="*100)
    print(analysis)

    return analysis, grid_result_df, wealth_df, ret_df, best_params


# ============================================================================
# LOAD DATA & RUN STRATEGY
# ============================================================================

print("\n" + "="*100)
print("COMPREHENSIVE BUBBLE-BASED TRADING STRATEGY")
print("="*100)

# Load daily data
print("\nLoading daily OHLC data...")
try:
    daily_close = pd.read_parquet("data/cache/daily_close.parquet")
    print(f"Loaded daily data: {daily_close.shape}")
except:
    print("WARNING: daily_close.parquet not found")
    daily_close = None

if daily_close is None:
    print("CRITICAL: Cannot proceed without daily data")
    sys.exit(1)

# Create data frame for strategy
# The function expects df with df["Close"] = DataFrame of all stock prices
class DataContainer:
    def __init__(self, close_data):
        self.Close = close_data

    def __getitem__(self, key):
        if key == "Close":
            return self.Close
        raise KeyError(f"Key {key} not found")

df_data = DataContainer(daily_close.copy())

# Run strategy with default parameters
analysis, grid_results, wealth_df, ret_df, best_params = run_strategy(
    df_data,
    lookback=140,
    holding_period=40,
    LongShort_flag=True,
    top=5,
    bar=0.0,
    trading_days=252,
    bubble_indicator_grid=["QQQ", "SPY", "Momentum"],
    ma_window_grid=[120],
    z_window_grid=[240],
    hedge_bubble_entry_grid=[0.85],
    hedge_alloc_grid=[0.5],
    hedge_hold_days_grid=[40],
    low_bubble_entry_grid=[-0.9, -0.89, -0.88],
    momentum_extra_leverage_grid=[0.25, 0.3, 0.35],
    leverage_hold_days_grid=[30, 40, 50, 55, 60],
    leverage_cost_annual=0.10,
)

# Save results
print("\nSaving results...")
os.makedirs("results", exist_ok=True)

grid_results.to_csv("results/bubble_strategy_grid_results.csv", index=False)
analysis.to_excel("results/bubble_strategy_analysis.xlsx")

print("\n" + "="*100)
print("COMPLETE")
print("="*100)
print(f"\nBest Strategy Performance:")
print(f"  Sharpe: {best_params['Sharpe Ratio']:.4f}")
print(f"  Return: {best_params['Total Return']:+.2%}")
print(f"  MaxDD: {best_params['Max Drawdown']:.2%}")
print(f"\nFiles saved:")
print(f"  - results/bubble_strategy_grid_results.csv")
print(f"  - results/bubble_strategy_analysis.xlsx")
