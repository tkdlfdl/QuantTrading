"""
RUN WITH EXACT FUNCTION FROM PROVIDED CODE
===========================================
Using the exact run_momentum_bubble_hedge_and_low_bubble_leverage function
This should produce Total Return: 2204644.309377% with Sharpe: 1.409475
"""

import sys, warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product

print("="*120)
print("RUNNING WITH EXACT PROVIDED CODE STRUCTURE")
print("="*120)

# Load data
print("\nLoading extended data (1997-2026)...")
close_data = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
print(f"Loaded: {close_data.shape}")

# Helper functions (EXACT from provided code)
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

    z = (
        residual - residual.rolling(z_window).mean()
    ) / residual.rolling(z_window).std()

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


# Data container (EXACT from provided code)
class DataContainer:
    def __init__(self, close_data):
        self.Close = close_data

    def __getitem__(self, key):
        if key == "Close":
            return self.Close
        raise KeyError(f"Key {key} not found")


df_data = DataContainer(close_data.copy())

# EXACT parameters from code output
print("\n" + "="*120)
print("EXACT PARAMETERS FROM CODE")
print("="*120)

bubble_indicator = "Momentum"
ma_window = 120
z_window = 240
hedge_bubble_entry = 0.85
hedge_alloc = 0.5
hedge_hold_days = 40
low_bubble_entry = -0.88
momentum_extra_leverage = 0.25
leverage_hold_days = 50

print(f"""
bubble_indicator: {bubble_indicator}
ma_window: {ma_window}
z_window: {z_window}
hedge_bubble_entry: {hedge_bubble_entry}
hedge_alloc: {hedge_alloc}
hedge_hold_days: {hedge_hold_days}
low_bubble_entry: {low_bubble_entry}
momentum_extra_leverage: {momentum_extra_leverage}
total_momentum_exposure_when_levered: {1.0 + momentum_extra_leverage}
leverage_hold_days: {leverage_hold_days}
leverage_cost_annual: 0.1

Expected Results:
  Total Return: 2204644.309377%
  Sharpe Ratio: 1.409475
  Sortino Ratio: 2.092009
  Max Drawdown: -0.652802
""")

# Main strategy function (EXACT from provided code)
close = df_data["Close"].copy()

for col in ["QQQ", "SPY"]:
    if col not in close.columns:
        print(f"WARNING: {col} not in data")

vix_col = get_vix_col(close)
has_uvxy = "UVXY" in close.columns
has_vix = vix_col is not None

if has_uvxy and has_vix:
    hedge_mode = "HYBRID_UVXY_SYNTHETIC_VIX"
elif has_uvxy:
    hedge_mode = "UVXY_ONLY"
elif has_vix:
    hedge_mode = "SYNTHETIC_VIX_ONLY"
else:
    hedge_mode = "NO_HEDGE"

print(f"\nHedge mode: {hedge_mode}")
print(f"Has UVXY: {has_uvxy}, Has VIX: {has_vix}")

# Calculate returns (EXACT from provided code)
lookback = 140
holding_period = 40
top = 5
trading_days = 252
bar = 0.0
LongShort_flag = True

ret_daily_df = close.pct_change().ffill().fillna(0)
ret_df_mom = close.pct_change(lookback).ffill().fillna(0)

print(f"\nGenerating momentum signals ({len(ret_df_mom)} rows)...")

strategy_returns = []

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
            long_signal = (
                np.sign(ret_df_mom.iloc[:, ranked_idx[:long_num]].iloc[i - 1:i])
                .abs()
            )

            long_ret = long_signal.mul(
                np.array(
                    ret_daily_df.iloc[:, ranked_idx[:long_num]].iloc[j:j + 1]
                )[0]
            )

            long_part = np.mean(long_ret, axis=1)[0] * long_num

        if LongShort_flag and short_num > 0:
            short_signal = (
                np.sign(ret_df_mom.iloc[:, ranked_idx[-short_num:]].iloc[i - 1:i])
                .abs()
                * -1
            )

            short_ret = short_signal.mul(
                np.array(
                    ret_daily_df.iloc[:, ranked_idx[-short_num:]].iloc[j:j + 1]
                )[0]
            )

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
ret_df = ret_df.drop(columns=["Hedge_Source"]).dropna()

base_wealth = (1 + ret_df).cumprod()
base_wealth = base_wealth / base_wealth.iloc[0]

print(f"Generated {len(ret_df)} daily signals")

# Calculate bubble score
if bubble_indicator == "QQQ":
    indicator_price = close.loc[ret_df.index, "QQQ"]
elif bubble_indicator == "SPY":
    indicator_price = close.loc[ret_df.index, "SPY"]
elif bubble_indicator == "Momentum":
    indicator_price = base_wealth["Momentum"]

bubble_score = calculate_bubble_score_proxy(
    indicator_price,
    ma_window=ma_window,
    z_window=z_window,
)

raw_hedge_signal = bubble_score > hedge_bubble_entry
hedge_trade_signal = raw_hedge_signal.shift(1).fillna(False)

raw_leverage_signal = bubble_score < low_bubble_entry
leverage_trade_signal = raw_leverage_signal.shift(1).fillna(False)

# Run strategy (EXACT from provided code)
print("\nRunning strategy with EXACT parameters...")

daily_leverage_cost = 0.10 / trading_days
strategy_ret = []
hedge_exposure = []
leverage_exposure = []

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

    current_hedge_alloc = 0.0
    current_extra_leverage = 0.0

    if hedge_remaining_days > 0:
        current_hedge_alloc = hedge_alloc

        momentum_component = (
            (1.0 - current_hedge_alloc) * base_momentum
        )

        uvxy_hedge_component = (
            current_hedge_alloc * ret_df.loc[date, "Hedge_Return"]
        )

        daily_ret = momentum_component + uvxy_hedge_component

        hedge_remaining_days -= 1

    elif leverage_remaining_days > 0:
        current_extra_leverage = momentum_extra_leverage

        leveraged_momentum_component = (
            current_extra_leverage * base_momentum
        )

        leverage_cost_component = (
            -current_extra_leverage * daily_leverage_cost
        )

        daily_ret = (
            momentum_component
            + leveraged_momentum_component
            + leverage_cost_component
        )

        leverage_remaining_days -= 1

    else:
        daily_ret = momentum_component

    strategy_ret.append(daily_ret)
    hedge_exposure.append(current_hedge_alloc)
    leverage_exposure.append(current_extra_leverage)

temp_ret = pd.Series(strategy_ret, index=ret_df.index)
temp_wealth = (1 + temp_ret).cumprod()
temp_wealth = temp_wealth / temp_wealth.iloc[0]

# Calculate metrics (EXACT)
temp_std = temp_ret.std()
temp_sharpe = (
    np.sqrt(trading_days) * temp_ret.mean() / temp_std
    if temp_std != 0 else np.nan
)

downside = temp_ret[temp_ret < 0]
downside_std = downside.std()
temp_sortino = (
    np.sqrt(trading_days) * temp_ret.mean() / downside_std
    if downside_std != 0 else np.nan
)

temp_return = temp_wealth.iloc[-1] - 1
temp_mdd = (temp_wealth / temp_wealth.cummax() - 1).min()

print("\n" + "="*120)
print("ACTUAL RESULTS (Using EXACT Code)")
print("="*120)

print(f"""
Total Return: {temp_return:.2%} ({temp_return*100:.2f}%)
Final Wealth: {temp_wealth.iloc[-1]:,.0f}x
Annual Return: {(1 + temp_return) ** (1/29.42) - 1:.2%}
Sharpe Ratio: {temp_sharpe:.6f}
Sortino Ratio: {temp_sortino:.6f}
Max Drawdown: {temp_mdd:.6f}

COMPARISON TO EXPECTED:
Expected Total Return: 2204644.31%
Actual Total Return: {temp_return*100:.2f}%
Expected Sharpe: 1.409475
Actual Sharpe: {temp_sharpe:.6f}
Expected Sortino: 2.092009
Actual Sortino: {temp_sortino:.6f}
""")

if abs(temp_return * 100 - 2204644.31) < 1000:
    print("[OK] MATCH! Results align with expected values")
else:
    print("[INFO] Different result - investigating why...")
    print(f"Difference: {(temp_return * 100) - 2204644.31:.2f}%")

print("\n" + "="*120)
print("ANALYSIS COMPLETE")
print("="*120)

