"""
Momentum + High-Bubble Hedge + Low-Bubble Leverage strategy.

Expects df["Close"] to be a wide DataFrame with ticker symbols as columns.
Required columns: QQQ, SPY
Optional columns: UVXY, ^VIX (or VIX) — enables hedge modes

Entry point: run_momentum_bubble_hedge_and_low_bubble_leverage()
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

from backtest.metrics import performance_stats, yearly_performance_stats


# ---------------------------------------------------------------------------
# Strategy-specific helpers
# ---------------------------------------------------------------------------

def calculate_bubble_score_proxy(price: pd.Series, ma_window: int = 252, z_window: int = 252) -> pd.Series:
    log_price = np.log(price)
    fair_value = price.rolling(ma_window).mean()
    log_fair_value = np.log(fair_value)
    residual = log_price - log_fair_value
    z = (residual - residual.rolling(z_window).mean()) / residual.rolling(z_window).std()
    return np.tanh(z / 2)


def get_vix_col(close: pd.DataFrame) -> str | None:
    if "^VIX" in close.columns:
        return "^VIX"
    if "VIX" in close.columns:
        return "VIX"
    return None


def synthetic_uvxy_return_from_vix(vix_ret: float, date: pd.Timestamp) -> float:
    if date < pd.Timestamp("2018-02-28"):
        leverage, decay = 2.0, 0.0020
    else:
        leverage, decay = 1.5, 0.0015
    return leverage * vix_ret - decay - 0.25 * (vix_ret ** 2)


# ---------------------------------------------------------------------------
# Main strategy runner
# ---------------------------------------------------------------------------

def run_momentum_bubble_hedge_and_low_bubble_leverage(
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
    close = df["Close"].copy()

    for col in ["QQQ", "SPY"]:
        if col not in close.columns:
            raise ValueError(f"df['Close'] must contain {col}.")

    for indicator in bubble_indicator_grid:
        if indicator not in ["QQQ", "SPY", "Momentum"]:
            raise ValueError(
                "bubble_indicator_grid can only contain 'QQQ', 'SPY', or 'Momentum'."
            )

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

    print("Hedge mode:", hedge_mode)
    print("VIX column:", vix_col)
    print("Has UVXY:", has_uvxy)

    ret_daily_df = close.pct_change().ffill().fillna(0)
    ret_df_mom = close.pct_change(lookback).ffill().fillna(0)

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
                    np.array(ret_daily_df.iloc[:, ranked_idx[:long_num]].iloc[j:j + 1])[0]
                )
                long_part = np.mean(long_ret, axis=1).iloc[0] * long_num

            if LongShort_flag and short_num > 0:
                short_signal = (
                    np.sign(ret_df_mom.iloc[:, ranked_idx[-short_num:]].iloc[i - 1:i])
                    .abs()
                    * -1
                )
                short_ret = short_signal.mul(
                    np.array(ret_daily_df.iloc[:, ranked_idx[-short_num:]].iloc[j:j + 1])[0]
                )
                short_part = np.mean(short_ret, axis=1).iloc[0] * short_num

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

    best_sharpe = -np.inf
    best_ret_series = None
    best_return_decomposition = None
    best_decomposition_wealth = None
    best_hedge_exposure = None
    best_leverage_exposure = None
    best_bubble_score = None
    best_hedge_signal = None
    best_leverage_signal = None
    best_params = None

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
                momentum_component = (1.0 - current_hedge_alloc) * base_momentum
                uvxy_hedge_component = current_hedge_alloc * ret_df.loc[date, "Hedge_Return"]
                daily_ret = momentum_component + uvxy_hedge_component
                hedge_remaining_days -= 1

            elif leverage_remaining_days > 0:
                current_extra_leverage = momentum_extra_leverage
                leveraged_momentum_component = current_extra_leverage * base_momentum
                leverage_cost_component = -current_extra_leverage * daily_leverage_cost
                daily_ret = (
                    momentum_component
                    + leveraged_momentum_component
                    + leverage_cost_component
                )
                leverage_remaining_days -= 1

            else:
                daily_ret = momentum_component

            strategy_ret.append({
                "Total": daily_ret,
                "Momentum_Component": momentum_component,
                "UVXY_Hedge_Component": uvxy_hedge_component,
                "Leveraged_Momentum_Component": leveraged_momentum_component,
                "Leverage_Cost_Component": leverage_cost_component,
            })

            hedge_exposure.append(current_hedge_alloc)
            leverage_exposure.append(current_extra_leverage)

        temp_ret_df = pd.DataFrame(strategy_ret, index=ret_df.index)
        temp_ret = temp_ret_df["Total"].rename("Momentum_HighBubbleHedge_LowBubbleLeverage")

        temp_wealth = (1 + temp_ret).cumprod()
        temp_wealth = temp_wealth / temp_wealth.iloc[0]

        decomposition_wealth = (1 + temp_ret_df).cumprod()
        decomposition_wealth = decomposition_wealth / decomposition_wealth.iloc[0]

        temp_std = temp_ret.std()
        temp_sharpe = (
            np.sqrt(trading_days) * temp_ret.mean() / temp_std if temp_std != 0 else np.nan
        )

        downside = temp_ret[temp_ret < 0]
        downside_std = downside.std()
        temp_sortino = (
            np.sqrt(trading_days) * temp_ret.mean() / downside_std
            if downside_std != 0 else np.nan
        )

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
            "total_momentum_exposure_when_levered": 1.0 + momentum_extra_leverage,
            "leverage_hold_days": leverage_hold_days,
            "leverage_cost_annual": leverage_cost_annual,
            "Total Return": temp_return,
            "Sharpe Ratio": temp_sharpe,
            "Sortino Ratio": temp_sortino,
            "Max Drawdown": temp_mdd,
            "Final Wealth": temp_wealth.iloc[-1],
        }

        grid_results.append(row)

        if pd.notna(temp_sharpe) and temp_sharpe > best_sharpe:
            best_sharpe = temp_sharpe
            best_ret_series = temp_ret
            best_return_decomposition = temp_ret_df.copy()
            best_decomposition_wealth = decomposition_wealth.copy()
            best_hedge_exposure = pd.Series(
                hedge_exposure, index=ret_df.index, name="UVXY_Hedge_Exposure"
            )
            best_leverage_exposure = pd.Series(
                leverage_exposure, index=ret_df.index, name="Momentum_Extra_Leverage"
            )
            best_hedge_signal = hedge_trade_signal.rename("Hedge_Trade_Signal_Next_Day")
            best_leverage_signal = leverage_trade_signal.rename("Momentum_Leverage_Signal_Next_Day")
            best_bubble_score = bubble_score.rename("BubbleScore_Proxy")
            best_params = row

    grid_result_df = pd.DataFrame(grid_results).sort_values("Sharpe Ratio", ascending=False)

    ret_df["Momentum_HighBubbleHedge_LowBubbleLeverage"] = best_ret_series

    wealth_df = (1 + ret_df).cumprod()
    wealth_df = wealth_df / wealth_df.iloc[0]

    analysis = performance_stats(ret_df, wealth_df, trading_days)
    yearly_analysis = yearly_performance_stats(ret_df, wealth_df, trading_days)

    print("Best Parameters:")
    print(pd.Series(best_params))
    print("\nBest Return Decomposition Sum:")
    print(best_return_decomposition.sum())

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
        5, 1, figsize=(18, 22), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1, 1, 2]}
    )

    for col in wealth_df.columns:
        if col != "Hedge_Return":
            ax1.plot(wealth_df.index, wealth_df[col], label=col, linewidth=2)
    ax1.set_title(
        f"Momentum + High Bubble Hedge + Low Bubble Leverage | Indicator={best_params['bubble_indicator']}"
    )
    ax1.set_ylabel("Cumulative Wealth")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    ax2.plot(
        best_bubble_score.index, best_bubble_score,
        label=f"BubbleScore Proxy: {best_params['bubble_indicator']}"
    )
    ax2.axhline(best_params["hedge_bubble_entry"], linestyle="--", label="High Bubble Hedge Entry")
    ax2.axhline(best_params["low_bubble_entry"], linestyle="--", label="Low Bubble Leverage Entry")
    ax2.axhline(0, linestyle="--", linewidth=1)
    ax2.set_ylim(-1, 1)
    ax2.set_ylabel("BubbleScore")
    ax2.grid(True)
    ax2.legend(loc="upper left")

    ax3.plot(best_hedge_exposure.index, best_hedge_exposure, label="UVXY / Synthetic VIX Hedge Exposure")
    ax3.fill_between(best_hedge_exposure.index, 0, best_hedge_exposure, alpha=0.3)
    ax3.set_ylabel("Hedge Alloc")
    ax3.grid(True)
    ax3.legend(loc="upper left")

    ax4.plot(best_leverage_exposure.index, best_leverage_exposure, label="Extra Momentum Leverage")
    ax4.fill_between(best_leverage_exposure.index, 0, best_leverage_exposure, alpha=0.3)
    ax4.set_ylabel("Extra Lev")
    ax4.grid(True)
    ax4.legend(loc="upper left")

    for col in [
        "Momentum_Component", "UVXY_Hedge_Component",
        "Leveraged_Momentum_Component", "Leverage_Cost_Component", "Total",
    ]:
        ax5.plot(
            best_decomposition_wealth.index, best_decomposition_wealth[col],
            label=col, linewidth=2
        )
    ax5.set_title("Return Decomposition")
    ax5.set_ylabel("Component Wealth")
    ax5.set_xlabel("Date")
    ax5.grid(True)
    ax5.legend(loc="upper left")

    plt.tight_layout()
    plt.show()

    return (
        analysis,
        yearly_analysis,
        wealth_df,
        ret_df,
        grid_result_df,
        best_return_decomposition,
        best_decomposition_wealth,
        best_hedge_exposure,
        best_leverage_exposure,
        best_bubble_score,
        best_hedge_signal,
        best_leverage_signal,
        hedge_source_series,
    )
