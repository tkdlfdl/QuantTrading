"""
Portfolio allocation engine.

Takes multiple strategy return series and allocates capital between them
using either fixed weights or momentum-based dynamic allocation.

Two methods:
  1. "fixed"    — grid search over fixed weight combinations
  2. "momentum" — allocate proportionally to each strategy's recent performance,
                  rebalance every hold_period days
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

from backtest.metrics import performance_stats, yearly_performance_stats


def _sharpe(r: pd.Series, trading_days: int = 252) -> float:
    std = r.std()
    return np.sqrt(trading_days) * r.mean() / std if std > 0 else np.nan


def _make_weight_grid(names: list[str], step: float = 0.1) -> list[dict]:
    """Generate all weight combinations that sum to 1.0 for N strategies."""
    if len(names) == 2:
        weights = []
        w = 0.0
        while w <= 1.0 + 1e-9:
            weights.append({names[0]: round(w, 2), names[1]: round(1 - w, 2)})
            w += step
        return weights
    # For 3+ strategies: use a coarser grid
    weights = []
    steps = np.arange(0, 1.0 + step, step).round(2)
    for combo in product(steps, repeat=len(names)):
        if abs(sum(combo) - 1.0) < 1e-9:
            weights.append(dict(zip(names, combo)))
    return weights


def _apply_max_alloc(weights: dict, max_alloc: float) -> dict:
    """
    Cap each strategy weight at max_alloc, redistributing excess
    proportionally to the remaining uncapped strategies. Iterates
    until stable (handles cascading caps).
    """
    if max_alloc >= 1.0:
        return weights
    w = {k: float(v) for k, v in weights.items()}
    for _ in range(len(w) + 1):
        capped   = {k: min(v, max_alloc) for k, v in w.items()}
        excess   = sum(w[k] - capped[k] for k in w)
        if excess < 1e-9:
            return capped
        uncapped = {k: v for k, v in capped.items() if v < max_alloc - 1e-9}
        u_total  = sum(uncapped.values())
        if u_total <= 0:
            return capped
        for k in uncapped:
            capped[k] += excess * capped[k] / u_total
        w = capped
    return w


def _momentum_weights(
    returns: pd.DataFrame,
    lookback: int,
    i: int,
    max_alloc: float = 1.0,
) -> dict[str, float]:
    """
    Compute allocation weights at rebalancing point i based on
    recent cumulative return over [i-lookback, i].
    Positive performance only; if all negative → equal weight.
    max_alloc caps any single strategy's weight.
    """
    window = returns.iloc[max(0, i - lookback):i]
    if len(window) == 0:
        n = len(returns.columns)
        return _apply_max_alloc({col: 1/n for col in returns.columns}, max_alloc)

    cum_ret = (1 + window).prod() - 1
    pos     = cum_ret.clip(lower=0)
    total   = pos.sum()

    if total > 0:
        raw = (pos / total).to_dict()
    else:
        n   = len(returns.columns)
        raw = {col: 1/n for col in returns.columns}

    return _apply_max_alloc(raw, max_alloc)


def run_portfolio_allocation(
    returns_dict: dict[str, pd.Series],
    method: str = "both",              # "fixed", "momentum", or "both"

    # Fixed allocation
    weight_step: float = 0.1,          # granularity of weight grid

    # Momentum allocation
    lookback_grid: list    = [20, 40, 60, 120],
    hold_period_grid: list = [5, 10, 20, 40],
    max_alloc_grid: list   = [1.0],   # max weight per strategy (1.0 = uncapped)

    trading_days: int = 252,
) -> tuple:
    """
    Allocate capital between multiple strategies.

    Args:
        returns_dict: {strategy_name: daily_return_series}
        method: "fixed", "momentum", or "both"

    Returns:
        best_ret, best_wealth, best_weights_ts, grid_df, best_params
    """
    names = list(returns_dict.keys())

    # Align all series to common dates
    ret_df = pd.DataFrame(returns_dict).dropna(how="all").fillna(0)
    ret_df = ret_df.sort_index()
    print(f"Portfolio period: {ret_df.index[0].date()} to {ret_df.index[-1].date()}")
    print(f"Strategies: {names}")
    print(f"Days: {len(ret_df)}\n")

    grid_results = []
    best_sharpe  = -np.inf
    best_ret     = None
    best_wealth  = None
    best_weights_ts = None
    best_params  = None

    # ── Fixed allocation ───────────────────────────────────────────────────
    if method in ("fixed", "both"):
        weight_combos = _make_weight_grid(names, step=weight_step)
        print(f"Fixed: testing {len(weight_combos)} weight combinations...")

        for weights in weight_combos:
            port_ret = sum(ret_df[name] * w for name, w in weights.items())
            wealth   = (1 + port_ret).cumprod()
            wealth   = wealth / wealth.iloc[0]
            sharpe   = _sharpe(port_ret, trading_days)
            downside = port_ret[port_ret < 0].std()
            sortino  = np.sqrt(trading_days) * port_ret.mean() / downside if downside > 0 else np.nan
            mdd      = (wealth / wealth.cummax() - 1).min()

            row = {
                "method":      "fixed",
                "hold_period": "-",
                "lookback":    "-",
                **{f"w_{n}": weights[n] for n in names},
                "Sharpe":      sharpe,
                "Sortino":     sortino,
                "Total Return": wealth.iloc[-1] - 1,
                "Max DD":      mdd,
            }
            grid_results.append(row)

            if pd.notna(sharpe) and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_ret    = port_ret.rename("Portfolio")
                best_wealth = wealth.rename("Portfolio")
                best_weights_ts = pd.DataFrame(
                    [{**weights, "date": ret_df.index[0]}]
                ).set_index("date")
                best_params = row

    # ── Momentum-based allocation ──────────────────────────────────────────
    if method in ("momentum", "both"):
        n_mom = len(lookback_grid) * len(hold_period_grid) * len(max_alloc_grid)
        print(f"Momentum: testing {n_mom} combos "
              f"({len(lookback_grid)} lookback × {len(hold_period_grid)} hold × "
              f"{len(max_alloc_grid)} max_alloc)...")

        for lookback, hold_period, max_alloc in product(lookback_grid, hold_period_grid, max_alloc_grid):
            if lookback >= len(ret_df):
                continue

            port_ret_rows  = []
            weights_rows   = []

            for i in range(lookback, len(ret_df) - 1, hold_period):
                weights    = _momentum_weights(ret_df, lookback, i, max_alloc)
                hold_end   = min(i + hold_period, len(ret_df))
                hold_dates = ret_df.index[i:hold_end]

                weights_rows.append({"date": ret_df.index[i], **weights})

                for date in hold_dates:
                    r = sum(ret_df.loc[date, name] * w for name, w in weights.items())
                    port_ret_rows.append({"date": date, "ret": r})

            if not port_ret_rows:
                continue

            port_ret = pd.DataFrame(port_ret_rows).set_index("date")["ret"]
            port_ret = port_ret[~port_ret.index.duplicated(keep="last")]
            wealth   = (1 + port_ret).cumprod()
            wealth   = wealth / wealth.iloc[0]
            sharpe   = _sharpe(port_ret, trading_days)
            downside = port_ret[port_ret < 0].std()
            sortino  = np.sqrt(trading_days) * port_ret.mean() / downside if downside > 0 else np.nan
            mdd      = (wealth / wealth.cummax() - 1).min()

            row = {
                "method":      "momentum",
                "lookback":    lookback,
                "hold_period": hold_period,
                "max_alloc":   max_alloc,
                **{f"w_{n}": "-" for n in names},
                "Sharpe":      sharpe,
                "Sortino":     sortino,
                "Total Return": wealth.iloc[-1] - 1,
                "Max DD":      mdd,
            }
            grid_results.append(row)

            if pd.notna(sharpe) and sharpe > best_sharpe:
                best_sharpe    = sharpe
                best_ret       = port_ret.rename("Portfolio")
                best_wealth    = wealth.rename("Portfolio")
                best_weights_ts = pd.DataFrame(weights_rows).set_index("date")
                best_params    = row

    grid_df = pd.DataFrame(grid_results).sort_values("Sharpe", ascending=False)

    # ── Performance summary ────────────────────────────────────────────────
    # Include individual strategies for comparison
    all_rets = ret_df.copy()
    all_rets["Portfolio"] = best_ret
    all_wealth = (1 + all_rets).cumprod()
    all_wealth = all_wealth / all_wealth.iloc[0]

    analysis       = performance_stats(all_rets, all_wealth, trading_days)
    yearly_analysis = yearly_performance_stats(all_rets, all_wealth, trading_days)

    print(f"\nBest allocation: {best_params}")

    # ── Chart ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 14),
                             gridspec_kw={"height_ratios": [3, 1.5, 1]})

    for col in all_wealth.columns:
        lw = 3 if col == "Portfolio" else 1.5
        axes[0].plot(all_wealth.index, all_wealth[col], label=col, linewidth=lw)
    axes[0].set_title(
        f"Portfolio Allocation | method={best_params['method']} | "
        f"lookback={best_params.get('lookback','-')} | hold={best_params.get('hold_period','-')}"
    )
    axes[0].set_ylabel("Cumulative Wealth")
    axes[0].legend()
    axes[0].grid(True)

    # Weight history
    if best_weights_ts is not None and isinstance(best_weights_ts.index, pd.DatetimeIndex):
        for col in names:
            if col in best_weights_ts.columns:
                axes[1].step(best_weights_ts.index,
                             best_weights_ts[col].astype(float),
                             label=f"w_{col}", where="post")
    axes[1].set_ylabel("Weights")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()
    axes[1].grid(True)

    dd = (best_wealth / best_wealth.cummax() - 1)
    axes[2].fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
    axes[2].set_ylabel("Drawdown")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    return analysis, yearly_analysis, best_ret, best_wealth, best_weights_ts, grid_df, best_params
