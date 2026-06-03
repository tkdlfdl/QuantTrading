"""
QQQ Hourly Bubble Score Strategy
==================================
Apply bubble score proxy directly to QQQ hourly close prices.

Signal (no lookahead — score at bar T uses data through T, signal fires at T+1 open):
  bubble_score < -threshold  →  LONG  QQQ at next bar open, hold X hours
  bubble_score > +threshold  →  SHORT QQQ at next bar open, hold X hours

Bubble score formula (same as reddit strategy):
  residual  = log(close) − log(rolling_mean(close, ma_window))
  z_score   = (residual − mean(residual, z_window)) / std(residual, z_window)
  bubble    = tanh(z_score / 2)   ← bounded (-1, +1)

Grid:
  ma_window   : MA window in hours (fair value)
  z_window    : z-score normalization window in hours
  threshold   : extreme low threshold (long entry)
  hold_hours  : how many bars to hold after entry
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import product

TRADING_HOURS_PER_YEAR = 252 * 6.5   # approx annual trading hours


def calculate_bubble_score(close: pd.Series, ma_window: int, z_window: int) -> pd.Series:
    close       = close.replace(0, np.nan).ffill()
    log_close   = np.log(close)
    fair_value  = close.rolling(ma_window).mean()
    residual    = log_close - np.log(fair_value)
    z           = (residual - residual.rolling(z_window).mean()) / residual.rolling(z_window).std()
    return np.tanh(z / 2)


def run_qqq_bubble_hourly(
    hourly_open:  pd.Series,                    # QQQ hourly open prices
    hourly_close: pd.Series,                    # QQQ hourly close prices
    ma_window_grid:  list = [20, 50, 100],      # hours for rolling MA
    z_window_grid:   list = [50, 100, 200],     # hours for z-score window
    threshold_grid:  list = [0.5, 0.6, 0.7, 0.8, 0.9],  # entry threshold
    hold_hours_grid: list = [1, 2, 4, 8, 24],  # bars to hold after entry
    transaction_cost: float = 0.001,            # 0.1% round-trip per trade
    short_borrow_rate: float = 0.08,            # 8%/yr borrowing cost on short positions
    enable_short: bool = True,                  # trade short side (score > +threshold)
) -> tuple[pd.Series, dict, pd.DataFrame]:
    """
    Returns
    -------
    best_daily_ret : daily return Series (trade P&L attributed to entry date)
    best_params    : dict of best parameters + metrics
    grid_df        : full grid results sorted by Sharpe
    """
    assert len(hourly_open) == len(hourly_close), "open/close must be same length"
    assert hourly_open.index.equals(hourly_close.index), "indexes must match"

    ho = hourly_open.copy()
    hc = hourly_close.copy()
    n  = len(ho)

    total = len(ma_window_grid) * len(z_window_grid) * len(threshold_grid) * len(hold_hours_grid)
    print(f"Grid search: {total} combinations (QQQ hourly bubble)...")

    grid_results = []
    best_sharpe  = -np.inf
    best_daily   = None
    best_params  = None

    # Pre-compute bubble scores for each (ma, z) combo
    score_cache: dict[tuple, pd.Series] = {}
    for ma, z in product(ma_window_grid, z_window_grid):
        score_cache[(ma, z)] = calculate_bubble_score(hc, ma, z)

    for ma, z, thresh, hold in product(
        ma_window_grid, z_window_grid, threshold_grid, hold_hours_grid
    ):
        raw_scores = score_cache[(ma, z)]
        # Shift by 1 bar: signal at bar i uses score from bar i-1 (no lookahead)
        signal_scores = raw_scores.shift(1)

        # Hourly short borrow cost
        hourly_borrow = short_borrow_rate / (252 * 6.5)

        trades: list[dict] = []
        in_trade_until = -1   # bar index — no new trade while in_trade_until > current

        for i in range(1, n - hold):
            if i <= in_trade_until:
                continue

            sig = signal_scores.iloc[i]
            if pd.isna(sig):
                continue

            # Determine trade direction
            if sig < -thresh:
                direction = 1   # LONG
            elif enable_short and sig > thresh:
                direction = -1  # SHORT
            else:
                continue

            # Entry at open of bar i, exit at close of bar i+hold-1
            entry_price = ho.iloc[i]
            exit_idx    = min(i + hold - 1, n - 1)
            exit_price  = hc.iloc[exit_idx]

            if entry_price <= 0 or pd.isna(entry_price) or pd.isna(exit_price):
                continue

            raw_ret    = (exit_price / entry_price - 1) * direction
            borrow_cost = hourly_borrow * hold if direction == -1 else 0.0
            net_ret    = raw_ret - transaction_cost - borrow_cost
            entry_dt   = ho.index[i]
            exit_dt    = hc.index[exit_idx]

            trades.append({
                "entry_bar":  i,
                "exit_bar":   exit_idx,
                "entry_dt":   entry_dt,
                "exit_dt":    exit_dt,
                "entry_price":entry_price,
                "exit_price": exit_price,
                "direction":  "LONG" if direction == 1 else "SHORT",
                "raw_ret":    raw_ret,
                "borrow":     borrow_cost,
                "net_ret":    net_ret,
                "score":      sig,
            })
            in_trade_until = exit_idx   # no overlapping trades

        if len(trades) < 5:
            grid_results.append(dict(
                ma_window=ma, z_window=z, threshold=thresh, hold_hours=hold,
                Sharpe=np.nan, Sortino=np.nan, Total_Return=np.nan,
                Max_DD=np.nan, n_trades=len(trades), Win_Rate=np.nan,
            ))
            continue

        # Build daily return series (attribute trade P&L to entry date)
        tdf       = pd.DataFrame(trades)
        tdf["date"] = tdf["entry_dt"].dt.normalize()
        daily     = tdf.groupby("date")["net_ret"].sum()

        # Extend through end of hourly data so idle days beyond last signal are included
        data_end  = hc.index[-1].normalize()
        all_dates = pd.date_range(daily.index.min(), data_end, freq="B")
        daily_full = daily.reindex(all_dates, fill_value=0.0)

        wealth = (1 + daily_full).cumprod(); wealth = wealth / wealth.iloc[0]
        mdd    = float((wealth / wealth.cummax() - 1).min())
        sh     = float(np.sqrt(252) * daily_full.mean() / daily_full.std()) if daily_full.std() > 0 else np.nan
        ds     = daily_full[daily_full < 0].std()
        so     = float(np.sqrt(252) * daily_full.mean() / ds) if ds > 0 else np.nan
        wr     = float((tdf["net_ret"] > 0).mean())
        tot    = float(wealth.iloc[-1] - 1)

        n_long  = int((tdf["direction"] == "LONG").sum())
        n_short = int((tdf["direction"] == "SHORT").sum())
        row = dict(
            ma_window=ma, z_window=z, threshold=thresh, hold_hours=hold,
            Sharpe=sh, Sortino=so, Total_Return=tot, Max_DD=mdd,
            n_trades=len(trades), n_long=n_long, n_short=n_short,
            Win_Rate=wr, Avg_Net_Ret=float(tdf["net_ret"].mean()),
        )
        grid_results.append(row)

        if pd.notna(sh) and sh > best_sharpe:
            best_sharpe = sh
            best_daily  = daily_full.rename("QQQ_Bubble")
            best_params = row

    grid_df = pd.DataFrame(grid_results).sort_values("Sharpe", ascending=False)

    if best_params:
        print(f"\nBest Parameters:")
        for k, v in best_params.items():
            print(f"  {k:<18} {v}")

    return best_daily, best_params, grid_df
