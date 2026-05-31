"""
Intraday Mean-Reversion Strategy
=================================
Signal (end of day T, no lookahead):
  daily_ret_T > rolling_mean + sigma * rolling_std  → SHORT at T+1 open
  daily_ret_T < rolling_mean - sigma * rolling_std  → LONG  at T+1 open

Execution (day T+1):
  Entry : first hourly bar's open  (market open)
  Exit  : close of bar at index hold_hours-1  (hold_hours after open)

Daily returns before the 2-year hourly window use the long-history daily data,
so the rolling lookback is always well-populated.

Grid parameters:
  sigma       : z-score threshold  [1.5, 2.0, 2.5, 3.0]
  hold_hours  : hours to hold      [1, 2, 4, 8]
  lookback    : rolling window (days) [20, 60, 120]
  top_n       : max positions per side per day [5, 10, 20]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import product

TRADING_DAYS = 252


def _build_day_index(hourly_open: pd.DataFrame) -> dict:
    """Map each calendar date → sorted list of hourly timestamps for that day."""
    day_idx: dict[object, list] = {}
    for ts in hourly_open.index:
        d = ts.date()
        day_idx.setdefault(d, []).append(ts)
    # keep timestamps in order
    for d in day_idx:
        day_idx[d].sort()
    return day_idx


def _sharpe(r: pd.Series) -> float:
    std = r.std()
    return float(np.sqrt(TRADING_DAYS) * r.mean() / std) if std > 0 else np.nan


def _sortino(r: pd.Series) -> float:
    ds = r[r < 0].std()
    return float(np.sqrt(TRADING_DAYS) * r.mean() / ds) if ds > 0 else np.nan


def run_intraday_mean_reversion(
    daily_close: pd.DataFrame,
    hourly_open: pd.DataFrame,
    hourly_close: pd.DataFrame,
    sigma_grid: list      = [1.5, 2.0, 2.5, 3.0],
    hold_hours_grid: list = [1, 2, 4, 8],
    lookback_grid: list   = [20, 60, 120],
    top_n_grid: list      = [5, 10, 20],
    transaction_cost: float = 0.001,
) -> tuple[pd.Series, dict, pd.DataFrame]:
    """
    Returns
    -------
    best_ret   : daily return Series for best parameter combo
    best_params: dict of best parameters + performance metrics
    grid_df    : full grid search results sorted by Sharpe
    """
    # ── Align universes ────────────────────────────────────────────────────
    daily_close.index = pd.to_datetime(daily_close.index)
    common_tickers = [t for t in daily_close.columns
                      if t in hourly_open.columns and t in hourly_close.columns]
    print(f"Common tickers (daily ∩ hourly): {len(common_tickers)}")

    daily_ret = daily_close[common_tickers].pct_change()

    # ── Build hourly day index ─────────────────────────────────────────────
    ho = hourly_open[common_tickers]
    hc = hourly_close[common_tickers]
    day_idx = _build_day_index(ho)
    hourly_trading_days = sorted(day_idx.keys())

    # (signal_date, exec_date) pairs — signal on T, execute on T+1
    trade_pairs = list(zip(hourly_trading_days[:-1], hourly_trading_days[1:]))
    print(f"Hourly window: {hourly_trading_days[0]} to {hourly_trading_days[-1]}")
    print(f"Trade days available: {len(trade_pairs)}\n")

    # Pre-compute rolling stats for all lookbacks at once (faster)
    roll_cache: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for lb in lookback_grid:
        roll_cache[lb] = (
            daily_ret.rolling(lb).mean(),
            daily_ret.rolling(lb).std(),
        )

    # ── Grid search ────────────────────────────────────────────────────────
    total = len(lookback_grid) * len(sigma_grid) * len(hold_hours_grid) * len(top_n_grid)
    print(f"Grid search: {total} combinations...")

    grid_results = []
    best_sharpe  = -np.inf
    best_ret     = None
    best_params  = None

    for lookback, sigma, hold_hours, top_n in product(
        lookback_grid, sigma_grid, hold_hours_grid, top_n_grid
    ):
        roll_mean, roll_std = roll_cache[lookback]

        ret_rows: list[tuple] = []  # (exec_date, daily_return)

        for sig_date, exec_date in trade_pairs:
            sig_ts = pd.Timestamp(sig_date)
            if sig_ts not in daily_ret.index:
                continue

            r  = daily_ret.loc[sig_ts]
            mu = roll_mean.loc[sig_ts]
            sd = roll_std.loc[sig_ts]

            valid = sd > 0
            z = ((r - mu) / sd).where(valid).dropna()

            # Select candidates
            long_cands  = z[z < -sigma].nsmallest(top_n)    # most negative z → long
            short_cands = z[z >  sigma].nlargest(top_n)     # most positive z → short

            positions: dict[str, int] = {}
            for t in long_cands.index:
                positions[t] = 1
            for t in short_cands.index:
                positions[t] = -1

            if not positions:
                ret_rows.append((exec_date, 0.0))
                continue

            # Hourly execution
            bars = day_idx.get(exec_date, [])
            if not bars:
                ret_rows.append((exec_date, 0.0))
                continue

            entry_ts = bars[0]
            exit_ts  = bars[min(hold_hours - 1, len(bars) - 1)]

            pos_rets = []
            for ticker, direction in positions.items():
                ep = ho.at[entry_ts, ticker]
                xp = hc.at[exit_ts, ticker]
                if pd.isna(ep) or pd.isna(xp) or ep <= 0:
                    continue
                raw_ret = (xp / ep - 1) * direction - transaction_cost
                pos_rets.append(raw_ret)

            daily_port_ret = float(np.mean(pos_rets)) if pos_rets else 0.0
            ret_rows.append((exec_date, daily_port_ret))

        if not ret_rows:
            continue

        exec_dates = [pd.Timestamp(d) for d, _ in ret_rows]
        rets       = [r for _, r in ret_rows]
        ret_series = pd.Series(rets, index=exec_dates, name="IntradayMR")
        ret_series = ret_series[~ret_series.index.duplicated(keep="last")]

        wealth = (1 + ret_series).cumprod()
        wealth = wealth / wealth.iloc[0]
        sh     = _sharpe(ret_series)
        so     = _sortino(ret_series)
        mdd    = float((wealth / wealth.cummax() - 1).min())
        n_tr   = int((ret_series != 0).sum())

        row = dict(
            lookback=lookback, sigma=sigma,
            hold_hours=hold_hours, top_n=top_n,
            Sharpe=sh, Sortino=so,
            Total_Return=float(wealth.iloc[-1] - 1),
            Max_DD=mdd, n_trades=n_tr,
        )
        grid_results.append(row)

        if pd.notna(sh) and sh > best_sharpe:
            best_sharpe = sh
            best_ret    = ret_series
            best_params = row

    grid_df = pd.DataFrame(grid_results).sort_values("Sharpe", ascending=False)

    if best_params:
        print("\nBest Parameters:")
        for k, v in best_params.items():
            print(f"  {k:<15} {v}")

    return best_ret, best_params, grid_df
