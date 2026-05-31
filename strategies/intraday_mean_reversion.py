"""
Intraday Mean-Reversion + Momentum Flip Strategy
==================================================
Phase 1 — Mean-Reversion (first 1 hour):
  daily_ret_T > mean + sigma*std  →  SHORT at T+1 open, hold 1h
  daily_ret_T < mean - sigma*std  →  LONG  at T+1 open, hold 1h

Phase 2 — Momentum flip (after first hour):
  Reverse direction: ex-short → LONG, ex-long → SHORT
  Hold until: end of same day (0) OR next 1/2/3/4 trading days (grid)

Total trade P&L = Phase1 return + Phase2 return − 2×transaction_cost − short_borrow
Short borrow: 8%/yr daily rate
  Phase 1 SHORT (1h): 8% / 252 / 6.5 ≈ 0.005% per hour
  Phase 2 SHORT (Nd): 8% / 252 × N days (0=same-day uses 0.5d)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import product

TRADING_DAYS = 252


def _build_day_index(hourly_open: pd.DataFrame) -> dict:
    day_idx: dict = {}
    for ts in hourly_open.index:
        d = ts.date()
        day_idx.setdefault(d, []).append(ts)
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
    sigma_grid: list           = [3.5, 4.0, 4.5, 5.0],
    flip_hold_days_grid: list  = [0, 1, 2, 3, 4],   # 0=same day EOD
    lookback_grid: list        = [20, 60, 120],
    top_n_grid: list           = [5, 10, 20],
    transaction_cost: float    = 0.001,              # per phase (0.1% each leg)
    short_borrow_rate: float   = 0.08,               # 8% annual borrowing cost on short positions
) -> tuple[pd.Series, dict, pd.DataFrame]:
    """
    Returns
    -------
    best_ret    : daily return Series (total trade P&L on entry date)
    best_params : dict with best parameters + metrics
    grid_df     : full grid sorted by Sharpe
    """
    daily_close.index = pd.to_datetime(daily_close.index)
    common = [t for t in daily_close.columns
              if t in hourly_open.columns and t in hourly_close.columns]
    print(f"Common tickers: {len(common)}")

    daily_ret = daily_close[common].pct_change()
    ho = hourly_open[common]
    hc = hourly_close[common]

    day_idx = _build_day_index(ho)
    trading_days_list = sorted(day_idx.keys())
    day_to_idx = {d: i for i, d in enumerate(trading_days_list)}

    # (signal_date, exec_date) pairs
    trade_pairs = list(zip(trading_days_list[:-1], trading_days_list[1:]))
    print(f"Hourly window: {trading_days_list[0]} to {trading_days_list[-1]}")
    print(f"Trade opportunities: {len(trade_pairs)}\n")

    # Pre-compute rolling stats
    roll_cache: dict[int, tuple] = {}
    for lb in lookback_grid:
        roll_cache[lb] = (
            daily_ret.rolling(lb).mean(),
            daily_ret.rolling(lb).std(),
        )

    total = len(lookback_grid) * len(sigma_grid) * len(flip_hold_days_grid) * len(top_n_grid)
    print(f"Grid search: {total} combinations...")

    grid_results = []
    best_sharpe  = -np.inf
    best_ret     = None
    best_params  = None

    for lookback, sigma, flip_hold, top_n in product(
        lookback_grid, sigma_grid, flip_hold_days_grid, top_n_grid
    ):
        roll_mean, roll_std = roll_cache[lookback]
        ret_rows: list[tuple] = []

        for sig_date, exec_date in trade_pairs:
            sig_ts = pd.Timestamp(sig_date)
            if sig_ts not in daily_ret.index:
                continue

            r  = daily_ret.loc[sig_ts]
            mu = roll_mean.loc[sig_ts]
            sd = roll_std.loc[sig_ts]

            valid = sd > 0
            z = ((r - mu) / sd).where(valid).dropna()

            long_cands  = z[z < -sigma].nsmallest(top_n)
            short_cands = z[z >  sigma].nlargest(top_n)

            positions: dict[str, int] = {}
            for t in long_cands.index:
                positions[t] = 1
            for t in short_cands.index:
                positions[t] = -1

            if not positions:
                ret_rows.append((exec_date, 0.0))
                continue

            # ── Phase 1 bars (1h mean-reversion) ──────────────────────────
            bars_exec = day_idx.get(exec_date, [])
            if len(bars_exec) < 2:
                ret_rows.append((exec_date, 0.0))
                continue

            p1_entry_ts = bars_exec[0]   # open of bar 0
            p1_exit_ts  = bars_exec[0]   # close of bar 0 (end of hour 1)
            p2_entry_ts = bars_exec[1]   # open of bar 1 (flip entry)

            # ── Phase 2 exit (flip direction, variable hold) ───────────────
            exec_i = day_to_idx.get(exec_date)
            if exec_i is None:
                ret_rows.append((exec_date, 0.0))
                continue

            if flip_hold == 0:
                p2_exit_day = exec_date
            else:
                p2_exit_i = exec_i + flip_hold
                if p2_exit_i >= len(trading_days_list):
                    ret_rows.append((exec_date, 0.0))
                    continue
                p2_exit_day = trading_days_list[p2_exit_i]

            bars_p2_exit = day_idx.get(p2_exit_day, [])
            if not bars_p2_exit:
                ret_rows.append((exec_date, 0.0))
                continue
            p2_exit_ts = bars_p2_exit[-1]  # close of last bar (EOD)

            # ── Borrow cost rates ──────────────────────────────────────────
            # Phase 1 is always 1 hour; Phase 2 holds flip_hold days (0 = ~half day)
            TRADING_HOURS   = 6.5
            daily_borrow    = short_borrow_rate / TRADING_DAYS
            hourly_borrow   = daily_borrow / TRADING_HOURS
            p2_hold_days    = flip_hold if flip_hold >= 1 else 0.5

            # ── Compute per-position returns ───────────────────────────────
            pos_rets = []
            for ticker, direction in positions.items():
                try:
                    # Phase 1: mean-reversion (1 hour)
                    ep1 = ho.at[p1_entry_ts, ticker]
                    xp1 = hc.at[p1_exit_ts,  ticker]
                    if pd.isna(ep1) or pd.isna(xp1) or ep1 <= 0:
                        continue
                    # Borrow cost only when Phase 1 is SHORT (direction == -1)
                    p1_borrow = hourly_borrow if direction == -1 else 0.0
                    p1_ret = (xp1 / ep1 - 1) * direction - transaction_cost - p1_borrow

                    # Phase 2: momentum flip (-direction, holds p2_hold_days)
                    ep2 = ho.at[p2_entry_ts, ticker]
                    xp2 = hc.at[p2_exit_ts,  ticker]
                    if pd.isna(ep2) or pd.isna(xp2) or ep2 <= 0:
                        continue
                    # Borrow cost only when Phase 2 is SHORT (-direction == -1, i.e. direction == 1)
                    p2_borrow = daily_borrow * p2_hold_days if direction == 1 else 0.0
                    p2_ret = (xp2 / ep2 - 1) * (-direction) - transaction_cost - p2_borrow

                    pos_rets.append(p1_ret + p2_ret)
                except (KeyError, TypeError):
                    continue

            daily_port_ret = float(np.mean(pos_rets)) if pos_rets else 0.0
            ret_rows.append((exec_date, daily_port_ret))

        if not ret_rows:
            continue

        ret_series = pd.Series(
            [r for _, r in ret_rows],
            index=[pd.Timestamp(d) for d, _ in ret_rows],
            name="IntradayMR",
        ).pipe(lambda s: s[~s.index.duplicated(keep="last")])

        wealth = (1 + ret_series).cumprod()
        wealth = wealth / wealth.iloc[0]
        sh  = _sharpe(ret_series)
        so  = _sortino(ret_series)
        mdd = float((wealth / wealth.cummax() - 1).min())
        n_tr = int((ret_series != 0).sum())

        row = dict(
            lookback=lookback, sigma=sigma,
            flip_hold_days=flip_hold, top_n=top_n,
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
            print(f"  {k:<18} {v}")

    return best_ret, best_params, grid_df
