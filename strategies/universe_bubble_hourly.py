"""
Universe Hourly Bubble Score Strategy — Long/Short
====================================================
For each stock in SP500+NASDAQ100:
  Compute bubble score on hourly close prices.

Signal (rebalance every hold_hours bars, no-lookahead):
  Lowest bubble scores  → LONG  (extreme oversold)
  Highest bubble scores → SHORT (extreme overbought)

Portfolio:
  Equal-weight within each side.
  If both sides active: 50% weight each.

Grid:
  ma_window   : MA window in hours (fair value baseline)
  z_window    : z-score normalization window in hours
  threshold   : minimum |score| to enter (filters weak signals)
  hold_hours  : rebalancing frequency / hold period
  top_n       : stocks per side
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import product

TRADING_DAYS = 252


def _bubble_scores_matrix(
    close: pd.DataFrame,   # bars × tickers
    ma_window: int,
    z_window: int,
) -> pd.DataFrame:
    """Vectorized bubble score for all tickers at once."""
    log_c    = np.log(close.replace(0, np.nan).ffill())
    fair     = close.rolling(ma_window).mean()
    residual = log_c - np.log(fair)
    z = ((residual - residual.rolling(z_window).mean())
         / residual.rolling(z_window).std())
    return np.tanh(z / 2)


def run_universe_bubble_hourly(
    hourly_open:  pd.DataFrame,       # bars × tickers
    hourly_close: pd.DataFrame,       # bars × tickers
    ma_window_grid:  list = [20, 50, 100],
    z_window_grid:   list = [50, 100, 200],
    threshold_grid:  list = [0.5, 0.6, 0.7, 0.8, 0.9],
    hold_hours_grid: list = [1, 2, 4, 8],
    top_n_grid:      list = [5, 10, 20],
    transaction_cost: float = 0.001,
    short_borrow_rate: float = 0.08,
) -> tuple[pd.Series, dict, pd.DataFrame]:
    """
    Returns
    -------
    best_daily_ret : daily return Series
    best_params    : dict with best parameters + metrics
    grid_df        : full grid sorted by Sharpe
    """
    ho = hourly_open.copy()
    hc = hourly_close.copy()
    tickers = hc.columns.tolist()
    n = len(hc)

    hourly_borrow = short_borrow_rate / (TRADING_DAYS * 6.5)

    total = (len(ma_window_grid) * len(z_window_grid) * len(threshold_grid)
             * len(hold_hours_grid) * len(top_n_grid))
    print(f"Universe: {len(tickers)} tickers  |  {n} hourly bars")
    print(f"Pre-computing bubble scores for {len(ma_window_grid)*len(z_window_grid)} "
          f"(ma, z) combos...")

    # Pre-compute and shift bubble score matrices (no-lookahead)
    score_cache: dict[tuple, pd.DataFrame] = {}
    for ma, z in product(ma_window_grid, z_window_grid):
        raw = _bubble_scores_matrix(hc, ma, z)
        score_cache[(ma, z)] = raw.shift(1)   # shift: signal known at bar i uses score from i-1
    print("Done. Starting grid search...")

    grid_results = []
    best_sharpe  = -np.inf
    best_daily   = None
    best_params  = None

    for ma, z, thresh, hold, top_n in product(
        ma_window_grid, z_window_grid, threshold_grid, hold_hours_grid, top_n_grid
    ):
        scores = score_cache[(ma, z)]
        borrow = hourly_borrow * hold   # total borrow cost per short trade

        trade_rets: list[tuple] = []   # (bar_idx, port_return)

        # Rebalance every hold_hours bars (non-overlapping)
        for i in range(max(ma, z) + 1, n - hold, hold):
            sig = scores.iloc[i].dropna()
            if sig.empty:
                continue

            long_cands  = sig[sig < -thresh].nsmallest(top_n)
            short_cands = sig[sig >  thresh].nlargest(top_n)

            has_long  = len(long_cands)  > 0
            has_short = len(short_cands) > 0
            if not has_long and not has_short:
                continue

            weight = 0.5 if (has_long and has_short) else 1.0
            exit_i = min(i + hold - 1, n - 1)

            pos_rets = []

            if has_long:
                for tkr in long_cands.index:
                    ep = ho.iat[i,      ho.columns.get_loc(tkr)]
                    xp = hc.iat[exit_i, hc.columns.get_loc(tkr)]
                    if pd.isna(ep) or pd.isna(xp) or ep <= 0:
                        continue
                    pos_rets.append(((xp/ep - 1) - transaction_cost) * weight)

            if has_short:
                for tkr in short_cands.index:
                    ep = ho.iat[i,      ho.columns.get_loc(tkr)]
                    xp = hc.iat[exit_i, hc.columns.get_loc(tkr)]
                    if pd.isna(ep) or pd.isna(xp) or ep <= 0:
                        continue
                    pos_rets.append((-(xp/ep - 1) - transaction_cost - borrow) * weight)

            if pos_rets:
                trade_rets.append((i, float(np.mean(pos_rets))))

        if len(trade_rets) < 5:
            grid_results.append(dict(
                ma_window=ma, z_window=z, threshold=thresh,
                hold_hours=hold, top_n=top_n,
                Sharpe=np.nan, Sortino=np.nan, Total_Return=np.nan,
                Max_DD=np.nan, n_trades=len(trade_rets), Win_Rate=np.nan,
            ))
            continue

        # Build daily return series
        rows = [(hc.index[i].normalize(), r) for i, r in trade_rets]
        daily_s = (pd.DataFrame(rows, columns=["date","ret"])
                   .groupby("date")["ret"].sum())
        data_end  = hc.index[-1].normalize()
        all_dates  = pd.date_range(daily_s.index.min(), data_end, freq="B")
        daily_full = daily_s.reindex(all_dates, fill_value=0.0)

        wealth = (1 + daily_full).cumprod(); wealth = wealth / wealth.iloc[0]
        std    = daily_full.std()
        sh     = float(np.sqrt(TRADING_DAYS) * daily_full.mean() / std) if std > 0 else np.nan
        ds     = daily_full[daily_full < 0].std()
        so     = float(np.sqrt(TRADING_DAYS) * daily_full.mean() / ds) if ds > 0 else np.nan
        mdd    = float((wealth / wealth.cummax() - 1).min())
        wr     = float(sum(r > 0 for _, r in trade_rets) / len(trade_rets))

        row = dict(
            ma_window=ma, z_window=z, threshold=thresh,
            hold_hours=hold, top_n=top_n,
            Sharpe=sh, Sortino=so,
            Total_Return=float(wealth.iloc[-1] - 1),
            Max_DD=mdd, n_trades=len(trade_rets), Win_Rate=wr,
        )
        grid_results.append(row)

        if pd.notna(sh) and sh > best_sharpe:
            best_sharpe = sh
            best_daily  = daily_full.rename("UniverseBubble")
            best_params = row

    grid_df = pd.DataFrame(grid_results).sort_values("Sharpe", ascending=False)

    if best_params:
        print(f"\nBest Parameters:")
        for k, v in best_params.items():
            print(f"  {k:<18} {v}")

    return best_daily, best_params, grid_df
