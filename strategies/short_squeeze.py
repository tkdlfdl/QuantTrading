"""
Short Squeeze Momentum Strategy
================================
Universe: NASDAQ100 + S&P500 stocks with historically high short interest.
Signal  : Anomalous intraday upward move (hourly return Z-score > threshold)
          on a stock that is heavily shorted.
Entry   : LONG at open of bar t+1 (no lookahead).
Exit    : Close of bar t + hold_hours.
Costs   : 0.1% round-trip transaction cost + 8%/yr short borrow (for future
          short-side extensions; this version is long-only).

Grid search:
  signal_sigma   - Z-score threshold to trigger entry
  hold_hours     - bars to hold after entry
  lookback_hours - rolling window for Z-score calculation
  top_n          - max concurrent positions per signal bar
  min_ret        - minimum raw hourly return required alongside Z-score

Returns
-------
best_daily_ret : daily return Series (trade P&L on entry date)
best_params    : dict of best params + metrics
grid_df        : full grid sorted by Sharpe
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import product

TRADING_DAYS = 252


def _sharpe(r: pd.Series) -> float:
    s = r.std()
    return float(np.sqrt(TRADING_DAYS) * r.mean() / s) if s > 0 else np.nan


def _sortino(r: pd.Series) -> float:
    ds = r[r < 0].std()
    return float(np.sqrt(TRADING_DAYS) * r.mean() / ds) if ds > 0 else np.nan


def run_short_squeeze(
    hourly_open:  pd.DataFrame,        # shape: [timestamp x ticker]
    hourly_close: pd.DataFrame,        # shape: [timestamp x ticker]
    short_universe: list[str],         # tickers with high short interest (pre-filtered)

    signal_sigma_grid:   list = [2.0, 2.5, 3.0, 3.5],
    hold_hours_grid:     list = [1, 2, 3, 4],
    lookback_hours_grid: list = [20, 40, 80, 130],
    top_n_grid:          list = [5, 10, 20],   # max positions per signal bar
    min_ret_grid:        list = [0.003, 0.005, 0.010],  # min raw return to trigger

    transaction_cost:   float = 0.001,   # 0.1% one-way per trade
) -> tuple[pd.Series, dict, pd.DataFrame]:
    """
    For each signal bar t:
      1. Compute hourly return = close[t] / open[t] - 1  for each stock
      2. Compare to rolling mean/std over lookback_hours
      3. Z-score > signal_sigma AND hourly_ret > min_ret  -> LONG signal
      4. Enter LONG at open[t+1], exit at close[t+hold_hours]
      5. P&L = (exit - entry) / entry - transaction_cost
    """
    # Restrict to universe members present in both dataframes
    universe = [t for t in short_universe
                if t in hourly_open.columns and t in hourly_close.columns]
    print(f"  Universe: {len(universe)} tickers present in hourly cache")
    if not universe:
        return None, None, pd.DataFrame()

    ho = hourly_open[universe].copy()
    hc = hourly_close[universe].copy()
    n  = len(ho)

    # Pre-compute hourly returns (close/open - 1) for each bar
    bar_ret = hc / ho - 1
    bar_ret = bar_ret.replace([np.inf, -np.inf], np.nan)

    # Build bar index: bar i -> date
    bar_dates = ho.index

    total = (len(signal_sigma_grid) * len(hold_hours_grid) *
             len(lookback_hours_grid) * len(top_n_grid) * len(min_ret_grid))
    print(f"  Grid: {total} combinations...")

    grid_results = []
    best_sharpe  = -np.inf
    best_daily   = None
    best_params  = None

    # Pre-compute rolling stats for each lookback
    roll_cache: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for lb in lookback_hours_grid:
        roll_cache[lb] = (
            bar_ret.rolling(lb).mean(),
            bar_ret.rolling(lb).std(),
        )

    for lb, sigma, hold, top_n, min_ret in product(
        lookback_hours_grid, signal_sigma_grid, hold_hours_grid,
        top_n_grid, min_ret_grid
    ):
        roll_mean, roll_std = roll_cache[lb]

        trades: list[dict] = []
        # Track which bars each ticker is locked in a trade (no overlap)
        # Using a simpler approach: track per-ticker when they exit
        ticker_busy_until: dict[str, int] = {}

        for i in range(lb, n - hold - 1):
            # Z-score at bar i using data through bar i (no lookahead)
            ret_i  = bar_ret.iloc[i]
            mu_i   = roll_mean.iloc[i]
            sd_i   = roll_std.iloc[i]

            valid  = (sd_i > 0) & ret_i.notna() & mu_i.notna()
            z_i    = ((ret_i - mu_i) / sd_i).where(valid)

            # Filter: Z > sigma AND raw return > min_ret
            candidates = z_i[(z_i > sigma) & (ret_i > min_ret)].nlargest(top_n)

            if candidates.empty:
                continue

            entry_bar = i + 1
            exit_bar  = min(i + hold, n - 1)

            for ticker in candidates.index:
                # Skip if ticker still in a trade
                if ticker_busy_until.get(ticker, -1) >= entry_bar:
                    continue

                ep = ho.iloc[entry_bar][ticker]
                xp = hc.iloc[exit_bar][ticker]

                if pd.isna(ep) or pd.isna(xp) or ep <= 0:
                    continue

                raw_ret = xp / ep - 1
                net_ret = raw_ret - transaction_cost

                entry_dt = bar_dates[entry_bar]
                exit_dt  = bar_dates[exit_bar]

                trades.append({
                    "entry_bar":  entry_bar,
                    "exit_bar":   exit_bar,
                    "entry_dt":   entry_dt,
                    "exit_dt":    exit_dt,
                    "ticker":     ticker,
                    "z_score":    float(z_i[ticker]),
                    "bar_ret":    float(ret_i[ticker]),
                    "raw_ret":    raw_ret,
                    "net_ret":    net_ret,
                })
                ticker_busy_until[ticker] = exit_bar

        if len(trades) < 5:
            grid_results.append(dict(
                lookback=lb, sigma=sigma, hold_hours=hold, top_n=top_n, min_ret=min_ret,
                Sharpe=np.nan, Sortino=np.nan, Total_Return=np.nan,
                Max_DD=np.nan, n_trades=len(trades), Win_Rate=np.nan,
            ))
            continue

        tdf = pd.DataFrame(trades)
        # Average P&L per signal bar (entry date) across all concurrent positions
        tdf["entry_date"] = tdf["entry_dt"].dt.normalize()
        daily = tdf.groupby("entry_date")["net_ret"].mean()

        data_end  = bar_dates[-1].normalize()
        all_dates = pd.date_range(daily.index.min(), data_end, freq="B")
        daily_full = daily.reindex(all_dates, fill_value=0.0)

        wealth = (1 + daily_full).cumprod(); wealth = wealth / wealth.iloc[0]
        mdd    = float((wealth / wealth.cummax() - 1).min())
        sh     = _sharpe(daily_full)
        so     = _sortino(daily_full)
        wr     = float((tdf["net_ret"] > 0).mean())
        tot    = float(wealth.iloc[-1] - 1)

        avg_hold_bars = float(tdf["exit_bar"].sub(tdf["entry_bar"]).mean()) if len(tdf) else 0.0

        row = dict(
            lookback=lb, sigma=sigma, hold_hours=hold, top_n=top_n, min_ret=min_ret,
            Sharpe=sh, Sortino=so, Total_Return=tot, Max_DD=mdd,
            n_trades=len(trades), Win_Rate=wr,
            Avg_Z=float(tdf["z_score"].mean()),
            Avg_BarRet=float(tdf["bar_ret"].mean()),
            Avg_NetRet=float(tdf["net_ret"].mean()),
        )
        grid_results.append(row)

        if pd.notna(sh) and sh > best_sharpe:
            best_sharpe = sh
            best_daily  = daily_full.rename("ShortSqueeze")
            best_params = row

    grid_df = pd.DataFrame(grid_results).sort_values("Sharpe", ascending=False)

    if best_params:
        print("\n  Best Parameters:")
        for k, v in best_params.items():
            print(f"    {k:<18} {v}")

    return best_daily, best_params, grid_df
