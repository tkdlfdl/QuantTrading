"""
Contrarian Bubble Score Strategy -- Long-Only, Hourly
=====================================================
Signal: buy when per-stock bubble score < -threshold (extreme oversold)
Exit:   close of bar at (signal_bar + hold_hours)
Universe: S&P500 + NASDAQ100

Bubble score = tanh(z/2)
  z = (log_residual - MA(log_residual)) / STD(log_residual)
  log_residual = log(close) - log(rolling_mean(close))

No lookahead: signal at bar t uses score from bar t-1.
Entry at open of bar t+1, exit at close of bar (t + hold_hours).

Daily attribution (actual intraday):
  entry day  : open -> day close
  middle days: prev day close -> day close  (actual close-to-close)
  exit day   : prev day close -> exit close

TC: 0.1% per trade, spread as TC/hold_hours per active day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import product

TRADING_DAYS = 252
TC = 0.001


def _bubble_matrix(close: pd.DataFrame, ma_window: int) -> np.ndarray:
    """Single-MA bubble score array (T x U), fillna(0)."""
    lp = np.log(close.replace(0, np.nan).ffill())
    fair = close.rolling(ma_window, min_periods=ma_window // 2).mean()
    residual = lp - np.log(fair.replace(0, np.nan))
    z = (
        (residual - residual.rolling(ma_window, min_periods=ma_window // 2).mean())
        / residual.rolling(ma_window, min_periods=ma_window // 2).std()
    )
    return np.tanh(z / 2).fillna(0.0).values.astype(np.float32)


def run_contrarian_bubble_hourly(
    hourly_open: pd.DataFrame,
    hourly_close: pd.DataFrame,
    ma_window_grid: list = [52, 104, 156, 208],
    buy_threshold_grid: list = [0.9, 0.8, 0.7, 0.6, 0.5],
    hold_hours_grid: list = [4, 8, 13, 26, 52],
    top_n_grid: list = [5, 10, 20],
    transaction_cost: float = TC,
) -> tuple[pd.Series | None, dict | None, pd.DataFrame]:
    """
    Grid-search the Contrarian Bubble strategy over hourly data.

    Daily P&L uses actual intraday attribution:
      - Entry day:   entry open -> day close
      - Middle days: close-to-close from actual daily prices
      - Exit day:    prev close -> exit bar close

    Parameters
    ----------
    hourly_open, hourly_close : bars x tickers (same index/columns)
    ma_window_grid      : MA window in hours for bubble computation
    buy_threshold_grid  : enter when bubble < -threshold (positive value)
    hold_hours_grid     : hourly bars to hold
    top_n_grid          : max positions per signal bar
    transaction_cost    : one-way TC, spread over hold days per active day

    Returns
    -------
    best_daily_ret : pd.Series daily returns for best combo (by Sharpe)
    best_params    : dict of best parameters + metrics
    grid_df        : full grid sorted by Sharpe descending
    """
    ho = hourly_open.copy()
    hc = hourly_close.copy()
    T, U = hc.shape
    idx = hc.index

    opens = ho.values.astype(np.float32)
    prices = hc.values.astype(np.float32)

    # ── Daily index infrastructure ─────────────────────────────────────────
    bar_date = idx.normalize()
    trading_days_arr = np.array(sorted(bar_date.unique()))
    trading_days_ts = pd.DatetimeIndex(trading_days_arr)
    date_to_int = {d: i for i, d in enumerate(trading_days_arr)}
    bar_day_int = np.array([date_to_int[d] for d in bar_date], dtype=np.int32)
    D = len(trading_days_arr)

    # Last bar index per trading day
    day_last = np.zeros(D, dtype=np.int32)
    for t in range(T):
        day_last[bar_day_int[t]] = t

    # Pre-compute close-to-close daily returns per ticker (D x U)
    daily_close = prices[day_last]                                 # (D, U)
    daily_ret_cc = np.zeros((D, U), dtype=np.float32)
    daily_ret_cc[1:] = daily_close[1:] / np.maximum(daily_close[:-1], 1e-8) - 1
    daily_ret_cc = np.clip(daily_ret_cc, -0.20, 0.20)

    # ── Pre-compute shifted bubble scores ──────────────────────────────────
    total = (
        len(ma_window_grid) * len(buy_threshold_grid)
        * len(hold_hours_grid) * len(top_n_grid)
    )
    print(f"Universe: {U} tickers  |  {T} hourly bars  "
          f"({idx[0].date()} -> {idx[-1].date()})")
    print(f"Pre-computing bubble scores for {len(ma_window_grid)} MA windows...")

    score_cache: dict[int, np.ndarray] = {}
    for ma in ma_window_grid:
        raw = _bubble_matrix(hc, ma)
        # shift(1): signal at bar t uses score from bar t-1
        shifted = np.empty_like(raw)
        shifted[0] = 0.0
        shifted[1:] = raw[:-1]
        score_cache[ma] = shifted
        pct = (raw < -0.8).mean() * 100
        print(f"  MA={ma}h  pct<-0.8: {pct:.2f}%")

    print(f"Grid search: {total} combinations "
          f"({len(ma_window_grid)} MA x {len(buy_threshold_grid)} thr "
          f"x {len(hold_hours_grid)} hold x {len(top_n_grid)} topN)...")

    grid_results = []
    best_sharpe = -np.inf
    best_daily: pd.Series | None = None
    best_params: dict | None = None

    for ma, thresh, hold, top_n in product(
        ma_window_grid, buy_threshold_grid, hold_hours_grid, top_n_grid
    ):
        bub = score_cache[ma]
        warmup = ma + 1
        free_at = np.zeros(U, dtype=np.int32)

        daily_num = np.zeros(D, dtype=np.float64)
        daily_den = np.zeros(D, dtype=np.float64)

        for t in range(warmup, T - hold - 1):
            scores = bub[t]
            available = (scores < -thresh) & (free_at <= t)
            if not available.any():
                continue

            avail_idx = np.where(available)[0]
            n_pick = min(top_n, len(avail_idx))
            chosen = avail_idx[np.argpartition(scores[avail_idx], n_pick - 1)[:n_pick]]

            entry_bar = t + 1
            exit_bar = min(t + hold, T - 1)
            entry_day = bar_day_int[entry_bar]
            exit_day = bar_day_int[exit_bar]
            days = np.arange(entry_day, exit_day + 1)

            for s in chosen:
                ep = opens[entry_bar, s]
                xp = prices[exit_bar, s]
                if ep <= 0 or xp <= 0 or not (np.isfinite(ep) and np.isfinite(xp)):
                    continue

                # Actual intraday daily attribution
                day_rets = daily_ret_cc[days, s].copy()

                # Entry day: open -> day close (actual open-to-close)
                dc_entry = prices[day_last[entry_day], s]
                day_rets[0] = (dc_entry / ep - 1) if dc_entry > 0 else 0.0

                # Exit day: prev close -> exit bar close
                if len(days) > 1:
                    prev_c = prices[day_last[exit_day - 1], s]
                    day_rets[-1] = (xp / prev_c - 1) if prev_c > 0 else 0.0

                day_rets = np.clip(day_rets, -0.20, 0.20)
                daily_num[days] += day_rets
                daily_den[days] += 1.0

            free_at[chosen] = exit_bar

        active = daily_den > 0
        port = np.zeros(D, dtype=np.float64)
        # TC spread per active day (total TC / hold_hours per day)
        port[active] = daily_num[active] / daily_den[active] - transaction_cost / hold

        s_daily = pd.Series(port, index=trading_days_ts)
        n_active = int(active.sum())

        if n_active < 5:
            grid_results.append(dict(
                ma_window=ma, buy_threshold=thresh, hold_hours=hold, top_n=top_n,
                Sharpe=np.nan, Sortino=np.nan, Total_Return=np.nan,
                Max_DD=np.nan, n_trades=n_active, Win_Rate=np.nan,
            ))
            continue

        wealth = (1 + s_daily).cumprod()
        std = s_daily.std()
        sh = float(np.sqrt(TRADING_DAYS) * s_daily.mean() / std) if std > 0 else np.nan
        ds = s_daily[s_daily < 0].std(ddof=0)
        so = float(np.sqrt(TRADING_DAYS) * s_daily.mean() / ds) if ds > 0 else np.nan
        mdd = float((wealth / wealth.cummax() - 1).min())
        tr = float(wealth.iloc[-1] - 1)
        act_s = s_daily[active]
        wr = float((act_s > 0).mean()) if len(act_s) > 0 else 0.5

        row = dict(
            ma_window=ma, buy_threshold=thresh, hold_hours=hold, top_n=top_n,
            Sharpe=sh, Sortino=so, Total_Return=tr,
            Max_DD=mdd, n_trades=n_active, Win_Rate=wr,
        )
        grid_results.append(row)

        if pd.notna(sh) and sh > best_sharpe:
            best_sharpe = sh
            best_daily = s_daily.rename("ContrarianBubble")
            best_params = row

    grid_df = pd.DataFrame(grid_results).sort_values("Sharpe", ascending=False)

    if best_params:
        print("\nBest Parameters:")
        for k, v in best_params.items():
            print(f"  {k:<18} {v}")

    return best_daily, best_params, grid_df
