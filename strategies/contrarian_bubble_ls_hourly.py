"""
Contrarian Bubble Score Strategy -- Long/Short, Hourly
=======================================================
LONG  : buy top-N most oversold  (bubble < -buy_threshold)
SHORT : sell top-N most overbought (bubble > +short_threshold)

Expects mean reversion on both sides.

Portfolio weighting:
  Both sides active  -> 50% long + 50% short
  One side only      -> 100% that side

Daily attribution (actual intraday):
  Entry day  : entry open  -> day close
  Middle days: prev close  -> day close  (actual close-to-close)
  Exit day   : prev close  -> exit close

TC : 0.1% per trade, spread as TC/hold_hours per active day
Borrow: 8% annual on short legs, spread per active day
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import product

TRADING_DAYS = 252
TC = 0.001
SHORT_BORROW_ANN = 0.08


def _bubble_matrix(close: pd.DataFrame, ma_window: int) -> np.ndarray:
    lp = np.log(close.replace(0, np.nan).ffill())
    fair = close.rolling(ma_window, min_periods=ma_window // 2).mean()
    residual = lp - np.log(fair.replace(0, np.nan))
    z = (
        (residual - residual.rolling(ma_window, min_periods=ma_window // 2).mean())
        / residual.rolling(ma_window, min_periods=ma_window // 2).std()
    )
    return np.tanh(z / 2).fillna(0.0).values.astype(np.float32)


def _metrics(s: pd.Series, label: str) -> dict:
    """Compute standard metrics for a daily return series."""
    if s is None or len(s) < 10:
        return {}
    wealth = (1 + s).cumprod()
    std = s.std()
    sh = float(np.sqrt(TRADING_DAYS) * s.mean() / std) if std > 0 else np.nan
    ds = s[s < 0].std(ddof=0)
    so = float(np.sqrt(TRADING_DAYS) * s.mean() / ds) if ds > 0 else np.nan
    mdd = float((wealth / wealth.cummax() - 1).min())
    tr = float(wealth.iloc[-1] - 1)
    act = s[s != 0]
    wr = float((act > 0).mean()) if len(act) > 0 else np.nan
    n_active = int((s != 0).sum())
    return dict(
        label=label, Sharpe=sh, Sortino=so,
        Total_Return=tr, Max_DD=mdd,
        Win_Rate=wr, Active_Days=n_active,
    )


def run_contrarian_bubble_ls_hourly(
    hourly_open: pd.DataFrame,
    hourly_close: pd.DataFrame,
    ma_window_grid: list = [52, 104, 156],
    buy_threshold_grid: list = [0.8, 0.9],
    short_threshold_grid: list = [0.8, 0.85, 0.9, 0.95],
    hold_hours_grid: list = [8, 13],
    top_n_grid: list = [10, 20],
    transaction_cost: float = TC,
    short_borrow_rate: float = SHORT_BORROW_ANN,
) -> tuple[dict | None, pd.DataFrame]:
    """
    Grid-search the Contrarian Bubble L/S strategy.

    Returns
    -------
    best_result : dict with keys:
        'long'     -> pd.Series daily returns (long side only)
        'short'    -> pd.Series daily returns (short side only)
        'combined' -> pd.Series daily returns (50/50 portfolio)
        'params'   -> dict of best parameters (ranked by combined Sharpe)
        'long_metrics', 'short_metrics', 'combined_metrics'
    grid_df : full grid sorted by combined Sharpe
    """
    ho = hourly_open.copy()
    hc = hourly_close.copy()
    T, U = hc.shape
    idx = hc.index

    opens = ho.values.astype(np.float32)
    prices = hc.values.astype(np.float32)

    # Daily infrastructure
    bar_date = idx.normalize()
    trading_days_arr = np.array(sorted(bar_date.unique()))
    trading_days_ts = pd.DatetimeIndex(trading_days_arr)
    date_to_int = {d: i for i, d in enumerate(trading_days_arr)}
    bar_day_int = np.array([date_to_int[d] for d in bar_date], dtype=np.int32)
    D = len(trading_days_arr)

    day_last = np.zeros(D, dtype=np.int32)
    for t in range(T):
        day_last[bar_day_int[t]] = t

    daily_close = prices[day_last]
    daily_ret_cc = np.zeros((D, U), dtype=np.float32)
    daily_ret_cc[1:] = daily_close[1:] / np.maximum(daily_close[:-1], 1e-8) - 1
    daily_ret_cc = np.clip(daily_ret_cc, -0.20, 0.20)

    # Hourly borrow cost per active day = ann_rate / (252 * 6.5) * hold_hours
    # (will be computed per combo inside loop)

    total = (len(ma_window_grid) * len(buy_threshold_grid)
             * len(short_threshold_grid) * len(hold_hours_grid) * len(top_n_grid))
    print(f"Universe: {U} tickers  |  {T} hourly bars  "
          f"({idx[0].date()} -> {idx[-1].date()})")
    print(f"Pre-computing bubble scores for {len(ma_window_grid)} MA windows...")

    score_cache: dict[int, np.ndarray] = {}
    for ma in ma_window_grid:
        raw = _bubble_matrix(hc, ma)
        shifted = np.empty_like(raw)
        shifted[0] = 0.0
        shifted[1:] = raw[:-1]
        score_cache[ma] = shifted
        pct_long = (raw < -0.8).mean() * 100
        pct_short = (raw > 0.8).mean() * 100
        print(f"  MA={ma}h  pct<-0.8: {pct_long:.2f}%  pct>+0.8: {pct_short:.2f}%")

    print(f"Grid search: {total} combinations...")

    grid_results = []
    best_combined_sharpe = -np.inf
    best_result = None

    for ma, buy_thresh, short_thresh, hold, top_n in product(
        ma_window_grid, buy_threshold_grid, short_threshold_grid,
        hold_hours_grid, top_n_grid
    ):
        bub = score_cache[ma]
        warmup = ma + 1
        # Daily borrow rate: 8% annual / 252 trading days, deducted each active day
        borrow_daily = short_borrow_rate / TRADING_DAYS

        free_long = np.zeros(U, dtype=np.int32)
        free_short = np.zeros(U, dtype=np.int32)

        # Separate accumulators for long and short
        long_num = np.zeros(D, dtype=np.float64)
        long_den = np.zeros(D, dtype=np.float64)
        short_num = np.zeros(D, dtype=np.float64)
        short_den = np.zeros(D, dtype=np.float64)

        for t in range(warmup, T - hold - 1):
            scores = bub[t]

            # ── Long side ──────────────────────────────────────────────
            avail_long = (scores < -buy_thresh) & (free_long <= t)
            if avail_long.any():
                avail_idx = np.where(avail_long)[0]
                n_pick = min(top_n, len(avail_idx))
                chosen = avail_idx[
                    np.argpartition(scores[avail_idx], n_pick - 1)[:n_pick]
                ]
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
                    day_rets = daily_ret_cc[days, s].copy()
                    dc_entry = prices[day_last[entry_day], s]
                    day_rets[0] = (dc_entry / ep - 1) if dc_entry > 0 else 0.0
                    if len(days) > 1:
                        prev_c = prices[day_last[exit_day - 1], s]
                        day_rets[-1] = (xp / prev_c - 1) if prev_c > 0 else 0.0
                    day_rets = np.clip(day_rets, -0.20, 0.20)
                    long_num[days] += day_rets
                    long_den[days] += 1.0

                free_long[chosen] = exit_bar

            # ── Short side ─────────────────────────────────────────────
            avail_short = (scores > short_thresh) & (free_short <= t)
            if avail_short.any():
                avail_idx = np.where(avail_short)[0]
                n_pick = min(top_n, len(avail_idx))
                # Most overbought = highest scores
                chosen = avail_idx[
                    np.argpartition(-scores[avail_idx], n_pick - 1)[:n_pick]
                ]
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
                    # Short return = negative of price change
                    day_rets = -daily_ret_cc[days, s].copy()
                    dc_entry = prices[day_last[entry_day], s]
                    day_rets[0] = -(dc_entry / ep - 1) if dc_entry > 0 else 0.0
                    if len(days) > 1:
                        prev_c = prices[day_last[exit_day - 1], s]
                        day_rets[-1] = -(xp / prev_c - 1) if prev_c > 0 else 0.0
                    # Daily borrow: 8%/252 per calendar day the short is held
                    day_rets -= borrow_daily
                    day_rets = np.clip(day_rets, -0.20, 0.20)
                    short_num[days] += day_rets
                    short_den[days] += 1.0

                free_short[chosen] = exit_bar

        # Build daily series for each side
        long_active = long_den > 0
        short_active = short_den > 0

        long_port = np.zeros(D)
        short_port = np.zeros(D)

        if long_active.any():
            long_port[long_active] = (
                long_num[long_active] / long_den[long_active]
                - transaction_cost / hold
            )
        if short_active.any():
            short_port[short_active] = (
                short_num[short_active] / short_den[short_active]
                - transaction_cost / hold
                # borrow already deducted per position in the inner loop
            )

        s_long = pd.Series(long_port, index=trading_days_ts)
        s_short = pd.Series(short_port, index=trading_days_ts)

        # Combined: 50/50 when both active, 100% when one side only
        both = long_active & short_active
        only_long = long_active & ~short_active
        only_short = ~long_active & short_active

        combined = np.zeros(D)
        combined[both] = 0.5 * long_port[both] + 0.5 * short_port[both]
        combined[only_long] = long_port[only_long]
        combined[only_short] = short_port[only_short]
        s_combined = pd.Series(combined, index=trading_days_ts)

        if (long_active | short_active).sum() < 5:
            grid_results.append(dict(
                ma_window=ma, buy_threshold=buy_thresh, short_threshold=short_thresh,
                hold_hours=hold, top_n=top_n,
                Long_Sharpe=np.nan, Short_Sharpe=np.nan, Combined_Sharpe=np.nan,
                Long_Return=np.nan, Short_Return=np.nan, Combined_Return=np.nan,
                Long_MaxDD=np.nan, Short_MaxDD=np.nan, Combined_MaxDD=np.nan,
            ))
            continue

        def _sh(s):
            std = s.std()
            return float(np.sqrt(TRADING_DAYS) * s.mean() / std) if std > 0 else np.nan

        def _tr(s):
            return float((1 + s).cumprod().iloc[-1] - 1)

        def _mdd(s):
            w = (1 + s).cumprod()
            return float((w / w.cummax() - 1).min())

        row = dict(
            ma_window=ma, buy_threshold=buy_thresh, short_threshold=short_thresh,
            hold_hours=hold, top_n=top_n,
            Long_Sharpe=_sh(s_long), Short_Sharpe=_sh(s_short),
            Combined_Sharpe=_sh(s_combined),
            Long_Return=_tr(s_long), Short_Return=_tr(s_short),
            Combined_Return=_tr(s_combined),
            Long_MaxDD=_mdd(s_long), Short_MaxDD=_mdd(s_short),
            Combined_MaxDD=_mdd(s_combined),
            Long_ActiveDays=int(long_active.sum()),
            Short_ActiveDays=int(short_active.sum()),
        )
        grid_results.append(row)

        csh = row["Combined_Sharpe"]
        if pd.notna(csh) and csh > best_combined_sharpe:
            best_combined_sharpe = csh
            best_result = dict(
                params=row,
                long=s_long.rename("Long"),
                short=s_short.rename("Short"),
                combined=s_combined.rename("Combined"),
                long_metrics=_metrics(s_long, "Long"),
                short_metrics=_metrics(s_short, "Short"),
                combined_metrics=_metrics(s_combined, "Combined"),
            )

    grid_df = pd.DataFrame(grid_results).sort_values(
        "Combined_Sharpe", ascending=False
    )
    return best_result, grid_df
