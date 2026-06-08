"""
live/signals.py
===============
Forward signal math for each book, using the LOCKED best parameters (no grid search).
Strictly causal: every signal at bar/day t uses data only through t.

Provides:
  load_panels()                     -> dict of aligned price panels (hourly + daily)
  bubble_score_hourly(prices, ma)   -> per-stock hourly bubble score  (Books B universe, D)
  qqq_bubble(qqq, ma)               -> QQQ hourly bubble score         (Book B trigger)
  momentum_hours(prices, lb)        -> trailing hourly momentum         (Books B, D rank)
  zscore_daily(daily_ret, lb)       -> daily return Z-score             (Book C)
  daily_momentum_rank(close, lb)    -> 140d momentum ranking            (Book A)

The execution/settlement engine (engine.py / settle.py) consumes these to decide
entries on each newly-completed trading day.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C


# ─────────────────────────────────────────────────────────────────────
# DATA LOADING / ALIGNMENT
# ─────────────────────────────────────────────────────────────────────
def _floor_dedup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = df.index.floor("h")
    return df[~df.index.duplicated(keep="last")]


def load_panels() -> dict:
    """
    Load and align all price panels used by the books.
    Returns a dict with:
      hourly_close, hourly_open : DataFrames [T x U]  (S&P500+NASDAQ100 universe)
      qqq_hourly                : Series [T]           (QQQ close, hour-floored)
      daily_close               : DataFrame [D x U']   (full daily history incl. UVXY/VIX)
      tickers                   : list[str]            (hourly universe columns)
      idx_h                     : DatetimeIndex        (hourly bar index)
    """
    hc = _floor_dedup(pd.read_parquet(C.MERGED_HOURLY_CLOSE))
    ho = _floor_dedup(pd.read_parquet(C.MERGED_HOURLY_OPEN))
    idx = hc.index.intersection(ho.index)
    hc, ho = hc.loc[idx], ho.loc[idx]

    daily = pd.read_parquet(C.DAILY_CLOSE)

    # Universe = S&P500+NASDAQ100 columns present in hourly cache.
    try:
        from data.universe import get_universe
        univ = set(get_universe())
    except Exception:
        univ = set(daily.columns)
    tickers = [c for c in hc.columns if c in univ]

    hc = hc[tickers].ffill()
    ho = ho[tickers].ffill()

    qqq = None
    try:
        q = pd.read_parquet(C.QQQ_HOURLY_CLOSE)["QQQ"]
        q.index = q.index.floor("h")
        q = q[~q.index.duplicated(keep="last")]
        qqq = q.reindex(idx).ffill()
    except Exception:
        if "QQQ" in hc.columns:
            qqq = hc["QQQ"]

    return dict(hourly_close=hc, hourly_open=ho, qqq_hourly=qqq,
                daily_close=daily, tickers=tickers, idx_h=idx)


# ─────────────────────────────────────────────────────────────────────
# BUBBLE SCORE (hourly, per-stock) — Books B universe rank input & D
# ─────────────────────────────────────────────────────────────────────
def bubble_score_hourly(prices: pd.DataFrame, ma_window: int) -> pd.DataFrame:
    """
    bubble = tanh(z/2),  z = normalised log-price residual vs rolling mean.
    Causal: score at bar t uses data through t only.
    """
    log_p = np.log(prices.replace(0, np.nan).ffill())
    fair  = prices.rolling(ma_window, min_periods=ma_window // 2).mean()
    res   = log_p - np.log(fair.replace(0, np.nan))
    z     = ((res - res.rolling(ma_window, min_periods=ma_window // 2).mean())
             / res.rolling(ma_window, min_periods=ma_window // 2).std())
    return np.tanh(z / 2).fillna(0)


def qqq_bubble(qqq: pd.Series, ma_window: int) -> pd.Series:
    log_p = np.log(qqq.replace(0, np.nan).ffill())
    fair  = qqq.rolling(ma_window, min_periods=ma_window // 2).mean()
    res   = log_p - np.log(fair.replace(0, np.nan))
    z     = ((res - res.rolling(ma_window, min_periods=ma_window // 2).mean())
             / res.rolling(ma_window, min_periods=ma_window // 2).std())
    return np.tanh(z / 2).fillna(0)


def momentum_hours(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Trailing compound return over `lookback` hourly bars (causal)."""
    return prices.pct_change(lookback).fillna(0)


# ─────────────────────────────────────────────────────────────────────
# DAILY Z-SCORE — Book C
# ─────────────────────────────────────────────────────────────────────
def zscore_daily(daily_ret: pd.DataFrame, lookback: int) -> pd.DataFrame:
    mu  = daily_ret.rolling(lookback).mean()
    sd  = daily_ret.rolling(lookback).std()
    z   = (daily_ret - mu) / sd.replace(0, np.nan)
    return z


# ─────────────────────────────────────────────────────────────────────
# DAILY MOMENTUM RANK — Book A
# ─────────────────────────────────────────────────────────────────────
def daily_momentum_rank(close: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """140-day momentum (compound return) per stock, causal."""
    return close.pct_change(lookback)


def daily_bubble_on_curve(equity: pd.Series, ma_days: int, z_days: int) -> pd.Series:
    """Bubble score on Book A's own equity curve (leverage/hedge trigger)."""
    lp   = np.log(np.maximum(equity, 1e-9))
    fair = lp.rolling(ma_days).mean()
    r    = lp - fair
    z    = (r - r.rolling(z_days).mean()) / r.rolling(z_days).std()
    return np.tanh(z / 2)
