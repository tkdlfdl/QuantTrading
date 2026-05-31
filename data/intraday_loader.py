"""
Loads two layers of price data for the intraday mean-reversion strategy:
  1. Daily close  — long history (from 2018) for signal lookback window
  2. Hourly OHLCV — last 730 days (yfinance max) for trade execution
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

_DAILY_CACHE  = CACHE_DIR / "daily_close.parquet"
_HOURLY_O_CACHE = CACHE_DIR / "hourly_open.parquet"
_HOURLY_C_CACHE = CACHE_DIR / "hourly_close.parquet"

_CHUNK = 100          # tickers per yfinance batch
_HRS_START = (datetime.now() - timedelta(days=729)).strftime("%Y-%m-%d")
_DAILY_START = "2018-01-01"


def _batch_download(tickers: list[str], chunk: int = _CHUNK, **kwargs) -> pd.DataFrame:
    """Download in chunks and concat to avoid yfinance timeouts."""
    frames = []
    for i in range(0, len(tickers), chunk):
        batch = tickers[i : i + chunk]
        print(f"  batch {i//chunk + 1}/{math.ceil(len(tickers)/chunk)} ({len(batch)} tickers)...")
        try:
            raw = yf.download(batch, progress=False, **kwargs)
            if not raw.empty:
                frames.append(raw)
        except Exception as e:
            print(f"  warning: batch failed: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def _extract_price(raw: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    """Pull a single price field from a (possibly MultiIndex) download result."""
    if isinstance(raw.columns, pd.MultiIndex):
        if field in raw.columns.get_level_values(0):
            df = raw[field]
        else:
            return pd.DataFrame()
    else:
        if field in raw.columns:
            df = raw[[field]].rename(columns={field: tickers[0]})
        else:
            return pd.DataFrame()
    return df


def load_daily_close(
    tickers: list[str],
    start: str = _DAILY_START,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Daily adjusted close prices, full history from `start`."""
    if use_cache and _DAILY_CACHE.exists():
        df = pd.read_parquet(_DAILY_CACHE)
        missing = [t for t in tickers if t not in df.columns]
        if not missing:
            print(f"Daily close: loaded {len(df.columns)} tickers from cache.")
            return df[tickers].ffill()
        print(f"Daily cache missing {len(missing)} tickers — re-downloading all.")

    print(f"Downloading daily close for {len(tickers)} tickers from {start}...")
    raw = _batch_download(tickers, start=start, auto_adjust=True)
    if raw.empty:
        return pd.DataFrame()

    close = _extract_price(raw, "Close", tickers)
    close.index = pd.to_datetime(close.index)
    close = close.ffill().dropna(axis="columns", how="all")

    close.to_parquet(_DAILY_CACHE)
    print(f"Daily close: {len(close)} days × {len(close.columns)} tickers saved.")
    return close


def load_hourly_bars(
    tickers: list[str],
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hourly open and close prices for last 730 days.
    Returns (hourly_open, hourly_close) with tz-naive DatetimeIndex.
    """
    if use_cache and _HOURLY_O_CACHE.exists() and _HOURLY_C_CACHE.exists():
        ho = pd.read_parquet(_HOURLY_O_CACHE)
        hc = pd.read_parquet(_HOURLY_C_CACHE)
        missing = [t for t in tickers if t not in ho.columns]
        # also refresh if data is older than 1 day
        last_ts = ho.index[-1]
        data_age_days = (datetime.now() - last_ts).days
        if not missing and data_age_days < 2:
            print(f"Hourly bars: loaded {len(ho.columns)} tickers from cache (last: {last_ts.date()}).")
            return ho[tickers], hc[tickers]
        print(f"Hourly cache stale or missing {len(missing)} tickers — re-downloading.")

    print(f"Downloading 1h bars for {len(tickers)} tickers from {_HRS_START}...")
    raw = _batch_download(tickers, start=_HRS_START, interval="1h", auto_adjust=True)
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    hourly_open  = _extract_price(raw, "Open",  tickers)
    hourly_close = _extract_price(raw, "Close", tickers)

    for df in (hourly_open, hourly_close):
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

    common = list(set(hourly_open.columns) & set(hourly_close.columns))
    hourly_open  = hourly_open[common].ffill()
    hourly_close = hourly_close[common].ffill()

    hourly_open.to_parquet(_HOURLY_O_CACHE)
    hourly_close.to_parquet(_HOURLY_C_CACHE)
    print(f"Hourly bars: {len(hourly_open)} bars × {len(common)} tickers saved.")
    return hourly_open, hourly_close
