"""
Loads price data for intraday strategies.

  1. Daily close  — long history (from 2018) for signal lookback
  2. Hourly OHLCV — two sources merged:
       a. Alpaca (free tier, IEX feed) — from 2016 to ~2 years ago
       b. yfinance — last 730 days (max available)
     Together they provide hourly bars back to 2016.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

_DAILY_CACHE    = CACHE_DIR / "daily_close.parquet"
_HOURLY_O_CACHE = CACHE_DIR / "hourly_open.parquet"       # yfinance-only (730d)
_HOURLY_C_CACHE = CACHE_DIR / "hourly_close.parquet"
_MERGED_O_CACHE = CACHE_DIR / "merged_hourly_open.parquet"  # Alpaca+yfinance (full history)
_MERGED_C_CACHE = CACHE_DIR / "merged_hourly_close.parquet"

_CHUNK = 100          # tickers per yfinance batch
_HRS_START   = (datetime.now() - timedelta(days=729)).strftime("%Y-%m-%d")
_DAILY_START = "2018-01-01"
_ALPACA_START = "2016-01-01"   # Alpaca IEX free tier history start
_ALPACA_O_CACHE = CACHE_DIR / "alpaca_hourly_open.parquet"
_ALPACA_C_CACHE = CACHE_DIR / "alpaca_hourly_close.parquet"


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


def load_alpaca_hourly_bars(
    tickers: list[str],
    start: str = _ALPACA_START,
    end: str | None = None,
    api_key: str | None = None,
    secret_key: str | None = None,
    use_cache: bool = True,
    feed: str = "iex",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hourly bars from Alpaca free tier (IEX feed).
    Provides history back to ~2016, complementing yfinance's 730-day limit.

    Requires free Alpaca account: https://alpaca.markets
    Set ALPACA_API_KEY and ALPACA_SECRET_KEY env vars, or pass directly.
    """
    if use_cache and _ALPACA_O_CACHE.exists() and _ALPACA_C_CACHE.exists():
        ho = pd.read_parquet(_ALPACA_O_CACHE)
        hc = pd.read_parquet(_ALPACA_C_CACHE)
        missing = [t for t in tickers if t not in ho.columns]
        last_ts = ho.index[-1]
        data_age_days = (datetime.now().date() - last_ts.date()).days
        if not missing and data_age_days < 7:
            print(f"Alpaca cache: {len(ho.columns)} tickers (last: {last_ts.date()}).")
            return ho[tickers], hc[tickers]
        print(f"Alpaca cache stale or missing {len(missing)} tickers — re-downloading.")

    from data.fetchers.alpaca_fetcher import fetch_alpaca_bars
    # Only fetch up to yfinance overlap point (avoid duplication)
    alpaca_end = end or _HRS_START
    print(f"Downloading Alpaca 1h bars ({start} → {alpaca_end}, feed={feed})...")
    ho, hc = fetch_alpaca_bars(
        tickers=tickers, start=start, end=alpaca_end,
        timeframe="1Hour", feed=feed,
        api_key=api_key, secret_key=secret_key,
        chunk_size=50,
    )
    if ho.empty:
        return pd.DataFrame(), pd.DataFrame()

    ho.to_parquet(_ALPACA_O_CACHE)
    hc.to_parquet(_ALPACA_C_CACHE)
    print(f"Alpaca: {len(ho)} bars × {len(ho.columns)} tickers saved.")
    return ho, hc


def load_hourly_bars(
    tickers: list[str],
    use_cache: bool = True,
    use_alpaca: bool = False,
    alpaca_api_key: str | None = None,
    alpaca_secret_key: str | None = None,
    alpaca_feed: str = "iex",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hourly open and close prices.
    If use_alpaca=True: merges Alpaca history (2016+) with yfinance (last 730d).
    Otherwise: yfinance only (last 730 days).
    Returns (hourly_open, hourly_close) with tz-naive DatetimeIndex.
    """
    # ── yfinance (recent 730 days) ─────────────────────────────────────────
    yf_cached = use_cache and _HOURLY_O_CACHE.exists() and _HOURLY_C_CACHE.exists()
    if yf_cached:
        yf_ho = pd.read_parquet(_HOURLY_O_CACHE)
        yf_hc = pd.read_parquet(_HOURLY_C_CACHE)
        missing = [t for t in tickers if t not in yf_ho.columns]
        last_ts = yf_ho.index[-1]
        data_age_days = (datetime.now().date() - last_ts.date()).days
        if not missing and data_age_days < 7:
            print(f"yfinance cache: {len(yf_ho.columns)} tickers (last: {last_ts.date()}).")
        else:
            yf_cached = False

    if not yf_cached:
        print(f"Downloading yfinance 1h bars from {_HRS_START}...")
        raw = _batch_download(tickers, start=_HRS_START, interval="1h", auto_adjust=True)
        if raw.empty:
            return pd.DataFrame(), pd.DataFrame()
        yf_ho = _extract_price(raw, "Open",  tickers)
        yf_hc = _extract_price(raw, "Close", tickers)
        for df in (yf_ho, yf_hc):
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
        common = list(set(yf_ho.columns) & set(yf_hc.columns))
        yf_ho = yf_ho[common].ffill()
        yf_hc = yf_hc[common].ffill()
        yf_ho.to_parquet(_HOURLY_O_CACHE)
        yf_hc.to_parquet(_HOURLY_C_CACHE)
        print(f"yfinance: {len(yf_ho)} bars × {len(yf_ho.columns)} tickers saved.")

    if not use_alpaca:
        # Check if merged cache already exists and is fresh
        if (_MERGED_O_CACHE.exists() and _MERGED_C_CACHE.exists()):
            mo = pd.read_parquet(_MERGED_O_CACHE)
            mc = pd.read_parquet(_MERGED_C_CACHE)
            missing_m = [t for t in tickers if t not in mo.columns]
            age_m = (datetime.now().date() - mo.index[-1].date()).days
            if not missing_m and age_m < 7:
                print(f"Merged cache: {len(mo.columns)} tickers "
                      f"({mo.index[0].date()} → {mo.index[-1].date()}).")
                return mo[tickers], mc[tickers]
        avail = [t for t in tickers if t in yf_ho.columns]
        return yf_ho[avail], yf_hc[avail]

    # ── Alpaca (extended history) ──────────────────────────────────────────
    al_ho, al_hc = load_alpaca_hourly_bars(
        tickers=tickers, start=_ALPACA_START, end=_HRS_START,
        api_key=alpaca_api_key, secret_key=alpaca_secret_key,
        use_cache=use_cache, feed=alpaca_feed,
    )

    if al_ho.empty:
        print("Alpaca returned no data — falling back to yfinance only.")
        avail = [t for t in tickers if t in yf_ho.columns]
        return yf_ho[avail], yf_hc[avail]

    # ── Merge: extend yfinance history with Alpaca — keep ALL yfinance tickers ─
    print("Merging Alpaca + yfinance (keeping all tickers)...")

    # Base: all yfinance tickers
    merged_o = yf_ho.copy()
    merged_c = yf_hc.copy()

    # Prepend Alpaca bars from before yfinance window
    yf_start  = yf_ho.index[0]
    al_hist_o = al_ho[al_ho.index < yf_start]
    al_hist_c = al_hc[al_hc.index < yf_start]

    if not al_hist_o.empty:
        common_t = [t for t in al_hist_o.columns if t in merged_o.columns]
        ext_idx  = al_hist_o.index.union(merged_o.index)
        merged_o = merged_o.reindex(ext_idx)
        merged_c = merged_c.reindex(ext_idx)
        merged_o.loc[al_hist_o.index, common_t] = al_hist_o[common_t].values
        merged_c.loc[al_hist_o.index, common_t] = al_hist_c[common_t].values
        merged_o = merged_o.sort_index().ffill()
        merged_c = merged_c.sort_index().ffill()
        print(f"Extended {len(common_t)} tickers with Alpaca history "
              f"({al_hist_o.index[0].date()} → {al_hist_o.index[-1].date()})")

    n_tickers = len(merged_o.columns)
    print(f"Merged: {len(merged_o)} bars × {n_tickers} tickers  "
          f"({merged_o.index[0].date()} → {merged_o.index[-1].date()})")

    # Save to dedicated merged cache (never overwrites yfinance-only cache)
    merged_o.to_parquet(_MERGED_O_CACHE)
    merged_c.to_parquet(_MERGED_C_CACHE)
    print(f"Saved to {_MERGED_O_CACHE.name}")
    return merged_o, merged_c
