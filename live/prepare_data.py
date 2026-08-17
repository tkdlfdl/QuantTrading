"""
live/prepare_data.py
====================
Phase 0 — ensure price caches are current.

Strategy:
  - Try to refresh the merged hourly caches via data/intraday_loader.load_hourly_bars,
    which self-manages staleness (re-downloads only if cache >7 days old or missing tickers).
  - Keep the daily-close panel current for Book A by appending day-close rows derived from
    the (refreshed) hourly-close panel for any dates beyond the historical extended file.
  - Non-fatal: on any network/refresh error, fall back to the existing cache so the engine
    can still settle and report.

Returns the latest fully-completed trading date available in the hourly cache.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C


def refresh_hourly(verbose=True) -> bool:
    """Attempt to refresh merged hourly caches. Returns True on success."""
    try:
        from data.universe import get_universe
        from data.intraday_loader import load_hourly_bars
        univ = get_universe()
        load_hourly_bars(univ, use_cache=True)   # self-stale-managed; writes merged_*.parquet
        if verbose:
            print("  [prepare] hourly caches refreshed (or already fresh).")
        return True
    except Exception as e:
        if verbose:
            print(f"  [prepare] hourly refresh skipped/failed: {e}")
        return False


def sync_daily_close(verbose=True) -> None:
    """
    Append recent daily closes (derived from the hourly close panel) onto the extended
    daily-close parquet, so Book A stays current beyond the static historical file.
    """
    try:
        hc = pd.read_parquet(C.MERGED_HOURLY_CLOSE)
        hc.index = pd.to_datetime(hc.index)
        daily_from_hourly = hc.groupby(hc.index.normalize()).last()
        daily_from_hourly.index = pd.to_datetime(daily_from_hourly.index)

        if C.DAILY_CLOSE.exists():
            base = pd.read_parquet(C.DAILY_CLOSE)
            base.index = pd.to_datetime(base.index)
            new_dates = daily_from_hourly.index[daily_from_hourly.index > base.index.max()]
            if len(new_dates) > 0:
                add = daily_from_hourly.loc[new_dates].reindex(columns=base.columns)
                merged = pd.concat([base, add])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                merged.to_parquet(C.DAILY_CLOSE)
                if verbose:
                    print(f"  [prepare] appended {len(new_dates)} daily-close rows for Book A.")
            elif verbose:
                print("  [prepare] daily-close panel already current.")
    except Exception as e:
        if verbose:
            print(f"  [prepare] daily-close sync skipped: {e}")


def latest_complete_date() -> pd.Timestamp:
    """The latest fully-completed trading day present in the hourly cache."""
    hc = pd.read_parquet(C.MERGED_HOURLY_CLOSE)
    hc.index = pd.to_datetime(hc.index)
    return hc.index.normalize().max()


def prepare(refresh: bool = True, verbose: bool = True) -> pd.Timestamp:
    if refresh:
        refresh_hourly(verbose=verbose)
        refresh_daily_ohlcv(verbose=verbose)
        sync_daily_close(verbose=verbose)
    last = latest_complete_date()
    if verbose:
        print(f"  [prepare] latest complete trading date in cache: {last.date()}")
    return last


def refresh_daily_ohlcv(verbose=True, lookback_days=14) -> bool:
    """Refresh DuckDB daily OHLCV (needed by Book G — official open prices).
    Batch-downloads the last `lookback_days` of daily bars for the cached
    universe via yfinance and upserts rows not already present."""
    try:
        import yfinance as yf, duckdb
        import pandas as _pd
        from datetime import datetime, timedelta
        from data.universe import get_cached_universe
        univ = get_cached_universe()
        if not univ:
            return False
        start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        raw = yf.download(univ, start=start, interval="1d", auto_adjust=True,
                          progress=False, group_by="ticker", threads=True)
        rows = []
        for t in univ:
            try:
                df = raw[t].dropna(subset=["Close"])
            except Exception:
                continue
            for ts, r in df.iterrows():
                rows.append((_pd.Timestamp(ts).to_pydatetime(), t, "1d",
                             float(r["Open"]), float(r["High"]), float(r["Low"]),
                             float(r["Close"]), float(r.get("Volume", 0) or 0)))
        if not rows:
            return False
        con = duckdb.connect(str(C.SENTIMENT_DB))
        con.register("_stage", _pd.DataFrame(rows,
            columns=["ts","symbol","interval","open","high","low","close","volume"]))
        con.execute("""
            INSERT INTO ohlcv
            SELECT s.* FROM _stage s
            LEFT JOIN ohlcv o ON o.ts = s.ts AND o.symbol = s.symbol AND o.interval = s.interval
            WHERE o.ts IS NULL
        """)
        con.close()
        if verbose:
            print(f"  [prepare] daily OHLCV refreshed ({len(rows)} candidate rows).")
        return True
    except Exception as e:
        if verbose:
            print(f"  [prepare] daily OHLCV refresh failed: {e}")
        return False
