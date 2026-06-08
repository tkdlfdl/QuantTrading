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
        sync_daily_close(verbose=verbose)
    last = latest_complete_date()
    if verbose:
        print(f"  [prepare] latest complete trading date in cache: {last.date()}")
    return last
