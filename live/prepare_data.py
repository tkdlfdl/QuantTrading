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


def _rebase_to_panel(base: pd.DataFrame, add: pd.DataFrame) -> pd.DataFrame:
    """Rebase each column of `add` onto the panel's price basis using the last
    date present in both (factor = base/add there). Columns with no overlap
    are left NaN — better a gap than a phantom jump. (Queue #34: mixed
    adjustment bases created +100..900% splice moves that entered Book A's
    momentum ranking.)"""
    join = [dt for dt in add.index if dt in base.index]
    out = pd.DataFrame(np.nan, index=add.index, columns=base.columns)
    for c in base.columns:
        if c not in add.columns:
            continue
        for dt in reversed(join):
            b, v = base.at[dt, c], add.at[dt, c]
            if np.isfinite(b) and np.isfinite(v) and v > 0:
                out[c] = add[c] * (b / v)
                break
    return out


def _guard_jumps(base: pd.DataFrame, new_rows: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """NaN-out appended cells implying a >100% 1-day move vs the previous
    panel value (loaders ffill over the gap). Real >100% days exist but are
    rare; a false NaN costs one stale close, a false jump poisons rankings."""
    prev = base.iloc[-1]
    guarded = new_rows.copy()
    for dt in guarded.index:
        row = guarded.loc[dt]
        jump = (row / prev - 1).abs() > 1.0
        bad = row.index[jump.fillna(False)]
        if len(bad):
            if verbose:
                print(f"  [prepare] JUMP GUARD {dt.date()}: masked {list(bad)[:6]}"
                      f"{'...' if len(bad) > 6 else ''}")
            guarded.loc[dt, bad] = np.nan
        prev = row.combine_first(prev)
    return guarded


def sync_daily_close(verbose=True) -> None:
    """
    Extend the daily-close parquet past the historical file so Book A stays
    current. Primary source: yfinance auto-adjusted daily closes, REBASED
    per-ticker at the join (consistent basis — queue #34 root fix). Fallback
    when the fetch fails: hourly-derived closes, same rebase + jump guard.
    """
    try:
        if not C.DAILY_CLOSE.exists():
            return
        base = pd.read_parquet(C.DAILY_CLOSE)
        base.index = pd.to_datetime(base.index)
        last = base.index.max()

        add = None
        try:
            import yfinance as yf
            start = (last - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
            fetch = yf.download(list(base.columns), start=start, interval="1d",
                                auto_adjust=True, progress=False, threads=True)["Close"]
            fetch.index = pd.to_datetime(fetch.index).tz_localize(None)
            if fetch.index.max() > last:
                add = _rebase_to_panel(base, fetch)
        except Exception as e:
            if verbose:
                print(f"  [prepare] adjusted daily fetch failed ({e}); hourly fallback.")
        if add is None:
            hc = pd.read_parquet(C.MERGED_HOURLY_CLOSE)
            hc.index = pd.to_datetime(hc.index)
            dfh = hc.groupby(hc.index.normalize()).last()
            dfh.index = pd.to_datetime(dfh.index)
            add = _rebase_to_panel(base, dfh[dfh.index > last - pd.Timedelta(days=10)])

        new_rows = add[add.index > last]
        # only append fully-completed trading days (hourly cache is the clock;
        # a live intraday "close" from yfinance would poison the last row)
        try:
            complete_through = latest_complete_date()
            new_rows = new_rows[new_rows.index <= complete_through]
        except Exception:
            pass
        if len(new_rows) == 0:
            if verbose:
                print("  [prepare] daily-close panel already current.")
            return
        new_rows = _guard_jumps(base, new_rows, verbose=verbose)
        merged = pd.concat([base, new_rows])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.to_parquet(C.DAILY_CLOSE)
        if verbose:
            print(f"  [prepare] appended {len(new_rows)} daily-close rows (adjusted basis).")
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
        refresh_etf_daily(verbose=verbose)
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


def refresh_etf_daily(verbose=True) -> bool:
    """Refresh the cross-asset ETF daily-close cache (Book X)."""
    try:
        import yfinance as yf
        import pandas as _pd
        etfs = C.PARAMS["X"]["etf_universe"]
        raw = yf.download(etfs, start="2002-01-01", interval="1d",
                          auto_adjust=True, progress=False)["Close"]
        raw.index = _pd.to_datetime(raw.index).tz_localize(None)
        # NEVER overwrite good data with worse data (2026-08-18 incident: a
        # network outage returned an EMPTY download and this function wiped
        # the cache, crashing replay_X and the whole settle).
        path = C.CACHE_DIR / "etf_daily_close.parquet"
        if raw.shape[0] < 1000:
            if verbose:
                print(f"  [prepare] ETF refresh returned {raw.shape[0]} rows — "
                      "REFUSING to overwrite cache.")
            return False
        if path.exists():
            old = _pd.read_parquet(path)
            if len(old) and _pd.to_datetime(old.index).max() > raw.index.max():
                if verbose:
                    print("  [prepare] ETF refresh older than cache — keeping cache.")
                return False
        raw.to_parquet(path)
        if verbose:
            print(f"  [prepare] ETF daily cache refreshed ({raw.shape[0]} rows).")
        return True
    except Exception as e:
        if verbose:
            print(f"  [prepare] ETF refresh failed: {e}")
        return False
