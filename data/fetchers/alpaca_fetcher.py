"""
Alpaca Historical Data Fetcher — free paper-trading tier.

Free tier uses IEX feed (covers most liquid US stocks).
Provides 1h bars back to ~2016 for most tickers.

Setup (one-time):
  1. Sign up free at https://alpaca.markets
  2. Go to Paper Trading → API Keys → Generate
  3. Set env vars or pass keys directly:
       ALPACA_API_KEY    = "PK..."
       ALPACA_SECRET_KEY = "..."

Install:
  pip install alpaca-py
"""
from __future__ import annotations

import os
import time
import math
import logging
from datetime import datetime, timedelta

import pandas as pd

log = logging.getLogger(__name__)

_CHUNK_MONTHS = 6       # fetch 6-month windows (Alpaca handles it fine)
_SLEEP        = 0.3     # seconds between requests (free tier: ~200 req/min)


def _get_client(api_key: str | None, secret_key: str | None):
    """Build Alpaca StockHistoricalDataClient."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
    except ImportError:
        raise ImportError(
            "alpaca-py not installed. Run: pip install alpaca-py"
        )
    key    = api_key    or os.environ.get("ALPACA_API_KEY")
    secret = secret_key or os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise ValueError(
            "Alpaca credentials required.\n"
            "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables,\n"
            "or pass api_key/secret_key arguments."
        )
    return StockHistoricalDataClient(key, secret)


def fetch_alpaca_bars(
    tickers: list[str],
    start: str = "2016-01-01",
    end: str | None = None,
    timeframe: str = "1Hour",
    feed: str = "iex",          # "iex" = free, "sip" = paid
    api_key: str | None = None,
    secret_key: str | None = None,
    chunk_size: int = 50,       # tickers per batch
    sleep: float = _SLEEP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch hourly OHLCV bars from Alpaca for a list of tickers.

    Returns
    -------
    (hourly_open, hourly_close) : DataFrames with DatetimeIndex (tz-naive UTC)
                                  columns = tickers
    """
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError:
        raise ImportError("Run: pip install alpaca-py")

    client = _get_client(api_key, secret_key)
    end_dt  = datetime.utcnow() if end is None else datetime.strptime(end, "%Y-%m-%d")
    start_dt = datetime.strptime(start, "%Y-%m-%d")

    tf_map = {
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day":  TimeFrame(1, TimeFrameUnit.Day),
        "1Min":  TimeFrame(1, TimeFrameUnit.Minute),
    }
    tf = tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Hour))

    all_opens  = []
    all_closes = []

    n_chunks = math.ceil(len(tickers) / chunk_size)
    for ci, i in enumerate(range(0, len(tickers), chunk_size)):
        batch = tickers[i : i + chunk_size]
        print(f"  Alpaca batch {ci+1}/{n_chunks} ({len(batch)} tickers)...",
              end=" ", flush=True)
        try:
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=tf,
                start=start_dt,
                end=end_dt,
                feed=feed,
                adjustment="all",    # adjusted for splits/dividends
            )
            bars = client.get_stock_bars(req).df

            if bars.empty:
                print("no data")
                continue

            # bars has MultiIndex (symbol, timestamp) → unstack to wide
            bars = bars.reset_index()
            bars["timestamp"] = pd.to_datetime(bars["timestamp"]).dt.tz_localize(None)

            open_wide  = bars.pivot(index="timestamp", columns="symbol", values="open")
            close_wide = bars.pivot(index="timestamp", columns="symbol", values="close")

            all_opens.append(open_wide)
            all_closes.append(close_wide)
            print(f"ok ({len(bars)} rows)")

        except Exception as e:
            print(f"ERROR: {e}")
            log.warning(f"Alpaca batch {ci+1} failed: {e}")

        time.sleep(sleep)

    if not all_opens:
        return pd.DataFrame(), pd.DataFrame()

    ho = (pd.concat(all_opens,  axis=1).sort_index()
          .ffill().dropna(axis="columns", how="all"))
    hc = (pd.concat(all_closes, axis=1).sort_index()
          .ffill().dropna(axis="columns", how="all"))

    # Keep only common columns
    common = ho.columns.intersection(hc.columns)
    return ho[common], hc[common]
