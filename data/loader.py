"""
Load data from DuckDB into the wide-panel format strategies expect.

  df = load_close_panel(["QQQ", "SPY", "AAPL"], interval="1d", start="2015-01-01")
  close = df["Close"]   # DataFrame, columns = ticker symbols
"""
from __future__ import annotations

import pandas as pd
from data.db.client import get_conn
from data.ingestion import ingest


def load_close_panel(
    symbols: list[str],
    interval: str = "1d",
    start: str = None,
    end: str = None,
    auto_ingest: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Returns {"Close": wide_df} where wide_df has symbols as columns and
    timestamps as the index. Compatible with all strategy runner functions.

    auto_ingest=True fetches any missing data from Yahoo Finance before loading.
    """
    if auto_ingest:
        for s in symbols:
            n = ingest(s, interval, start, end)
            if n > 0:
                print(f"  Ingested {n} new rows for {s} [{interval}]")

    conn = get_conn()

    conditions = [f"interval = ?"]
    params = [interval]

    if start:
        conditions.append("ts >= ?")
        params.append(start)
    if end:
        conditions.append("ts <= ?")
        params.append(end)

    placeholders = ", ".join(["?" for _ in symbols])
    conditions.append(f"symbol IN ({placeholders})")
    params.extend(symbols)

    where = " AND ".join(conditions)

    df = conn.execute(
        f"SELECT ts, symbol, open, high, low, close, volume FROM ohlcv WHERE {where} ORDER BY ts",
        params,
    ).df()

    if df.empty:
        return {"Close": pd.DataFrame(), "Open": pd.DataFrame(),
                "High": pd.DataFrame(), "Low": pd.DataFrame(), "Volume": pd.DataFrame()}

    panel = {}
    for col in ["open", "high", "low", "close", "volume"]:
        wide = df.pivot(index="ts", columns="symbol", values=col)
        wide.index.name = "Date"
        wide.index = pd.to_datetime(wide.index)
        # Reindex to requested symbol order (preserves column order for strategies)
        existing = [s for s in symbols if s in wide.columns]
        panel[col.capitalize()] = wide[existing]

    return panel
