"""
Fetch → normalise → store pipeline.

Incremental by default: only inserts rows not already in the DB.
Call ingest() for a single symbol/interval or ingest_batch() for many.
"""
from __future__ import annotations

import pandas as pd

from data.db.client import get_conn
from data.fetchers.yfinance_fetcher import YFinanceFetcher

_fetcher = YFinanceFetcher()


def ingest(symbol: str, interval: str, start: str = None, end: str = None) -> int:
    df = _fetcher.fetch(symbol, interval, start, end)
    if df.empty:
        return 0

    conn = get_conn()
    conn.register("_staging", df)

    before = conn.execute(
        "SELECT COUNT(*) FROM ohlcv WHERE symbol = ? AND interval = ?", [symbol, interval]
    ).fetchone()[0]

    conn.execute("""
        INSERT INTO ohlcv
        SELECT s.*
        FROM _staging s
        LEFT JOIN ohlcv o
               ON o.ts = s.ts AND o.symbol = s.symbol AND o.interval = s.interval
        WHERE o.ts IS NULL
    """)

    after = conn.execute(
        "SELECT COUNT(*) FROM ohlcv WHERE symbol = ? AND interval = ?", [symbol, interval]
    ).fetchone()[0]
    inserted = after - before

    conn.execute("""
        INSERT INTO fetch_log (symbol, interval, fetched_from, fetched_to, rows_inserted)
        VALUES (?, ?, ?, ?, ?)
    """, [symbol, interval, df["ts"].min(), df["ts"].max(), inserted])

    conn.unregister("_staging")
    return inserted


def ingest_batch(
    symbols: list[str],
    intervals: list[str],
    start: str = None,
    end: str = None,
) -> dict:
    results: dict = {}
    for symbol in symbols:
        results[symbol] = {}
        for interval in intervals:
            try:
                n = ingest(symbol, interval, start, end)
                print(f"  {symbol:<12} {interval:<5}  {n:>6} rows inserted")
                results[symbol][interval] = n
            except Exception as exc:
                print(f"  {symbol:<12} {interval:<5}  ERROR: {exc}")
                results[symbol][interval] = -1
    return results


def ingest_universe(
    symbols: list[str],
    interval: str = "1d",
    start: str = None,
    end: str = None,
    chunk_size: int = 100,
) -> int:
    """
    Efficiently ingest a large universe using batch yfinance downloads.
    Downloads chunk_size tickers at once instead of one per API call.
    Returns total rows inserted.
    """
    conn = get_conn()
    chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
    total_inserted = 0

    for idx, chunk in enumerate(chunks):
        print(f"  Chunk {idx + 1}/{len(chunks)}: downloading {len(chunk)} tickers...", end=" ", flush=True)
        try:
            df = _fetcher.fetch_batch(chunk, interval, start, end)
            if df.empty:
                print("no data")
                continue

            conn.register("_staging", df)

            placeholders = ", ".join(["?" for _ in chunk])
            before = conn.execute(
                f"SELECT COUNT(*) FROM ohlcv WHERE interval = ? AND symbol IN ({placeholders})",
                [interval] + chunk,
            ).fetchone()[0]

            conn.execute("""
                INSERT INTO ohlcv
                SELECT s.*
                FROM _staging s
                LEFT JOIN ohlcv o
                       ON o.ts = s.ts AND o.symbol = s.symbol AND o.interval = s.interval
                WHERE o.ts IS NULL
            """)

            after = conn.execute(
                f"SELECT COUNT(*) FROM ohlcv WHERE interval = ? AND symbol IN ({placeholders})",
                [interval] + chunk,
            ).fetchone()[0]

            inserted = after - before
            total_inserted += inserted
            conn.unregister("_staging")
            print(f"{inserted:,} rows inserted")

        except Exception as exc:
            print(f"ERROR: {exc}")

    return total_inserted


def export_parquet(symbol: str, interval: str) -> str:
    """Export one symbol+interval to a parquet file and return its path."""
    from config.settings import PARQUET_DIR
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    path = PARQUET_DIR / f"{symbol.replace('=', '_')}_{interval}.parquet"

    conn = get_conn()
    conn.execute(f"""
        COPY (
            SELECT * FROM ohlcv
            WHERE symbol = '{symbol}' AND interval = '{interval}'
            ORDER BY ts
        ) TO '{path}' (FORMAT PARQUET)
    """)
    return str(path)
