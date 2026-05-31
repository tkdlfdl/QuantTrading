"""
Aggregate raw Reddit posts → daily sentiment scores per symbol.
Stores results in the sentiment_posts and sentiment_daily DB tables.
"""
from __future__ import annotations

import logging
import pandas as pd
import numpy as np

from data.db.client import get_conn

log = logging.getLogger(__name__)


def store_posts(posts_df: pd.DataFrame) -> int:
    """Insert raw posts into DB, skip duplicates. Returns rows inserted."""
    if posts_df.empty:
        return 0

    conn = get_conn()
    conn.register("_posts_staging", posts_df)

    before = conn.execute("SELECT COUNT(*) FROM sentiment_posts").fetchone()[0]
    conn.execute("""
        INSERT INTO sentiment_posts
        SELECT s.*
        FROM _posts_staging s
        LEFT JOIN sentiment_posts p ON p.post_id = s.post_id AND p.symbol = s.symbol
        WHERE p.post_id IS NULL
    """)
    after = conn.execute("SELECT COUNT(*) FROM sentiment_posts").fetchone()[0]
    conn.unregister("_posts_staging")
    return after - before


def aggregate_daily(symbols: list[str] | None = None) -> pd.DataFrame:
    """
    Aggregate raw posts to daily weighted sentiment per symbol.
    Weight = log(1 + upvotes + num_comments) so viral posts matter more.
    Upserts into sentiment_daily table.
    Returns the aggregated DataFrame.
    """
    conn = get_conn()

    if symbols:
        placeholders = ", ".join(["?" for _ in symbols])
        raw = conn.execute(
            f"SELECT * FROM sentiment_posts WHERE symbol IN ({placeholders})", symbols
        ).df()
    else:
        raw = conn.execute("SELECT * FROM sentiment_posts").df()

    if raw.empty:
        log.warning("No raw sentiment posts found in DB.")
        return pd.DataFrame()

    raw["ts"] = pd.to_datetime(raw["ts"])
    raw["date"] = raw["ts"].dt.date
    raw["weight"] = np.log1p(raw["upvotes"] + raw["num_comments"]).clip(lower=1)

    def wavg(group):
        w = group["weight"]
        c = group["compound"]
        total_w = w.sum()
        return {
            "avg_compound":      c.mean(),
            "weighted_compound": (c * w).sum() / total_w if total_w > 0 else 0.0,
            "mention_count":     len(group),
            "post_count":        group["post_id"].nunique(),
        }

    agg = (
        raw.groupby(["date", "symbol"])
        .apply(wavg, include_groups=False)
        .apply(pd.Series)
        .reset_index()
    )
    agg["date"] = pd.to_datetime(agg["date"])

    # Upsert into sentiment_daily
    conn.register("_daily_staging", agg)
    conn.execute("DELETE FROM sentiment_daily WHERE symbol IN (SELECT DISTINCT symbol FROM _daily_staging)")
    conn.execute("INSERT INTO sentiment_daily SELECT * FROM _daily_staging")
    conn.unregister("_daily_staging")

    log.info(f"Aggregated {len(agg)} symbol-days into sentiment_daily.")
    return agg


def load_sentiment_panel(
    symbols: list[str],
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Load daily sentiment as a wide DataFrame: index=date, columns=symbol.
    Missing days filled with 0 (neutral sentiment).
    """
    conn = get_conn()

    conditions = []
    params: list = []

    placeholders = ", ".join(["?" for _ in symbols])
    conditions.append(f"symbol IN ({placeholders})")
    params.extend(symbols)

    if start:
        conditions.append("date >= ?")
        params.append(start)
    if end:
        conditions.append("date <= ?")
        params.append(end)

    where = " AND ".join(conditions)
    df = conn.execute(
        f"SELECT date, symbol, weighted_compound FROM sentiment_daily WHERE {where} ORDER BY date",
        params,
    ).df()

    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    panel = df.pivot(index="date", columns="symbol", values="weighted_compound")

    # Reindex to only requested symbols that exist
    available = [s for s in symbols if s in panel.columns]
    panel = panel[available]

    # Fill missing dates with NaN, then 0
    all_dates = pd.date_range(panel.index.min(), panel.index.max(), freq="B")
    panel = panel.reindex(all_dates).fillna(0)
    panel.index.name = "date"

    return panel
