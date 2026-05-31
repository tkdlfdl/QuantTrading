"""
StockTwits sentiment fetcher — no authentication required.

StockTwits is purpose-built for stock sentiment. Each message has an explicit
Bullish/Bearish label from the user, making it cleaner than VADER on Reddit text.

Rate limit: 200 requests/hour (unauthenticated).
"""
from __future__ import annotations

import time
import logging
from datetime import datetime, timezone

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

log    = logging.getLogger(__name__)
_vader = SentimentIntensityAnalyzer()
_BASE  = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"


def _label_to_score(label: str | None, text: str) -> float:
    """Convert StockTwits label → compound score, fall back to VADER."""
    if label == "Bullish":
        return 0.6
    if label == "Bearish":
        return -0.6
    return _vader.polarity_scores(text)["compound"]


def fetch_stocktwits(
    ticker: str,
    max_pages: int = 20,
    sleep: float = 0.4,
) -> pd.DataFrame:
    """
    Fetch recent StockTwits messages for a ticker.
    Paginates backward using max_id; returns up to max_pages * 30 messages.
    """
    url     = _BASE.format(symbol=ticker)
    rows    = []
    max_id  = None

    for page in range(max_pages):
        params: dict = {"limit": 30}
        if max_id:
            params["max"] = max_id

        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 429:
                log.warning(f"StockTwits rate limit hit for {ticker}. Sleeping 60s.")
                time.sleep(60)
                continue
            if resp.status_code != 200:
                log.warning(f"StockTwits {resp.status_code} for {ticker}")
                break

            data = resp.json()
            msgs = data.get("messages", [])
            if not msgs:
                break

            for msg in msgs:
                sentiment = (msg.get("entities") or {}).get("sentiment") or {}
                label     = sentiment.get("basic")
                text      = msg.get("body", "")
                score     = _label_to_score(label, text)

                rows.append({
                    "post_id":   str(msg["id"]),
                    "symbol":    ticker,
                    "subreddit": "stocktwits",
                    "ts":        datetime.strptime(
                                     msg["created_at"], "%Y-%m-%dT%H:%M:%SZ"
                                 ).replace(tzinfo=None),
                    "title":     text[:500],
                    "upvotes":   (msg.get("likes") or {}).get("total", 0),
                    "num_comments": 0,
                    "compound":  score,
                    "pos":       max(score, 0),
                    "neg":       min(score, 0),
                    "neu":       1.0 - abs(score),
                })

            max_id = msgs[-1]["id"] - 1
            time.sleep(sleep)

        except Exception as e:
            log.warning(f"StockTwits error for {ticker} page {page}: {e}")
            break

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).drop_duplicates("post_id")
