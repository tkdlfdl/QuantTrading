"""
PullPush fetcher — community Pushshift mirror, no Reddit credentials needed.
Endpoint: https://api.pullpush.io/reddit/search/submission/

Provides historical Reddit posts going back years without any API key.
"""
from __future__ import annotations

import time
import logging
from datetime import datetime

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

log    = logging.getLogger(__name__)
_vader = SentimentIntensityAnalyzer()
_BASE  = "https://api.pullpush.io/reddit/search/submission/"
_SUBS  = ["wallstreetbets", "stocks", "investing", "StockMarket", "options"]


def _score(text: str) -> dict:
    s = _vader.polarity_scores(text)
    return {"compound": s["compound"], "pos": s["pos"], "neg": s["neg"], "neu": s["neu"]}


def fetch_pullpush(
    ticker: str,
    start: str = "2020-01-01",
    end: str | None = None,
    subreddits: list[str] = _SUBS,
    size: int = 100,
    sleep: float = 1.0,
) -> pd.DataFrame:
    """
    Fetch Reddit posts via PullPush (no credentials needed).
    Paginates through results in 100-post chunks.
    """
    start_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
    end_ts   = int(datetime.utcnow().timestamp()) if end is None else \
               int(datetime.strptime(end, "%Y-%m-%d").timestamp())

    rows = []

    for sr in subreddits:
        after = start_ts
        while after < end_ts:
            params = {
                "q":         f"${ticker}",
                "subreddit": sr,
                "after":     after,
                "before":    end_ts,
                "size":      size,
                "sort":      "asc",
            }
            try:
                resp = requests.get(_BASE, params=params, timeout=15)
                if resp.status_code == 429:
                    log.warning("PullPush rate limit. Sleeping 30s.")
                    time.sleep(30)
                    continue
                if resp.status_code != 200:
                    log.warning(f"PullPush {resp.status_code} for {ticker}/{sr}")
                    break

                data  = resp.json().get("data", [])
                if not data:
                    break

                for post in data:
                    text = f"{post.get('title', '')} {post.get('selftext', '')}"
                    sc   = _score(text)
                    rows.append({
                        "post_id":      post.get("id", ""),
                        "symbol":       ticker,
                        "subreddit":    sr,
                        "ts":           datetime.utcfromtimestamp(post.get("created_utc", 0)),
                        "title":        post.get("title", "")[:500],
                        "upvotes":      max(post.get("score", 0), 0),
                        "num_comments": post.get("num_comments", 0),
                        **sc,
                    })

                # Advance window to after last post
                after = max(p.get("created_utc", after) for p in data) + 1

                if len(data) < size:
                    break

                time.sleep(sleep)

            except Exception as e:
                log.warning(f"PullPush error {ticker}/{sr}: {e}")
                break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates("post_id")
    df["ts"] = pd.to_datetime(df["ts"])
    return df.sort_values("ts").reset_index(drop=True)
