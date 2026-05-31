"""
Reddit public JSON fetcher — no API credentials needed.

Uses Reddit's public *.json endpoints (same feed as the website).
Works for any public subreddit. Returns up to 1000 posts per search.
Rate limit: ~1 req/sec to be safe.
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

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

SUBREDDITS = ["wallstreetbets", "stocks", "investing", "StockMarket"]


def _score(text: str) -> dict:
    s = _vader.polarity_scores(text)
    return {"compound": s["compound"], "pos": s["pos"],
            "neg": s["neg"], "neu": s["neu"]}


def fetch_reddit_json(
    ticker: str,
    subreddits: list[str] = SUBREDDITS,
    limit: int = 100,
    sleep: float = 2.0,
) -> pd.DataFrame:
    """
    Fetch posts mentioning $TICKER from Reddit's public JSON API.
    No credentials required — uses the same feed as reddit.com.
    Returns up to limit posts per subreddit.
    """
    rows = []

    for sr in subreddits:
        url    = f"https://www.reddit.com/r/{sr}/search.json"
        params = {
            "q":           f"${ticker}",
            "sort":        "new",
            "limit":       limit,
            "restrict_sr": "on",
            "t":           "all",
        }

        try:
            resp = requests.get(url, headers=_HEADERS, params=params, timeout=10)
            if resp.status_code == 429:
                log.warning(f"Reddit rate limit for {ticker}/{sr}. Sleeping 30s.")
                time.sleep(30)
                continue
            if resp.status_code != 200:
                log.warning(f"Reddit JSON {resp.status_code} for {ticker}/{sr}")
                time.sleep(sleep)
                continue

            posts = resp.json().get("data", {}).get("children", [])
            for post in posts:
                p    = post.get("data", {})
                text = f"{p.get('title', '')} {p.get('selftext', '')}"
                sc   = _score(text)
                rows.append({
                    "post_id":      p.get("id", ""),
                    "symbol":       ticker,
                    "subreddit":    sr,
                    "ts":           datetime.utcfromtimestamp(p.get("created_utc", 0)),
                    "title":        p.get("title", "")[:500],
                    "upvotes":      max(p.get("score", 0), 0),
                    "num_comments": p.get("num_comments", 0),
                    **sc,
                })

            time.sleep(sleep)

        except Exception as e:
            log.warning(f"Reddit JSON error {ticker}/{sr}: {e}")
            time.sleep(sleep)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates("post_id")
    df["ts"] = pd.to_datetime(df["ts"])
    return df.sort_values("ts").reset_index(drop=True)
