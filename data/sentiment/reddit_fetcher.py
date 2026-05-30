"""
Reddit post fetcher using PRAW (recent data) with Pushshift fallback (historical).

PRAW gives ~1-2 years via search. Pushshift (via pmaw) extends further back.
All posts are scored with VADER inline and stored in the DB.

Reddit API credentials required — create an app at https://www.reddit.com/prefs/apps
Then set in config/settings.py or pass directly.
"""
from __future__ import annotations

import time
import logging
from datetime import datetime, timezone, timedelta

import pandas as pd
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

log = logging.getLogger(__name__)

SUBREDDITS = ["wallstreetbets", "stocks", "investing", "StockMarket", "options"]
_vader = SentimentIntensityAnalyzer()


def _score(text: str) -> dict:
    s = _vader.polarity_scores(text)
    return {"compound": s["compound"], "pos": s["pos"], "neg": s["neg"], "neu": s["neu"]}


def _make_reddit(client_id: str, client_secret: str, user_agent: str = "QuantTrading/1.0") -> praw.Reddit:
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        ratelimit_seconds=5,
    )


# ---------------------------------------------------------------------------
# PRAW — recent posts (up to ~1 year via search)
# ---------------------------------------------------------------------------

def fetch_recent(
    ticker: str,
    reddit: praw.Reddit,
    subreddits: list[str] = SUBREDDITS,
    limit: int = 500,
) -> pd.DataFrame:
    """
    Fetch recent posts mentioning $TICKER via PRAW search.
    Returns DataFrame with VADER scores attached.
    """
    sub  = reddit.subreddit("+".join(subreddits))
    rows = []

    for sort in ["new", "top"]:
        try:
            for post in sub.search(f"${ticker}", limit=limit, sort=sort, time_filter="all"):
                text = f"{post.title} {post.selftext}"
                sc   = _score(text)
                rows.append({
                    "post_id":      post.id,
                    "symbol":       ticker,
                    "subreddit":    post.subreddit.display_name.lower(),
                    "ts":           datetime.fromtimestamp(post.created_utc, tz=timezone.utc).replace(tzinfo=None),
                    "title":        post.title[:500],
                    "upvotes":      max(post.score, 0),
                    "num_comments": post.num_comments,
                    **sc,
                })
        except Exception as e:
            log.warning(f"PRAW error for {ticker} ({sort}): {e}")
        time.sleep(1)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates("post_id")
    return df


# ---------------------------------------------------------------------------
# Pushshift (via pmaw) — historical posts
# ---------------------------------------------------------------------------

def fetch_historical(
    ticker: str,
    start: str,
    end: str | None = None,
    subreddits: list[str] = SUBREDDITS,
    limit: int = 10_000,
) -> pd.DataFrame:
    """
    Fetch historical posts via Pushshift (pmaw).
    start / end: 'YYYY-MM-DD' strings.
    Falls back gracefully if Pushshift is unavailable.
    """
    try:
        from pmaw import PushshiftAPI
        api = PushshiftAPI()
    except Exception:
        log.warning("pmaw not available or Pushshift unreachable — skipping historical fetch.")
        return pd.DataFrame()

    start_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
    end_ts   = int(datetime.strptime(end, "%Y-%m-%d").timestamp()) if end else int(datetime.utcnow().timestamp())

    rows = []
    for sr in subreddits:
        try:
            posts = api.search_submissions(
                q=f"${ticker}",
                subreddit=sr,
                after=start_ts,
                before=end_ts,
                limit=limit,
                safe_exit=True,
            )
            for post in posts:
                title = post.get("title", "")
                body  = post.get("selftext", "")
                text  = f"{title} {body}"
                sc    = _score(text)
                rows.append({
                    "post_id":      post.get("id", ""),
                    "symbol":       ticker,
                    "subreddit":    sr,
                    "ts":           datetime.utcfromtimestamp(post.get("created_utc", 0)),
                    "title":        title[:500],
                    "upvotes":      max(post.get("score", 0), 0),
                    "num_comments": post.get("num_comments", 0),
                    **sc,
                })
        except Exception as e:
            log.warning(f"Pushshift error for {ticker}/{sr}: {e}")

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).drop_duplicates("post_id")


# ---------------------------------------------------------------------------
# Combined fetch — try historical first, then top up with PRAW
# ---------------------------------------------------------------------------

def fetch_all(
    ticker: str,
    reddit: praw.Reddit,
    start: str = "2020-01-01",
    subreddits: list[str] = SUBREDDITS,
) -> pd.DataFrame:
    """Fetch all available posts: Pushshift for history + PRAW for recent."""
    dfs = []

    hist = fetch_historical(ticker, start=start, subreddits=subreddits)
    if not hist.empty:
        dfs.append(hist)

    recent = fetch_recent(ticker, reddit, subreddits=subreddits)
    if not recent.empty:
        dfs.append(recent)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True).drop_duplicates("post_id")
    combined["ts"] = pd.to_datetime(combined["ts"])
    combined = combined.sort_values("ts").reset_index(drop=True)
    return combined
