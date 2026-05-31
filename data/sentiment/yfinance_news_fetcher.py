"""
Yahoo Finance news fetcher — no credentials needed.
Handles both old and new yfinance response formats.
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

log    = logging.getLogger(__name__)
_vader = SentimentIntensityAnalyzer()


def _parse_article(article: dict, ticker: str) -> dict | None:
    """Handle both yfinance <0.2 and >=1.4 article formats."""
    # New format: {'id': ..., 'content': {'title': ..., 'pubDate': ..., ...}}
    content = article.get("content", {})
    if content:
        title     = content.get("title", "")
        pub_ts    = content.get("pubDate", "")
        publisher = (content.get("provider") or {}).get("displayName", "news")
        post_id   = article.get("id", "")
        try:
            ts = datetime.strptime(pub_ts[:19], "%Y-%m-%dT%H:%M:%S")
        except Exception:
            ts = datetime.utcnow()

    # Old format: {'title': ..., 'uuid': ..., 'providerPublishTime': ..., ...}
    else:
        title     = article.get("title", "")
        publisher = article.get("publisher", "news")
        post_id   = article.get("uuid", "")
        ts        = datetime.utcfromtimestamp(article.get("providerPublishTime", 0))

    if not title:
        return None

    sc = _vader.polarity_scores(title)
    return {
        "post_id":      post_id or f"{ticker}_{ts}",
        "symbol":       ticker,
        "subreddit":    f"yf_{publisher}",
        "ts":           ts,
        "title":        title[:500],
        "upvotes":      0,
        "num_comments": 0,
        "compound":     sc["compound"],
        "pos":          sc["pos"],
        "neg":          sc["neg"],
        "neu":          sc["neu"],
    }


def fetch_yfinance_news(ticker: str) -> pd.DataFrame:
    try:
        news = yf.Ticker(ticker).news
        if not news:
            return pd.DataFrame()
    except Exception as e:
        log.warning(f"yfinance news error for {ticker}: {e}")
        return pd.DataFrame()

    rows = [r for a in news if (r := _parse_article(a, ticker)) is not None]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates("post_id")
    df["ts"] = pd.to_datetime(df["ts"])
    return df.sort_values("ts").reset_index(drop=True)
