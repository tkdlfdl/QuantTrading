"""
GDELT (Global Database of Events, Language, and Tone) news fetcher.
No authentication, no API key — completely free, historical data back to 2015.

Queries the GDELT DOC 2.0 API for news articles mentioning each company,
scores titles with VADER to produce sentiment.

Rate limit: ~1 req/sec recommended.
"""
from __future__ import annotations

import time
import logging
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import requests
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

log    = logging.getLogger(__name__)
_vader = SentimentIntensityAnalyzer()
_BASE  = "https://api.gdeltproject.org/api/v2/doc/doc"

# Cache company names to avoid repeated yfinance calls
_name_cache: dict[str, str] = {}


def _get_company_name(ticker: str) -> str:
    if ticker in _name_cache:
        return _name_cache[ticker]
    try:
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName") or ticker
        # Remove common suffixes for cleaner search
        for suffix in [" Inc.", " Inc", " Corp.", " Corp", " Ltd.", " Ltd",
                       " Co.", " Co", " plc", " PLC", " N.V.", " NV"]:
            name = name.replace(suffix, "")
        _name_cache[ticker] = name.strip()
    except Exception:
        _name_cache[ticker] = ticker
    return _name_cache[ticker]


def _fetch_chunk(query: str, start: datetime, end: datetime, max_records: int = 250) -> list[dict]:
    params = {
        "query":         query,
        "mode":          "artlist",
        "format":        "json",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime":   end.strftime("%Y%m%d%H%M%S"),
        "maxrecords":    max_records,
        "sort":          "DateDesc",
        "sourcelang":    "English",
    }
    try:
        resp = requests.get(_BASE, params=params, timeout=8)
        if resp.status_code != 200:
            return []
        return resp.json().get("articles", [])
    except Exception as e:
        log.warning(f"GDELT error: {e}")
        return []


def fetch_gdelt(
    ticker: str,
    start: str = "2020-01-01",
    end: str | None = None,
    chunk_months: int = 3,
    sleep: float = 1.5,
) -> pd.DataFrame:
    """
    Fetch news articles for a ticker from GDELT.
    Chunks requests by quarter to maximise article coverage.
    Scores titles with VADER.
    """
    company = _get_company_name(ticker)
    query   = f'"{company}" (stock OR earnings OR shares OR investor)'

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end, "%Y-%m-%d") if end else datetime.utcnow()

    rows  = []
    chunk = timedelta(days=180)  # 6-month windows — 3 API calls per 1.5 years
    cur   = start_dt

    while cur < end_dt:
        chunk_end = min(cur + chunk, end_dt)
        articles  = _fetch_chunk(query, cur, chunk_end)

        for art in articles:
            title    = art.get("title", "")
            date_str = art.get("seendate", "")
            url      = art.get("url", "")

            if not title:
                continue
            try:
                ts = datetime.strptime(date_str[:15], "%Y%m%dT%H%M%S")
            except Exception:
                continue

            sc = _vader.polarity_scores(title)
            rows.append({
                "post_id":      url[:200] or f"{ticker}_{ts}",
                "symbol":       ticker,
                "subreddit":    f"gdelt_{art.get('domain', 'news')}",
                "ts":           ts,
                "title":        title[:500],
                "upvotes":      0,
                "num_comments": 0,
                "compound":     sc["compound"],
                "pos":          sc["pos"],
                "neg":          sc["neg"],
                "neu":          sc["neu"],
            })

        cur = chunk_end + timedelta(seconds=1)
        time.sleep(sleep)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates("post_id")
    df["ts"] = pd.to_datetime(df["ts"])
    return df.sort_values("ts").reset_index(drop=True)
