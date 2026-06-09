"""
Reddit subreddit-feed scanner — no credentials, efficient daily refresh.

Instead of searching per-ticker (515 requests), it pulls the recent feed from a
handful of finance subreddits (a few requests), extracts ticker mentions against
the universe, scores each post with VADER, and returns a posts DataFrame in the
sentiment_posts schema (post_id, symbol, subreddit, ts, title, upvotes,
num_comments, compound, pos, neg, neu).
"""
from __future__ import annotations
import re, time, logging
from datetime import datetime, timezone, timedelta
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

log = logging.getLogger(__name__)
_vader = SentimentIntensityAnalyzer()

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/122.0.0.0 Safari/537.36",
            "Accept": "application/json"}

SUBREDDITS = ["wallstreetbets", "stocks", "investing", "StockMarket", "options"]
FEEDS = ["new", "hot"]

# Common all-caps words that collide with tickers — excluded from bare-token matching
_STOP = {"A","I","DD","CEO","CFO","IPO","ETF","USA","FDA","SEC","FOMO","YOLO","FUD",
         "ATH","EPS","PE","PT","TA","IMO","IMHO","LOL","WTF","USD","GDP","CPI","FED",
         "EOD","AH","PM","OG","RH","WSB","ER","PR","AI","EV","IT","ON","BE","OR","SO",
         "GO","NOW","ALL","ANY","CAN","NEW","ONE","OUT","BIG","BUY","RED","HOT","TOP",
         "ARE","FOR","YOU","THE","AND","NOT","BUT","WIN","CASH","CALL","PUT","HOLD",
         "MOON","BEAR","BULL","LONG","SHORT","GAIN","LOSS","RISK","Q1","Q2","Q3","Q4"}

_CASHTAG = re.compile(r"\$([A-Za-z]{1,5})\b")
_BARE    = re.compile(r"\b([A-Z]{2,5})\b")


def _score(text: str) -> dict:
    s = _vader.polarity_scores(text or "")
    return {"compound": s["compound"], "pos": s["pos"], "neg": s["neg"], "neu": s["neu"]}


def _extract_tickers(text: str, universe: set) -> set:
    found = set()
    for m in _CASHTAG.findall(text or ""):
        u = m.upper()
        if u in universe:
            found.add(u)
    for m in _BARE.findall(text or ""):
        if m in universe and m not in _STOP:
            found.add(m)
    return found


def _fetch_feed(subreddit: str, feed: str, limit: int, sleep: float):
    url = f"https://www.reddit.com/r/{subreddit}/{feed}.json?limit={limit}"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=12)
        r.raise_for_status()
        children = r.json().get("data", {}).get("children", [])
        time.sleep(sleep)
        return children
    except Exception as e:
        log.warning(f"feed {subreddit}/{feed} failed: {e}")
        return []


_COLS = ["post_id","symbol","subreddit","ts","title",
         "upvotes","num_comments","compound","pos","neg","neu"]


def _row(pid, tk, sub, ts, title, ups, ncom):
    sc = _score(title)
    return {"post_id": pid, "symbol": tk, "subreddit": sub,
            "ts": ts, "title": (title or "")[:300],
            "upvotes": int(ups or 0), "num_comments": int(ncom or 0), **sc}


# ── Source 1: PRAW (Reddit official API — reliable, needs credentials) ──────
def scan_via_praw(universe, client_id, client_secret, subreddits=SUBREDDITS,
                  limit=200, verbose=True) -> pd.DataFrame:
    import praw
    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret,
                         user_agent="QuantTrading-sentiment/1.0", ratelimit_seconds=10)
    uni = {u.upper() for u in universe}
    rows = []
    for sub in subreddits:
        try:
            posts = list(reddit.subreddit(sub).new(limit=limit))
        except Exception as e:
            if verbose: print(f"  r/{sub}: PRAW error {e}")
            continue
        n = 0
        for p in posts:
            text = f"{p.title} {getattr(p,'selftext','') or ''}"
            tks = _extract_tickers(text, uni)
            if not tks: continue
            ts = datetime.fromtimestamp(p.created_utc, tz=timezone.utc).replace(tzinfo=None)
            for tk in tks:
                rows.append(_row(p.id, tk, sub, ts, p.title,
                                 getattr(p,"ups",0), getattr(p,"num_comments",0))); n += 1
        if verbose: print(f"  r/{sub}: {len(posts)} posts -> {n} ticker mentions")
    df = pd.DataFrame(rows, columns=_COLS) if rows else pd.DataFrame(columns=_COLS)
    return df.drop_duplicates(subset=["post_id","symbol"]) if not df.empty else df


# ── Source 2: PullPush (no creds, best-effort, rate-limited) ───────────────
def scan_via_pullpush(universe, subreddits=SUBREDDITS, days=2,
                      size=100, sleep=8.0, retries=3, verbose=True) -> pd.DataFrame:
    after = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    uni = {u.upper() for u in universe}
    rows = []
    for sub in subreddits:
        data = None
        for attempt in range(retries):
            try:
                url = (f"https://api.pullpush.io/reddit/search/submission/"
                       f"?subreddit={sub}&after={after}&size={size}&sort=desc")
                r = requests.get(url, headers=_HEADERS, timeout=25)
                if r.status_code == 429:
                    time.sleep(sleep * (attempt + 2)); continue
                r.raise_for_status()
                data = r.json().get("data", [])
                break
            except Exception as e:
                if verbose: print(f"  pullpush r/{sub} attempt {attempt+1}: {e}")
                time.sleep(sleep)
        time.sleep(sleep)
        if not data:
            if verbose: print(f"  r/{sub}: no data (rate-limited/empty)")
            continue
        n = 0
        for p in data:
            text = f"{p.get('title','')} {p.get('selftext','') or ''}"
            tks = _extract_tickers(text, uni)
            if not tks: continue
            ts = datetime.fromtimestamp(p.get("created_utc",0), tz=timezone.utc).replace(tzinfo=None)
            for tk in tks:
                rows.append(_row(p.get("id",""), tk, sub, ts, p.get("title",""),
                                 p.get("ups",0), p.get("num_comments",0))); n += 1
        if verbose: print(f"  r/{sub}: {len(data)} posts -> {n} ticker mentions")
    df = pd.DataFrame(rows, columns=_COLS) if rows else pd.DataFrame(columns=_COLS)
    return df.drop_duplicates(subset=["post_id","symbol"]) if not df.empty else df
