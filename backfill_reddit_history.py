"""
backfill_reddit_history.py
==========================
Backfill historical Reddit sentiment via Arctic Shift (free, no credentials).
https://arctic-shift.photon-reddit.com

Fetches ALL posts from finance subreddits in weekly chunks, extracts ticker
mentions, scores with VADER, and stores in the DuckDB sentiment DB.

Approach: subreddit-level with cursor pagination -- 5 subs x N weeks.
Arctic Shift has full Reddit history from 2005 onward, no rate-limit issues.

Usage:
    python backfill_reddit_history.py                     # 2019-01-01 to start of existing data
    python backfill_reddit_history.py --start 2021-01-01 --end 2022-12-31
    python backfill_reddit_history.py --start 2019-01-01 --chunk-days 7 --sleep 1.5

Estimated time: ~5-8 hours for 5 years (2019-2024). Run overnight.
"""
from __future__ import annotations
import sys, time, argparse, datetime, logging
import requests
import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
BASE        = "https://arctic-shift.photon-reddit.com/api/posts/search"
SUBREDDITS  = ["wallstreetbets", "stocks", "investing", "StockMarket", "options"]
PAGE_LIMIT  = 100       # max posts per request (Arctic Shift supports 100)
SLEEP_OK    = 1.5       # seconds between successful page requests
SLEEP_429   = 60.0      # seconds on rate-limit
SLEEP_ERR   = 10.0      # seconds on errors
MAX_RETRY   = 3
TIMEOUT     = 30        # seconds per request
FIELDS      = "id,title,selftext,score,num_comments,created_utc"
HEADERS     = {"User-Agent": "QuantTrading-backfill/1.0"}

_vader = SentimentIntensityAnalyzer()

# ── Ticker extraction ─────────────────────────────────────────────────────────
import re
_STOP = {
    "A","I","DD","CEO","CFO","IPO","ETF","USA","FDA","SEC","FOMO","YOLO","FUD",
    "ATH","EPS","PE","PT","TA","IMO","IMHO","LOL","WTF","USD","GDP","CPI","FED",
    "EOD","AH","PM","OG","RH","WSB","ER","PR","AI","EV","IT","ON","BE","OR","SO",
    "GO","NOW","ALL","ANY","CAN","NEW","ONE","OUT","BIG","BUY","RED","HOT","TOP",
    "ARE","FOR","YOU","THE","AND","NOT","BUT","WIN","CASH","CALL","PUT","HOLD",
    "MOON","BEAR","BULL","LONG","SHORT","GAIN","LOSS","RISK","Q1","Q2","Q3","Q4",
}
_CASHTAG = re.compile(r"\$([A-Za-z]{1,5})\b")
_BARE    = re.compile(r"\b([A-Z]{2,5})\b")

def _extract_tickers(text: str, universe: set) -> set:
    found = set()
    for m in _CASHTAG.findall(text or ""):
        if m.upper() in universe:
            found.add(m.upper())
    for m in _BARE.findall(text or ""):
        if m in universe and m not in _STOP:
            found.add(m)
    return found

def _score(text: str) -> dict:
    s = _vader.polarity_scores(text or "")
    return {"compound": s["compound"], "pos": s["pos"], "neg": s["neg"], "neu": s["neu"]}


# ── Arctic Shift paginator ────────────────────────────────────────────────────
def fetch_subreddit_window(subreddit: str, after_ts: int, before_ts: int,
                           sleep: float) -> list[dict]:
    """
    Paginate Arctic Shift for one subreddit over [after_ts, before_ts].
    Returns list of raw post dicts.
    """
    all_posts = []
    cursor    = after_ts

    while cursor < before_ts:
        params = {
            "subreddit": subreddit,
            "after":     cursor,
            "before":    before_ts,
            "limit":     PAGE_LIMIT,
            "sort":      "asc",
            "fields":    FIELDS,
        }
        data = None
        for attempt in range(MAX_RETRY):
            try:
                r = requests.get(BASE, params=params, headers=HEADERS, timeout=TIMEOUT)
                if r.status_code == 429:
                    log.warning(f"    Rate limit on r/{subreddit}. Sleeping {SLEEP_429}s...")
                    time.sleep(SLEEP_429)
                    continue
                if r.status_code != 200:
                    log.warning(f"    r/{subreddit} HTTP {r.status_code}. Retrying...")
                    time.sleep(SLEEP_ERR)
                    continue
                data = r.json().get("data", [])
                break
            except Exception as e:
                log.warning(f"    r/{subreddit} attempt {attempt+1}: {e}")
                time.sleep(SLEEP_ERR)

        if not data:
            break

        all_posts.extend(data)
        last_ts = max(p.get("created_utc", cursor) for p in data)
        if last_ts <= cursor:
            break
        cursor = last_ts + 1

        if len(data) < PAGE_LIMIT:
            break   # last page

        time.sleep(sleep)

    return all_posts


# ── Build sentiment rows ──────────────────────────────────────────────────────
def posts_to_rows(raw: list[dict], subreddit: str, universe: set) -> list[dict]:
    rows = []
    for p in raw:
        text = f"{p.get('title','') or ''} {p.get('selftext','') or ''}"
        tks  = _extract_tickers(text, universe)
        if not tks:
            continue
        ts  = datetime.datetime.utcfromtimestamp(p.get("created_utc", 0))
        sc  = _score(p.get("title", ""))
        for tk in tks:
            rows.append({
                "post_id":      str(p.get("id", "")),
                "symbol":       tk,
                "subreddit":    subreddit,
                "ts":           ts,
                "title":        (p.get("title", "") or "")[:300],
                "upvotes":      max(int(p.get("score", 0) or 0), 0),
                "num_comments": int(p.get("num_comments", 0) or 0),
                **sc,
            })
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",      default="2019-01-01")
    parser.add_argument("--end",        default=None,
                        help="End date (default: day before earliest existing post)")
    parser.add_argument("--chunk-days", type=int, default=7,
                        help="Days per chunk (default: 7)")
    parser.add_argument("--sleep",      type=float, default=SLEEP_OK,
                        help=f"Sleep between requests in seconds (default: {SLEEP_OK})")
    parser.add_argument("--subreddits", default=",".join(SUBREDDITS))
    args = parser.parse_args()

    from data.universe import get_universe
    from data.sentiment.aggregator import store_posts, aggregate_daily

    universe = {u.upper() for u in get_universe()}
    log.info(f"Universe: {len(universe)} tickers")

    start_dt = datetime.datetime.strptime(args.start, "%Y-%m-%d")

    if args.end:
        end_dt = datetime.datetime.strptime(args.end, "%Y-%m-%d")
    else:
        import duckdb
        con = duckdb.connect("data/market_data.duckdb", read_only=True)
        row = con.execute("SELECT MIN(ts) FROM sentiment_posts").fetchone()
        con.close()
        if row and row[0]:
            end_dt = pd.Timestamp(row[0]).to_pydatetime().replace(
                hour=0, minute=0, second=0, microsecond=0)
            log.info(f"Auto end: {end_dt.date()} (before earliest existing post)")
        else:
            end_dt = datetime.datetime.utcnow()

    if start_dt >= end_dt:
        log.info("Nothing to backfill (start >= end). Done.")
        return

    subs  = [s.strip() for s in args.subreddits.split(",")]
    chunk = datetime.timedelta(days=args.chunk_days)

    chunks = []
    cur = start_dt
    while cur < end_dt:
        chunks.append((cur, min(cur + chunk, end_dt)))
        cur += chunk

    total_req_est = len(chunks) * len(subs)
    log.info(f"Backfill: {start_dt.date()} -> {end_dt.date()}  "
             f"chunk={args.chunk_days}d  subs={subs}")
    log.info(f"Chunks: {len(chunks)}  Est. min requests: {total_req_est}  "
             f"(more for high-volume subs like WSB)")

    total_posts = 0
    total_new   = 0
    t0 = time.time()

    for ci, (cs, ce) in enumerate(chunks):
        after_ts  = int(cs.timestamp())
        before_ts = int(ce.timestamp())
        chunk_rows = []

        for sub in subs:
            raw  = fetch_subreddit_window(sub, after_ts, before_ts, args.sleep)
            rows = posts_to_rows(raw, sub, universe)
            chunk_rows.extend(rows)
            log.info(f"  [{ci+1}/{len(chunks)}] r/{sub:15s} "
                     f"{cs.date()} -> {ce.date()}: "
                     f"{len(raw):4d} posts -> {len(rows):4d} ticker mentions")
            time.sleep(args.sleep)

        if chunk_rows:
            df    = pd.DataFrame(chunk_rows).drop_duplicates(subset=["post_id","symbol"])
            n_new = store_posts(df)
            total_posts += len(df)
            total_new   += n_new

        # Progress + ETA every 4 chunks
        if (ci + 1) % 4 == 0 or ci == 0:
            elapsed  = time.time() - t0
            pct      = (ci + 1) / len(chunks)
            eta_s    = (elapsed / pct) * (1 - pct) if pct > 0 else 0
            eta_h    = eta_s / 3600
            log.info(f"  Progress: {ci+1}/{len(chunks)} chunks | "
                     f"{total_new} new posts | ETA {eta_h:.1f}h")

    # Final aggregation over all touched symbols
    log.info("Re-aggregating sentiment_daily...")
    agg = aggregate_daily()
    elapsed_h = (time.time() - t0) / 3600

    log.info(f"Done in {elapsed_h:.1f}h. {total_new} new posts stored.")
    if not agg.empty:
        log.info(f"sentiment_daily now: {agg['date'].min().date()} -> "
                 f"{agg['date'].max().date()}  ({len(agg)} symbol-day rows)")


if __name__ == "__main__":
    main()
