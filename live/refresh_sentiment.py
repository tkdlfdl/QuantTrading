"""
live/refresh_sentiment.py
=========================
Daily Reddit-sentiment refresh for Book E.

Scans finance subreddits for fresh ticker mentions, scores them (VADER), stores
the posts, and re-aggregates into the `sentiment_daily` table so Book E trades on
current sentiment instead of stale data.

Source preference:
  1. PRAW (Reddit official API)  — RELIABLE. Needs free Reddit app credentials
     (REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET env, or live/state/reddit_creds.json).
  2. PullPush (no credentials)   — best-effort fallback; rate-limited & lagged.

Usage:
  python -m live.refresh_sentiment
  python -m live.refresh_sentiment --source pullpush   # force fallback
"""
from __future__ import annotations
import sys, datetime as dt
from . import config as C


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    force = "--source" in argv and argv[argv.index("--source")+1] or None

    from data.universe import get_universe
    from data.sentiment import reddit_feed_scanner as RFS
    from data.sentiment.aggregator import store_posts, aggregate_daily

    universe = get_universe()
    cid, csec = C.load_reddit_creds()

    print(f"Sentiment refresh — {dt.date.today()}")
    if force == "pullpush" or not (cid and csec):
        if not (cid and csec) and force != "pullpush":
            print("  No Reddit API credentials — using PullPush fallback (best-effort).")
        print("  Source: PullPush")
        posts = RFS.scan_via_pullpush(universe)
    else:
        print("  Source: PRAW (Reddit official API)")
        posts = RFS.scan_via_praw(universe, cid, csec)

    if posts.empty:
        print("  No fresh ticker mentions fetched (source blocked/empty). "
              "sentiment_daily unchanged.")
        return

    n_new = store_posts(posts)
    print(f"  Fetched {len(posts)} ticker-mentions; {n_new} new posts stored.")
    # re-aggregate only the symbols we touched (fast, targeted upsert)
    syms = sorted(posts["symbol"].unique().tolist())
    agg = aggregate_daily(symbols=syms)
    if not agg.empty:
        latest = agg.sort_values("date").tail(1)["date"].iloc[0]
        today_rows = agg[agg["date"] == agg["date"].max()]
        print(f"  Re-aggregated {len(syms)} symbols; latest sentiment date now {agg['date'].max().date()}.")
        top = today_rows.sort_values("mention_count", ascending=False).head(10)
        print("  Most-discussed today:")
        for _, r in top.iterrows():
            print(f"    {r['symbol']:<6} mentions={int(r['mention_count']):<3} "
                  f"sentiment={r['weighted_compound']:+.2f}")
    print("Done.")


if __name__ == "__main__":
    main()
