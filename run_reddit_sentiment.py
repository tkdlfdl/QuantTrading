"""
Reddit Sentiment Bubble Strategy — full pipeline.

Steps:
  1. Fetch Reddit posts for each ticker (PRAW + Pushshift)
  2. Store raw posts + aggregate daily scores in DB
  3. Load price data via yf.download()
  4. Run sentiment bubble backtest with grid search

Setup required (one-time):
  Create a Reddit app at https://www.reddit.com/prefs/apps
  Set your credentials below or in environment variables.

Usage:
    python run_reddit_sentiment.py
"""
import sys, os
sys.path.insert(0, ".")

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import yfinance as yf
import pandas as pd
import requests
from io import StringIO

from data.db.schema import init
from data.universe import get_reddit_universe, get_top_by_marketcap, get_universe
from data.sentiment.reddit_fetcher import fetch_all, _make_reddit, SUBREDDITS
from data.sentiment.aggregator import store_posts, aggregate_daily, load_sentiment_panel
from strategies.reddit_sentiment_bubble import run_reddit_sentiment_bubble

# ── Reddit API credentials ─────────────────────────────────────────────────
REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID",     "YOUR_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "YOUR_CLIENT_SECRET")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT",    "QuantTrading/1.0")

SENTIMENT_START = "2020-01-01"
PRICE_START     = "2020-01-01"
MIN_MENTIONS    = 10

# ── Universe mode ──────────────────────────────────────────────────────────
# "reddit"   → NASDAQ100 + WSB classics   (~120 tickers, ~30-45 min)  ← default
# "top100"   → top 100 by market cap      (~100 tickers, ~25-35 min)
# "top50"    → top 50 by market cap       (~50  tickers, ~15-20 min)
# "full"     → NASDAQ100 + S&P500         (~516 tickers, ~2-5 hrs)
UNIVERSE_MODE = "reddit"


def step1_fetch_sentiment(symbols: list[str], reddit) -> None:
    """Fetch Reddit posts for each symbol and store in DB."""
    total_stored = 0
    for i, sym in enumerate(symbols):
        print(f"  [{i+1}/{len(symbols)}] Fetching {sym}...", end=" ", flush=True)
        try:
            posts = fetch_all(sym, reddit, start=SENTIMENT_START, subreddits=SUBREDDITS)
            if posts.empty:
                print("no posts")
                continue
            n = store_posts(posts)
            total_stored += n
            print(f"{len(posts)} posts, {n} new stored")
        except Exception as e:
            print(f"ERROR: {e}")
    print(f"\nTotal new posts stored: {total_stored:,}")


def step2_aggregate() -> pd.DataFrame:
    """Aggregate raw posts to daily sentiment scores."""
    print("Aggregating daily sentiment scores...")
    agg = aggregate_daily()
    print(f"Aggregated {len(agg):,} symbol-day rows.")
    return agg


def step3_load_prices(symbols: list[str]) -> pd.DataFrame:
    """Download price panel via yf.download()."""
    print(f"\nDownloading price data for {len(symbols)} tickers from {PRICE_START}...")
    df = yf.download(tickers=symbols, start=PRICE_START, progress=True, auto_adjust=True)
    close = df["Close"].ffill().dropna(axis="columns")
    print(f"Price panel: {close.shape}")
    return close


def step4_backtest(symbols: list[str], price_panel: pd.DataFrame) -> None:
    """Load sentiment panel, align with prices, run strategy."""
    sentiment_panel = load_sentiment_panel(symbols, start=PRICE_START)
    if sentiment_panel.empty:
        print("No sentiment data found. Run fetch step first.")
        return

    # Keep only symbols present in both panels
    common_syms = [s for s in sentiment_panel.columns if s in price_panel.columns]
    sentiment_panel = sentiment_panel[common_syms]
    price_panel     = price_panel[common_syms]

    print(f"\nRunning backtest on {len(common_syms)} symbols with sentiment data...")
    print(f"Sentiment period: {sentiment_panel.index[0].date()} to {sentiment_panel.index[-1].date()}")
    print(f"Price period:     {price_panel.index[0].date()} to {price_panel.index[-1].date()}\n")

    analysis, yearly_analysis, best_wealth, best_ret, grid_df, best_params = \
        run_reddit_sentiment_bubble(
            sentiment_panel=sentiment_panel,
            price_panel=price_panel,
            holding_period_grid=[5, 10, 20, 40],
            short_threshold_grid=[0.5, 0.6, 0.7, 0.8, 0.9],
            long_threshold_grid=[-0.5, -0.6, -0.7, -0.8, -0.9],
            top_n_grid=[5, 10, 20],
            ma_window_grid=[30, 60],
            z_window_grid=[60, 120],
            sentiment_scale_grid=[0.05],
            min_mentions=MIN_MENTIONS,
        )

    S = "=" * 60
    print(f"\n{S}\nOVERALL PERFORMANCE\n{S}")
    print(analysis.to_string())

    print(f"\n{S}\nYEARLY PERFORMANCE\n{S}")
    print(yearly_analysis.to_string())

    print(f"\n{S}\nTOP 10 GRID RESULTS BY SHARPE\n{S}")
    cols = ["holding_period", "short_threshold", "long_threshold",
            "top_n", "ma_window", "z_window",
            "Sharpe Ratio", "Sortino Ratio", "Total Return", "Max Drawdown"]
    print(grid_df[cols].head(10).to_string(index=False))


def build_universe() -> list[str]:
    if UNIVERSE_MODE == "reddit":
        return get_reddit_universe()
    elif UNIVERSE_MODE == "top100":
        return get_top_by_marketcap(get_universe(), n=100)
    elif UNIVERSE_MODE == "top50":
        return get_top_by_marketcap(get_universe(), n=50)
    else:  # "full"
        return get_universe()


if __name__ == "__main__":
    init()
    symbols = build_universe()
    print(f"Universe mode: {UNIVERSE_MODE} → {len(symbols)} tickers\n")

    # Validate credentials before starting long fetch
    if REDDIT_CLIENT_ID == "YOUR_CLIENT_ID":
        print("\nReddit API credentials not set.")
        print("Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables.")
        print("Or edit run_reddit_sentiment.py directly.")
        print("\nTo create Reddit API credentials:")
        print("  1. Go to https://www.reddit.com/prefs/apps")
        print("  2. Click 'create another app'")
        print("  3. Select 'script', fill in name/description")
        print("  4. Copy the client_id (under app name) and secret")
        sys.exit(1)

    reddit = _make_reddit(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT)

    print("\n=== Step 1: Fetching Reddit sentiment data ===")
    step1_fetch_sentiment(symbols, reddit)

    print("\n=== Step 2: Aggregating daily scores ===")
    step2_aggregate()

    print("\n=== Step 3: Loading price data ===")
    price_panel = step3_load_prices(symbols)

    print("\n=== Step 4: Running backtest ===")
    step4_backtest(symbols, price_panel)
