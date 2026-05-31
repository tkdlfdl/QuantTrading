"""
Reddit Sentiment Bubble Strategy — full pipeline.

Data sources (no Reddit API credentials required):
  - PullPush: community Pushshift mirror, historical Reddit posts
  - StockTwits: explicit bullish/bearish labels, no auth needed

Usage:
    python run_reddit_sentiment.py
"""
import sys
sys.path.insert(0, ".")

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_fig_n = [0]
def _save(*a, **k):
    _fig_n[0] += 1
    plt.savefig(f"sentiment_chart_{_fig_n[0]}.png", dpi=120, bbox_inches="tight")
    print(f"[Chart {_fig_n[0]} saved as sentiment_chart_{_fig_n[0]}.png]")
plt.show = _save

from data.db.schema import init
from data.universe import get_universe, get_top_by_marketcap
from data.sentiment.gdelt_fetcher import fetch_gdelt
from data.sentiment.yfinance_news_fetcher import fetch_yfinance_news
from data.sentiment.aggregator import store_posts, aggregate_daily, load_sentiment_panel
from strategies.reddit_sentiment_bubble import run_reddit_sentiment_bubble

# ── Config ─────────────────────────────────────────────────────────────────
SENTIMENT_START = "2024-01-01"   # 1.5 years — ~6 GDELT chunks per ticker (~17 min total)
PRICE_START     = "2024-01-01"
MIN_MENTIONS    = 3
TOP_N           = 50


# ── Step 1: Build universe ─────────────────────────────────────────────────
def build_universe() -> list[str]:
    universe = get_universe()
    return get_top_by_marketcap(universe, n=TOP_N)


# ── Step 2: Fetch sentiment ────────────────────────────────────────────────
def fetch_sentiment(symbols: list[str]) -> None:
    """
    Two no-credential sources:
      1. GDELT — historical news back to 2022, VADER scored
      2. Yahoo Finance news — recent articles (top-up)
    """
    import pandas as pd
    total = 0
    for i, sym in enumerate(symbols):
        print(f"  [{i+1}/{len(symbols)}] {sym}...", end=" ", flush=True)
        dfs = []

        # GDELT: historical news
        try:
            df = fetch_gdelt(sym, start=SENTIMENT_START, sleep=0.5)
            if not df.empty:
                dfs.append(df)
                print(f"GDELT:{len(df)}", end=" ", flush=True)
        except Exception as e:
            print(f"GDELT:ERR", end=" ", flush=True)

        # yfinance: recent news top-up
        try:
            df = fetch_yfinance_news(sym)
            if not df.empty:
                dfs.append(df)
                print(f"YF:{len(df)}", end=" ", flush=True)
        except Exception:
            pass

        if dfs:
            combined = pd.concat(dfs, ignore_index=True).drop_duplicates("post_id")
            n = store_posts(combined)
            total += n
            print(f"-> {n} stored")
        else:
            print("no data")

    print(f"\nTotal new posts stored: {total:,}")


# ── Step 3: Aggregate ──────────────────────────────────────────────────────
def aggregate() -> None:
    print("Aggregating daily sentiment scores...")
    agg = aggregate_daily()
    print(f"Aggregated {len(agg):,} symbol-day rows.")


# ── Step 4: Prices ─────────────────────────────────────────────────────────
def load_prices(symbols: list[str]):
    print(f"\nDownloading prices for {len(symbols)} tickers from {PRICE_START}...")
    raw   = yf.download(tickers=symbols, start=PRICE_START, progress=True, auto_adjust=True)
    close = raw["Close"].ffill().dropna(axis="columns")
    print(f"Price panel: {close.shape}")
    return close


# ── Step 5: Backtest ───────────────────────────────────────────────────────
def backtest(symbols: list[str], price_panel) -> None:
    sentiment_panel = load_sentiment_panel(symbols, start=PRICE_START)
    if sentiment_panel.empty:
        print("No sentiment data in DB. Run fetch step first.")
        return

    common = [s for s in sentiment_panel.columns if s in price_panel.columns]
    print(f"\nSymbols with sentiment + price data: {len(common)}")
    print(f"Sentiment: {sentiment_panel.index[0].date()} to {sentiment_panel.index[-1].date()}")
    print(f"Price:     {price_panel.index[0].date()} to {price_panel.index[-1].date()}\n")

    (analysis, yearly_analysis, best_wealth, best_ret, grid_df, best_params) = \
        run_reddit_sentiment_bubble(
            sentiment_panel=sentiment_panel[common],
            price_panel=price_panel[common],
            holding_period_grid=[1, 2, 3, 5],
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


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init()

    print("=== Step 1: Building universe ===")
    symbols = build_universe()
    print(f"Universe: {len(symbols)} tickers\n")

    print("=== Step 2: Fetching sentiment (PullPush + StockTwits) ===")
    fetch_sentiment(symbols)

    print("\n=== Step 3: Aggregating daily scores ===")
    aggregate()

    print("\n=== Step 4: Loading prices ===")
    price_panel = load_prices(symbols)

    print("\n=== Step 5: Running backtest ===")
    backtest(symbols, price_panel)
