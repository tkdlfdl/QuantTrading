"""
Multi-strategy portfolio runner.

Combines:
  1. Momentum + Bubble Hedge + Leverage (from run_momentum_bubble.py)
  2. Reddit 4-zone Sentiment (from run_reddit_sentiment.py)

Allocates capital between them using:
  - Fixed weights (grid search: 0%, 10%, ... 100% per strategy)
  - Momentum-based (allocate proportional to recent strategy performance,
    hold for hold_period days, then rebalance based on new lookback)

Usage:
    python run_portfolio.py
"""
import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_n = [0]
def _save(*a, **k):
    _n[0] += 1
    plt.savefig(f"portfolio_chart_{_n[0]}.png", dpi=120, bbox_inches="tight")
    print(f"[Chart saved: portfolio_chart_{_n[0]}.png]")
plt.show = _save

sys.path.insert(0, ".")
import yfinance as yf
import pandas as pd
import requests
from io import StringIO

from data.db.schema import init
from data.universe import get_universe, get_top_by_marketcap
from data.sentiment.aggregator import load_sentiment_panel
from strategies.momentum_bubble_hedge import run_momentum_bubble_hedge_and_low_bubble_leverage
from strategies.reddit_sentiment_bubble import run_reddit_sentiment_bubble
from portfolio.constructor import run_portfolio_allocation

# ── Config ─────────────────────────────────────────────────────────────────
# Use overlapping period so both strategies have data
START   = "2024-01-01"

# Momentum strategy best params (from full backtest)
MOM_PARAMS = dict(
    lookback=140, holding_period=40, LongShort_flag=True, top=5,
    bubble_indicator_grid=["QQQ", "SPY", "Momentum"],
    ma_window_grid=[120], z_window_grid=[240],
    hedge_bubble_entry_grid=[0.85], hedge_alloc_grid=[0.5], hedge_hold_days_grid=[40],
    low_bubble_entry_grid=[-0.88],
    momentum_extra_leverage_grid=[0.25],
    leverage_hold_days_grid=[50],
)

# Reddit sentiment best params
RED_PARAMS = dict(
    holding_period_grid=[5],
    mild_threshold_grid=[0.5],
    extreme_threshold_grid=[0.9],
    top_n_grid=[5],
    ma_window_grid=[30],
    z_window_grid=[60],
    sentiment_scale_grid=[0.05],
    min_mentions=3,
)


def get_momentum_returns(start: str) -> pd.Series:
    """Run momentum strategy and return daily return series."""
    print("=== Running Momentum strategy ===")
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    session = requests.Session()
    session.headers.update(HEADERS)

    def scrape(url, col):
        html = session.get(url, timeout=20)
        for t in pd.read_html(StringIO(html.text)):
            if col in t.columns:
                return t[col].dropna().astype(str).str.strip().str.replace(".", "-", regex=False).unique().tolist()
        return []

    sp500     = scrape("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol")
    nasdaq100 = scrape("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker")
    ticker    = list(set(nasdaq100 + sp500 + ["TMF", "TLT"]))
    stock_list = (ticker + ["QQQ", "SPY", "UVXY", "^VIX", "^FVX",
                            "2Y", "US2Y", "DGS2", "10Y", "US10Y", "^TNX", "DGS10"])

    print(f"Downloading {len(stock_list)} tickers from {start}...")
    df_mom = yf.download(tickers=stock_list, start=start, progress=True)
    close  = df_mom.bfill().ffill().dropna(axis="columns")["Close"].copy()
    df     = {"Close": close}

    results = run_momentum_bubble_hedge_and_low_bubble_leverage(df, **MOM_PARAMS)
    # results[3] = ret_df, column "Momentum_HighBubbleHedge_LowBubbleLeverage"
    ret_df = results[3]
    return ret_df["Momentum_HighBubbleHedge_LowBubbleLeverage"].rename("Momentum")


def get_reddit_returns(start: str) -> pd.Series:
    """Run Reddit sentiment strategy — top-50 by market cap (best signal quality)."""
    print("\n=== Running Reddit Sentiment strategy ===")
    universe = get_top_by_marketcap(get_universe(), n=50)
    print(f"Universe: {len(universe)} tickers (top-50 by mktcap)")
    raw      = yf.download(tickers=universe, start=start, progress=False, auto_adjust=True)
    price    = raw["Close"].ffill().dropna(axis="columns")
    sent     = load_sentiment_panel(universe, start=start)
    common   = [s for s in sent.columns if s in price.columns]
    print(f"Symbols with sentiment: {len(common)}")

    results = run_reddit_sentiment_bubble(
        sentiment_panel=sent[common],
        price_panel=price[common],
        transaction_cost=0.001,   # 0.1% round-trip per position
        cash_rate=0.02,           # 2% annual on idle cash
        short_borrow_rate=0.08,   # 8% annual short borrow cost
        **RED_PARAMS,
    )
    return results[3].rename("Reddit")  # best_ret


if __name__ == "__main__":
    init()

    # ── Get individual strategy returns ────────────────────────────────────
    mom_ret    = get_momentum_returns(START)
    reddit_ret = get_reddit_returns(START)

    # ── Align to common dates ──────────────────────────────────────────────
    common_idx = mom_ret.index.intersection(reddit_ret.index)
    mom_ret    = mom_ret.loc[common_idx]
    reddit_ret = reddit_ret.loc[common_idx]

    print(f"\nOverlapping period: {common_idx[0].date()} to {common_idx[-1].date()}")
    print(f"Momentum  — Sharpe: {(mom_ret.mean()/mom_ret.std()*(252**0.5)):.3f}")
    print(f"Reddit    — Sharpe: {(reddit_ret.mean()/reddit_ret.std()*(252**0.5)):.3f}")

    # ── Run portfolio allocation ───────────────────────────────────────────
    print("\n=== Portfolio Allocation ===")
    (
        analysis, yearly_analysis,
        best_ret, best_wealth, best_weights_ts,
        grid_df, best_params,
    ) = run_portfolio_allocation(
        returns_dict={
            "Momentum": mom_ret,
            "Reddit":   reddit_ret,
        },
        method="both",
        weight_step=0.1,
        lookback_grid=[20, 40, 60, 120],
        hold_period_grid=[5, 10, 20, 40],
    )

    S = "=" * 60
    print(f"\n{S}\nOVERALL PERFORMANCE\n{S}")
    print(analysis.to_string())

    print(f"\n{S}\nYEARLY PERFORMANCE\n{S}")
    print(yearly_analysis.to_string())

    print(f"\n{S}\nTOP 10 ALLOCATIONS BY SHARPE\n{S}")
    print(grid_df.head(10).to_string(index=False))
