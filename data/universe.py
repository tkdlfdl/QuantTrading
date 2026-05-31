"""
Fetch live index constituent lists from Wikipedia.

    from data.universe import get_universe
    tickers = get_universe()                    # NASDAQ 100 + S&P 500, deduplicated
    tickers = get_universe(sp500=False)         # NASDAQ 100 only
    tickers = get_reddit_universe()             # NASDAQ 100 + WSB classics (~120 tickers)
    tickers = get_top_by_marketcap(tickers, 50) # top 50 by market cap
"""
from __future__ import annotations

from io import StringIO
import pandas as pd
import requests
import yfinance as yf

_NDX_URL   = "https://en.wikipedia.org/wiki/Nasdaq-100"
_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_HEADERS   = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}


def _fetch_tables(url: str) -> list[pd.DataFrame]:
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    return pd.read_html(StringIO(resp.text))


def _clean(ticker: str) -> str:
    return ticker.strip().replace(".", "-")


def get_nasdaq100() -> list[str]:
    tables = _fetch_tables(_NDX_URL)
    for t in tables:
        if "Ticker" in t.columns:
            return sorted(_clean(s) for s in t["Ticker"].dropna().tolist())
    raise RuntimeError("NASDAQ-100 ticker column not found on Wikipedia page.")


def get_sp500() -> list[str]:
    tables = _fetch_tables(_SP500_URL)
    for t in tables:
        if "Symbol" in t.columns:
            return sorted(_clean(s) for s in t["Symbol"].dropna().tolist())
    raise RuntimeError("S&P 500 Symbol column not found on Wikipedia page.")


def get_universe(nasdaq100: bool = True, sp500: bool = True) -> list[str]:
    """Return deduplicated sorted ticker list for the requested indices."""
    tickers: set[str] = set()
    if nasdaq100:
        ndx = get_nasdaq100()
        tickers.update(ndx)
        print(f"NASDAQ 100: {len(ndx)} tickers")
    if sp500:
        sp  = get_sp500()
        tickers.update(sp)
        print(f"S&P 500:    {len(sp)} tickers")
    combined = sorted(tickers)
    print(f"Combined universe (unique): {len(combined)} tickers")
    return combined


# Stocks consistently popular on WSB / Reddit that may not be in NASDAQ 100
_WSB_CLASSICS = [
    "GME", "AMC", "BBBY", "BB", "NOK",        # original meme stocks
    "PLTR", "SOFI", "HOOD", "COIN", "RIVN",    # newer retail favourites
    "LCID", "NIO", "WISH", "CLOV", "CLNE",
    "SPCE", "MARA", "RIOT", "SNDL", "SENS",
    "SQ",  "PYPL", "SNAP", "UBER", "LYFT",
    "ARKK", "SPY",  "QQQ",
]


def get_reddit_universe() -> list[str]:
    """
    NASDAQ 100 + hand-picked WSB classics.
    ~120 tickers — good balance of coverage vs. data-collection speed.
    """
    ndx = get_nasdaq100()
    combined = sorted(set(ndx) | set(_WSB_CLASSICS))
    print(f"Reddit universe: {len(combined)} tickers (NASDAQ100 + WSB classics)")
    return combined


def get_top_by_marketcap(symbols: list[str], n: int = 100) -> list[str]:
    """
    Filter `symbols` to the top-N by market cap using yfinance.
    Downloads info in batches of 50. Falls back to full list if yfinance fails.
    """
    print(f"Fetching market caps for {len(symbols)} tickers (top {n} kept)...")
    caps: dict[str, float] = {}
    batch_size = 50

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        try:
            tickers_obj = yf.Tickers(" ".join(batch))
            for sym in batch:
                try:
                    caps[sym] = tickers_obj.tickers[sym].info.get("marketCap", 0) or 0
                except Exception:
                    caps[sym] = 0
        except Exception:
            for sym in batch:
                caps[sym] = 0

    ranked = sorted(caps, key=lambda s: caps[s], reverse=True)
    top    = ranked[:n]
    print(f"Top {n} by market cap selected (min cap: ${caps.get(top[-1], 0):,.0f})")
    return sorted(top)
