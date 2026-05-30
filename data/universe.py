"""
Fetch live index constituent lists from Wikipedia.

    from data.universe import get_universe
    tickers = get_universe()          # NASDAQ 100 + S&P 500, deduplicated
    tickers = get_universe(sp500=False)   # NASDAQ 100 only
"""
from __future__ import annotations

from io import StringIO
import pandas as pd
import requests

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
