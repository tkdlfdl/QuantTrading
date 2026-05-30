"""
Run Momentum + High-Bubble Hedge + Low-Bubble Leverage strategy.

Data pipeline matches GetData.py exactly:
  - Single yf.download() call for all tickers
  - bfill -> ffill -> dropna(axis='columns')

This produces identical results to running my_new_strategy.py locally.

Usage:
    python run_momentum_bubble.py
"""
import sys
sys.path.insert(0, ".")

import yfinance as yf
import pandas as pd
import requests
from io import StringIO

from data.db.schema import init
from strategies.momentum_bubble_hedge import run_momentum_bubble_hedge_and_low_bubble_leverage

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
START = "1997-01-01"


def get_sp500() -> list[str]:
    session = requests.Session()
    session.headers.update(HEADERS)
    html = session.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", timeout=20)
    df = pd.read_html(StringIO(html.text))[0]
    return df["Symbol"].dropna().astype(str).str.strip().str.replace(".", "-", regex=False).unique().tolist()


def get_nasdaq100() -> list[str]:
    session = requests.Session()
    session.headers.update(HEADERS)
    html = session.get("https://en.wikipedia.org/wiki/Nasdaq-100", timeout=20)
    for t in pd.read_html(StringIO(html.text)):
        if "Ticker" in t.columns:
            return t["Ticker"].dropna().astype(str).str.strip().str.replace(".", "-", regex=False).unique().tolist()


def load_panel(start: str = START) -> dict:
    sp500     = get_sp500()
    nasdaq100 = get_nasdaq100()
    print(f"S&P 500: {len(sp500)}, NASDAQ-100: {len(nasdaq100)}")

    ticker = list(set(nasdaq100 + sp500 + ["TMF", "TLT"]))
    stock_list = (ticker + ["QQQ", "SPY", "UVXY", "^VIX", "^FVX",
                            "2Y", "US2Y", "DGS2", "10Y", "US10Y", "^TNX", "DGS10"])

    print(f"Downloading {len(stock_list)} tickers from {start}...")
    df_mom = yf.download(tickers=stock_list, start=start, progress=True)
    close  = df_mom.bfill().ffill().dropna(axis="columns")["Close"].copy()

    print(f"Panel: {close.shape} | {close.index[0].date()} to {close.index[-1].date()}")
    print(f"TLT={('TLT' in close.columns)}, TMF={('TMF' in close.columns)}, "
          f"^TNX={('^TNX' in close.columns)}, ^FVX={('^FVX' in close.columns)}, "
          f"UVXY={('UVXY' in close.columns)}")
    return {"Close": close}


if __name__ == "__main__":
    init()
    df = load_panel(START)

    (
        analysis,
        yearly_analysis,
        wealth_df,
        ret_df,
        grid_result_df,
        best_return_decomposition,
        best_decomposition_wealth,
        best_hedge_exposure,
        best_leverage_exposure,
        best_bubble_score,
        best_hedge_signal,
        best_leverage_signal,
        best_treasury_bubble_df,
        best_signal_treasury_snapshot,
        hedge_source_series,
    ) = run_momentum_bubble_hedge_and_low_bubble_leverage(
        df,
        lookback=140,
        holding_period=40,
        LongShort_flag=True,
        top=5,
        bubble_indicator_grid=["QQQ", "SPY", "Momentum"],
        ma_window_grid=[120],
        z_window_grid=[240],
        hedge_bubble_entry_grid=[0.85],
        hedge_alloc_grid=[0.5],
        hedge_hold_days_grid=[40],
        low_bubble_entry_grid=[-0.9, -0.89, -0.88],
        momentum_extra_leverage_grid=[0.25, 0.3, 0.35],
        leverage_hold_days_grid=[30, 40, 50, 55, 60],
    )

    S = "=" * 60
    print(f"\n{S}\nOVERALL PERFORMANCE\n{S}")
    print(analysis.to_string())

    print(f"\n{S}\nTOP 10 GRID RESULTS BY SHARPE\n{S}")
    cols = ["bubble_indicator", "low_bubble_entry", "momentum_extra_leverage",
            "leverage_hold_days", "Sharpe Ratio", "Sortino Ratio", "Total Return", "Max Drawdown"]
    print(grid_result_df[cols].head(10).to_string(index=False))
