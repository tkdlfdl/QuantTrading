"""
DOWNLOAD EXTENDED HISTORICAL DATA (1997-2026)
==============================================
Uses Wikipedia to scrape current S&P 500 and Nasdaq-100 constituents
Downloads from 1997 to capture financial crisis, tech crash, COVID, etc.
"""

import sys, warnings, os
warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf
from io import StringIO
import requests
from datetime import datetime

print("="*100)
print("DOWNLOADING EXTENDED HISTORICAL DATA (1997-2026) FROM WIKIPEDIA CONSTITUENTS")
print("="*100)

# Setup headers and session
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

session = requests.Session()
session.headers.update(HEADERS)

def normalize_tickers(series):
    """Normalize ticker symbols for Yahoo Finance"""
    return (
        series.dropna()
        .astype(str)
        .str.strip()
        .str.replace(".", "-", regex=False)  # for Yahoo Finance
        .unique()
        .tolist()
    )

def get_sp500():
    """Scrape S&P 500 constituents from Wikipedia"""
    print("\nFetching S&P 500 constituents from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        html = session.get(url, timeout=20)
        html.raise_for_status()
        df = pd.read_html(StringIO(html.text))[0]
        tickers = normalize_tickers(df["Symbol"])
        print(f"  Found {len(tickers)} S&P 500 stocks")
        return tickers
    except Exception as e:
        print(f"  ERROR fetching S&P 500: {e}")
        return []

def get_nasdaq100():
    """Scrape Nasdaq-100 constituents from Wikipedia"""
    print("Fetching Nasdaq-100 constituents from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        html = session.get(url, timeout=20)
        html.raise_for_status()
        tables = pd.read_html(StringIO(html.text))

        for df in tables:
            cols = [str(c).strip() for c in df.columns]
            if "Ticker" in cols:
                tickers = normalize_tickers(df["Ticker"])
                print(f"  Found {len(tickers)} Nasdaq-100 stocks")
                return tickers

        raise ValueError("Nasdaq-100 ticker table not found")
    except Exception as e:
        print(f"  ERROR fetching Nasdaq-100: {e}")
        return []

# Get tickers
sp500 = get_sp500()
nasdaq100 = get_nasdaq100()

# Combine and add additional tickers
stock_list = list(set(nasdaq100 + sp500))
additional_tickers = ['QQQ', 'SPY', 'UVXY', '^VIX']
bond_tickers = ['TMF', 'TLT']
yield_tickers = ['DGS2', 'DGS10', '^FVX', '^TNX']

stock_list_full = stock_list + additional_tickers + bond_tickers + yield_tickers

print(f"\n{'='*100}")
print("TICKER LIST SUMMARY")
print(f"{'='*100}")
print(f"S&P 500 constituents: {len(sp500)}")
print(f"Nasdaq-100 constituents: {len(nasdaq100)}")
print(f"Combined (deduplicated): {len(stock_list)}")
print(f"Additional (indices, bonds, yields): {len(additional_tickers + bond_tickers + yield_tickers)}")
print(f"Total tickers to download: {len(stock_list_full)}")
print(f"\nSample tickers: {stock_list_full[:20]}")

# Download data
print(f"\n{'='*100}")
print("DOWNLOADING HISTORICAL DATA (1997-01-01 to 2026-06-06)")
print(f"{'='*100}")

start_date = "1997-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")

print(f"Start: {start_date}")
print(f"End: {end_date}")
print(f"Duration: ~29 years")
print(f"\nThis will take several minutes...\n")

try:
    df_mom = yf.download(
        tickers=stock_list_full,
        start=start_date,
        end=end_date,
        interval="1d",
        progress=True,
    )
    print(f"\nDownloaded: {df_mom.shape}")
except Exception as e:
    print(f"ERROR during download: {e}")
    sys.exit(1)

# Data preprocessing (as provided)
print(f"\n{'='*100}")
print("DATA PREPROCESSING")
print(f"{'='*100}")

print("Step 1: Backward fill (for holidays/gaps)")
df2_mom = df_mom.bfill(axis='rows')

print("Step 2: Forward fill (for missing data)")
df3_mom = df2_mom.ffill(axis='rows')

print("Step 3: Drop columns with all NaN values")
df4_mom = df3_mom.dropna(axis='columns')

print(f"Final data shape: {df4_mom.shape}")
print(f"Columns remaining: {df4_mom.shape[1]}")
print(f"Date range: {df4_mom.index[0].date()} to {df4_mom.index[-1].date()}")
print(f"Missing data after fill: {df4_mom.isna().sum().sum() / (df4_mom.shape[0] * df4_mom.shape[1]) * 100:.2f}%")

# Extract close prices
print("\nExtracting close prices...")
if isinstance(df4_mom.columns, pd.MultiIndex):
    df_close = df4_mom["Close"].copy()
else:
    df_close = df4_mom[["Close"]].copy()
    if isinstance(df_close.columns, pd.Index):
        df_close.columns = ["Close"] if len(df_close.columns) == 1 else df_close.columns

print(f"Close data shape: {df_close.shape}")

# Save data
os.makedirs("data/cache", exist_ok=True)

print(f"\n{'='*100}")
print("SAVING DATA")
print(f"{'='*100}")

# Parquet format
parquet_file = "data/cache/daily_close_extended_1997_2026.parquet"
df_close.to_parquet(parquet_file)
print(f"Saved: {parquet_file}")
print(f"File size: {os.path.getsize(parquet_file) / (1024*1024):.2f} MB")

# CSV format
csv_file = "data/cache/daily_close_extended_1997_2026.csv"
df_close.to_csv(csv_file)
print(f"Saved: {csv_file}")
print(f"File size: {os.path.getsize(csv_file) / (1024*1024):.2f} MB")

# Summary
print(f"\n{'='*100}")
print("DATA SUMMARY")
print(f"{'='*100}")
print(f"""
Period: {df_close.index[0].date()} to {df_close.index[-1].date()}
Duration: {(df_close.index[-1] - df_close.index[0]).days / 365.25:.2f} years

Data Coverage:
  Total rows: {len(df_close)}
  Total columns: {df_close.shape[1]}
  Data completeness: {(1 - df_close.isna().sum().sum() / (df_close.shape[0] * df_close.shape[1])) * 100:.2f}%

Columns by category:
  - Stocks: {len([c for c in df_close.columns if c not in ['QQQ', 'SPY', 'UVXY', '^VIX', 'TMF', 'TLT', 'DGS2', 'DGS10', '^FVX', '^TNX']])}
  - Indices: 4 (QQQ, SPY, UVXY, ^VIX)
  - Bonds: 2 (TMF, TLT)
  - Yields: 4 (DGS2, DGS10, ^FVX, ^TNX)

Historical Events Covered:
  - 2000-2002: Tech/Dot-com crash
  - 2008-2009: Financial crisis
  - 2015: Volatility spike
  - 2020: COVID-19 pandemic
  - 2022: Tech bear market
  - 2023-2026: Bull market recovery

Status: READY FOR EXTENDED BACKTESTING (1997-2026)
""")

print(f"Next step: Run extended backtest using:")
print(f"  python run_daily_momentum_extended_1997_2026.py")
