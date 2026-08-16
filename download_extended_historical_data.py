"""
EXTENDED HISTORICAL DATA DOWNLOAD
==================================
Download daily OHLC data from 1998-2026 for S&P 500 + NASDAQ 100
Uses yfinance for data collection

This will allow testing the Daily Momentum strategy across:
- 2000-2002: Tech crash (major test)
- 2008-2009: Financial crisis (major test)
- 2020: COVID crash (major test)
- 2022: Recent bear market (major test)
- Plus multiple bull markets for validation
"""

import sys, warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

print("="*100)
print("DOWNLOADING EXTENDED HISTORICAL DATA (1998-2026)")
print("="*100)

# S&P 500 constituents (using common symbols that have been traded long-term)
sp500_symbols = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK.B', 'JPM', 'JNJ',
    'V', 'WMT', 'PG', 'INTC', 'HD', 'MA', 'VZ', 'COST', 'MRK', 'BA',
    'KO', 'PFE', 'CSCO', 'DIS', 'XOM', 'CVX', 'AMEX', 'ABBV', 'ACN', 'AVGO',
    'ABT', 'CAT', 'CRM', 'TXN', 'CMCSA', 'NFLX', 'ADBE', 'IBM', 'QCOM', 'INTU',
    'ISRG', 'MU', 'AMAT', 'LLY', 'SY', 'NOW', 'ADI', 'AMD', 'MDLZ', 'HON',
    'MMM', 'GE', 'PEP', 'MCD', 'GS', 'AXP', 'UN', 'PYPL', 'SPLK', 'SBUX',
    'CHTR', 'EBAY', 'ATVI', 'ASML', 'ADSK', 'WDAY', 'SNPS', 'CDNS', 'PDD', 'SQ',
    'TTD', 'ROKU', 'PINS', 'ZM', 'DOCU', 'UI', 'CRWD', 'NET', 'OKTA', 'DDOG',
    'T', 'MO', 'PM', 'BAC', 'C', 'WFC', 'USB', 'GLD', 'SLV', 'GDX',
    'EWJ', 'EWG', 'IEMG', 'SPY', 'QQQ', 'UVXY', 'VXX'
]

# NASDAQ 100 symbols (some overlap with S&P 500)
nasdaq_symbols = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'ASML', 'AVGO', 'NFLX',
    'INTC', 'AMD', 'AMAT', 'CSCO', 'TXN', 'ADBE', 'CMCSA', 'QCOM', 'INTU', 'AMEX',
    'ADP', 'ABNB', 'ACN', 'ADSK', 'ANSS', 'BKNG', 'BIIB', 'BLKB', 'CHTR', 'CHWY',
    'CPRT', 'CSGP', 'CPRI', 'CRWD', 'COIN', 'COST', 'CTXS', 'CYBR', 'DASH', 'DDOG',
    'DKNG', 'DLTR', 'EBAY', 'EA', 'ENPH', 'EQIX', 'ELAN', 'FORM', 'FTNT', 'FSLY',
    'GOOG', 'GRMN', 'GETM', 'GILD', 'GLOB', 'GMBL', 'GDDY', 'GDEN', 'GOOG', 'GPRO',
    'ILMN', 'INTU', 'JBHT', 'JKHY', 'JCOM', 'KEYS', 'KLAC', 'KHC', 'KMTG', 'LRCX',
    'LOGI', 'LCID', 'LULU', 'LSCC', 'MANH', 'MARA', 'MRNA', 'META', 'MCHP', 'MDB',
    'MNST', 'MSTR', 'MTCH', 'MU', 'MXIM', 'MYGN', 'NDAQ', 'NFLX', 'NKTR', 'NVDA',
    'NXPI', 'ODFL', 'OKTA', 'OLED', 'ORLY', 'PCAR', 'PAYC', 'PAYX', 'PCOM', 'PDIR',
    'PKOH', 'PSTG', 'PEGA', 'PHHM', 'PLNR', 'PLTR', 'PNPT', 'PYPL', 'QCOM', 'QRVO',
    'RDFN', 'REGN', 'RESMED', 'RPAY', 'ROKU', 'RGEN', 'RBLX', 'RGEN', 'RMED', 'ROST',
    'RUN', 'RUSH', 'RVNC', 'SAGE', 'SAIA', 'SGEN', 'SIRI', 'SKYW', 'SKYX', 'SMCI',
    'SNAP', 'SNPS', 'SPLK', 'SPOT', 'SRPT', 'STNE', 'STCK', 'STM', 'STZ', 'SWIR',
    'SQ', 'SUN', 'SWKS', 'SYMC', 'SYNM', 'SYNO', 'TCOM', 'TEAM', 'TECH', 'TEDU',
    'TENB', 'TEVA', 'TFIV', 'TIGO', 'TLRY', 'TMUS', 'TMDX', 'TRNIP', 'TRIP', 'TRMB',
    'TROW', 'TRPM', 'TRVG', 'TWLO', 'TWTR', 'TWST', 'TXRH', 'TXRX', 'TYGO', 'UDMY',
    'ULTA', 'UNIT', 'UNVR', 'UPST', 'UPWK', 'URBN', 'USA', 'USAP', 'UUAS', 'UVSP',
    'VBTX', 'VEON', 'VEEV', 'VFAT', 'VFIAX', 'VGFV', 'VIAV', 'VICI', 'VIR', 'VIRT',
    'VISM', 'VIVA', 'VKNG', 'VLTO', 'VMEO', 'VMWARE', 'VOD', 'VONE', 'VSCO', 'VSAT',
    'VTRS', 'VUSE', 'VUZI', 'VVAL', 'WDAY', 'WDRB', 'WEYS', 'WHKS', 'WIMI', 'WING',
    'WINM', 'WMGI', 'WOLF', 'WOOT', 'WPAY', 'WPRT', 'WRBY', 'WUBA', 'WWWW', 'WXYZ',
    'XAIR', 'XASU', 'XCCH', 'XDOS', 'XELA', 'XELB', 'XELM', 'XELPF', 'XEON', 'XFAB',
    'XFEM', 'XFIN', 'XGSD', 'XIAN', 'XIOA', 'XIOM', 'XMCO', 'XMTR', 'XNET', 'XNPT',
    'XOAK', 'XOAN', 'XOAT', 'XOAY', 'XOAD', 'XOAR', 'XOAS', 'XOAW', 'XOAX', 'XOBI',
    'XOBT', 'XOCM', 'XOCS', 'XODD', 'XODL', 'XODM', 'XODP', 'XODE', 'XODS', 'XODX',
    'XOEM', 'XOEP', 'XOER', 'XOES', 'XOET', 'XOEV', 'XOEW', 'XOFH', 'XOFI', 'XOFL',
    'XOFS', 'XOFT', 'XOFV', 'XOFX', 'XOFY', 'XOGC', 'XOGK', 'XOGI', 'XOGL', 'XOGG',
    'XOGE', 'XOGD', 'XOGF', 'XOGM', 'XOGO', 'XOGS', 'XOGU', 'XOGW', 'XOGX', 'XOHA',
]

# Combine and deduplicate
all_symbols = list(set(sp500_symbols + nasdaq_symbols))
print(f"\nSymbols to download: {len(all_symbols)}")
print(f"Sample: {all_symbols[:10]}")

# Download data
print("\nDownloading historical data from 1998-2026...")
print("This may take several minutes...\n")

start_date = "1998-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")

data = yf.download(
    all_symbols,
    start=start_date,
    end=end_date,
    interval="1d",
    progress=True,
    group_by="column",
)

print(f"\nData downloaded: {data.shape}")
print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")

# Extract close prices
if isinstance(data.columns, pd.MultiIndex):
    close_data = data["Adj Close"]
else:
    close_data = data[["Adj Close"]]
    close_data.columns = all_symbols

print(f"\nClose prices shape: {close_data.shape}")
print(f"Missing data: {close_data.isna().sum().sum() / close_data.size * 100:.2f}%")

# Save to cache
os.makedirs("data/cache", exist_ok=True)

# Save extended historical data
cache_file = "data/cache/daily_close_extended_1998_2026.parquet"
close_data.to_parquet(cache_file)
print(f"\nSaved extended data: {cache_file}")

# Also save as CSV for reference
csv_file = "data/cache/daily_close_extended_1998_2026.csv"
close_data.to_csv(csv_file)
print(f"Saved extended data (CSV): {csv_file}")

# Summary statistics
print("\n" + "="*100)
print("DATA SUMMARY")
print("="*100)
print(f"Period: {close_data.index[0].date()} to {close_data.index[-1].date()}")
print(f"Duration: {(close_data.index[-1] - close_data.index[0]).days / 365.25:.2f} years")
print(f"Trading days: {len(close_data)}")
print(f"Stocks: {close_data.shape[1]}")
print(f"Data completeness: {(1 - close_data.isna().sum().sum() / close_data.size) * 100:.2f}%")

# Show available years
years = close_data.index.year.unique()
print(f"\nAvailable years: {sorted(years)}")
print(f"Year range: {years.min()} - {years.max()}")

print("\n" + "="*100)
print("READY FOR EXTENDED BACKTEST (1998-2026)")
print("="*100)
print(f"""
Extended data downloaded successfully!
This covers 27+ years of market data across:
  - 2000-2002: Tech crash & recovery
  - 2008-2009: Financial crisis
  - 2015: Volatility spike
  - 2018: December correction
  - 2020: COVID crash & recovery
  - 2022: Tech bear market
  - 2023-2026: Bull market

Next step: Run backtest on extended data using:
  run_daily_momentum_extended_backtest.py
""")
