from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH  = DATA_DIR / "market_data.duckdb"
PARQUET_DIR = DATA_DIR / "cache"

EQUITY_SYMBOLS  = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"]
FUTURES_SYMBOLS = ["ES=F", "NQ=F", "CL=F", "GC=F"]
FOREX_SYMBOLS   = ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]

ALL_SYMBOLS = EQUITY_SYMBOLS + FUTURES_SYMBOLS + FOREX_SYMBOLS

# yfinance max lookback in days per interval
YF_LOOKBACK_DAYS = {
    "1m":  7,
    "2m":  60,
    "5m":  60,
    "15m": 60,
    "30m": 60,
    "1h":  730,
    "1d":  None,
}
