"""
Entry point: initialise DB and run a sample ingest.
    python main.py
"""
from data.db.schema import init
from data.ingestion import ingest_batch
from config.settings import EQUITY_SYMBOLS

if __name__ == "__main__":
    init()
    print("\nIngesting sample equity data (1h, last 30 days)...")
    ingest_batch(EQUITY_SYMBOLS[:3], intervals=["1h", "1d"])
    print("\nDone. Data stored in data/market_data.duckdb")
