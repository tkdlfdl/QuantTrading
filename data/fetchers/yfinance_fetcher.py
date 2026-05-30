from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

from data.fetchers.base import BaseFetcher
from config.settings import YF_LOOKBACK_DAYS


class YFinanceFetcher(BaseFetcher):

    def fetch(self, symbol: str, interval: str, start: str = None, end: str = None) -> pd.DataFrame:
        start = self._clamp_start(interval, start)

        raw = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )

        if raw.empty:
            return pd.DataFrame()

        df = raw.copy()

        # Flatten MultiIndex columns (yfinance >=0.2 uses Price/Ticker MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Reset index first so Date/Datetime becomes a regular column
        df = df.reset_index()

        # Lowercase everything
        df.columns = [str(col).lower() for col in df.columns]

        # index column is "datetime" for intraday, "date" for daily
        ts_col = "datetime" if "datetime" in df.columns else "date"
        df = df.rename(columns={ts_col: "ts"})

        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
        df["symbol"] = symbol
        df["interval"] = interval

        return df[["ts", "symbol", "interval", "open", "high", "low", "close", "volume"]].dropna()

    @staticmethod
    def _clamp_start(interval: str, start: str | None) -> str | None:
        limit_days = YF_LOOKBACK_DAYS.get(interval)
        if limit_days is None:
            return start
        earliest = (datetime.utcnow() - timedelta(days=limit_days - 1)).strftime("%Y-%m-%d")
        if start is None or start < earliest:
            return earliest
        return start
