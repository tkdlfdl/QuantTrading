from abc import ABC, abstractmethod
import pandas as pd


class BaseFetcher(ABC):
    @abstractmethod
    def fetch(self, symbol: str, interval: str, start: str = None, end: str = None) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
            ts, symbol, interval, open, high, low, close, volume
        ts is timezone-naive UTC. Returns empty DataFrame on failure.
        """
