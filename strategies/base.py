from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    """
    Every strategy inherits from this class.
    Implement generate_signals() to return a Series of {-1, 0, 1}
    aligned to the input DataFrame's index.
    """

    name: str = "base"
    params: dict = {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Args:
            data: OHLCV DataFrame with columns [ts, open, high, low, close, volume]
        Returns:
            pd.Series of float: 1=long, -1=short, 0=flat, indexed same as data
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.params})"
