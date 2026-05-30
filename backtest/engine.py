"""
Vectorised backtesting engine.

Usage:
    from backtest.engine import Backtester
    from strategies.my_strategy import MyStrategy

    bt = Backtester(data=df, strategy=MyStrategy(), initial_cash=100_000)
    result = bt.run()
    print(result.summary())
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field

from strategies.base import Strategy
from backtest.metrics import compute_metrics


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    signals: pd.Series
    metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"{'Metric':<25} {'Value':>12}"]
        lines.append("-" * 38)
        for k, v in self.metrics.items():
            lines.append(f"{k:<25} {v:>12}")
        return "\n".join(lines)


class Backtester:
    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        initial_cash: float = 100_000,
        commission: float = 0.001,   # 0.1% per trade
    ):
        self.data = data.copy().reset_index(drop=True)
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.commission = commission

    def run(self) -> BacktestResult:
        signals = self.strategy.generate_signals(self.data)

        close = self.data["close"].values
        pos = signals.shift(1).fillna(0).values   # trade on next bar open

        # Daily returns of the strategy
        raw_returns = pd.Series(close).pct_change().fillna(0).values
        strat_returns = pos * raw_returns

        # Subtract commission on position changes
        trades = np.abs(np.diff(pos, prepend=0))
        strat_returns -= trades * self.commission

        equity = pd.Series(
            self.initial_cash * (1 + strat_returns).cumprod(),
            index=self.data.index,
        )
        returns = pd.Series(strat_returns, index=self.data.index)

        metrics = compute_metrics(returns, equity)

        return BacktestResult(
            equity=equity,
            returns=returns,
            signals=signals,
            metrics=metrics,
        )
