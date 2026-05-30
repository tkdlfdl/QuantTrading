"""
Run the Momentum + High-Bubble Hedge + Low-Bubble Leverage strategy.

Usage:
    python run_momentum_bubble.py
"""
import sys
sys.path.insert(0, ".")

from data.db.schema import init
from data.loader import load_close_panel
from strategies.momentum_bubble_hedge import run_momentum_bubble_hedge_and_low_bubble_leverage

# Symbols: momentum universe + optional hedge instruments
SYMBOLS = ["QQQ", "SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "^VIX"]
START   = "2015-01-01"

if __name__ == "__main__":
    init()

    print(f"Loading data for {SYMBOLS} from {START}...")
    df = load_close_panel(SYMBOLS, interval="1d", start=START)

    print(f"\nClose panel shape: {df['Close'].shape}")
    print(f"Date range: {df['Close'].index[0]} → {df['Close'].index[-1]}")
    print(f"Columns: {list(df['Close'].columns)}\n")

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

    print("\n=== Overall Performance ===")
    print(analysis.to_string())

    print("\n=== Yearly Performance ===")
    print(yearly_analysis.to_string())

    print("\n=== Top 10 Grid Results by Sharpe ===")
    print(grid_result_df.head(10).to_string(index=False))
