"""
Run Momentum + High-Bubble Hedge + Low-Bubble Leverage strategy
on the full NASDAQ 100 + S&P 500 universe.

Usage:
    python run_momentum_bubble.py
"""
import sys
sys.path.insert(0, ".")

from data.db.schema import init
from data.universe import get_universe
from data.ingestion import ingest, ingest_universe
from data.loader import load_close_panel
from strategies.momentum_bubble_hedge import run_momentum_bubble_hedge_and_low_bubble_leverage

# Always-present extras: index ETFs for buy-hold comparison + VIX for hedging
EXTRAS = ["QQQ", "SPY", "^VIX", "UVXY"]
START  = "1998-01-01"

if __name__ == "__main__":
    init()

    # --- Build universe --------------------------------------------------
    print("Fetching index constituent lists from Wikipedia...")
    universe = get_universe(nasdaq100=True, sp500=True)
    all_symbols = EXTRAS + [s for s in universe if s not in EXTRAS]
    print(f"Total symbols to ingest: {len(all_symbols)}\n")

    # --- Ingest (incremental — skips rows already in DB) -----------------
    print("Ingesting EXTRAS individually...")
    for s in EXTRAS:
        n = ingest(s, "1d", START)
        if n > 0:
            print(f"  {s}: {n} rows inserted")

    print("\nIngesting universe (batch download)...")
    total = ingest_universe(universe, interval="1d", start=START, chunk_size=100)
    print(f"\nTotal new rows inserted: {total:,}\n")

    # --- Load wide panel -------------------------------------------------
    print("Loading close panel from DB...")
    df = load_close_panel(all_symbols, interval="1d", start=START, auto_ingest=False)
    close = df["Close"]

    # Drop columns with too many NaNs (< 5 years of data = <1260 bars)
    valid = close.columns[close.notna().sum() >= 1260].tolist()
    # Always keep QQQ, SPY, ^VIX even if sparse
    for must_have in ["QQQ", "SPY", "^VIX"]:
        if must_have in close.columns and must_have not in valid:
            valid.append(must_have)
    df["Close"] = close[valid]

    print(f"Close panel: {df['Close'].shape}  |  "
          f"{df['Close'].index[0].date()} → {df['Close'].index[-1].date()}")
    print(f"Tickers after quality filter: {len(valid)}\n")

    # --- Run strategy ----------------------------------------------------
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
        top=20,                          # larger top-N for bigger universe
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

    print("\n" + "=" * 60)
    print("OVERALL PERFORMANCE")
    print("=" * 60)
    print(analysis.to_string())

    print("\n" + "=" * 60)
    print("TOP 10 GRID RESULTS BY SHARPE")
    print("=" * 60)
    cols = ["bubble_indicator", "low_bubble_entry", "momentum_extra_leverage",
            "leverage_hold_days", "Sharpe Ratio", "Sortino Ratio", "Total Return", "Max Drawdown"]
    print(grid_result_df[cols].head(10).to_string(index=False))
