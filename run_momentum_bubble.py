"""
Run Momentum + High-Bubble Hedge + Low-Bubble Leverage strategy
on the full NASDAQ 100 + S&P 500 universe.

Preprocessing matches GetData.py:
  bfill -> ffill -> dropna(axis='columns')

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

# Extras: hedge instruments + bond ETFs + treasury yields (matching GetData.py)
EXTRAS  = ["QQQ", "SPY", "^VIX", "UVXY", "TLT", "TMF", "^TNX", "^FVX"]
START   = "1997-01-01"

if __name__ == "__main__":
    init()

    # --- Build universe --------------------------------------------------
    print("Fetching index constituent lists from Wikipedia...")
    universe    = get_universe(nasdaq100=True, sp500=True)
    all_symbols = EXTRAS + [s for s in universe if s not in EXTRAS]
    print(f"Total symbols: {len(all_symbols)}\n")

    # --- Ingest (incremental) --------------------------------------------
    print("Ingesting extras...")
    for s in EXTRAS:
        n = ingest(s, "1d", START)
        if n > 0:
            print(f"  {s}: {n} rows inserted")

    print("\nIngesting universe (batch)...")
    total = ingest_universe(universe, interval="1d", start=START, chunk_size=100)
    print(f"Total new rows inserted: {total:,}\n")

    # --- Load & preprocess (matching GetData.py) -------------------------
    print("Loading close panel from DB...")
    df    = load_close_panel(all_symbols, interval="1d", start=START, auto_ingest=False)
    close = df["Close"]

    # Replicate GetData.py: bfill -> ffill -> dropna(axis='columns')
    # bfill fills pre-IPO NaN with first available price (0 return pre-IPO)
    # dropna removes any ticker with no data at all (failed downloads)
    close = close.bfill().ffill().dropna(axis="columns")

    # Always keep required columns even if they somehow got dropped
    for must_have in ["QQQ", "SPY"]:
        if must_have not in close.columns:
            raise ValueError(f"Required column {must_have} missing after preprocessing.")

    df["Close"] = close

    print(f"Close panel: {close.shape}  |  "
          f"{close.index[0].date()} to {close.index[-1].date()}")
    print(f"Columns: {close.shape[1]} tickers")
    print(f"Bond ETFs present: TLT={('TLT' in close.columns)}, TMF={('TMF' in close.columns)}")
    print(f"Treasury yields present: ^TNX={( '^TNX' in close.columns)}, ^FVX={('^FVX' in close.columns)}\n")

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
