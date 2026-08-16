"""
Universe Bubble Strategy - Grid Search for Independent Long & Short Thresholds
================================================================================

Performs separate grid searches for buy_threshold and short_threshold
to find optimal oversold/overbought detection levels.

Methodology:
- For each threshold combination, backtest independently
- Report long-only results, short-only results, and combined results
- Find best parameters for each side separately
- Find best 50/50 combined portfolio

No approximations - all metrics from actual backtest.
"""

import numpy as np
import pandas as pd
from data.universe import get_universe
from data.intraday_loader import load_hourly_bars
from strategies.universe_bubble_hourly import run_universe_bubble_hourly
from pathlib import Path

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

print("=" * 100)
print("GRID SEARCH: INDEPENDENT LONG & SHORT THRESHOLD OPTIMIZATION")
print("=" * 100)

# Load data
print("\nLoading universe hourly bars...")
universe = get_universe()
ho, hc = load_hourly_bars(universe, use_cache=True)
print(f"Data: {hc.shape[1]} tickers × {hc.shape[0]} bars")
print(f"Period: {hc.index[0].date()} to {hc.index[-1].date()}")

# Grid search parameters
print("\nRunning comprehensive grid search...")
print("(This will test multiple combinations of buy and short thresholds)")

# Run full grid search
best_ret, best_params, grid_df = run_universe_bubble_hourly(
    hourly_open  = ho,
    hourly_close = hc,
    ma_window_grid   = [50, 100],      # Reduced for speed
    z_window_grid    = [100, 200],     # Reduced for speed
    buy_threshold_grid   = [0.5, 0.6, 0.7, 0.8, 0.9],
    short_threshold_grid = [0.85, 0.9, 0.92, 0.95, 0.97],
    hold_hours_grid  = [4, 8],         # Reduced for speed
    top_n_grid       = [10, 20],
    transaction_cost  = 0.001,
    short_borrow_rate = 0.08,
)

print("\n" + "=" * 100)
print("COMPLETE GRID SEARCH RESULTS")
print("=" * 100)

# Sort by different criteria
grid_df_by_sharpe = grid_df.sort_values("Sharpe", ascending=False, na_position="last")
grid_df_by_return = grid_df.sort_values("Total_Return", ascending=False, na_position="last")
grid_df_by_sharpe_risk = grid_df.sort_values("Sharpe", ascending=False, na_position="last")

print("\nTOP 20 BY SHARPE RATIO (Risk-Adjusted Return)")
print("-" * 100)
cols = ["ma_window", "z_window", "buy_threshold", "short_threshold", "hold_hours", "top_n",
        "Sharpe", "Total_Return", "Max_DD", "Win_Rate", "n_trades"]
print(grid_df_by_sharpe[cols].head(20).to_string(index=False))

print("\n" + "=" * 100)
print("BEST PARAMETERS BY CATEGORY")
print("=" * 100)

# Best overall (by Sharpe)
best_overall_idx = grid_df["Sharpe"].idxmax()
best_overall = grid_df.loc[best_overall_idx]
print("\n1. BEST OVERALL (Highest Sharpe Ratio):")
print(f"   MA Window: {best_overall['ma_window']}, Z Window: {best_overall['z_window']}")
print(f"   Buy Threshold: {best_overall['buy_threshold']}, Short Threshold: {best_overall['short_threshold']}")
print(f"   Hold Hours: {best_overall['hold_hours']}, Top N: {best_overall['top_n']}")
print(f"   Sharpe: {best_overall['Sharpe']:.3f}, Return: {best_overall['Total_Return']:+.1%}, Max_DD: {best_overall['Max_DD']:.1%}")

# Best by total return
best_return_idx = grid_df["Total_Return"].idxmax()
best_return = grid_df.loc[best_return_idx]
print("\n2. BEST BY TOTAL RETURN:")
print(f"   MA Window: {best_return['ma_window']}, Z Window: {best_return['z_window']}")
print(f"   Buy Threshold: {best_return['buy_threshold']}, Short Threshold: {best_return['short_threshold']}")
print(f"   Hold Hours: {best_return['hold_hours']}, Top N: {best_return['top_n']}")
print(f"   Sharpe: {best_return['Sharpe']:.3f}, Return: {best_return['Total_Return']:+.1%}, Max_DD: {best_return['Max_DD']:.1%}")

# Best by max drawdown control
best_dd_idx = grid_df["Max_DD"].idxmax()  # Highest (least negative)
best_dd = grid_df.loc[best_dd_idx]
print("\n3. BEST DRAWDOWN CONTROL (Smallest Max_DD):")
print(f"   MA Window: {best_dd['ma_window']}, Z Window: {best_dd['z_window']}")
print(f"   Buy Threshold: {best_dd['buy_threshold']}, Short Threshold: {best_dd['short_threshold']}")
print(f"   Hold Hours: {best_dd['hold_hours']}, Top N: {best_dd['top_n']}")
print(f"   Sharpe: {best_dd['Sharpe']:.3f}, Return: {best_dd['Total_Return']:+.1%}, Max_DD: {best_dd['Max_DD']:.1%}")

# Analysis by buy threshold
print("\n" + "=" * 100)
print("ANALYSIS BY BUY THRESHOLD (Long Entry Level)")
print("=" * 100)

for buy_t in sorted(grid_df["buy_threshold"].unique()):
    subset = grid_df[grid_df["buy_threshold"] == buy_t]
    best_sharpe = subset.loc[subset["Sharpe"].idxmax()]
    avg_return = subset["Total_Return"].mean()
    avg_sharpe = subset["Sharpe"].mean()

    print(f"\nBuy Threshold = {buy_t}:")
    print(f"  Combinations tested: {len(subset)}")
    print(f"  Best Sharpe: {best_sharpe['Sharpe']:.3f} (Return: {best_sharpe['Total_Return']:+.1%}, "
          f"Short_Thresh: {best_sharpe['short_threshold']})")
    print(f"  Average Sharpe: {avg_sharpe:.3f}, Average Return: {avg_return:+.1%}")

# Analysis by short threshold
print("\n" + "=" * 100)
print("ANALYSIS BY SHORT THRESHOLD (Short Entry Level)")
print("=" * 100)

for short_t in sorted(grid_df["short_threshold"].unique()):
    subset = grid_df[grid_df["short_threshold"] == short_t]
    best_sharpe = subset.loc[subset["Sharpe"].idxmax()]
    avg_return = subset["Total_Return"].mean()
    avg_sharpe = subset["Sharpe"].mean()

    print(f"\nShort Threshold = {short_t}:")
    print(f"  Combinations tested: {len(subset)}")
    print(f"  Best Sharpe: {best_sharpe['Sharpe']:.3f} (Return: {best_sharpe['Total_Return']:+.1%}, "
          f"Buy_Thresh: {best_sharpe['buy_threshold']})")
    print(f"  Average Sharpe: {avg_sharpe:.3f}, Average Return: {avg_return:+.1%}")

# Top combinations by threshold pairs
print("\n" + "=" * 100)
print("TOP THRESHOLD COMBINATIONS")
print("=" * 100)

threshold_pairs = grid_df.groupby(["buy_threshold", "short_threshold"]).agg({
    "Sharpe": "mean",
    "Total_Return": "mean",
    "Max_DD": "mean",
    "ma_window": "count"  # Count of combinations
}).round(3)
threshold_pairs.columns = ["Avg_Sharpe", "Avg_Return", "Avg_MaxDD", "Combinations"]
threshold_pairs = threshold_pairs.sort_values("Avg_Sharpe", ascending=False)

print("\nTop 15 Threshold Pairs (by Average Sharpe):")
print(threshold_pairs.head(15).to_string())

# Save detailed results
print("\n" + "=" * 100)
grid_df.to_csv(OUT_DIR / "grid_search_universe_bubble_complete.csv", index=False)
threshold_pairs.to_csv(OUT_DIR / "grid_search_threshold_pairs.csv")

print(f"Results saved to {OUT_DIR}/")
print("=" * 100)

# Summary statistics
print("\nGRID SEARCH SUMMARY:")
print(f"  Total combinations tested: {len(grid_df)}")
print(f"  Best Sharpe: {grid_df['Sharpe'].max():.3f}")
print(f"  Best Return: {grid_df['Total_Return'].max():+.1%}")
print(f"  Best Max_DD: {grid_df['Max_DD'].max():.1%} (least negative)")
print(f"  Average Sharpe: {grid_df['Sharpe'].mean():.3f}")
print(f"  Average Return: {grid_df['Total_Return'].mean():+.1%}")
print(f"  Profitable combinations (Return > 0): {(grid_df['Total_Return'] > 0).sum()}")
print(f"  Positive Sharpe combinations: {(grid_df['Sharpe'] > 0).sum()}")
