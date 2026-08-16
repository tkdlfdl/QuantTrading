"""
Comprehensive Grid Search Backtest: Fixed Weight vs Momentum Allocation
Tests all combinations of Books A, B, C, D, E with historical data (2024-2026)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 100)
print("COMPREHENSIVE GRID SEARCH BACKTEST: Fixed Weight vs Momentum Allocation")
print("=" * 100)

csv_path = Path("results") / "portfolio_5book_daily.csv"
df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index)
df = df.fillna(0)

print(f"\nData loaded: {csv_path.name}")
print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
print(f"Books: {list(df.columns)}")
print(f"Trading days: {len(df)}")
print(f"Data shape: {df.shape}")

# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(returns_series):
    """Calculate performance metrics from daily returns."""
    r = returns_series.dropna()
    if len(r) == 0 or (r == 0).all():
        return {
            "annual_return": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_dd": 0.0,
            "cum_return": 0.0,
            "win_rate": 0.0,
        }

    # Annual metrics
    cum = (1 + r).prod() - 1
    years = len(r) / 252
    ann = (1 + cum) ** (1 / years) - 1 if years > 0 else cum

    # Sharpe
    rf_daily = 0.02 / 252
    vol = r.std()
    sharpe = ((r.mean() - rf_daily) / vol * np.sqrt(252)) if vol > 0 else 0.0

    # Sortino (downside risk)
    down_std = r[r < 0].std()
    sortino = ((r.mean() - rf_daily) / down_std * np.sqrt(252)) if down_std and down_std > 0 else 0.0

    # Max drawdown
    wealth = (1 + r).cumprod()
    max_dd = float((wealth / wealth.cummax() - 1).min())

    # Win rate
    win_rate = float((r > 0).mean())

    return {
        "annual_return": ann,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "cum_return": cum,
        "win_rate": win_rate,
    }


# ============================================================================
# GRID SEARCH 1: FIXED WEIGHT ALLOCATIONS
# ============================================================================

print("\n" + "=" * 100)
print("GRID SEARCH 1: FIXED WEIGHT ALLOCATIONS")
print("=" * 100)

books = ["A", "B", "C", "D", "E"]
weight_options = [0, 0.1, 0.2, 0.3, 0.4, 0.5]  # More granular
results_fixed = []

combo_count = 0
for weights_tuple in product(weight_options, repeat=len(books)):
    total = sum(weights_tuple)
    # Only keep combinations that sum to 100%
    if abs(total - 1.0) < 0.001:
        combo_count += 1
        weights = dict(zip(books, weights_tuple))

        # Calculate portfolio return
        port_ret = sum(df[book] * weights[book] for book in books)
        metrics = calculate_metrics(port_ret)

        results_fixed.append({
            "A": weights["A"],
            "B": weights["B"],
            "C": weights["C"],
            "D": weights["D"],
            "E": weights["E"],
            "Ann_Return": metrics["annual_return"],
            "Sharpe": metrics["sharpe"],
            "Sortino": metrics["sortino"],
            "Max_DD": metrics["max_dd"],
            "Cum_Return": metrics["cum_return"],
            "Win_Rate": metrics["win_rate"],
        })

results_fixed_df = pd.DataFrame(results_fixed)
results_fixed_df = results_fixed_df.sort_values("Sharpe", ascending=False)

print(f"\nTotal weight combinations tested: {len(results_fixed_df)}")
print(f"\nTop 20 allocations by Sharpe ratio:\n")
print(results_fixed_df.head(20).to_string(index=False))

# Find best by different metrics
best_sharpe = results_fixed_df.iloc[0]
best_return = results_fixed_df.loc[results_fixed_df["Ann_Return"].idxmax()]
best_dd = results_fixed_df.loc[results_fixed_df["Max_DD"].idxmax()]

print(f"\n\nBest by different metrics:")
print(f"  Sharpe:     {best_sharpe['Sharpe']:.4f} (A:{best_sharpe['A']:.1%} B:{best_sharpe['B']:.1%} C:{best_sharpe['C']:.1%} D:{best_sharpe['D']:.1%} E:{best_sharpe['E']:.1%})")
print(f"  Ann Return: {best_return['Ann_Return']:.2%} (A:{best_return['A']:.1%} B:{best_return['B']:.1%} C:{best_return['C']:.1%} D:{best_return['D']:.1%} E:{best_return['E']:.1%})")
print(f"  Max DD:     {best_dd['Max_DD']:.2%} (A:{best_dd['A']:.1%} B:{best_dd['B']:.1%} C:{best_dd['C']:.1%} D:{best_dd['D']:.1%} E:{best_dd['E']:.1%})")


# ============================================================================
# GRID SEARCH 2: MOMENTUM ALLOCATION
# ============================================================================

print("\n" + "=" * 100)
print("GRID SEARCH 2: MOMENTUM ALLOCATION (Rolling Sharpe Rebalance)")
print("=" * 100)

lookback_windows = [20, 30, 60, 90, 120]
results_momentum = []

for lookback in lookback_windows:
    print(f"\n  Testing lookback={lookback}d...")

    # Create weights time series based on rolling Sharpe
    weights_ts = pd.DataFrame(index=df.index, columns=books, dtype=float)

    for i, date in enumerate(df.index):
        if i < lookback:
            # Equal weight during warmup
            w = {book: 1.0 / len(books) for book in books}
        else:
            # Compute rolling Sharpe for past `lookback` days
            start_idx = i - lookback
            sharpes = {}
            for book in books:
                rets = df[book].iloc[start_idx:i]
                metrics = calculate_metrics(rets)
                sharpe = max(metrics['sharpe'], 0.1)  # Floor at 0.1
                sharpes[book] = sharpe

            # Normalize to sum to 1
            total = sum(sharpes.values())
            w = {book: sharpes[book] / total for book in sharpes.keys()}

        for book in w:
            weights_ts.loc[date, book] = w[book]

    # Calculate portfolio return
    port_ret = (df * weights_ts).sum(axis=1)
    metrics = calculate_metrics(port_ret)

    results_momentum.append({
        "Lookback_Days": lookback,
        "Ann_Return": metrics["annual_return"],
        "Sharpe": metrics["sharpe"],
        "Sortino": metrics["sortino"],
        "Max_DD": metrics["max_dd"],
        "Cum_Return": metrics["cum_return"],
        "Win_Rate": metrics["win_rate"],
    })

results_momentum_df = pd.DataFrame(results_momentum)
results_momentum_df = results_momentum_df.sort_values("Sharpe", ascending=False)

print(f"\nMomentum Allocation Results (sorted by Sharpe):\n")
print(results_momentum_df.to_string(index=False))

best_moma = results_momentum_df.iloc[0]
print(f"\nBest MomAlloc: lookback={int(best_moma['Lookback_Days'])}d, Sharpe={best_moma['Sharpe']:.4f}, Ann={best_moma['Ann_Return']:.2%}")


# ============================================================================
# COMPARISON & SUMMARY
# ============================================================================

print("\n" + "=" * 100)
print("COMPARISON: Top Candidates")
print("=" * 100)

top_fixed = results_fixed_df.iloc[0]
top_fixed_desc = f"A:{top_fixed['A']:.0%} B:{top_fixed['B']:.0%} C:{top_fixed['C']:.0%} D:{top_fixed['D']:.0%} E:{top_fixed['E']:.0%}"

equal_weight_row = results_fixed_df[
    (results_fixed_df["A"] == 0.2) &
    (results_fixed_df["B"] == 0.2) &
    (results_fixed_df["C"] == 0.2) &
    (results_fixed_df["D"] == 0.2) &
    (results_fixed_df["E"] == 0.2)
]

comparison = pd.DataFrame([
    {
        "Allocation": f"Fixed (Best): {top_fixed_desc}",
        "Ann Return": f"{top_fixed['Ann_Return']:.2%}",
        "Sharpe": f"{top_fixed['Sharpe']:.4f}",
        "Max DD": f"{top_fixed['Max_DD']:.2%}",
        "Sortino": f"{top_fixed['Sortino']:.4f}",
    },
    {
        "Allocation": f"Momentum: {int(best_moma['Lookback_Days'])}d lookback",
        "Ann Return": f"{best_moma['Ann_Return']:.2%}",
        "Sharpe": f"{best_moma['Sharpe']:.4f}",
        "Max DD": f"{best_moma['Max_DD']:.2%}",
        "Sortino": f"{best_moma['Sortino']:.4f}",
    },
])

if not equal_weight_row.empty:
    eq = equal_weight_row.iloc[0]
    comparison = pd.concat([comparison, pd.DataFrame([{
        "Allocation": "Equal-Weight (20% each)",
        "Ann Return": f"{eq['Ann_Return']:.2%}",
        "Sharpe": f"{eq['Sharpe']:.4f}",
        "Max DD": f"{eq['Max_DD']:.2%}",
        "Sortino": f"{eq['Sortino']:.4f}",
    }])], ignore_index=True)

print("\n" + comparison.to_string(index=False))

# ============================================================================
# SAVE RESULTS
# ============================================================================

out_file = Path("results") / "comprehensive_grid_backtest.xlsx"

with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
    # Summary
    comparison.to_excel(writer, sheet_name="Summary", index=False)

    # Fixed weight results
    results_fixed_sorted = results_fixed_df.sort_values("Sharpe", ascending=False)
    results_fixed_sorted.to_excel(writer, sheet_name="Fixed_Weight_Search", index=False)

    # Momentum allocation results
    results_momentum_df.to_excel(writer, sheet_name="Momentum_Alloc_Search", index=False)

print(f"\n[SAVED] Results: {out_file}")

print("\n" + "=" * 100)
print("GRID SEARCH COMPLETE")
print("=" * 100 + "\n")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print("\nFINAL RECOMMENDATIONS:\n")

print(f"✓ BEST FIXED WEIGHT ALLOCATION:")
print(f"  Allocation: A:{top_fixed['A']:.0%} B:{top_fixed['B']:.0%} C:{top_fixed['C']:.0%} D:{top_fixed['D']:.0%} E:{top_fixed['E']:.0%}")
print(f"  Sharpe: {top_fixed['Sharpe']:.4f}")
print(f"  Annual Return: {top_fixed['Ann_Return']:.2%}")
print(f"  Max Drawdown: {top_fixed['Max_DD']:.2%}")
print(f"  Sortino: {top_fixed['Sortino']:.4f}")

print(f"\n✓ BEST MOMENTUM ALLOCATION:")
print(f"  Lookback: {int(best_moma['Lookback_Days'])} days")
print(f"  Sharpe: {best_moma['Sharpe']:.4f}")
print(f"  Annual Return: {best_moma['Ann_Return']:.2%}")
print(f"  Max Drawdown: {best_moma['Max_DD']:.2%}")
print(f"  Sortino: {best_moma['Sortino']:.4f}")

if top_fixed['Sharpe'] > best_moma['Sharpe']:
    print(f"\n🏆 WINNER: Fixed Weight")
    print(f"  Sharpe advantage: +{(top_fixed['Sharpe'] - best_moma['Sharpe']):.4f} ({(top_fixed['Sharpe']/best_moma['Sharpe']-1)*100:.1f}% better)")
else:
    print(f"\n🏆 WINNER: Momentum Allocation")
    print(f"  Sharpe advantage: +{(best_moma['Sharpe'] - top_fixed['Sharpe']):.4f} ({(best_moma['Sharpe']/top_fixed['Sharpe']-1)*100:.1f}% better)")
