"""
Comprehensive Grid Search Backtest with Proper Strategy Parameters
Tests fixed weight and momentum allocations using verified 2024-2026 daily returns

Data: Verified from live paper trading engine (2024-04-01 to 2026-06-01)
Strategies: A, B, C, D, E with documented optimal parameters
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import warnings

warnings.filterwarnings('ignore')

print("=" * 130)
print("GRID SEARCH BACKTEST: FIXED WEIGHT + MOMENTUM ALLOCATION (2024-2026)")
print("=" * 130)

# ============================================================================
# LOAD VERIFIED DATA
# ============================================================================

print("\n[LOADING VERIFIED 2024-2026 DATA]")

csv_path = Path("results") / "portfolio_5book_daily.csv"

if not csv_path.exists():
    print(f"ERROR: {csv_path} not found")
    exit(1)

df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index)

print(f"[OK] Data loaded: {len(df)} trading days")
print(f"     Period: {df.index[0].date()} to {df.index[-1].date()}")
print(f"     Strategies: {list(df.columns)}")

# Data availability
print(f"\n[DATA AVAILABILITY]")
for book in df.columns:
    active_days = (df[book] != 0).sum()
    pct = (active_days / len(df)) * 100
    print(f"  {book}: {active_days}/{len(df)} days ({pct:.1f}%)")

# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(returns_series):
    """Calculate metrics from daily returns."""
    r = returns_series.dropna()

    if len(r) == 0 or (r == 0).all():
        return {
            "annual_return": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_dd": 0.0,
            "cum_return": 0.0,
            "volatility": 0.0,
        }

    # Annual metrics
    cum = (1 + r).prod() - 1
    years = len(r) / 252
    ann = (1 + cum) ** (1 / years) - 1 if years > 0 else cum

    # Sharpe (from daily data, not yearly)
    rf_daily = 0.02 / 252
    vol = r.std()
    sharpe = ((r.mean() - rf_daily) / vol * np.sqrt(252)) if vol > 0 else 0.0

    # Sortino (downside risk)
    down_std = r[r < 0].std()
    sortino = ((r.mean() - rf_daily) / down_std * np.sqrt(252)) if down_std and down_std > 0 else 0.0

    # Max drawdown
    wealth = (1 + r).cumprod()
    max_dd = float((wealth / wealth.cummax() - 1).min())

    # Volatility
    volatility = vol * np.sqrt(252)

    return {
        "annual_return": ann,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "cum_return": cum,
        "volatility": volatility,
    }

# ============================================================================
# INDIVIDUAL STRATEGY PERFORMANCE
# ============================================================================

print("\n" + "=" * 130)
print("INDIVIDUAL STRATEGY PERFORMANCE (2024-2026)")
print("=" * 130)

individual_results = []

for book in df.columns:
    metrics = calculate_metrics(df[book])
    individual_results.append({
        "Book": book,
        "Annual_Return": metrics["annual_return"],
        "Sharpe": metrics["sharpe"],
        "Sortino": metrics["sortino"],
        "Max_DD": metrics["max_dd"],
        "Volatility": metrics["volatility"],
        "Cum_Return": metrics["cum_return"],
    })

    status = "EXCELLENT" if metrics["sharpe"] > 1.5 else "GOOD" if metrics["sharpe"] > 1.0 else "WEAK" if metrics["sharpe"] > 0.5 else "POOR"
    print(f"\nBook {book}: {status}")
    print(f"  Annual Return: {metrics['annual_return']:>7.2%}")
    print(f"  Sharpe Ratio:  {metrics['sharpe']:>7.4f}")
    print(f"  Sortino:       {metrics['sortino']:>7.4f}")
    print(f"  Max DD:        {metrics['max_dd']:>7.2%}")
    print(f"  Volatility:    {metrics['volatility']:>7.2%}")

individual_df = pd.DataFrame(individual_results)
individual_df = individual_df.sort_values("Sharpe", ascending=False)

# ============================================================================
# YEARLY BREAKDOWN
# ============================================================================

print("\n" + "=" * 130)
print("YEARLY BREAKDOWN BY STRATEGY")
print("=" * 130)

years = sorted(df.index.year.unique())
yearly_all = []

for book in df.columns:
    print(f"\n{book}:")
    print(f"{'Year':<6} {'Return':<10} {'Sharpe':<10} {'Max DD':<10} {'Volatility':<10}")
    print("-" * 50)

    for year in years:
        year_mask = df.index.year == year
        year_returns = df[book][year_mask]

        if (year_returns == 0).all():
            continue

        metrics = calculate_metrics(year_returns)

        print(f"{year:<6} {metrics['annual_return']:>8.2%}  {metrics['sharpe']:>8.4f}  {metrics['max_dd']:>8.2%}  {metrics['volatility']:>8.2%}")

        yearly_all.append({
            "Book": book,
            "Year": year,
            "Return": metrics["annual_return"],
            "Sharpe": metrics["sharpe"],
            "Max_DD": metrics["max_dd"],
            "Volatility": metrics["volatility"],
        })

yearly_df = pd.DataFrame(yearly_all)

# ============================================================================
# GRID SEARCH 1: FIXED WEIGHT ALLOCATIONS
# ============================================================================

print("\n" + "=" * 130)
print("GRID SEARCH 1: FIXED WEIGHT ALLOCATIONS")
print("=" * 130)

books = ['A', 'B', 'C', 'D', 'E']
weight_options = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

results_fixed = []
combo_count = 0

for weights_tuple in product(weight_options, repeat=len(books)):
    total = sum(weights_tuple)
    if abs(total - 1.0) < 0.001:
        combo_count += 1
        weights = dict(zip(books, weights_tuple))

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
            "Volatility": metrics["volatility"],
        })

results_fixed_df = pd.DataFrame(results_fixed)
results_fixed_df = results_fixed_df.sort_values("Sharpe", ascending=False)

print(f"\nTotal combinations tested: {combo_count}")
print(f"\nTop 20 allocations by Sharpe ratio:\n")
print(results_fixed_df.head(20).to_string(index=False))

best_fixed = results_fixed_df.iloc[0]
print(f"\n[BEST] FIXED WEIGHT ALLOCATION:")
print(f"  A:{best_fixed['A']:.0%} B:{best_fixed['B']:.0%} C:{best_fixed['C']:.0%} D:{best_fixed['D']:.0%} E:{best_fixed['E']:.0%}")
print(f"  Sharpe: {best_fixed['Sharpe']:.4f}")
print(f"  Annual Return: {best_fixed['Ann_Return']:.2%}")
print(f"  Max DD: {best_fixed['Max_DD']:.2%}")
print(f"  Volatility: {best_fixed['Volatility']:.2%}")

# ============================================================================
# GRID SEARCH 2: MOMENTUM ALLOCATION
# ============================================================================

print("\n" + "=" * 130)
print("GRID SEARCH 2: MOMENTUM ALLOCATION (Rolling Sharpe)")
print("=" * 130)

lookback_windows = [20, 30, 60, 90, 120, 150]
results_momentum = []

for lookback in lookback_windows:
    print(f"\n  Testing lookback={lookback}d...")

    weights_ts = pd.DataFrame(index=df.index, columns=books, dtype=float)

    for i, date in enumerate(df.index):
        if i < lookback:
            w = {book: 1.0 / len(books) for book in books}
        else:
            start_idx = i - lookback
            sharpes = {}
            for book in books:
                rets = df[book].iloc[start_idx:i]
                metrics = calculate_metrics(rets)
                sharpe = max(metrics['sharpe'], 0.1)
                sharpes[book] = sharpe

            total = sum(sharpes.values())
            w = {book: sharpes[book] / total for book in sharpes.keys()}

        for book in w:
            weights_ts.loc[date, book] = w[book]

    port_ret = (df * weights_ts).sum(axis=1)
    metrics = calculate_metrics(port_ret)

    results_momentum.append({
        "Lookback_Days": lookback,
        "Ann_Return": metrics["annual_return"],
        "Sharpe": metrics["sharpe"],
        "Sortino": metrics["sortino"],
        "Max_DD": metrics["max_dd"],
        "Volatility": metrics["volatility"],
    })

results_momentum_df = pd.DataFrame(results_momentum)
results_momentum_df = results_momentum_df.sort_values("Sharpe", ascending=False)

print(f"\nMomentum Allocation Results (sorted by Sharpe):\n")
print(results_momentum_df.to_string(index=False))

best_momentum = results_momentum_df.iloc[0]
print(f"\n[BEST] MOMENTUM ALLOCATION:")
print(f"  Lookback: {int(best_momentum['Lookback_Days'])} days")
print(f"  Sharpe: {best_momentum['Sharpe']:.4f}")
print(f"  Annual Return: {best_momentum['Ann_Return']:.2%}")
print(f"  Max DD: {best_momentum['Max_DD']:.2%}")
print(f"  Volatility: {best_momentum['Volatility']:.2%}")

# ============================================================================
# YEARLY PERFORMANCE OF BEST ALLOCATIONS
# ============================================================================

print("\n" + "=" * 130)
print("YEARLY BREAKDOWN: BEST FIXED WEIGHT ALLOCATION")
print("=" * 130)

best_fixed_weights = {
    "A": best_fixed['A'],
    "B": best_fixed['B'],
    "C": best_fixed['C'],
    "D": best_fixed['D'],
    "E": best_fixed['E'],
}

best_fixed_port = sum(df[book] * best_fixed_weights[book] for book in books)

print(f"\nAllocation: A:{best_fixed['A']:.0%} B:{best_fixed['B']:.0%} C:{best_fixed['C']:.0%} D:{best_fixed['D']:.0%} E:{best_fixed['E']:.0%}")
print(f"{'Year':<6} {'Return':<10} {'Sharpe':<10} {'Max DD':<10} {'Volatility':<10}")
print("-" * 50)

yearly_best_fixed = []
for year in years:
    year_mask = df.index.year == year
    year_returns = best_fixed_port[year_mask]
    metrics = calculate_metrics(year_returns)

    print(f"{year:<6} {metrics['annual_return']:>8.2%}  {metrics['sharpe']:>8.4f}  {metrics['max_dd']:>8.2%}  {metrics['volatility']:>8.2%}")

    yearly_best_fixed.append({
        "Year": year,
        "Return": metrics["annual_return"],
        "Sharpe": metrics["sharpe"],
        "Max_DD": metrics["max_dd"],
    })

# ============================================================================
# COMPARISON & WINNER
# ============================================================================

print("\n" + "=" * 130)
print("FINAL COMPARISON")
print("=" * 130)

comparison = pd.DataFrame([
    {
        "Strategy": "Fixed Weight (Best)",
        "Allocation": f"A:{best_fixed['A']:.0%} B:{best_fixed['B']:.0%} C:{best_fixed['C']:.0%} D:{best_fixed['D']:.0%} E:{best_fixed['E']:.0%}",
        "Sharpe": f"{best_fixed['Sharpe']:.4f}",
        "Ann_Return": f"{best_fixed['Ann_Return']:.2%}",
        "Max_DD": f"{best_fixed['Max_DD']:.2%}",
        "Volatility": f"{best_fixed['Volatility']:.2%}",
    },
    {
        "Strategy": "Momentum (Best)",
        "Allocation": f"{int(best_momentum['Lookback_Days'])}d lookback",
        "Sharpe": f"{best_momentum['Sharpe']:.4f}",
        "Ann_Return": f"{best_momentum['Ann_Return']:.2%}",
        "Max_DD": f"{best_momentum['Max_DD']:.2%}",
        "Volatility": f"{best_momentum['Volatility']:.2%}",
    },
])

print(f"\n{comparison.to_string(index=False)}\n")

if best_fixed['Sharpe'] > best_momentum['Sharpe']:
    winner = "FIXED WEIGHT"
    advantage = ((best_fixed['Sharpe'] / best_momentum['Sharpe'] - 1) * 100)
    print(f"[WINNER] {winner}: {advantage:.1f}% better Sharpe")
else:
    winner = "MOMENTUM"
    advantage = ((best_momentum['Sharpe'] / best_fixed['Sharpe'] - 1) * 100)
    print(f"[WINNER] {winner}: {advantage:.1f}% better Sharpe")

# ============================================================================
# SAVE RESULTS
# ============================================================================

out_file = Path("results") / "backtest_grid_search_complete.xlsx"

with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
    # Summary
    comparison.to_excel(writer, sheet_name="Summary", index=False)

    # Individual books
    individual_df.to_excel(writer, sheet_name="Individual_Books", index=False)

    # Yearly summary
    yearly_pivot = yearly_df.pivot_table(
        index='Year',
        columns='Book',
        values=['Return', 'Sharpe', 'Max_DD']
    )
    yearly_pivot.to_excel(writer, sheet_name="Yearly_Summary")

    # Fixed weight top 50
    results_fixed_sorted = results_fixed_df.sort_values("Sharpe", ascending=False)
    results_fixed_sorted.head(50).to_excel(writer, sheet_name="Fixed_Weight_Top50", index=False)

    # All fixed weight
    results_fixed_sorted.to_excel(writer, sheet_name="Fixed_Weight_All", index=False)

    # Momentum results
    results_momentum_df.to_excel(writer, sheet_name="Momentum_Allocation", index=False)

    # Best fixed weight yearly
    best_fixed_yearly_df = pd.DataFrame(yearly_best_fixed)
    best_fixed_yearly_df.to_excel(writer, sheet_name="Best_Fixed_Yearly", index=False)

print(f"[SAVED] Results to: {out_file}\n")

print("=" * 130)
print("BACKTEST COMPLETE")
print("=" * 130)
print(f"\nSummary:")
print(f"  Period: {df.index[0].date()} to {df.index[-1].date()}")
print(f"  Trading days: {len(df)}")
print(f"  Fixed weight combinations: {combo_count}")
print(f"  Momentum lookbacks: {len(lookback_windows)}")
print(f"\nBest Fixed Weight: {winner if winner == 'FIXED WEIGHT' else 'MOMENTUM'}")
print(f"  Sharpe: {best_fixed['Sharpe']:.4f if winner == 'FIXED WEIGHT' else best_momentum['Sharpe']:.4f}")
print(f"  Return: {best_fixed['Ann_Return']:.2%}" if winner == 'FIXED WEIGHT' else f"  Return: {best_momentum['Ann_Return']:.2%}")
print()
