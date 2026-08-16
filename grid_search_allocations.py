"""
Grid search backtest for portfolio allocations.

Tests:
1. MomAlloc: Different lookback windows (30d, 60d, 90d, 120d)
2. FixedEW: Different weight combinations across A, B, C, D, E
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import warnings

warnings.filterwarnings('ignore')


def load_daily_returns():
    """Load actual daily returns from 2024-2026."""
    csv_path = Path(__file__).parent / "results" / "portfolio_5book_daily.csv"
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df = df.fillna(0)
    return df


def calculate_metrics(returns_series, name=""):
    """Calculate comprehensive metrics from returns."""
    r = returns_series.dropna()
    if len(r) == 0 or (r == 0).all():
        return {
            "name": name,
            "n_days": 0,
            "annual_return": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_dd": 0.0,
            "cum_return": 0.0,
            "win_rate": 0.0,
            "vol": 0.0,
        }

    # Annual metrics
    cum = (1 + r).prod() - 1
    years = len(r) / 252
    ann = (1 + cum) ** (1 / years) - 1 if years > 0 else cum

    # Sharpe
    rf_daily = 0.02 / 252
    vol = r.std()
    sharpe = ((r.mean() - rf_daily) / vol * np.sqrt(252)) if vol > 0 else 0.0

    # Sortino
    down_std = r[r < 0].std()
    sortino = ((r.mean() - rf_daily) / down_std * np.sqrt(252)) if down_std and down_std > 0 else 0.0

    # Max drawdown
    wealth = (1 + r).cumprod()
    max_dd = float((wealth / wealth.cummax() - 1).min())

    # Win rate
    win_rate = float((r > 0).mean())

    return {
        "name": name,
        "n_days": len(r),
        "annual_return": ann,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "cum_return": cum,
        "win_rate": win_rate,
        "vol": vol,
    }


def grid_search_fixed_weights(daily_returns):
    """Grid search different fixed weight allocations."""
    print("\n" + "="*100)
    print("GRID SEARCH 1: FIXED WEIGHT ALLOCATIONS")
    print("="*100)

    books = ["A", "B", "C", "D", "E"]

    # Generate weight combinations (each book gets 0%, 10%, 20%, 30%, 40%)
    # Constrained to sum to 100%
    weight_options = [0, 0.1, 0.2, 0.3, 0.4]
    results = []

    # Generate all valid weight combinations
    combo_count = 0
    for weights_tuple in product(weight_options, repeat=len(books)):
        total = sum(weights_tuple)
        if abs(total - 1.0) < 0.001:  # Sum to 100%
            combo_count += 1

            weights = dict(zip(books, weights_tuple))

            # Skip all-zero weights
            if sum(weights.values()) < 0.99:
                continue

            # Calculate portfolio return
            port_ret = sum(daily_returns[book] * weights[book] for book in books)
            metrics = calculate_metrics(port_ret, name=f"A:{weights['A']:.0%} B:{weights['B']:.0%} C:{weights['C']:.0%} D:{weights['D']:.0%} E:{weights['E']:.0%}")

            results.append({
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

    results_df = pd.DataFrame(results)

    # Sort by Sharpe
    results_df = results_df.sort_values("Sharpe", ascending=False)

    print(f"\nTotal weight combinations tested: {len(results_df)}")
    print(f"\nTop 15 allocations by Sharpe ratio:")
    print("\n" + results_df.head(15).to_string(index=False))

    # Find best by different metrics
    print(f"\n\nBest by different metrics:")
    print(f"  Sharpe:     {results_df.iloc[0]['Sharpe']:.3f} (A:{results_df.iloc[0]['A']:.0%} B:{results_df.iloc[0]['B']:.0%} C:{results_df.iloc[0]['C']:.0%} D:{results_df.iloc[0]['D']:.0%} E:{results_df.iloc[0]['E']:.0%})")

    best_return = results_df.loc[results_df["Ann_Return"].idxmax()]
    print(f"  Ann Return: {best_return['Ann_Return']:.2%} (A:{best_return['A']:.0%} B:{best_return['B']:.0%} C:{best_return['C']:.0%} D:{best_return['D']:.0%} E:{best_return['E']:.0%})")

    best_risk = results_df.loc[results_df["Max_DD"].idxmax()]
    print(f"  Max DD:     {best_risk['Max_DD']:.2%} (A:{best_risk['A']:.0%} B:{best_risk['B']:.0%} C:{best_risk['C']:.0%} D:{best_risk['D']:.0%} E:{best_risk['E']:.0%})")

    return results_df


def grid_search_momentum_allocation(daily_returns):
    """Grid search different momentum allocation lookback windows."""
    print("\n" + "="*100)
    print("GRID SEARCH 2: MOMENTUM ALLOCATION (Rolling Sharpe Rebalance)")
    print("="*100)

    books = ["A", "B", "C", "D", "E"]
    lookback_windows = [20, 30, 60, 90, 120]
    results = []

    for lookback in lookback_windows:
        print(f"\n  Testing lookback={lookback}d...")

        # Create weights time series based on rolling Sharpe
        weights_ts = pd.DataFrame(index=daily_returns.index, columns=books, dtype=float)

        for i, date in enumerate(daily_returns.index):
            if i < lookback:
                # Equal weight during warmup
                w = {book: 1.0 / len(books) for book in books}
            else:
                # Compute rolling Sharpe for past `lookback` days
                start_idx = i - lookback
                sharpes = {}
                for book in books:
                    rets = daily_returns[book].iloc[start_idx:i]
                    metrics = calculate_metrics(rets)
                    sharpe = max(metrics['sharpe'], 0.1)  # Floor at 0.1
                    sharpes[book] = sharpe

                # Normalize to sum to 1
                total = sum(sharpes.values())
                w = {book: sharpes[book] / total for book in sharpes.keys()}

            for book in w:
                weights_ts.loc[date, book] = w[book]

        # Calculate portfolio return
        port_ret = (daily_returns * weights_ts).sum(axis=1)
        metrics = calculate_metrics(port_ret, name=f"lookback={lookback}d")

        results.append({
            "Lookback_Days": lookback,
            "Ann_Return": metrics["annual_return"],
            "Sharpe": metrics["sharpe"],
            "Sortino": metrics["sortino"],
            "Max_DD": metrics["max_dd"],
            "Cum_Return": metrics["cum_return"],
            "Win_Rate": metrics["win_rate"],
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Sharpe", ascending=False)

    print(f"\nMomentum Allocation Results (sorted by Sharpe):")
    print("\n" + results_df.to_string(index=False))

    best = results_df.iloc[0]
    print(f"\nBest MomAlloc: lookback={int(best['Lookback_Days'])}d, Sharpe={best['Sharpe']:.3f}, Ann={best['Ann_Return']:.2%}")

    return results_df


def compare_allocations(fixed_results, moma_results):
    """Compare all tested allocations."""
    print("\n" + "="*100)
    print("ALLOCATION COMPARISON: Top Candidates")
    print("="*100)

    # Get top fixed allocation
    top_fixed = fixed_results.iloc[0]
    top_fixed_desc = f"Fixed: A={top_fixed['A']:.0%} B={top_fixed['B']:.0%} C={top_fixed['C']:.0%} D={top_fixed['D']:.0%} E={top_fixed['E']:.0%}"

    # Get top momentum allocation
    top_moma = moma_results.iloc[0]
    top_moma_desc = f"MomAlloc: lookback={int(top_moma['Lookback_Days'])}d"

    # Also include equal-weight for reference
    equal_w = {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2}
    equal_port = 0.2 * (fixed_results.index == 0).sum()  # Find equal weight in results
    equal_w_row = fixed_results[
        (fixed_results["A"] == 0.2) &
        (fixed_results["B"] == 0.2) &
        (fixed_results["C"] == 0.2) &
        (fixed_results["D"] == 0.2) &
        (fixed_results["E"] == 0.2)
    ]

    comparison = []
    comparison.append({
        "Allocation": top_fixed_desc,
        "Ann Return": f"{top_fixed['Ann_Return']:.2%}",
        "Sharpe": f"{top_fixed['Sharpe']:.3f}",
        "Max DD": f"{top_fixed['Max_DD']:.2%}",
        "Cum Return": f"{top_fixed['Cum_Return']:.2%}",
    })

    comparison.append({
        "Allocation": top_moma_desc,
        "Ann Return": f"{top_moma['Ann_Return']:.2%}",
        "Sharpe": f"{top_moma['Sharpe']:.3f}",
        "Max DD": f"{top_moma['Max_DD']:.2%}",
        "Cum Return": f"{top_moma['Cum_Return']:.2%}",
    })

    if not equal_w_row.empty:
        eq = equal_w_row.iloc[0]
        comparison.append({
            "Allocation": "Equal-Weight (20% each)",
            "Ann Return": f"{eq['Ann_Return']:.2%}",
            "Sharpe": f"{eq['Sharpe']:.3f}",
            "Max DD": f"{eq['Max_DD']:.2%}",
            "Cum Return": f"{eq['Cum_Return']:.2%}",
        })

    comp_df = pd.DataFrame(comparison)
    print("\n" + comp_df.to_string(index=False))

    return comp_df


def save_results(fixed_df, moma_df, comp_df):
    """Save grid search results to Excel."""
    out_file = Path(__file__).parent / "results" / "grid_search_allocations.xlsx"

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        # Summary
        comp_df.to_excel(writer, sheet_name="Summary", index=False)

        # Fixed weight results
        fixed_sorted = fixed_df.sort_values("Sharpe", ascending=False)
        fixed_sorted.to_excel(writer, sheet_name="Fixed_Weight_Search", index=False)

        # Momentum allocation results
        moma_df.to_excel(writer, sheet_name="MomAlloc_Search", index=False)

    print(f"\n[SAVED] Results: {out_file}")


def main():
    print("\n" + "="*100)
    print("GRID SEARCH BACKTEST: Portfolio Allocations".center(100))
    print("="*100)

    # Load data
    print("\nLoading daily returns (2024-2026)...")
    daily_returns = load_daily_returns()
    print(f"  Period: {daily_returns.index[0].date()} to {daily_returns.index[-1].date()}")
    print(f"  Books: {list(daily_returns.columns)}")
    print(f"  Trading days: {len(daily_returns)}")

    # Grid search 1: Fixed weights
    fixed_results = grid_search_fixed_weights(daily_returns)

    # Grid search 2: Momentum allocation
    moma_results = grid_search_momentum_allocation(daily_returns)

    # Compare top allocations
    comp_results = compare_allocations(fixed_results, moma_results)

    # Save results
    save_results(fixed_results, moma_results, comp_results)

    print("\n" + "="*100)
    print("GRID SEARCH COMPLETE".center(100))
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
