"""
Comprehensive Backtest: Strategies A, B, C, D, E (2019-2026)
Following BACKTEST_METHODOLOGY_GUIDELINES.md standards

Strategies:
- A: Daily Momentum + Leverage + UVXY Hedge
- B: QQQ Bubble Hourly (started 2020-07-27)
- C: Intraday MR + Momentum Flip (started 2019-01-02)
- D: Contrarian Bubble (started 2019-01-02)
- E: Reddit Sentiment Long-Only (started 2024-01-01)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from data.loader import load_close_panel
import warnings

warnings.filterwarnings('ignore')

print("=" * 120)
print("COMPREHENSIVE BACKTEST: STRATEGIES A, B, C, D, E (2019-2026)")
print("=" * 120)

# ============================================================================
# LOAD HISTORICAL DATA (2019-2026)
# ============================================================================

print("\n[LOADING DATA FROM 2019-2026...]")

# Check if we have a pre-computed file
csv_path = Path("results") / "portfolio_5book_daily_2019_2026.csv"

if csv_path.exists():
    print(f"Loading from cache: {csv_path.name}")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
else:
    print("Cache not found. Will use available 2024-2026 data.")
    print("Note: Earlier strategies (2019-2023) require separate computation.")
    csv_path_short = Path("results") / "portfolio_5book_daily.csv"
    if csv_path_short.exists():
        print(f"Using available data: {csv_path_short.name}")
        df = pd.read_csv(csv_path_short, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
    else:
        print("ERROR: No data files found")
        exit(1)

print(f"\nData loaded:")
print(f"  Period: {df.index[0].date()} to {df.index[-1].date()}")
print(f"  Duration: {(df.index[-1] - df.index[0]).days / 365.25:.2f} years")
print(f"  Books: {list(df.columns)}")
print(f"  Trading days: {len(df)}")

# Check data availability
print(f"\n[DATA AVAILABILITY]")
for book in df.columns:
    non_zero_days = (df[book] != 0).sum()
    pct = (non_zero_days / len(df)) * 100
    first_date = df[df[book] != 0].index[0] if non_zero_days > 0 else "N/A"
    print(f"  Book {book}: {non_zero_days}/{len(df)} days ({pct:.1f}%) - First: {first_date}")

# ============================================================================
# METRICS CALCULATION FUNCTIONS
# ============================================================================

def calculate_metrics(returns_series, name=""):
    """Calculate metrics from daily returns following proper Sharpe methodology."""
    r = returns_series.dropna()

    if len(r) == 0 or (r == 0).all():
        return {
            "annual_return": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_dd": 0.0,
            "cum_return": 0.0,
            "win_rate": 0.0,
            "volatility": 0.0,
        }

    # Annual return (geometric)
    cum = (1 + r).prod() - 1
    years = len(r) / 252
    ann = (1 + cum) ** (1 / years) - 1 if years > 0 else cum

    # Sharpe Ratio (CORRECT: from daily data, not yearly)
    rf_daily = 0.02 / 252
    vol = r.std()
    sharpe = ((r.mean() - rf_daily) / vol * np.sqrt(252)) if vol > 0 else 0.0

    # Sortino (downside risk)
    down_std = r[r < 0].std()
    sortino = ((r.mean() - rf_daily) / down_std * np.sqrt(252)) if down_std and down_std > 0 else 0.0

    # Max drawdown (from continuous wealth curve, not year-end)
    wealth = (1 + r).cumprod()
    max_dd = float((wealth / wealth.cummax() - 1).min())

    # Win rate
    win_rate = float((r > 0).mean())

    # Volatility (annualized)
    volatility = vol * np.sqrt(252)

    return {
        "annual_return": ann,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "cum_return": cum,
        "win_rate": win_rate,
        "volatility": volatility,
    }

# ============================================================================
# INDIVIDUAL BOOK BACKTESTS
# ============================================================================

print("\n" + "=" * 120)
print("INDIVIDUAL BOOK PERFORMANCE")
print("=" * 120)

individual_results = []

for book in df.columns:
    book_returns = df[book]
    metrics = calculate_metrics(book_returns, name=book)

    individual_results.append({
        "Book": book,
        "Annual_Return": metrics["annual_return"],
        "Sharpe": metrics["sharpe"],
        "Sortino": metrics["sortino"],
        "Max_DD": metrics["max_dd"],
        "Volatility": metrics["volatility"],
        "Win_Rate": metrics["win_rate"],
        "Cum_Return": metrics["cum_return"],
        "Observations": len(book_returns[book_returns != 0]),
    })

    print(f"\nBook {book}:")
    print(f"  Annual Return: {metrics['annual_return']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe']:.4f}")
    print(f"  Sortino Ratio: {metrics['sortino']:.4f}")
    print(f"  Max Drawdown: {metrics['max_dd']:.2%}")
    print(f"  Volatility: {metrics['volatility']:.2%}")
    print(f"  Win Rate: {metrics['win_rate']:.1%}")
    print(f"  Total Return: {metrics['cum_return']:+.2%}")

individual_df = pd.DataFrame(individual_results)
print(f"\n\n{individual_df.to_string(index=False)}\n")

# ============================================================================
# YEARLY BREAKDOWN FOR EACH BOOK
# ============================================================================

print("\n" + "=" * 120)
print("YEARLY BREAKDOWN FOR EACH BOOK")
print("=" * 120)

years = sorted(df.index.year.unique())

for book in df.columns:
    print(f"\n{'='*80}")
    print(f"BOOK {book} - Yearly Breakdown")
    print(f"{'='*80}")

    yearly_data = []

    for year in years:
        year_mask = df.index.year == year
        year_returns = df[book][year_mask]

        if (year_returns == 0).all():
            yearly_data.append({
                "Year": year,
                "Return": 0.0,
                "Sharpe": 0.0,
                "MaxDD": 0.0,
                "Observations": 0,
                "Note": "No data"
            })
            continue

        metrics = calculate_metrics(year_returns)

        yearly_data.append({
            "Year": year,
            "Return": metrics["annual_return"],
            "Sharpe": metrics["sharpe"],
            "MaxDD": metrics["max_dd"],
            "Volatility": metrics["volatility"],
            "Observations": len(year_returns[year_returns != 0]),
        })

    yearly_df = pd.DataFrame(yearly_data)
    print(f"\n{yearly_df.to_string(index=False)}\n")

# ============================================================================
# EQUAL WEIGHT PORTFOLIO
# ============================================================================

print("\n" + "=" * 120)
print("EQUAL WEIGHT PORTFOLIO (20% each book)")
print("=" * 120)

equal_weight_returns = df.mean(axis=1)
metrics = calculate_metrics(equal_weight_returns, "Equal Weight")

print(f"\nAllocation: 20% A + 20% B + 20% C + 20% D + 20% E")
print(f"Annual Return: {metrics['annual_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe']:.4f}")
print(f"Max Drawdown: {metrics['max_dd']:.2%}")
print(f"Volatility: {metrics['volatility']:.2%}")
print(f"Win Rate: {metrics['win_rate']:.1%}")

# Yearly breakdown
print(f"\nYearly Breakdown:")
yearly_eq = []
for year in years:
    year_mask = df.index.year == year
    year_returns = equal_weight_returns[year_mask]
    metrics_y = calculate_metrics(year_returns)
    yearly_eq.append({
        "Year": year,
        "Return": metrics_y["annual_return"],
        "Sharpe": metrics_y["sharpe"],
        "MaxDD": metrics_y["max_dd"],
    })

yearly_eq_df = pd.DataFrame(yearly_eq)
print(f"\n{yearly_eq_df.to_string(index=False)}\n")

# ============================================================================
# BEST FIXED WEIGHT ALLOCATION
# ============================================================================

print("\n" + "=" * 120)
print("BEST FIXED WEIGHT ALLOCATION")
print("=" * 120)

# Test 30A + 20C + 50D (best from grid search)
best_alloc = {"A": 0.30, "B": 0.0, "C": 0.20, "D": 0.50, "E": 0.0}

best_returns = sum(df[book] * best_alloc[book] for book in df.columns)
metrics = calculate_metrics(best_returns, "Best Fixed")

print(f"\nAllocation: 30% A + 0% B + 20% C + 50% D + 0% E")
print(f"Annual Return: {metrics['annual_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe']:.4f}")
print(f"Max Drawdown: {metrics['max_dd']:.2%}")
print(f"Volatility: {metrics['volatility']:.2%}")
print(f"Win Rate: {metrics['win_rate']:.1%}")

# Yearly breakdown
print(f"\nYearly Breakdown:")
yearly_best = []
for year in years:
    year_mask = df.index.year == year
    year_returns = best_returns[year_mask]
    metrics_y = calculate_metrics(year_returns)
    yearly_best.append({
        "Year": year,
        "Return": metrics_y["annual_return"],
        "Sharpe": metrics_y["sharpe"],
        "MaxDD": metrics_y["max_dd"],
    })

yearly_best_df = pd.DataFrame(yearly_best)
print(f"\n{yearly_best_df.to_string(index=False)}\n")

# ============================================================================
# SUMMARY COMPARISON
# ============================================================================

print("\n" + "=" * 120)
print("SUMMARY COMPARISON")
print("=" * 120)

comparison_data = []

for book in df.columns:
    metrics = calculate_metrics(df[book])
    comparison_data.append({
        "Strategy": f"Book {book} (Single)",
        "Ann_Return": metrics["annual_return"],
        "Sharpe": metrics["sharpe"],
        "Max_DD": metrics["max_dd"],
        "Volatility": metrics["volatility"],
    })

# Add portfolios
eq_metrics = calculate_metrics(equal_weight_returns)
comparison_data.append({
    "Strategy": "Equal Weight (20/20/20/20/20)",
    "Ann_Return": eq_metrics["annual_return"],
    "Sharpe": eq_metrics["sharpe"],
    "Max_DD": eq_metrics["max_dd"],
    "Volatility": eq_metrics["volatility"],
})

best_metrics = calculate_metrics(best_returns)
comparison_data.append({
    "Strategy": "Best Fixed (30A+20C+50D)",
    "Ann_Return": best_metrics["annual_return"],
    "Sharpe": best_metrics["sharpe"],
    "Max_DD": best_metrics["max_dd"],
    "Volatility": best_metrics["volatility"],
})

comp_df = pd.DataFrame(comparison_data)
comp_df = comp_df.sort_values("Sharpe", ascending=False)

print(f"\n{comp_df.to_string(index=False)}\n")

# ============================================================================
# SAVE RESULTS
# ============================================================================

out_file = Path("results") / "backtest_all_strategies_2019_2026.xlsx"

with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
    # Summary comparison
    comp_df.to_excel(writer, sheet_name="Summary", index=False)

    # Individual books
    individual_df.to_excel(writer, sheet_name="Individual_Books", index=False)

    # Yearly comparisons for each book
    for book in df.columns:
        yearly_data = []
        for year in years:
            year_mask = df.index.year == year
            year_returns = df[book][year_mask]
            metrics_y = calculate_metrics(year_returns)
            yearly_data.append({
                "Year": year,
                "Return": metrics_y["annual_return"],
                "Sharpe": metrics_y["sharpe"],
                "MaxDD": metrics_y["max_dd"],
                "Observations": len(year_returns[year_returns != 0]),
            })
        yearly_book_df = pd.DataFrame(yearly_data)
        yearly_book_df.to_excel(writer, sheet_name=f"Book_{book}_Yearly", index=False)

    # Equal weight
    yearly_eq_df.to_excel(writer, sheet_name="EqualWeight_Yearly", index=False)

    # Best fixed
    yearly_best_df.to_excel(writer, sheet_name="BestFixed_Yearly", index=False)

print(f"[SAVED] Results to: {out_file}")

print("\n" + "=" * 120)
print("BACKTEST COMPLETE")
print("=" * 120)
print(f"\nKey Findings:")
print(f"  - Period: {df.index[0].date()} to {df.index[-1].date()}")
print(f"  - Books analyzed: A, B, C, D, E (5 strategies)")
print(f"  - Daily observations: {len(df)} trading days")
print(f"  - Best book by Sharpe: {comp_df.iloc[0]['Strategy']}")
print(f"  - Best portfolio by Sharpe: {comp_df.iloc[1]['Strategy']}")
print(f"  - Methodology: Daily data, Sharpe = (mean_daily - rf) / std_daily * sqrt(252)")
print()
