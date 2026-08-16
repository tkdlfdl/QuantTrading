"""
Comprehensive backtest from 2019 onwards using documented yearly data + recent daily data.
Grid search allocations for Books A, B, C, D, E from 2019-2026.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_documented_yearly_data():
    """Load yearly returns from documented backtests (1997-2026)."""
    csv_path = Path(__file__).parent / "results" / "portfolio_4strategy_tc025_yearly.csv"
    df = pd.read_csv(csv_path)

    # Extract the columns we need for 2019+
    df_2019 = df[df["Year"] >= 2019].copy()

    return df_2019


def load_recent_daily_data():
    """Load actual daily returns from 2024-2026."""
    csv_path = Path(__file__).parent / "results" / "portfolio_5book_daily.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df.fillna(0)


def calculate_annual_metrics(returns_series, name=""):
    """Calculate metrics from returns series."""
    if len(returns_series) == 0:
        return {
            "n_periods": 0,
            "annual_return": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "cum_return": 0.0,
            "vol": 0.0,
        }

    # Annual return (already given as yearly data)
    cum = (1 + returns_series).prod() - 1

    # Sharpe (from returns)
    rf_annual = 0.02
    excess = returns_series - rf_annual
    sharpe = (excess.mean() / returns_series.std() * np.sqrt(len(returns_series))) if returns_series.std() > 0 else 0.0

    # Max drawdown (from cumulative)
    wealth = (1 + returns_series).cumprod()
    max_dd = float((wealth / wealth.cummax() - 1).min())

    # Annualized return
    years = len(returns_series)
    ann = (1 + cum) ** (1 / years) - 1 if years > 0 else cum

    return {
        "n_periods": len(returns_series),
        "annual_return": ann,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "cum_return": cum,
        "vol": returns_series.std(),
    }


def backtest_allocations_yearly_data(yearly_df):
    """Grid search allocations using yearly data."""
    print("\n" + "="*100)
    print("BACKTEST 1: 2019-2026 Using Documented Yearly Data (Books A, B, C, D)")
    print("="*100)

    years_2019 = yearly_df[yearly_df["Year"] >= 2019].copy()
    print(f"\nYears covered: {int(years_2019['Year'].min())} to {int(years_2019['Year'].max())}")
    print(f"Data points: {len(years_2019)}")

    # Extract book returns
    book_returns = {}
    for book in ["A", "B", "C", "D"]:
        col = f"Ret_{book}"
        if col in years_2019.columns:
            book_returns[book] = years_2019[col].values
        else:
            print(f"  Warning: Column Ret_{book} not found")

    print(f"\nBooks available: {list(book_returns.keys())}")

    # Show individual book performance
    print("\n" + "-"*100)
    print("Individual Book Performance (2019-2026):")
    print("-"*100)

    individual_metrics = {}
    for book, returns in book_returns.items():
        returns_series = pd.Series(returns)
        metrics = calculate_annual_metrics(returns_series)
        individual_metrics[book] = metrics

        print(f"Book {book}:")
        print(f"  Annual Return (geometric): {metrics['annual_return']:.2%}")
        print(f"  Cumulative Return: {metrics['cum_return']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe']:.3f}")
        print(f"  Max Drawdown: {metrics['max_dd']:.2%}")
        print(f"  Volatility: {metrics['vol']:.2%}")

    # Grid search allocations
    print("\n" + "-"*100)
    print("Grid Search: Fixed Weight Allocations (2019-2026)")
    print("-"*100)

    results = []
    weight_options = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    from itertools import product
    for weights_tuple in product(weight_options, repeat=len(book_returns)):
        total = sum(weights_tuple)
        if abs(total - 1.0) < 0.001:  # Sum to 100%
            weights = dict(zip(sorted(book_returns.keys()), weights_tuple))

            # Calculate portfolio return
            port_ret = sum(book_returns[b] * weights[b] for b in book_returns.keys())
            port_ret = pd.Series(port_ret)

            metrics = calculate_annual_metrics(port_ret)

            results.append({
                **weights,
                "Ann_Return": metrics["annual_return"],
                "Sharpe": metrics["sharpe"],
                "Max_DD": metrics["max_dd"],
                "Cum_Return": metrics["cum_return"],
                "Vol": metrics["vol"],
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Sharpe", ascending=False)

    print(f"\nTotal combinations tested: {len(results_df)}")
    print(f"\nTop 10 allocations by Sharpe (2019-2026):")
    print(results_df.head(10).to_string(index=False))

    best = results_df.iloc[0]
    print(f"\n\nBest Allocation (by Sharpe):")
    print(f"  A: {best['A']:.0%}  B: {best['B']:.0%}  C: {best['C']:.0%}  D: {best['D']:.0%}")
    print(f"  Sharpe: {best['Sharpe']:.3f}")
    print(f"  Ann Return: {best['Ann_Return']:.2%}")
    print(f"  Max DD: {best['Max_DD']:.2%}")

    return results_df, individual_metrics


def backtest_recent_period_enhanced(daily_df):
    """Enhanced grid search on recent 2024-2026 daily data."""
    if daily_df is None:
        return None

    print("\n" + "="*100)
    print("BACKTEST 2: 2024-2026 Using Actual Daily Data (Books A, B, C, D, E)")
    print("="*100)

    books = ["A", "B", "C", "D", "E"]
    print(f"\nPeriod: {daily_df.index[0].date()} to {daily_df.index[-1].date()}")
    print(f"Trading days: {len(daily_df)}")

    # Calculate individual metrics
    print("\nIndividual Book Performance (2024-2026 Daily):")
    individual_metrics = {}
    for book in books:
        rets = daily_df[book].dropna()
        cum = (1 + rets).prod() - 1
        years = len(rets) / 252
        ann = (1 + cum) ** (1 / years) - 1 if years > 0 else cum

        rf_daily = 0.02 / 252
        sharpe = ((rets.mean() - rf_daily) / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0

        wealth = (1 + rets).cumprod()
        max_dd = float((wealth / wealth.cummax() - 1).min())

        individual_metrics[book] = {
            "annual_return": ann,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "cum_return": cum,
        }

        print(f"  {book}: Sharpe {sharpe:.3f}, Ann {ann:.2%}, MaxDD {max_dd:.2%}")

    # Grid search
    results = []
    weight_options = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    from itertools import product
    for weights_tuple in product(weight_options, repeat=len(books)):
        total = sum(weights_tuple)
        if abs(total - 1.0) < 0.001:
            weights = dict(zip(books, weights_tuple))

            port_ret = sum(daily_df[b] * weights[b] for b in books)
            rets = port_ret.dropna()
            cum = (1 + rets).prod() - 1
            years = len(rets) / 252
            ann = (1 + cum) ** (1 / years) - 1 if years > 0 else cum

            rf_daily = 0.02 / 252
            sharpe = ((rets.mean() - rf_daily) / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0

            wealth = (1 + rets).cumprod()
            max_dd = float((wealth / wealth.cummax() - 1).min())

            results.append({
                **weights,
                "Ann_Return": ann,
                "Sharpe": sharpe,
                "Max_DD": max_dd,
                "Cum_Return": cum,
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Sharpe", ascending=False)

    print(f"\nTotal combinations tested: {len(results_df)}")
    print(f"\nTop 10 allocations by Sharpe (2024-2026 Daily):")
    print(results_df.head(10).to_string(index=False))

    best = results_df.iloc[0]
    print(f"\n\nBest Allocation (by Sharpe):")
    weights_str = " ".join(f"{b}:{best[b]:.0%}" for b in books)
    print(f"  {weights_str}")
    print(f"  Sharpe: {best['Sharpe']:.3f}")
    print(f"  Ann Return: {best['Ann_Return']:.2%}")
    print(f"  Max DD: {best['Max_DD']:.2%}")

    return results_df


def compare_periods(yearly_results, daily_results):
    """Compare 2019-2026 vs 2024-2026 results."""
    print("\n" + "="*100)
    print("COMPARISON: 2019-2026 (Yearly Data) vs 2024-2026 (Daily Data)")
    print("="*100)

    print("\n2019-2026 Best (4 books: A,B,C,D):")
    best_yearly = yearly_results.iloc[0]
    print(f"  A: {best_yearly['A']:.0%}  B: {best_yearly['B']:.0%}  C: {best_yearly['C']:.0%}  D: {best_yearly['D']:.0%}")
    print(f"  Sharpe: {best_yearly['Sharpe']:.3f}, Ann: {best_yearly['Ann_Return']:.2%}, MaxDD: {best_yearly['Max_DD']:.2%}")

    if daily_results is not None:
        print("\n2024-2026 Best (5 books: A,B,C,D,E):")
        best_daily = daily_results.iloc[0]
        books = ["A", "B", "C", "D", "E"]
        weights_str = " ".join(f"{b}:{best_daily[b]:.0%}" for b in books)
        print(f"  {weights_str}")
        print(f"  Sharpe: {best_daily['Sharpe']:.3f}, Ann: {best_daily['Ann_Return']:.2%}, MaxDD: {best_daily['Max_DD']:.2%}")

        print("\nObservations:")
        print(f"  - Different time periods show different optimal allocations")
        print(f"  - 2019-2026 Sharpe: {best_yearly['Sharpe']:.3f} vs 2024-2026 Sharpe: {best_daily['Sharpe']:.3f}")
        print(f"  - Suggests strategies have different performance in different market regimes")


def main():
    print("\n" + "="*100)
    print("COMPREHENSIVE BACKTEST: 2019-2026 Grid Search with Allocation Optimization")
    print("="*100)

    # Load data
    print("\nLoading data...")
    yearly_df = load_documented_yearly_data()
    daily_df = load_recent_daily_data()

    # Backtest 1: 2019-2026 yearly data (4 books)
    yearly_results, yearly_metrics = backtest_allocations_yearly_data(yearly_df)

    # Backtest 2: 2024-2026 daily data (5 books)
    daily_results = backtest_recent_period_enhanced(daily_df)

    # Compare
    compare_periods(yearly_results, daily_results)

    # Save results
    print("\n" + "="*100)
    print("SAVING RESULTS")
    print("="*100)

    out_dir = Path(__file__).parent / "results"
    out_file = out_dir / "backtest_2019_onwards_comprehensive.xlsx"

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        yearly_results.sort_values("Sharpe", ascending=False).to_excel(
            writer, sheet_name="2019-2026_Yearly_Search", index=False
        )
        if daily_results is not None:
            daily_results.sort_values("Sharpe", ascending=False).to_excel(
                writer, sheet_name="2024-2026_Daily_Search", index=False
            )

    print(f"[SAVED] {out_file}")

    print("\n" + "="*100)
    print("BACKTEST COMPLETE")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
