"""
Corrected backtest from 2019 onwards using actual max drawdown data from CSV.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_documented_yearly_data():
    """Load yearly returns WITH actual max drawdowns from CSV."""
    csv_path = Path(__file__).parent / "results" / "portfolio_4strategy_tc025_yearly.csv"
    df = pd.read_csv(csv_path)
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


def backtest_2019_with_correct_dd(yearly_df):
    """Backtest 2019-2026 using ACTUAL max drawdown data from CSV."""
    print("\n" + "="*100)
    print("BACKTEST 1: 2019-2026 (CORRECTED with Actual Max Drawdown Data)")
    print("="*100)

    years_2019 = yearly_df[yearly_df["Year"] >= 2019].copy()
    print(f"\nYears: {int(years_2019['Year'].min())} to {int(years_2019['Year'].max())}")
    print(f"Data points: {len(years_2019)}")

    # Extract returns and max drawdowns from CSV
    books = ["A", "B", "C", "D"]
    book_returns = {}
    book_max_dd = {}

    for book in books:
        ret_col = f"Ret_{book}"
        dd_col = f"DD_{book}"

        if ret_col in years_2019.columns and dd_col in years_2019.columns:
            book_returns[book] = years_2019[ret_col].values
            book_max_dd[book] = years_2019[dd_col].values  # Actual max drawdowns per year

    print(f"\nBooks: {list(book_returns.keys())}")

    # Show individual book performance
    print("\n" + "-"*100)
    print("Individual Book Performance (2019-2026):")
    print("-"*100)

    individual_metrics = {}
    for book in books:
        returns = book_returns[book]
        max_dds = book_max_dd[book]

        returns_series = pd.Series(returns)
        cum = (1 + returns_series).prod() - 1
        years = len(returns_series)
        ann = (1 + cum) ** (1 / years) - 1 if years > 0 else cum

        # Sharpe from yearly returns
        rf_annual = 0.02
        excess = returns_series - rf_annual
        sharpe = (excess.mean() / returns_series.std() * np.sqrt(years)) if returns_series.std() > 0 else 0.0

        # Max drawdown is the WORST drawdown across all years
        worst_dd = max_dds.min()  # Most negative value

        individual_metrics[book] = {
            "annual_return": ann,
            "sharpe": sharpe,
            "max_dd": worst_dd,
            "cum_return": cum,
        }

        print(f"Book {book}:")
        print(f"  Annual Return (geometric): {ann:.2%}")
        print(f"  Cumulative Return: {cum:.2%}")
        print(f"  Sharpe Ratio (yearly): {sharpe:.3f}")
        print(f"  Max Drawdown (worst year): {worst_dd:.2%}")
        print(f"  Yearly drawdowns: {[f'{dd:.2%}' for dd in max_dds]}")

    # Grid search allocations
    print("\n" + "-"*100)
    print("Grid Search: Fixed Weight Allocations (2019-2026)")
    print("-"*100)

    results = []
    weight_options = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    from itertools import product
    for weights_tuple in product(weight_options, repeat=len(book_returns)):
        total = sum(weights_tuple)
        if abs(total - 1.0) < 0.001:
            weights = dict(zip(sorted(book_returns.keys()), weights_tuple))

            # Calculate portfolio return (yearly compounding)
            port_ret = sum(book_returns[b] * weights[b] for b in book_returns.keys())
            port_ret = pd.Series(port_ret)

            cum = (1 + port_ret).prod() - 1
            years_count = len(port_ret)
            ann = (1 + cum) ** (1 / years_count) - 1 if years_count > 0 else cum

            rf_annual = 0.02
            excess = port_ret - rf_annual
            sharpe = (excess.mean() / port_ret.std() * np.sqrt(years_count)) if port_ret.std() > 0 else 0.0

            # Portfolio max drawdown: weighted average of worst individual drawdowns
            # (Conservative: worst case is all holdings drawdown simultaneously)
            port_max_dd = sum(weights[b] * individual_metrics[b]["max_dd"] for b in book_returns.keys())

            results.append({
                **weights,
                "Ann_Return": ann,
                "Sharpe": sharpe,
                "Max_DD": port_max_dd,
                "Cum_Return": cum,
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Sharpe", ascending=False)

    print(f"\nTotal combinations: {len(results_df)}")
    print(f"\nTop 15 allocations by Sharpe (2019-2026):")
    print(results_df.head(15)[["A", "B", "C", "D", "Ann_Return", "Sharpe", "Max_DD"]].to_string(index=False))

    best = results_df.iloc[0]
    print(f"\n\nBest Allocation by Sharpe (2019-2026):")
    print(f"  A: {best['A']:.0%}  B: {best['B']:.0%}  C: {best['C']:.0%}  D: {best['D']:.0%}")
    print(f"  Sharpe: {best['Sharpe']:.3f}")
    print(f"  Ann Return: {best['Ann_Return']:.2%}")
    print(f"  Max DD: {best['Max_DD']:.2%}")

    # Show top allocations by other metrics
    print(f"\n\nTop 5 by Different Metrics:")
    best_return = results_df.loc[results_df["Ann_Return"].idxmax()]
    print(f"  Highest Return: {best_return['Ann_Return']:.2%} (A:{best_return['A']:.0%} D:{best_return['D']:.0%})")

    best_dd = results_df.loc[results_df["Max_DD"].idxmax()]
    print(f"  Best Max DD: {best_dd['Max_DD']:.2%} (A:{best_dd['A']:.0%} D:{best_dd['D']:.0%})")

    return results_df, individual_metrics


def backtest_recent_daily(daily_df):
    """Backtest 2024-2026 using daily data with correct max drawdowns."""
    if daily_df is None:
        return None

    print("\n" + "="*100)
    print("BACKTEST 2: 2024-2026 Using Daily Data (CORRECTED)")
    print("="*100)

    books = ["A", "B", "C", "D", "E"]
    print(f"\nPeriod: {daily_df.index[0].date()} to {daily_df.index[-1].date()}")
    print(f"Trading days: {len(daily_df)}")

    # Calculate individual metrics with proper max drawdown
    print("\nIndividual Book Performance (2024-2026):")
    individual_metrics = {}
    for book in books:
        rets = daily_df[book].dropna()
        cum = (1 + rets).prod() - 1
        years = len(rets) / 252
        ann = (1 + cum) ** (1 / years) - 1 if years > 0 else cum

        rf_daily = 0.02 / 252
        sharpe = ((rets.mean() - rf_daily) / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0

        # CORRECT max drawdown calculation
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

            # CORRECT max drawdown
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

    print(f"\nTotal combinations: {len(results_df)}")
    print(f"\nTop 10 allocations by Sharpe (2024-2026):")
    books_cols = [b for b in books if b in results_df.columns]
    print(results_df.head(10)[books_cols + ["Ann_Return", "Sharpe", "Max_DD"]].to_string(index=False))

    best = results_df.iloc[0]
    weights_str = " ".join(f"{b}:{best[b]:.0%}" for b in books)
    print(f"\n\nBest Allocation by Sharpe (2024-2026):")
    print(f"  {weights_str}")
    print(f"  Sharpe: {best['Sharpe']:.3f}")
    print(f"  Ann Return: {best['Ann_Return']:.2%}")
    print(f"  Max DD: {best['Max_DD']:.2%}")

    return results_df


def main():
    print("\n" + "="*100)
    print("CORRECTED BACKTEST: 2019-2026 with Proper Max Drawdown Calculations")
    print("="*100)

    # Load data
    yearly_df = load_documented_yearly_data()
    daily_df = load_recent_daily_data()

    # Backtest with correct max drawdown
    yearly_results, yearly_metrics = backtest_2019_with_correct_dd(yearly_df)
    daily_results = backtest_recent_daily(daily_df)

    # Save
    print("\n" + "="*100)
    print("SAVING RESULTS")
    print("="*100)

    out_dir = Path(__file__).parent / "results"
    out_file = out_dir / "backtest_2019_corrected.xlsx"

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        yearly_results.sort_values("Sharpe", ascending=False).to_excel(
            writer, sheet_name="2019-2026_Yearly", index=False
        )
        if daily_results is not None:
            daily_results.sort_values("Sharpe", ascending=False).to_excel(
                writer, sheet_name="2024-2026_Daily", index=False
            )
        # Individual metrics
        ind_df = pd.DataFrame(yearly_metrics).T
        ind_df.to_excel(writer, sheet_name="2019-2026_Individual")

    print(f"[SAVED] {out_file}")
    print("\n" + "="*100)


if __name__ == "__main__":
    main()
