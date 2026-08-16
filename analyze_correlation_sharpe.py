"""
Investigate why combining strategies reduced Sharpe ratio.
Check: correlations, individual Sharpes, and diversification benefit.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_daily_returns():
    """Load actual daily returns."""
    csv_path = Path(__file__).parent / "results" / "portfolio_5book_daily.csv"
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df = df.fillna(0)
    return df


def calculate_metrics(returns_series):
    """Calculate comprehensive metrics."""
    r = returns_series.dropna()
    if len(r) == 0 or (r == 0).all():
        return {
            "annual_return": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_dd": 0.0,
            "cum_return": 0.0,
            "vol": 0.0,
            "win_rate": 0.0,
        }

    cum = (1 + r).prod() - 1
    years = len(r) / 252
    ann = (1 + cum) ** (1 / years) - 1 if years > 0 else cum

    rf_daily = 0.02 / 252
    vol = r.std()
    sharpe = ((r.mean() - rf_daily) / vol * np.sqrt(252)) if vol > 0 else 0.0

    down_std = r[r < 0].std()
    sortino = ((r.mean() - rf_daily) / down_std * np.sqrt(252)) if down_std and down_std > 0 else 0.0

    wealth = (1 + r).cumprod()
    max_dd = float((wealth / wealth.cummax() - 1).min())
    win_rate = float((r > 0).mean())

    return {
        "annual_return": ann,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "cum_return": cum,
        "vol": vol,
        "win_rate": win_rate,
    }


def main():
    print("\n" + "="*100)
    print("CORRELATION & SHARPE ANALYSIS: Why Did Combined Strategies Underperform?")
    print("="*100)

    # Load data
    daily_returns = load_daily_returns()
    books = ["A", "B", "C", "D", "E"]

    print(f"\nData Period: {daily_returns.index[0].date()} to {daily_returns.index[-1].date()}")
    print(f"Trading Days: {len(daily_returns)}")

    # ========================================================================
    # PART 1: Individual Strategy Performance in 2024-2026
    # ========================================================================
    print("\n" + "="*100)
    print("PART 1: Individual Strategy Performance (2024-2026 Backtest Period)")
    print("="*100)

    individual_metrics = {}
    individual_results = []

    for book in books:
        metrics = calculate_metrics(daily_returns[book])
        individual_metrics[book] = metrics
        individual_results.append({
            "Book": book,
            "Ann Return": f"{metrics['annual_return']:.2%}",
            "Sharpe": f"{metrics['sharpe']:.3f}",
            "Volatility": f"{metrics['vol']:.2%}",
            "Max DD": f"{metrics['max_dd']:.2%}",
            "Win Rate": f"{metrics['win_rate']:.2%}",
        })

    ind_df = pd.DataFrame(individual_results)
    print("\n" + ind_df.to_string(index=False))

    # Ranked by Sharpe
    ranked = sorted(individual_metrics.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    print("\n\nRanked by Sharpe (2024-2026 Backtest Period):")
    for i, (book, metrics) in enumerate(ranked, 1):
        print(f"  {i}. Book {book}: Sharpe {metrics['sharpe']:.3f}, Ann {metrics['annual_return']:.2%}, Vol {metrics['vol']:.2%}")

    # ========================================================================
    # PART 2: Correlation Matrix
    # ========================================================================
    print("\n" + "="*100)
    print("PART 2: Strategy Correlation Matrix (2024-2026)")
    print("="*100)

    corr = daily_returns.corr()
    print("\n" + corr.to_string())

    print("\n\nCorrelation Observations:")
    for i, book1 in enumerate(books):
        for book2 in books[i+1:]:
            corr_val = corr.loc[book1, book2]
            print(f"  {book1}-{book2}: {corr_val:+.3f}", end="")
            if abs(corr_val) < 0.3:
                print(" [LOW correlation]")
            elif abs(corr_val) < 0.6:
                print(" [MEDIUM correlation]")
            else:
                print(" [HIGH correlation]")

    # ========================================================================
    # PART 3: Best Individual vs Best Combined
    # ========================================================================
    print("\n" + "="*100)
    print("PART 3: Best Individual vs Best Combined Portfolio")
    print("="*100)

    best_individual = ranked[0]
    print(f"\nBest Individual Strategy: Book {best_individual[0]}")
    print(f"  Sharpe: {best_individual[1]['sharpe']:.3f}")
    print(f"  Ann Return: {best_individual[1]['annual_return']:.2%}")
    print(f"  Volatility: {best_individual[1]['vol']:.2%}")

    # Best combined from grid search
    best_combined_alloc = {"A": 0.30, "B": 0.0, "C": 0.20, "D": 0.40, "E": 0.10}
    combined_ret = sum(daily_returns[b] * best_combined_alloc[b] for b in books)
    combined_metrics = calculate_metrics(combined_ret)

    print(f"\nBest Combined Portfolio: 30% A + 20% C + 40% D + 10% E")
    print(f"  Sharpe: {combined_metrics['sharpe']:.3f}")
    print(f"  Ann Return: {combined_metrics['annual_return']:.2%}")
    print(f"  Volatility: {combined_metrics['vol']:.2%}")

    print(f"\nComparison:")
    print(f"  Sharpe Delta: {combined_metrics['sharpe'] - best_individual[1]['sharpe']:+.3f}")
    print(f"  Return Delta: {combined_metrics['annual_return'] - best_individual[1]['annual_return']:+.2%}")
    print(f"  Vol Delta: {combined_metrics['vol'] - best_individual[1]['vol']:+.2%}")

    # ========================================================================
    # PART 4: Correlation Impact on Volatility
    # ========================================================================
    print("\n" + "="*100)
    print("PART 4: Theoretical vs Actual Volatility Reduction")
    print("="*100)

    alloc = best_combined_alloc
    weights = np.array([alloc["A"], alloc["B"], alloc["C"], alloc["D"], alloc["E"]])
    cov = daily_returns.cov()

    # Theoretical portfolio variance
    theo_var = weights @ cov @ weights
    theo_vol = np.sqrt(theo_var)

    print(f"\nTheoretical Portfolio Volatility (from covariance matrix):")
    print(f"  {theo_vol:.4f} ({theo_vol*100:.2f}%)")

    print(f"\nActual Portfolio Volatility (observed from daily returns):")
    print(f"  {combined_metrics['vol']:.4f} ({combined_metrics['vol']*100:.2f}%)")

    print(f"\nWeighted Average of Individual Volatilities:")
    weighted_vol = sum(alloc[b] * individual_metrics[b]["vol"] for b in books)
    print(f"  {weighted_vol:.4f} ({weighted_vol*100:.2f}%)")

    print(f"\nVolatility Reduction from Diversification:")
    reduction = weighted_vol - combined_metrics['vol']
    reduction_pct = (reduction / weighted_vol) * 100 if weighted_vol > 0 else 0
    print(f"  {reduction:.4f} ({reduction_pct:.2f}% reduction)")

    if reduction > 0:
        print(f"  [OK] Diversification DID reduce volatility as expected")
    else:
        print(f"  [FAIL] Diversification FAILED to reduce volatility (negative correlation??)")

    # ========================================================================
    # PART 5: Why Sharpe Decreased
    # ========================================================================
    print("\n" + "="*100)
    print("PART 5: Why Did Sharpe Decrease Despite Diversification?")
    print("="*100)

    print(f"\nSharpe = (Return - Rf) / Volatility")
    print(f"\nBest Individual (Book {best_individual[0]}):")
    print(f"  Return:     {best_individual[1]['annual_return']:.2%}")
    print(f"  Volatility: {best_individual[1]['vol']:.2%}")
    print(f"  Sharpe:     {best_individual[1]['sharpe']:.3f}")

    print(f"\nBest Combined (30A+20C+40D+10E):")
    print(f"  Return:     {combined_metrics['annual_return']:.2%}")
    print(f"  Volatility: {combined_metrics['vol']:.2%}")
    print(f"  Sharpe:     {combined_metrics['sharpe']:.3f}")

    print(f"\nThe Issue:")
    print(f"  - Return DECREASED by {combined_metrics['annual_return'] - best_individual[1]['annual_return']:+.2%}")
    print(f"  - Volatility DECREASED by {combined_metrics['vol'] - best_individual[1]['vol']:+.2%}")
    print(f"  - But return decrease (-{best_individual[1]['annual_return'] - combined_metrics['annual_return']:.2%}) is")
    print(f"    LARGER than volatility decrease (-{best_individual[1]['vol'] - combined_metrics['vol']:.2%})")
    print(f"  - So Sharpe decreased")

    print(f"\nWhy did return decrease?")
    return_impact = sum(alloc[b] * (individual_metrics[b]["annual_return"] - best_individual[1]['annual_return']) for b in books)
    print(f"  - Because we're blending {best_individual[0]} (highest alpha)")
    print(f"    with lower-alpha strategies A, C, E")
    print(f"  - Expected blended return from allocation: {return_impact + best_individual[1]['annual_return']:.2%}")

    print(f"\nConclusion:")
    print(f"  When combining strategies, you get:")
    print(f"  ??Lower volatility (from diversification)")
    print(f"  ??Lower returns (from diluting best strategy)")
    print(f"  => If volatility reduction < return reduction, Sharpe falls")

    # ========================================================================
    # PART 6: When Does Diversification Help?
    # ========================================================================
    print("\n" + "="*100)
    print("PART 6: When Does Diversification Actually Help Sharpe?")
    print("="*100)

    print(f"""
Diversification helps Sharpe when:

  1. Strategies are NEGATIVELY correlated
     - Adding a low-alpha hedge reduces overall volatility significantly
     - Example: Growth + Defensive strategies

  2. All strategies have SIMILAR alpha but different sources
     - Combining different signals reduces idiosyncratic risk
     - Example: Multiple momentum indicators

  3. Volatility reduction > Return dilution
     - Mathematical: (A+B Ret) / (A+B Vol) > A Ret / A Vol

Current Situation (2024-2026):
  - Book D dominates on Sharpe (2.665)
  - Other strategies are lower quality (Sharpe < D)
  - Adding lower-quality strategies DILUTES the portfolio
  - Result: Lower Sharpe than pure Book D

Better Approach:
  - If D is clearly best ??Use D only
  - If D+others are similar quality ??Diversify
  - If we want defensive hedge ??Add uncorrelated hedge
""")

    # ========================================================================
    # PART 7: Recommendations
    # ========================================================================
    print("="*100)
    print("PART 7: Recommendations Based on This Analysis")
    print("="*100)

    print(f"""
Option 1: PURE STRATEGY (Maximum Sharpe)
  - Deploy: 100% Book D
  - Sharpe: {best_individual[1]['sharpe']:.3f}
  - Ann Return: {best_individual[1]['annual_return']:.2%}
  - Risk: Concentrated, single-strategy risk
  - Best if: You trust Book D and want max Sharpe

Option 2: DIVERSIFIED (Lower Risk, Slightly Lower Sharpe)
  - Deploy: 30% A + 20% C + 40% D + 10% E
  - Sharpe: {combined_metrics['sharpe']:.3f}
  - Ann Return: {combined_metrics['annual_return']:.2%}
  - Risk: Lower individual strategy risk
  - Best if: You want hedge against D underperforming

Option 3: QUALITY FILTER (Recommended)
  - Only combine strategies with SIMILAR quality (Sharpe > 1.5)
  - Current: Only B (1.660) and D (2.665) qualify
  - Deploy: Blend B + D only, exclude lower-Sharpe A/C/E
  - Expected: Sharpe ~{0.5*individual_metrics['B']['sharpe'] + 0.5*individual_metrics['D']['sharpe']:.3f}
""")


if __name__ == "__main__":
    main()

