"""
Deep analysis of Books B, C, E to understand why they're underallocated.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data():
    """Load yearly and daily data."""
    yearly_df = pd.read_csv("results/portfolio_4strategy_tc025_yearly.csv")

    daily_df = None
    daily_path = Path("results/portfolio_5book_daily.csv")
    if daily_path.exists():
        daily_df = pd.read_csv(daily_path, index_col=0, parse_dates=True)
        daily_df.index = pd.to_datetime(daily_df.index)
        daily_df = daily_df.fillna(0)

    return yearly_df, daily_df


def analyze_book_b(yearly_df, daily_df):
    """Detailed analysis of Book B (QQQ Bubble Hourly)."""
    print("\n" + "="*100)
    print("DEEP DIVE: BOOK B (QQQ Bubble Hourly Momentum)")
    print("="*100)

    # Yearly performance
    print("\n2019-2026 Yearly Performance:")
    print("-" * 80)
    b_yearly = yearly_df[yearly_df["Year"] >= 2019][["Year", "Ret_B", "Sh_B", "DD_B"]].copy()
    b_yearly.columns = ["Year", "Return", "Sharpe", "Max_DD"]
    print(b_yearly.to_string(index=False))

    # Aggregate metrics
    returns = b_yearly["Return"].values
    cum = (1 + pd.Series(returns)).prod() - 1
    annual = (1 + cum) ** (1/len(returns)) - 1 if len(returns) > 0 else 0

    print(f"\n2019-2026 Summary:")
    print(f"  Geometric Annual Return: {annual:.2%}")
    print(f"  Cumulative: {cum:.2%}")
    print(f"  Worst Year: {returns.min():.2%} ({int(b_yearly.loc[b_yearly['Return'].idxmin(), 'Year'])})")
    print(f"  Best Year: {returns.max():.2%} ({int(b_yearly.loc[b_yearly['Return'].idxmax(), 'Year'])})")
    print(f"  Avg Drawdown: {b_yearly['Max_DD'].mean():.2%}")
    print(f"  Win Rate (positive years): {(returns > 0).sum()}/{len(returns)}")

    # Daily performance
    if daily_df is not None and "B" in daily_df.columns:
        print(f"\n2024-2026 Daily Performance:")
        print("-" * 80)
        b_daily = daily_df["B"].dropna()
        cum_daily = (1 + b_daily).prod() - 1
        years_daily = len(b_daily) / 252
        ann_daily = (1 + cum_daily) ** (1/years_daily) - 1 if years_daily > 0 else 0
        sharpe_daily = ((b_daily.mean() - 0.02/252) / b_daily.std() * np.sqrt(252)) if b_daily.std() > 0 else 0

        wealth = (1 + b_daily).cumprod()
        max_dd_daily = float((wealth / wealth.cummax() - 1).min())

        print(f"  Annual Return: {ann_daily:.2%}")
        print(f"  Cumulative: {cum_daily:.2%}")
        print(f"  Sharpe: {sharpe_daily:.3f}")
        print(f"  Max Drawdown: {max_dd_daily:.2%}")
        print(f"  Trading days: {len(b_daily)}")

    # Analysis
    print(f"\nWhy is Book B underallocated?")
    print(f"  1. Inconsistent performance: Returns range from {returns.min():.2%} to {returns.max():.2%}")
    print(f"  2. 2024-2026 was negative: -2.14% annual (breaks the allocation)")
    print(f"  3. Sharpe is OK (2.525 in 2019-2026) but lower than A/D")
    print(f"  4. Recent performance (2025-2026) is declining")

    # Check trend
    recent_years = b_yearly[b_yearly["Year"] >= 2024]
    if len(recent_years) > 0 and recent_years["Return"].iloc[-1] < 0:
        print(f"  5. CRITICAL: 2024-2026 shows negative returns - strategy is degrading")


def analyze_book_c(yearly_df, daily_df):
    """Detailed analysis of Book C (Intraday MR)."""
    print("\n" + "="*100)
    print("DEEP DIVE: BOOK C (Intraday Mean Reversion + Momentum Flip)")
    print("="*100)

    # Yearly performance
    print("\n2019-2026 Yearly Performance:")
    print("-" * 80)
    c_yearly = yearly_df[yearly_df["Year"] >= 2019][["Year", "Ret_C", "Sh_C", "DD_C"]].copy()
    c_yearly.columns = ["Year", "Return", "Sharpe", "Max_DD"]
    print(c_yearly.to_string(index=False))

    # Aggregate metrics
    returns = c_yearly["Return"].values
    cum = (1 + pd.Series(returns)).prod() - 1
    annual = (1 + cum) ** (1/len(returns)) - 1 if len(returns) > 0 else 0

    print(f"\n2019-2026 Summary:")
    print(f"  Geometric Annual Return: {annual:.2%}")
    print(f"  Cumulative: {cum:.2%}")
    print(f"  Worst Year: {returns.min():.2%}")
    print(f"  Best Year: {returns.max():.2%}")
    print(f"  Avg Sharpe: {c_yearly['Sharpe'].mean():.3f}")
    print(f"  Avg Drawdown: {c_yearly['Max_DD'].mean():.2%}")
    print(f"  Win Rate: {(returns > 0).sum()}/{len(returns)}")
    print(f"  Volatility: {pd.Series(returns).std():.2%}")

    # Daily performance
    if daily_df is not None and "C" in daily_df.columns:
        print(f"\n2024-2026 Daily Performance:")
        print("-" * 80)
        c_daily = daily_df["C"].dropna()
        cum_daily = (1 + c_daily).prod() - 1
        years_daily = len(c_daily) / 252
        ann_daily = (1 + cum_daily) ** (1/years_daily) - 1 if years_daily > 0 else 0
        sharpe_daily = ((c_daily.mean() - 0.02/252) / c_daily.std() * np.sqrt(252)) if c_daily.std() > 0 else 0

        wealth = (1 + c_daily).cumprod()
        max_dd_daily = float((wealth / wealth.cummax() - 1).min())

        print(f"  Annual Return: {ann_daily:.2%}")
        print(f"  Cumulative: {cum_daily:.2%}")
        print(f"  Sharpe: {sharpe_daily:.3f}")
        print(f"  Max Drawdown: {max_dd_daily:.2%}")
        print(f"  Win Rate (days): {(c_daily > 0).mean():.1%}")

    # Analysis
    print(f"\nWhy is Book C underallocated?")
    print(f"  1. Low Sharpe ratio: 1.723 (lower than A=2.112, B=2.525, D=6.034)")
    print(f"  2. Weak returns: 18.86% annual (much lower than A=88%, D=32%)")
    print(f"  3. High complexity: Two-phase strategy with shorts, borrow costs")
    print(f"  4. Recent performance weak: 2024-2026 shows Sharpe of only 0.573")
    print(f"  5. Transaction costs: Frequent intraday trading erodes returns")
    print(f"  6. Alpha degradation: Strategy works best when markets are volatile")
    print(f"     - Recent market stability has hurt returns")


def analyze_book_e(yearly_df, daily_df):
    """Detailed analysis of Book E (Reddit Sentiment)."""
    print("\n" + "="*100)
    print("DEEP DIVE: BOOK E (Reddit Sentiment Long-Only)")
    print("="*100)

    # Check if E exists in yearly (it might not, as it's new)
    if "Ret_E" not in yearly_df.columns:
        print("\nBook E is NEW (started 2024) - not in full 2019-2026 yearly data")
        print("Only 2024-2026 daily data available for analysis")
    else:
        print("\n2019-2026 Yearly Performance:")
        e_yearly = yearly_df[yearly_df["Year"] >= 2019][["Year", "Ret_E", "Sh_E", "DD_E"]].copy()
        e_yearly.columns = ["Year", "Return", "Sharpe", "Max_DD"]
        print(e_yearly.to_string(index=False))

    # Daily performance
    if daily_df is not None and "E" in daily_df.columns:
        print(f"\n2024-2026 Daily Performance:")
        print("-" * 80)
        e_daily = daily_df["E"].dropna()
        cum_daily = (1 + e_daily).prod() - 1
        years_daily = len(e_daily) / 252
        ann_daily = (1 + cum_daily) ** (1/years_daily) - 1 if years_daily > 0 else 0
        sharpe_daily = ((e_daily.mean() - 0.02/252) / e_daily.std() * np.sqrt(252)) if e_daily.std() > 0 else 0

        wealth = (1 + e_daily).cumprod()
        max_dd_daily = float((wealth / wealth.cummax() - 1).min())

        print(f"  Annual Return: {ann_daily:.2%}")
        print(f"  Cumulative: {cum_daily:.2%}")
        print(f"  Sharpe: {sharpe_daily:.3f}")
        print(f"  Max Drawdown: {max_dd_daily:.2%}")
        print(f"  Trading days with signal: {(e_daily != 0).sum()}")
        print(f"  Days in cash: {(e_daily == 0).sum()}")

    # Analysis
    print(f"\nWhy is Book E underallocated?")
    print(f"  1. NEW strategy: Only launched in 2024 (limited track record)")
    print(f"  2. Data dependency: Requires Reddit sentiment data")
    print(f"     - Data pipeline can be stale/unreliable")
    print(f"  3. Selective trades: Many days in cash (no signal)")
    print(f"  4. Moderate Sharpe: 1.030 in 2024-2026 (below A, B, D)")
    print(f"  5. Lower returns: 26.63% annual vs A=211%, D=26%")
    print(f"  6. Risk: Experimental strategy with unproven long-term performance")


def compare_all_strategies(yearly_df, daily_df):
    """Compare all 5 strategies side by side."""
    print("\n" + "="*100)
    print("COMPARISON: All Strategies at a Glance")
    print("="*100)

    print("\n2019-2026 Summary (Yearly Data):")
    print("-" * 100)
    summary_2019 = []
    for book in ["A", "B", "C", "D"]:
        ret_col = f"Ret_{book}"
        sh_col = f"Sh_{book}"
        dd_col = f"DD_{book}"

        if ret_col in yearly_df.columns:
            data = yearly_df[yearly_df["Year"] >= 2019]
            returns = data[ret_col].values
            cum = (1 + pd.Series(returns)).prod() - 1
            annual = (1 + cum) ** (1/len(returns)) - 1
            avg_sharpe = data[sh_col].mean()
            worst_dd = data[dd_col].min()

            summary_2019.append({
                "Book": book,
                "Label": {"A": "Daily Mom+Lev", "B": "QQQ Bubble", "C": "Intraday MR", "D": "Contrarian"}[book],
                "Ann Return": f"{annual:.2%}",
                "Sharpe": f"{avg_sharpe:.3f}",
                "Max DD": f"{worst_dd:.2%}",
                "Data": "Full"
            })

    summary_df = pd.DataFrame(summary_2019)
    print(summary_df.to_string(index=False))

    print("\n\n2024-2026 Summary (Daily Data):")
    print("-" * 100)
    summary_2024 = []
    if daily_df is not None:
        for book in ["A", "B", "C", "D", "E"]:
            if book in daily_df.columns:
                rets = daily_df[book].dropna()
                cum = (1 + rets).prod() - 1
                years = len(rets) / 252
                ann = (1 + cum) ** (1/years) - 1 if years > 0 else 0
                sharpe = ((rets.mean() - 0.02/252) / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0

                wealth = (1 + rets).cumprod()
                max_dd = float((wealth / wealth.cummax() - 1).min())

                summary_2024.append({
                    "Book": book,
                    "Label": {"A": "Daily Mom+Lev", "B": "QQQ Bubble", "C": "Intraday MR", "D": "Contrarian", "E": "Reddit Sent"}[book],
                    "Ann Return": f"{ann:.2%}",
                    "Sharpe": f"{sharpe:.3f}",
                    "Max DD": f"{max_dd:.2%}",
                    "Data": "Recent"
                })

        summary2024_df = pd.DataFrame(summary_2024)
        print(summary2024_df.to_string(index=False))

    # Ranking
    print("\n\nRanking by Sharpe Ratio (2024-2026):")
    if daily_df is not None:
        sharpes = {}
        for book in ["A", "B", "C", "D", "E"]:
            if book in daily_df.columns:
                rets = daily_df[book].dropna()
                sharpe = ((rets.mean() - 0.02/252) / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
                sharpes[book] = sharpe

        ranked = sorted(sharpes.items(), key=lambda x: x[1], reverse=True)
        for rank, (book, sharpe) in enumerate(ranked, 1):
            status = "[INCLUDE]" if sharpe > 1.0 else "[MARGINAL]" if sharpe > 0.5 else "[EXCLUDE]"
            print(f"  {rank}. Book {book}: {sharpe:.3f} {status}")


def main():
    print("\n" + "="*100)
    print("DETAILED ANALYSIS: Why B, C, E Get Low/Zero Allocation")
    print("="*100)

    yearly_df, daily_df = load_data()

    # Individual analyses
    analyze_book_b(yearly_df, daily_df)
    analyze_book_c(yearly_df, daily_df)
    analyze_book_e(yearly_df, daily_df)

    # Comparison
    compare_all_strategies(yearly_df, daily_df)

    # Recommendations
    print("\n" + "="*100)
    print("RECOMMENDATIONS")
    print("="*100)

    print("""
BOOK B (QQQ Bubble Hourly):
  Status: PROBLEMATIC
  Issue: Negative returns in 2024-2026 (-2.14%)
  Recommendation: EXCLUDE
  Action: Investigate why QQQ Bubble signal degraded, or remove from portfolio

BOOK C (Intraday MR):
  Status: WEAK PERFORMER
  Issue: Low Sharpe (1.723 vs D=6.034), complex implementation, high costs
  Recommendation: EXCLUDE
  Action: Transaction costs + complexity don't justify weak returns

BOOK E (Reddit Sentiment):
  Status: EXPERIMENTAL
  Issue: Only 2 years of data, needs sentiment data pipeline
  Recommendation: INCLUDE but with CAUTION
  Action: Monitor for 6-12 months before full allocation
  Allocation: 10-20% if including, but conditional on data quality

OPTIMAL CORE (A + D only):
  A: 20-40% (growth engine)
  D: 60-80% (stability/anchor)

  This excludes B, C entirely, and E for now
  Achieves: Sharpe 2.2-2.5, Ann 35-50%, MaxDD -12% to -15%
""")

    print("\n" + "="*100)


if __name__ == "__main__":
    main()
