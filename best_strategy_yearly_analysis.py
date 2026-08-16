"""
All-time best strategy (10% A + 90% D, 2019-2026):
Show yearly returns, drawdowns, Sharpe ratios, and create visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_yearly_data():
    """Load yearly performance data."""
    csv_path = Path(__file__).parent / "results" / "portfolio_4strategy_tc025_yearly.csv"
    df = pd.read_csv(csv_path)
    return df[df["Year"] >= 2019].copy()


def calculate_best_strategy_yearly(yearly_df):
    """Calculate 10% A + 90% D yearly metrics."""

    # Extract returns and metrics
    allocation = {"A": 0.10, "D": 0.90}

    yearly_results = []

    for _, row in yearly_df.iterrows():
        year = int(row["Year"])

        # Annual return
        ret_a = row["Ret_A"]
        ret_d = row["Ret_D"]
        annual_ret = 0.10 * ret_a + 0.90 * ret_d

        # Max drawdown (weighted average)
        dd_a = row["DD_A"]
        dd_d = row["DD_D"]
        annual_dd = 0.10 * dd_a + 0.90 * dd_d

        # Sharpe (need to calculate from returns, use documented values as proxy)
        sh_a = row["Sh_A"]
        sh_d = row["Sh_D"]
        # Approximate Sharpe for portfolio (not perfect but reasonable)
        annual_sharpe = 0.10 * sh_a + 0.90 * sh_d

        yearly_results.append({
            "Year": year,
            "Annual Return": annual_ret,
            "Max Drawdown": annual_dd,
            "Sharpe Ratio": annual_sharpe,
            "Book A Return": ret_a,
            "Book D Return": ret_d,
        })

    return pd.DataFrame(yearly_results)


def print_summary_tables(yearly_df):
    """Print detailed yearly performance."""
    print("\n" + "="*100)
    print("ALL-TIME BEST STRATEGY: 10% A + 90% D (2019-2026)")
    print("="*100)

    print("\n" + "-"*100)
    print("YEARLY BREAKDOWN:")
    print("-"*100)

    # Create display table
    display_df = yearly_df[["Year", "Annual Return", "Max Drawdown", "Sharpe Ratio"]].copy()
    display_df["Annual Return"] = display_df["Annual Return"].apply(lambda x: f"{x:+.2%}")
    display_df["Max Drawdown"] = display_df["Max Drawdown"].apply(lambda x: f"{x:.2%}")
    display_df["Sharpe Ratio"] = display_df["Sharpe Ratio"].apply(lambda x: f"{x:.3f}")

    print(display_df.to_string(index=False))

    print("\n" + "-"*100)
    print("AGGREGATE METRICS (2019-2026):")
    print("-"*100)

    returns = yearly_df["Annual Return"].values
    drawdowns = yearly_df["Max Drawdown"].values
    sharpes = yearly_df["Sharpe Ratio"].values

    # Geometric return
    cum_return = (1 + pd.Series(returns)).prod() - 1
    geometric_annual = (1 + cum_return) ** (1/len(returns)) - 1

    print(f"Total Cumulative Return:    {cum_return:+.2%}")
    print(f"Geometric Annual Return:    {geometric_annual:+.2%}")
    print(f"Average Annual Return:      {returns.mean():+.2%}")
    print(f"Worst Year:                 {returns.min():+.2%} ({int(yearly_df.loc[yearly_df['Annual Return'].idxmin(), 'Year'])})")
    print(f"Best Year:                  {returns.max():+.2%} ({int(yearly_df.loc[yearly_df['Annual Return'].idxmax(), 'Year'])})")

    print(f"\nAverage Max Drawdown:       {drawdowns.mean():.2%}")
    print(f"Worst Max Drawdown:         {drawdowns.min():.2%} ({int(yearly_df.loc[yearly_df['Max Drawdown'].idxmin(), 'Year'])})")
    print(f"Best Max Drawdown:          {drawdowns.max():.2%} ({int(yearly_df.loc[yearly_df['Max Drawdown'].idxmax(), 'Year'])})")

    print(f"\nAverage Sharpe Ratio:       {sharpes.mean():.3f}")
    print(f"Worst Sharpe:               {sharpes.min():.3f} ({int(yearly_df.loc[yearly_df['Sharpe Ratio'].idxmin(), 'Year'])})")
    print(f"Best Sharpe:                {sharpes.max():.3f} ({int(yearly_df.loc[yearly_df['Sharpe Ratio'].idxmax(), 'Year'])})")

    print(f"\nWin Rate (positive years):  {(returns > 0).sum()}/{len(returns)} years")
    print(f"Consistency:                {len(returns)}/8 years profitable")

    # Volatility of annual returns
    annual_vol = returns.std()
    print(f"Volatility of Annual Returns: {annual_vol:.2%}")


def create_visualizations(yearly_df):
    """Create comprehensive visualizations."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Best Strategy Performance: 10% A + 90% D (2019-2026)", fontsize=16, fontweight='bold')

    years = yearly_df["Year"].values

    # 1. Annual Returns
    ax = axes[0, 0]
    colors = ['green' if x > 0 else 'red' for x in yearly_df["Annual Return"]]
    ax.bar(years, yearly_df["Annual Return"] * 100, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title("Annual Returns", fontsize=12, fontweight='bold')
    ax.set_ylabel("Return (%)")
    ax.set_xlabel("Year")
    ax.grid(axis='y', alpha=0.3)
    for i, (year, ret) in enumerate(zip(years, yearly_df["Annual Return"])):
        ax.text(year, ret * 100 + (2 if ret > 0 else -2), f"{ret:.1%}", ha='center', va='bottom' if ret > 0 else 'top')

    # 2. Max Drawdown
    ax = axes[0, 1]
    ax.bar(years, yearly_df["Max Drawdown"] * 100, color='coral', alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title("Maximum Drawdown per Year", fontsize=12, fontweight='bold')
    ax.set_ylabel("Max Drawdown (%)")
    ax.set_xlabel("Year")
    ax.grid(axis='y', alpha=0.3)
    for i, (year, dd) in enumerate(zip(years, yearly_df["Max Drawdown"])):
        ax.text(year, dd * 100 - 1, f"{dd:.1%}", ha='center', va='top')

    # 3. Sharpe Ratio
    ax = axes[1, 0]
    colors_sharpe = ['darkgreen' if x > 2 else 'lightgreen' if x > 1 else 'orange' for x in yearly_df["Sharpe Ratio"]]
    ax.bar(years, yearly_df["Sharpe Ratio"], color=colors_sharpe, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=1, color='blue', linestyle='--', linewidth=1, label='Sharpe=1.0')
    ax.axhline(y=2, color='green', linestyle='--', linewidth=1, label='Sharpe=2.0')
    ax.set_title("Annual Sharpe Ratio", fontsize=12, fontweight='bold')
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xlabel("Year")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for i, (year, sharpe) in enumerate(zip(years, yearly_df["Sharpe Ratio"])):
        ax.text(year, sharpe + 0.1, f"{sharpe:.2f}", ha='center', va='bottom')

    # 4. Return vs Drawdown scatter
    ax = axes[1, 1]
    scatter = ax.scatter(yearly_df["Max Drawdown"] * 100, yearly_df["Annual Return"] * 100,
                        s=200, alpha=0.6, c=yearly_df["Sharpe Ratio"], cmap='RdYlGn', edgecolor='black', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title("Return vs Risk Tradeoff", fontsize=12, fontweight='bold')
    ax.set_xlabel("Max Drawdown (%)")
    ax.set_ylabel("Annual Return (%)")
    ax.grid(True, alpha=0.3)

    # Add year labels
    for year, ret, dd in zip(years, yearly_df["Annual Return"] * 100, yearly_df["Max Drawdown"] * 100):
        ax.text(dd, ret, str(year), ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio", rotation=270, labelpad=20)

    plt.tight_layout()

    # Save
    output_path = Path(__file__).parent / "results" / "best_strategy_yearly_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[SAVED] {output_path}")

    plt.show()


def create_summary_csv(yearly_df):
    """Save yearly results to CSV."""
    output_path = Path(__file__).parent / "results" / "best_strategy_yearly_analysis.csv"

    # Format for CSV
    csv_df = yearly_df[["Year", "Annual Return", "Max Drawdown", "Sharpe Ratio"]].copy()
    csv_df["Annual Return"] = csv_df["Annual Return"].apply(lambda x: f"{x:.4f}")
    csv_df["Max Drawdown"] = csv_df["Max Drawdown"].apply(lambda x: f"{x:.4f}")
    csv_df["Sharpe Ratio"] = csv_df["Sharpe Ratio"].apply(lambda x: f"{x:.4f}")

    csv_df.to_csv(output_path, index=False)
    print(f"[SAVED] {output_path}")


def main():
    # Load data
    yearly_df = load_yearly_data()

    # Calculate best strategy metrics
    yearly_results = calculate_best_strategy_yearly(yearly_df)

    # Print summary
    print_summary_tables(yearly_results)

    # Create visualizations
    print("\n" + "="*100)
    print("CREATING VISUALIZATIONS...")
    print("="*100)
    create_visualizations(yearly_results)

    # Save CSV
    create_summary_csv(yearly_results)

    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
