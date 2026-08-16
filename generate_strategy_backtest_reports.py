"""
Generate Complete Strategy Backtest Reports (2019-2026)
With: Daily Returns, Yearly Sharpe, Yearly Returns, Yearly Max Drawdown
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

print("=" * 120)
print("STRATEGY BACKTEST REPORT GENERATOR (2019-2026)")
print("=" * 120)

# Load daily returns
df_daily = pd.read_csv('results/portfolio_5book_daily_2019_2026_reconstructed.csv',
                        index_col='Date', parse_dates=True)

print(f"\nLoaded daily returns: {df_daily.shape[0]} trading days")
print(f"Strategies: {', '.join(df_daily.columns)}")

strategy_info = {
    'A': {
        'name': 'Daily Momentum + Leverage + UVXY',
        'period': '1997-2026 (30 years)',
        'sharpe': 1.41,
        'annual': 85,
        'maxdd': -65,
        'status': 'EXCELLENT'
    },
    'B': {
        'name': 'QQQ Bubble Hourly Momentum',
        'period': '2020-2026 (5.85 years)',
        'sharpe': 1.66,
        'annual': 17.19,
        'maxdd': -16.25,
        'status': 'EXCELLENT'
    },
    'C': {
        'name': 'Intraday Mean Reversion + Momentum Flip',
        'period': '2019-2026 (7.4 years)',
        'sharpe': 0.98,
        'annual': 26.8,
        'maxdd': -20.8,
        'status': 'WEAK'
    },
    'D': {
        'name': 'Contrarian Bubble',
        'period': '2019-2026 (7.4 years)',
        'sharpe': 2.65,
        'annual': 38.08,
        'maxdd': -10.09,
        'status': 'EXCELLENT'
    },
    'E': {
        'name': 'Reddit Sentiment Long',
        'period': '2024-2026 (2.2 years)',
        'sharpe': 0.509,
        'annual': 8.5,
        'maxdd': -22.3,
        'status': 'EXPERIMENTAL'
    }
}

os.makedirs('results', exist_ok=True)

# =============================================================================
# Generate reports for each strategy
# =============================================================================

for strategy in df_daily.columns:
    print(f"\n{'=' * 120}")
    print(f"STRATEGY {strategy}: {strategy_info[strategy]['name']}")
    print(f"{'=' * 120}")

    ret_series = df_daily[strategy].dropna()

    if len(ret_series) == 0:
        print(f"  No data available")
        continue

    # =========================================================================
    # Overall Metrics
    # =========================================================================

    wealth = (1 + ret_series).cumprod()
    total_return = wealth.iloc[-1] / wealth.iloc[0] - 1
    years = len(ret_series) / 252
    annual_return = (wealth.iloc[-1] / wealth.iloc[0]) ** (1 / years) - 1
    sharpe = (ret_series.mean() / ret_series.std() * np.sqrt(252)) if ret_series.std() > 0 else 0
    sortino = (ret_series.mean() / ret_series[ret_series < 0].std() * np.sqrt(252)) if ret_series[ret_series < 0].std() > 0 else 0
    maxdd = ((wealth / wealth.cummax()) - 1).min()
    win_rate = (ret_series > 0).sum() / len(ret_series)

    print(f"\nOVERALL METRICS (2019-2026):")
    print(f"  Trading Days: {len(ret_series):,}")
    print(f"  Total Return: {total_return:+.2%}")
    print(f"  Annual Return: {annual_return:+.2%}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Sortino Ratio: {sortino:.2f}")
    print(f"  Max Drawdown: {maxdd:.2%}")
    print(f"  Win Rate: {win_rate:.1%}")

    # =========================================================================
    # Yearly Breakdown
    # =========================================================================

    print(f"\nYEARLY BREAKDOWN:")
    print(f"{'Year':<8} {'Days':<8} {'Return':<12} {'Sharpe':<10} {'Sortino':<10} {'Max DD':<10} {'Win Rate':<10}")
    print("-" * 80)

    yearly_results = []

    for year in range(2019, 2027):
        year_mask = ret_series.index.year == year
        year_ret = ret_series[year_mask]

        if len(year_ret) == 0:
            continue

        year_wealth = (1 + year_ret).cumprod()
        year_total = year_wealth.iloc[-1] / year_wealth.iloc[0] - 1
        year_sharpe = (year_ret.mean() / year_ret.std() * np.sqrt(252)) if year_ret.std() > 0 else 0
        year_sortino = (year_ret.mean() / year_ret[year_ret < 0].std() * np.sqrt(252)) if year_ret[year_ret < 0].std() > 0 else 0
        year_maxdd = ((year_wealth / year_wealth.cummax()) - 1).min()
        year_win_rate = (year_ret > 0).sum() / len(year_ret)

        print(f"{year:<8} {len(year_ret):<8} {year_total:+9.2%}    {year_sharpe:6.2f}     {year_sortino:6.2f}     {year_maxdd:7.2%}    {year_win_rate:7.1%}")

        yearly_results.append({
            'Year': year,
            'Days': len(year_ret),
            'Total Return': year_total,
            'Annual Return': year_total,  # For single year, total = annual
            'Sharpe': year_sharpe,
            'Sortino': year_sortino,
            'Max Drawdown': year_maxdd,
            'Win Rate': year_win_rate,
        })

    # =========================================================================
    # Save to CSV
    # =========================================================================

    # Daily returns
    ret_df = pd.DataFrame({f'{strategy}_Daily_Return': ret_series})
    ret_df.to_csv(f'results/{strategy}_daily_returns_2019_2026.csv')

    # Yearly metrics
    yearly_df = pd.DataFrame(yearly_results)
    yearly_df.to_csv(f'results/{strategy}_yearly_metrics_2019_2026.csv', index=False)

    print(f"\n  [Saved] {strategy}_daily_returns_2019_2026.csv")
    print(f"  [Saved] {strategy}_yearly_metrics_2019_2026.csv")

# =============================================================================
# Summary Table
# =============================================================================

print(f"\n{'=' * 120}")
print("SUMMARY: ALL STRATEGIES 2019-2026")
print(f"{'=' * 120}\n")

summary_rows = []

for strategy in df_daily.columns:
    ret_series = df_daily[strategy].dropna()

    if len(ret_series) == 0:
        continue

    wealth = (1 + ret_series).cumprod()
    total_return = wealth.iloc[-1] / wealth.iloc[0] - 1
    years = len(ret_series) / 252
    annual_return = (wealth.iloc[-1] / wealth.iloc[0]) ** (1 / years) - 1
    sharpe = (ret_series.mean() / ret_series.std() * np.sqrt(252)) if ret_series.std() > 0 else 0
    maxdd = ((wealth / wealth.cummax()) - 1).min()
    win_rate = (ret_series > 0).sum() / len(ret_series)

    summary_rows.append({
        'Strategy': f"{strategy}: {strategy_info[strategy]['name'][:40]}",
        'Days': len(ret_series),
        'Total Return %': total_return * 100,
        'Annual %': annual_return * 100,
        'Sharpe': sharpe,
        'Max DD %': maxdd * 100,
        'Win Rate %': win_rate * 100,
    })

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values('Sharpe', ascending=False)

for idx, row in summary_df.iterrows():
    print(f"{row['Strategy']}")
    print(f"  Total: {row['Total Return %']:+.1f}% | Annual: {row['Annual %']:+.1f}% | "
          f"Sharpe: {row['Sharpe']:6.2f} | MaxDD: {row['Max DD %']:6.1f}% | Win: {row['Win Rate %']:5.1f}%\n")

summary_df.to_csv('results/ALL_STRATEGIES_SUMMARY_2019_2026.csv', index=False)
print(f"\n[Saved] ALL_STRATEGIES_SUMMARY_2019_2026.csv")

# =============================================================================
# Comparison with Documented Performance
# =============================================================================

print(f"\n{'=' * 120}")
print("COMPARISON: DOCUMENTED VS ACTUAL (2019-2026)")
print(f"{'=' * 120}\n")

comparison = []

for strategy in df_daily.columns:
    ret_series = df_daily[strategy].dropna()

    if len(ret_series) == 0:
        continue

    wealth = (1 + ret_series).cumprod()
    years = len(ret_series) / 252
    annual_return = (wealth.iloc[-1] / wealth.iloc[0]) ** (1 / years) - 1
    sharpe = (ret_series.mean() / ret_series.std() * np.sqrt(252)) if ret_series.std() > 0 else 0
    maxdd = ((wealth / wealth.cummax()) - 1).min()

    doc_sharpe = strategy_info[strategy]['sharpe']
    doc_annual = strategy_info[strategy]['annual']

    comparison.append({
        'Strategy': strategy,
        'Documented Sharpe': doc_sharpe,
        'Actual Sharpe': sharpe,
        'Sharpe Diff': sharpe - doc_sharpe,
        'Documented Annual %': doc_annual,
        'Actual Annual %': annual_return * 100,
        'Annual Diff %': (annual_return * 100) - doc_annual,
    })

comp_df = pd.DataFrame(comparison)

print(f"{'Strat':<8} {'Doc Sharpe':<12} {'Act Sharpe':<12} {'Diff':<10} "
      f"{'Doc Ann %':<12} {'Act Ann %':<12} {'Diff %':<10}")
print("-" * 90)

for idx, row in comp_df.iterrows():
    print(f"{row['Strategy']:<8} {row['Documented Sharpe']:<12.2f} {row['Actual Sharpe']:<12.2f} "
          f"{row['Sharpe Diff']:+.2f}     {row['Documented Annual %']:<12.1f} "
          f"{row['Actual Annual %']:<12.1f} {row['Annual Diff %']:+.1f}")

comp_df.to_csv('results/COMPARISON_DOCUMENTED_VS_ACTUAL.csv', index=False)
print(f"\n[Saved] COMPARISON_DOCUMENTED_VS_ACTUAL.csv")

print(f"\n{'=' * 120}")
print("BACKTEST REPORT GENERATION COMPLETE")
print(f"{'=' * 120}")

print("\nGenerated Files:")
for strategy in df_daily.columns:
    print(f"  - {strategy}_daily_returns_2019_2026.csv")
    print(f"  - {strategy}_yearly_metrics_2019_2026.csv")
print(f"  - ALL_STRATEGIES_SUMMARY_2019_2026.csv")
print(f"  - COMPARISON_DOCUMENTED_VS_ACTUAL.csv")

