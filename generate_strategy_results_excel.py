"""
Generate Individual Strategy Results Excel File
Uses best parameters from strat.md and STRATEGY_PARAMETERS_REFERENCE.md
Saves comprehensive results for each strategy A, B, C, D, E
"""

import pandas as pd
import numpy as np
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

print("=" * 120)
print("GENERATING INDIVIDUAL STRATEGY RESULTS (Based on strat.md Best Parameters)")
print("=" * 120)

# ============================================================================
# STRATEGY DATA FROM strat.md (VERIFIED RESULTS)
# ============================================================================

strategies = {
    'A': {
        'name': 'Daily Momentum + Leverage + UVXY',
        'status': 'PRODUCTION READY',
        'backtest_period': '1997-2026',
        'duration_years': 30,
        'sharpe': 1.41,
        'annual_return': 0.85,
        'max_dd': -0.6528,
        'volatility': None,
        'positive_years': '24/30 (80%)',
        'win_rate': None,
        'parameters': {
            'Momentum Lookback': '140 days',
            'Rebalance Frequency': '40 days',
            'Long Positions': 'Top 5',
            'Short Positions': 'Bottom 5',
            'Position Size': '10% each',
            'Leverage Multiplier': '1.25x',
            'Leverage Threshold': 'Bubble < -0.88',
            'UVXY Hedge Threshold': 'Bubble > 0.85',
            'Bubble MA Window': '120 days',
            'Universe': '524 stocks (S&P500 + NASDAQ100)',
            'Data Frequency': 'Daily OHLC',
            'Transaction Cost': '0.5% per 40-day cycle',
        },
        'yearly_data': {
            2022: {'return': 0.244, 'sharpe': 0.76, 'max_dd': -0.195},
            2023: {'return': 0.565, 'sharpe': 1.41, 'max_dd': -0.298},
            2024: {'return': 0.741, 'sharpe': 1.38, 'max_dd': -0.202},
            2025: {'return': 1.76, 'sharpe': 1.97, 'max_dd': -0.384},
        },
        'notes': [
            'Tested over 30 years (1997-2026)',
            'Only -1.53% in 2022 bear market',
            '24 positive years out of 30',
            'Leveraged momentum with UVXY hedge',
            'Grid search optimal on long history'
        ]
    },

    'B': {
        'name': 'QQQ Bubble Hourly Momentum',
        'status': 'EXCELLENT',
        'backtest_period': '2020-07-27 to 2026',
        'duration_years': 5.85,
        'sharpe': 1.6603,
        'annual_return': 0.1719,
        'max_dd': -0.1625,
        'volatility': None,
        'positive_years': '6/6 (100%)',
        'win_rate': '84.6%',
        'parameters': {
            'Bubble MA Window': '500h (~77 trading days)',
            'Entry Threshold': '-0.8 (extreme undervaluation)',
            'Momentum Lookback': '40h (~1 week)',
            'Hold Period': '52h (~1.6 weeks)',
            'Top-N Stocks': '5 (highest momentum)',
            'Cash Duration': '~88% of hours',
            'Signal Frequency': '~4-5 trades per year',
            'Total Trades': '26 (4.45 trades/year)',
            'Transaction Cost': '0.1% per trade',
            'Universe': '405 stocks',
            'Data Frequency': 'Hourly OHLC',
        },
        'yearly_data': {
            2021: {'return': 0.2060, 'sharpe': 4.874, 'max_dd': -0.0864},
            2022: {'return': 0.0405, 'sharpe': 0.412, 'max_dd': -0.1112},
            2023: {'return': 0.0914, 'sharpe': 1.031, 'max_dd': -0.0566},
            2024: {'return': 0.1970, 'sharpe': 1.352, 'max_dd': -0.0726},
            2025: {'return': 0.1353, 'sharpe': 1.339, 'max_dd': -0.1171},
            2026: {'return': 0.3587, 'sharpe': 3.070, 'max_dd': -0.1625},
        },
        'notes': [
            '100% positive years (never down)',
            '2022: +4.05% while QQQ -32.39% (36x better!)',
            'Selective trading (~4-5 trades/year)',
            '88% time in cash (low drawdown)',
            'Very high win rate (84.6%)',
            'Excellent bear market hedge'
        ]
    },

    'C': {
        'name': 'Intraday Mean Reversion + Momentum Flip',
        'status': 'WEAK',
        'backtest_period': '2019-2026',
        'duration_years': 7.4,
        'sharpe': 0.98,
        'annual_return': 0.268,
        'max_dd': -0.2081,
        'volatility': None,
        'positive_years': 'Varies',
        'win_rate': None,
        'parameters': {
            'Z-Score Window': '20 days',
            'Entry Signal': 'Z > 4.0 or Z < -4.0',
            'Signal Threshold': '4 sigma (extreme moves)',
            'Top-N Stocks': '5 (highest |Z|)',
            'Phase 1 Hold': '1 hour (mean reversion)',
            'Phase 2 Hold': '3 days (momentum flip)',
            'Position Size': 'Equal-weight across 5',
            'Total Trades': '127 (~17/year)',
            'Transaction Cost': '0.1% per phase (double cost)',
            'Data Frequency': 'Daily + Hourly',
        },
        'yearly_data': {
            2021: {'return': 1.047, 'sharpe': 1.37, 'max_dd': None},
            2022: {'return': 0.306, 'sharpe': 1.16, 'max_dd': None},
            2024: {'return': -0.0140, 'sharpe': 0.07, 'max_dd': -0.2081},
        },
        'notes': [
            'Below 1.0 Sharpe threshold (0.98)',
            'Volatility dependent (only fires when Z > 4.0)',
            'Requires margin account (shorting)',
            'Negative performance in 2024 low-vol',
            'Two phases per trade = double costs',
            'RECOMMENDATION: Exclude from portfolio'
        ]
    },

    'D': {
        'name': 'Contrarian Bubble',
        'status': 'EXCELLENT (GRID SEARCH OPTIMAL)',
        'backtest_period': '2019-2026',
        'duration_years': 7.4,
        'sharpe': 2.6475,
        'annual_return': 0.3808,
        'max_dd': -0.1009,
        'volatility': 0.1437,
        'positive_years': '8/8 (100%)',
        'win_rate': '56.6%',
        'parameters': {
            'MA Window': '104h (~16 trading days)',
            'Entry Threshold': '-0.8 (extreme undervaluation)',
            'Hold Period': '13h (~2 trading sessions)',
            'Top-N Stocks': '20 (most depressed)',
            'Position Size': 'Equal-weight (5% each)',
            'Universe': '515 stocks (S&P500 + NASDAQ100)',
            'Signal Frequency': '~4% of hourly bars',
            'Active Days': '~80% of trading days',
            'Transaction Cost': '0.1% per trade',
            'Data Frequency': 'Hourly OHLC',
            'Grid Search Tested': '1,008 combinations',
            'Grid Positive': '75% (751 combos)',
        },
        'yearly_data': {
            2019: {'return': 0.1691, 'sharpe': 1.989, 'max_dd': -0.0238},
            2020: {'return': 0.2083, 'sharpe': 3.399, 'max_dd': -0.0538},
            2021: {'return': 0.5831, 'sharpe': 4.010, 'max_dd': -0.0265},
            2022: {'return': 0.5613, 'sharpe': 2.400, 'max_dd': -0.1009},
            2023: {'return': 0.3951, 'sharpe': 2.924, 'max_dd': -0.0591},
            2024: {'return': 0.3793, 'sharpe': 3.082, 'max_dd': -0.0492},
            2025: {'return': 0.3519, 'sharpe': 2.339, 'max_dd': -0.0677},
            2026: {'return': 0.2040, 'sharpe': 3.534, 'max_dd': -0.0271},
        },
        'notes': [
            'Highest Sharpe ratio (2.65)',
            '100% positive years (never down)',
            '+56.13% in 2022 bear market',
            'Very low max DD (-10.09%)',
            'Grid search: 75% of combos profitable',
            'Works in all market regimes',
            'Top tier performance',
            'RECOMMENDATION: Core allocation (50%)'
        ]
    },

    'E': {
        'name': 'Reddit Sentiment Long',
        'status': 'EXPERIMENTAL',
        'backtest_period': '2024-2026',
        'duration_years': 2.2,
        'sharpe': 0.509,
        'annual_return': 0.0848,
        'max_dd': -0.2234,
        'volatility': None,
        'positive_years': '2/3 (started 2024)',
        'win_rate': None,
        'parameters': {
            'Data Source': 'Daily Reddit sentiment',
            'Sentiment Window': 'Daily aggregation',
            'Capitulation Signal': 'Sentiment << historical mean',
            'Hype Signal': 'Sentiment > +0.5',
            'Peak Signal': 'Near maximum sentiment',
            'Position Type': 'Long-only (no shorts)',
            'Auto-Drop Threshold': 'If data > 5 days stale',
            'Data Availability': '566 days (from 2024-01-01)',
            'Subreddits': '/r/investing, /r/stocks, /r/wallstreetbets',
        },
        'yearly_data': {
            2024: {'return': 0.0848, 'sharpe': -0.380, 'max_dd': -0.1411},
            2025: {'return': 0.5014, 'sharpe': 1.427, 'max_dd': -0.1434},
            2026: {'return': 0.3924, 'sharpe': 1.682, 'max_dd': -0.0648},
        },
        'notes': [
            'Very limited data (only 2.2 years)',
            'Recent launch (2024-01-01)',
            'Below 1.0 Sharpe threshold (0.509)',
            'Data dependency - requires fresh sentiment daily',
            'Stale data risk - auto-drops if delayed',
            'Recent performance improving (2025-2026)',
            'RECOMMENDATION: Monitor, exclude until Sharpe > 1.0'
        ]
    }
}

# ============================================================================
# CREATE EXCEL WORKBOOK
# ============================================================================

print("\n[CREATING EXCEL WORKBOOK]")

out_file = Path("results") / "individual_strategy_results.xlsx"

with pd.ExcelWriter(out_file, engine='openpyxl') as writer:

    # ====================================================================
    # SHEET 1: SUMMARY
    # ====================================================================

    print("  Writing: Summary")

    summary_data = []
    for book_id, book_info in strategies.items():
        summary_data.append({
            'Book': book_id,
            'Strategy': book_info['name'],
            'Status': book_info['status'],
            'Period': book_info['backtest_period'],
            'Years': book_info['duration_years'],
            'Annual Return': f"{book_info['annual_return']:.2%}",
            'Sharpe': f"{book_info['sharpe']:.4f}",
            'Max DD': f"{book_info['max_dd']:.2%}",
            'Positive Years': book_info['positive_years'],
            'Recommendation': 'CORE' if book_id in ['A', 'D'] else 'SECONDARY' if book_id == 'B' else 'EXCLUDE' if book_id == 'C' else 'MONITOR'
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

    # ====================================================================
    # SHEET 2-6: INDIVIDUAL STRATEGY DETAILS
    # ====================================================================

    for book_id, book_info in strategies.items():

        print(f"  Writing: Book {book_id} - {book_info['name']}")

        sheet_name = f'Book {book_id}'

        # Overall Performance
        overall_data = pd.DataFrame({
            'Metric': [
                'Strategy Name',
                'Status',
                'Backtest Period',
                'Duration (Years)',
                'Annual Return',
                'Sharpe Ratio',
                'Sortino Ratio',
                'Max Drawdown',
                'Volatility',
                'Positive Years',
                'Win Rate',
                'Total Trades',
            ],
            'Value': [
                book_info['name'],
                book_info['status'],
                book_info['backtest_period'],
                f"{book_info['duration_years']:.2f}",
                f"{book_info['annual_return']:.2%}",
                f"{book_info['sharpe']:.4f}",
                'N/A',
                f"{book_info['max_dd']:.2%}",
                f"{book_info['volatility']:.2%}" if book_info['volatility'] else 'N/A',
                book_info['positive_years'],
                book_info['win_rate'] if book_info['win_rate'] else 'N/A',
                'N/A',
            ]
        })

        overall_data.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0, index=False)

        # Parameters
        param_data = pd.DataFrame({
            'Parameter': list(book_info['parameters'].keys()),
            'Value': list(book_info['parameters'].values())
        })

        startrow = len(overall_data) + 3
        param_data.to_excel(writer, sheet_name=sheet_name, startrow=startrow, startcol=0, index=False)

        # Yearly Performance
        if book_info['yearly_data']:
            yearly_rows = []
            for year, metrics in sorted(book_info['yearly_data'].items()):
                yearly_rows.append({
                    'Year': year,
                    'Return': f"{metrics['return']:.2%}",
                    'Sharpe': f"{metrics['sharpe']:.4f}",
                    'Max DD': f"{metrics['max_dd']:.2%}" if metrics['max_dd'] else 'N/A',
                })

            yearly_df = pd.DataFrame(yearly_rows)
            startrow = len(overall_data) + len(param_data) + 6
            yearly_df.to_excel(writer, sheet_name=sheet_name, startrow=startrow, startcol=0, index=False)

        # Notes
        notes_df = pd.DataFrame({
            'Notes': book_info['notes']
        })

        startrow = len(overall_data) + len(param_data) + len(yearly_rows if book_info['yearly_data'] else []) + 9
        notes_df.to_excel(writer, sheet_name=sheet_name, startrow=startrow, startcol=0, index=False, header=True)

    # ====================================================================
    # SHEET 7: COMPARISON
    # ====================================================================

    print("  Writing: Comparison")

    comparison_data = []
    for book_id in ['A', 'B', 'C', 'D', 'E']:
        book = strategies[book_id]
        comparison_data.append({
            'Book': book_id,
            'Name': book['name'],
            'Sharpe': book['sharpe'],
            'Return': book['annual_return'],
            'Max DD': book['max_dd'],
            'Status': book['status'],
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Sharpe', ascending=False)
    comparison_df.to_excel(writer, sheet_name='Comparison', index=False)

print(f"\n[SAVED] {out_file}")

# ============================================================================
# PRINT SUMMARY
# ============================================================================

print("\n" + "=" * 120)
print("INDIVIDUAL STRATEGY RESULTS SUMMARY")
print("=" * 120)

for book_id in ['D', 'A', 'B', 'E', 'C']:
    book = strategies[book_id]
    print(f"\n[{book_id}] {book['name']}")
    print(f"    Sharpe: {book['sharpe']:.4f}")
    print(f"    Return: {book['annual_return']:.2%}")
    print(f"    Max DD: {book['max_dd']:.2%}")
    print(f"    Period: {book['backtest_period']}")
    print(f"    Status: {book['status']}")

print("\n" + "=" * 120)
print("OPTIMAL PORTFOLIO (Based on Results)")
print("=" * 120)

print("\nRECOMMENDED: 50% D + 30% A + 20% B")
print("  50% Book D (Contrarian Bubble) - Sharpe 2.65, Max DD -10%")
print("  30% Book A (Momentum + Leverage) - Sharpe 1.41, Return 85%")
print("  20% Book B (QQQ Bubble) - Sharpe 1.66, 100% positive years")
print("\nEXCLUDE: Book C (Sharpe 0.98 < 1.0), Book E (Limited data, Sharpe 0.509)")

print("\n" + "=" * 120)
print("EXCEL FILE GENERATED")
print("=" * 120)
print(f"\nLocation: {out_file}")
print("\nSheets included:")
print("  - Summary: Quick overview of all strategies")
print("  - Book A: Daily Momentum + Leverage + UVXY details")
print("  - Book B: QQQ Bubble Hourly details")
print("  - Book C: Intraday MR details")
print("  - Book D: Contrarian Bubble details")
print("  - Book E: Reddit Sentiment details")
print("  - Comparison: All strategies ranked by Sharpe")
print()
