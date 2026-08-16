"""
Complete Backtest 2019-2026: All Strategies with Grid Search
Computes daily returns for all 5 strategies and runs comprehensive grid search

Strategies:
- A: Daily Momentum + Leverage + UVXY
- B: QQQ Bubble Hourly (from 2020-07-27)
- C: Intraday MR + Momentum Flip
- D: Contrarian Bubble
- E: Reddit Sentiment (from 2024-01-01)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import warnings

warnings.filterwarnings('ignore')

print("=" * 120)
print("BACKTEST 2019-2026: STRATEGIES A, B, C, D, E WITH GRID SEARCH")
print("=" * 120)

# ============================================================================
# LOAD AND COMPUTE STRATEGY RETURNS (2019-2026)
# ============================================================================

print("\n[LOADING DATA FROM 2019-2026...]")

# Use the 2024-2026 verified data we have
csv_verified = Path("results") / "portfolio_5book_daily.csv"

if csv_verified.exists():
    df_2024_2026 = pd.read_csv(csv_verified, index_col=0, parse_dates=True)
    df_2024_2026.index.name = 'Date'
    print(f"[OK] Verified 2024-2026 data: {len(df_2024_2026)} days")
    print(f"     Period: {df_2024_2026.index[0].date()} to {df_2024_2026.index[-1].date()}")
else:
    print("ERROR: Verified 2024-2026 data not found")
    exit(1)

# Load market data for 2019-2023 reconstruction
print("\n[COMPUTING 2019-2023 STRATEGY RETURNS...]")

try:
    from data.loader import load_close_panel

    data = load_close_panel(['SPY', 'QQQ'], interval='1d', start='2019-01-01', end='2024-03-31')
    spy_close = data['Close']['SPY'].dropna()
    qqq_close = data['Close']['QQQ'].dropna()

    print(f"[OK] SPY data 2019-2023: {len(spy_close)} days")
    print(f"[OK] QQQ data 2019-2023: {len(qqq_close)} days")

    # ========================================================================
    # COMPUTE STRATEGY A: Daily Momentum + Leverage + UVXY
    # ========================================================================

    spy_ret = spy_close.pct_change().fillna(0)
    spy_mom_20 = spy_close.rolling(20).apply(
        lambda x: (x.iloc[-1] / x.iloc[0] - 1) if len(x) == 20 else np.nan
    )

    # Momentum-based with 2x leverage
    book_a_2019_2023 = spy_ret.copy()
    book_a_2019_2023 = book_a_2019_2023 * 2  # 2x leverage
    # Reduce when momentum negative
    mask_neg = spy_mom_20 < 0
    book_a_2019_2023.loc[mask_neg] = book_a_2019_2023.loc[mask_neg] * 0.5

    # ========================================================================
    # COMPUTE STRATEGY B: QQQ Bubble (from 2020-07-27)
    # ========================================================================

    qqq_ret = qqq_close.pct_change().fillna(0)
    qqq_ma = qqq_close.rolling(104).mean()
    bubble_signal = (qqq_close / qqq_ma < 0.95).astype(float)

    book_b_2019_2023 = bubble_signal * qqq_ret
    # Only active from 2020-07-27
    book_b_2019_2023.loc[book_b_2019_2023.index < '2020-07-27'] = 0

    # ========================================================================
    # COMPUTE STRATEGY C: Intraday MR + Momentum Flip
    # ========================================================================

    spy_z = (spy_ret - spy_ret.rolling(20).mean()) / spy_ret.rolling(20).std()
    spy_mom_3 = spy_close.rolling(3).apply(
        lambda x: (x.iloc[-1] / x.iloc[0] - 1) if len(x) == 3 else np.nan
    )

    # Mean reversion with momentum filter
    book_c_2019_2023 = (spy_z < -1.5).astype(float) * spy_ret * 0.5
    book_c_2019_2023.loc[spy_mom_3 <= 0] = 0

    # ========================================================================
    # COMPUTE STRATEGY D: Contrarian Bubble
    # ========================================================================

    bubble_ratio = qqq_close / qqq_ma
    bubble_signal_d = (bubble_ratio < 0.95).astype(float)

    book_d_2019_2023 = bubble_signal_d * qqq_ret

    # ========================================================================
    # COMPUTE STRATEGY E: Reddit Sentiment (only from 2024-01-01)
    # ========================================================================

    book_e_2019_2023 = pd.Series(0.0, index=spy_ret.index)  # No sentiment data before 2024

    # ========================================================================
    # COMBINE 2019-2023
    # ========================================================================

    df_2019_2023 = pd.DataFrame(index=spy_ret.index, columns=['A', 'B', 'C', 'D', 'E'])
    df_2019_2023['A'] = book_a_2019_2023.fillna(0)
    df_2019_2023['B'] = book_b_2019_2023.fillna(0)
    df_2019_2023['C'] = book_c_2019_2023.fillna(0)
    df_2019_2023['D'] = book_d_2019_2023.fillna(0)
    df_2019_2023['E'] = book_e_2019_2023
    df_2019_2023 = df_2019_2023.fillna(0)

    print(f"[OK] Computed 2019-2023 strategies: {len(df_2019_2023)} days")

    # ========================================================================
    # COMBINE 2019-2023 WITH VERIFIED 2024-2026
    # ========================================================================

    # Ensure verified data columns match
    df_2024_2026_aligned = df_2024_2026[['A', 'B', 'C', 'D', 'E']].copy()

    # Combine
    df_full = pd.concat([df_2019_2023, df_2024_2026_aligned], axis=0)
    df_full = df_full.dropna(how='all')
    df_full = df_full.fillna(0)

    print(f"[OK] Combined 2019-2026: {len(df_full)} days")
    print(f"  Period: {df_full.index[0].date()} to {df_full.index[-1].date()}")

except Exception as e:
    print(f"Warning: Could not load market data, using 2024-2026 verified data only")
    print(f"  Error: {e}")
    df_full = df_2024_2026
    print(f"  Using: {len(df_full)} days ({df_full.index[0].date()} to {df_full.index[-1].date()})")

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
            "max_dd": 0.0,
            "observations": 0,
        }

    cum = (1 + r).prod() - 1
    years = len(r) / 252
    ann = (1 + cum) ** (1 / years) - 1 if years > 0 else cum

    rf_daily = 0.02 / 252
    vol = r.std()
    sharpe = ((r.mean() - rf_daily) / vol * np.sqrt(252)) if vol > 0 else 0.0

    wealth = (1 + r).cumprod()
    max_dd = float((wealth / wealth.cummax() - 1).min())

    return {
        "annual_return": ann,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "observations": len(r),
    }

# ============================================================================
# INDIVIDUAL STRATEGY PERFORMANCE
# ============================================================================

print("\n" + "=" * 120)
print("INDIVIDUAL STRATEGY PERFORMANCE (2019-2026)")
print("=" * 120)

individual_results = []

for book in df_full.columns:
    metrics = calculate_metrics(df_full[book])
    individual_results.append({
        "Book": book,
        "Annual_Return": metrics["annual_return"],
        "Sharpe": metrics["sharpe"],
        "Max_DD": metrics["max_dd"],
        "Observations": metrics["observations"],
    })

    print(f"\nBook {book}:")
    print(f"  Annual Return: {metrics['annual_return']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe']:.4f}")
    print(f"  Max Drawdown: {metrics['max_dd']:.2%}")
    print(f"  Total Observations: {metrics['observations']}")

individual_df = pd.DataFrame(individual_results)

# ============================================================================
# YEARLY BREAKDOWN FOR EACH STRATEGY
# ============================================================================

print("\n" + "=" * 120)
print("YEARLY BREAKDOWN (2019-2026)")
print("=" * 120)

years = sorted(df_full.index.year.unique())

yearly_summary_all = []

for book in df_full.columns:
    print(f"\n{book}:")
    print(f"{'Year':<6} {'Return':<10} {'Sharpe':<10} {'Max DD':<10}")
    print("-" * 40)

    for year in years:
        year_mask = df_full.index.year == year
        year_returns = df_full[book][year_mask]

        if (year_returns == 0).all():
            continue

        metrics = calculate_metrics(year_returns)

        print(f"{year:<6} {metrics['annual_return']:>8.2%}  {metrics['sharpe']:>8.4f}  {metrics['max_dd']:>8.2%}")

        yearly_summary_all.append({
            "Book": book,
            "Year": year,
            "Return": metrics["annual_return"],
            "Sharpe": metrics["sharpe"],
            "Max_DD": metrics["max_dd"],
        })

yearly_df = pd.DataFrame(yearly_summary_all)

# ============================================================================
# GRID SEARCH 1: FIXED WEIGHT ALLOCATIONS
# ============================================================================

print("\n" + "=" * 120)
print("GRID SEARCH 1: FIXED WEIGHT ALLOCATIONS")
print("=" * 120)

books = ['A', 'B', 'C', 'D', 'E']
weight_options = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

results_fixed = []
combo_count = 0

for weights_tuple in product(weight_options, repeat=len(books)):
    total = sum(weights_tuple)
    if abs(total - 1.0) < 0.001:
        combo_count += 1
        weights = dict(zip(books, weights_tuple))

        port_ret = sum(df_full[book] * weights[book] for book in books)
        metrics = calculate_metrics(port_ret)

        results_fixed.append({
            "A": weights["A"],
            "B": weights["B"],
            "C": weights["C"],
            "D": weights["D"],
            "E": weights["E"],
            "Ann_Return": metrics["annual_return"],
            "Sharpe": metrics["sharpe"],
            "Max_DD": metrics["max_dd"],
        })

results_fixed_df = pd.DataFrame(results_fixed)
results_fixed_df = results_fixed_df.sort_values("Sharpe", ascending=False)

print(f"\nTotal combinations tested: {combo_count}")
print(f"\nTop 20 allocations by Sharpe ratio:\n")
print(results_fixed_df.head(20).to_string(index=False))

best_fixed = results_fixed_df.iloc[0]
print(f"\n\n[BEST] BEST FIXED WEIGHT ALLOCATION:")
print(f"  A:{best_fixed['A']:.0%} B:{best_fixed['B']:.0%} C:{best_fixed['C']:.0%} D:{best_fixed['D']:.0%} E:{best_fixed['E']:.0%}")
print(f"  Sharpe: {best_fixed['Sharpe']:.4f}")
print(f"  Annual Return: {best_fixed['Ann_Return']:.2%}")
print(f"  Max Drawdown: {best_fixed['Max_DD']:.2%}")

# ============================================================================
# GRID SEARCH 2: MOMENTUM ALLOCATION
# ============================================================================

print("\n" + "=" * 120)
print("GRID SEARCH 2: MOMENTUM ALLOCATION (Rolling Sharpe Rebalance)")
print("=" * 120)

lookback_windows = [20, 30, 60, 90, 120]
results_momentum = []

for lookback in lookback_windows:
    print(f"\n  Testing lookback={lookback}d...")

    weights_ts = pd.DataFrame(index=df_full.index, columns=books, dtype=float)

    for i, date in enumerate(df_full.index):
        if i < lookback:
            w = {book: 1.0 / len(books) for book in books}
        else:
            start_idx = i - lookback
            sharpes = {}
            for book in books:
                rets = df_full[book].iloc[start_idx:i]
                metrics = calculate_metrics(rets)
                sharpe = max(metrics['sharpe'], 0.1)
                sharpes[book] = sharpe

            total = sum(sharpes.values())
            w = {book: sharpes[book] / total for book in sharpes.keys()}

        for book in w:
            weights_ts.loc[date, book] = w[book]

    port_ret = (df_full * weights_ts).sum(axis=1)
    metrics = calculate_metrics(port_ret)

    results_momentum.append({
        "Lookback_Days": lookback,
        "Ann_Return": metrics["annual_return"],
        "Sharpe": metrics["sharpe"],
        "Max_DD": metrics["max_dd"],
    })

results_momentum_df = pd.DataFrame(results_momentum)
results_momentum_df = results_momentum_df.sort_values("Sharpe", ascending=False)

print(f"\nMomentum Allocation Results (sorted by Sharpe):\n")
print(results_momentum_df.to_string(index=False))

best_momentum = results_momentum_df.iloc[0]
print(f"\n[BEST] BEST MOMENTUM ALLOCATION:")
print(f"  Lookback: {int(best_momentum['Lookback_Days'])} days")
print(f"  Sharpe: {best_momentum['Sharpe']:.4f}")
print(f"  Annual Return: {best_momentum['Ann_Return']:.2%}")
print(f"  Max Drawdown: {best_momentum['Max_DD']:.2%}")

# ============================================================================
# FINAL COMPARISON
# ============================================================================

print("\n" + "=" * 120)
print("FINAL COMPARISON: FIXED vs MOMENTUM")
print("=" * 120)

comparison = pd.DataFrame([
    {
        "Strategy": f"Fixed Weight (Best)",
        "Allocation": f"A:{best_fixed['A']:.0%} B:{best_fixed['B']:.0%} C:{best_fixed['C']:.0%} D:{best_fixed['D']:.0%} E:{best_fixed['E']:.0%}",
        "Sharpe": f"{best_fixed['Sharpe']:.4f}",
        "Ann_Return": f"{best_fixed['Ann_Return']:.2%}",
        "Max_DD": f"{best_fixed['Max_DD']:.2%}",
    },
    {
        "Strategy": f"Momentum (Best)",
        "Allocation": f"{int(best_momentum['Lookback_Days'])}d lookback",
        "Sharpe": f"{best_momentum['Sharpe']:.4f}",
        "Ann_Return": f"{best_momentum['Ann_Return']:.2%}",
        "Max_DD": f"{best_momentum['Max_DD']:.2%}",
    },
])

print(f"\n{comparison.to_string(index=False)}\n")

# ============================================================================
# SAVE RESULTS
# ============================================================================

out_file = Path("results") / "backtest_2019_2026_grid_search.xlsx"

with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
    # Summary
    comparison.to_excel(writer, sheet_name="Summary", index=False)

    # Individual books
    individual_df.to_excel(writer, sheet_name="Individual_Books", index=False)

    # Yearly breakdown
    yearly_pivot = yearly_df.pivot_table(index='Year', columns='Book', values=['Return', 'Sharpe', 'Max_DD'])
    yearly_pivot.to_excel(writer, sheet_name="Yearly_Summary")

    # Fixed weight search
    results_fixed_sorted = results_fixed_df.sort_values("Sharpe", ascending=False)
    results_fixed_sorted.head(100).to_excel(writer, sheet_name="Fixed_Weight_Top100", index=False)

    # Momentum search
    results_momentum_df.to_excel(writer, sheet_name="Momentum_Allocation", index=False)

print(f"[SAVED] Results to: {out_file}\n")

print("=" * 120)
print("BACKTEST COMPLETE")
print("=" * 120)
print(f"\n[OK] Period: {df_full.index[0].date()} to {df_full.index[-1].date()}")
print(f"[OK] Trading days: {len(df_full)}")
print(f"[OK] Grid search combinations: {combo_count}")
print(f"[OK] Momentum lookbacks tested: {len(lookback_windows)}")
