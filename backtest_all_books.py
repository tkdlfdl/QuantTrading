"""
Comprehensive backtest for Books A-E with FixedEW and MomAlloc portfolios.

Uses live/ engine primitives to replay all books with locked parameters,
then combines them using fixed equal-weight and momentum-allocation rules.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add live engine to path
sys.path.insert(0, str(Path(__file__).parent))

from live import config as C
from live import signals as S
from live import settle as St
from live import engine as E

import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load all required price panels."""
    print("[1/6] Loading price data...")
    try:
        panels = S.load_panels()
        print(f"  [OK] Hourly close: {panels['hourly_close'].shape}")
        print(f"  [OK] Hourly open: {panels['hourly_open'].shape}")
        print(f"  [OK] Daily close: {panels['daily_close'].shape}")
        return panels
    except Exception as e:
        print(f"  [ERROR] Error loading data: {e}")
        raise


def replay_all_books(panels):
    """Replay all 5 books with locked parameters."""
    print("\n[2/6] Replaying Books A-E...")

    books_data = {}

    # Book D (Contrarian Bubble)
    print("  → Book D (Contrarian Bubble)...")
    try:
        ret_d, pos_d, closed_d = St.replay_D(panels)
        books_data["D"] = ret_d
        print(f"    [OK] {len(ret_d)} days")
    except Exception as e:
        print(f"    [ERROR] Error: {e}")

    # Book B (QQQ Bubble Hourly)
    print("  -> Book B (QQQ Bubble Hourly)...")
    try:
        ret_b, pos_b, closed_b = St.replay_B(panels)
        books_data["B"] = ret_b
        print(f"    [OK] {len(ret_b)} days")
    except Exception as e:
        print(f"    [ERROR] Error: {e}")

    # Book C (Intraday MR)
    print("  -> Book C (Intraday MR)...")
    try:
        ret_c, pos_c, closed_c = St.replay_C(panels)
        books_data["C"] = ret_c
        print(f"    [OK] {len(ret_c)} days")
    except Exception as e:
        print(f"    [ERROR] Error: {e}")

    # Book A (Daily Momentum + Leverage + UVXY)
    print("  -> Book A (Daily Momentum + Leverage + UVXY)...")
    try:
        ret_a, pos_a, closed_a = St.replay_A(panels)
        books_data["A"] = ret_a
        print(f"    [OK] {len(ret_a)} days")
    except Exception as e:
        print(f"    [ERROR] Error: {e}")

    # Book E (Reddit Sentiment)
    print("  -> Book E (Reddit Sentiment)...")
    try:
        ret_e, pos_e, closed_e = St.replay_E(panels)
        books_data["E"] = ret_e
        print(f"    [OK] {len(ret_e)} days")
    except Exception as e:
        print(f"    [ERROR] Error: {e}")

    # Align all books to common period
    print("\n  Aligning books to common period...")
    all_dates = set()
    for ret in books_data.values():
        all_dates.update(ret.index)
    common_idx = sorted(all_dates)

    for book_name in list(books_data.keys()):
        books_data[book_name] = books_data[book_name].reindex(common_idx).fillna(0)

    print(f"  [OK] Common period: {len(common_idx)} days ({common_idx[0].date()} to {common_idx[-1].date()})")

    return books_data


def create_fixed_ew_portfolio(books_data):
    """Fixed equal-weight across all books."""
    print("\n[3/6] Creating FixedEW portfolio (equal-weight)...")
    weights = {book: 1.0 / len(books_data) for book in books_data.keys()}

    returns = pd.concat([
        books_data[book] * weights[book]
        for book in sorted(books_data.keys())
    ], axis=1).sum(axis=1)

    print(f"  Weights: {' | '.join(f'{b}={w:.1%}' for b, w in sorted(weights.items()))}")
    return returns, weights


def create_momentum_alloc_portfolio(books_data, window=60):
    """Momentum-weighted portfolio using rolling Sharpe."""
    print(f"\n[4/6] Creating MomAlloc portfolio (rolling Sharpe {window}d, rebalance daily)...")

    common_idx = books_data["A"].index
    weights_ts = pd.DataFrame(index=common_idx, columns=sorted(books_data.keys()), dtype=float)

    for i, date in enumerate(common_idx):
        if i < window:
            # Use equal weight during warmup
            w = {book: 1.0 / len(books_data) for book in books_data.keys()}
        else:
            # Compute rolling Sharpe for past `window` days
            start_idx = i - window
            sharpes = {}
            for book in books_data.keys():
                rets = books_data[book].iloc[start_idx:i]
                sharpe = E.metrics_from_returns(rets).get('sharpe', 0.0)
                sharpes[book] = max(sharpe, 0.1)  # floor at 0.1 to avoid negatives

            # Normalize to sum to 1
            total = sum(sharpes.values())
            w = {book: sharpes[book] / total for book in sharpes.keys()}

        for book in w:
            weights_ts.loc[date, book] = w[book]

    # Compute portfolio returns
    returns = (pd.concat([
        books_data[book] * weights_ts[book]
        for book in sorted(books_data.keys())
    ], axis=1).sum(axis=1))

    return returns, weights_ts


def compute_metrics(returns, label=""):
    """Compute comprehensive metrics."""
    metrics = E.metrics_from_returns(returns)

    # Additional metrics
    equity = (1 + returns).cumprod()

    return {
        "Label": label,
        "Days": len(returns),
        "Start": returns.index[0].date(),
        "End": returns.index[-1].date(),
        "Annual Return": f"{metrics['ann_ret']:.2%}",
        "Total Return": f"{metrics['cum_ret']:.2%}",
        "Sharpe Ratio": f"{metrics['sharpe']:.3f}",
        "Sortino Ratio": f"{metrics['sortino']:.3f}",
        "Max Drawdown": f"{metrics['maxdd']:.2%}",
        "Volatility": f"{metrics['vol_ann']:.2%}",
        "Win Rate": f"{metrics['win_rate']:.2%}",
    }


def generate_report(books_data, fixed_ew_ret, fixed_ew_wts, moma_ret, moma_wts):
    """Generate comprehensive performance report."""
    print("\n[5/6] Computing metrics...")

    results = []

    # Individual books
    for book in sorted(books_data.keys()):
        metrics = compute_metrics(books_data[book], label=f"Book {book} ({C.BOOK_LABELS[book]})")
        results.append(metrics)

    # Portfolios
    results.append(compute_metrics(fixed_ew_ret, label="FixedEW Portfolio"))
    results.append(compute_metrics(moma_ret, label="MomAlloc Portfolio"))

    df = pd.DataFrame(results)
    return df


def save_results(books_data, fixed_ew_ret, moma_ret, report_df):
    """Save results to Excel and CSV."""
    print("\n[6/6] Saving results...")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    # Excel file with multiple sheets
    with pd.ExcelWriter(out_dir / "backtest_all_books_comprehensive.xlsx", engine="openpyxl") as writer:
        # Summary metrics
        report_df.to_excel(writer, sheet_name="Summary", index=False)

        # Daily returns for each book
        for book in sorted(books_data.keys()):
            books_data[book].to_frame(name=f"Book {book}").to_excel(
                writer, sheet_name=f"Book {book} Daily Returns"
            )

        # Portfolio daily returns
        fixed_ew_ret.to_frame(name="FixedEW").to_excel(writer, sheet_name="FixedEW Daily Returns")
        moma_ret.to_frame(name="MomAlloc").to_excel(writer, sheet_name="MomAlloc Daily Returns")

        # Cumulative returns
        cum_df = pd.DataFrame({
            **{f"Book {book}": (1 + books_data[book]).cumprod() for book in sorted(books_data.keys())},
            "FixedEW": (1 + fixed_ew_ret).cumprod(),
            "MomAlloc": (1 + moma_ret).cumprod(),
        })
        cum_df.to_excel(writer, sheet_name="Cumulative Returns")

    print(f"  [OK] Saved: {out_dir / 'backtest_all_books_comprehensive.xlsx'}")

    # Print summary
    print("\n" + "="*120)
    print("BACKTEST SUMMARY")
    print("="*120)
    print(report_df.to_string(index=False))
    print("="*120)


def main():
    print("="*120)
    print("COMPREHENSIVE BACKTEST: Books A-E with FixedEW & MomAlloc")
    print("="*120)

    try:
        # Load data
        panels = load_data()

        # Replay all books
        books_data = replay_all_books(panels)

        # Create portfolios
        fixed_ew_ret, fixed_ew_wts = create_fixed_ew_portfolio(books_data)
        moma_ret, moma_wts = create_momentum_alloc_portfolio(books_data)

        # Generate report
        report_df = generate_report(books_data, fixed_ew_ret, fixed_ew_wts, moma_ret, moma_wts)

        # Save results
        save_results(books_data, fixed_ew_ret, moma_ret, report_df)

        print("\n[DONE] Backtest complete!")

    except Exception as e:
        print(f"\n[FATAL] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
