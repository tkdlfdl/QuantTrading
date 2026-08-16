"""
Backtest for Books A-E focusing on recent period (2020-2026) with both allocations.

Uses live/ engine to replay books with locked parameters, creates FixedEW and MomAlloc
portfolios, and generates comprehensive comparison.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).parent))

from live import config as C
from live import signals as S
from live import settle as St
from live import engine as E

warnings.filterwarnings('ignore')


def load_data():
    """Load all required price panels."""
    print("[1/7] Loading price data...")
    panels = S.load_panels()
    print(f"  [OK] Hourly: {panels['hourly_close'].shape[0]:,} bars x {panels['hourly_close'].shape[1]} tickers")
    print(f"  [OK] Daily:  {panels['daily_close'].shape[0]:,} bars x {panels['daily_close'].shape[1]} tickers")
    print(f"  [OK] Period: {panels['idx_h'][0].date()} to {panels['idx_h'][-1].date()}")
    return panels


def replay_all_books(panels):
    """Replay all 5 books with locked parameters."""
    print("\n[2/7] Replaying Books A-E...")

    books_data = {}

    # Book D (Contrarian Bubble)
    print("  [REPLAY] Book D (Contrarian Bubble Score)...")
    ret_d, _, _ = St.replay_D(panels)
    books_data["D"] = ret_d
    print(f"    -> {len(ret_d)} days, Sharpe={E.metrics_from_returns(ret_d)['sharpe']:.3f}")

    # Book B (QQQ Bubble Hourly)
    print("  [REPLAY] Book B (QQQ Bubble Hourly Momentum)...")
    ret_b, _, _ = St.replay_B(panels)
    books_data["B"] = ret_b
    print(f"    -> {len(ret_b)} days, Sharpe={E.metrics_from_returns(ret_b)['sharpe']:.3f}")

    # Book C (Intraday MR)
    print("  [REPLAY] Book C (Intraday MR + Momentum Flip)...")
    ret_c, _, _ = St.replay_C(panels)
    books_data["C"] = ret_c
    print(f"    -> {len(ret_c)} days, Sharpe={E.metrics_from_returns(ret_c)['sharpe']:.3f}")

    # Book A (Daily Momentum + Leverage + UVXY)
    print("  [REPLAY] Book A (Daily Momentum + Leverage + UVXY)...")
    ret_a, _, _ = St.replay_A(panels)
    books_data["A"] = ret_a
    metrics_a = E.metrics_from_returns(ret_a)
    print(f"    -> {len(ret_a)} days, Sharpe={metrics_a['sharpe']:.3f}")

    # Book E (Reddit Sentiment)
    print("  [REPLAY] Book E (Reddit Sentiment Long-Only)...")
    ret_e, _, _ = St.replay_E(panels)
    books_data["E"] = ret_e
    print(f"    -> {len(ret_e)} days, Sharpe={E.metrics_from_returns(ret_e)['sharpe']:.3f}")

    # Align all books to common period
    print("\n[3/7] Aligning books to common period...")
    all_dates = set()
    for ret in books_data.values():
        all_dates.update(ret.index)
    common_idx = sorted(all_dates)
    print(f"  Common dates: {len(common_idx)} days ({common_idx[0].date()} to {common_idx[-1].date()})")

    # Align to common period
    for book_name in list(books_data.keys()):
        orig_len = len(books_data[book_name])
        books_data[book_name] = books_data[book_name].reindex(common_idx).fillna(0)
        print(f"    {book_name}: aligned {orig_len} -> {len(books_data[book_name])} days")

    return books_data


def filter_recent_period(books_data, start_date="2020-01-01"):
    """Filter to recent period only."""
    print(f"\n[4/7] Filtering to recent period (>= {start_date})...")
    start = pd.Timestamp(start_date)
    books_filtered = {}
    for book, ret in books_data.items():
        ret_filt = ret[ret.index >= start]
        books_filtered[book] = ret_filt
        m = E.metrics_from_returns(ret_filt)
        print(f"  {book}: {len(ret_filt)} days, Sharpe={m['sharpe']:.3f}, Ann={m['ann_ret']:.2%}")

    return books_filtered


def create_portfolios(books_data):
    """Create FixedEW and MomAlloc portfolios."""
    print("\n[5/7] Creating portfolio allocations...")

    # Fixed Equal-Weight
    print("  [CREATE] FixedEW (equal-weight 20% each)...")
    weights_ew = {book: 1.0 / len(books_data) for book in books_data.keys()}
    ret_ew = pd.concat([
        books_data[book] * weights_ew[book]
        for book in sorted(books_data.keys())
    ], axis=1).sum(axis=1)
    m_ew = E.metrics_from_returns(ret_ew)
    print(f"    -> Sharpe={m_ew['sharpe']:.3f}, Ann={m_ew['ann_ret']:.2%}, MaxDD={m_ew['maxdd']:.2%}")

    # Momentum-Allocated (rolling 60d Sharpe)
    print("  [CREATE] MomAlloc (rolling 60d Sharpe rebalanced daily)...")
    common_idx = books_data["A"].index
    weights_ts = pd.DataFrame(index=common_idx, columns=sorted(books_data.keys()), dtype=float)

    for i, date in enumerate(common_idx):
        if i < 60:
            w = {book: 1.0 / len(books_data) for book in books_data.keys()}
        else:
            start_idx = i - 60
            sharpes = {}
            for book in books_data.keys():
                rets = books_data[book].iloc[start_idx:i]
                sharpe = max(E.metrics_from_returns(rets).get('sharpe', 0.0), 0.05)
                sharpes[book] = sharpe

            total = sum(sharpes.values())
            w = {book: sharpes[book] / total for book in sharpes.keys()}

        for book in w:
            weights_ts.loc[date, book] = w[book]

    ret_moma = (pd.concat([
        books_data[book] * weights_ts[book]
        for book in sorted(books_data.keys())
    ], axis=1).sum(axis=1))

    m_moma = E.metrics_from_returns(ret_moma)
    print(f"    -> Sharpe={m_moma['sharpe']:.3f}, Ann={m_moma['ann_ret']:.2%}, MaxDD={m_moma['maxdd']:.2%}")

    return ret_ew, weights_ew, ret_moma, weights_ts


def generate_summary_table(books_data, ret_ew, ret_moma):
    """Generate comprehensive summary."""
    print("\n[6/7] Generating summary metrics...")

    results = []

    # Individual books
    for book in sorted(books_data.keys()):
        m = E.metrics_from_returns(books_data[book])
        results.append({
            "Strategy": f"Book {book}: {C.BOOK_LABELS[book]}",
            "Days": len(books_data[book]),
            "Ann Return": f"{m['ann_ret']:.2%}",
            "Total Return": f"{m['cum_ret']:.2%}",
            "Sharpe": f"{m['sharpe']:.3f}",
            "Sortino": f"{m['sortino']:.3f}",
            "Max DD": f"{m['maxdd']:.2%}",
            "Vol": f"{m['vol_ann']:.2%}",
            "Win Rate": f"{m['win_rate']:.2%}",
        })

    # Portfolios
    m_ew = E.metrics_from_returns(ret_ew)
    results.append({
        "Strategy": "FixedEW: Equal-Weight (20% each)",
        "Days": len(ret_ew),
        "Ann Return": f"{m_ew['ann_ret']:.2%}",
        "Total Return": f"{m_ew['cum_ret']:.2%}",
        "Sharpe": f"{m_ew['sharpe']:.3f}",
        "Sortino": f"{m_ew['sortino']:.3f}",
        "Max DD": f"{m_ew['maxdd']:.2%}",
        "Vol": f"{m_ew['vol_ann']:.2%}",
        "Win Rate": f"{m_ew['win_rate']:.2%}",
    })

    m_moma = E.metrics_from_returns(ret_moma)
    results.append({
        "Strategy": "MomAlloc: Rolling 60d Sharpe (daily rebal)",
        "Days": len(ret_moma),
        "Ann Return": f"{m_moma['ann_ret']:.2%}",
        "Total Return": f"{m_moma['cum_ret']:.2%}",
        "Sharpe": f"{m_moma['sharpe']:.3f}",
        "Sortino": f"{m_moma['sortino']:.3f}",
        "Max DD": f"{m_moma['maxdd']:.2%}",
        "Vol": f"{m_moma['vol_ann']:.2%}",
        "Win Rate": f"{m_moma['win_rate']:.2%}",
    })

    return pd.DataFrame(results)


def save_results(books_data, ret_ew, ret_moma, summary_df):
    """Save detailed results to Excel."""
    print("\n[7/7] Saving results to Excel...")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "backtest_all_books_recent.xlsx"

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        # Summary
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Daily returns
        returns_df = pd.DataFrame({f"Book {b}": books_data[b] for b in sorted(books_data.keys())})
        returns_df["FixedEW"] = ret_ew
        returns_df["MomAlloc"] = ret_moma
        returns_df.to_excel(writer, sheet_name="Daily Returns")

        # Cumulative returns
        cum_df = (1 + returns_df).cumprod()
        cum_df.to_excel(writer, sheet_name="Cumulative Returns")

        # Yearly performance
        yearly = returns_df.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        yearly.index = yearly.index.year
        yearly.to_excel(writer, sheet_name="Yearly Returns")

    print(f"  [SAVED] {out_file}")
    return out_file


def main():
    print("="*110)
    print("BACKTEST: Books A-E with FixedEW & MomAlloc (Recent Period: 2020+)")
    print("="*110)

    try:
        # Load and replay
        panels = load_data()
        books_data = replay_all_books(panels)

        # Filter to recent period
        books_data = filter_recent_period(books_data, start_date="2020-01-01")

        # Create portfolios
        ret_ew, weights_ew, ret_moma, weights_ts = create_portfolios(books_data)

        # Generate summary
        summary_df = generate_summary_table(books_data, ret_ew, ret_moma)

        # Save results
        out_file = save_results(books_data, ret_ew, ret_moma, summary_df)

        # Print summary
        print("\n" + "="*110)
        print("BACKTEST SUMMARY (2020-2026)")
        print("="*110)
        print(summary_df.to_string(index=False))
        print("="*110)

        print(f"\n[SUCCESS] Backtest complete! Results saved to {out_file.name}")

    except Exception as e:
        print(f"\n[FATAL] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
