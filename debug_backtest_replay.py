"""
Debug script: Compare documented performance vs live/settle.py replay results.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from live import config as C
from live import signals as S
from live import settle as St
from live import engine as E


def check_book_d():
    """Debug Book D (Contrarian Bubble) vs documented performance."""
    print("\n" + "="*80)
    print("BOOK D: Contrarian Bubble Score")
    print("="*80)
    print("\nDocumented (2019-2026, 7.41 years):")
    print("  - Total Return: +993.56%")
    print("  - Annual Return: 38.08%")
    print("  - Sharpe: 2.6475")

    panels = S.load_panels()
    ret_d, pos_d, closed_d = St.replay_D(panels)

    # Filter to 2019+
    ret_d_recent = ret_d[ret_d.index >= "2019-01-01"]
    m = E.metrics_from_returns(ret_d_recent)

    print(f"\nMy Backtest ({ret_d_recent.index[0].date()} to {ret_d_recent.index[-1].date()}):")
    print(f"  - Total Return: {m['cum_ret']:.2%}")
    print(f"  - Annual Return: {m['ann_ret']:.2%}")
    print(f"  - Sharpe: {m['sharpe']:.4f}")
    print(f"  - Max DD: {m['maxdd']:.2%}")
    print(f"  - N days: {len(ret_d_recent)}")

    # Show first few days
    print(f"\nFirst 10 days of returns:")
    print(ret_d_recent.head(10))

    # Check for NaNs/zeros
    print(f"\nData quality:")
    print(f"  - Total days: {len(ret_d_recent)}")
    print(f"  - Non-zero days: {(ret_d_recent != 0).sum()}")
    print(f"  - Zero days: {(ret_d_recent == 0).sum()}")
    print(f"  - NaN days: {ret_d_recent.isna().sum()}")
    print(f"  - Max daily return: {ret_d_recent.max():.2%}")
    print(f"  - Min daily return: {ret_d_recent.min():.2%}")


def check_book_b():
    """Debug Book B (QQQ Bubble Hourly) vs documented performance."""
    print("\n" + "="*80)
    print("BOOK B: QQQ Bubble Hourly Momentum")
    print("="*80)
    print("\nDocumented (2020-2026, 5.85 years):")
    print("  - Total Return: 152.87%")
    print("  - Annual Return: 17.19%")
    print("  - Sharpe: 1.6603")

    panels = S.load_panels()
    ret_b, pos_b, closed_b = St.replay_B(panels)

    # Filter to 2020+
    ret_b_recent = ret_b[ret_b.index >= "2020-01-01"]
    m = E.metrics_from_returns(ret_b_recent)

    print(f"\nMy Backtest ({ret_b_recent.index[0].date()} to {ret_b_recent.index[-1].date()}):")
    print(f"  - Total Return: {m['cum_ret']:.2%}")
    print(f"  - Annual Return: {m['ann_ret']:.2%}")
    print(f"  - Sharpe: {m['sharpe']:.4f}")
    print(f"  - Max DD: {m['maxdd']:.2%}")
    print(f"  - N days: {len(ret_b_recent)}")

    print(f"\nFirst 10 days of returns:")
    print(ret_b_recent.head(10))

    print(f"\nData quality:")
    print(f"  - Total days: {len(ret_b_recent)}")
    print(f"  - Non-zero days: {(ret_b_recent != 0).sum()}")
    print(f"  - Zero days: {(ret_b_recent == 0).sum()}")
    print(f"  - NaN days: {ret_b_recent.isna().sum()}")
    print(f"  - Max daily return: {ret_b_recent.max():.2%}")
    print(f"  - Min daily return: {ret_b_recent.min():.2%}")


def check_book_c():
    """Debug Book C (Intraday MR) vs documented performance."""
    print("\n" + "="*80)
    print("BOOK C: Intraday MR + Momentum Flip")
    print("="*80)
    print("\nDocumented (2019-2026, 7.41 years):")
    print("  - Total Return: +480.97%")
    print("  - Annual Return: 26.78%")
    print("  - Sharpe: 0.9828")

    panels = S.load_panels()
    ret_c, pos_c, closed_c = St.replay_C(panels)

    # Filter to 2019+
    ret_c_recent = ret_c[ret_c.index >= "2019-01-01"]
    m = E.metrics_from_returns(ret_c_recent)

    print(f"\nMy Backtest ({ret_c_recent.index[0].date()} to {ret_c_recent.index[-1].date()}):")
    print(f"  - Total Return: {m['cum_ret']:.2%}")
    print(f"  - Annual Return: {m['ann_ret']:.2%}")
    print(f"  - Sharpe: {m['sharpe']:.4f}")
    print(f"  - Max DD: {m['maxdd']:.2%}")
    print(f"  - N days: {len(ret_c_recent)}")

    print(f"\nFirst 10 days of returns:")
    print(ret_c_recent.head(10))

    print(f"\nData quality:")
    print(f"  - Total days: {len(ret_c_recent)}")
    print(f"  - Non-zero days: {(ret_c_recent != 0).sum()}")
    print(f"  - Zero days: {(ret_c_recent == 0).sum()}")
    print(f"  - NaN days: {ret_c_recent.isna().sum()}")
    print(f"  - Max daily return: {ret_c_recent.max():.2%}")
    print(f"  - Min daily return: {ret_c_recent.min():.2%}")


if __name__ == "__main__":
    try:
        check_book_b()
        check_book_d()
        check_book_c()
        print("\n" + "="*80)
        print("DIAGNOSIS COMPLETE")
        print("="*80)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
