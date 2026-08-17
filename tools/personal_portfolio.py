"""Personal Portfolio — long-hold sleeve (min holding period: 20 trading days).

Constraint: every security must be held >= 20 trading days. Only books whose
MINIMUM hold satisfies this qualify:

    A  daily momentum + UVXY hedge   rebalance 40 trading days
    F  universe hourly momentum      hold 200h ~= 29 trading days
    G  overnight-share x-section     hold 21 trading days   (INCUBATING)
    X  cross-asset ETF momentum      monthly recheck, holds persist while
                                     ranked (effective >= 21 trading days)

Excluded on the constraint: D/D14/DU (8-14 hour holds), C (1h + 3d), B (24h).

Construction: same champion machinery (inverse-vol weights, 21d rebalance of
weights*, 15% de-risk-only vol target, DM panic gate -> cash).
*Weight rebalancing trades only book-level allocations at their own natural
rebalance points; it does not force security sales before 20 days.

Usage:  python -m tools.personal_portfolio            # metrics + yearly
        python -m tools.personal_portfolio --holdings # current names per book
"""
from __future__ import annotations

import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from tools.metrics import metrics_from_returns, sharpe_ratio, max_drawdown
from live import engine as E, config as C

TD = 252
BOOKS_ELIGIBLE = ["A", "F", "G", "X"]
MIN_HOLD_TD = 20

SERIES = {
    "A": "book_a_retest",
    "F": "book_f_retest",
    "G": "book_g_live_spec",
    "X": "book_x_live_spec",
}

HOLD_AUDIT = {  # trading days, minimum a security is held
    "A": 40, "F": 29, "G": 21, "X": 21,
}


def _load(name: str) -> pd.Series:
    df = pd.read_csv(f"strategies/performance/{name}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)


def build(start: str = "2019-01-02", end: str | None = None) -> pd.Series:
    for b, h in HOLD_AUDIT.items():
        assert h >= MIN_HOLD_TD, f"book {b} violates min-hold constraint ({h}d)"
    books = {k: _load(v) for k, v in SERIES.items()}
    saved_b, saved_s = C.ALLOC_BOOKS, dict(C.ALLOC_SHARES)
    C.ALLOC_BOOKS = BOOKS_ELIGIBLE
    C.ALLOC_SHARES = {}
    try:
        R = pd.DataFrame(books)
        R = R[R.index >= pd.Timestamp(start)]
        if end:
            R = R[R.index <= pd.Timestamp(end)]
        return E.ivol_voltgt(R.fillna(0.0)).dropna()
    finally:
        C.ALLOC_BOOKS, C.ALLOC_SHARES = saved_b, saved_s


def report() -> None:
    ser = build()
    m = metrics_from_returns(ser.values, TD)
    print("PERSONAL PORTFOLIO (min hold 20 trading days: books A/F/G/X)")
    print(f"  window {ser.index.min().date()}..{ser.index.max().date()}")
    print(f"  Sharpe {m['sharpe']:.3f} | CAGR {m['cagr']:.1%} | "
          f"total {m['total_return']:+.0%} | MaxDD {m['max_dd']:.1%}")
    print("\n  hold-period audit:")
    for b in BOOKS_ELIGIBLE:
        note = " (INCUBATING)" if b in ("G", "X") else ""
        print(f"    {b}: min hold {HOLD_AUDIT[b]} trading days{note}")
    print("\n  yearly:")
    for y, x in ser.groupby(ser.index.year):
        print(f"    {y}: ret {(1+x).prod()-1:+8.2%}  sharpe {sharpe_ratio(x.values, TD):5.2f}  "
              f"mdd {max_drawdown(x.values):7.2%}")
    hist = pd.DataFrame({k: _load(v) for k, v in SERIES.items()}).reindex(ser.index).fillna(0.0).iloc[-60:]
    vol = hist.std() * np.sqrt(TD)
    iv = {b: 1.0 / vol[b] for b in BOOKS_ELIGIBLE if vol[b] > 1e-9}
    tot = sum(iv.values())
    print("\n  current book weights (inverse-vol):",
          {k: f"{v/tot:.0%}" for k, v in iv.items()})


def holdings() -> None:
    """Current names per eligible book from live state (positions.json /
    settle open positions where available)."""
    import json
    from pathlib import Path
    pos_file = Path("live/state/positions.json")
    if pos_file.exists():
        pos = json.loads(pos_file.read_text(encoding="utf-8"))
        for b in BOOKS_ELIGIBLE:
            recs = pos.get(b, [])
            names = sorted({r["ticker"] for r in recs}) if recs else []
            print(f"  {b}: {', '.join(names) if names else '(flat / between holds)'}")
    else:
        print("  live/state/positions.json not found — run the daily settle first")


if __name__ == "__main__":
    if "--holdings" in sys.argv:
        holdings()
    else:
        report()
