"""
live/report.py
==============
Render the human-readable track record (track_record.md) and dashboard.png from
the persisted equity.csv / positions.json / trades.csv.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from . import engine as E


def _load_equity():
    if not C.EQUITY_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(C.EQUITY_FILE, parse_dates=["date"])
    return df


def render():
    eq = _load_equity()
    meta = json.loads(C.META_FILE.read_text(encoding="utf-8")) if C.META_FILE.exists() else {}
    positions = json.loads(C.POSITIONS_FILE.read_text(encoding="utf-8")) if C.POSITIONS_FILE.exists() else {}
    n_open = {b: len(positions.get(b, [])) for b in C.BOOKS}

    lines = []
    lines.append("# Paper-Trading Track Record")
    lines.append("")
    lines.append(f"**Inception:** {meta.get('inception','-')}  ")
    lines.append(f"**Last settled:** {meta.get('last_settled_date','-')}  ")
    lines.append(f"**Capital per book:** ${meta.get('capital_per_book', C.CAPITAL_PER_BOOK):,.0f}  ")
    lines.append(f"**Transaction cost:** {C.TC_ONE_WAY*100:.2f}% one-way "
                 f"({C.TC_ONE_WAY*200:.2f}% round-trip)  ")
    lines.append("")

    if eq.empty:
        lines.append("_No settled trading days yet - track record starts flat and will "
                     "populate after the first completed session._")
        C.TRACK_RECORD_MD.write_text("\n".join(lines), encoding="utf-8")
        print(f"  [report] wrote {C.TRACK_RECORD_MD} (empty track record)")
        return

    # latest snapshot per book
    lines.append("## Current Standings")
    lines.append("")
    lines.append("| Book | Strategy | Cum Ret | Ann Ret | Sharpe | Sortino | MaxDD | "
                 "Win% | Today P&L | Open Pos |")
    lines.append("|------|----------|--------:|--------:|-------:|--------:|------:|"
                 "-----:|----------:|---------:|")

    last_date = eq["date"].max()
    for b in C.ALL_BOOKS:
        sub = eq[eq.book == b].sort_values("date")
        if sub.empty:
            continue
        rets = pd.Series(sub["daily_ret"].values, index=sub["date"].values)
        m = E.metrics_from_returns(rets)
        today = sub[sub.date == last_date]
        today_pnl = float(today["daily_ret"].iloc[0]) if not today.empty else 0.0
        today_dollar = today_pnl * C.CAPITAL_PER_BOOK
        op = n_open.get(b, "-")
        lines.append(
            f"| {b} | {C.BOOK_LABELS[b]} | {m['cum_ret']*100:+.1f}% | "
            f"{m['ann_ret']*100:+.1f}% | {m['sharpe']:.2f} | {m['sortino']:.2f} | "
            f"{m['maxdd']*100:.1f}% | {m['win_rate']*100:.0f}% | "
            f"{today_pnl*100:+.2f}% (${today_dollar:+,.0f}) | {op} |"
        )

    lines.append("")
    n_settled = eq["date"].nunique()
    lines.append(f"_Settled trading days: {n_settled}_  ")
    if C.TRADES_FILE.exists():
        tr = pd.read_csv(C.TRADES_FILE)
        lines.append(f"_Closed trades logged (B+D): {len(tr)}_  ")

    C.TRACK_RECORD_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [report] wrote {C.TRACK_RECORD_MD}")

    _dashboard(eq)


def _dashboard(eq: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("Paper-Trading Track Record", fontsize=13, fontweight="bold")

    colors = {"A":"steelblue","B":"green","C":"orange","D":"crimson",
              "FixedEW":"darkred","MomAlloc":"purple"}

    # 1. Equity curves
    ax = axes[0]
    for b in C.ALL_BOOKS:
        sub = eq[eq.book == b].sort_values("date")
        if sub.empty:
            continue
        lw = 2.4 if b in C.PORTFOLIOS else 1.4
        ax.plot(sub["date"], sub["equity"], lw=lw, color=colors.get(b), label=b)
    ax.set_title("Equity ($)", fontweight="bold")
    ax.legend(fontsize=8, ncol=3); ax.grid(True, alpha=0.3)

    # 2. Drawdown
    ax = axes[1]
    for b in C.ALL_BOOKS:
        sub = eq[eq.book == b].sort_values("date")
        if sub.empty:
            continue
        lw = 2.0 if b in C.PORTFOLIOS else 1.2
        ax.plot(sub["date"], sub["drawdown"]*100, lw=lw, color=colors.get(b), label=b)
    ax.set_title("Drawdown (%)", fontweight="bold")
    ax.legend(fontsize=8, ncol=3); ax.grid(True, alpha=0.3)

    # 3. Rolling 30-day Sharpe of the two portfolios
    ax = axes[2]
    for b in C.PORTFOLIOS:
        sub = eq[eq.book == b].sort_values("date")
        if sub.empty:
            continue
        rets = pd.Series(sub["daily_ret"].values, index=pd.to_datetime(sub["date"].values))
        rs = E.rolling_sharpe(rets, 30)
        ax.plot(sub["date"], rs.values, lw=2.0, color=colors.get(b), label=b)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("30-day Rolling Sharpe (portfolios)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(C.DASHBOARD_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [report] wrote {C.DASHBOARD_PNG}")
