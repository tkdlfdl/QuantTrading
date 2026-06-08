"""
live/run_daily.py
=================
Idempotent daily orchestrator for the paper-trading track record.

Each run:
  0. PREPARE  — refresh caches (self-stale-managed), find latest complete trading date
  1. SETTLE   — re-replay all books, rebuild the live (>= inception) equity/metrics through
                the latest complete date  (deterministic → idempotent, no duplicate rows)
  2. PLAN     — write the next session's plan from data through the latest close
  3. REPORT   — render track_record.md + dashboard.png

Flags:
  --reset            wipe state and start flat
  --replay-last N    seed inception N business days before the latest cached date
                     (dry-run sanity check against the backtests, then use --reset)
  --no-refresh       skip the network data refresh (use existing cache)

Usage:
  python -m live.run_daily
  python -m live.run_daily --replay-last 60
  python -m live.run_daily --reset
"""
from __future__ import annotations
import sys
import json
import numpy as np
import pandas as pd

from . import config as C
from . import engine as E
from . import signals as S
from . import settle as SETTLE
from . import plan as PLAN
from . import report as REPORT
from . import prepare_data as PREP


def _running_equity_rows(book: str, rets: pd.Series, capital: float) -> pd.DataFrame:
    """Per-date equity + expanding metrics for one book."""
    r = rets.dropna()
    if r.empty:
        return pd.DataFrame()
    eq = capital * (1 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1
    # expanding Sharpe / Sortino (inception-to-date)
    rf_d = C.RISK_FREE_ANN / C.TRADING_DAYS
    exc = r - rf_d
    em = exc.expanding(min_periods=2).mean()
    es = r.expanding(min_periods=2).std()
    sharpe = (em / es * np.sqrt(C.TRADING_DAYS)).fillna(0.0)
    # downside std expanding
    neg = r.where(r < 0, np.nan)
    ds = neg.expanding(min_periods=2).std(ddof=0)
    sortino = (em / ds * np.sqrt(C.TRADING_DAYS)).replace([np.inf, -np.inf], 0).fillna(0.0)
    cum = eq / capital - 1
    return pd.DataFrame({
        "date": r.index, "book": book, "daily_ret": r.values,
        "equity": eq.values, "cum_ret": cum.values, "drawdown": dd.values,
        "sharpe_itd": sharpe.values, "sortino_itd": sortino.values,
    })


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    reset = "--reset" in argv
    no_refresh = "--no-refresh" in argv
    replay_last = None
    if "--replay-last" in argv:
        replay_last = int(argv[argv.index("--replay-last") + 1])

    C.ensure_dirs()
    store = E.StateStore()

    if reset:
        for f in (C.META_FILE, C.POSITIONS_FILE, C.EQUITY_FILE, C.TRADES_FILE):
            if f.exists():
                f.unlink()
        store = E.StateStore()
        print("  [run] state reset to flat.")

    # ── Phase 0: prepare data ───────────────────────────────────────
    print("Phase 0 — prepare data")
    last_complete = PREP.prepare(refresh=not no_refresh)

    # ── Determine inception (forward-only) ──────────────────────────
    if store.meta is None:
        if replay_last:
            inception = (last_complete - pd.offsets.BDay(replay_last)).normalize()
        else:
            inception = pd.Timestamp.now().normalize()   # forward-only: start today
        store.init_meta(str(inception.date()))
        print(f"  [run] inception set to {inception.date()} "
              f"({'replay' if replay_last else 'forward-only'}).")
    inception = pd.Timestamp(store.inception)

    # ── Phase 1: settle (replay all books, rebuild live equity) ─────
    print("Phase 1 — settle (replay books)")
    panels = S.load_panels()
    book_rets, positions, trades_df = SETTLE.replay_all(panels)

    # restrict to live window and to completed days
    live = book_rets[(book_rets.index >= inception) & (book_rets.index <= last_complete)]
    if live.empty:
        print(f"  [run] no completed trading days >= inception ({inception.date()}); "
              f"track record starts flat.")
        store.positions = positions
        store.meta["last_settled_date"] = str(inception.date())
        store.save()
    else:
        # portfolio combiners on the live window
        abcd = live[["A", "B", "C", "D"]]
        fixedew = E.fixed_ew(abcd)
        momalloc = E.mom_alloc(abcd)
        allr = abcd.copy()
        allr["FixedEW"] = fixedew
        allr["MomAlloc"] = momalloc
        # align all books to the common live calendar; missing = flat (no position)
        allr = allr.fillna(0.0)

        cap = store.meta["capital_per_book"]
        rows = [_running_equity_rows(b, allr[b], cap) for b in C.ALL_BOOKS]
        equity = pd.concat([df for df in rows if not df.empty], ignore_index=True)

        # keep only closed trades whose entry is within the live window
        if not trades_df.empty and "entry_ts" in trades_df.columns:
            ets = pd.to_datetime(trades_df["entry_ts"])
            trades_df = trades_df[ets >= inception].reset_index(drop=True)

        store.equity = equity
        store.positions = positions
        store.trades = trades_df
        store.meta["last_settled_date"] = str(live.index.max().date())
        store.save()
        print(f"  [run] settled {live.index.min().date()} → {live.index.max().date()} "
              f"({live.shape[0]} trading days, {len(C.ALL_BOOKS)} books).")

    # ── Phase 2: plan next session ──────────────────────────────────
    print("Phase 2 — plan next session")
    plan = PLAN.generate_plan(panels, last_complete)
    path = PLAN.write_plan(plan)
    store.meta["last_planned_date"] = plan["for_session"]
    store.save()
    print(f"  [run] wrote plan for {plan['for_session']} -> {path}")

    # ── Phase 3: report ─────────────────────────────────────────────
    print("Phase 3 — report")
    REPORT.render()
    print("Done.")


if __name__ == "__main__":
    main()
