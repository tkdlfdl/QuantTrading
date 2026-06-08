"""
live/run_intraday.py
====================
Intraday execution loop — mirrors the MomAlloc book into an Alpaca paper account.

Each tick:
  0. market-open gate (skip weekends / outside ~09:30-16:00 ET) unless --force
  1. refresh today's bars (intraday_data)
  2. compute MomAlloc per-book weights from the live track record (equity.csv)
  3. tick the live position book -> blended per-symbol target shares
  4. reconcile via broker (DRY-RUN logs orders; LIVE submits paper orders)
  5. append broker_equity.csv

Flags:
  --once     run a single tick and exit
  --force    bypass the market-hours gate (for dry-run testing)
  --live     submit real paper orders (requires credentials)

Usage:
  python -m live.run_intraday --once --force          # dry-run one tick
  python -m live.run_intraday --once --live           # live one tick (needs keys)
"""
from __future__ import annotations
import sys
import datetime as dt
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

from . import config as C
from . import engine as E
from . import intraday_data as ID
from . import live_book as LB
from . import broker as BR


# ── market gate ──────────────────────────────────────────────────────
def market_open_now() -> bool:
    try:
        et = dt.datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        et = dt.datetime.utcnow() - dt.timedelta(hours=5)
    if et.weekday() >= 5:
        return False
    o = et.replace(hour=C.MARKET_OPEN_ET[0], minute=C.MARKET_OPEN_ET[1], second=0, microsecond=0)
    c = et.replace(hour=C.MARKET_CLOSE_ET[0], minute=C.MARKET_CLOSE_ET[1], second=0, microsecond=0)
    return o <= et <= c


# ── MomAlloc per-book weights from the live track record ─────────────
def book_weights_from_equity() -> dict:
    """Latest 60d-Sharpe weights per book (clip>=0, normalise); equal-weight fallback."""
    if not C.EQUITY_FILE.exists():
        return {b: 0.25 for b in C.BOOKS}
    eq = pd.read_csv(C.EQUITY_FILE, parse_dates=["date"])
    piv = eq.pivot_table(index="date", columns="book", values="daily_ret")
    piv = piv.reindex(columns=C.BOOKS)
    if len(piv) < C.MOM_ALLOC_MIN_DAYS:
        return {b: 0.25 for b in C.BOOKS}
    win = piv.tail(C.MOM_ALLOC_WINDOW)
    sh = (win.mean() / win.std() * np.sqrt(C.TRADING_DAYS)).clip(lower=0).fillna(0)
    tot = sh.sum()
    if tot <= 0:
        return {b: 0.25 for b in C.BOOKS}
    return {b: float(sh.get(b, 0) / tot) for b in C.BOOKS}


def _append_broker_equity(acct, diag, n_orders):
    C.ensure_dirs()
    row = dict(ts=dt.datetime.now().isoformat(timespec="seconds"),
               equity=acct["equity"], cash=acct["cash"],
               connected=acct["live"], gross=diag["gross"],
               n_target=diag["n_target"], n_orders=n_orders,
               nA=diag["n_open"]["A"], nB=diag["n_open"]["B"],
               nC=diag["n_open"]["C"], nD=diag["n_open"]["D"])
    df = pd.DataFrame([row])
    try:
        if C.BROKER_EQUITY_FILE.exists():
            df.to_csv(C.BROKER_EQUITY_FILE, mode="a", header=False, index=False)
        else:
            df.to_csv(C.BROKER_EQUITY_FILE, index=False)
    except PermissionError:
        print("  [intraday] broker_equity.csv locked (concurrent run) — skipped this append.")


def tick(live=False, force=False, verbose=True):
    if not force and not market_open_now():
        if verbose:
            print("  [intraday] market closed -> skip tick.")
        return

    broker = BR.Broker(live=live)
    if verbose:
        print(f"  [intraday] broker: {broker.status_line()}")

    panels = ID.refresh_today(verbose=verbose)
    bw = book_weights_from_equity()
    if verbose:
        print(f"  [intraday] book weights: " +
              ", ".join(f"{b}={bw[b]:.2f}" for b in C.BOOKS))

    acct = broker.get_account()
    book = LB.LiveBook()
    target_w, shares, prices, diag, attribution = book.tick(
        panels, dt.datetime.now(), bw, acct["equity"])
    book.save()

    orders = broker.reconcile(shares, prices, attribution)
    _append_broker_equity(acct, diag, len(orders))

    if verbose:
        longs = sum(1 for v in shares.values() if v > 0)
        shorts = sum(1 for v in shares.values() if v < 0)
        print(f"  [intraday] target: {len(shares)} symbols ({longs} long / {shorts} short), "
              f"gross={diag['gross']:.2f}, open A/B/C/D={diag['n_open']}")
        print(f"  [intraday] reconcile: {len(orders)} orders ({broker.status_line()})")
        for o in orders[:8]:
            print(f"      {o['side'].upper():4} {o['qty']:>5} {o['symbol']:<6} "
                  f"${o['notional']:>10,.0f}  [{o['mode']}]")
        if len(orders) > 8:
            print(f"      ... +{len(orders)-8} more (see {C.ORDERS_LOG})")


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    live = "--live" in argv
    force = "--force" in argv
    once = "--once" in argv or True   # default single tick (scheduler fires us hourly)
    print(f"Intraday tick  (live={live}, force={force})")
    tick(live=live, force=force)
    print("Done.")


if __name__ == "__main__":
    main()
