"""
live/pnl.py
===========
Quick live P&L / positions snapshot from the Alpaca paper account.

Usage:
  python -m live.pnl              # summary + positions by strategy
  python -m live.pnl --short      # one-line summary only
  python -m live.pnl --orders     # also list recent orders
"""
from __future__ import annotations
import sys
import json
from . import config as C

START_EQUITY = 100_000.0


def _client():
    k, s = C.load_alpaca_creds()
    if not (k and s):
        print("No Alpaca credentials found (live/state/alpaca_creds.json or env vars).")
        sys.exit(1)
    from alpaca.trading.client import TradingClient
    return TradingClient(k, s, paper=True)


def _book_map():
    """symbol -> strategy book, from the live position book state."""
    m = {}
    if C.LIVE_POSITIONS_FILE.exists():
        st = json.loads(C.LIVE_POSITIONS_FILE.read_text(encoding="utf-8"))
        b = st.get("books", {})
        for s in b.get("A", {}).get("longs", []):  m[s] = "A"
        for s in b.get("A", {}).get("shorts", []): m[s] = "A"
        for x in b.get("B", []): m[x["symbol"]] = "B"
        for x in b.get("C", []): m[x["symbol"]] = "C"
        for x in b.get("D", []): m[x["symbol"]] = "D"
    return m


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    cli = _client()
    a = cli.get_account()
    eq = float(a.equity); leq = float(a.last_equity)
    pos = cli.get_all_positions()
    unreal = sum(float(p.unrealized_pl) for p in pos)
    invested = sum(abs(float(p.market_value)) for p in pos)

    tot_pl = eq - START_EQUITY
    tot_ret = (eq / START_EQUITY - 1) * 100
    day_pl = eq - leq
    day_ret = (eq / leq - 1) * 100 if leq else 0.0

    if "--short" in argv:
        print("Equity ${:,.0f}  |  Total {:+,.0f} ({:+.2f}%)  |  Day {:+,.0f} ({:+.2f}%)  |  "
              "{} positions".format(eq, tot_pl, tot_ret, day_pl, day_ret, len(pos)))
        return

    print("=" * 46)
    print("  ALPACA PAPER P&L")
    print("=" * 46)
    print("  Equity:        {:>14,.2f}".format(eq))
    print("  Total P&L:     {:>+14,.2f}   ({:+.2f}%)".format(tot_pl, tot_ret))
    print("  Day P&L:       {:>+14,.2f}   ({:+.2f}%)".format(day_pl, day_ret))
    print("  Unrealized:    {:>+14,.2f}".format(unreal))
    print("  Cash:          {:>14,.2f}".format(float(a.cash)))
    print("  Invested:      {:>14,.2f}   ({:.0f}% of equity)".format(invested, invested / eq * 100))

    if not pos:
        print("\n  No open positions.")
        return

    bm = _book_map()
    print("\n  POSITIONS ({}) — by strategy".format(len(pos)))
    print("  {:<6}{:>4}{:>5}{:>10}{:>9}{:>8}".format("Sym", "Bk", "Qty", "MktVal", "P&L", "P&L%"))
    print("  " + "-" * 40)
    for p in sorted(pos, key=lambda x: float(x.unrealized_pl)):
        print("  {:<6}{:>4}{:>5}{:>10,.0f}{:>+9,.0f}{:>+7.1f}%".format(
            p.symbol, bm.get(p.symbol, "?"), p.qty,
            float(p.market_value), float(p.unrealized_pl),
            float(p.unrealized_plpc) * 100))

    # P&L by book
    by = {}
    for p in pos:
        b = bm.get(p.symbol, "?")
        by[b] = by.get(b, 0.0) + float(p.unrealized_pl)
    print("  " + "-" * 40)
    labels = {"A": "Momentum", "B": "QQQ Bubble", "C": "Intraday MR",
              "D": "Contrarian", "?": "Unattributed"}
    print("  P&L by book:")
    for b in sorted(by):
        print("    {} ({:<11}): {:>+9,.0f}".format(b, labels.get(b, b), by[b]))

    if "--orders" in argv:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        from collections import Counter
        orders = cli.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=100))
        print("\n  ORDERS: {} total  {}".format(
            len(orders), dict(Counter(str(o.status).split(".")[-1] for o in orders))))


if __name__ == "__main__":
    main()
