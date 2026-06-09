"""
live/eod_report.py
==================
End-of-day Excel report. Run after the close (or anytime) to record the day and
append to a master workbook that tracks EVERY trading day.

  python -m live.eod_report

Source-of-truth CSVs (appended/updated idempotently per date, in live/state/):
  eod_actions.csv    every buy/sell/exit with book + signal + action type
  eod_book_daily.csv per (date,book): day P&L, cum P&L, allocation, return, Sharpe, MaxDD
  eod_summary.csv    per date: equity, total/day P&L, return, Sharpe, MaxDD, gross
  eod_positions.csv  EOD position snapshot per (date,book,symbol)

The Excel workbook live/reports/track_record.xlsx is regenerated from these CSVs each run,
so it is always consistent and idempotent (re-running the same day overwrites that day's rows).

Per-book P&L uses Alpaca's per-position `unrealized_intraday_pl` (P&L since prior close),
summed per book. Realized P&L from intraday exits is small and shown as the
account-vs-attributed residual in the Summary sheet.
"""
from __future__ import annotations
import sys, json, datetime as dt
import numpy as np
import pandas as pd
from . import config as C

START_EQUITY = 100_000.0
RF = C.RISK_FREE_ANN
TD = C.TRADING_DAYS

EOD_ACTIONS   = C.STATE_DIR / "eod_actions.csv"
EOD_BOOKDAILY = C.STATE_DIR / "eod_book_daily.csv"
EOD_SUMMARY   = C.STATE_DIR / "eod_summary.csv"
EOD_POSITIONS = C.STATE_DIR / "eod_positions.csv"
XLSX          = C.REPORTS_DIR / "track_record.xlsx"

BOOK_LABEL = {"A": "Momentum", "B": "QQQ Bubble", "C": "Intraday MR",
              "D": "Contrarian", "E": "Reddit Sentiment"}


# ─────────────────────────────────────────────────────────────────────
def _client():
    k, s = C.load_alpaca_creds()
    if not (k and s):
        print("No Alpaca credentials — cannot build EOD report.")
        sys.exit(1)
    from alpaca.trading.client import TradingClient
    return TradingClient(k, s, paper=True)


def _book_map():
    m = {}
    if C.LIVE_POSITIONS_FILE.exists():
        b = json.loads(C.LIVE_POSITIONS_FILE.read_text(encoding="utf-8")).get("books", {})
        for s in b.get("A", {}).get("longs", []):  m[s] = "A"
        for s in b.get("A", {}).get("shorts", []): m[s] = "A"
        for x in b.get("B", []): m[x["symbol"]] = "B"
        for x in b.get("C", []): m[x["symbol"]] = "C"
        for x in b.get("D", []): m[x["symbol"]] = "D"
        for s in b.get("E", {}).get("longs", []): m[s] = "E"
    return m


def _book_weights():
    """Current MomAlloc allocation per book (from the simulated track record)."""
    if not C.EQUITY_FILE.exists():
        return {b: 0.25 for b in C.BOOKS}
    eq = pd.read_csv(C.EQUITY_FILE, parse_dates=["date"])
    piv = eq.pivot_table(index="date", columns="book", values="daily_ret").reindex(columns=C.BOOKS)
    if len(piv) < C.MOM_ALLOC_MIN_DAYS:
        return {b: 0.25 for b in C.BOOKS}
    win = piv.tail(C.MOM_ALLOC_WINDOW)
    sh = (win.mean() / win.std() * np.sqrt(TD)).clip(lower=0).fillna(0)
    tot = sh.sum()
    return {b: float(sh.get(b, 0) / tot) for b in C.BOOKS} if tot > 0 else {b: 0.25 for b in C.BOOKS}


def _upsert(path, df_new, key_cols):
    """Append df_new to the CSV, replacing any existing rows matching key_cols values."""
    if path.exists():
        old = pd.read_csv(path)
        keyvals = set(map(tuple, df_new[key_cols].astype(str).values))
        mask = ~old[key_cols].astype(str).apply(tuple, axis=1).isin(keyvals)
        out = pd.concat([old[mask], df_new], ignore_index=True)
    else:
        out = df_new
    out.to_csv(path, index=False)
    return out


def _metrics(daily_ret: pd.Series):
    r = daily_ret.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0, 0.0
    sh = (r - RF / TD).mean() / r.std() * np.sqrt(TD)
    w = (1 + r).cumprod()
    dd = (w / w.cummax() - 1).min()
    return float(sh), float(dd)


# ─────────────────────────────────────────────────────────────────────
def build(today=None):
    today = today or dt.date.today().isoformat()
    cli = _client()
    acct = cli.get_account()
    equity = float(acct.equity); last_eq = float(acct.last_equity)
    positions = cli.get_all_positions()
    bm = _book_map()
    weights = _book_weights()

    # ── 1. ACTIONS (today's orders, enriched) ────────────────────────
    acts = []
    if C.ORDERS_LOG.exists():
        for line in C.ORDERS_LOG.read_text(encoding="utf-8").splitlines():
            try:
                o = json.loads(line)
            except Exception:
                continue
            if not o.get("ts", "").startswith(today):
                continue
            acts.append(dict(
                date=today, time=o["ts"][11:19], book=o.get("book", "?"),
                symbol=o["symbol"], action=o.get("action", ""),
                side=o["side"].upper(), qty=o["qty"], price=o.get("price", 0),
                notional=o.get("notional", 0), mode=o.get("mode", ""),
                signal=o.get("signal"), signal_desc=o.get("signal_desc", "")))
    if acts:
        _upsert(EOD_ACTIONS, pd.DataFrame(acts), ["date", "time", "symbol", "action"])

    # ── 2. POSITIONS snapshot + per-book day P&L ─────────────────────
    pos_rows, book_pnl, book_notional, book_npos = [], {}, {}, {}
    for p in positions:
        b = bm.get(p.symbol, "?")
        side = (p.side.value if hasattr(p.side, "value") else str(p.side)).split(".")[-1]
        intraday = float(getattr(p, "unrealized_intraday_pl", 0) or 0)
        mv = float(p.market_value)
        pos_rows.append(dict(date=today, book=b, symbol=p.symbol, side=side, qty=p.qty,
                             mkt_value=round(mv, 2), unreal_pnl=round(float(p.unrealized_pl), 2),
                             day_pnl=round(intraday, 2)))
        book_pnl[b] = book_pnl.get(b, 0.0) + intraday
        book_notional[b] = book_notional.get(b, 0.0) + abs(mv)
        book_npos[b] = book_npos.get(b, 0) + 1
    if pos_rows:
        _upsert(EOD_POSITIONS, pd.DataFrame(pos_rows), ["date", "symbol"])

    # ── 3. BOOK DAILY (append today, then compute return/Sharpe/MaxDD over history) ──
    bd_today = []
    for b in C.BOOKS:
        notional = book_notional.get(b, 0.0)
        pnl = book_pnl.get(b, 0.0)
        ret = (pnl / notional) if notional > 0 else 0.0
        bd_today.append(dict(date=today, book=b, day_pnl=round(pnl, 2),
                             n_pos=book_npos.get(b, 0), notional=round(notional, 2),
                             alloc_weight=round(weights.get(b, 0), 4), day_ret=ret))
    bd = _upsert(EOD_BOOKDAILY, pd.DataFrame(bd_today), ["date", "book"])
    bd["date"] = pd.to_datetime(bd["date"])
    bd = bd.sort_values(["book", "date"])
    # cumulative + metrics per book over the accumulating series
    bd["cum_pnl"] = bd.groupby("book")["day_pnl"].cumsum()
    sh_map, dd_map = {}, {}
    for b, g in bd.groupby("book"):
        sh_map[b], dd_map[b] = _metrics(g.set_index("date")["day_ret"])
    bd["sharpe_itd"] = bd["book"].map(sh_map)
    bd["maxdd_itd"]  = bd["book"].map(dd_map)
    bd["date"] = bd["date"].dt.date.astype(str)
    bd.to_csv(EOD_BOOKDAILY, index=False)

    # ── 4. SUMMARY (account level) ───────────────────────────────────
    attributed = sum(book_pnl.values())
    gross = sum(book_notional.values())
    day_pnl = equity - last_eq
    summ_today = pd.DataFrame([dict(
        date=today, equity=round(equity, 2),
        total_pnl=round(equity - START_EQUITY, 2),
        total_ret=round((equity / START_EQUITY - 1) * 100, 3),
        day_pnl=round(day_pnl, 2),
        day_ret=round((equity / last_eq - 1) * 100, 3) if last_eq else 0,
        gross_exposure=round(gross, 2),
        gross_pct=round(gross / equity * 100, 1) if equity else 0,
        cash=round(float(acct.cash), 2),
        attributed_book_pnl=round(attributed, 2),
        unattributed=round(day_pnl - attributed, 2))])
    sm = _upsert(EOD_SUMMARY, summ_today, ["date"])
    sm["date"] = pd.to_datetime(sm["date"])
    sm = sm.sort_values("date")
    sh, dd = _metrics(sm.set_index("date")["day_ret"] / 100.0)
    sm["sharpe_itd"] = round(sh, 4)
    sm["maxdd_itd_pct"] = round(dd * 100, 2)
    sm["date"] = sm["date"].dt.date.astype(str)
    sm.to_csv(EOD_SUMMARY, index=False)

    # ── 5. WRITE EXCEL ───────────────────────────────────────────────
    _write_excel()
    print(f"EOD report for {today}:")
    print(f"  Equity ${equity:,.2f}  Day P&L ${day_pnl:+,.2f}  Total ${equity-START_EQUITY:+,.2f} "
          f"({(equity/START_EQUITY-1)*100:+.2f}%)")
    for b in C.BOOKS:
        if book_npos.get(b):
            print(f"  Book {b} ({BOOK_LABEL[b]}): day P&L ${book_pnl.get(b,0):+,.0f}  "
                  f"{book_npos[b]} pos  alloc {weights.get(b,0)*100:.0f}%  "
                  f"Sharpe {sh_map.get(b,0):.2f}  MaxDD {dd_map.get(b,0)*100:.1f}%")
    print(f"  Actions logged today: {len(acts)}")
    print(f"  Workbook: {XLSX}")


def _write_excel():
    def _rd(p):
        return pd.read_csv(p) if p.exists() else pd.DataFrame()
    summary = _rd(EOD_SUMMARY)
    bookdaily = _rd(EOD_BOOKDAILY)
    actions = _rd(EOD_ACTIONS)
    positions = _rd(EOD_POSITIONS)

    # latest per-book performance pivot for quick scan
    perf = pd.DataFrame()
    if not bookdaily.empty:
        bookdaily["book_label"] = bookdaily["book"].map(BOOK_LABEL).fillna(bookdaily["book"])
    if not actions.empty:
        actions = actions.sort_values(["date", "time"], ascending=[False, False])

    with pd.ExcelWriter(XLSX, engine="openpyxl") as xw:
        (summary.sort_values("date", ascending=False) if not summary.empty else summary
         ).to_excel(xw, sheet_name="Summary", index=False)
        (bookdaily.sort_values(["date", "book"], ascending=[False, True]) if not bookdaily.empty
         else bookdaily).to_excel(xw, sheet_name="BookPerformance", index=False)
        actions.to_excel(xw, sheet_name="Actions", index=False)
        (positions.sort_values(["date", "book"], ascending=[False, True]) if not positions.empty
         else positions).to_excel(xw, sheet_name="Positions", index=False)


def main(argv=None):
    build()


if __name__ == "__main__":
    main()
