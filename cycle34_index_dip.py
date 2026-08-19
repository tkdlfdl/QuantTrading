"""
Cycle 34 — index-level dip-buying (user idea; registry #168).
All specs pre-registered; every cell reported. Costs 0.1%/side.
Judge: standalone + FULL-STACK marginal @0.25 for the strongest
pre-registered cell FAMILY (not argmax cell — the family verdict).
"""
from __future__ import annotations
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from tools.metrics import metrics_from_returns
from tools.significance import sharpe_delta_test
from live import engine as E, config as C

TD, TC, END = 252, 0.001, "2026-07-08"
t0 = time.time()

def load(n):
    df = pd.read_csv(f"strategies/performance/{n}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)

champ = load("portfolio_champ_d14")
d8q, d14 = load("book_d8_quiet"), load("book_d14")
ju = d8q.index.union(d14.index)
dblend = 0.5 * d8q.reindex(ju).fillna(0) + 0.5 * d14.reindex(ju).fillna(0)
FULL = {"A": load("book_a_retest"), "C": load("book_c_overlap_cap"),
        "D": dblend, "F": load("book_f_retest"),
        "G": load("book_g_live_spec"), "X": load("book_x_live_spec"),
        "DU": load("book_du_live_spec"), "FD": load("book_fdip_40h")}
worst = champ.nsmallest(int(len(champ) * 0.05)).index
SHARES = {"D": 2.0, "G": .25, "X": .25, "DU": .25, "FD": .25}
ORDER = ["A", "C", "D", "F", "G", "X", "DU", "FD"]


def build_port(books, alloc, shares):
    sb, ss = C.ALLOC_BOOKS, dict(C.ALLOC_SHARES)
    C.ALLOC_BOOKS = alloc
    C.ALLOC_SHARES = shares
    try:
        R = pd.DataFrame(books)
        R = R[(R.index >= "2019-01-02") & (R.index <= END)].fillna(0.0)
        return E.ivol_voltgt(R).dropna()
    finally:
        C.ALLOC_BOOKS, C.ALLOC_SHARES = sb, ss


full_stack = build_port(FULL, ORDER, SHARES)
print(f"full stack (v4): {metrics_from_returns(full_stack.values, TD)['sharpe']:.3f}")


def stats(name, s, do_stack=False):
    s = s[s.index <= END].dropna()
    ms = metrics_from_returns(s.values, TD)
    s19 = s.reindex(champ.index).fillna(0)
    cc = float(np.corrcoef(s19, champ)[0, 1])
    sd = float(s.reindex(worst).fillna(0).mean())
    line = (f"  {name}: {s.index.min().date()}..: Sharpe {ms['sharpe']:.2f} "
            f"CAGR {ms['cagr']:.1%} MaxDD {ms['max_dd']:.0%} | corr {cc:+.2f} "
            f"| stress {sd:+.2%}/d")
    if do_stack:
        ser = build_port({**FULL, "N": s}, ORDER + ["N"], {**SHARES, "N": .25})
        r = sharpe_delta_test(ser, full_stack)
        m = metrics_from_returns(ser.values, TD)
        sig = "  <-- ADDS" if r["significant_p10"] and r["delta"] > 0 else ""
        line += (f"\n    FULL-STACK +N @0.25: {m['sharpe']:.3f} "
                 f"(delta {r['delta']:+.3f}, p {r['p_one_sided']:.3f}){sig}")
    print(line, flush=True)


daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)

# ══════════ A1: daily dip-buy ══════════
print("\n=== A1 daily index dip-buy (hold 10d, overlaps extend) ===")
for sym in ("SPY", "QQQ"):
    px = daily[sym].dropna()
    r = px.pct_change()
    for thr in (-0.02, -0.03):
        drop = r <= thr
        arr = np.zeros(len(px))
        di = np.where(drop.values)[0]
        for i in di:
            arr[i + 1:min(i + 11, len(arr))] = 1.0
        pos = pd.Series(arr, index=px.index)
        s = (r * pos.shift(0) - pos.diff().abs().fillna(0) * TC).dropna()
        s = s[s.index >= "1997-06-01"]
        n_ev = int(drop.sum())
        stats(f"{sym} drop<={thr:.0%} ({n_ev} events, "
              f"in-mkt {pos.mean():.0%})", s)

# ══════════ A2: hourly sudden drop (QQQ) ══════════
print(f"\n=== A2 hourly sudden-drop QQQ (-1.5%/1h -> hold 24h) ({time.time()-t0:.0f}s) ===")
try:
    q = pd.read_parquet("data/cache/qqq_hourly_close.parquet")["QQQ"]
    q.index = pd.to_datetime(q.index).floor("h")
    q = q[~q.index.duplicated(keep="last")].dropna()
    qr = q.pct_change()
    drop_h = qr <= -0.015
    posh = np.zeros(len(q))
    for i in np.where(drop_h.values)[0]:
        posh[i + 1:min(i + 25, len(posh))] = 1.0
    posh = pd.Series(posh, index=q.index)
    sh = (qr * posh - posh.diff().abs().fillna(0) * TC).dropna()
    # aggregate to daily for judging
    sd_ = (1 + sh).groupby(sh.index.normalize()).prod() - 1
    sd_.index = pd.to_datetime(sd_.index)
    print(f"  events: {int(drop_h.sum())}, window {q.index.min().date()}..")
    stats("QQQ hourly dip", sd_)
except Exception as e:
    print(f"  hourly variant failed: {e}")

# ══════════ B: FOMC-drop conditional ══════════
print(f"\n=== B FOMC-drop conditional ({time.time()-t0:.0f}s) ===")
fomc = pd.read_csv("data/cache/fomc_dates.csv", parse_dates=["date"])
spy = daily["SPY"].dropna()
r = spy.pct_change()
fd = set(fomc["date"].dt.normalize())
is_f = pd.Series([d.normalize() in fd for d in spy.index], index=spy.index)
for cond, lbl in ((r <= -0.01, "FOMC-day <= -1% (dip)"),
                  (r >= 0.01, "FOMC-day >= +1% (control)")):
    ev = is_f & cond
    n_ev = int(ev.sum())
    for hold in (5, 10):
        pos = np.zeros(len(spy))
        for i in np.where(ev.values)[0]:
            pos[i + 1:min(i + hold + 1, len(pos))] = 1.0
        pos = pd.Series(pos, index=spy.index)
        s = (r * pos - pos.diff().abs().fillna(0) * TC).dropna()
        s = s[s.index >= "1997-06-01"]
        stats(f"{lbl} hold {hold}d ({n_ev} ev, in-mkt {pos.mean():.0%})", s)

# family stack test: strongest PRE-REGISTERED family = QQQ -3% daily (deepest
# dips, the user's core idea) — run its stack marginal
print(f"\n=== family stack test: QQQ <=-3% hold 10d ({time.time()-t0:.0f}s) ===")
px = daily["QQQ"].dropna()
r = px.pct_change()
drop = r <= -0.03
pos = np.zeros(len(px))
for i in np.where(drop.values)[0]:
    pos[i + 1:min(i + 11, len(pos))] = 1.0
pos = pd.Series(pos, index=px.index)
s = (r * pos - pos.diff().abs().fillna(0) * TC).dropna()
s = s[s.index >= "1999-06-01"]
stats("QQQ deep-dip family", s, do_stack=True)
print(f"total {time.time()-t0:.0f}s")
