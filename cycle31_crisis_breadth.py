"""
Cycle 31 (manual) — crisis-alpha + rare-event tactical candidates
(registry #160-161). Full gates vs the v4 stack.

 (A) TSMOM defensive: TLT and GLD each long when px > 200d MA (checked at
     month-end, signal lagged 1d), else cash; 50/50; 0.1%/side on switches.
 (B) Breadth thrust: breadth = fraction of universe above 20d MA. Thrust =
     breadth crosses from <0.40 to >0.60 within 10 sessions. Long QQQ for
     63 trading days from the next session. 0.1%/side.
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


def judge(name, stream):
    stream = stream[stream.index <= END].dropna()
    ms = metrics_from_returns(stream.values, TD)
    s19 = stream.reindex(champ.index).fillna(0)
    cc = float(np.corrcoef(s19, champ)[0, 1])
    sd = float(stream.reindex(worst).fillna(0).mean())
    y22 = stream[(stream.index >= "2022-01-01") & (stream.index <= "2022-12-31")]
    r22 = float((1 + y22).prod() - 1) if len(y22) else np.nan
    print(f"  {name}: {stream.index.min().date()}..: Sharpe {ms['sharpe']:.2f} "
          f"CAGR {ms['cagr']:.1%} MaxDD {ms['max_dd']:.0%} | 2022 {r22:+.1%} | "
          f"corr {cc:+.2f} | stress {sd:+.2%}/d")
    ser = build_port({**FULL, "N": stream}, ORDER + ["N"], {**SHARES, "N": .25})
    r = sharpe_delta_test(ser, full_stack)
    m = metrics_from_returns(ser.values, TD)
    sig = "  <-- ADDS ON FULL STACK" if r["significant_p10"] and r["delta"] > 0 else ""
    print(f"    FULL-STACK +N @0.25: {m['sharpe']:.3f} (delta {r['delta']:+.3f}, "
          f"p {r['p_one_sided']:.3f}){sig}")


etf = pd.read_parquet("data/cache/etf_daily_close.parquet")
etf.index = pd.to_datetime(etf.index)
daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)

# ══════════ (A) TSMOM defensive ══════════
print(f"\n=== (A) TSMOM defensive (TLT+GLD 200d trend) ===")
pos = pd.DataFrame(0.0, index=etf.index, columns=["TLT", "GLD"])
month_end = etf.index.to_series().dt.month.diff().fillna(1) != 0
sig = (etf[["TLT", "GLD"]] > etf[["TLT", "GLD"]].rolling(200).mean()).shift(1)
tgt = sig[month_end].astype(float) * 0.5
pos = tgt.reindex(etf.index).ffill().fillna(0.0)
r_etf = etf[["TLT", "GLD"]].pct_change()
sA = ((r_etf.fillna(0) * pos.shift(1)).sum(axis=1)
      - pos.diff().abs().sum(axis=1).fillna(0) * TC)
sA = sA[sA.index >= "2005-06-01"].dropna()
judge("TSMOM defensive", sA)

# ══════════ (B) breadth thrust ══════════
print(f"\n=== (B) breadth thrust ({time.time()-t0:.0f}s) ===")
stocks = [c for c in daily.columns if not c.startswith("^")
          and c not in {"UVXY", "SPY", "QQQ"}]
px = daily[stocks].ffill()
breadth = (px > px.rolling(20).mean()).mean(axis=1)
low = breadth.rolling(10).min()
thrust = (breadth > 0.60) & (low < 0.40)
thrust = thrust & ~thrust.shift(1).fillna(False)      # rising edge only
qqq = daily["QQQ"].dropna() if "QQQ" in daily.columns else daily["SPY"].dropna()
qr = qqq.pct_change()
posB = pd.Series(0.0, index=qqq.index)
# registered intent: RARE events — a thrust is only valid if no position is
# already on (the raw trigger re-fires while breadth oscillates above 0.6)
active_until = -1
th_days = []
for d in thrust[thrust].index:
    loc = qqq.index.searchsorted(d)
    if loc <= active_until:
        continue
    th_days.append(d)
    posB.iloc[loc + 1:min(loc + 64, len(qqq.index))] = 1.0
    active_until = loc + 63
print(f"  thrust events (deduped): {len(th_days)} "
      f"({', '.join(d.strftime('%Y-%m') for d in th_days[-6:])})")
sB = (qr * posB.shift(0) - posB.diff().abs().fillna(0) * TC).dropna()
sB = sB[sB.index >= "1999-06-01"]
print(f"  in-market: {float((posB > 0).mean()):.0%} of days")
judge("breadth thrust", sB)
print(f"\ntotal {time.time()-t0:.0f}s")
