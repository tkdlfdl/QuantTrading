"""
Cycle 24 (d) — Frazzini-Lamont earnings-announcement premium on the cached
universe earnings calendar (data/cache/earnings_dates.parquet).

Long each name from close(T-2) to close(T+1) around its announcement date
(T = first trading day >= calendar date), equal-weight across concurrent
events, 0.1%/side on traded weight, splice-guarded. Only years with >200
events (coverage floor) enter the sample — no approximation on thin years.

Gates: standalone, corr(champ), stress days, full-stack marginal @0.25, LW p.
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
        "DU": load("book_du_live_spec")}
worst = champ.nsmallest(int(len(champ) * 0.05)).index


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


full_stack = build_port(FULL, ["A", "C", "D", "F", "G", "X", "DU"],
                        {"D": 2.0, "G": .25, "X": .25, "DU": .25})

edf = pd.read_parquet("data/cache/earnings_dates.parquet")
edf["date"] = pd.to_datetime(edf["date"])
per_yr = edf.groupby(edf["date"].dt.year).size()
ok_years = per_yr[per_yr > 200]
start_yr = int(ok_years.index.min())
print(f"calendar: {len(edf)} rows, {edf['symbol'].nunique()} tickers; "
      f"coverage years {start_yr}..{int(ok_years.index.max())}")

daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)
stocks = [c for c in daily.columns
          if not c.startswith("^") and c not in {"SPY", "UVXY", "QQQ"}]
px = daily[stocks].ffill()
ret = px.pct_change()
jump = ret.abs().rolling(252).max()

pos = np.zeros((len(px.index), len(stocks)))
col = {s: i for i, s in enumerate(stocks)}
n_ev = 0
for _, row in edf.iterrows():
    d_, s_ = row["date"], row["symbol"]
    if d_.year < start_yr or s_ not in col:
        continue
    loc = px.index.searchsorted(d_)
    if loc < 3 or loc >= len(px.index) - 2:
        continue
    j = jump.iat[loc, col[s_]]
    if np.isfinite(j) and j > 1.0:
        continue
    pos[loc - 1:loc + 2, col[s_]] = 1.0
    n_ev += 1
print(f"events used: {n_ev}  ({time.time()-t0:.0f}s)")
posdf = pd.DataFrame(pos, index=px.index, columns=stocks)
w = posdf.div(posdf.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
strat = ((ret.fillna(0) * w.shift(1)).sum(axis=1)
         - w.diff().abs().sum(axis=1).fillna(0) * TC)
strat = strat[strat.index >= f"{start_yr}-06-01"].dropna()

ms = metrics_from_returns(strat.values, TD)
s19 = strat.reindex(champ.index).fillna(0)
cc = float(np.corrcoef(s19, champ)[0, 1])
sd = float(strat.reindex(worst).fillna(0).mean())
in_mkt = float((posdf.sum(axis=1) > 0).mean())
print(f"standalone {strat.index.min().date()}..: Sharpe {ms['sharpe']:.2f} "
      f"CAGR {ms['cagr']:.1%} MaxDD {ms['max_dd']:.0%} | corr(champ) {cc:+.2f} | "
      f"stress {sd:+.2%}/d | in-market {in_mkt:.0%} of days")
ser = build_port({**FULL, "N": strat},
                 ["A", "C", "D", "F", "G", "X", "DU", "N"],
                 {"D": 2.0, "G": .25, "X": .25, "DU": .25, "N": .25})
r = sharpe_delta_test(ser, full_stack)
m = metrics_from_returns(ser.values, TD)
sig = "  <-- ADDS ON FULL STACK" if r["significant_p10"] and r["delta"] > 0 else ""
print(f"FULL-STACK +N @0.25: {m['sharpe']:.3f} (delta {r['delta']:+.3f}, "
      f"p {r['p_one_sided']:.3f}){sig}")
print(f"total {time.time()-t0:.0f}s")
