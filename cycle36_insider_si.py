"""
Cycle 36 — insider + short-interest data classes (queue #49/#50) and the
user's SHORT-SQUEEZE ride idea. All specs pre-registered; all cells reported.

 (A) INSIDER cluster-buy drift (Cohen-Malloy-Pomorski-lite): >=2 officer/
     director open-market buys (code P) within 21d for a name -> enter close
     T+1, hold 63d, equal-weight open events. Universe-filtered.
 (B) LOW-SI tilt (Rapach-Ringgenberg-Zhou cross-sec): at each settlement
     date, long the 50 LOWEST short-ratio names (SI shares / shares
     outstanding), rebalance per settlement (~bi-monthly).
 (C) SHORT-SQUEEZE ride (user idea): eligibility = TOP-quintile short ratio
     at latest settlement; trigger = 5d return >= +10% (ignition); enter
     close of trigger day, hold {5,10}d. Family pre-named: hold 10.
     Control cell: same trigger on BOTTOM-quintile SI names (if the "squeeze"
     return is just momentum ignition, control matches; if the SHORT
     COVERING is the fuel, high-SI must beat low-SI).

Costs 0.1%/side. Gates: standalone, corr, stress, FULL-STACK @0.25, LW p.
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


def judge(name, stream, do_stack=True):
    stream = stream[stream.index <= END].dropna()
    if len(stream) < 200:
        print(f"  {name}: too short ({len(stream)})")
        return
    ms = metrics_from_returns(stream.values, TD)
    s19 = stream.reindex(champ.index).fillna(0)
    cc = float(np.corrcoef(s19, champ)[0, 1])
    sd = float(stream.reindex(worst).fillna(0).mean())
    print(f"  {name}: {stream.index.min().date()}..: Sharpe {ms['sharpe']:.2f} "
          f"CAGR {ms['cagr']:.1%} MaxDD {ms['max_dd']:.0%} | corr {cc:+.2f} | "
          f"stress {sd:+.2%}/d")
    if do_stack:
        ser = build_port({**FULL, "N": stream}, ORDER + ["N"], {**SHARES, "N": .25})
        r = sharpe_delta_test(ser, full_stack)
        m = metrics_from_returns(ser.values, TD)
        sig = "  <-- ADDS" if r["significant_p10"] and r["delta"] > 0 else ""
        print(f"    FULL-STACK +N @0.25: {m['sharpe']:.3f} "
              f"(delta {r['delta']:+.3f}, p {r['p_one_sided']:.3f}){sig}")


daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)
stocks = [c for c in daily.columns if not c.startswith("^")
          and c not in {"UVXY", "SPY", "QQQ"}]
px = daily[stocks].ffill()
ret = px.pct_change()
jump = ret.abs().rolling(252).max()
idx = px.index
col = {s: i for i, s in enumerate(stocks)}

# ══════════ (A) insider cluster buys ══════════
print("\n=== (A) insider cluster-buy drift ===")
ins = pd.read_parquet("data/cache/insider_purchases.parquet")
ins = ins[ins["is_officer_dir"] & ins["symbol"].isin(stocks)]
ins["trans_date"] = pd.to_datetime(ins["trans_date"])
ins = ins[(ins["trans_date"] >= "2018-06-01") & (ins["trans_date"] <= END)]
print(f"  officer/director buys in universe: {len(ins)} "
      f"({ins['symbol'].nunique()} names)")
cnt = (ins.groupby(["symbol", "trans_date"]).size().rename("n")
       .reset_index())
cnt_p = cnt.pivot_table(index="trans_date", columns="symbol", values="n",
                        aggfunc="sum").reindex(idx).fillna(0.0)
roll = cnt_p.rolling(21).sum()
cluster = (roll >= 2) & (cnt_p > 0)          # a buy today + >=2 in 21d
pos = np.zeros((len(idx), len(stocks)))
n_ev = 0
for d_i, d in enumerate(idx):
    if d < pd.Timestamp("2018-07-01"):
        continue
    row = cluster.iloc[d_i]
    for s_ in row.index[row]:
        if s_ not in col:
            continue
        if np.isfinite(jump.iat[d_i, col[s_]]) and jump.iat[d_i, col[s_]] > 1.0:
            continue
        pos[d_i + 2:min(d_i + 65, len(idx)), col[s_]] = 1.0
        n_ev += 1
print(f"  cluster events: {n_ev}")
posdf = pd.DataFrame(pos, index=idx, columns=stocks)
w = posdf.div(posdf.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
sA = ((ret.fillna(0) * w.shift(1)).sum(axis=1)
      - w.diff().abs().sum(axis=1).fillna(0) * TC)
sA = sA[sA.index >= "2018-08-01"].dropna()
judge("insider cluster (>=2 in 21d, hold 63d)", sA)

# ══════════ SI panel prep ══════════
print(f"\n=== SI panel ({time.time()-t0:.0f}s) ===")
si = pd.read_parquet("data/cache/short_interest.parquet")
si["date"] = pd.to_datetime(si["date"])
si = si[si["symbol"].isin(stocks)]
shares = pd.read_parquet("data/cache/shares_outstanding.parquet")
si_p = si.pivot_table(index="date", columns="symbol", values="si_shares",
                      aggfunc="last").sort_index()
sh_al = shares.reindex(columns=si_p.columns).reindex(si_p.index, method="ffill")
si_ratio = (si_p / sh_al).clip(0, 1)
print(f"  settlements: {len(si_p)}, names covered: "
      f"{si_ratio.notna().sum(axis=1).median():.0f}/settle")

# ══════════ (B) low-SI tilt ══════════
print("\n=== (B) low-SI tilt (50 lowest short-ratio) ===")
posB = pd.DataFrame(0.0, index=idx, columns=stocks)
settles = list(si_ratio.index)
for k, d in enumerate(settles):
    row = si_ratio.loc[d].dropna()
    if len(row) < 200:
        continue
    lows = row.nsmallest(50).index
    loc = idx.searchsorted(d) + 3            # publication lag ~2 business days
    until = idx.searchsorted(settles[k + 1]) + 3 if k + 1 < len(settles) else len(idx) - 1
    cols = [posB.columns.get_loc(c) for c in lows if c in posB.columns]
    posB.iloc[loc:min(until, len(idx)), cols] = 1.0 / 50
w = posB.shift(1)
sB = ((ret.fillna(0) * w).sum(axis=1)
      - w.diff().abs().sum(axis=1).fillna(0) * TC)
sB = sB[sB.index >= "2019-03-01"].dropna()
judge("low-SI tilt", sB)

# ══════════ (C) short-squeeze ride ══════════
print(f"\n=== (C) short-squeeze ride ({time.time()-t0:.0f}s) ===")
si_daily = si_ratio.reindex(idx, method="ffill").shift(3)   # publication lag
r5 = px.pct_change(5)
for quint, lbl in ((0.8, "TOP-quintile SI (squeeze)"),
                   (0.2, "BOTTOM-quintile SI (control)")):
    for hold in (5, 10):
        posC = np.zeros((len(idx), len(stocks)))
        n_ev = 0
        q_th = si_daily.quantile(quint, axis=1) if quint == 0.8 else None
        hi = (si_daily.ge(si_daily.quantile(0.8, axis=1), axis=0)
              if quint == 0.8
              else si_daily.le(si_daily.quantile(0.2, axis=1), axis=0))
        trig = (r5 >= 0.10) & hi & (jump <= 1.0)
        for d_i in range(len(idx)):
            if idx[d_i] < pd.Timestamp("2019-03-01"):
                continue
            row = trig.iloc[d_i]
            for s_ in row.index[row]:
                posC[d_i + 1:min(d_i + hold + 1, len(idx)), col[s_]] = 1.0
                n_ev += 1
        pC = pd.DataFrame(posC, index=idx, columns=stocks)
        w = pC.div(pC.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).shift(1)
        sC = ((ret.fillna(0) * w).sum(axis=1)
              - w.diff().abs().sum(axis=1).fillna(0) * TC)
        sC = sC[sC.index >= "2019-03-01"].dropna()
        fam = (quint == 0.8 and hold == 10)
        print(f"  -- {lbl} hold {hold}d ({n_ev} entries):")
        judge(f"{lbl} h{hold}", sC, do_stack=fam)
print(f"\ntotal {time.time()-t0:.0f}s")
