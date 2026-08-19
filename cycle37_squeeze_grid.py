"""
Cycle 37 — squeeze-ride structure grid (#175 follow-up, user-directed).
27 cells: eligibility {SIratio q80, SIratio q90, DTC q80} x trigger
{5d>=10%, 5d>=15%, 1d>=5%} x hold {5,10,21}. ALL cells reported.
Stack tests ONLY for cells passing the PRE-STATED structure criterion:
standalone Sharpe > 1.0 AND corr(champ) < 0.30. No argmax adoption; any
adoption still needs rule 4b on the family.
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

daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)
stocks = [c for c in daily.columns if not c.startswith("^")
          and c not in {"UVXY", "SPY", "QQQ"}]
px = daily[stocks].ffill()
ret = px.pct_change()
jump = ret.abs().rolling(252).max()
idx = px.index
col = {s: i for i, s in enumerate(stocks)}

si = pd.read_parquet("data/cache/short_interest.parquet")
si["date"] = pd.to_datetime(si["date"])
si = si[si["symbol"].isin(stocks)]
shares_o = pd.read_parquet("data/cache/shares_outstanding.parquet")
si_p = si.pivot_table(index="date", columns="symbol", values="si_shares",
                      aggfunc="last").sort_index()
dtc_p = si.pivot_table(index="date", columns="symbol", values="dtc",
                       aggfunc="last").sort_index()
sh_al = shares_o.reindex(columns=si_p.columns).reindex(si_p.index, method="ffill")
si_ratio = (si_p / sh_al).clip(0, 1)
si_daily = si_ratio.reindex(idx, method="ffill").shift(3)
dtc_daily = dtc_p.reindex(idx, method="ffill").shift(3)
r5 = px.pct_change(5)
r1 = ret

ELIG = {
    "SIq80": si_daily.ge(si_daily.quantile(0.80, axis=1), axis=0),
    "SIq90": si_daily.ge(si_daily.quantile(0.90, axis=1), axis=0),
    "DTCq80": dtc_daily.ge(dtc_daily.quantile(0.80, axis=1), axis=0),
}
TRIG = {
    "5d10": r5 >= 0.10,
    "5d15": r5 >= 0.15,
    "1d5": r1 >= 0.05,
}

rows = []
candidates = []
for en, emask in ELIG.items():
    for tn, tmask in TRIG.items():
        trig = emask & tmask & (jump <= 1.0)
        tv = trig.values
        ev_locs = [(d_i, np.where(tv[d_i])[0]) for d_i in range(len(idx))
                   if idx[d_i] >= pd.Timestamp("2019-03-01") and tv[d_i].any()]
        for hold in (5, 10, 21):
            pos = np.zeros((len(idx), len(stocks)))
            n_ev = 0
            for d_i, cols_ in ev_locs:
                for c_ in cols_:
                    pos[d_i + 1:min(d_i + hold + 1, len(idx)), c_] = 1.0
                    n_ev += 1
            pdf = pd.DataFrame(pos, index=idx, columns=stocks)
            w = pdf.div(pdf.sum(axis=1).replace(0, np.nan), axis=0) \
                   .fillna(0.0).shift(1)
            s = ((ret.fillna(0) * w).sum(axis=1)
                 - w.diff().abs().sum(axis=1).fillna(0) * TC)
            s = s[(s.index >= "2019-03-01") & (s.index <= END)].dropna()
            ms = metrics_from_returns(s.values, TD)
            cc = float(np.corrcoef(s.reindex(champ.index).fillna(0), champ)[0, 1])
            sd = float(s.reindex(worst).fillna(0).mean())
            tag = f"{en}|{tn}|h{hold}"
            rows.append((tag, n_ev, ms["sharpe"], ms["cagr"], ms["max_dd"], cc, sd))
            if ms["sharpe"] > 1.0 and cc < 0.30:
                candidates.append((tag, s))
            print(f"{tag:16s} ev {n_ev:5d} | Sharpe {ms['sharpe']:5.2f} "
                  f"CAGR {ms['cagr']:6.1%} MaxDD {ms['max_dd']:5.0%} | "
                  f"corr {cc:+.2f} | stress {sd:+.2%}/d", flush=True)

print(f"\nstructure-criterion cells (Sharpe>1.0 & corr<0.30): "
      f"{len(candidates)}  [{time.time()-t0:.0f}s]")
for tag, s in candidates:
    ser = build_port({**FULL, "N": s}, ORDER + ["N"], {**SHARES, "N": .25})
    r = sharpe_delta_test(ser, full_stack)
    m = metrics_from_returns(ser.values, TD)
    sig = "  <-- ADDS" if r["significant_p10"] and r["delta"] > 0 else ""
    print(f"  STACK {tag}: {m['sharpe']:.3f} (delta {r['delta']:+.3f}, "
          f"p {r['p_one_sided']:.3f}){sig}")
print(f"total {time.time()-t0:.0f}s")
