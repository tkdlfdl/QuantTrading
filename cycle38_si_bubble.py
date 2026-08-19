"""
Cycle 38 — user idea: HIGH short interest x DEEPLY NEGATIVE bubble score
(daily, user's exact formula) -> buy and hold (grid).

Signal (verbatim user spec): fair = price.rolling(252).mean();
residual = log(price) - log(fair); z = (residual - roll252.mean)/roll252.std;
score = tanh(z/2). Signal lagged 1 day.

Grid (all cells reported): SI-elig {top-quintile, bottom-quintile CONTROL} x
bubble thr {-0.6, -0.8} x hold {10, 21, 42}. Family pre-named:
SI-top / thr -0.8 / hold 21. Structure criterion for extra stack tests:
Sharpe > 1.0 and corr < 0.30. Costs 0.1%/side.
Prior: #126 (D formula daily-alone weak 0.28-0.46) — the test is whether SI
fuel resurrects it; the control isolates the SI contribution.
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

# ---- user's bubble-score proxy, verbatim ----
log_price = np.log(px)
fair_value = px.rolling(252).mean()
log_fair = np.log(fair_value)
residual = log_price - log_fair
z = (residual - residual.rolling(252).mean()) / residual.rolling(252).std()
score = np.tanh(z / 2).shift(1)               # lagged 1 day

si = pd.read_parquet("data/cache/short_interest.parquet")
si["date"] = pd.to_datetime(si["date"])
si = si[si["symbol"].isin(stocks)]
shares_o = pd.read_parquet("data/cache/shares_outstanding.parquet")
si_p = si.pivot_table(index="date", columns="symbol", values="si_shares",
                      aggfunc="last").sort_index()
sh_al = shares_o.reindex(columns=si_p.columns).reindex(si_p.index, method="ffill")
si_ratio = (si_p / sh_al).clip(0, 1)
si_daily = si_ratio.reindex(idx, method="ffill").shift(3)
hi_si = si_daily.ge(si_daily.quantile(0.80, axis=1), axis=0)
lo_si = si_daily.le(si_daily.quantile(0.20, axis=1), axis=0)

candidates = []
fam = None
for elig, en in ((hi_si, "SI-TOP"), (lo_si, "SI-LOW ctrl")):
    for thr in (-0.6, -0.8):
        sigm = (score < thr) & elig.reindex(columns=score.columns).fillna(False) \
               & (jump <= 1.0)
        sv = sigm.values
        ev_locs = [(d_i, np.where(sv[d_i])[0]) for d_i in range(len(idx))
                   if idx[d_i] >= pd.Timestamp("2019-03-01") and sv[d_i].any()]
        for hold in (10, 21, 42):
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
            if len(s) < 200 or n_ev < 30:
                print(f"{en}|thr{thr}|h{hold}: too few events ({n_ev})")
                continue
            ms = metrics_from_returns(s.values, TD)
            cc = float(np.corrcoef(s.reindex(champ.index).fillna(0), champ)[0, 1])
            sd = float(s.reindex(worst).fillna(0).mean())
            in_mkt = float((pdf.sum(axis=1) > 0).mean())
            tag = f"{en}|thr{thr}|h{hold}"
            print(f"{tag:22s} ev {n_ev:5d} in-mkt {in_mkt:4.0%} | "
                  f"Sharpe {ms['sharpe']:5.2f} CAGR {ms['cagr']:6.1%} "
                  f"MaxDD {ms['max_dd']:5.0%} | corr {cc:+.2f} | "
                  f"stress {sd:+.2%}/d", flush=True)
            if en == "SI-TOP" and (ms["sharpe"] > 1.0 and cc < 0.30):
                candidates.append((tag, s))
            if en == "SI-TOP" and thr == -0.8 and hold == 21:
                fam = (tag, s)

print(f"\n[{time.time()-t0:.0f}s] stack tests: family + structure-criterion cells")
tested = set()
for tag, s in ([fam] if fam else []) + candidates:
    if tag in tested:
        continue
    tested.add(tag)
    ser = build_port({**FULL, "N": s}, ORDER + ["N"], {**SHARES, "N": .25})
    r = sharpe_delta_test(ser, full_stack)
    m = metrics_from_returns(ser.values, TD)
    sig = "  <-- ADDS" if r["significant_p10"] and r["delta"] > 0 else ""
    print(f"  STACK {tag}: {m['sharpe']:.3f} (delta {r['delta']:+.3f}, "
          f"p {r['p_one_sided']:.3f}){sig}")
print(f"total {time.time()-t0:.0f}s")
