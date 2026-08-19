"""
Cycle 33 — the first Book-F refinement battery (registry #166).

Reconstruction: top-5 by 750h return, hold 200h, non-overlapping blocks,
equal_weight_daily_pnl kernel (validated exact vs official engines), one-way
TC 0.001 spread over hold (research engine convention), splice guard
(trailing-year >100% move exclusion, live replay_F parity).

ANCHOR: reconstruction must match recorded book_f_retest on 2019+ window
(+/-0.05 Sharpe, corr >= 0.99) or VOID.

Variants (pre-registered):
 (i)   TRANCHE-2: 0.5 x (offset-0 + offset-100h) streams.
 (ii)  RISK-ADJ: rank by 750h return / 750h daily-vol.
 (iii) SKIP-WEEK: rank by 750h return excluding last 35h.
Judge: champion swap test (F -> variant) on quiet-champion config + LW p;
winners also checked on the full v4 stack.
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

TD, END = 252, "2026-07-08"
t0 = time.time()

def load(n):
    df = pd.read_csv(f"strategies/performance/{n}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)

hc = pd.read_parquet("data/cache/merged_hourly_close.parquet")
hc.index = pd.to_datetime(hc.index)
ho = pd.read_parquet("data/cache/merged_hourly_open.parquet")
ho.index = pd.to_datetime(ho.index)
common = sorted((set(ho.columns) & set(hc.columns)) - {"SPY"})
hcc, hoo = hc[common].ffill(), ho[common].ffill()
idx = hcc.index
prices = hcc.values.astype(np.float64)
opens = hoo.values.astype(np.float64)
T, U = prices.shape

# splice guard: exclude names with any 1-day |move|>100% in trailing year
day = idx.normalize()
dclose = hcc.groupby(day).last()
jump_d = dclose.pct_change().abs().rolling(252).max()
jump_bars = jump_d.reindex(day).values

bar_day = idx.normalize().values
tdays = np.unique(bar_day)
d2i = {d: i for i, d in enumerate(tdays)}
bdi = np.array([d2i[d] for d in bar_day], dtype=np.int32)
D = len(tdays)
day_last = np.zeros(D, dtype=np.int32)
day_first = np.zeros(D, dtype=np.int32)
for t in range(T):
    day_last[bdi[t]] = t
for t in range(T - 1, -1, -1):
    day_first[bdi[t]] = t
daily_close = prices[day_last]
dret = np.zeros_like(daily_close)
dret[1:] = daily_close[1:] / np.maximum(daily_close[:-1], 1e-8) - 1
dret = np.clip(dret, -0.20, 0.20)

LOOK, HOLD, TOPN = 750, 200, 5
mom = hcc.pct_change(LOOK).values
ret_h = hcc.pct_change()
vol750 = ret_h.rolling(LOOK).std().values
mom_skip = hcc.pct_change(LOOK - 35).shift(35).values


def run_f(score, offset=0):
    sc = score.copy()
    sc[~np.isfinite(sc)] = -np.inf
    sc[jump_bars > 1.0] = -np.inf
    trades = []
    i = LOOK + 5 + offset
    while i < T - 1:
        row = sc[i]
        if np.isfinite(row).sum() >= 50 and (row > -np.inf).sum() >= TOPN:
            top = np.argpartition(-row, TOPN - 1)[:TOPN]
            eb, xb = i + 1, min(i + HOLD, T - 1)
            trades.append((eb, xb, list(top), +1))
            i += HOLD
        else:
            i += 1
    port = E.equal_weight_daily_pnl(trades, prices, opens, idx, day_last,
                                    day_first, bdi, dret, U, 0.001, HOLD)
    s = pd.Series(port, index=pd.to_datetime(tdays)).dropna()
    return s[(s.index >= "2019-01-02") & (s.index <= END)]


# ---- anchor ----
f_rec = run_f(mom)
off = load("book_f_retest")
off = off[(off.index >= "2019-01-02") & (off.index <= END)].dropna()
m_r = metrics_from_returns(f_rec.values, TD)
m_o = metrics_from_returns(off.values, TD)
jj = f_rec.index.intersection(off.index)
cc = float(np.corrcoef(f_rec.loc[jj], off.loc[jj])[0, 1])
print(f"ANCHOR F: reconstructed {m_r['sharpe']:.3f} vs official {m_o['sharpe']:.3f} "
      f"(diff {m_r['sharpe']-m_o['sharpe']:+.3f}, corr {cc:.4f})")
if abs(m_r["sharpe"] - m_o["sharpe"]) > 0.05 or cc < 0.99:
    print("ANCHOR FAILED — VOID")
    sys.exit(1)

# ---- champion machinery ----
d8q, d14 = load("book_d8_quiet"), load("book_d14")
ju = d8q.index.union(d14.index)
dblend = 0.5 * d8q.reindex(ju).fillna(0) + 0.5 * d14.reindex(ju).fillna(0)
BOOKS = {"A": load("book_a_retest"), "C": load("book_c_overlap_cap"),
         "D": dblend, "F": off}


def champ_with(f_ser):
    sb, ss = C.ALLOC_BOOKS, dict(C.ALLOC_SHARES)
    C.ALLOC_BOOKS = ["A", "C", "D", "F"]
    C.ALLOC_SHARES = {"D": 2.0}
    try:
        R = pd.DataFrame({**BOOKS, "F": f_ser})
        R = R[(R.index >= "2019-01-02") & (R.index <= END)].fillna(0.0)
        return E.ivol_voltgt(R).dropna()
    finally:
        C.ALLOC_BOOKS, C.ALLOC_SHARES = sb, ss


champ_base = champ_with(off)
print(f"champion base: {metrics_from_returns(champ_base.values, TD)['sharpe']:.3f}")

variants = {}
tr0 = f_rec
tr100 = run_f(mom, offset=100)
juT = tr0.index.union(tr100.index)
variants["(i) tranche-2"] = (0.5 * tr0.reindex(juT).fillna(0)
                             + 0.5 * tr100.reindex(juT).fillna(0))
variants["(ii) risk-adj"] = run_f(mom / (vol750 + 1e-9))
variants["(iii) skip-week"] = run_f(mom_skip)

for name, s_ in variants.items():
    m_ = metrics_from_returns(s_.dropna().values, TD)
    ser = champ_with(s_)
    r = sharpe_delta_test(ser, champ_base)
    m = metrics_from_returns(ser.values, TD)
    sig = "  <-- SIGNIFICANT" if r["significant_p10"] and r["delta"] > 0 else ""
    print(f"{name}: standalone {m_['sharpe']:.2f}/{m_['max_dd']:.0%} "
          f"(raw F {m_o['sharpe']:.2f}/{m_o['max_dd']:.0%}) | champ "
          f"{m['sharpe']:.3f} (delta {r['delta']:+.3f}, p {r['p_one_sided']:.3f})"
          f"{sig}  [{time.time()-t0:.0f}s]")
print(f"total {time.time()-t0:.0f}s")
