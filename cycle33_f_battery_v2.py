"""
Cycle 33 v2 — F battery on an EXACT copy of live replay_F (v1 VOIDed:
reconstruction missed the hourly splice guard, the row!=0 validity rule,
the step-by-hold-always loop, and the 2x TC convention).

Anchor: param copy (default args) vs actual replay_F on the same panels at
research TC — must match to ~0.000. Baseline = engine-now (recorded
book_f_retest is from an older cache state; like-for-like rule from cycle23).
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
import live.signals as S
import live.settle as SET

TD, END = 252, "2026-07-08"
t0 = time.time()

def load(n):
    df = pd.read_csv(f"strategies/performance/{n}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)

panels = S.load_panels()
hc, ho, idx = panels["hourly_close"], panels["hourly_open"], panels["idx_h"]
tickers = panels["tickers"]
prices = hc.values.astype(np.float64)
opens = ho.values.astype(np.float64)
U, T = len(tickers), len(idx)
p = dict(C.PARAMS["F"])
p["tc_one_way"] = 0.001                      # research cost basis
mom_default = S.momentum_hours(hc, p["lookback_hours"]).values
tdays, bdi, day_last, day_first, dret = SET._daily_infra(idx, prices)
jump = hc.pct_change().abs().rolling(p["lookback_hours"], min_periods=1).max().values
last_bar = T - 1


def run_f(score=None, offset=0):
    """EXACT copy of replay_F's loop, parametrized by score matrix + offset."""
    mom = mom_default if score is None else score
    warmup = p["lookback_hours"] + 1 + offset
    hold_h, top_n = p["hold_hours"], p["top_n"]
    trades = []
    i = warmup
    while i < T - 1:
        row = mom[i]
        valid = np.where(np.isfinite(row) & (row != 0) & ~(jump[i] > 1.0))[0]
        if len(valid) >= top_n:
            chosen = valid[np.argpartition(row[valid], -top_n)[-top_n:]]
            eb = i + 1
            xb = min(i + hold_h, last_bar)
            trades.append((eb, xb, list(chosen), +1))
        i += hold_h
    port = E.equal_weight_daily_pnl(trades, prices, opens, idx, day_last,
                                    day_first, bdi, dret, U,
                                    2 * p["tc_one_way"], hold_h)
    s = pd.Series(port, index=pd.to_datetime(tdays)).dropna()
    return s[(s.index >= "2019-01-02") & (s.index <= END)]


# ---- anchor: copy vs actual replay_F at same TC ----
saved_tc = C.PARAMS["F"]["tc_one_way"]
C.PARAMS["F"]["tc_one_way"] = 0.001
off_ser, _, _ = SET.replay_F(panels)
C.PARAMS["F"]["tc_one_way"] = saved_tc
off_ser = off_ser.dropna()
off_ser = off_ser[(off_ser.index >= "2019-01-02") & (off_ser.index <= END)]
mine = run_f()
m_m, m_o = metrics_from_returns(mine.values, TD), metrics_from_returns(off_ser.values, TD)
jj = mine.index.intersection(off_ser.index)
cc = float(np.corrcoef(mine.loc[jj], off_ser.loc[jj])[0, 1])
print(f"ANCHOR: copy {m_m['sharpe']:.3f} vs replay_F {m_o['sharpe']:.3f} "
      f"(diff {m_m['sharpe']-m_o['sharpe']:+.3f}, corr {cc:.4f})")
if abs(m_m["sharpe"] - m_o["sharpe"]) > 0.02 or cc < 0.999:
    print("ANCHOR FAILED — VOID")
    sys.exit(1)

# ---- champion machinery (engine-now F as base) ----
d8q, d14 = load("book_d8_quiet"), load("book_d14")
ju = d8q.index.union(d14.index)
dblend = 0.5 * d8q.reindex(ju).fillna(0) + 0.5 * d14.reindex(ju).fillna(0)
BOOKS = {"A": load("book_a_retest"), "C": load("book_c_overlap_cap"),
         "D": dblend, "F": off_ser}


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


champ_base = champ_with(off_ser)
print(f"champion base (engine-now F): "
      f"{metrics_from_returns(champ_base.values, TD)['sharpe']:.3f}")

ret_h = hc.pct_change()
vol750 = ret_h.rolling(p["lookback_hours"]).std().values
skip = S.momentum_hours(hc, p["lookback_hours"] - 35)
skip_m = skip.shift(35).values

variants = {}
tr100 = run_f(offset=100)
juT = mine.index.union(tr100.index)
variants["(i) tranche-2 (0/100h)"] = (0.5 * mine.reindex(juT).fillna(0)
                                      + 0.5 * tr100.reindex(juT).fillna(0))
ra = mom_default / (vol750 + 1e-9)
ra[vol750 <= 0] = np.nan
variants["(ii) risk-adj rank"] = run_f(score=ra)
variants["(iii) skip-week"] = run_f(score=skip_m)

for name, s_ in variants.items():
    m_ = metrics_from_returns(s_.dropna().values, TD)
    ser = champ_with(s_)
    r = sharpe_delta_test(ser, champ_base)
    m = metrics_from_returns(ser.values, TD)
    sig = "  <-- SIGNIFICANT" if r["significant_p10"] and r["delta"] > 0 else ""
    print(f"{name}: standalone {m_['sharpe']:.2f}/{m_['max_dd']:.0%} "
          f"(F now {m_o['sharpe']:.2f}/{m_o['max_dd']:.0%}) | champ "
          f"{m['sharpe']:.3f} (delta {r['delta']:+.3f}, p {r['p_one_sided']:.3f})"
          f"{sig}  [{time.time()-t0:.0f}s]")
print(f"total {time.time()-t0:.0f}s")
