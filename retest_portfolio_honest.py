"""
HONEST combined-portfolio re-estimation from the 2026-08-15 retest series.

Inputs: strategies/performance/book_{a,b,c,d,f}_retest_daily.csv — all books
scored with locked params + proper daily attribution (C = MTM rewrite).

Configs (mirroring run_portfolio_abdc.py exactly, incl. 50% cap redistribution):
  - Fixed EW  A+F+D+C+B   (documented: Sharpe 2.344, CAGR 47.5%, MaxDD -18.3%)
  - MomAlloc  lb=20 rb=60 (documented: Sharpe 2.170, CAGR 61.9%, MaxDD -28.8%)
  - Fixed EW  without B   (Book B fate analysis)
  - MomAlloc  without B
Start 2019-01-02. B activates at its first retest date (2020-07-27).
Records the two headline configs via tools/record.py.
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from tools.metrics import sharpe_ratio, max_drawdown, metrics_from_returns
from tools.record import record_performance

TD = 252
MAX_ALLOC = 0.50
START = "2019-01-02"

def load(name):
    df = pd.read_csv(f"strategies/performance/{name}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)

books = {k: load(f"book_{k}_retest") for k in ["a", "b", "c", "d", "f"]}
full_idx = sorted(set().union(*[s.index for s in books.values()]))
full_idx = pd.DatetimeIndex([d for d in full_idx if d >= pd.Timestamp(START)])

vals, act = {}, {}
for k, s in books.items():
    first = s.index.min()
    v = s.reindex(full_idx).fillna(0.0).values
    vals[k] = v
    act[k] = np.array([d >= first for d in full_idx])
n = len(full_idx)
print(f"Portfolio window: {full_idx[0].date()} .. {full_idx[-1].date()} ({n} days)")
for k in books: print(f"  {k.upper()}: active from {books[k].index.min().date()}")

def fixed_ew(keys):
    port = np.zeros(n)
    for i in range(n):
        active = [k for k in keys if act[k][i]]
        if active:
            port[i] = np.mean([vals[k][i] for k in active])
    return pd.Series(port, index=full_idx)

def mom_alloc(keys, lookback=20, rebalance=60):
    port = np.zeros(n)
    w = {k: 0.0 for k in keys}
    for i in range(n):
        use = {k: act[k][i] for k in keys}
        if i >= lookback and i % rebalance == 0:
            pos = {}
            for k in keys:
                if use[k] and act[k][max(i - lookback, 0)]:
                    pos[k] = max(0.0, float(np.prod(1 + vals[k][i-lookback:i]) - 1))
                else:
                    pos[k] = 0.0
            total = sum(pos.values())
            if total > 1e-9:
                r = {k: pos[k]/total for k in keys}
                for _ in range(8):
                    exc = sum(max(0.0, x - MAX_ALLOC) for x in r.values())
                    if exc < 1e-9: break
                    r = {k: min(x, MAX_ALLOC) for k, x in r.items()}
                    unc = [k for k in keys if r[k] < MAX_ALLOC and use[k]]
                    if unc:
                        each = exc / len(unc)
                        for k in unc: r[k] = min(r[k] + each, MAX_ALLOC)
                w = r
            else:
                na = sum(use.values()); w = {k: (1.0/na if use[k] else 0.0) for k in keys}
        elif i < lookback:
            na = sum(use.values()); w = {k: (1.0/na if use[k] else 0.0) for k in keys}
        port[i] = sum(w[k]*vals[k][i] for k in keys)
    return pd.Series(port, index=full_idx)

def show(tag, ser, doc=None):
    m = metrics_from_returns(ser.values, TD)
    line = (f"{tag:<26} Sharpe {m['sharpe']:.3f} | CAGR {m['cagr']:.1%} | "
            f"total {m['total_return']:+.1%} | MaxDD {m['max_dd']:.1%}")
    if doc: line += f"   [docs: {doc[0]:.3f} / {doc[1]:.1%}]"
    print(line)
    return ser, m

ALL = ["a", "b", "c", "d", "f"]
NO_B = ["a", "c", "d", "f"]

print("\n=== HONEST PORTFOLIO (retest inputs) vs DOCUMENTED ===")
few, m_few   = show("FixedEW A+F+D+C+B",  fixed_ew(ALL),  (2.344, -0.183))
mal, m_mal   = show("MomAlloc lb20 rb60", mom_alloc(ALL), (2.170, -0.288))
few2, m_few2 = show("FixedEW no-B",       fixed_ew(NO_B))
mal2, m_mal2 = show("MomAlloc no-B",      mom_alloc(NO_B))

print("\n=== Book B marginal contribution ===")
print(f"  FixedEW : with B {m_few['sharpe']:.3f}  without B {m_few2['sharpe']:.3f}  "
      f"delta {m_few['sharpe']-m_few2['sharpe']:+.3f}")
print(f"  MomAlloc: with B {m_mal['sharpe']:.3f}  without B {m_mal2['sharpe']:.3f}  "
      f"delta {m_mal['sharpe']-m_mal2['sharpe']:+.3f}")

print("\n=== Yearly (FixedEW honest) ===")
for y, x in few.groupby(few.index.year):
    print(f"  {y}: ret {(1+x).prod()-1:+8.2%}  sharpe {sharpe_ratio(x.values,TD):6.2f}  "
          f"mdd {max_drawdown(x.values):8.2%}")

for nm, ser, extra in [
    ("portfolio_fixed_ew_honest", few, {"books": "A+B+C+D+F retest series", "note": "honest re-estimate"}),
    ("portfolio_momalloc_honest", mal, {"books": "A+B+C+D+F retest series", "lb": 20, "rb": 60}),
]:
    record_performance(name=nm, dates=ser.index, returns=ser.values,
        params={"config": nm}, data_period=f"{ser.index.min().date()}..{ser.index.max().date()}",
        periods_per_year=TD, extra=extra)
    print(f"recorded -> strategies/performance/{nm}_*")
