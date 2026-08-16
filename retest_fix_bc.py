"""Fix-up pass for the all-books retest: Book B (correct signal timing) and
Book C (record with ASCII-safe labels after cp949 console crash)."""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from live.config import PARAMS
from strategies.intraday_mean_reversion import run_intraday_mean_reversion
from strategies.qqq_bubble_hourly import calculate_bubble_score
from tools.metrics import sharpe_ratio, max_drawdown, metrics_from_returns
from tools.record import record_performance

TD, TC = 252, 0.001

def report(book, ser, convention, params, doc):
    ser = ser.dropna()
    m = metrics_from_returns(ser.values, TD)
    print(f"\nBOOK {book} [{convention}] params={params}")
    print(f"  retest : Sharpe {m['sharpe']:.3f} | CAGR {m['cagr']:.1%} | "
          f"total {m['total_return']:.1%} | MaxDD {m['max_dd']:.1%} | days {m['n']}")
    print(f"  docs   : Sharpe {doc[0]:.3f} | MaxDD {doc[1]:.1%} -> dSharpe {m['sharpe']-doc[0]:+.2f}")
    for y, x in ser.groupby(ser.index.year):
        print(f"    {y}: ret {(1+x).prod()-1:+8.2%}  sharpe {sharpe_ratio(x.values,TD):6.2f}  "
              f"mdd {max_drawdown(x.values):8.2%}")
    name = f"book_{book.lower()}_retest"
    record_performance(name=name, dates=ser.index, returns=ser.values, params=params,
        data_period=f"{ser.index.min().date()}..{ser.index.max().date()}",
        periods_per_year=TD,
        extra={"convention": convention, "retest": "2026-08-15 all-books retest"})
    print(f"  recorded -> strategies/performance/{name}_*")

# ── BOOK B: engine-matched timing (signal known at bar i = score from i-1; enter bar i open)
p = PARAMS["B"]
q = pd.read_parquet("data/cache/qqq_hourly.parquet")
q.index = pd.to_datetime(q.index)
qo, qc = q["open"].dropna(), q["close"].dropna()
idx = qo.index.intersection(qc.index); qo, qc = qo.loc[idx], qc.loc[idx]
sig = calculate_bubble_score(qc, p["qqq_bubble_ma_hours"], p["z_window_hours"]).shift(1)
nq = len(qc); hold = p["hold_hours"]
pos = np.zeros(nq); em = np.zeros(nq, bool); xm = np.zeros(nq, bool)
i = 0
while i < nq - hold:
    if pd.notna(sig.iloc[i]) and sig.iloc[i] < p["threshold"]:
        pos[i:i+hold] = 1.0; em[i] = True; xm[min(i+hold-1, nq-1)] = True
        i += hold
    else:
        i += 1
qo_np, qc_np = qo.values, qc.values
o2c = np.where(qo_np > 0, qc_np/qo_np - 1, 0.0)
c_prev = np.concatenate([[qc_np[0]], qc_np[:-1]])
c2c = np.where(c_prev > 0, qc_np/c_prev - 1, 0.0)
br = np.where(em, o2c, c2c) * pos
br[em] -= TC; br[xm] -= TC
sb = pd.Series(br, index=idx)
daily_b = sb.groupby(sb.index.normalize()).apply(lambda g: float((1+g).prod()-1))
daily_b.index = pd.to_datetime(daily_b.index)
report("B", daily_b, "PROPER bar-by-bar daily, engine-matched timing",
       {k: p[k] for k in ("qqq_bubble_ma_hours","z_window_hours","threshold","hold_hours")},
       (1.106, -0.0594))

# ── BOOK C: rerun + record (ASCII-safe)
ho = pd.read_parquet("data/cache/merged_hourly_open.parquet")
hc = pd.read_parquet("data/cache/merged_hourly_close.parquet")
ho.index = pd.to_datetime(ho.index); hc.index = pd.to_datetime(hc.index)
common = sorted(set(ho.columns) & set(hc.columns)); ho, hc = ho[common], hc[common]
daily_all = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily_all.index = pd.to_datetime(daily_all.index)
pC = PARAMS["C"]
dc = daily_all[[c for c in common if c in daily_all.columns]]
ret_c, _, _ = run_intraday_mean_reversion(
    dc, ho, hc, sigma_grid=[pC["sigma"]], flip_hold_days_grid=[pC["flip_hold_days"]],
    lookback_grid=[pC["z_lookback_days"]], top_n_grid=[pC["top_n"]])
report("C", ret_c, "BLOCK entry-date (flagged: needs MTM rewrite)",
       {k: pC[k] for k in ("sigma","z_lookback_days","flip_hold_days","top_n")},
       (0.927, -0.2081))
