"""
Queue #40 — D closing-auction entry tilt, anchored A/B.

Signals that fire in the last 2 bars of a trading day: variant defers the
entry to the NEXT day's first bar (open) instead of the next hourly bar.
Rationale: EOD loser rebound literature (Goyal-Jegadeesh-Wu corpus, #120) +
the closing auction being the cheapest venue; late-day capitulations may
keep falling into the close.

ANCHOR: immediate-entry loop must reproduce the official engine on the
current cache (+/-0.05, corr>=0.995) or VOID.
"""
from __future__ import annotations
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from tools.metrics import metrics_from_returns
from tools.significance import sharpe_delta_test

TD, END = 252, "2026-07-08"
t0 = time.time()

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

import strategies.contrarian_bubble_hourly as CB
raw = CB._bubble_matrix(hcc, 104)
bub = np.empty_like(raw)
bub[0] = 0.0
bub[1:] = raw[:-1]

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

thr, top_n, warmup, hold_h = -0.8, 20, 105, 8
TC1 = 0.001


def run(defer_late: bool) -> pd.Series:
    free_at = np.zeros(U, dtype=np.int64)
    daily_num = np.zeros(D, dtype=np.float64)
    daily_den = np.zeros(D, dtype=np.float64)
    for t in range(warmup, T - hold_h - 1):
        scores = bub[t]
        available = (scores < thr) & (free_at <= t)
        if not available.any():
            continue
        ai = np.where(available)[0]
        npick = min(top_n, len(ai))
        chosen = ai[np.argpartition(scores[ai], npick - 1)[:npick]]
        # late-day signal: t is one of the last 2 bars of its day
        is_late = t >= day_last[bdi[t]] - 1
        if defer_late and is_late:
            d_next = bdi[t] + 1
            if d_next >= D:
                continue
            eb = day_first[d_next]
        else:
            eb = t + 1
        xb = min(eb - 1 + hold_h, T - 1)
        for s in chosen:
            ep = opens[eb, s]
            xp = prices[xb, s]
            if ep <= 0 or xp <= 0 or not (np.isfinite(ep) and np.isfinite(xp)):
                free_at[s] = xb
                continue
            ed, xd = bdi[eb], bdi[xb]
            days = np.arange(ed, xd + 1)
            dr = dret[days, s].copy()
            dc_entry = prices[day_last[ed], s]
            dr[0] = (dc_entry / ep - 1) if dc_entry > 0 else 0.0
            if len(days) > 1:
                pc = prices[day_last[xd - 1], s]
                dr[-1] = (xp / pc - 1) if pc > 0 else 0.0
            dr = np.clip(dr, -0.20, 0.20)
            daily_num[days] += dr
            daily_den[days] += 1.0
            free_at[s] = xb
    active = daily_den > 0
    port = np.zeros(D, dtype=np.float64)
    port[active] = daily_num[active] / daily_den[active] - TC1 / hold_h
    ser = pd.Series(port, index=pd.to_datetime(tdays))
    return ser[(ser.index >= "2019-01-02") & (ser.index <= END)]


off_ser, _, _ = CB.run_contrarian_bubble_hourly(
    hoo, hcc, ma_window_grid=[104], buy_threshold_grid=[0.8],
    hold_hours_grid=[8], top_n_grid=[20])
off_ser = off_ser.dropna()
off_ser = off_ser[(off_ser.index >= "2019-01-02") & (off_ser.index <= END)]
m_off = metrics_from_returns(off_ser.values, TD)

base = run(defer_late=False)
m_b = metrics_from_returns(base.values, TD)
cc = float(np.corrcoef(base.reindex(off_ser.index).fillna(0), off_ser)[0, 1])
print(f"ANCHOR immediate: {m_b['sharpe']:.3f} vs official-now {m_off['sharpe']:.3f} "
      f"(diff {m_b['sharpe']-m_off['sharpe']:+.3f}, corr {cc:.4f})")
if abs(m_b["sharpe"] - m_off["sharpe"]) > 0.05 or cc < 0.995:
    print("ANCHOR FAILED — VOID")
    sys.exit(1)

tilt = run(defer_late=True)
m_t = metrics_from_returns(tilt.values, TD)
r = sharpe_delta_test(tilt, base)
sig = "  <-- SIGNIFICANT" if r["significant_p10"] else ""
print(f"defer-late-entries: Sharpe {m_t['sharpe']:.3f} MaxDD {m_t['max_dd']:.1%} "
      f"(delta {r['delta']:+.3f}, p {r['p_one_sided']:.3f}){sig}")
print(f"total {time.time()-t0:.0f}s")
