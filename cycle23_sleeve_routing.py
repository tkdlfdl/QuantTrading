"""
Queue #24 — sleeve assignment by reversal speed (Dai et al. 2024 FAJ, #94).
UNLOCKED by #23: gap-heavy decliners rebound slower (f14-f8 spread 2x
gap-light's); deepest decliners have the largest horizon spread.

Variant: single loop, same selection as official D8; each chosen name's HOLD
is routed by its trailing-decline composition at signal time:
  slow cohort (overnight-share of 35-bar decline above median, or deepest
  tercile) -> hold 14h;  fast cohort -> hold 8h.
Two routings tested: by gap-share, by depth (score).

ANCHOR: the official blend 0.5*(fixed-8 + fixed-14) built with THIS loop must
match 0.5*(engine-8 + engine-14) on the current cache. The routed variant is
then compared against that blend (like-for-like: same engine, same names).
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
for t in range(T):
    day_last[bdi[t]] = t
daily_close = prices[day_last]
dret = np.zeros_like(daily_close)
dret[1:] = daily_close[1:] / np.maximum(daily_close[:-1], 1e-8) - 1
dret = np.clip(dret, -0.20, 0.20)

thr, top_n, warmup = -0.8, 20, 105
TC1, LOOK = 0.001, 35


def gap_share(t: int, s: int) -> float:
    """Overnight share of the trailing LOOK-bar decline (diagnostic method)."""
    p_now, p_then = prices[t, s], prices[t - LOOK, s]
    if not (np.isfinite(p_now) and np.isfinite(p_then)) or p_then <= 0 or p_now <= 0:
        return 0.5
    total = np.log(p_now / p_then)
    if total >= 0:
        return 0.5
    on = 0.0
    for tt in range(t - LOOK + 1, t + 1):
        if bdi[tt] != bdi[tt - 1]:
            pc, po = prices[tt - 1, s], opens[tt, s]
            if np.isfinite(pc) and np.isfinite(po) and pc > 0 and po > 0:
                on += np.log(po / pc)
    return float(np.clip(on / total, -1, 2))


def run(mode: str) -> pd.Series:
    """mode: 'fixed8' | 'fixed14' | 'route_gap' | 'route_depth'."""
    free_at = np.zeros(U, dtype=np.int64)
    daily_num = np.zeros(D, dtype=np.float64)
    daily_den = np.zeros(D, dtype=np.float64)
    hold_sum, hold_cnt = 0.0, 0
    for t in range(max(warmup, LOOK + 1), T - 14 - 1):
        scores = bub[t]
        available = (scores < thr) & (free_at <= t)
        if not available.any():
            continue
        ai = np.where(available)[0]
        npick = min(top_n, len(ai))
        chosen = ai[np.argpartition(scores[ai], npick - 1)[:npick]]
        eb = t + 1
        if mode == "route_depth":
            med = np.median(scores[chosen])
        for s in chosen:
            if mode == "fixed8":
                hold = 8
            elif mode == "fixed14":
                hold = 14
            elif mode == "route_gap":
                hold = 14 if gap_share(t, s) > 0.33 else 8
            else:  # route_depth: deeper than the block median -> slow sleeve
                hold = 14 if scores[s] <= med else 8
            xb = min(t + hold, T - 1)
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
            hold_sum += hold
            hold_cnt += 1
            free_at[s] = xb
    active = daily_den > 0
    eff_hold = (hold_sum / hold_cnt) if hold_cnt else 8
    port = np.zeros(D, dtype=np.float64)
    port[active] = daily_num[active] / daily_den[active] - TC1 / eff_hold
    ser = pd.Series(port, index=pd.to_datetime(tdays))
    return ser[(ser.index >= "2019-01-02") & (ser.index <= END)]


# official engine blend on current cache = anchor target
offs = {}
for h in (8, 14):
    s_, _, _ = CB.run_contrarian_bubble_hourly(
        hoo, hcc, ma_window_grid=[104], buy_threshold_grid=[0.8],
        hold_hours_grid=[h], top_n_grid=[20])
    s_ = s_.dropna()
    offs[h] = s_[(s_.index >= "2019-01-02") & (s_.index <= END)]
ju = offs[8].index.union(offs[14].index)
off_blend = 0.5 * offs[8].reindex(ju).fillna(0) + 0.5 * offs[14].reindex(ju).fillna(0)
m_ob = metrics_from_returns(off_blend.values, TD)

f8 = run("fixed8")
f14 = run("fixed14")
jm = f8.index.union(f14.index)
my_blend = 0.5 * f8.reindex(jm).fillna(0) + 0.5 * f14.reindex(jm).fillna(0)
m_mb = metrics_from_returns(my_blend.values, TD)
cc = float(np.corrcoef(my_blend.reindex(off_blend.index).fillna(0), off_blend)[0, 1])
print(f"ANCHOR blend: {m_mb['sharpe']:.3f} vs official blend {m_ob['sharpe']:.3f} "
      f"(diff {m_mb['sharpe']-m_ob['sharpe']:+.3f}, corr {cc:.4f})  [{time.time()-t0:.0f}s]")
if abs(m_mb["sharpe"] - m_ob["sharpe"]) > 0.05 or cc < 0.995:
    print("ANCHOR FAILED — VOID")
    sys.exit(1)

for mode in ("route_gap", "route_depth"):
    v = run(mode)
    m_v = metrics_from_returns(v.values, TD)
    r = sharpe_delta_test(v, my_blend)
    sig = "  <-- SIGNIFICANT" if r["significant_p10"] else ""
    print(f"{mode}: Sharpe {m_v['sharpe']:.3f} MaxDD {m_v['max_dd']:.1%} "
          f"(delta vs blend {r['delta']:+.3f}, p {r['p_one_sided']:.3f}){sig}  "
          f"[{time.time()-t0:.0f}s]")
print(f"total {time.time()-t0:.0f}s")
