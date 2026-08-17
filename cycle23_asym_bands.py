"""
Queue #26 — D asymmetric entry/exit bands, ANCHORED redo.

Custom per-name loop. Stage 1 (ANCHOR): fixed hold=8, per-name daily
attribution identical to the research engine -> must reproduce official
book_d_retest 2.740 +/- 0.05 or VOID.
Stage 2: lazy exit — exit at the first bar where the (shifted) score has
recovered above -0.4, capped at 21h; entry rule unchanged. Costs: 0.1%/side,
TC spread over each trade's ACTUAL hold length.

Verdict gates: standalone vs official D8; if promising, champion-level test
comes later (blend + engine) — this script answers the mechanism question.
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
TC1 = 0.001  # official engine convention: ONE-WAY TC spread over hold hours


def run_variant(exit_mode: str, hold_fix: int = 8, exit_band: float = -0.4,
                cap_h: int = 21) -> pd.Series:
    """exit_mode='fixed' (anchor) or 'lazy' (score>band or cap)."""
    free_at = np.zeros(U, dtype=np.int64)
    daily_num = np.zeros(D, dtype=np.float64)
    daily_den = np.zeros(D, dtype=np.float64)
    hold_sum = 0.0
    hold_cnt = 0
    loop_end = T - (hold_fix if exit_mode == "fixed" else cap_h) - 1
    for t in range(warmup, loop_end):
        scores = bub[t]
        available = (scores < thr) & (free_at <= t)
        if not available.any():
            continue
        ai = np.where(available)[0]
        npick = min(top_n, len(ai))
        chosen = ai[np.argpartition(scores[ai], npick - 1)[:npick]]
        eb = t + 1
        for s in chosen:
            if exit_mode == "fixed":
                xb = min(t + hold_fix, T - 1)
            else:
                xb = min(t + cap_h, T - 1)
                for tt in range(eb + 1, min(t + cap_h, T - 1) + 1):
                    if bub[tt, s] > exit_band:
                        xb = tt
                        break
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
            hold_sum += xb - eb + 1
            hold_cnt += 1
            free_at[s] = xb
    active = daily_den > 0
    # engine convention: flat one-way TC / hold-hours per active day; for the
    # variable-exit variant use the variant's own MEAN hold hours
    eff_hold = (hold_sum / hold_cnt) if hold_cnt else hold_fix
    port = np.zeros(D, dtype=np.float64)
    port[active] = daily_num[active] / daily_den[active] - TC1 / eff_hold
    ser = pd.Series(port, index=pd.to_datetime(tdays))
    return ser[(ser.index >= "2019-01-02") & (ser.index <= END)]


# anchor target: official engine regenerated on the CURRENT cache
off_ser, _, _ = CB.run_contrarian_bubble_hourly(
    hoo, hcc, ma_window_grid=[104], buy_threshold_grid=[0.8],
    hold_hours_grid=[8], top_n_grid=[20])
off_ser = off_ser.dropna()
off_ser = off_ser[(off_ser.index >= "2019-01-02") & (off_ser.index <= END)]
m_off = metrics_from_returns(off_ser.values, TD)

anchor = run_variant("fixed")
m_a = metrics_from_returns(anchor.values, TD)
cc = float(np.corrcoef(anchor.reindex(off_ser.index).fillna(0), off_ser)[0, 1])
print(f"ANCHOR fixed-8: {m_a['sharpe']:.3f} vs official-engine-now "
      f"{m_off['sharpe']:.3f} (diff {m_a['sharpe']-m_off['sharpe']:+.3f}, "
      f"corr {cc:.4f})  [{time.time()-t0:.0f}s]")
if abs(m_a["sharpe"] - m_off["sharpe"]) > 0.05 or cc < 0.995:
    print("ANCHOR FAILED — VOID")
    sys.exit(1)

for band, cap in [(-0.4, 21), (-0.6, 21), (-0.4, 14), (-0.7, 14)]:
    lazy = run_variant("lazy", exit_band=band, cap_h=cap)
    m_l = metrics_from_returns(lazy.values, TD)
    r = sharpe_delta_test(lazy, anchor)
    sig = "  <-- SIGNIFICANT" if r["significant_p10"] else ""
    print(f"lazy band {band} cap {cap}h: Sharpe {m_l['sharpe']:.3f} "
          f"MaxDD {m_l['max_dd']:.1%} (delta {r['delta']:+.3f}, "
          f"p {r['p_one_sided']:.3f}){sig}  [{time.time()-t0:.0f}s]")
print(f"total {time.time()-t0:.0f}s")
