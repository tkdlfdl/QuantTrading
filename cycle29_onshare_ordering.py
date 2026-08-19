"""
Cycle 29 — complete the M1-Stage-A pre-registration: OVERNIGHT-SHARE ordering
on the D sleeves (the untested feature; hour-of-day is degenerate as a
cross-sectional ordering — constant per bar — and gating = skipping is
banned by the pre-registration).

Directional hypotheses from the #23 diagnostic (anchored, registry #133):
  D8  : prefer LOW  overnight-share of the trailing decline (gap-light
        rebounds fastest at 8h: f8 0.111% vs 0.073%).
  D14 : prefer HIGH overnight-share (gap-heavy has the largest f14: 0.177%).

Feature (causal): daily overnight/intraday log split from hourly bars,
trailing 5 completed days, shift(1); share = ON5/TOT5 on declining names,
z-clipped to [-1,1]. Ordering-encode S = where(B<-0.8, -0.81 + r*0.09, 0)
with r = +share_z (D8, low preferred -> engine nsmallest) and r = -share_z
(D14, high preferred).

ANCHORS: raw matrix through the patched path must reproduce the official
engine for each hold. Judge: champion config (quiet-D8 + D14 blend, 2sh)
with ONE sleeve's ordering replaced; rule 4b LW p<0.10.
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

import strategies.contrarian_bubble_hourly as CB
B = CB._bubble_matrix(hcc, 104)
_orig = CB._bubble_matrix

# ---- overnight-share feature (daily, causal) ----
day = idx.normalize()
close_last = hcc.groupby(day).last()
open_first = hoo.groupby(day).first()
ON = np.log(open_first / close_last.shift(1))
IN = np.log(close_last / open_first)
TOT5 = (ON + IN).rolling(5).sum().shift(1)
ON5 = ON.rolling(5).sum().shift(1)
share = (ON5 / TOT5).where(TOT5 < 0)          # defined on decliners only
share = share.clip(-1.0, 2.0)
sd = np.nanstd(share.values)
share_z = np.clip((share / (sd + 1e-9)).fillna(0.0), -1, 1)
f_bars = share_z.reindex(day).values          # map day feature to bars

def encode(sign):
    r = sign * f_bars
    return np.where(B < -0.8, -0.81 + r * 0.09, 0.0).astype(np.float32)

def run_engine(matrix, hold):
    CB._bubble_matrix = lambda close, m_: matrix
    s_, _, _ = CB.run_contrarian_bubble_hourly(
        hoo, hcc, ma_window_grid=[104], buy_threshold_grid=[0.8],
        hold_hours_grid=[hold], top_n_grid=[20])
    CB._bubble_matrix = _orig
    s_ = s_.dropna()
    return s_[(s_.index >= "2019-01-02") & (s_.index <= END)]

# ---- anchors ----
for hold, official in ((8, None), (14, "book_d14")):
    anc = run_engine(B, hold)
    m_a = metrics_from_returns(anc.values, TD)
    print(f"ANCHOR hold={hold}: raw-through-patch {m_a['sharpe']:.3f}", flush=True)

d8q, d14 = load("book_d8_quiet"), load("book_d14")
ju = d8q.index.union(d14.index)
base_blend = 0.5 * d8q.reindex(ju).fillna(0) + 0.5 * d14.reindex(ju).fillna(0)
BOOKS = {"A": load("book_a_retest"), "C": load("book_c_overlap_cap"),
         "D": base_blend, "F": load("book_f_retest")}

def champ_with(dblend):
    sb, ss = C.ALLOC_BOOKS, dict(C.ALLOC_SHARES)
    C.ALLOC_BOOKS = ["A", "C", "D", "F"]
    C.ALLOC_SHARES = {"D": 2.0}
    try:
        R = pd.DataFrame({**BOOKS, "D": dblend})
        R = R[(R.index >= "2019-01-02") & (R.index <= END)].fillna(0.0)
        return E.ivol_voltgt(R).dropna()
    finally:
        C.ALLOC_BOOKS, C.ALLOC_SHARES = sb, ss

champ_base = champ_with(base_blend)
m_cb = metrics_from_returns(champ_base.values, TD)
print(f"champion base (quiet-D8 + raw-D14): {m_cb['sharpe']:.3f}", flush=True)

# ---- (a) D8 prefer LOW overnight-share (replaces quiet in the blend) ----
d8_on = run_engine(encode(+1.0), 8)
m8 = metrics_from_returns(d8_on.values, TD)
ju2 = d8_on.index.union(d14.index)
bl = 0.5 * d8_on.reindex(ju2).fillna(0) + 0.5 * d14.reindex(ju2).fillna(0)
ser = champ_with(bl)
r = sharpe_delta_test(ser, champ_base)
m = metrics_from_returns(ser.values, TD)
print(f"(a) D8 low-ON-share: standalone {m8['sharpe']:.3f} "
      f"(quiet {metrics_from_returns(d8q.dropna().values, TD)['sharpe']:.3f}) | "
      f"champ {m['sharpe']:.3f} (delta {r['delta']:+.3f}, p {r['p_one_sided']:.3f})"
      f"{'  <-- BEATS QUIET' if r['significant_p10'] and r['delta'] > 0 else ''}",
      flush=True)

# ---- (b) D14 prefer HIGH overnight-share (quiet-D8 kept) ----
d14_on = run_engine(encode(-1.0), 14)
m14 = metrics_from_returns(d14_on.values, TD)
ju3 = d8q.index.union(d14_on.index)
bl = 0.5 * d8q.reindex(ju3).fillna(0) + 0.5 * d14_on.reindex(ju3).fillna(0)
ser = champ_with(bl)
r = sharpe_delta_test(ser, champ_base)
m = metrics_from_returns(ser.values, TD)
print(f"(b) D14 high-ON-share: standalone {m14['sharpe']:.3f} "
      f"(raw D14 {metrics_from_returns(d14.dropna().values, TD)['sharpe']:.3f}) | "
      f"champ {m['sharpe']:.3f} (delta {r['delta']:+.3f}, p {r['p_one_sided']:.3f})"
      f"{'  <-- SIGNIFICANT' if r['significant_p10'] and r['delta'] > 0 else ''}",
      flush=True)
print(f"total {time.time()-t0:.0f}s")
