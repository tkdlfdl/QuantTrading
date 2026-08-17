"""
Queue #23 — D path-composition diagnostic (analysis only, anchored redo).

For every official D8 entry (raw ordering, research basis), decompose the
trailing 5-day decline into OVERNIGHT (close->open gaps) vs INTRADAY
(open->close) components; measure forward rebound at 8h and 14h by cohort.

ANCHOR: the reconstructed entry loop + equal-weight P&L at 0.1%/side must
reproduce official book_d_retest (2.740) within +/-0.05 Sharpe, else VOID.

Informs #24 (sleeve routing by reversal speed): if cohorts show a material
8h-vs-14h horizon spread, routing has legs; if flat, #24 dies here.
"""
from __future__ import annotations
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from tools.metrics import metrics_from_returns
from live import engine as E, config as C
import live.signals as S

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
# official engine applies shift(1): signal at bar t uses score from t-1
bub = np.empty_like(raw)
bub[0] = 0.0
bub[1:] = raw[:-1]

# ---- official-selection loop (identical to research engine) ----
thr, hold_h, top_n = -0.8, 8, 20
warmup = 105
free_at = np.zeros(U, dtype=np.int64)
trades = []
last_bar = T - 1
for t in range(warmup, T - hold_h - 1):
    scores = bub[t]
    avail = (scores < thr) & (free_at <= t)
    if not avail.any():
        continue
    ai = np.where(avail)[0]
    npick = min(top_n, len(ai))
    chosen = ai[np.argpartition(scores[ai], npick - 1)[:npick]]
    eb = t + 1
    xb = min(t + hold_h, last_bar)
    trades.append((eb, xb, list(chosen), +1))
    free_at[chosen] = xb
print(f"trades blocks: {len(trades)} ({time.time()-t0:.0f}s)")

# ---- anchor: daily P&L must reproduce official D8 ----
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

# official engine convention: ONE-WAY TC spread over hold (tc/hold per active
# day) — equal_weight_daily_pnl's tc_round_trip/hold matches with 0.001
port = E.equal_weight_daily_pnl(trades, prices, opens, idx, day_last, day_first,
                                bdi, dret, U, 0.001, hold_h)
ser = pd.Series(port, index=pd.to_datetime(tdays)).dropna()
ser = ser[(ser.index >= "2019-01-02") & (ser.index <= END)]
m = metrics_from_returns(ser.values, TD)
# anchor target: OFFICIAL ENGINE regenerated on the current cache (the
# recorded book_d_retest is from an older cache state; the anchor rule
# compares like-for-like inputs)
off_ser, _, _ = CB.run_contrarian_bubble_hourly(
    hoo, hcc, ma_window_grid=[104], buy_threshold_grid=[0.8],
    hold_hours_grid=[8], top_n_grid=[20])
off_ser = off_ser.dropna()
off_ser = off_ser[(off_ser.index >= "2019-01-02") & (off_ser.index <= END)]
m_off = metrics_from_returns(off_ser.values, TD)
cc = float(np.corrcoef(ser.reindex(off_ser.index).fillna(0), off_ser)[0, 1])
print(f"ANCHOR: reconstructed {m['sharpe']:.3f} vs official-engine-now "
      f"{m_off['sharpe']:.3f} (diff {m['sharpe']-m_off['sharpe']:+.3f}, corr {cc:.4f})")
if abs(m["sharpe"] - m_off["sharpe"]) > 0.05 or cc < 0.995:
    print("ANCHOR FAILED — VOID diagnostic")
    sys.exit(1)

# ---- decomposition per entry ----
LOOK = 35  # 5 trading days of hourly bars
day_first_of_bar = day_first[bdi]           # first bar index of each bar's day
rows = []
for eb, xb, chosen, _ in trades:
    t = eb - 1                              # signal bar
    if t < LOOK + 1:
        continue
    for s in chosen:
        p_now = prices[t, s]
        p_then = prices[t - LOOK, s]
        if not (np.isfinite(p_now) and np.isfinite(p_then)) or p_then <= 0 or p_now <= 0:
            continue
        total = np.log(p_now / p_then)
        # overnight: sum of log(first bar open of day / prev day last close)
        on = 0.0
        for tt in range(t - LOOK + 1, t + 1):
            if bdi[tt] != bdi[tt - 1]:      # day boundary
                pc, po = prices[tt - 1, s], opens[tt, s]
                if np.isfinite(pc) and np.isfinite(po) and pc > 0 and po > 0:
                    on += np.log(po / pc)
        e8 = min(t + 8, last_bar); e14 = min(t + 14, last_bar)
        entry = opens[eb, s]
        if not np.isfinite(entry) or entry <= 0:
            continue
        f8 = prices[e8, s] / entry - 1
        f14 = prices[e14, s] / entry - 1
        rows.append((total, on, total - on, bub[t, s], f8, f14))
df = pd.DataFrame(rows, columns=["trail", "overnight", "intraday", "score", "f8", "f14"])
df = df[df["trail"] < 0]                    # true decliners only
df["on_share"] = (df["overnight"] / df["trail"]).clip(-1, 2)
print(f"entries analysed: {len(df)} ({time.time()-t0:.0f}s)")

print("\n--- rebound by OVERNIGHT-SHARE tercile (of the decline) ---")
df["cohort"] = pd.qcut(df["on_share"], 3, labels=["gap-light", "mixed", "gap-heavy"])
g = df.groupby("cohort")[["f8", "f14"]].agg(["mean", "count"])
print(g.to_string())
print("\n--- rebound by DECLINE SPEED (trail depth tercile) ---")
df["speed"] = pd.qcut(df["trail"], 3, labels=["deepest", "middle", "shallowest"])
print(df.groupby("speed")[["f8", "f14"]].mean().to_string())
print("\n--- horizon spread (f14 - f8) by cohort: does routing have legs? ---")
df["h_spread"] = df["f14"] - df["f8"]
print(df.groupby("cohort")["h_spread"].mean().to_string())
print(df.groupby("speed")["h_spread"].mean().to_string())
print(f"\nALL entries: f8 {df['f8'].mean():+.3%}  f14 {df['f14'].mean():+.3%}  "
      f"spread {df['h_spread'].mean():+.3%}")
print(f"total {time.time()-t0:.0f}s")
