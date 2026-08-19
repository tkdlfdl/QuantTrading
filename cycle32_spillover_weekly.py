"""
Cycle 32 (manual) — registry #163-164.

 (A) Sector-peer intraday spillover (JFQA 2025): at each hour t, signal_i =
     z of mean trailing-5h return of sector-mates (excluding self). Long
     top-20, hold 4h, per-name non-overlap. P&L via the validated
     equal_weight_daily_pnl kernel (exact-match vs official engine, cycle23),
     round-trip 0.2% spread over the hold.
 (B) Weekly sector-neutral reversal (DLS): every 5th session, rank 5d
     sector-demeaned returns, long bottom-20 equal-weight, hold 5d.
     0.1%/side.

Gates: standalone, corr(champ), stress, FULL-STACK marginal @0.25, LW p.
"""
from __future__ import annotations
import sys, warnings, time, json
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


def judge(name, stream):
    stream = stream[stream.index <= END].dropna()
    ms = metrics_from_returns(stream.values, TD)
    s19 = stream.reindex(champ.index).fillna(0)
    cc = float(np.corrcoef(s19, champ)[0, 1])
    sd = float(stream.reindex(worst).fillna(0).mean())
    print(f"  {name}: {stream.index.min().date()}..: Sharpe {ms['sharpe']:.2f} "
          f"CAGR {ms['cagr']:.1%} MaxDD {ms['max_dd']:.0%} | corr {cc:+.2f} | "
          f"stress {sd:+.2%}/d")
    ser = build_port({**FULL, "N": stream}, ORDER + ["N"], {**SHARES, "N": .25})
    r = sharpe_delta_test(ser, full_stack)
    m = metrics_from_returns(ser.values, TD)
    sig = "  <-- ADDS ON FULL STACK" if r["significant_p10"] and r["delta"] > 0 else ""
    print(f"    FULL-STACK +N @0.25: {m['sharpe']:.3f} (delta {r['delta']:+.3f}, "
          f"p {r['p_one_sided']:.3f}){sig}")


sector_map = json.load(open("data/cache/sector_map.json", encoding="utf-8"))

# ══════════ (A) sector-peer intraday spillover ══════════
print(f"\n=== (A) sector-peer intraday spillover (hourly) ===")
hc = pd.read_parquet("data/cache/merged_hourly_close.parquet")
hc.index = pd.to_datetime(hc.index)
ho = pd.read_parquet("data/cache/merged_hourly_open.parquet")
ho.index = pd.to_datetime(ho.index)
common = sorted((set(ho.columns) & set(hc.columns)) - {"SPY"})
common = [c for c in common if c in sector_map]
hcc, hoo = hc[common].ffill(), ho[common].ffill()
idx = hcc.index
prices = hcc.values.astype(np.float64)
opens = hoo.values.astype(np.float64)
T, U = prices.shape
r5 = hcc.pct_change(5)

sectors = pd.Series({c: sector_map[c] for c in common})
sec_names = sorted(set(sectors))
sec_mat = np.zeros((U, len(sec_names)))
for i, c in enumerate(common):
    sec_mat[i, sec_names.index(sectors[c])] = 1.0
counts = sec_mat.sum(axis=0)

R5 = r5.values
peer_sum = np.nan_to_num(R5) @ sec_mat            # T x S sector sums
valid = np.isfinite(R5).astype(float) @ sec_mat   # counts of valid names
peer_mean_ex = np.zeros_like(R5)
for j in range(len(sec_names)):
    members = sec_mat[:, j] == 1.0
    denom = np.maximum(valid[:, j:j+1] - 1, 1)
    pm = (peer_sum[:, j:j+1] - np.nan_to_num(R5[:, members])) / denom
    peer_mean_ex[:, members] = pm
sig_m = peer_mean_ex
sig_m = (sig_m - np.nanmean(sig_m, axis=1, keepdims=True)) / \
        (np.nanstd(sig_m, axis=1, keepdims=True) + 1e-9)
sig_m[~np.isfinite(R5)] = np.nan
sig_lag = np.empty_like(sig_m)
sig_lag[0] = np.nan
sig_lag[1:] = sig_m[:-1]                          # signal lagged 1 bar

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

HOLD, TOPN = 4, 20
free_at = np.zeros(U, dtype=np.int64)
trades = []
for t in range(10, T - HOLD - 1):
    row = sig_lag[t]
    avail = np.isfinite(row) & (free_at <= t)
    if avail.sum() < 50:
        continue
    ai = np.where(avail)[0]
    chosen = ai[np.argpartition(-row[ai], TOPN - 1)[:TOPN]]   # HIGHEST peer signal
    eb, xb = t + 1, min(t + HOLD, T - 1)
    trades.append((eb, xb, list(chosen), +1))
    free_at[chosen] = xb
print(f"  trade blocks: {len(trades)}  ({time.time()-t0:.0f}s)")
port = E.equal_weight_daily_pnl(trades, prices, opens, idx, day_last, day_first,
                                bdi, dret, U, 0.002, HOLD)
sA = pd.Series(port, index=pd.to_datetime(tdays)).dropna()
sA = sA[sA.index >= "2019-06-01"]
judge("peer intraday spillover", sA)
# control: LOWEST peer signal (reversal direction) — the paper predicts continuation
free_at = np.zeros(U, dtype=np.int64)
trades = []
for t in range(10, T - HOLD - 1):
    row = sig_lag[t]
    avail = np.isfinite(row) & (free_at <= t)
    if avail.sum() < 50:
        continue
    ai = np.where(avail)[0]
    chosen = ai[np.argpartition(row[ai], TOPN - 1)[:TOPN]]
    eb, xb = t + 1, min(t + HOLD, T - 1)
    trades.append((eb, xb, list(chosen), +1))
    free_at[chosen] = xb
port = E.equal_weight_daily_pnl(trades, prices, opens, idx, day_last, day_first,
                                bdi, dret, U, 0.002, HOLD)
sA2 = pd.Series(port, index=pd.to_datetime(tdays)).dropna()
sA2 = sA2[sA2.index >= "2019-06-01"]
m2 = metrics_from_returns(sA2[sA2.index <= END].values, TD)
print(f"    control (lowest peer signal): Sharpe {m2['sharpe']:.2f}")

# ══════════ (B) weekly sector-neutral reversal ══════════
print(f"\n=== (B) weekly sector-neutral reversal ({time.time()-t0:.0f}s) ===")
daily_px = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily_px.index = pd.to_datetime(daily_px.index)
stocks = [c for c in daily_px.columns if not c.startswith("^")
          and c not in {"UVXY", "SPY", "QQQ"} and c in sector_map]
px = daily_px[stocks].ffill()
ret = px.pct_change()
jump = ret.abs().rolling(252).max()
r5d = px.pct_change(5)
sec_ser = pd.Series({c: sector_map[c] for c in stocks})
demean = r5d.sub(r5d.T.groupby(sec_ser).transform("mean").T)
posB = pd.DataFrame(0.0, index=px.index, columns=stocks)
dates = px.index
i = 260
while i < len(dates) - 6:
    row = demean.iloc[i].dropna()
    row = row[jump.iloc[i][row.index] <= 1.0]
    if len(row) >= 100:
        losers = row.nsmallest(20).index
        cols = [posB.columns.get_loc(c) for c in losers]
        posB.iloc[i + 1:i + 6, cols] = 1.0 / 20
        i += 5
    else:
        i += 1
w = posB.shift(1)
sB = ((ret.fillna(0) * w).sum(axis=1)
      - w.diff().abs().sum(axis=1).fillna(0) * 0.001)
sB = sB[sB.index >= "1999-06-01"].dropna()
judge("weekly sector-neutral reversal", sB)
print(f"\ntotal {time.time()-t0:.0f}s")
