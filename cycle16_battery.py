"""
Cycle 16 — four structural candidates.

 (a) F phase-tranching (RTL): 4 offset cohorts of the F machinery (offsets
     0/50/100/150 bars), averaged. CORE SWAP: replaces F in the champion.
     Anchor: offset-0 cohort must reproduce 1.548.
 (b) D8 threshold barbell: average of 0.7/0.8/0.9-threshold D8 sleeves
     (official engine, patched thresholds). CORE SWAP for the d8 half of the
     D blend. Anchor: 0.8 sleeve = 2.740.
 (c) Cross-asset capitulation (SYNTHESIS): D-style bubble score on the 15-ETF
     daily panel (50d MA, 50d z, tanh), buy bottom-2 with score < -0.8,
     hold 10d. Satellite @0.25 shares.
 (d) Sector mean reversion: worst-2 of 11 sectors by 21d return, hold 21d.
     Satellite @0.25 shares.

Champion baseline 2.781 (portfolio_champ_d14). LW gate everywhere.
"""
from __future__ import annotations
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from tools.metrics import metrics_from_returns
from tools.significance import sharpe_delta_test
from tools.record import record_performance, record_improvement
from live import engine as E, config as C

TD, TC, END = 252, 0.001, "2026-07-08"
t0 = time.time()

def load(n):
    df = pd.read_csv(f"strategies/performance/{n}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)
champ = load("portfolio_champ_d14")
d8, d14 = load("book_d_retest"), load("book_d14")
ju = d8.index.union(d14.index)
dblend = 0.5*d8.reindex(ju).fillna(0) + 0.5*d14.reindex(ju).fillna(0)
BASE = {"A": load("book_a_retest"), "C": load("book_c_overlap_cap"),
        "D": dblend, "F": load("book_f_retest")}

def champ_with(books, alloc, shares):
    saved_b, saved_s = C.ALLOC_BOOKS, dict(C.ALLOC_SHARES)
    C.ALLOC_BOOKS = alloc; C.ALLOC_SHARES = shares
    try:
        R = pd.DataFrame(books)
        R = R[(R.index >= "2019-01-02") & (R.index <= END)].fillna(0.0)
        return E.ivol_voltgt(R).dropna()
    finally:
        C.ALLOC_BOOKS, C.ALLOC_SHARES = saved_b, saved_s

hc = pd.read_parquet("data/cache/merged_hourly_close.parquet"); hc.index = pd.to_datetime(hc.index)
ho = pd.read_parquet("data/cache/merged_hourly_open.parquet"); ho.index = pd.to_datetime(ho.index)
common = sorted((set(ho.columns) & set(hc.columns)) - {"SPY"})
hcc, hoo = hc[common], ho[common]
hc_np = hcc.values.astype(np.float32); ho_np = hoo.values.astype(np.float32)
bar_ts = hcc.index; n, U = hc_np.shape

# ═════════ (a) F phase-tranching ═════════
print("=== (a) F phase-tranching (4 cohorts) ===")
LB, HOLD, TOPN = 750, 200, 5
with np.errstate(divide="ignore", invalid="ignore"):
    mom = hc_np / np.vstack([np.full((LB, U), np.nan, dtype=np.float32), hc_np[:-LB]]) - 1
    o2c = np.where((ho_np>0)&np.isfinite(ho_np)&np.isfinite(hc_np), hc_np/ho_np-1, 0.0)
    cp = np.vstack([hc_np[:1], hc_np[:-1]])
    c2c = np.where((cp>0)&np.isfinite(cp)&np.isfinite(hc_np), hc_np/cp-1, 0.0)
jump = pd.DataFrame(hc_np, index=bar_ts).pct_change().abs().rolling(LB, min_periods=1).max().values

def run_f_offset(offset):
    H = np.zeros((n, U), dtype=np.float32)
    em = np.zeros(n, bool); xm = np.zeros(n, bool)
    i = LB + offset
    while i + HOLD < n:
        row = mom[i]
        valid = np.where(np.isfinite(row) & ~(jump[i] > 1.0))[0]
        if len(valid) >= TOPN:
            top = valid[np.argsort(row[valid])[-TOPN:]]
            H[i+1:i+HOLD+1, top] = 1.0/TOPN; em[i+1] = True; xm[i+HOLD] = True
        i += HOLD
    br = np.where(em[:, None], o2c, c2c)
    port = (H*br).sum(1); port[em] -= TC; port[xm] -= TC
    port[H.sum(1) == 0] = 0.0
    s = pd.Series(port.astype(float), index=bar_ts)
    d = s.groupby(s.index.normalize()).apply(lambda g: float((1+g).prod()-1))
    d.index = pd.to_datetime(d.index)
    return d

cohorts = [run_f_offset(k) for k in (0, 50, 100, 150)]
m0f = metrics_from_returns(cohorts[0][cohorts[0].index <= END].values, TD)
print(f"  anchor (offset 0): {m0f['sharpe']:.3f} {'OK' if abs(m0f['sharpe']-1.548)<0.06 else 'FAIL - VOID (a)'}")
if abs(m0f['sharpe']-1.548) < 0.06:
    idxu = cohorts[0].index
    for cs in cohorts[1:]: idxu = idxu.union(cs.index)
    F_tr = sum(cs.reindex(idxu).fillna(0) for cs in cohorts) / 4
    F_tr = F_tr[F_tr.index <= END]
    mtr = metrics_from_returns(F_tr.values, TD)
    print(f"  F tranched standalone: Sharpe {mtr['sharpe']:.3f} | CAGR {mtr['cagr']:.1%} | MaxDD {mtr['max_dd']:.1%} "
          f"(raw F: 1.548/-39.8%)")
    books = dict(BASE); books["F"] = F_tr
    ser = champ_with(books, ["A","C","D","F"], {"D": 2.0})
    mp = metrics_from_returns(ser.values, TD)
    rr = sharpe_delta_test(ser, champ)
    print(f"  champ with F-tranched: Sharpe {mp['sharpe']:.3f} (delta {rr['delta']:+.3f}, p {rr['p_one_sided']:.3f})"
          f"{'  <-- CANDIDATE' if rr['significant_p10'] else ''}")
    if rr["significant_p10"]:
        record_performance(name="book_f_tranched", dates=F_tr.index, returns=F_tr.values,
            params={"cohorts": 4, "offsets": [0,50,100,150], "lb": LB, "hold": HOLD},
            data_period=f"{F_tr.index.min().date()}..{F_tr.index.max().date()}", periods_per_year=TD,
            extra={"cycle": "Cycle 16a", "paper": "Hoffstein RTL 2019; JT 1993 cohorts"})
        record_improvement("Book F phase-tranching (4 cohorts)",
            "Hoffstein et al. 2019 JII; JT 1993 (registry #86)",
            metrics_from_returns(champ.values, TD), mp, ["book_f_tranched"],
            f"p={rr['p_one_sided']:.3f}; F standalone {mtr['sharpe']:.2f} vs 1.548")
        print("  recorded book_f_tranched")

# ═════════ (b) D8 threshold barbell ═════════
print(f"\n=== (b) D8 threshold barbell 0.7/0.8/0.9 ({time.time()-t0:.0f}s) ===")
from strategies.contrarian_bubble_hourly import run_contrarian_bubble_hourly
sleeves = {}
for thr in (0.7, 0.8, 0.9):
    dts, _, _ = run_contrarian_bubble_hourly(hoo, hcc, ma_window_grid=[104],
        buy_threshold_grid=[thr], hold_hours_grid=[8], top_n_grid=[20])
    sleeves[thr] = dts.dropna()
m08 = metrics_from_returns(sleeves[0.8][sleeves[0.8].index <= END].values, TD)
print(f"  anchor (0.8): {m08['sharpe']:.3f} {'OK' if abs(m08['sharpe']-2.740)<0.05 else 'FAIL - VOID (b)'}")
if abs(m08['sharpe']-2.740) < 0.05:
    idxu = sleeves[0.8].index
    for s_ in sleeves.values(): idxu = idxu.union(s_.index)
    d8_bar = sum(s_.reindex(idxu).fillna(0) for s_ in sleeves.values()) / 3
    dblend_bar = 0.5*d8_bar.reindex(ju.union(idxu)).fillna(0) + 0.5*d14.reindex(ju.union(idxu)).fillna(0)
    books = dict(BASE); books["D"] = dblend_bar
    ser = champ_with(books, ["A","C","D","F"], {"D": 2.0})
    mp = metrics_from_returns(ser.values, TD)
    rr = sharpe_delta_test(ser, champ)
    print(f"  champ with D-barbell: Sharpe {mp['sharpe']:.3f} (delta {rr['delta']:+.3f}, p {rr['p_one_sided']:.3f})"
          f"{'  <-- CANDIDATE' if rr['significant_p10'] else ''}")
    if rr["significant_p10"]:
        record_performance(name="book_d8_barbell", dates=d8_bar.index, returns=d8_bar.values,
            params={"thresholds": [0.7, 0.8, 0.9], "ma": 104, "hold": 8},
            data_period=f"{d8_bar.index.min().date()}..{d8_bar.index.max().date()}", periods_per_year=TD,
            extra={"cycle": "Cycle 16b"})
        record_improvement("D8 threshold barbell (0.7/0.8/0.9 ensemble)",
            "Parameter-ensemble theory (registry #86/#100)",
            metrics_from_returns(champ.values, TD), mp, ["book_d8_barbell"],
            f"p={rr['p_one_sided']:.3f}")
        print("  recorded book_d8_barbell")

# ═════════ (c) cross-asset capitulation ═════════
print(f"\n=== (c) cross-asset capitulation ({time.time()-t0:.0f}s) ===")
etf = pd.read_parquet("data/cache/etf_daily_close.parquet"); etf.index = pd.to_datetime(etf.index)
re_ = etf.pct_change()
lp = np.log(etf.replace(0, np.nan))
fair = etf.rolling(50).mean()
res = lp - np.log(fair)
z = (res - res.rolling(50).mean()) / res.rolling(50).std()
score = np.tanh(z/2).shift(1)
dates = etf.index; n2 = len(dates)
rows = []; i = 120
while i + 10 < n2:
    d = dates[i]; s_ = score.loc[d].dropna()
    deep = s_[s_ < -0.8]
    if len(deep) >= 1:
        picks = deep.nsmallest(2).index
        fwd = re_.iloc[i:i+10][picks]
        pr = fwd.mean(axis=1).fillna(0.0); pr.iloc[0] -= 2*TC
        for dt, x in pr.items(): rows.append((dt, float(x)))
        i += 10
    else:
        rows.append((d, 0.0)); i += 1
s_c = pd.Series(dict(rows)).sort_index(); s_c = s_c[~s_c.index.duplicated(keep="last")]
ms = metrics_from_returns(s_c.values, TD)
print(f"  standalone: Sharpe {ms['sharpe']:.2f} | CAGR {ms['cagr']:.1%} | MaxDD {ms['max_dd']:.0%} | "
      f"in-market {(s_c != 0).mean():.0%}")
books = dict(BASE); books["N"] = s_c
ser = champ_with(books, ["A","C","D","F","N"], {"D": 2.0, "N": 0.25})
mp = metrics_from_returns(ser.values, TD)
rr = sharpe_delta_test(ser, champ)
print(f"  champ+N @0.25: Sharpe {mp['sharpe']:.3f} (delta {rr['delta']:+.3f}, p {rr['p_one_sided']:.3f})"
      f"{'  <-- CANDIDATE' if rr['significant_p10'] else ''}")
if rr["significant_p10"]:
    record_performance(name="book_xasset_capit", dates=s_c.index, returns=s_c.values,
        params={"ma": 50, "z": 50, "thr": -0.8, "hold": 10, "max_n": 2},
        data_period=f"{s_c.index.min().date()}..{s_c.index.max().date()}", periods_per_year=TD,
        extra={"cycle": "Cycle 16c", "origin": "SYNTHESIS: D bubble score on ETF panel"})
    print("  recorded book_xasset_capit")

# ═════════ (d) sector mean reversion ═════════
print(f"\n=== (d) sector mean reversion ===")
sec = pd.read_parquet("data/cache/sector_etf_close.parquet"); sec.index = pd.to_datetime(sec.index)
rs = sec.pct_change()
r21 = sec.pct_change(21).shift(1)
dates = sec.index; n3 = len(dates)
rows = []; i = 60
while i + 21 < n3:
    d = dates[i]; m_ = r21.loc[d].dropna()
    if len(m_) >= 6:
        picks = m_.nsmallest(2).index
        fwd = rs.iloc[i:i+21][picks]
        pr = fwd.mean(axis=1).fillna(0.0); pr.iloc[0] -= 2*TC
        for dt, x in pr.items(): rows.append((dt, float(x)))
        i += 21
    else:
        rows.append((d, 0.0)); i += 1
s_d = pd.Series(dict(rows)).sort_index(); s_d = s_d[~s_d.index.duplicated(keep="last")]
ms = metrics_from_returns(s_d[s_d.index >= "2000-06-01"].values, TD)
print(f"  standalone: Sharpe {ms['sharpe']:.2f} | CAGR {ms['cagr']:.1%} | MaxDD {ms['max_dd']:.0%}")
books = dict(BASE); books["N"] = s_d
ser = champ_with(books, ["A","C","D","F","N"], {"D": 2.0, "N": 0.25})
mp = metrics_from_returns(ser.values, TD)
rr = sharpe_delta_test(ser, champ)
print(f"  champ+N @0.25: Sharpe {mp['sharpe']:.3f} (delta {rr['delta']:+.3f}, p {rr['p_one_sided']:.3f})"
      f"{'  <-- CANDIDATE' if rr['significant_p10'] else ''}")
print(f"\ntotal {time.time()-t0:.0f}s")
