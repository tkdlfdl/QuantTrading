"""
Cycle 15 — four new-book candidates on in-hand data (no new sweep needed).

 (i)   Sector rotation: top-3 of 11 SPDR sectors by 12-1 momentum, monthly.
 (ii)  Cross-asset weekly reversal: long worst-3 of 15 ETFs by 5d return, 5d hold.
 (iii) D x momentum double-sort (SYNTHESIS): official D engine, but eligible
       set masked to stocks with positive 126d momentum ("dips in uptrends").
       Anchor: unmasked run must reproduce 2.740.
 (iv)  Credit-regime rotation: HYG/LQD 21d relative strength -> SPY else TLT,
       weekly check.

All: 0.1%/side, signals lagged, judged standalone + champ delta @0.25 shares
with LW significance. Champion baseline 2.781 (portfolio_champ_d14).
"""
from __future__ import annotations
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import yfinance as yf
from tools.metrics import metrics_from_returns, sharpe_ratio, max_drawdown
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
worst = champ.nsmallest(int(len(champ)*0.05)).index

def judge(name, stream, record_name=None, paper=""):
    s19 = stream.reindex(champ.index).fillna(0)
    cc = float(np.corrcoef(s19, champ)[0, 1])
    ms = metrics_from_returns(stream.dropna().values, TD)
    saved_b, saved_s = C.ALLOC_BOOKS, dict(C.ALLOC_SHARES)
    C.ALLOC_BOOKS = ["A","C","D","F","N"]; C.ALLOC_SHARES = {"D": 2.0, "N": 0.25}
    try:
        books = dict(BASE); books["N"] = stream
        R = pd.DataFrame(books)
        R = R[(R.index >= "2019-01-02") & (R.index <= END)].fillna(0.0)
        ser = E.ivol_voltgt(R).dropna()
    finally:
        C.ALLOC_BOOKS, C.ALLOC_SHARES = saved_b, saved_s
    mp = metrics_from_returns(ser.values, TD)
    rr = sharpe_delta_test(ser, champ)
    sd = float(stream.reindex(worst).fillna(0).mean())
    print(f"  standalone Sharpe {ms['sharpe']:.2f} CAGR {ms['cagr']:.1%} MaxDD {ms['max_dd']:.0%} | "
          f"corr {cc:+.2f} | stress {sd:+.2%}/d")
    print(f"  champ+N @0.25: Sharpe {mp['sharpe']:.3f} (delta {rr['delta']:+.3f}, p {rr['p_one_sided']:.3f})"
          f"{'  <-- CANDIDATE' if rr['significant_p10'] and mp['max_dd'] >= -0.10 else ''}")
    if record_name and rr["significant_p10"] and mp["max_dd"] >= -0.10:
        record_performance(name=record_name, dates=stream.dropna().index, returns=stream.dropna().values,
            params={}, data_period=f"{stream.index.min().date()}..{stream.index.max().date()}",
            periods_per_year=TD, extra={"cycle": "Cycle 15", "paper": paper})
        print(f"  recorded {record_name}")
    return rr

# ═════════ (i) sector rotation ═════════
print("=== (i) sector rotation (11 SPDR sectors) ===")
SECT = ["XLK","XLE","XLF","XLV","XLI","XLP","XLY","XLB","XLU","XLRE","XLC"]
sec = yf.download(SECT, start="1999-01-01", interval="1d", auto_adjust=True, progress=False)["Close"]
sec.index = pd.to_datetime(sec.index).tz_localize(None)
sec.to_parquet("data/cache/sector_etf_close.parquet")
rs = sec.pct_change()
mom = sec.pct_change(252).shift(21)
month_end = sec.index.to_series().dt.month.diff().fillna(1) != 0
targets = pd.DataFrame(index=sec.index[month_end], columns=sec.columns, dtype=float)
for d in targets.index:
    m_ = mom.loc[d].dropna()
    row = pd.Series(0.0, index=sec.columns)
    if len(m_) >= 6:
        for t in m_.nlargest(3).index: row[t] = 1.0/3
    targets.loc[d] = row
pos = targets.reindex(sec.index).ffill().fillna(0.0).shift(1)
s_i = ((rs.fillna(0)*pos).sum(axis=1) - pos.diff().abs().sum(axis=1).fillna(0)*TC)
s_i = s_i[s_i.index >= "2000-06-01"].dropna()
judge("sector rotation", s_i, "book_sector_rot", "sector momentum (SPDR)")

# ═════════ (ii) cross-asset weekly reversal ═════════
print(f"\n=== (ii) cross-asset weekly reversal ===")
etf = pd.read_parquet("data/cache/etf_daily_close.parquet")
etf.index = pd.to_datetime(etf.index)
re_ = etf.pct_change()
r5 = etf.pct_change(5).shift(1)
dates = etf.index; n = len(dates)
rows = []; i = 260
while i + 5 < n:
    d = dates[i]; m_ = r5.loc[d].dropna()
    if len(m_) >= 8:
        picks = m_.nsmallest(3).index
        fwd = re_.iloc[i:i+5][picks]
        pr = fwd.mean(axis=1).fillna(0.0); pr.iloc[0] -= 2*TC
        for dt, x in pr.items(): rows.append((dt, float(x)))
        i += 5
    else:
        rows.append((d, 0.0)); i += 1
s_ii = pd.Series(dict(rows)).sort_index()
s_ii = s_ii[~s_ii.index.duplicated(keep="last")]
judge("xasset weekly reversal", s_ii, "book_xasset_rev", "cross-asset ST reversal")

# ═════════ (iii) D x momentum double-sort ═════════
print(f"\n=== (iii) D x momentum double-sort (dips in uptrends) ===")
import strategies.contrarian_bubble_hourly as CB
hc = pd.read_parquet("data/cache/merged_hourly_close.parquet"); hc.index = pd.to_datetime(hc.index)
ho = pd.read_parquet("data/cache/merged_hourly_open.parquet"); ho.index = pd.to_datetime(ho.index)
common = sorted((set(ho.columns) & set(hc.columns)) - {"SPY"})
hcc, hoo = hc[common], ho[common]
B_raw = CB._bubble_matrix(hcc, 104)
mom126d = hcc.pct_change(126*7).values          # ~126 trading days in hourly bars
_orig = CB._bubble_matrix
MASK = None
def _patched(close, ma):
    S = B_raw.copy()
    if MASK is not None:
        S = np.where(MASK, S, 0.0)              # ineligible -> score 0 (never < -0.8)
    return S.astype(np.float32)
CB._bubble_matrix = _patched
# anchor
MASK = None
dA, _, _ = CB.run_contrarian_bubble_hourly(hoo, hcc, ma_window_grid=[104],
    buy_threshold_grid=[0.8], hold_hours_grid=[8], top_n_grid=[20])
mA = metrics_from_returns(dA.dropna()[dA.dropna().index <= END].values, TD)
print(f"  anchor: {mA['sharpe']:.3f} {'OK' if abs(mA['sharpe']-2.740)<0.05 else 'FAIL - VOID'}")
if abs(mA['sharpe']-2.740) < 0.05:
    MASK = mom126d > 0                          # only dips in uptrends
    dS, _, _ = CB.run_contrarian_bubble_hourly(hoo, hcc, ma_window_grid=[104],
        buy_threshold_grid=[0.8], hold_hours_grid=[8], top_n_grid=[20])
    dS = dS.dropna(); dS = dS[dS.index <= END]
    mS = metrics_from_returns(dS.values, TD)
    print(f"  D8-uptrend-only standalone: Sharpe {mS['sharpe']:.3f} | CAGR {mS['cagr']:.1%} | MaxDD {mS['max_dd']:.1%}")
    judge("D-dip-in-uptrend as extra book", dS, None, "")
CB._bubble_matrix = _orig

# ═════════ (iv) credit-regime rotation ═════════
print(f"\n=== (iv) credit-regime rotation (HYG/LQD -> SPY|TLT) ===")
hyg, lqd = etf["HYG"].dropna(), etf["LQD"].dropna()
spy_e, tlt = etf["SPY"].dropna(), etf["TLT"].dropna()
relstr = (hyg/lqd).pct_change(21)
week = etf.index.to_series().dt.isocalendar().week.diff().fillna(1) != 0
risk_on = (relstr > 0).where(week).ffill().shift(1).fillna(False)
r_spy, r_tlt = spy_e.pct_change(), tlt.pct_change()
s_iv = pd.Series(np.where(risk_on.reindex(etf.index), r_spy.reindex(etf.index), r_tlt.reindex(etf.index)), index=etf.index)
chg = risk_on.astype(float).diff().abs().fillna(0).reindex(etf.index).fillna(0)
s_iv = (s_iv - chg*2*TC).dropna()
s_iv = s_iv[s_iv.index >= "2008-01-01"]
judge("credit-regime rotation", s_iv, "book_credit_regime", "HYG/LQD risk switch")
print(f"\ntotal {time.time()-t0:.0f}s")
