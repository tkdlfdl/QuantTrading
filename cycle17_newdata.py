"""
Cycle 17 — NEW DATA + three strategies on it.

DATA (all free yfinance daily, cached):
  - Yield curve: ^TNX ^FVX ^TYX ^IRX (10y/5y/30y/13w)          -> yields_daily.parquet
  - Extended ETF universe: current 15 + countries (EWJ EWG EWU EWZ FXI INDA
    EWY EWT) + GDX XME TIP SHY + crypto (BTC-USD ETH-USD)      -> etf_extended_close.parquet

STRATEGIES:
 (a) Expanded cross-asset momentum: X's exact spec (12-1, top-3, monthly,
     ungated) on the ~29-asset universe. Redundancy-checked vs current X.
 (b) Crypto trend book: BTC+ETH 50/50, long/flat by 20d>100d MA cross, daily
     check, 0.25% costs (crypto spreads). Satellite @0.25 shares on FULL stack.
 (c) Curve tilt: 10y-13w slope momentum (21d change) -> long TLT when curve
     rallying (yields falling), else flat. Satellite test.

Gates: standard (standalone + corr + champ delta + LW p) PLUS full-stack
marginal (champ+G+X+DU) — the Cycle 16 lesson.
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
from tools.record import record_performance
from live import engine as E, config as C

TD, TC, END = 252, 0.001, "2026-07-08"
t0 = time.time()

# ══════════ DATA LOAD ══════════
YIELDS = ["^TNX", "^FVX", "^TYX", "^IRX"]
yl = yf.download(YIELDS, start="1995-01-01", interval="1d", auto_adjust=False, progress=False)["Close"]
yl.index = pd.to_datetime(yl.index).tz_localize(None)
yl.to_parquet("data/cache/yields_daily.parquet")
print(f"yields: {yl.shape} {yl.index.min().date()}..{yl.index.max().date()}")

BASE15 = ["TLT","IEF","GLD","SLV","DBC","USO","UUP","EEM","EFA","IWM","VNQ","HYG","LQD","SPY","QQQ"]
NEW = ["EWJ","EWG","EWU","EWZ","FXI","INDA","EWY","EWT","GDX","XME","TIP","SHY","BTC-USD","ETH-USD"]
ext = yf.download(BASE15 + NEW, start="2002-01-01", interval="1d", auto_adjust=True, progress=False)["Close"]
ext.index = pd.to_datetime(ext.index).tz_localize(None)
# crypto trades 7d/wk — align to NYSE calendar via the SPY row mask
ext = ext[ext["SPY"].notna()]
ext.to_parquet("data/cache/etf_extended_close.parquet")
print(f"extended ETF panel: {ext.shape}; crypto from {ext['BTC-USD'].first_valid_index().date()}")

def load(n):
    df = pd.read_csv(f"strategies/performance/{n}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)
champ = load("portfolio_champ_d14")
d8, d14 = load("book_d_retest"), load("book_d14")
ju = d8.index.union(d14.index)
dblend = 0.5*d8.reindex(ju).fillna(0) + 0.5*d14.reindex(ju).fillna(0)
BASE = {"A": load("book_a_retest"), "C": load("book_c_overlap_cap"), "D": dblend, "F": load("book_f_retest")}
FULL = dict(BASE); FULL.update({"G": load("book_g_live_spec"), "X": load("book_x_live_spec"), "DU": load("book_du_live_spec")})
worst = champ.nsmallest(int(len(champ)*0.05)).index

def build_port(books, alloc, shares):
    saved_b, saved_s = C.ALLOC_BOOKS, dict(C.ALLOC_SHARES)
    C.ALLOC_BOOKS = alloc; C.ALLOC_SHARES = shares
    try:
        R = pd.DataFrame(books)
        R = R[(R.index >= "2019-01-02") & (R.index <= END)].fillna(0.0)
        return E.ivol_voltgt(R).dropna()
    finally:
        C.ALLOC_BOOKS, C.ALLOC_SHARES = saved_b, saved_s

full_stack = build_port(FULL, ["A","C","D","F","G","X","DU"], {"D":2.0,"G":.25,"X":.25,"DU":.25})

def judge(name, stream, crypto_tc=False):
    s19 = stream.reindex(champ.index).fillna(0)
    cc = float(np.corrcoef(s19, champ)[0,1])
    ms = metrics_from_returns(stream.dropna().values, TD)
    sd = float(stream.reindex(worst).fillna(0).mean())
    print(f"  standalone Sharpe {ms['sharpe']:.2f} CAGR {ms['cagr']:.1%} MaxDD {ms['max_dd']:.0%} | "
          f"corr(champ) {cc:+.2f} | stress {sd:+.2%}/d")
    # vs bare champion
    ser1 = build_port({**BASE, "N": stream}, ["A","C","D","F","N"], {"D":2.0,"N":.25})
    r1 = sharpe_delta_test(ser1, champ)
    # vs FULL stack (the decisive test)
    ser2 = build_port({**FULL, "N": stream}, ["A","C","D","F","G","X","DU","N"],
                      {"D":2.0,"G":.25,"X":.25,"DU":.25,"N":.25})
    r2 = sharpe_delta_test(ser2, full_stack)
    print(f"  vs champion: delta {r1['delta']:+.3f} p {r1['p_one_sided']:.3f} | "
          f"vs FULL stack: delta {r2['delta']:+.3f} p {r2['p_one_sided']:.3f}"
          f"{'  <-- ADDS ON FULL STACK' if r2['significant_p10'] else ''}")
    return r2

# ══════════ (a) expanded cross-asset momentum ══════════
print(f"\n=== (a) expanded X (29 assets incl. countries+crypto) ({time.time()-t0:.0f}s) ===")
r = ext.pct_change()
mom = ext.pct_change(252).shift(21)
month_end = ext.index.to_series().dt.month.diff().fillna(1) != 0
targets = pd.DataFrame(index=ext.index[month_end], columns=ext.columns, dtype=float)
for d in targets.index:
    m_ = mom.loc[d].dropna()
    row = pd.Series(0.0, index=ext.columns)
    if len(m_) >= 8:
        for t in m_.nlargest(3).index: row[t] = 1.0/3
    targets.loc[d] = row
pos = targets.reindex(ext.index).ffill().fillna(0.0).shift(1)
sa = ((r.fillna(0)*pos).sum(axis=1) - pos.diff().abs().sum(axis=1).fillna(0)*TC)
sa = sa[sa.index >= "2003-06-01"].dropna()
r2a = judge("expanded-X", sa)
x_cur = load("book_x_live_spec")
jx = sa.index.intersection(x_cur.index)
print(f"  corr(expanded-X, current X): {np.corrcoef(sa.loc[jx], x_cur.loc[jx])[0,1]:+.2f}")
# swap test: replace X with expanded in full stack
swap = dict(FULL); swap["X"] = sa
ser_sw = build_port(swap, ["A","C","D","F","G","X","DU"], {"D":2.0,"G":.25,"X":.25,"DU":.25})
rsw = sharpe_delta_test(ser_sw, full_stack)
print(f"  SWAP current X -> expanded: delta {rsw['delta']:+.3f} p {rsw['p_one_sided']:.3f}"
      f"{'  <-- UPGRADE CANDIDATE' if rsw['significant_p10'] else ''}")

# ══════════ (b) crypto trend book ══════════
print(f"\n=== (b) crypto trend (BTC+ETH, 20/100 MA long/flat) ===")
CTC = 0.0025
cr = ext[["BTC-USD","ETH-USD"]].copy()
crr = cr.pct_change()
sig = (cr.rolling(20).mean() > cr.rolling(100).mean()).shift(1).fillna(False)
pos_c = sig.astype(float) * 0.5
sb = (crr.fillna(0)*pos_c).sum(axis=1) - pos_c.diff().abs().sum(axis=1).fillna(0)*CTC
sb = sb[sb.index >= "2015-01-01"].dropna()
judge("crypto trend", sb)

# ══════════ (c) curve tilt ══════════
print(f"\n=== (c) yield-curve tilt (TLT when curve rallying) ===")
slope = (yl["^TNX"] - yl["^IRX"]).reindex(ext.index).ffill()
tnx_mom = yl["^TNX"].diff(21).reindex(ext.index).ffill()
sig_c = (tnx_mom < 0).shift(1).fillna(False)          # yields falling -> long TLT
tlt_r = ext["TLT"].pct_change()
sc = pd.Series(np.where(sig_c, tlt_r, 0.0), index=ext.index)
chg = sig_c.astype(float).diff().abs().fillna(0)
sc = (sc - chg*2*TC).dropna()
sc = sc[sc.index >= "2003-06-01"]
judge("curve tilt", sc)
print(f"\ntotal {time.time()-t0:.0f}s")
