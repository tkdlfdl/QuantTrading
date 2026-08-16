"""
Improvement Cycle 12 — three introspection-driven candidates (registry bench
notes; the origin class that produced both prior wins).

 (a) HAR-X overlay: Cycle 7's HAR failed on the noisy daily r^2 proxy. Now use
     SPY INTRADAY realized vol (sum of squared hourly returns per day — real RV,
     newly available from the SPY ingestion) as an exogenous regressor:
       log(port_r2_{t+1}) ~ [1, log port r2 1/5/22d, log SPY RV_d]
     expanding OLS, refit 21d, forecasts lagged. De-risk-only scale as before.
 (b) Book H cash-sleeve: H (breadth capitulation, standalone 0.896) was benched
     because ivol over-weights its 67%-cash profile. Instead: fund H ONLY from
     the vol-target's idle cash — port = scale*champ + min(1-scale, 0.5)*H.
     Gross <= 1 always (no leverage); H gets capital only when the overlay has
     parked some AND H is in a trade.
 (c) D-aggressive sleeve: official D engine at hold=14h (Cycle 3: Sharpe 2.64,
     total +872%, MDD -12.1%) added as a FIFTH book next to D8 —
     CAGR play with ivol handling the risk balance.

All judged on the recorded-series window (through 2026-07-08) vs champion
anchor 2.580 / 43.5% / -8.9%. Signals lagged; anchors enforced.
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from tools.metrics import metrics_from_returns
from tools.record import record_performance, record_improvement

TD, TGT = 252, 0.15
CHAMP = dict(sharpe=2.580, cagr=0.435, max_dd=-0.089)
END = "2026-07-08"   # recorded-series window for apples-to-apples

def load(n):
    df = pd.read_csv(f"strategies/performance/{n}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)

BOOKS = {"A": load("book_a_retest"), "C": load("book_c_overlap_cap"),
         "D": load("book_d_retest"), "F": load("book_f_retest")}

daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)
spy_d = daily["SPY"].dropna()

def build_champ(books, ret_scale=None):
    idx = sorted(set().union(*[s.index for s in books.values()]))
    idx = pd.DatetimeIndex([d for d in idx if pd.Timestamp("2019-01-02") <= d <= pd.Timestamp(END)])
    R = pd.DataFrame({k: books[k].reindex(idx).fillna(0.0) for k in books})
    sr = spy_d.pct_change()
    panic = ((spy_d.pct_change(504) < 0) &
             (sr.rolling(63).std()*np.sqrt(TD) >
              (sr.rolling(63).std()*np.sqrt(TD)).rolling(756).quantile(0.8))
            ).shift(1).reindex(idx).fillna(False)
    cols = list(R.columns)
    iA, iF = cols.index("A"), cols.index("F")
    iDs = [i for i, c in enumerate(cols) if c.startswith("D")]
    n = len(R); port = np.zeros(n); w = None
    for t in range(n):
        if w is None or t % 21 == 0:
            hist = R.iloc[max(0, t-60):t]
            v = hist.std() * np.sqrt(TD)
            iv = np.array([1.0/x if np.isfinite(x) and x > 1e-9 else 0.0 for x in v])
            w = iv/iv.sum() if iv.sum() else np.ones(len(cols))/len(cols)
        wt = w.copy()
        if bool(panic.iloc[t]):
            freed = 0.5*(wt[iA]+wt[iF]); wt[iA] *= 0.5; wt[iF] *= 0.5
            for i_ in iDs: wt[i_] += freed/len(iDs)
        port[t] = float(wt @ R.iloc[t].values)
    pre = pd.Series(port, index=idx)
    rv = pre.rolling(20).std().shift(1) * np.sqrt(TD)
    scale = (TGT/rv).clip(upper=1.0).fillna(1.0)
    return pre, scale, pre*scale

pre0, scale0, anchor = build_champ(BOOKS)
m0 = metrics_from_returns(anchor.values, TD)
print(f"anchor: Sharpe {m0['sharpe']:.3f} | CAGR {m0['cagr']:.1%} | MaxDD {m0['max_dd']:.1%} "
      f"{'OK' if abs(m0['sharpe']-CHAMP['sharpe'])<0.05 else 'FAILED - VOID'}")
if abs(m0["sharpe"] - CHAMP["sharpe"]) >= 0.05: sys.exit(1)
idx = anchor.index

# ═════════ (a) HAR-X with SPY intraday RV ═════════
print("\n=== (a) HAR-X overlay (SPY intraday RV regressor) ===")
hc = pd.read_parquet("data/cache/merged_hourly_close.parquet")
hc.index = pd.to_datetime(hc.index)
spy_h = hc["SPY"].dropna()
hr = spy_h.pct_change()
rv_spy = (hr**2).groupby(hr.index.normalize()).sum()      # true daily RV from hourly
rv_spy.index = pd.to_datetime(rv_spy.index)
rv_spy = rv_spy.reindex(idx)
r2 = (pre0**2).values
EPS = 1e-10
n = len(idx)
lx = np.log(np.maximum(r2, EPS))
xw = np.log(np.maximum(pd.Series(r2).rolling(5).mean().values, EPS))
xm = np.log(np.maximum(pd.Series(r2).rolling(22).mean().values, EPS))
xs = np.log(np.maximum(rv_spy.values, EPS))               # SPY intraday RV (exog)
has_spy = np.isfinite(rv_spy.values) & (rv_spy.values > 0)
X_all = np.column_stack([np.ones(n), lx, xw, xm, xs])
vhat = np.full(n, np.nan); beta = None; s2r = 0.0
for t in range(n):
    if t >= 252 and t % 21 == 0:
        rows = np.arange(22, t-1)
        rows = rows[has_spy[rows]]
        X, y = X_all[rows], lx[rows+1]
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        if ok.sum() > 60:
            b, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
            s2r = float((y[ok] - X[ok] @ b).var()); beta = b
    if beta is not None and has_spy[t-1] and np.isfinite(X_all[t-1]).all():
        vhat[t] = np.sqrt(np.exp(float(X_all[t-1] @ beta) + 0.5*s2r) * TD)
trail = (pre0.rolling(20).std().shift(1) * np.sqrt(TD)).values
sig = np.where(np.isfinite(vhat), vhat, trail)            # fallback where no forecast
scale_a = np.minimum(1.0, TGT/np.where(np.isfinite(sig) & (sig > 1e-9), sig, np.inf))
scale_a = np.where(np.isfinite(sig), scale_a, 1.0)
ser_a = pre0 * scale_a
m_a = metrics_from_returns(ser_a.values, TD)
print(f"  HAR-X: Sharpe {m_a['sharpe']:.3f} | CAGR {m_a['cagr']:.1%} | MaxDD {m_a['max_dd']:.1%} | "
      f"avg expo {np.nanmean(scale_a):.0%}")
if m_a["sharpe"] > m0["sharpe"] + 0.05 and m_a["max_dd"] >= CHAMP["max_dd"]*1.2:
    record_performance(name="portfolio_champ_harx", dates=ser_a.index, returns=ser_a.values,
        params={"overlay": "HAR-X (SPY intraday RV)", "target": TGT},
        data_period=f"{idx.min().date()}..{idx.max().date()}", periods_per_year=TD,
        extra={"cycle": "Cycle 12a", "paper": "Corsi 2009 HAR-X; own SPY-hourly RV"})
    record_improvement("HAR-X vol forecast (SPY intraday RV) for champion overlay",
        "Corsi 2009 + own intraday RV (registry #60 revisit)", CHAMP, m_a,
        ["portfolio_champ_harx"])
    print("  IMPROVED -> recorded")
else:
    print("  no gain")

# ═════════ (b) Book H cash-sleeve ═════════
print("\n=== (b) Book H funded from vol-target idle cash ===")
H = load("book_h_breadth_timing").reindex(idx).fillna(0.0)
sleeve = np.minimum(1.0 - scale0.values, 0.5)             # idle cash, capped 50%
ser_b = anchor + pd.Series(sleeve, index=idx) * H
m_b = metrics_from_returns(ser_b.values, TD)
used = float(np.mean((sleeve > 0.01) & (H.values != 0)))
print(f"  champ + H-sleeve: Sharpe {m_b['sharpe']:.3f} | CAGR {m_b['cagr']:.1%} | "
      f"MaxDD {m_b['max_dd']:.1%} | sleeve active {used:.1%} of days")
if (m_b["sharpe"] > m0["sharpe"] + 0.05 or m_b["cagr"] > m0["cagr"] + 0.05) \
        and m_b["max_dd"] >= CHAMP["max_dd"]*1.2:
    record_performance(name="portfolio_champ_hsleeve", dates=ser_b.index, returns=ser_b.values,
        params={"sleeve": "min(1-scale, 0.5) x H", "H": "breadth capitulation"},
        data_period=f"{idx.min().date()}..{idx.max().date()}", periods_per_year=TD,
        extra={"cycle": "Cycle 12b", "origin": "ORIGINAL (fixes ivol cash-drag artifact)"})
    record_improvement("Book H funded from vol-target idle cash (conditional sleeve)",
        "ORIGINAL — registry #83 revisit", CHAMP, m_b, ["portfolio_champ_hsleeve"])
    print("  IMPROVED -> recorded")
else:
    print("  no gain")

# ═════════ (c) D14 aggressive sleeve as 5th book ═════════
print("\n=== (c) D14 sleeve (official engine, hold=14h) ===")
from strategies.contrarian_bubble_hourly import run_contrarian_bubble_hourly
ho = pd.read_parquet("data/cache/merged_hourly_open.parquet"); ho.index = pd.to_datetime(ho.index)
common = sorted((set(ho.columns) & set(hc.columns)) - {"SPY"})
d14, _, _ = run_contrarian_bubble_hourly(ho[common], hc[common],
    ma_window_grid=[104], buy_threshold_grid=[0.8], hold_hours_grid=[14], top_n_grid=[20])
d14 = d14.dropna(); d14 = d14[d14.index <= END]
m14 = metrics_from_returns(d14.values, TD)
print(f"  D14 standalone: Sharpe {m14['sharpe']:.3f} | CAGR {m14['cagr']:.1%} | MaxDD {m14['max_dd']:.1%}")
books5 = dict(BOOKS); books5["D14"] = d14
_, _, ser_c = build_champ(books5)
m_c = metrics_from_returns(ser_c.values, TD)
print(f"  champ + D14: Sharpe {m_c['sharpe']:.3f} | CAGR {m_c['cagr']:.1%} | MaxDD {m_c['max_dd']:.1%}")
if (m_c["sharpe"] > m0["sharpe"] + 0.05 or m_c["cagr"] > m0["cagr"] + 0.05) \
        and m_c["max_dd"] >= CHAMP["max_dd"]*1.2:
    record_performance(name="portfolio_champ_d14", dates=ser_c.index, returns=ser_c.values,
        params={"books": "A,C,D8,D14,F"},
        data_period=f"{ser_c.index.min().date()}..{ser_c.index.max().date()}", periods_per_year=TD,
        extra={"cycle": "Cycle 12c", "origin": "registry #45 note (D 14h variant)"})
    record_improvement("D14 aggressive sleeve added as fifth book",
        "Own grid finding (registry #45 note)", CHAMP, m_c, ["portfolio_champ_d14"])
    print("  IMPROVED -> recorded")
else:
    print("  no gain")
