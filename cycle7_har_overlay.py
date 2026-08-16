"""
Improvement Cycle 7 — queue #11 (+#15): better vol forecasts for the champion's
vol-target overlay.

The overlay scale = min(1, 15% / sigma_hat) is the champion's biggest lever;
a better sigma_hat transmits directly to the frontier (Corsi 2009 HAR; Bates-
Granger 1969 combination). Variants, all strictly leak-free (forecast for day t
uses data through t-1 only; HAR refit every 21d on EXPANDING window, min 252 obs):

  trail20   : 20d trailing std (CURRENT — sanity anchor, must give 2.580)
  ewma94    : RiskMetrics EWMA lambda=0.94
  har       : HAR on log daily r^2 proxy (1/5/22d components), log-normal
              bias-corrected, expanding OLS
  committee : inverse-QLIKE-weighted combo of the three (trailing 63d QLIKE,
              weights lagged 1 day)   [#15]

Judged on the pre-overlay champion series (ivol + DM panic gate over
A / C-capped / D / F). Baseline champion: 2.580 / 43.5% / -8.9%.
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from tools.metrics import metrics_from_returns
from tools.record import record_performance, record_improvement

TD = 252
TGT = 0.15
CHAMP = dict(sharpe=2.580, cagr=0.435, max_dd=-0.089)

def load(n):
    df = pd.read_csv(f"strategies/performance/{n}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)

BOOKS = {"A": load("book_a_retest"), "C": load("book_c_overlap_cap"),
         "D": load("book_d_retest"), "F": load("book_f_retest")}
idx = sorted(set().union(*[s.index for s in BOOKS.values()]))
idx = pd.DatetimeIndex([d for d in idx if d >= pd.Timestamp("2019-01-02")])
R = pd.DataFrame({k: BOOKS[k].reindex(idx).fillna(0.0) for k in BOOKS})

daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)
spy = daily["SPY"].dropna()
ret24 = spy.pct_change(504)
vol63 = spy.pct_change().rolling(63).std() * np.sqrt(TD)
q80 = vol63.rolling(756).quantile(0.80)
panic = ((ret24 < 0) & (vol63 > q80)).shift(1).reindex(idx).fillna(False)

# pre-overlay champion series (ivol + panic gate, no vol scaling)
cols = list(R.columns); iA, iF, iD = cols.index("A"), cols.index("F"), cols.index("D")
n = len(R); port = np.zeros(n); w = None
for t in range(n):
    if w is None or t % 21 == 0:
        hist = R.iloc[max(0, t-60):t]
        vol = hist.std() * np.sqrt(TD)
        iv = np.array([1.0/v if np.isfinite(v) and v > 1e-9 else 0.0 for v in vol])
        w = iv/iv.sum() if iv.sum() else np.ones(len(cols))/len(cols)
    wt = w.copy()
    if bool(panic.iloc[t]):
        freed = 0.5*(wt[iA]+wt[iF]); wt[iA] *= 0.5; wt[iF] *= 0.5; wt[iD] += freed
    port[t] = float(wt @ R.iloc[t].values)
base = pd.Series(port, index=idx)
r2 = (base**2).values
EPS = 1e-10

# ── forecast variants: sigma_hat[t] = annualized vol forecast for day t ─────
def f_trail20():
    return (base.rolling(20).std().shift(1) * np.sqrt(TD)).values

def f_ewma94():
    v = np.full(n, np.nan); s2 = None
    for t in range(1, n):
        x = r2[t-1]
        s2 = x if s2 is None else 0.94*s2 + 0.06*x
        v[t] = np.sqrt(s2 * TD)
    return v

def f_har(refit=21, min_obs=252):
    lx = np.log(np.maximum(r2, EPS))
    x_d = pd.Series(lx, index=idx)
    x_w = pd.Series(np.log(np.maximum(pd.Series(r2).rolling(5).mean().values, EPS)), index=idx)
    x_m = pd.Series(np.log(np.maximum(pd.Series(r2).rolling(22).mean().values, EPS)), index=idx)
    X_all = np.column_stack([np.ones(n), x_d.values, x_w.values, x_m.values])
    v = np.full(n, np.nan); beta = None; sig2_res = 0.0
    for t in range(n):
        if t >= min_obs and (t % refit == 0 or beta is None):
            # fit on rows 22..t-1: predict lx[i+1] from X_all[i]
            rows = np.arange(22, t-1)
            X, y = X_all[rows], lx[rows+1]
            ok = np.isfinite(X).all(1) & np.isfinite(y)
            if ok.sum() > 60:
                b, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
                res = y[ok] - X[ok] @ b
                beta, sig2_res = b, float(res.var())
        if beta is not None and np.isfinite(X_all[t-1]).all():
            ly = float(X_all[t-1] @ beta) + 0.5*sig2_res     # log-normal correction
            v[t] = np.sqrt(np.exp(ly) * TD)
    return v

def qlike(s2hat, x2):
    ratio = np.maximum(x2, EPS) / np.maximum(s2hat, EPS)
    return ratio - np.log(ratio) - 1.0

def f_committee(members):
    V = np.column_stack(members)                # annualized sigma forecasts
    S2 = (V**2) / TD                            # daily variance forecasts
    L = np.full_like(S2, np.nan)
    for j in range(S2.shape[1]):
        L[:, j] = qlike(S2[:, j], r2)
    Ldf = pd.DataFrame(L).rolling(63).mean().shift(1).values   # lagged avg loss
    v = np.full(n, np.nan)
    for t in range(n):
        lt, st = Ldf[t], S2[t]
        ok = np.isfinite(lt) & np.isfinite(st)
        if ok.sum() == 0: continue
        wgt = 1.0/np.maximum(lt[ok], 1e-6); wgt /= wgt.sum()
        v[t] = np.sqrt(float(wgt @ st[ok]) * TD)
    return v

variants = {}
variants["trail20"] = f_trail20()
variants["ewma94"] = f_ewma94()
variants["har"] = f_har()
variants["committee"] = f_committee([variants["trail20"], variants["ewma94"], variants["har"]])

print(f"{'variant':<11}{'Sharpe':>8}{'CAGR':>8}{'MaxDD':>8}{'avg expo':>10}")
res = {}
for name, v in variants.items():
    scale = np.minimum(1.0, TGT / np.where(np.isfinite(v) & (v > 1e-9), v, np.inf))
    scale = np.where(np.isfinite(v), scale, 1.0)
    ser = base * scale
    m = metrics_from_returns(ser.values, TD)
    res[name] = (ser, m, scale)
    print(f"{name:<11}{m['sharpe']:>8.3f}{m['cagr']:>8.1%}{m['max_dd']:>8.1%}{scale.mean():>10.0%}")

anchor = res["trail20"][1]
if abs(anchor["sharpe"] - CHAMP["sharpe"]) > 0.05:
    print(f"\nANCHOR FAILED ({anchor['sharpe']:.3f} vs 2.580) — VOID"); sys.exit(1)
print("\nanchor OK")

best_name = max([k for k in res if k != "trail20"], key=lambda k: res[k][1]["sharpe"])
ser_b, m_b, _ = res[best_name]
if m_b["sharpe"] > anchor["sharpe"] + 0.05 and m_b["max_dd"] >= CHAMP["max_dd"]*1.2:
    # robustness: winner across target vols
    print(f"\nrobustness ({best_name}):")
    wins = tot = 0
    v = variants[best_name]
    for tgt in [0.12, 0.15, 0.18]:
        scale = np.minimum(1.0, tgt / np.where(np.isfinite(v) & (v > 1e-9), v, np.inf))
        scale = np.where(np.isfinite(v), scale, 1.0)
        m = metrics_from_returns((base*scale).values, TD)
        ok = m["sharpe"] >= anchor["sharpe"] and m["max_dd"] >= -0.12
        wins += ok; tot += 1
        print(f"  tgt {tgt:.0%}: Sharpe {m['sharpe']:.3f} MaxDD {m['max_dd']:.1%} {'ok' if ok else '--'}")
    if wins >= 2:
        nm = f"portfolio_champ_{best_name}"
        record_performance(name=nm, dates=ser_b.index, returns=ser_b.values,
            params={"overlay": best_name, "target": TGT},
            data_period=f"{idx.min().date()}..{idx.max().date()}", periods_per_year=TD,
            extra={"cycle": "Cycle 7 #11/#15",
                   "paper": "Corsi 2009 HAR; Bates-Granger 1969 combination"})
        record_improvement(f"Vol-target overlay forecast upgraded to {best_name}",
            "Corsi 2009 J.Fin.Econometrics; Bates & Granger 1969 (registry #60, #77)",
            CHAMP, m_b, [nm], f"robustness {wins}/{tot} target-vol grid")
        print(f"IMPROVED -> recorded {nm}")
    else:
        print("improvement not robust across targets — not recorded")
else:
    print(f"\nno gain: best alt {best_name} {m_b['sharpe']:.3f} vs anchor {anchor['sharpe']:.3f}")
