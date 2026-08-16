"""
Portfolio-construction shootout on the honest retest series.

Methods (classic literature; citations reconciled in research/papers_read.md):
  ew         : Fixed equal weight (baseline; DeMiguel-Garlappi-Uppal 1/N)
  ivol       : Inverse trailing volatility (naive risk parity)
  erc        : Equal Risk Contribution (Maillard-Roncalli-Teiletche 2010)
  hrp        : Hierarchical Risk Parity (Lopez de Prado 2016)
  sharpe_w   : Trailing-Sharpe weighted (floor 0)
  ew_voltgt  : Fixed EW + portfolio vol-target overlay (Moreira-Muir style,
               de-risk only: scale = min(1, target_vol / realized_vol))
  ivol_voltgt: ivol + same overlay

Rules: no leverage (weights sum to 1; overlay only de-risks, unallocated = cash
at 0%). Estimation uses trailing windows only (no lookahead): weights computed
from data through t-1, applied at t. Rebalance every 21 trading days.
Book sets: ALL (A,B,C,D,F) and NO_B (drop-B decision). Inter-book reallocation
costs not charged — identical treatment to the documented Fixed EW baseline.

Improvement bar (registry rule): Sharpe > 2.060 or CAGR > 48.0% with
MaxDD <= ~1.2x baseline (-19.0% -> not worse than ~-23%).
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list

from tools.metrics import sharpe_ratio, max_drawdown, metrics_from_returns
from tools.record import record_performance

TD = 252
REBAL = 21
VOL_WIN = 60          # trailing window for vol/corr estimation
SHARPE_WIN = 120
TGT_VOL = 0.15        # 15% annualized portfolio vol target
START = "2019-01-02"

def load(name):
    df = pd.read_csv(f"strategies/performance/{name}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)

BOOKS = {k: load(f"book_{k}_retest") for k in ["a", "b", "c", "d", "f"]}

def build_panel(keys):
    idx = sorted(set().union(*[BOOKS[k].index for k in keys]))
    idx = pd.DatetimeIndex([d for d in idx if d >= pd.Timestamp(START)])
    R = pd.DataFrame({k: BOOKS[k].reindex(idx) for k in keys})
    first = {k: BOOKS[k].index.min() for k in keys}
    act = pd.DataFrame({k: [d >= first[k] for d in idx] for k in keys}, index=idx)
    return R.fillna(0.0), act

# ── weight engines (use data through t-1 only) ─────────────────────────────
def w_ew(R, act, t):
    a = act.iloc[t]
    w = a / a.sum() if a.sum() else a * 0.0
    return w.astype(float)

def w_ivol(R, act, t):
    a = act.iloc[t]
    hist = R.iloc[max(0, t-VOL_WIN):t]
    vol = hist.std() * np.sqrt(TD)
    iv = pd.Series(0.0, index=R.columns)
    for k in R.columns:
        if a[k] and vol[k] > 1e-9:
            iv[k] = 1.0 / vol[k]
    return iv / iv.sum() if iv.sum() else w_ew(R, act, t)

def w_erc(R, act, t, iters=200):
    a = act.iloc[t]
    keys = [k for k in R.columns if a[k]]
    hist = R.iloc[max(0, t-VOL_WIN):t][keys]
    if len(hist) < 20 or len(keys) < 2:
        return w_ew(R, act, t)
    S = hist.cov().values * TD
    nk = len(keys)
    x = np.ones(nk) / nk
    for _ in range(iters):                      # cyclical coordinate descent
        for i in range(nk):
            sig_x = S @ x
            if sig_x[i] <= 0: continue
            x[i] = x[i] * float(np.sqrt((x @ sig_x / nk) / (x[i] * sig_x[i]))) if x[i]*sig_x[i] > 0 else x[i]
        x = np.maximum(x, 1e-9); x = x / x.sum()
    w = pd.Series(0.0, index=R.columns); w[keys] = x
    return w

def w_hrp(R, act, t):
    a = act.iloc[t]
    keys = [k for k in R.columns if a[k]]
    hist = R.iloc[max(0, t-VOL_WIN):t][keys]
    # drop zero-variance books (e.g. just-activated) — corr undefined
    keys = [k for k in keys if hist[k].std() > 1e-10]
    hist = hist[keys]
    if len(hist) < 20 or len(keys) < 2:
        return w_ew(R, act, t)
    corr = hist.corr().values
    cov = hist.cov().values * TD
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, 1))
    order = leaves_list(linkage(dist[np.triu_indices(len(keys), 1)], method="single"))
    w = np.ones(len(keys))
    clusters = [list(order)]
    while clusters:                              # recursive bisection
        cl = clusters.pop(0)
        if len(cl) <= 1: continue
        half = len(cl)//2
        c1, c2 = cl[:half], cl[half:]
        def cvar(c):
            sub = cov[np.ix_(c, c)]
            iv = 1.0/np.clip(np.diag(sub), 1e-12, None)
            wv = iv/iv.sum()
            return float(wv @ sub @ wv)
        v1, v2 = cvar(c1), cvar(c2)
        alpha = 1 - v1/(v1+v2) if (v1+v2) > 0 else 0.5
        for i in c1: w[i] *= alpha
        for i in c2: w[i] *= (1-alpha)
        clusters += [c1, c2]
    out = pd.Series(0.0, index=R.columns)
    for pos, ki in enumerate(keys):
        out[ki] = w[pos]
    return out / out.sum()

def w_sharpe(R, act, t):
    a = act.iloc[t]
    hist = R.iloc[max(0, t-SHARPE_WIN):t]
    s = pd.Series(0.0, index=R.columns)
    for k in R.columns:
        if a[k]:
            sd = hist[k].std()
            s[k] = max(0.0, hist[k].mean()/sd*np.sqrt(TD)) if sd > 1e-12 else 0.0
    return s/s.sum() if s.sum() > 1e-9 else w_ew(R, act, t)

def run_alloc(R, act, weight_fn, vol_target=False):
    n = len(R)
    port = np.zeros(n)
    w = None
    for t in range(n):
        if w is None or t % REBAL == 0:
            w = weight_fn(R, act, t)
        port[t] = float((w * R.iloc[t]).sum())
    ser = pd.Series(port, index=R.index)
    if vol_target:                               # de-risk-only overlay, no lookahead
        rv = ser.rolling(20).std().shift(1) * np.sqrt(TD)
        scale = (TGT_VOL / rv).clip(upper=1.0).fillna(1.0)
        ser = ser * scale
    return ser

def stats(tag, ser, base=None):
    m = metrics_from_returns(ser.values, TD)
    flag = ""
    if base:
        better_sh = m["sharpe"] > base[0]
        better_ret = m["cagr"] > base[1]
        dd_ok = m["max_dd"] >= base[2] * 1.2     # not more than 1.2x worse
        if (better_sh or better_ret) and dd_ok: flag = "  <-- IMPROVED"
    print(f"  {tag:<14} Sharpe {m['sharpe']:.3f} | CAGR {m['cagr']:.1%} | MaxDD {m['max_dd']:.1%}{flag}")
    return m

METHODS = {
    "ew":          (w_ew,     False),
    "ivol":        (w_ivol,   False),
    "erc":         (w_erc,    False),
    "hrp":         (w_hrp,    False),
    "sharpe_w":    (w_sharpe, False),
    "ew_voltgt":   (w_ew,     True),
    "ivol_voltgt": (w_ivol,   True),
}

all_results = {}
for set_name, keys in [("ALL(A,B,C,D,F)", ["a","b","c","d","f"]),
                       ("NO_B(A,C,D,F)",  ["a","c","d","f"])]:
    R, act = build_panel(keys)
    print(f"\n=== {set_name}  {R.index[0].date()}..{R.index[-1].date()} ===")
    base_m = None
    for name, (fn, vt) in METHODS.items():
        ser = run_alloc(R, act, fn, vol_target=vt)
        base = (base_m["sharpe"], base_m["cagr"], base_m["max_dd"]) if base_m else None
        m = stats(name, ser, base)
        if name == "ew": base_m = m
        all_results[(set_name, name)] = (ser, m)

# record best performers (vs its own set's EW baseline)
print("\n=== Recording improved configs ===")
for (set_name, name), (ser, m) in all_results.items():
    base = all_results[(set_name, "ew")][1]
    if name == "ew": continue
    if ((m["sharpe"] > base["sharpe"] or m["cagr"] > base["cagr"])
            and m["max_dd"] >= base["max_dd"] * 1.2):
        tag = f"portfolio_{name}_{'all' if 'ALL' in set_name else 'nob'}"
        record_performance(name=tag, dates=ser.index, returns=ser.values,
            params={"method": name, "books": set_name, "rebalance_days": REBAL,
                    "vol_window": VOL_WIN, "target_vol": TGT_VOL if "voltgt" in name else None},
            data_period=f"{ser.index.min().date()}..{ser.index.max().date()}",
            periods_per_year=TD,
            extra={"vs_baseline": f"EW {base['sharpe']:.3f}/{base['max_dd']:.1%}",
                   "study": "2026-08-15 construction shootout"})
        print(f"  recorded {tag}: Sharpe {m['sharpe']:.3f} MaxDD {m['max_dd']:.1%}")
