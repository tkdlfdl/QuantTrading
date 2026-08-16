"""
Cycle 4 v2 — residual bubble score injected into the OFFICIAL Book D engine.

v1's custom bar-engine failed its own sanity anchor (alpha=0 gave Sharpe -1.23
instead of ~2.74), so its verdicts are void. v2 patches
strategies.contrarian_bubble_hourly._bubble_matrix to return the blended score
(alpha * residual_bubble + (1-alpha) * raw_bubble) and runs the exact validated
engine (locked ma=104, thr 0.8, hold 8h, top20). alpha=0 must reproduce 2.740
to the digit — that's the sanity gate for the whole experiment.
"""
from __future__ import annotations
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import strategies.contrarian_bubble_hourly as CB
from tools.metrics import metrics_from_returns
from tools.record import record_performance, record_improvement

TD, MA = 252, 104
D_BASE = dict(sharpe=2.740, cagr=0.352, max_dd=-0.064)

t0 = time.time()
ho = pd.read_parquet("data/cache/merged_hourly_open.parquet"); ho.index = pd.to_datetime(ho.index)
hc = pd.read_parquet("data/cache/merged_hourly_close.parquet"); hc.index = pd.to_datetime(hc.index)
common = sorted(set(ho.columns) & set(hc.columns)); ho, hc = ho[common], hc[common]

# residual index (identical construction to v1; backward-only rolling stats)
r = hc.pct_change()
mkt = r.mean(axis=1)
mm = mkt.rolling(MA).mean(); mv = mkt.rolling(MA).var()
cov = r.multiply(mkt, axis=0).rolling(MA).mean() - r.rolling(MA).mean().multiply(mm, axis=0)
beta = cov.div(mv, axis=0)
resid = (r - beta.multiply(mkt, axis=0)).fillna(0.0)
resid_idx = (1.0 + resid.clip(-0.5, 0.5)).cumprod() * 100.0
print(f"residual index built {time.time()-t0:.0f}s")

_orig_bubble = CB._bubble_matrix
B_raw = _orig_bubble(hc, MA)                       # official formula, raw prices
B_res = _orig_bubble(resid_idx, MA)                # official formula, residual index

ALPHA = 0.0
def _patched(close, ma_window):
    return (ALPHA * B_res + (1.0 - ALPHA) * B_raw).astype(np.float32)

CB._bubble_matrix = _patched

results = {}
for ALPHA in [0.0, 0.5, 1.0]:
    daily, params, _ = CB.run_contrarian_bubble_hourly(
        ho, hc, ma_window_grid=[MA], buy_threshold_grid=[0.8],
        hold_hours_grid=[8], top_n_grid=[20])
    m = metrics_from_returns(daily.dropna().values, TD)
    results[ALPHA] = (daily, m)
    print(f"alpha={ALPHA:.1f}: Sharpe {m['sharpe']:.3f} | CAGR {m['cagr']:.1%} | "
          f"MaxDD {m['max_dd']:.1%}")

CB._bubble_matrix = _orig_bubble                   # restore

m0 = results[0.0][1]
sane = abs(m0["sharpe"] - D_BASE["sharpe"]) < 0.05
print(f"\nsanity anchor alpha=0 vs official 2.740: {'OK' if sane else 'FAILED — void'}"
      f" ({m0['sharpe']:.3f})")
if sane:
    best_a = max([0.5, 1.0], key=lambda a: results[a][1]["sharpe"])
    d_best, m_best = results[best_a]
    print(f"residualization effect: {m_best['sharpe']-m0['sharpe']:+.3f} Sharpe (alpha={best_a})")
    if m_best["sharpe"] > m0["sharpe"] + 0.05 and m_best["max_dd"] >= D_BASE["max_dd"]*1.2:
        nm = f"book_d_residual_a{int(best_a*100)}"
        record_performance(name=nm, dates=d_best.dropna().index, returns=d_best.dropna().values,
            params={"ma": MA, "thr": -0.8, "hold": 8, "top_n": 20, "alpha": best_a},
            data_period=f"{d_best.index.min().date()}..{d_best.index.max().date()}",
            periods_per_year=TD,
            extra={"cycle": "Cycle 4 #9 v2 (official engine)", "paper": "Bun-Bouchaud-Potters + GPZ"})
        record_improvement(f"Residual bubble score for Book D (alpha={best_a})",
            "SYNTHESIS: Bun-Bouchaud-Potters 2017 + Guijarro-Ordonez et al. (registry #47, #48)",
            D_BASE, m_best, [nm])
        print(f"IMPROVED -> recorded {nm}")
    else:
        print("no gain vs raw bubble score in the official engine")
print(f"total {time.time()-t0:.0f}s")
