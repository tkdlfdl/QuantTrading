"""
Improvement Cycle 4 — queue idea #9: residual bubble score for Book D.

Synthesis (Bun-Bouchaud-Potters 2017 RMT + Guijarro-Ordonez et al.
residualization): the top eigenvector of the hourly correlation matrix is the
market mode; a stock can look "oversold" merely because the market fell. Strip
it: residual return r_i - beta_i * r_mkt (rolling 104h beta vs the EW market
return = RMT-lite market mode), rebuild a residual price index, compute D's
exact bubble formula on that index, and select on a blend:

    score = alpha * residual_score + (1-alpha) * raw_score,  alpha in {0.5, 1}

Selection identical to Book D locked params: score.shift(1) < -0.8, top-20 most
negative, non-overlapping 8h holds. P&L bar-by-bar on REAL prices (entry next
bar open->close, then close->close), compounded to daily; 0.1%/side costs.
Baseline: Book D 2.740 / CAGR 35.2% / MaxDD -6.4%.
"""
from __future__ import annotations
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from tools.metrics import metrics_from_returns
from tools.record import record_performance, record_improvement

TD = 252
MA, THR, HOLD, TOPN = 104, -0.8, 8, 20
TC = 0.001
D_BASE = dict(sharpe=2.740, cagr=0.352, max_dd=-0.064)

t0 = time.time()
ho = pd.read_parquet("data/cache/merged_hourly_open.parquet"); ho.index = pd.to_datetime(ho.index)
hc = pd.read_parquet("data/cache/merged_hourly_close.parquet"); hc.index = pd.to_datetime(hc.index)
common = sorted(set(ho.columns) & set(hc.columns)); ho, hc = ho[common], hc[common]
print(f"panel {hc.shape} loaded {time.time()-t0:.0f}s")

# ── residual returns vs EW market mode (rolling beta, no lookahead) ────────
r = hc.pct_change()
mkt = r.mean(axis=1)                                   # EW market mode (RMT-lite)
mm = mkt.rolling(MA).mean()
mv = mkt.rolling(MA).var()
# rolling beta_i = cov(r_i, mkt)/var(mkt), all rolling windows (backward-only)
cov = r.multiply(mkt, axis=0).rolling(MA).mean() - r.rolling(MA).mean().multiply(mm, axis=0)
beta = cov.div(mv, axis=0)
resid = (r - beta.multiply(mkt, axis=0)).fillna(0.0)
resid_idx = (1.0 + resid.clip(-0.5, 0.5)).cumprod() * 100.0
print(f"residual panel built {time.time()-t0:.0f}s")

def bubble(close: pd.DataFrame) -> np.ndarray:
    lp = np.log(close.replace(0, np.nan).ffill())
    fair = close.rolling(MA, min_periods=MA//2).mean()
    res = lp - np.log(fair.replace(0, np.nan))
    z = (res - res.rolling(MA, min_periods=MA//2).mean()) / res.rolling(MA, min_periods=MA//2).std()
    return np.tanh(z/2).fillna(0.0).values.astype(np.float32)

B_raw = bubble(hc)
B_res = bubble(resid_idx)
print(f"bubble scores computed {time.time()-t0:.0f}s")

ho_np = ho.values.astype(np.float32); hc_np = hc.values.astype(np.float32)
n, U = hc_np.shape
bar_ts = hc.index
with np.errstate(divide="ignore", invalid="ignore"):
    o2c = np.where((ho_np > 0) & np.isfinite(ho_np) & np.isfinite(hc_np), hc_np/ho_np - 1, 0.0)
    c_prev = np.vstack([hc_np[:1], hc_np[:-1]])
    c2c = np.where((c_prev > 0) & np.isfinite(c_prev) & np.isfinite(hc_np), hc_np/c_prev - 1, 0.0)

def run_alpha(alpha: float):
    S = alpha * B_res + (1 - alpha) * B_raw          # blended score
    H = np.zeros((n, U), dtype=np.float32)
    em = np.zeros(n, bool); xm = np.zeros(n, bool)
    i = MA
    while i + HOLD < n - 1:
        s = S[i - 1]                                  # shift(1): score from prior bar
        cand = np.where(s < THR)[0]
        if len(cand):
            picks = cand[np.argsort(s[cand])[:TOPN]]
            H[i+1:i+HOLD+1, picks] = 1.0/len(picks)
            em[i+1] = True; xm[i+HOLD] = True
        i += HOLD
    bar_ret = np.where(em[:, None], o2c, c2c)
    port = (H * bar_ret).sum(axis=1)
    port[em] -= TC; port[xm] -= TC
    port[H.sum(axis=1) == 0] = 0.0
    s = pd.Series(port.astype(float), index=bar_ts)
    daily = s.groupby(s.index.normalize()).apply(lambda g: float((1+g).prod()-1))
    daily.index = pd.to_datetime(daily.index)
    return daily

print("\nalpha  Sharpe   CAGR    MaxDD")
results = {}
for alpha in [0.0, 0.5, 1.0]:
    d = run_alpha(alpha)
    m = metrics_from_returns(d.values, TD)
    results[alpha] = (d, m)
    print(f"{alpha:4.1f}  {m['sharpe']:6.3f}  {m['cagr']:6.1%}  {m['max_dd']:7.1%}")

# judge: alpha=0 is the bar-engine baseline (sanity vs official 2.740); compare
# alpha>0 against the SAME engine's alpha=0 to isolate the residualization effect,
# and against official D_BASE for the promotion decision.
m0 = results[0.0][1]
best_alpha = max([0.5, 1.0], key=lambda a: results[a][1]["sharpe"])
d_best, m_best = results[best_alpha]
gain = m_best["sharpe"] - m0["sharpe"]
print(f"\nresidualization effect (same engine): {gain:+.3f} Sharpe (alpha={best_alpha})")
if (m_best["sharpe"] > max(m0["sharpe"], D_BASE["sharpe"]) + 0.05
        and m_best["max_dd"] >= D_BASE["max_dd"] * 1.2):
    nm = f"book_d_residual_a{int(best_alpha*100)}"
    record_performance(name=nm, dates=d_best.index, returns=d_best.values,
        params={"ma": MA, "thr": THR, "hold": HOLD, "top_n": TOPN, "alpha": best_alpha},
        data_period=f"{d_best.index.min().date()}..{d_best.index.max().date()}",
        periods_per_year=TD,
        extra={"cycle": "Cycle 4 #9", "paper": "Bun-Bouchaud-Potters 2017 + Guijarro-Ordonez et al."})
    record_improvement(f"Residual bubble score for Book D (alpha={best_alpha})",
        "SYNTHESIS: Bun-Bouchaud-Potters 2017 Phys.Rep. + Guijarro-Ordonez-Pelger-Zanotti (registry #47, #48)",
        D_BASE, m_best, [nm])
    print(f"IMPROVED -> recorded {nm}")
else:
    print("no gain vs Book D locked — residualization does not beat raw bubble score")
print(f"total {time.time()-t0:.0f}s")
