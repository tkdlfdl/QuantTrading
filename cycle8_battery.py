"""
Improvement Cycle 8 battery — queue #12 (dispersion throttle, pre-test first)
and #13 (correlation-spike fragility gate).

#12 Stivers-Sun 2010 JFQA: high cross-sectional return dispersion predicts weak
    momentum profits. STAGE 1 pre-test on Book A over 1997-2018 (in-sample kill
    zone — cheap to falsify before touching the 2019+ OOS window): compare Book
    A mean return in high-dispersion (z>1) vs normal months. Proceed to Stage 2
    (throttle in champion) ONLY if the pre-test shows the effect.

#13 Preis 2012 / Kritzman 2011: mean pairwise correlation of the stock universe
    spikes ahead of fragile periods. Gate: when 60d mean pairwise correlation's
    z-score (vs trailing 3yr) > 1, throttle A/F by half and tighten vol target
    to 12%. Never touches D (registry #47 lesson). All signals lagged 1 day.

Baseline champion: 2.580 / 43.5% / -8.9% (anchor rebuilt internally).
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
CHAMP = dict(sharpe=2.580, cagr=0.435, max_dd=-0.089)

daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)
px = daily.drop(columns=[c for c in ("UVXY", "^VIX") if c in daily.columns])
rets = px.pct_change()

def load(n):
    df = pd.read_csv(f"strategies/performance/{n}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)

# ═══════ #12 Stage-1 pre-test: dispersion vs Book A returns, 1997-2018 ═══════
print("=== #12 dispersion pre-test (Book A, 1997-2018 in-sample) ===")
disp = rets.std(axis=1)                                  # cross-sectional dispersion
disp_z = ((disp - disp.rolling(252).mean()) / disp.rolling(252).std()).shift(1)
a = load("book_a_retest")
pre = a[(a.index >= "1998-01-01") & (a.index < "2019-01-01")]
z = disp_z.reindex(pre.index)
hi = pre[z > 1.0]; lo = pre[z <= 1.0]
t_num = hi.mean() - lo.mean()
t_den = np.sqrt(hi.var()/len(hi) + lo.var()/len(lo))
print(f"  high-disp days: {len(hi)} mean {hi.mean()*TD:+.1%}/yr | "
      f"normal days: {len(lo)} mean {lo.mean()*TD:+.1%}/yr | t={t_num/t_den:.2f}")
effect = (hi.mean() < lo.mean()) and abs(t_num/t_den) > 1.65
print(f"  pre-test verdict: {'EFFECT PRESENT -> stage 2' if effect else 'NO EFFECT -> kill (registry #78)'}")

# ═══════ champion rebuild helper (anchor) ═══════
BOOKS = {"A": load("book_a_retest"), "C": load("book_c_overlap_cap"),
         "D": load("book_d_retest"), "F": load("book_f_retest")}
idx = sorted(set().union(*[s.index for s in BOOKS.values()]))
idx = pd.DatetimeIndex([d for d in idx if d >= pd.Timestamp("2019-01-02")])
R = pd.DataFrame({k: BOOKS[k].reindex(idx).fillna(0.0) for k in BOOKS})
spy = daily["SPY"].dropna()
panic = ((spy.pct_change(504) < 0) &
         (spy.pct_change().rolling(63).std()*np.sqrt(TD) >
          (spy.pct_change().rolling(63).std()*np.sqrt(TD)).rolling(756).quantile(0.8))
        ).shift(1).reindex(idx).fillna(False)
cols = list(R.columns); iA, iF, iD = cols.index("A"), cols.index("F"), cols.index("D")

def run_champ(extra_throttle=None, tgt_series=None):
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
        if extra_throttle is not None and extra_throttle[t]:
            wt[iA] *= 0.5; wt[iF] *= 0.5      # freed weight -> cash (de-risk)
        port[t] = float(wt @ R.iloc[t].values)
    ser = pd.Series(port, index=idx)
    rv = ser.rolling(20).std().shift(1) * np.sqrt(TD)
    tgt = pd.Series(0.15, index=idx) if tgt_series is None else tgt_series
    return ser * (tgt/rv).clip(upper=1.0).fillna(1.0)

anchor = run_champ()
m0 = metrics_from_returns(anchor.values, TD)
print(f"\nanchor: Sharpe {m0['sharpe']:.3f} | MaxDD {m0['max_dd']:.1%} "
      f"{'OK' if abs(m0['sharpe']-CHAMP['sharpe'])<0.05 else 'FAILED - VOID'}")
if abs(m0["sharpe"] - CHAMP["sharpe"]) >= 0.05: sys.exit(1)

# ═══════ #13 correlation-spike fragility gate ═══════
print("\n=== #13 correlation-spike gate ===")
sub = rets.loc[:, rets.notna().mean() > 0.9]             # well-covered names
corr_mean = (sub.rolling(60).corr().groupby(level=0).mean().mean(axis=1)
             if False else None)
# efficient mean pairwise correlation: var of EW portfolio vs avg var
ew_var = sub.mean(axis=1).rolling(60).var()
avg_var = sub.rolling(60).var().mean(axis=1)
N = sub.notna().sum(axis=1).clip(lower=2)
rho = ((N * ew_var / avg_var) - 1) / (N - 1)             # implied mean pairwise corr
rho_z = ((rho - rho.rolling(756).mean()) / rho.rolling(756).std()).shift(1)
gate13 = (rho_z.reindex(idx) > 1.0).fillna(False).values
tgt13 = pd.Series(np.where(gate13, 0.12, 0.15), index=idx)
print(f"  gate active: {gate13.sum()}/{len(idx)} days ({gate13.mean():.1%})")
ser13 = run_champ(extra_throttle=gate13, tgt_series=tgt13)
m13 = metrics_from_returns(ser13.values, TD)
print(f"  champ + corr gate: Sharpe {m13['sharpe']:.3f} | CAGR {m13['cagr']:.1%} | MaxDD {m13['max_dd']:.1%}")
if (m13["sharpe"] > m0["sharpe"] + 0.05 or m13["cagr"] > m0["cagr"] + 0.05) \
        and m13["max_dd"] >= CHAMP["max_dd"]*1.2:
    record_performance(name="portfolio_champ_corrgate", dates=ser13.index, returns=ser13.values,
        params={"gate": "mean pairwise corr z>1 (60d vs 3yr)", "throttle": "A/F x0.5, tgt 12%"},
        data_period=f"{idx.min().date()}..{idx.max().date()}", periods_per_year=TD,
        extra={"cycle": "Cycle 8 #13", "paper": "Preis 2012 Sci.Rep.; Kritzman 2011 JPM"})
    record_improvement("Correlation-spike fragility gate on champion",
        "Preis et al. 2012; Kritzman et al. 2011 (registry #68, #71)", CHAMP, m13,
        ["portfolio_champ_corrgate"])
    print("  IMPROVED -> recorded")
else:
    print("  no gain")

# ═══════ #12 Stage 2 (only if pre-test passed) ═══════
if effect:
    print("\n=== #12 stage 2: dispersion throttle in champion ===")
    gate12 = (disp_z.reindex(idx) > 1.0).fillna(False).values
    ser12 = run_champ(extra_throttle=gate12)
    m12 = metrics_from_returns(ser12.values, TD)
    print(f"  champ + disp throttle: Sharpe {m12['sharpe']:.3f} | CAGR {m12['cagr']:.1%} | MaxDD {m12['max_dd']:.1%}")
    if (m12["sharpe"] > m0["sharpe"] + 0.05) and m12["max_dd"] >= CHAMP["max_dd"]*1.2:
        record_performance(name="portfolio_champ_dispgate", dates=ser12.index, returns=ser12.values,
            params={"gate": "CS dispersion z>1"}, data_period=f"{idx.min().date()}..{idx.max().date()}",
            periods_per_year=TD, extra={"cycle": "Cycle 8 #12", "paper": "Stivers-Sun 2010 JFQA"})
        record_improvement("Dispersion throttle on A/F", "Stivers & Sun 2010 JFQA (registry #78)",
            CHAMP, m12, ["portfolio_champ_dispgate"])
        print("  IMPROVED -> recorded")
    else:
        print("  no gain")
