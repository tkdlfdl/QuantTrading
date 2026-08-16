"""Sharpe-delta significance test (Ledoit & Wolf 2008, J. Empirical Finance).

Promotion decisions on small Sharpe deltas between highly-correlated return
series are statistically fragile (SE of the delta ~0.07 at our sample size and
~0.98 correlation). This tool tests H0: Sharpe(a) == Sharpe(b) with a
studentized stationary-bootstrap (Politis-Romano) confidence interval on the
difference. Adopted as process 2026-08-16 (econometrics sweep round 2, idea #1):
**an overlay/portfolio variant is only promotable if p < 0.10 one-sided.**

CLI:
  python -m tools.significance portfolio_champ_d14 portfolio_champion_cappedC
Library:
  from tools.significance import sharpe_delta_test
  res = sharpe_delta_test(rets_a, rets_b)   # dict: delta, se, p_one_sided, ci
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

TD = 252


def _sharpe(r: np.ndarray) -> float:
    s = r.std(ddof=1)
    return float(r.mean() / s * np.sqrt(TD)) if s > 0 else np.nan


def _stationary_bootstrap_idx(n: int, avg_block: float, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap index sequence of length n."""
    idx = np.empty(n, dtype=int)
    p = 1.0 / avg_block
    t = rng.integers(0, n)
    for i in range(n):
        idx[i] = t
        if rng.random() < p:
            t = rng.integers(0, n)
        else:
            t = (t + 1) % n
    return idx


def sharpe_delta_test(rets_a, rets_b, n_boot: int = 2000, avg_block: float = 10.0,
                      seed: int = 42) -> dict:
    """Test Sharpe(a) - Sharpe(b) on ALIGNED daily returns (paired days only)."""
    a = pd.Series(rets_a).astype(float)
    b = pd.Series(rets_b).astype(float)
    if isinstance(rets_a, pd.Series) and isinstance(rets_b, pd.Series):
        j = rets_a.index.intersection(rets_b.index)
        a, b = rets_a.loc[j].astype(float), rets_b.loc[j].astype(float)
    av, bv = a.values, b.values
    n = len(av)
    delta = _sharpe(av) - _sharpe(bv)
    corr = float(np.corrcoef(av, bv)[0, 1])

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for k in range(n_boot):
        idx = _stationary_bootstrap_idx(n, avg_block, rng)
        boot[k] = _sharpe(av[idx]) - _sharpe(bv[idx])
    se = float(boot.std(ddof=1))
    # one-sided p for H1: delta > 0, via centered bootstrap distribution
    p_one = float(np.mean((boot - boot.mean()) >= delta)) if delta > 0 else \
            float(np.mean((boot - boot.mean()) <= delta))
    ci = (float(np.percentile(boot, 5)), float(np.percentile(boot, 95)))
    return dict(n=n, corr=corr, delta=float(delta), se=se,
                p_one_sided=p_one, ci90=ci,
                significant_p10=bool(p_one < 0.10 and delta > 0))


def main(argv=None) -> int:
    args = argv or sys.argv[1:]
    if len(args) != 2:
        print("usage: python -m tools.significance <name_a> <name_b>   "
              "(recorded strategies; tests Sharpe(a) - Sharpe(b))")
        return 2
    def load(n):
        df = pd.read_csv(f"strategies/performance/{n}_daily.csv", parse_dates=["date"])
        return df.set_index("date")["ret"].astype(float)
    a, b = load(args[0]), load(args[1])
    r = sharpe_delta_test(a, b)
    print(f"{args[0]}  vs  {args[1]}")
    print(f"  n={r['n']} paired days | corr {r['corr']:.3f}")
    print(f"  Sharpe delta {r['delta']:+.3f} | bootstrap SE {r['se']:.3f} "
          f"| 90% CI [{r['ci90'][0]:+.3f}, {r['ci90'][1]:+.3f}]")
    print(f"  one-sided p = {r['p_one_sided']:.3f} -> "
          f"{'SIGNIFICANT (p<0.10) — promotable' if r['significant_p10'] else 'NOT significant — do not promote on this evidence'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
