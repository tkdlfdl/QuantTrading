"""
Regime overlays from the 60yr strategies survey, applied on top of the champion
allocator (ivol + 15% vol-target, no-B: Sharpe 2.501 / MaxDD -8.9%).

  PANIC gate (Daniel & Moskowitz 2016 JFE): momentum crashes cluster when the
    market is below water AND vol is high. State = SPY trailing 24-month return
    < 0 AND SPY 63d realized vol in its top quintile (trailing 3yr). In panic:
    cut momentum books (A, F) weights by half, redeploy freed weight to D.

  VIX-scale D (Nagel 2012 RFS): short-term reversal = liquidity provision; its
    conditional return rises with VIX. Scale D's weight by
    clip(VIX / 252d-median(VIX), 0.75, 1.75), then renormalize weights to 1.

Configs tested (all no-B, 21d rebalance, de-risk-only 15% vol target):
  champ            = ivol + voltgt                       (baseline champion)
  champ+panic      = + DM panic gate
  champ+vixD       = + Nagel VIX scaling of D
  champ+both       = + both overlays
All signals lagged 1 day (state at t uses data through t-1). No leverage.
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from tools.metrics import metrics_from_returns, sharpe_ratio, max_drawdown
from tools.record import record_performance

TD, REBAL, VOL_WIN, TGT_VOL = 252, 21, 60, 0.15
KEYS = ["a", "c", "d", "f"]
START = "2019-01-02"

def load(n):
    df = pd.read_csv(f"strategies/performance/{n}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)

B = {k: load(f"book_{k}_retest") for k in KEYS}
idx = sorted(set().union(*[s.index for s in B.values()]))
idx = pd.DatetimeIndex([d for d in idx if d >= pd.Timestamp(START)])
R = pd.DataFrame({k: B[k].reindex(idx).fillna(0.0) for k in KEYS})
n = len(R)

daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)
spy = daily["SPY"].dropna()
vix = daily["^VIX"].dropna()

# ── regime state series, all shifted 1 day (known before trading day) ──────
spy_ret24m = spy.pct_change(504)
spy_vol63 = spy.pct_change().rolling(63).std() * np.sqrt(TD)
vol_q80 = spy_vol63.rolling(756).quantile(0.80)
panic = ((spy_ret24m < 0) & (spy_vol63 > vol_q80)).shift(1).reindex(idx).fillna(False)

vix_med = vix.rolling(252).median()
vix_scale = (vix / vix_med).clip(0.75, 1.75).shift(1).reindex(idx).ffill().fillna(1.0)

print(f"Panic days in window: {int(panic.sum())}/{n} ({panic.mean():.1%})")

def run(panic_gate=False, vix_d=False):
    port = np.zeros(n); w = None
    for t in range(n):
        if w is None or t % REBAL == 0:
            hist = R.iloc[max(0, t-VOL_WIN):t]
            vol = hist.std() * np.sqrt(TD)
            iv = pd.Series(0.0, index=KEYS)
            for k in KEYS:
                if vol.get(k, 0) > 1e-9:
                    iv[k] = 1.0 / vol[k]
            w = iv / iv.sum() if iv.sum() else pd.Series(1/len(KEYS), index=KEYS)
        wt = w.copy()
        if panic_gate and bool(panic.iloc[t]):
            freed = 0.5 * (wt["a"] + wt["f"])
            wt["a"] *= 0.5; wt["f"] *= 0.5
            wt["d"] += freed                    # redeploy to D (reversal thrives in panics)
        if vix_d:
            wt["d"] *= float(vix_scale.iloc[t])
            wt = wt / wt.sum()                  # renormalize, no leverage
        port[t] = float((wt * R.iloc[t]).sum())
    ser = pd.Series(port, index=idx)
    rv = ser.rolling(20).std().shift(1) * np.sqrt(TD)   # vol-target overlay
    return ser * (TGT_VOL / rv).clip(upper=1.0).fillna(1.0)

CHAMP = (2.501, 0.452, -0.089)
configs = {
    "champ":       run(False, False),
    "champ+panic": run(True,  False),
    "champ+vixD":  run(False, True),
    "champ+both":  run(True,  True),
}
print(f"\n{'config':<14}{'Sharpe':>8}{'CAGR':>8}{'MaxDD':>8}")
for name, ser in configs.items():
    m = metrics_from_returns(ser.values, TD)
    imp = ""
    if name != "champ" and (m["sharpe"] > CHAMP[0] or m["cagr"] > CHAMP[1]) and m["max_dd"] >= CHAMP[2]*1.2:
        imp = "  <-- IMPROVED vs champion"
    print(f"{name:<14}{m['sharpe']:>8.3f}{m['cagr']:>8.1%}{m['max_dd']:>8.1%}{imp}")
    if imp:
        nm = f"portfolio_{name.replace('champ+','champ_')}_nob"
        record_performance(name=nm, dates=ser.index, returns=ser.values,
            params={"base": "ivol+voltgt nob", "overlay": name},
            data_period=f"{ser.index.min().date()}..{ser.index.max().date()}",
            periods_per_year=TD,
            extra={"papers": "Daniel-Moskowitz 2016 JFE / Nagel 2012 RFS",
                   "champion_baseline": "2.501/-8.9%"})
        print(f"   recorded -> strategies/performance/{nm}_*")

# yearly for the best config
best_name = max(configs, key=lambda k: metrics_from_returns(configs[k].values, TD)["sharpe"])
print(f"\nYearly [{best_name}]:")
ser = configs[best_name]
for y, x in ser.groupby(ser.index.year):
    print(f"  {y}: ret {(1+x).prod()-1:+8.2%}  sharpe {sharpe_ratio(x.values,TD):6.2f}  "
          f"mdd {max_drawdown(x.values):8.2%}")
