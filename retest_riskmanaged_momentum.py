"""
Risk-managed momentum overlay on Book F (Barroso & Santa-Clara 2015 JFE style).

WML momentum scaled to constant volatility using its own trailing realized vol
kills momentum crashes. Applied to Book F's honest retest series (Sharpe 1.548,
MaxDD -39.8%): scale_t = min(1, sigma_target / sigma_hat_t), where sigma_hat is
annualized vol from the past 126 days (6 months, per the paper), de-risk only
(no leverage per project rules). Grid over sigma_target. Same overlay on Book A
(daily momentum, MaxDD -65%) as a second test.
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from tools.metrics import sharpe_ratio, max_drawdown, metrics_from_returns
from tools.record import record_performance

TD = 252
VOL_WIN = 126   # Barroso-Santa-Clara: 6 months of daily returns

def load(name):
    df = pd.read_csv(f"strategies/performance/{name}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)

BASE = {
    "f": (1.548, -0.398),
    "a": (1.301, -0.654),
}

for book in ["f", "a"]:
    ser = load(f"book_{book}_retest")
    if book == "a":
        ser = ser[ser.index >= "2019-01-02"]   # align comparison window with F
    b_sh, b_dd = BASE[book]
    m0 = metrics_from_returns(ser.values, TD)
    print(f"\n=== BOOK {book.upper()} base: Sharpe {m0['sharpe']:.3f} | CAGR {m0['cagr']:.1%} | MaxDD {m0['max_dd']:.1%} ===")
    best = None
    for tgt in [0.10, 0.12, 0.15, 0.20, 0.25]:
        rv = ser.rolling(VOL_WIN).std().shift(1) * np.sqrt(TD)
        scale = (tgt / rv).clip(upper=1.0).fillna(1.0)
        managed = ser * scale
        m = metrics_from_returns(managed.values, TD)
        tag = ""
        if m["sharpe"] > m0["sharpe"] and m["max_dd"] >= m0["max_dd"] * 1.2:
            tag = "  <-- IMPROVED"
        print(f"  tgt {tgt:.0%}: Sharpe {m['sharpe']:.3f} | CAGR {m['cagr']:.1%} | "
              f"MaxDD {m['max_dd']:.1%} | avg exposure {scale.mean():.0%}{tag}")
        if tag and (best is None or m["sharpe"] > best[1]["sharpe"]):
            best = (tgt, m, managed)
    if best:
        tgt, m, managed = best
        nm = f"book_{book}_riskmanaged"
        record_performance(name=nm, dates=managed.index, returns=managed.values,
            params={"overlay": "vol-scale de-risk only", "target_vol": tgt, "vol_window": VOL_WIN},
            data_period=f"{managed.index.min().date()}..{managed.index.max().date()}",
            periods_per_year=TD,
            extra={"base": f"book_{book}_retest Sharpe {m0['sharpe']:.3f} MaxDD {m0['max_dd']:.1%}",
                   "paper": "Barroso & Santa-Clara 2015 JFE (risk-managed momentum)"})
        print(f"  recorded -> strategies/performance/{nm}_* (tgt {tgt:.0%})")
