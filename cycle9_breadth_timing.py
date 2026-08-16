"""
Improvement Cycle 9 — queue #16: NEW BOOK candidate "Oversold-Breadth
Capitulation Index Timing" (origin class 4: original synthesis, no paper).

Thesis: Book D's bubble score works stock-by-stock (300/300 combos positive).
When MANY stocks are simultaneously deep-oversold (breadth spike), that is
market-level capitulation — buy the INDEX and hold days, not hours. Fires
exactly when momentum books bleed -> expected negative conditional corr to A/F.

Spec (no look-ahead): breadth_t = fraction of universe with official D bubble
score < -0.8 at the last bar of day t; signal = breadth (lagged 1 day) above
its rolling 2-year 95th percentile (percentile also lagged); entry next day
at QQQ open; hold H in {5,10,20} trading days, non-overlapping; 0.1%/side.
Universe scores from the OFFICIAL engine's _bubble_matrix (no re-implementation).
New-book gates: honest Sharpe > 0.8, corr to D < 0.45, corr to A/F low/negative.
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from strategies.contrarian_bubble_hourly import _bubble_matrix
from tools.metrics import metrics_from_returns, sharpe_ratio, max_drawdown
from tools.record import record_performance, record_improvement

TD, MA, THR, TC = 252, 104, -0.8, 0.001

hc = pd.read_parquet("data/cache/merged_hourly_close.parquet")
hc.index = pd.to_datetime(hc.index)
uni = hc.drop(columns=[c for c in ("SPY",) if c in hc.columns])
B = _bubble_matrix(uni, MA)                      # official score, [bars x tickers]
valid = uni.notna().values
oversold = (B < THR) & valid
frac = oversold.sum(1) / np.maximum(valid.sum(1), 1)
breadth_bar = pd.Series(frac, index=hc.index)
breadth = breadth_bar.groupby(breadth_bar.index.normalize()).last()   # daily (last bar)
breadth.index = pd.to_datetime(breadth.index)

q = pd.read_parquet("data/cache/qqq_hourly.parquet")  # open, close
q.index = pd.to_datetime(q.index)
qo_d = q["open"].groupby(q.index.normalize()).first()  # day open
qc_d = q["close"].groupby(q.index.normalize()).last()  # day close
qo_d.index = pd.to_datetime(qo_d.index); qc_d.index = pd.to_datetime(qc_d.index)

days = breadth.index.intersection(qo_d.index).sort_values()
b = breadth.reindex(days)
p95 = b.rolling(504, min_periods=252).quantile(0.95)
sig = (b > p95).shift(1).fillna(False)                 # signal known before day t
print(f"days {len(days)} ({days.min().date()}..{days.max().date()}), "
      f"signal days: {int(sig.sum())} ({sig.mean():.1%})")

results = {}
for H in [5, 10, 20]:
    rows = []
    i = 0
    day_list = list(days)
    while i < len(day_list) - H - 1:
        d = day_list[i]
        if bool(sig.loc[d]):
            entry = qo_d.loc[day_list[i]]              # next-day open = day t open (sig lagged)
            # daily attribution: entry open->close day i, then close->close, exit close day i+H-1
            for j in range(i, i + H):
                dj = day_list[j]
                if j == i:
                    r = qc_d.loc[dj]/qo_d.loc[dj] - 1 - TC
                else:
                    r = qc_d.loc[dj]/qc_d.loc[day_list[j-1]] - 1
                if j == i + H - 1:
                    r -= TC
                rows.append((dj, float(r)))
            i += H                                     # non-overlapping
        else:
            rows.append((day_list[i], 0.0))
            i += 1
    ser = pd.Series(dict(rows)).sort_index()
    m = metrics_from_returns(ser.values, TD)
    results[H] = (ser, m)
    in_mkt = (ser != 0).mean()
    print(f"H={H:>2}: Sharpe {m['sharpe']:.3f} | CAGR {m['cagr']:.1%} | "
          f"MaxDD {m['max_dd']:.1%} | in-market {in_mkt:.0%}")

bestH = max(results, key=lambda h: results[h][1]["sharpe"])
ser, m = results[bestH]

def load(n):
    df = pd.read_csv(f"strategies/performance/{n}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)
corr = {}
for bk, nm in [("A","book_a_retest"),("C","book_c_overlap_cap"),
               ("D","book_d_retest"),("F","book_f_retest")]:
    x = load(nm); j = ser.index.intersection(x.index)
    corr[bk] = float(np.corrcoef(ser.loc[j], x.loc[j])[0,1])
print(f"\nbest H={bestH}; corr vs books:", {k: round(v,2) for k,v in corr.items()})
for y, x in ser.groupby(ser.index.year):
    print(f"  {y}: ret {(1+x).prod()-1:+7.2%}  sharpe {sharpe_ratio(x.values,TD):5.2f}  "
          f"mdd {max_drawdown(x.values):7.2%}")

if m["sharpe"] > 0.8 and corr["D"] < 0.45 and max(corr["A"], corr["F"]) < 0.45:
    nm = "book_h_breadth_timing"
    record_performance(name=nm, dates=ser.index, returns=ser.values,
        params={"ma": MA, "thr": THR, "pctile": 0.95, "window": 504, "hold": bestH},
        data_period=f"{ser.index.min().date()}..{ser.index.max().date()}",
        periods_per_year=TD,
        extra={"cycle": "Cycle 9 #16", "origin": "ORIGINAL synthesis (Book D edge -> index timing)",
               "corr_books": corr})
    record_improvement("NEW BOOK candidate: oversold-breadth capitulation index timing",
        "ORIGINAL — no paper; synthesis of Book D bubble score at market level",
        dict(sharpe=0.0, cagr=0.0, max_dd=0.0), m, [nm],
        f"hold {bestH}d; corr {corr}; PENDING: incubation per Phase 2, human sign-off")
    print(f"\nCANDIDATE PASSES -> recorded {nm}; next step = zero-weight incubation (human-gated)")
else:
    print(f"\nfails new-book gates (Sharpe {m['sharpe']:.2f}, corr {corr})")
