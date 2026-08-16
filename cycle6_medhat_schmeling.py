"""
Improvement Cycle 6 — queue idea #3: Medhat-Schmeling (2022 RFS)
turnover-conditioned short-term momentum. NEW-BOOK candidate.

Paper: 1-month momentum REVERSES among low-turnover stocks but CONTINUES among
high-turnover stocks; strongest in large caps (our universe).

Implementation (honest deviations noted):
- Turnover proxy: 21d volume sum / 252d volume sum (relative turnover
  intensity). We lack shares outstanding, so this is intensity vs own history,
  cross-sectionally ranked — a DEVIATION from the paper's share turnover;
  flagged for the registry.
- Monthly (21d) non-overlapping rebalance: within the TOP turnover-proxy
  quintile, long the top-10 stocks by trailing 21d return. Skip-day between
  formation and holding (signal from t-22..t-1, trade at t) — no look-ahead.
- Long-only (house style), equal weight, 0.1%/side costs, full history
  1998-2026 (earliest-data rule), daily attribution from daily closes.

Judged as a NEW BOOK: needs meaningful standalone Sharpe AND low correlation
to existing books to earn incubation (SELF_IMPROVEMENT_STRATEGY_PLAN Phase 2).
"""
from __future__ import annotations
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from data.db.client import get_conn
from tools.metrics import metrics_from_returns, sharpe_ratio, max_drawdown
from tools.record import record_performance, record_improvement

TD, REBAL, TOPN, TC = 252, 21, 10, 0.001
t0 = time.time()

con = get_conn()
px = con.execute("""
    SELECT ts, symbol, close, volume FROM ohlcv
    WHERE interval='1d' AND ts >= '1998-01-01' AND volume > 0
""").df()
px["ts"] = pd.to_datetime(px["ts"])
close = px.pivot_table(index="ts", columns="symbol", values="close").sort_index()
vol   = px.pivot_table(index="ts", columns="symbol", values="volume").sort_index()
print(f"panel {close.shape} loaded {time.time()-t0:.0f}s")

ret1d = close.pct_change()
mom21 = close.pct_change(21).shift(1)                      # formation t-22..t-1
tovr  = (vol.rolling(21).sum() / vol.rolling(252).sum()).shift(1)   # turnover proxy, lagged

dates = close.index
n = len(dates)
rows = []
i = 273                                                     # warmup 252+21
while i + REBAL < n:
    d = dates[i]
    m_ = mom21.loc[d]; t_ = tovr.loc[d]
    ok = m_.notna() & t_.notna() & close.loc[d].notna()
    if ok.sum() >= 50:
        t_ok = t_[ok]
        hi_t = t_ok[t_ok >= t_ok.quantile(0.8)].index        # top turnover quintile
        picks = m_[hi_t].nlargest(TOPN).index                # ST winners within it
        fwd = ret1d.iloc[i:i+REBAL][picks]
        pr = fwd.mean(axis=1).fillna(0.0)
        pr.iloc[0] -= 2 * TC
        for dt, x in pr.items(): rows.append((dt, float(x)))
    else:
        for j in range(i, i+REBAL): rows.append((dates[j], 0.0))
    i += REBAL

ser = pd.Series(dict(rows)).sort_index()
ser = ser[~ser.index.duplicated(keep="last")]
m = metrics_from_returns(ser.values, TD)
print(f"\nMS turnover-conditioned ST momentum (1998-2026): "
      f"Sharpe {m['sharpe']:.3f} | CAGR {m['cagr']:.1%} | MaxDD {m['max_dd']:.1%}")
for y, x in ser.groupby(ser.index.year):
    if y % 4 == 0 or y >= 2019:
        print(f"   {y}: ret {(1+x).prod()-1:+8.2%}  sharpe {sharpe_ratio(x.values,TD):6.2f}  "
              f"mdd {max_drawdown(x.values):8.2%}")

# variant: unconditioned ST momentum (isolate the turnover condition's value)
rows2 = []
i = 273
while i + REBAL < n:
    d = dates[i]
    m_ = mom21.loc[d]; ok = m_.notna() & close.loc[d].notna()
    if ok.sum() >= 50:
        picks = m_[ok].nlargest(TOPN).index
        fwd = ret1d.iloc[i:i+REBAL][picks]
        pr = fwd.mean(axis=1).fillna(0.0); pr.iloc[0] -= 2*TC
        for dt, x in pr.items(): rows2.append((dt, float(x)))
    else:
        for j in range(i, i+REBAL): rows2.append((dates[j], 0.0))
    i += REBAL
ser0 = pd.Series(dict(rows2)).sort_index(); ser0 = ser0[~ser0.index.duplicated(keep="last")]
m0 = metrics_from_returns(ser0.values, TD)
print(f"\nunconditioned ST momentum control:            "
      f"Sharpe {m0['sharpe']:.3f} | CAGR {m0['cagr']:.1%} | MaxDD {m0['max_dd']:.1%}")
print(f"turnover-condition effect: {m['sharpe']-m0['sharpe']:+.3f} Sharpe")

# correlation vs existing books (2019+ window)
def load(nm):
    df = pd.read_csv(f"strategies/performance/{nm}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)
corr = {}
s19 = ser[ser.index >= "2019-01-02"]
for b, nm in [("A","book_a_retest"),("C","book_c_overlap_cap"),("D","book_d_retest"),("F","book_f_retest")]:
    x = load(nm)
    j = s19.index.intersection(x.index)
    corr[b] = float(np.corrcoef(s19.loc[j], x.loc[j])[0,1])
print("corr vs books:", {k: round(v,2) for k,v in corr.items()})

if m["sharpe"] > 0.8 and m["sharpe"] > m0["sharpe"] + 0.1:
    record_performance(name="book_g_ms_stmom", dates=ser.index, returns=ser.values,
        params={"rebal": REBAL, "top_n": TOPN, "turnover_q": 0.8,
                "turnover_proxy": "vol21/vol252 (NOT share turnover — deviation)"},
        data_period=f"{ser.index.min().date()}..{ser.index.max().date()}", periods_per_year=TD,
        extra={"cycle": "Cycle 6 #3", "paper": "Medhat & Schmeling 2022 RFS",
               "corr_books": corr, "control_sharpe": m0["sharpe"]})
    record_improvement("NEW BOOK candidate: turnover-conditioned ST momentum",
        "Medhat & Schmeling 2022 RFS (registry #34)",
        dict(sharpe=m0["sharpe"], cagr=m0["cagr"], max_dd=m0["max_dd"]), m,
        ["book_g_ms_stmom"], f"vs unconditioned control; corr to books {corr}")
    print("CANDIDATE -> recorded book_g_ms_stmom (incubation path per Phase 2)")
else:
    print("no gain — condition adds nothing or absolute level too weak for a new book")
print(f"total {time.time()-t0:.0f}s")
