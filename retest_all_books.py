"""
RETEST ALL PRODUCTION BOOKS — locked params, latest data, daily-convention metrics.

Motivated by the Book E reliability audit (documented Sharpe 1.97 -> daily 0.63):
re-run every live book with its LOCKED live/config.py parameters on current data,
score everything with the project-standard daily Sharpe (mean*252 / std*sqrt252,
rf=0), record daily + historical performance via tools/record.py, and compare
against the documented numbers in BACKTEST_PERFORMANCE.md.

Attribution conventions per book (flagged in output):
  A : daily mark-to-market (run_book_a from run_hourly_momentum_proper)   PROPER
  B : replicated here bar-by-bar (entry open->close, then close->close)   PROPER
  C : existing engine — trade P&L attributed to ENTRY DATE (3d blocks)    BLOCK*
  D : existing engine — actual intraday daily attribution                 PROPER
  F : run_hourly_mom_combo (bar-by-bar, compounded within day)            PROPER
  * C's block convention is flagged; a full mark-to-market rewrite is follow-up.
"""
from __future__ import annotations
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from live.config import PARAMS
from strategies.contrarian_bubble_hourly import run_contrarian_bubble_hourly
from strategies.intraday_mean_reversion import run_intraday_mean_reversion
from strategies.qqq_bubble_hourly import calculate_bubble_score
from tools.metrics import sharpe_ratio, max_drawdown, metrics_from_returns
from tools.record import record_performance

TD = 252
TC = 0.001
SEP = "=" * 78

DOCUMENTED = {  # from BACKTEST_PERFORMANCE.md
    "A": dict(sharpe=1.41,  maxdd=-0.653),
    "B": dict(sharpe=1.106, maxdd=-0.0594),
    "C": dict(sharpe=0.927, maxdd=-0.2081),
    "D": dict(sharpe=2.71,  maxdd=-0.0641),
    "F": dict(sharpe=1.642, maxdd=-0.396),
}

t0 = time.time()
print("Loading data panels...")
ho = pd.read_parquet("data/cache/merged_hourly_open.parquet")
hc = pd.read_parquet("data/cache/merged_hourly_close.parquet")
ho.index = pd.to_datetime(ho.index); hc.index = pd.to_datetime(hc.index)
daily_all = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily_all.index = pd.to_datetime(daily_all.index)
common = sorted(set(ho.columns) & set(hc.columns))
ho, hc = ho[common], hc[common]
print(f"hourly: {len(ho)} bars x {len(common)} tickers "
      f"({ho.index.min()} .. {ho.index.max()}) | daily: {daily_all.shape} | {time.time()-t0:.0f}s")

RESULTS = {}

def report(book, ser, convention, params):
    ser = ser.dropna()
    m = metrics_from_returns(ser.values, TD)
    doc = DOCUMENTED.get(book, {})
    print(f"\n{SEP}\nBOOK {book}  [{convention}]  params={params}")
    print(f"  retest : Sharpe {m['sharpe']:.3f} | CAGR {m['cagr']:.1%} | "
          f"total {m['total_return']:.1%} | MaxDD {m['max_dd']:.1%} | days {m['n']}")
    if doc:
        print(f"  docs   : Sharpe {doc['sharpe']:.3f} | MaxDD {doc['maxdd']:.1%}"
              f"   -> dSharpe {m['sharpe']-doc['sharpe']:+.2f}")
    for y, x in ser.groupby(ser.index.year):
        print(f"    {y}: ret {(1+x).prod()-1:+8.2%}  sharpe {sharpe_ratio(x.values,TD):6.2f}  "
              f"mdd {max_drawdown(x.values):8.2%}")
    name = f"book_{book.lower()}_retest"
    record_performance(name=name, dates=ser.index, returns=ser.values,
        params=params, data_period=f"{ser.index.min().date()}..{ser.index.max().date()}",
        periods_per_year=TD,
        extra={"convention": convention, "retest": "2026-08-15 all-books retest"})
    RESULTS[book] = m
    print(f"  recorded -> strategies/performance/{name}_*")


# ═══════════════════════════════════════════════════════════════════════════
# BOOK D — contrarian bubble, locked ma=104 thr=0.8 hold=8 top20 (PROPER attribution)
# ═══════════════════════════════════════════════════════════════════════════
p = PARAMS["D"]
daily_d, params_d, _ = run_contrarian_bubble_hourly(
    ho, hc,
    ma_window_grid=[p["bubble_ma_hours"]],
    buy_threshold_grid=[abs(p["threshold"])],
    hold_hours_grid=[p["hold_hours"]],
    top_n_grid=[p["top_n"]],
)
report("D", daily_d, "PROPER intraday daily",
       {k: p[k] for k in ("bubble_ma_hours","threshold","hold_hours","top_n")})

# ═══════════════════════════════════════════════════════════════════════════
# BOOK F — hourly momentum lb=750 hold=200 top5 (PROPER bar-by-bar attribution)
# ═══════════════════════════════════════════════════════════════════════════
p = PARAMS["F"]
lb, hold, top_n = p["lookback_hours"], p["hold_hours"], p["top_n"]
hc_np = hc.values.astype(np.float32); ho_np = ho.values.astype(np.float32)
bar_ts = hc.index
n, n_tick = hc_np.shape
with np.errstate(divide="ignore", invalid="ignore"):
    mom_np = hc_np / np.vstack([np.full((lb, n_tick), np.nan, dtype=np.float32),
                                hc_np[:-lb]]) - 1

H = np.zeros((n, n_tick), dtype=np.float32)
entry_mask = np.zeros(n, dtype=bool); exit_mask = np.zeros(n, dtype=bool)
i = lb
while i + hold < n:
    row = mom_np[i]; valid = np.where(np.isfinite(row))[0]
    if len(valid) >= top_n:
        top_idx = valid[np.argsort(row[valid])[-top_n:]]
        H[i+1:i+hold+1, top_idx] = 1.0/top_n
        entry_mask[i+1] = True; exit_mask[i+hold] = True
    i += hold
with np.errstate(divide="ignore", invalid="ignore"):
    o2c = np.where((ho_np > 0) & np.isfinite(ho_np) & np.isfinite(hc_np), hc_np/ho_np - 1, 0.0)
    c_prev = np.vstack([hc_np[:1], hc_np[:-1]])
    c2c = np.where((c_prev > 0) & np.isfinite(c_prev) & np.isfinite(hc_np), hc_np/c_prev - 1, 0.0)
bar_ret = np.where(entry_mask[:, None], o2c, c2c)
port_h = (H * bar_ret).sum(axis=1)
port_h[entry_mask] -= TC; port_h[exit_mask] -= TC
port_h[H.sum(axis=1) == 0] = 0.0
s = pd.Series(port_h.astype(float), index=bar_ts)
daily_f = s.groupby(s.index.normalize()).apply(lambda g: float((1+g).prod()-1))
daily_f.index = pd.to_datetime(daily_f.index)
report("F", daily_f, "PROPER bar-by-bar daily", dict(lookback_hours=lb, hold_hours=hold, top_n=top_n))

# ═══════════════════════════════════════════════════════════════════════════
# BOOK B — QQQ bubble ma=200 z=100 buy<-0.8 hold=24 (replicated PROPER attribution)
# ═══════════════════════════════════════════════════════════════════════════
p = PARAMS["B"]
if "QQQ" in hc.columns:
    qo, qc = ho["QQQ"].dropna(), hc["QQQ"].dropna()
else:
    q = pd.read_parquet("data/cache/qqq_hourly.parquet")  # cols: open, close (11,775 bars)
    q.index = pd.to_datetime(q.index)
    qo, qc = q["open"].dropna(), q["close"].dropna()
idx = qo.index.intersection(qc.index)
qo, qc = qo.loc[idx], qc.loc[idx]
score = calculate_bubble_score(qc, p["qqq_bubble_ma_hours"], p["z_window_hours"]).shift(1)
nq = len(qc); hold = p["hold_hours"]
pos = np.zeros(nq); em = np.zeros(nq, bool); xm = np.zeros(nq, bool)
i = 0
while i < nq - hold - 1:
    if score.iloc[i] is not np.nan and pd.notna(score.iloc[i]) and score.iloc[i] < p["threshold"]:
        pos[i+1:i+hold+1] = 1.0; em[i+1] = True; xm[i+hold] = True
        i += hold
    else:
        i += 1
qo_np, qc_np = qo.values, qc.values
o2c = np.where(qo_np > 0, qc_np/qo_np - 1, 0.0)
c_prev = np.concatenate([[qc_np[0]], qc_np[:-1]])
c2c = np.where(c_prev > 0, qc_np/c_prev - 1, 0.0)
br = np.where(em, o2c, c2c) * pos
br[em] -= TC; br[xm] -= TC
sb = pd.Series(br, index=idx)
daily_b = sb.groupby(sb.index.normalize()).apply(lambda g: float((1+g).prod()-1))
daily_b.index = pd.to_datetime(daily_b.index)
try:
    report("B", daily_b, "PROPER bar-by-bar daily (replicated)",
           {k: p[k] for k in ("qqq_bubble_ma_hours","z_window_hours","threshold","hold_hours")})
except Exception as e:
    print(f"BOOK B FAILED: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# BOOK C — intraday MR sigma=4 lb=20 flip=3 top5 (existing engine, ENTRY-DATE blocks)
# ═══════════════════════════════════════════════════════════════════════════
p = PARAMS["C"]
dc = daily_all[[c for c in common if c in daily_all.columns]]
ret_c, params_c, _ = run_intraday_mean_reversion(
    dc, ho, hc,
    sigma_grid=[p["sigma"]],
    flip_hold_days_grid=[p["flip_hold_days"]],
    lookback_grid=[p["z_lookback_days"]],
    top_n_grid=[p["top_n"]],
)
try:
    report("C", ret_c, "BLOCK entry-date (flagged — needs MTM rewrite)",
           {k: p[k] for k in ("sigma","z_lookback_days","flip_hold_days","top_n")})
except Exception as e:
    print(f"BOOK C FAILED: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# BOOK A — daily momentum + leverage + UVXY hedge (PROPER daily attribution)
# ═══════════════════════════════════════════════════════════════════════════
A_PARAMS = PARAMS["A"]
def _bubble(equity, ma_days, z_days):
    lp = np.log(np.maximum(equity, 1e-9))
    fair = lp.rolling(ma_days).mean()
    r = lp - fair
    z = (r - r.rolling(z_days).mean()) / r.rolling(z_days).std()
    return np.tanh(z / 2)

def run_book_a(daily_close):
    pA = A_PARAMS
    close = daily_close.copy()
    lookback, holding, top = pA["lookback_days"], pA["rebalance_days"], pA["top_n"]
    ret_d = close.pct_change().ffill().fillna(0)
    ret_mom = close.pct_change(lookback).ffill().fillna(0)
    rows = []
    for i in range(lookback + 1, len(ret_mom), holding):
        ranked = np.argsort(ret_mom.iloc[i-1:i].rank(axis=1, ascending=False).values[0])
        for j in range(i, min(i + holding, len(ret_mom))):
            date = ret_d.index[j]
            lret = ret_d.iloc[j, ranked[:top]].mean()
            mom_r = lret - pA["tc_per_cycle"] / holding
            h_ret = 0.0
            if "UVXY" in close.columns and pd.notna(close.loc[date, "UVXY"]):
                h_ret = ret_d.loc[date, "UVXY"]
            elif "^VIX" in close.columns and pd.notna(close.loc[date, "^VIX"]):
                v = ret_d.loc[date, "^VIX"]
                h_ret = (2.0*v - 0.002 - 0.25*v**2 if date < pd.Timestamp("2018-02-28")
                         else 1.5*v - 0.0015 - 0.25*v**2)
            rows.append({"Date": date, "Momentum": mom_r, "Hedge": h_ret})
    dfA = pd.DataFrame(rows).set_index("Date").dropna()
    base_w = (1 + dfA).cumprod() / (1 + dfA).cumprod().iloc[0]
    bub = _bubble(base_w["Momentum"], pA["bubble_ma_days"], pA["bubble_z_days"])
    h_sig = (bub > pA["hedge_threshold"]).shift(1).fillna(False)
    l_sig = (bub < pA["lev_threshold"]).shift(1).fillna(False)
    lev_c = pA["lev_cost_ann"] / TD
    out = []; h_rem = l_rem = 0
    for date in dfA.index:
        if h_rem == 0 and h_sig.loc[date]: h_rem = pA["hedge_hold_days"]
        if l_rem == 0 and l_sig.loc[date]: l_rem = pA["lev_hold_days"]
        base = dfA.loc[date, "Momentum"]
        if h_rem > 0:
            r = (1 - pA["hedge_alloc"])*base + pA["hedge_alloc"]*dfA.loc[date, "Hedge"]
            h_rem -= 1
        elif l_rem > 0:
            r = base + pA["lev_mult"]*base - pA["lev_mult"]*lev_c
            l_rem -= 1
        else:
            r = base
        out.append(r)
    return pd.Series(out, index=dfA.index)

ret_a = run_book_a(daily_all)
report("A", ret_a, "PROPER daily MTM",
       {k: A_PARAMS[k] for k in ("lookback_days","rebalance_days","top_n",
                                 "lev_threshold","hedge_threshold")})

# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}\nSUMMARY — retested (daily convention) vs documented")
print(f"{'Book':<5}{'Retest Sharpe':>14}{'Doc Sharpe':>12}{'Delta':>8}{'Retest MaxDD':>14}{'Doc MaxDD':>11}")
for b in ["A","B","C","D","F"]:
    if b in RESULTS:
        m, d = RESULTS[b], DOCUMENTED[b]
        print(f"{b:<5}{m['sharpe']:>14.3f}{d['sharpe']:>12.3f}{m['sharpe']-d['sharpe']:>+8.2f}"
              f"{m['max_dd']:>14.1%}{d['maxdd']:>11.1%}")
print(f"\nTotal runtime: {time.time()-t0:.0f}s")
