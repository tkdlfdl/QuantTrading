"""
Cycle 14 — orthogonal program, queue #28-32 in test order.

Mission metric per orthogonal_ideation brief: worst-5%-day P&L + champion delta
(LW p<0.10) — NOT standalone Sharpe. Champion = blended-D config (2.788/-8.3%).

 #28 UVXY carry-regime study (diagnostic)
 #29 Complacency-timed UVXY convexity (breadth inversion + VRP low/falling +
     realized-vol ignition; 30 in-market days/yr budget) — 2019+ (breadth data)
 #30 Book F residual-momentum re-rank (rolling beta vs EW market, anchor 1.548)
 #31 Crisis-only short/flat SPY sleeve (TSMOM<0 AND 21d<0; 8%/yr borrow)
     — paper test only; live shorts need human sign-off
 #32 Utilities/SPY rotation (XLU fetched ad hoc if absent from panel)
"""
from __future__ import annotations
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from tools.metrics import metrics_from_returns
from tools.significance import sharpe_delta_test
from tools.record import record_performance, record_improvement
from live import engine as E, config as C

TD, TC, END = 252, 0.001, "2026-07-08"
t0 = time.time()

def load(n):
    df = pd.read_csv(f"strategies/performance/{n}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)

daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)
spy = daily["SPY"].dropna(); spy_r = spy.pct_change()
vix = daily["^VIX"].dropna()
uvxy = daily["UVXY"].dropna(); uvxy_r = uvxy.pct_change()

champ = load("portfolio_champ_d14")
worst = champ.nsmallest(int(len(champ)*0.05)).index

d8, d14 = load("book_d_retest"), load("book_d14")
ju = d8.index.union(d14.index)
dblend = 0.5*d8.reindex(ju).fillna(0) + 0.5*d14.reindex(ju).fillna(0)
BASE = {"A": load("book_a_retest"), "C": load("book_c_overlap_cap"),
        "D": dblend, "F": load("book_f_retest")}

def contrib(name, stream, shares=1.0):
    """Champion + candidate stream: metrics, delta, stress-day P&L."""
    saved_b, saved_s = C.ALLOC_BOOKS, dict(getattr(C, "ALLOC_SHARES", {}))
    C.ALLOC_BOOKS = ["A","C","D","F","X"]; C.ALLOC_SHARES = {"D": 2.0, "X": shares}
    try:
        books = dict(BASE); books["X"] = stream
        R = pd.DataFrame(books)
        R = R[(R.index >= "2019-01-02") & (R.index <= END)].fillna(0.0)
        ser = E.ivol_voltgt(R).dropna()
    finally:
        C.ALLOC_BOOKS, C.ALLOC_SHARES = saved_b, saved_s
    m = metrics_from_returns(ser.values, TD)
    r = sharpe_delta_test(ser, champ)
    sd = float(stream.reindex(worst).fillna(0).mean())
    print(f"  {name}: champ+X Sharpe {m['sharpe']:.3f} (delta {r['delta']:+.3f}, p {r['p_one_sided']:.3f}) "
          f"| MaxDD {m['max_dd']:.1%} | X stress-day P&L {sd:+.2%}/d")
    return m, r, sd

# ══════════ #28 UVXY carry-regime study ══════════
print("=== #28 UVXY carry regimes (2011+) ===")
rv21 = spy_r.rolling(21).std()*np.sqrt(TD)
regime = pd.DataFrame({"u": uvxy_r,
                       "vix_lvl": vix.reindex(uvxy_r.index),
                       "vix_vs_ma": (vix/vix.rolling(63).mean()).reindex(uvxy_r.index),
                       "rv_rising": (spy_r.rolling(5).std() > spy_r.rolling(21).std()).reindex(uvxy_r.index)}).dropna()
for tag, mask in [("VIX<15", regime.vix_lvl < 15), ("VIX 15-25", (regime.vix_lvl>=15)&(regime.vix_lvl<25)),
                  ("VIX>=25", regime.vix_lvl >= 25),
                  ("VIX>MA63 & rv_rising", (regime.vix_vs_ma>1)&(regime.rv_rising)),
                  ("VIX<MA63 & rv falling", (regime.vix_vs_ma<1)&(~regime.rv_rising))]:
    sub = regime.u[mask]
    print(f"  {tag:<24} n={len(sub):>5}  drift {sub.mean():+.2%}/d")

# ══════════ #29 complacency-timed convexity ══════════
print(f"\n=== #29 complacency-timed UVXY convexity ({time.time()-t0:.0f}s) ===")
from strategies.contrarian_bubble_hourly import _bubble_matrix
hc = pd.read_parquet("data/cache/merged_hourly_close.parquet"); hc.index = pd.to_datetime(hc.index)
uni = hc.drop(columns=[c for c in ("SPY",) if c in hc.columns])
B = _bubble_matrix(uni, 104)
valid = uni.notna().values
breadth_bar = pd.Series((( B < -0.8) & valid).sum(1) / np.maximum(valid.sum(1), 1), index=hc.index)
breadth = breadth_bar.groupby(breadth_bar.index.normalize()).last()
breadth.index = pd.to_datetime(breadth.index)

vrp = ((vix/100.0)**2 - (spy_r.rolling(21).std()**2 * TD)).dropna()
vrp_pct = vrp.rolling(756).rank(pct=True)
idx = breadth.index.intersection(uvxy_r.index).intersection(vrp.index)
complacent = (breadth.reindex(idx) <= breadth.reindex(idx).rolling(504, min_periods=252).quantile(0.20))
vrp_lowfall = (vrp_pct.reindex(idx) < 0.40) & (vrp.reindex(idx).diff(5) < 0)
ignite = (spy_r.rolling(5).std() > spy_r.rolling(21).std()).reindex(idx)
sig = (complacent & vrp_lowfall & ignite).shift(1).fillna(False)

rows, i, days = [], 0, list(idx)
in_mkt_year = {}
while i < len(days) - 5:
    d = days[i]; yr = d.year
    if bool(sig.loc[d]) and in_mkt_year.get(yr, 0) < 30:
        for k in range(5):
            dd = days[i+k]
            rr = float(uvxy_r.reindex([dd]).fillna(0).iloc[0])
            rows.append((dd, rr - (TC if k in (0,4) else 0)))
        in_mkt_year[yr] = in_mkt_year.get(yr, 0) + 5
        i += 5
    else:
        rows.append((days[i], 0.0)); i += 1
conv = pd.Series(dict(rows)).sort_index()
m29 = metrics_from_returns(conv.values, TD)
act = float((conv != 0).mean())
print(f"  standalone: Sharpe {m29['sharpe']:.2f} | CAGR {m29['cagr']:.1%} | in-market {act:.0%} of days")
m, r, sd = contrib("convexity book", conv, shares=0.5)
if r["significant_p10"]:
    record_performance(name="book_v_convexity", dates=conv.index, returns=conv.values,
        params={"breadth_q": 0.2, "vrp_pct": 0.4, "hold": 5, "budget": 30},
        data_period=f"{conv.index.min().date()}..{conv.index.max().date()}", periods_per_year=TD,
        extra={"cycle": "Cycle 14 #29", "origin": "ORIGINAL synthesis"})
    record_improvement("Complacency-timed UVXY convexity book", "ORIGINAL (registry #103-104 inputs)",
        metrics_from_returns(champ.values, TD), m, ["book_v_convexity"], f"p={r['p_one_sided']:.3f}")
    print("  IMPROVED -> recorded")

# ══════════ #30 F residual-momentum re-rank ══════════
print(f"\n=== #30 F residual-momentum re-rank ({time.time()-t0:.0f}s) ===")
ho = pd.read_parquet("data/cache/merged_hourly_open.parquet"); ho.index = pd.to_datetime(ho.index)
common = sorted((set(ho.columns) & set(hc.columns)) - {"SPY"})
hcc, hoo = hc[common], ho[common]
hc_np = hcc.values.astype(np.float32); ho_np = hoo.values.astype(np.float32)
bar_ts = hcc.index; n, U = hc_np.shape
LB, HOLD, TOPN = 750, 200, 5
r1 = pd.DataFrame(hc_np, index=bar_ts).pct_change()
mkt = r1.mean(axis=1)
mm = mkt.rolling(LB).mean(); mv = mkt.rolling(LB).var()
cov = r1.multiply(mkt, axis=0).rolling(LB).mean() - r1.rolling(LB).mean().multiply(mm, axis=0)
beta = cov.div(mv, axis=0).values.astype(np.float32)
mkt_idx = (1+mkt.fillna(0)).cumprod().values.astype(np.float32)
with np.errstate(divide="ignore", invalid="ignore"):
    mom = hc_np / np.vstack([np.full((LB, U), np.nan, dtype=np.float32), hc_np[:-LB]]) - 1
    mkt_mom = mkt_idx / np.concatenate([np.full(LB, np.nan, dtype=np.float32), mkt_idx[:-LB]]) - 1
resid_mom = mom - beta * mkt_mom[:, None]

def run_f(score_np):
    H = np.zeros((n, U), dtype=np.float32)
    em = np.zeros(n, bool); xm = np.zeros(n, bool)
    i = LB
    while i + HOLD < n:
        row = score_np[i]; valid_ = np.where(np.isfinite(row))[0]
        if len(valid_) >= TOPN:
            top = valid_[np.argsort(row[valid_])[-TOPN:]]
            H[i+1:i+HOLD+1, top] = 1.0/TOPN; em[i+1] = True; xm[i+HOLD] = True
        i += HOLD
    with np.errstate(divide="ignore", invalid="ignore"):
        o2c = np.where((ho_np>0)&np.isfinite(ho_np)&np.isfinite(hc_np), hc_np/ho_np-1, 0.0)
        cp = np.vstack([hc_np[:1], hc_np[:-1]])
        c2c = np.where((cp>0)&np.isfinite(cp)&np.isfinite(hc_np), hc_np/cp-1, 0.0)
    br = np.where(em[:, None], o2c, c2c)
    port = (H*br).sum(1); port[em] -= TC; port[xm] -= TC
    port[H.sum(1) == 0] = 0.0
    s = pd.Series(port.astype(float), index=bar_ts)
    d = s.groupby(s.index.normalize()).apply(lambda g: float((1+g).prod()-1))
    d.index = pd.to_datetime(d.index)
    return d

f_anchor = run_f(mom)
ma_ = metrics_from_returns(f_anchor[f_anchor.index <= END].values, TD)
print(f"  anchor (raw rank): Sharpe {ma_['sharpe']:.3f} {'OK' if abs(ma_['sharpe']-1.548)<0.06 else 'ANCHOR FAIL — VOID #30'}")
if abs(ma_['sharpe']-1.548) < 0.06:
    f_res = run_f(resid_mom)
    f_res = f_res[f_res.index <= END]
    mres = metrics_from_returns(f_res.values, TD)
    print(f"  residual rank F:   Sharpe {mres['sharpe']:.3f} | CAGR {mres['cagr']:.1%} | MaxDD {mres['max_dd']:.1%}")
    books = dict(BASE); books["F"] = f_res
    saved = C.ALLOC_BOOKS; C.ALLOC_BOOKS = ["A","C","D","F"]
    R = pd.DataFrame(books); R = R[(R.index >= "2019-01-02") & (R.index <= END)].fillna(0.0)
    ser30 = E.ivol_voltgt(R).dropna(); C.ALLOC_BOOKS = saved
    m30 = metrics_from_returns(ser30.values, TD)
    r30 = sharpe_delta_test(ser30, champ)
    print(f"  champ with resid-F: Sharpe {m30['sharpe']:.3f} (delta {r30['delta']:+.3f}, p {r30['p_one_sided']:.3f})")

# ══════════ #31 crisis-only short/flat SPY ══════════
print(f"\n=== #31 crisis-only short/flat SPY sleeve ===")
tsmom = spy.pct_change(252).shift(21)
fast = spy.pct_change(21)
short_sig = ((tsmom < 0) & (fast < 0)).shift(1).fillna(False)
pos = -short_sig.astype(float)
borrow = 0.08/TD
s31 = (spy_r*pos - pos.diff().abs().fillna(0)*TC + pos*borrow).dropna()  # pos negative -> pays borrow
s31 = s31[s31.index >= "1997-06-01"]
m31 = metrics_from_returns(s31.values, TD)
print(f"  standalone 1997+: Sharpe {m31['sharpe']:.2f} | CAGR {m31['cagr']:.1%} | "
      f"short {float((pos!=0).mean()):.0%} of days")
contrib("crisis-short sleeve", s31.reindex(champ.index).fillna(0), shares=0.5)

# ══════════ #32 utilities/SPY rotation ══════════
print(f"\n=== #32 utilities/SPY rotation ===")
if "XLU" in daily.columns:
    xlu = daily["XLU"].dropna()
else:
    try:
        import yfinance as yf
        xlu = yf.download("XLU", start="1999-01-01", auto_adjust=True, progress=False)["Close"]
        if isinstance(xlu, pd.DataFrame): xlu = xlu.iloc[:, 0]
        xlu.index = pd.to_datetime(xlu.index).tz_localize(None)
    except Exception as ex:
        xlu = None; print(f"  XLU unavailable ({ex}) — data-blocked")
if xlu is not None:
    rel = (xlu/spy).dropna()
    sig32 = (rel > rel.rolling(63).mean()).shift(1).fillna(False)   # utilities leading = defense
    xlu_r = xlu.pct_change()
    s32 = pd.Series(np.where(sig32, xlu_r.reindex(sig32.index), spy_r.reindex(sig32.index)),
                    index=sig32.index)
    chg = sig32.astype(int).diff().abs().fillna(0)
    s32 = (s32 - chg*2*TC).dropna()
    m32 = metrics_from_returns(s32[s32.index >= "2000-01-01"].values, TD)
    print(f"  standalone 2000+: Sharpe {m32['sharpe']:.2f} | CAGR {m32['cagr']:.1%} | MaxDD {m32['max_dd']:.1%}")
    contrib("XLU rotation", s32.reindex(champ.index).fillna(0))
print(f"\ntotal {time.time()-t0:.0f}s")
