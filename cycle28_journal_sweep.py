"""
Cycle 28 — tiered-journal sweep battery (registry #149-153).

 (a) Market intraday momentum (Gao et al. JFE 2018): hourly-bar version on
     SPY (fallback QQQ): predictor = overnight+first-hour return sign;
     position = that sign held for the LAST hour only. Costs 0.1%/side
     per round trip (2 legs/day when traded).
 (b) Heston-Sadka same-month seasonality: rank by average same-calendar-month
     return over past 10 years (min 5 obs); long top-20 equal-weight for the
     month. Splice-guarded.
 (c) George-Hwang 52wk-high: rank by px/252d-max; long top-20, monthly.
 (d) Barroso-Santa-Clara vol-scaled F: F book series scaled to 20% ann vol
     (trailing 63d), cap 2x — book-level constant-vol; compare vs raw F in
     champion slot (swap test).
 (e) GGR pairs: 252d formation on normalized prices (SSD), top-20 pairs,
     trade at |z|>2 (63d z), exit at zero-cross or 21d; both legs, 8%/yr
     borrow on short leg.

Gates: standalone from earliest data, corr(champ), stress days, FULL-STACK
marginal @0.25 (quiet-D basis + G/X/DU/FD), LW p. (d) is a swap test on F.
"""
from __future__ import annotations
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from tools.metrics import metrics_from_returns
from tools.significance import sharpe_delta_test
from live import engine as E, config as C

TD, TC, BORROW, END = 252, 0.001, 0.08, "2026-07-08"
t0 = time.time()

def load(n):
    df = pd.read_csv(f"strategies/performance/{n}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)

champ = load("portfolio_champ_d14")
d8q, d14 = load("book_d8_quiet"), load("book_d14")
ju = d8q.index.union(d14.index)
dblend = 0.5 * d8q.reindex(ju).fillna(0) + 0.5 * d14.reindex(ju).fillna(0)
FULL = {"A": load("book_a_retest"), "C": load("book_c_overlap_cap"),
        "D": dblend, "F": load("book_f_retest"),
        "G": load("book_g_live_spec"), "X": load("book_x_live_spec"),
        "DU": load("book_du_live_spec"), "FD": load("book_fdip_40h")}
worst = champ.nsmallest(int(len(champ) * 0.05)).index
SHARES = {"D": 2.0, "G": .25, "X": .25, "DU": .25, "FD": .25}
ORDER = ["A", "C", "D", "F", "G", "X", "DU", "FD"]


def build_port(books, alloc, shares):
    sb, ss = C.ALLOC_BOOKS, dict(C.ALLOC_SHARES)
    C.ALLOC_BOOKS = alloc
    C.ALLOC_SHARES = shares
    try:
        R = pd.DataFrame(books)
        R = R[(R.index >= "2019-01-02") & (R.index <= END)].fillna(0.0)
        return E.ivol_voltgt(R).dropna()
    finally:
        C.ALLOC_BOOKS, C.ALLOC_SHARES = sb, ss


full_stack = build_port(FULL, ORDER, SHARES)
m_fs = metrics_from_returns(full_stack.values, TD)
print(f"full stack (v4 basis): {m_fs['sharpe']:.3f}  [{time.time()-t0:.0f}s]")


def judge(name, stream):
    stream = stream[stream.index <= END].dropna()
    ms = metrics_from_returns(stream.values, TD)
    s19 = stream.reindex(champ.index).fillna(0)
    cc = float(np.corrcoef(s19, champ)[0, 1])
    sd = float(stream.reindex(worst).fillna(0).mean())
    rec = stream[stream.index >= "2015-01-01"]
    mr = metrics_from_returns(rec.values, TD) if len(rec) > 100 else {"sharpe": np.nan}
    print(f"  {name}: standalone {stream.index.min().date()}..: "
          f"Sharpe {ms['sharpe']:.2f} CAGR {ms['cagr']:.1%} MaxDD {ms['max_dd']:.0%} "
          f"| 2015+ {mr['sharpe']:.2f} | corr(champ) {cc:+.2f} | stress {sd:+.2%}/d")
    ser = build_port({**FULL, "N": stream}, ORDER + ["N"], {**SHARES, "N": .25})
    r = sharpe_delta_test(ser, full_stack)
    m = metrics_from_returns(ser.values, TD)
    sig = "  <-- ADDS ON FULL STACK" if r["significant_p10"] and r["delta"] > 0 else ""
    print(f"    FULL-STACK +N @0.25: {m['sharpe']:.3f} (delta {r['delta']:+.3f}, "
          f"p {r['p_one_sided']:.3f}){sig}")


daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)
stocks = [c for c in daily.columns if not c.startswith("^") and c not in {"UVXY", "SPY", "QQQ"}]
px = daily[stocks].ffill()
ret = px.pct_change()
jump = ret.abs().rolling(252).max()
month_end = px.index.to_series().dt.month.diff().fillna(1) != 0

# ══════════ (a) market intraday momentum ══════════
print(f"\n=== (a) market intraday momentum (hourly) ===")
hc = pd.read_parquet("data/cache/merged_hourly_close.parquet")
hc.index = pd.to_datetime(hc.index)
sym = "SPY" if "SPY" in hc.columns else None
if sym is None:
    q = pd.read_parquet("data/cache/qqq_hourly_close.parquet")["QQQ"]
    q.index = pd.to_datetime(q.index).floor("h")
    ser_h = q[~q.index.duplicated(keep="last")]
    sym = "QQQ"
else:
    ser_h = hc["SPY"].dropna()
print(f"  instrument: {sym} hourly, {ser_h.index.min().date()}..")
bar_day = ser_h.index.normalize()
days = pd.unique(bar_day)
rows = []
grp = ser_h.groupby(bar_day)
prev_close = None
for d, g in grp:
    if len(g) < 4:
        prev_close = g.iloc[-1]
        continue
    first_h = g.iloc[0]
    on_first = (first_h / prev_close - 1) if prev_close and prev_close > 0 else np.nan
    last_ret = g.iloc[-1] / g.iloc[-2] - 1
    rows.append((d, on_first, last_ret))
    prev_close = g.iloc[-1]
im = pd.DataFrame(rows, columns=["d", "pred", "lastret"]).set_index("d").dropna()
pos = np.sign(im["pred"])
strat_a = pos * im["lastret"] - 2 * TC - (pos < 0) * BORROW / TD / 6.5
strat_a.index = pd.to_datetime(strat_a.index)
judge("intraday momentum (last hour)", strat_a)
# also long-signal-only variant (skip shorts + their costs)
pos_l = (im["pred"] > 0).astype(float)
strat_a2 = pos_l * im["lastret"] - pos_l * 2 * TC
strat_a2.index = pd.to_datetime(strat_a2.index)
judge("intraday momentum long-only", strat_a2)

# ══════════ (b) Heston-Sadka same-month seasonality ══════════
print(f"\n=== (b) same-month seasonality ({time.time()-t0:.0f}s) ===")
mret = px.resample("ME").last().pct_change()
seas_pos = pd.DataFrame(0.0, index=px.index, columns=stocks)
me_dates = px.index[month_end]
for d in me_dates:
    m_ = d.month
    hist = mret[(mret.index.month == m_) & (mret.index < d - pd.Timedelta(days=20))]
    hist = hist.iloc[-10:]
    if len(hist) < 5:
        continue
    avg = hist.mean()
    ok = avg.dropna()
    ok = ok[jump.loc[:d].iloc[-1][ok.index] <= 1.0]
    if len(ok) < 100:
        continue
    top = ok.nlargest(20).index
    nxt = me_dates[me_dates > d]
    until = nxt[0] if len(nxt) else px.index[-1]
    seas_pos.loc[d:until, top] = 1.0 / 20
w = seas_pos.shift(1)
strat_b = ((ret.fillna(0) * w).sum(axis=1)
           - w.diff().abs().sum(axis=1).fillna(0) * TC)
strat_b = strat_b[strat_b.index >= "1999-06-01"].dropna()
judge("same-month seasonality", strat_b)

# ══════════ (c) 52-week-high momentum ══════════
print(f"\n=== (c) 52wk-high momentum ({time.time()-t0:.0f}s) ===")
prox = (px / px.rolling(252).max()).mask(jump > 1.0).shift(1)
gh_pos = pd.DataFrame(0.0, index=px.index, columns=stocks)
for d in me_dates:
    row = prox.loc[d].dropna()
    if len(row) < 100:
        continue
    top = row.nlargest(20).index
    nxt = me_dates[me_dates > d]
    until = nxt[0] if len(nxt) else px.index[-1]
    gh_pos.loc[d:until, top] = 1.0 / 20
w = gh_pos.shift(1)
strat_c = ((ret.fillna(0) * w).sum(axis=1)
           - w.diff().abs().sum(axis=1).fillna(0) * TC)
strat_c = strat_c[strat_c.index >= "1999-06-01"].dropna()
judge("52wk-high momentum", strat_c)
fB = load("book_f_retest")
jf = strat_c.index.intersection(fB.index)
print(f"    corr(52wk-high, F): {np.corrcoef(strat_c.loc[jf], fB.loc[jf])[0,1]:+.2f}")

# ══════════ (d) Barroso-Santa-Clara vol-scaled F (swap test) ══════════
print(f"\n=== (d) vol-scaled F swap ({time.time()-t0:.0f}s) ===")
rv = fB.rolling(63).std().shift(1) * np.sqrt(TD)
scale = (0.20 / rv).clip(upper=2.0).fillna(1.0)
f_scaled = fB * scale
mf, mfs = metrics_from_returns(fB.dropna().values, TD), metrics_from_returns(f_scaled.dropna().values, TD)
print(f"  F raw {mf['sharpe']:.2f}/{mf['max_dd']:.0%} -> vol-scaled "
      f"{mfs['sharpe']:.2f}/{mfs['max_dd']:.0%}")
swap = dict(FULL); swap["F"] = f_scaled
ser_sw = build_port(swap, ORDER, SHARES)
r_sw = sharpe_delta_test(ser_sw, full_stack)
m_sw = metrics_from_returns(ser_sw.values, TD)
sig = "  <-- UPGRADE CANDIDATE" if r_sw["significant_p10"] and r_sw["delta"] > 0 else ""
print(f"  SWAP F->vol-scaled in full stack: {m_sw['sharpe']:.3f} "
      f"(delta {r_sw['delta']:+.3f}, p {r_sw['p_one_sided']:.3f}){sig}")

# ══════════ (e) GGR pairs ══════════
print(f"\n=== (e) GGR pairs ({time.time()-t0:.0f}s) ===")
lpx = np.log(px.mask(jump > 1.0))
strat_rows = {}
form_dates = me_dates[(me_dates >= "2000-01-01")][::6]      # semiannual formation
for fd_ in form_dates:
    loc = px.index.searchsorted(fd_)
    if loc < 260 or loc + 2 >= len(px.index):
        continue
    win = lpx.iloc[loc - 252:loc]
    norm = win - win.iloc[0]
    valid = [c for c in stocks if norm[c].notna().all()]
    if len(valid) < 150:
        continue
    sub = norm[valid].values
    # SSD over a liquidity-limited subset for tractability: top 200 by data
    vsub = sub[:, :200] if sub.shape[1] > 200 else sub
    names = valid[:200]
    n = vsub.shape[1]
    ssd = np.full((n, n), np.inf)
    for i in range(n):
        diff = vsub[:, i:i+1] - vsub[:, i+1:]
        ssd[i, i+1:] = (diff ** 2).sum(axis=0)
    pairs = []
    used = set()
    for flat in np.argsort(ssd, axis=None):
        i, j = divmod(flat, n)
        if not np.isfinite(ssd[i, j]):
            break
        if i in used or j in used:
            continue
        pairs.append((names[i], names[j]))
        used.add(i); used.add(j)
        if len(pairs) >= 20:
            break
    # trade window: next 6 months
    end_loc = min(loc + 126, len(px.index) - 1)
    spread = lpx.iloc[loc - 63:end_loc]
    for a_, b_ in pairs:
        sp = spread[a_] - spread[b_]
        z = (sp - sp.rolling(63).mean()) / sp.rolling(63).std()
        z = z.iloc[63:]
        state = 0
        entry_i = None
        for dt_, zv in z.items():
            if not np.isfinite(zv):
                continue
            if state == 0 and abs(zv) > 2:
                state = -np.sign(zv)      # long the cheap leg
                entry_i = dt_
            elif state != 0 and (np.sign(zv) != -state or abs(zv) < 0.1):
                # realize P&L daily while open instead: handled below
                state = 0
        # simpler daily P&L: position = -sign(z) lagged, while |z|>0.5
        posz = (-np.sign(z) * (z.abs() > 2).astype(float)).replace(0, np.nan).ffill()
        posz = posz.where(z.abs() > 0.25).fillna(0).shift(1).fillna(0)
        ra = px[a_].pct_change().reindex(z.index)
        rb = px[b_].pct_change().reindex(z.index)
        pl = posz * (ra - rb) - posz.diff().abs().fillna(0) * 2 * TC - posz.abs() * BORROW / TD
        for dt_, v in pl.dropna().items():
            strat_rows.setdefault(dt_, []).append(v)
strat_e = pd.Series({d: np.mean(v) for d, v in strat_rows.items()}).sort_index()
strat_e.index = pd.to_datetime(strat_e.index)
judge("GGR pairs (top-20, 2sigma)", strat_e)
print(f"\ntotal {time.time()-t0:.0f}s")
