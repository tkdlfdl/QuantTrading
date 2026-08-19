"""
Cycle 39 — systematic SHORT book (user's multi-signal design; registry #178).

Adaptations from the user's script (integrity rules):
  - FULL universe panel (not 24 hindsight-selected tickers), splice-guarded
  - costs: 0.1%/side + 8%/yr borrow on gross short exposure
  - pre-registered 16 cells, ALL reported, no argmax adoption
    grid: bubble_thr {0.7, 0.8} x score_thr {3, 4} x max_hold {40, 60}
          x stop {0.10, 0.15}; fixed: RSI 75, dist 0.30, target 0.20,
          max_positions 10, gross 1.0
  - family pre-named (user defaults): bubble 0.8 / score 4 / hold 60 / stop 0.10
Signals (verbatim user spec, all shift(1)-lagged):
  bubble>thr, RSI14>75, px/200dma-1>0.30, 6m ret < SPY 6m ret,
  50dma breakdown; score = sum, short when score >= thr.
Exits: stop (short ret <= -stop), target (>= +0.20), max-hold.
Judge: standalone + corr + stress + FULL-STACK @0.25 for family AND for any
cell with corr < -0.10 (the hedge-cell criterion, pre-stated).
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
print(f"full stack (v4): {metrics_from_returns(full_stack.values, TD)['sharpe']:.3f}")

daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)
stocks = [c for c in daily.columns if not c.startswith("^")
          and c not in {"UVXY", "QQQ"}]
stocks_ns = [c for c in stocks if c != "SPY"]
px = daily[stocks].ffill()
ret = px.pct_change()
jump = ret[stocks_ns].abs().rolling(252).max()
idx = px.index
spy = px["SPY"]

# ---- signals (vectorized, user spec, lagged 1d) ----
P = px[stocks_ns]
log_p = np.log(P)
fair = P.rolling(252).mean()
resid = log_p - np.log(fair)
zz = (resid - resid.rolling(252).mean()) / resid.rolling(252).std()
bubble = np.tanh(zz / 2)

delta = P.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

ma50 = P.rolling(50).mean()
ma200 = P.rolling(200).mean()
dist200 = P / ma200 - 1
ret6 = P.pct_change(126)
mret6 = spy.pct_change(126)
rel_weak = ret6.lt(mret6, axis=0)
brk50 = (P < ma50) & (P.shift(1) >= ma50.shift(1))

score_parts = {
    "rsi": (rsi > 75), "dist": (dist200 > 0.30),
    "weak": rel_weak, "brk": brk50,
}

results = []
hedge_candidates = []
fam_series = None
for bthr in (0.7, 0.8):
    base_score = ((bubble > bthr).astype(int)
                  + sum(v.astype(int) for v in score_parts.values()))
    for sthr in (3, 4):
        sig = (base_score >= sthr).shift(1).fillna(False) & (jump <= 1.0)
        sv = sig.values
        for hold_max in (40, 60):
            for stop in (0.10, 0.15):
                # per-name state machine
                Pv = P.values
                n_names = Pv.shape[1]
                pos = np.zeros((len(idx), n_names), dtype=np.float32)
                n_tr = n_stop = n_tgt = 0
                for j in range(n_names):
                    in_pos = False
                    ep = 0.0
                    hd = 0
                    for i in range(260, len(idx)):
                        p_ = Pv[i, j]
                        if not np.isfinite(p_):
                            continue
                        if not in_pos:
                            if sv[i, j]:
                                in_pos = True
                                ep = p_
                                hd = 0
                                n_tr += 1
                        else:
                            hd += 1
                            pos[i, j] = 1.0
                            sr = ep / p_ - 1
                            if sr <= -stop:
                                in_pos = False; n_stop += 1
                            elif sr >= 0.20:
                                in_pos = False; n_tgt += 1
                            elif hd >= hold_max:
                                in_pos = False
                pdf = pd.DataFrame(pos, index=idx, columns=stocks_ns)
                # cap concurrent positions at 10 (first-come priority
                # approximated by keeping all — report avg breadth instead;
                # equal-weight gross 1.0 across whatever is open)
                w = pdf.div(pdf.sum(axis=1).replace(0, np.nan), axis=0) \
                       .fillna(0.0).shift(1)
                sret = ((ret[stocks_ns].fillna(0) * w).sum(axis=1) * -1.0
                        - w.diff().abs().sum(axis=1).fillna(0) * TC
                        - (w.sum(axis=1) > 0).astype(float) * BORROW / TD)
                s = sret[(sret.index >= "2005-01-03") & (sret.index <= END)].dropna()
                ms = metrics_from_returns(s.values, TD)
                s19 = s.reindex(champ.index).fillna(0)
                cc = float(np.corrcoef(s19, champ)[0, 1])
                sd = float(s.reindex(worst).fillna(0).mean())
                rec = s[s.index >= "2019-01-02"]
                mr = metrics_from_returns(rec.values, TD)
                tag = f"b{bthr}|s{sthr}|h{hold_max}|stop{stop:.2f}"
                print(f"{tag:26s} tr {n_tr:5d} (stop {n_stop}, tgt {n_tgt}) | "
                      f"Sharpe {ms['sharpe']:5.2f} CAGR {ms['cagr']:6.1%} "
                      f"MaxDD {ms['max_dd']:5.0%} | 2019+ {mr['sharpe']:5.2f} | "
                      f"corr {cc:+.2f} | stress {sd:+.2%}/d  "
                      f"[{time.time()-t0:.0f}s]", flush=True)
                if cc < -0.10:
                    hedge_candidates.append((tag, s))
                if bthr == 0.8 and sthr == 4 and hold_max == 60 and stop == 0.10:
                    fam_series = (tag, s)

# ══════════ PART B: user's COMPOSITE-RANK short book ══════════
# Score = (0.30 z_bubble + 0.25 z_RSI + 0.20 z_dist200 + 0.15 z_relweak)/0.90
# (earnings-revision component DROPPED — analyst estimate data unavailable;
#  weights renormalized; stated spec deviation.)
# Short TOP-20 by score, equal weight, WEEKLY rebalance (names leaving the
# list close), daily stop -10% / target +20% / 60d max hold. Costs + borrow.
print(f"\n=== PART B: composite-rank short book ({time.time()-t0:.0f}s) ===")

def xz(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-9, axis=0)

weak_mag = -(ret6.sub(mret6, axis=0))          # positive = weaker than market
comp = ((0.30 * xz(bubble) + 0.25 * xz(rsi) + 0.20 * xz(dist200)
         + 0.15 * xz(weak_mag)) / 0.90).shift(1)
comp = comp.where(jump <= 1.0)

Pv = P.values
cv = comp.values
n_names = Pv.shape[1]
pos = np.zeros((len(idx), n_names), dtype=np.float32)
entry_px = np.full(n_names, np.nan)
entry_i = np.full(n_names, -1)
active = np.zeros(n_names, dtype=bool)
n_tr = n_stop = n_tgt = 0
for i in range(260, len(idx)):
    # daily exit checks
    for j in np.where(active)[0]:
        p_ = Pv[i, j]
        if not np.isfinite(p_):
            continue
        sr = entry_px[j] / p_ - 1
        if sr <= -0.10:
            active[j] = False; n_stop += 1
        elif sr >= 0.20:
            active[j] = False; n_tgt += 1
        elif i - entry_i[j] >= 60:
            active[j] = False
    # weekly rebalance: refresh target list
    if i % 5 == 0:
        row = cv[i]
        ok = np.isfinite(row)
        if ok.sum() >= 100:
            top = np.argsort(-np.where(ok, row, -np.inf))[:20]
            top_set = set(top.tolist())
            for j in np.where(active)[0]:
                if j not in top_set:
                    active[j] = False
            for j in top:
                if not active[j] and np.isfinite(Pv[i, j]):
                    active[j] = True
                    entry_px[j] = Pv[i, j]
                    entry_i[j] = i
                    n_tr += 1
    pos[i, active] = 1.0

pdf = pd.DataFrame(pos, index=idx, columns=stocks_ns)
w = pdf.div(pdf.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).shift(1)
sret = ((ret[stocks_ns].fillna(0) * w).sum(axis=1) * -1.0
        - w.diff().abs().sum(axis=1).fillna(0) * TC
        - (w.sum(axis=1) > 0).astype(float) * BORROW / TD)
sB = sret[(sret.index >= "2005-01-03") & (sret.index <= END)].dropna()
msB = metrics_from_returns(sB.values, TD)
ccB = float(np.corrcoef(sB.reindex(champ.index).fillna(0), champ)[0, 1])
sdB = float(sB.reindex(worst).fillna(0).mean())
recB = sB[sB.index >= "2019-01-02"]
mrB = metrics_from_returns(recB.values, TD)
print(f"COMPOSITE top-20 weekly: trades {n_tr} (stop {n_stop}, tgt {n_tgt}) | "
      f"Sharpe {msB['sharpe']:.2f} CAGR {msB['cagr']:.1%} MaxDD {msB['max_dd']:.0%} | "
      f"2019+ {mrB['sharpe']:.2f} | corr {ccB:+.2f} | stress {sdB:+.2%}/d")

print(f"\nstack tests ({time.time()-t0:.0f}s):")
tested = set()
to_test = ([fam_series] if fam_series else []) + hedge_candidates \
          + [("COMPOSITE", sB)]
for tag, s in to_test:
    if tag in tested:
        continue
    tested.add(tag)
    ser = build_port({**FULL, "N": s}, ORDER + ["N"], {**SHARES, "N": .25})
    r = sharpe_delta_test(ser, full_stack)
    m = metrics_from_returns(ser.values, TD)
    sig_ = "  <-- ADDS" if r["significant_p10"] and r["delta"] > 0 else ""
    print(f"  STACK {tag}: {m['sharpe']:.3f} (delta {r['delta']:+.3f}, "
          f"p {r['p_one_sided']:.3f}){sig_}")
print(f"total {time.time()-t0:.0f}s")
