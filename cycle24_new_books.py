"""
Cycle 24 — queue-refill research sweep: structurally-different candidates.

 (a) RESIDUAL momentum (Blitz-Huij-Martens 2011): 252d rolling beta vs SPY,
     momentum on residual returns (252-21 window), top-20, monthly.
 (b) LOW-VOL long-only (Blitz-van Vliet 2007): bottom-50 by 252d vol,
     monthly rebalance, equal weight.
 (c) SECTOR weekly reversal: long worst-2 of 11 SPDRs by trailing 5d return,
     non-overlapping 5d holds.
 (d) EARNINGS-premium feasibility probe: how deep is yfinance's earnings-date
     history for the universe? (data gate only — no test if coverage thin)
 (e) F OVERNIGHT-share diagnostic (Lou-Polk-Skouras on Book F's holdings):
     analysis only — where does F's return accrue?

Gates for (a)-(c): standalone (earliest data), corr vs champion, stress-day
behavior, FULL-STACK marginal @0.25 shares (champ(quiet-D)+G+X+DU), LW p.
Costs 0.1%/side research basis. Signals lagged (shift) — no look-ahead.
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

TD, TC, END = 252, 0.001, "2026-07-08"
t0 = time.time()

def load(n):
    df = pd.read_csv(f"strategies/performance/{n}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)

champ = load("portfolio_champ_d14")
d8q, d14 = load("book_d8_quiet"), load("book_d14")
ju = d8q.index.union(d14.index)
dblend = 0.5 * d8q.reindex(ju).fillna(0) + 0.5 * d14.reindex(ju).fillna(0)
BASE = {"A": load("book_a_retest"), "C": load("book_c_overlap_cap"),
        "D": dblend, "F": load("book_f_retest")}
FULL = dict(BASE)
FULL.update({"G": load("book_g_live_spec"), "X": load("book_x_live_spec"),
             "DU": load("book_du_live_spec")})
worst = champ.nsmallest(int(len(champ) * 0.05)).index


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


full_stack = build_port(FULL, ["A", "C", "D", "F", "G", "X", "DU"],
                        {"D": 2.0, "G": .25, "X": .25, "DU": .25})
m_fs = metrics_from_returns(full_stack.values, TD)
print(f"full stack (quiet-D basis): {m_fs['sharpe']:.3f}  [{time.time()-t0:.0f}s]")


def judge(name, stream):
    stream = stream.dropna()
    ms = metrics_from_returns(stream.values, TD)
    s19 = stream.reindex(champ.index).fillna(0)
    cc = float(np.corrcoef(s19, champ)[0, 1])
    sd = float(stream.reindex(worst).fillna(0).mean())
    print(f"  standalone {stream.index.min().date()}..: Sharpe {ms['sharpe']:.2f} "
          f"CAGR {ms['cagr']:.1%} MaxDD {ms['max_dd']:.0%} | corr(champ) {cc:+.2f} | "
          f"stress {sd:+.2%}/d")
    ser = build_port({**FULL, "N": stream},
                     ["A", "C", "D", "F", "G", "X", "DU", "N"],
                     {"D": 2.0, "G": .25, "X": .25, "DU": .25, "N": .25})
    r = sharpe_delta_test(ser, full_stack)
    m = metrics_from_returns(ser.values, TD)
    sig = "  <-- ADDS ON FULL STACK" if r["significant_p10"] and r["delta"] > 0 else ""
    print(f"  FULL-STACK +N @0.25: {m['sharpe']:.3f} (delta {r['delta']:+.3f}, "
          f"p {r['p_one_sided']:.3f}){sig}")
    return r


daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)
EXCL = {"SPY", "UVXY", "^VIX", "QQQ"}
stocks = [c for c in daily.columns if c not in EXCL and not c.startswith("^")]
px = daily[stocks].ffill()
ret = px.pct_change()
spy_r = daily["SPY"].pct_change()

# splice guard (research convention): any 1-day |move|>100% in trailing year
jump = ret.abs().rolling(252).max()

# ══════════ (a) residual momentum ══════════
print(f"\n=== (a) residual momentum (BHM 2011) ({time.time()-t0:.0f}s) ===")
cov = ret.rolling(252).cov(spy_r)
var = spy_r.rolling(252).var()
beta = cov.div(var, axis=0)
resid = ret.sub(beta.mul(spy_r, axis=0))
rmom = resid.rolling(231).sum().shift(21)          # 252-21 residual momentum
rvol = resid.rolling(231).std().shift(21)
score = (rmom / rvol).mask(jump > 1.0)             # scaled, splice-guarded
month_end = px.index.to_series().dt.month.diff().fillna(1) != 0
tgt = pd.DataFrame(index=px.index[month_end], columns=stocks, dtype=float)
for d in tgt.index:
    srow = score.loc[d].dropna()
    row = pd.Series(0.0, index=stocks)
    if len(srow) >= 100:
        for t_ in srow.nlargest(20).index:
            row[t_] = 1.0 / 20
    tgt.loc[d] = row
pos = tgt.reindex(px.index).ffill().fillna(0.0).shift(1)
sa = ((ret.fillna(0) * pos).sum(axis=1)
      - pos.diff().abs().sum(axis=1).fillna(0) * TC)
sa = sa[sa.index >= "1999-06-01"].dropna()
judge("residual momentum", sa)
fB = load("book_f_retest")
jf = sa.index.intersection(fB.index)
print(f"  corr(resid-mom, F): {np.corrcoef(sa.loc[jf], fB.loc[jf])[0,1]:+.2f} | "
      f"corr(resid-mom, A): "
      f"{np.corrcoef(sa.reindex(BASE['A'].index).fillna(0), BASE['A'])[0,1]:+.2f}")

# ══════════ (b) low-vol long-only ══════════
print(f"\n=== (b) low-vol long-only (BvV 2007) ({time.time()-t0:.0f}s) ===")
vol252 = ret.rolling(252).std()
lv_score = vol252.mask(jump > 1.0).shift(1)
tgt = pd.DataFrame(index=px.index[month_end], columns=stocks, dtype=float)
for d in tgt.index:
    srow = lv_score.loc[d].dropna()
    srow = srow[srow > 0]
    row = pd.Series(0.0, index=stocks)
    if len(srow) >= 100:
        for t_ in srow.nsmallest(50).index:
            row[t_] = 1.0 / 50
    tgt.loc[d] = row
pos = tgt.reindex(px.index).ffill().fillna(0.0).shift(1)
sb_ = ((ret.fillna(0) * pos).sum(axis=1)
       - pos.diff().abs().sum(axis=1).fillna(0) * TC)
sb_ = sb_[sb_.index >= "1999-06-01"].dropna()
judge("low-vol long-only", sb_)

# ══════════ (c) sector weekly reversal ══════════
print(f"\n=== (c) sector SPDR weekly reversal ({time.time()-t0:.0f}s) ===")
sect = pd.read_parquet("data/cache/sector_etf_close.parquet")
sect.index = pd.to_datetime(sect.index)
sret = sect.pct_change()
r5 = sect.pct_change(5).shift(1)
pos_s = pd.DataFrame(0.0, index=sect.index, columns=sect.columns)
i = 10
dates = sect.index
while i < len(dates) - 5:
    row = r5.iloc[i].dropna()
    if len(row) >= 8:
        losers = row.nsmallest(2).index
        pos_s.iloc[i + 1:i + 6][losers] = 0.5
        i += 5
    else:
        i += 1
sc_ = ((sret.fillna(0) * pos_s).sum(axis=1)
       - pos_s.diff().abs().sum(axis=1).fillna(0) * TC)
sc_ = sc_[sc_.index >= sect.index[15]].dropna()
judge("sector weekly reversal", sc_)

# ══════════ (d) earnings-calendar feasibility probe ══════════
print(f"\n=== (d) earnings-date coverage probe ({time.time()-t0:.0f}s) ===")
try:
    import yfinance as yf
    from concurrent.futures import ThreadPoolExecutor
    probe = stocks[:15]
    def ed(t_):
        try:
            e = yf.Ticker(t_).get_earnings_dates(limit=60)
            if e is None or len(e) == 0:
                return t_, None, 0
            idx_ = pd.to_datetime(e.index).tz_localize(None)
            return t_, idx_.min(), len(idx_)
        except Exception:
            return t_, None, 0
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(ed, probe))
    ok = [(t_, d_, n_) for t_, d_, n_ in res if d_ is not None]
    if ok:
        starts = sorted(d_ for _, d_, _ in ok)
        print(f"  {len(ok)}/15 have data; earliest date median "
              f"{starts[len(starts)//2].date()}, min {starts[0].date()}, "
              f"avg entries {np.mean([n_ for _, _, n_ in ok]):.0f}")
        if starts[len(starts) // 2] > pd.Timestamp("2023-01-01"):
            print("  VERDICT: history too shallow for a >=2yr backtest with "
                  "in-sample discipline -> DATA-BLOCKED (registry note)")
        else:
            print("  VERDICT: coverage sufficient -> queue full acquisition")
    else:
        print("  no earnings data returned -> DATA-BLOCKED")
except Exception as e:
    print(f"  probe failed ({e}) -> DATA-BLOCKED")

# ══════════ (e) F overnight-share diagnostic ══════════
print(f"\n=== (e) Book F overnight-share diagnostic (LPS) ({time.time()-t0:.0f}s) ===")
hc = pd.read_parquet("data/cache/merged_hourly_close.parquet")
hc.index = pd.to_datetime(hc.index)
ho = pd.read_parquet("data/cache/merged_hourly_open.parquet")
ho.index = pd.to_datetime(ho.index)
common = sorted((set(ho.columns) & set(hc.columns)) - {"SPY"})
hcc, hoo = hc[common].ffill(), ho[common].ffill()
idx = hcc.index
prices, opens = hcc.values, hoo.values
T, U = prices.shape
bar_day = idx.normalize().values
bdi = pd.factorize(bar_day)[0]
mom = hcc.pct_change(750).values
on_sum, in_sum, blocks = 0.0, 0.0, 0
i = 800
while i < T - 201:
    row = mom[i]
    if np.isfinite(row).sum() >= 100:
        top = np.argsort(row)[-5:]
        for s in top:
            for tt in range(i + 2, i + 201):
                if not (np.isfinite(prices[tt - 1, s]) and np.isfinite(opens[tt, s])
                        and np.isfinite(prices[tt, s])):
                    continue
                if bdi[tt] != bdi[tt - 1]:
                    on_sum += np.log(max(opens[tt, s] / prices[tt - 1, s], 1e-6))
                    in_sum += np.log(max(prices[tt, s] / opens[tt, s], 1e-6))
                else:
                    in_sum += np.log(max(prices[tt, s] / prices[tt - 1, s], 1e-6))
        blocks += 1
        i += 200
    else:
        i += 1
tot = on_sum + in_sum
print(f"  {blocks} F holding blocks | log-return split: overnight {on_sum:+.2f} "
      f"({on_sum/tot:.0%}) vs intraday {in_sum:+.2f} ({in_sum/tot:.0%})")
print("  (LPS predicts momentum accrues overnight; if so, an overnight-only F "
      "variant pays 2x daily costs — viability depends on the split size)")
print(f"\ntotal {time.time()-t0:.0f}s")
