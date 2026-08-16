"""
Improvement Cycle 10 — queue #17-20: four NEW-BOOK candidates.

Each gets (a) full-history standalone backtest (earliest data, 0.1%/side costs,
no look-ahead: signals lagged, next-day execution), then (b) the standing
portfolio-contribution test vs champion 2.580/43.5%/-8.9% (anchor rebuilt).

 #17 GKM high-volume premium (Gervais-Kaniel-Mingelgrin 2001 JF): daily volume
     in top decile vs own trailing 50d AND |price z| < 3 (strip Book C overlap)
     -> long top-10 by volume shock, hold 20d, non-overlapping, monthly-ish.
 #18 Turn-of-month SPY (Etula+ 2020 RFS): long SPY from T-3 close to T+3 close
     each month, flat otherwise.
 #19 Overnight-share cross-section (LPS 2019): rank trailing 252d
     (overnight - intraday) cumulative spread; long top-10, 21d holds.
     Kill gate: corr to F > 0.40.
 #20 VRP-proxy SPY timing (Bollerslev-Tauchen-Zhou 2009): VRP = (VIX/100)^2 -
     realized 21d var; exposure tiers by VRP rolling-3yr percentile
     (>=60% -> 1.0, 30-60% -> 0.5, <30% -> 0.0). MUST beat a constant
     matched-average-exposure SPY benchmark (beta-not-alpha lesson).
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from data.db.client import get_conn
from tools.metrics import metrics_from_returns
from tools.record import record_performance, record_improvement

TD, TC = 252, 0.001
CHAMP = dict(sharpe=2.580, cagr=0.435, max_dd=-0.089)

con = get_conn()
px = con.execute("""SELECT ts, symbol, open, close, volume FROM ohlcv
                    WHERE interval='1d' AND ts >= '1997-01-01'""").df()
px["ts"] = pd.to_datetime(px["ts"])
close = px.pivot_table(index="ts", columns="symbol", values="close").sort_index()
openp = px.pivot_table(index="ts", columns="symbol", values="open").sort_index()
vol   = px.pivot_table(index="ts", columns="symbol", values="volume").sort_index()
ret1d = close.pct_change()
print(f"panel {close.shape}")

daily_ext = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily_ext.index = pd.to_datetime(daily_ext.index)
spy = daily_ext["SPY"].dropna()
vix = daily_ext["^VIX"].dropna()
spy_ret = spy.pct_change()

def load(n):
    df = pd.read_csv(f"strategies/performance/{n}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)
BASE_BOOKS = {"A": load("book_a_retest"), "C": load("book_c_overlap_cap"),
              "D": load("book_d_retest"), "F": load("book_f_retest")}

def build_champ(books):
    idx = sorted(set().union(*[s.index for s in books.values()]))
    idx = pd.DatetimeIndex([d for d in idx if d >= pd.Timestamp("2019-01-02")])
    R = pd.DataFrame({k: books[k].reindex(idx).fillna(0.0) for k in books})
    panic = ((spy.pct_change(504) < 0) &
             (spy_ret.rolling(63).std()*np.sqrt(TD) >
              (spy_ret.rolling(63).std()*np.sqrt(TD)).rolling(756).quantile(0.8))
            ).shift(1).reindex(idx).fillna(False)
    cols = list(R.columns)
    iA = cols.index("A"); iF = cols.index("F"); iD = cols.index("D")
    n = len(R); port = np.zeros(n); w = None
    for t in range(n):
        if w is None or t % 21 == 0:
            hist = R.iloc[max(0, t-60):t]
            v = hist.std() * np.sqrt(TD)
            iv = np.array([1.0/x if np.isfinite(x) and x > 1e-9 else 0.0 for x in v])
            w = iv/iv.sum() if iv.sum() else np.ones(len(cols))/len(cols)
        wt = w.copy()
        if bool(panic.iloc[t]):
            freed = 0.5*(wt[iA]+wt[iF]); wt[iA] *= 0.5; wt[iF] *= 0.5; wt[iD] += freed
        port[t] = float(wt @ R.iloc[t].values)
    ser = pd.Series(port, index=idx)
    rv = ser.rolling(20).std().shift(1) * np.sqrt(TD)
    return ser * (0.15/rv).clip(upper=1.0).fillna(1.0)

anchor_m = metrics_from_returns(build_champ(BASE_BOOKS).values, TD)
assert abs(anchor_m["sharpe"] - CHAMP["sharpe"]) < 0.05, "anchor failed"
print(f"anchor OK ({anchor_m['sharpe']:.3f})")

def judge(name, ser, corr_gate=None):
    ser = ser.dropna()
    m = metrics_from_returns(ser.values, TD)
    corr = {}
    for bk, s in BASE_BOOKS.items():
        j = ser.index.intersection(s.index)
        if len(j) > 100:
            corr[bk] = float(np.corrcoef(ser.loc[j], s.loc[j])[0,1])
    print(f"  standalone: Sharpe {m['sharpe']:.3f} | CAGR {m['cagr']:.1%} | "
          f"MaxDD {m['max_dd']:.1%} | corr {dict((k, round(v,2)) for k,v in corr.items())}")
    if corr_gate and any(corr.get(k, 0) > v for k, v in corr_gate.items()):
        print(f"  KILL GATE hit ({corr_gate}) — benched without portfolio test")
        return m, None
    books2 = dict(BASE_BOOKS); books2["X"] = ser
    port = build_champ(books2)
    mp = metrics_from_returns(port.values, TD)
    worth = (mp["sharpe"] > anchor_m["sharpe"] or mp["cagr"] > anchor_m["cagr"]) \
            and mp["max_dd"] >= anchor_m["max_dd"]*1.2
    print(f"  portfolio-contribution: champ+{name} Sharpe {mp['sharpe']:.3f} | "
          f"CAGR {mp['cagr']:.1%} | MaxDD {mp['max_dd']:.1%} -> "
          f"{'WORTH ADDING' if worth else 'bench'}")
    return m, (mp if worth else None)

# ═════════ #17 GKM high-volume premium ═════════
print("\n=== #17 GKM high-volume premium ===")
vshock = vol.rank(axis=0) * np.nan  # placeholder shape
v50m = vol.rolling(50).mean(); v50s = vol.rolling(50).std()
vz = ((vol - v50m) / v50s).shift(1)
rz = ((ret1d - ret1d.rolling(20).mean()) / ret1d.rolling(20).std()).shift(1)
dates = close.index; n = len(dates)
rows = []; i = 300
while i + 20 < n:
    d = dates[i]
    z = vz.loc[d].dropna()
    pz = rz.loc[d]
    elig = z[(z > 0)].index
    elig = [t for t in elig if pd.notna(pz.get(t)) and abs(pz[t]) < 3]
    if len(elig) >= 10:
        picks = z[elig].nlargest(10).index
        fwd = ret1d.iloc[i:i+20][picks]
        pr = fwd.mean(axis=1).fillna(0.0); pr.iloc[0] -= 2*TC
        for dt, x in pr.items(): rows.append((dt, float(x)))
        i += 20
    else:
        rows.append((d, 0.0)); i += 1
ser17 = pd.Series(dict(rows)).sort_index(); ser17 = ser17[~ser17.index.duplicated(keep="last")]
m17, w17 = judge("gkm_volume", ser17)

# ═════════ #18 turn-of-month SPY ═════════
print("\n=== #18 turn-of-month SPY ===")
sd = spy.index
month = pd.Series(sd.month, index=sd)
tom_pos = pd.Series(0.0, index=sd)
# position held from T-3 close to T+3 close: in-market on days T-2..T+3 (returns)
day_in_month = pd.Series(np.arange(len(sd)), index=sd).groupby([sd.year, sd.month]).cumcount()
days_left = pd.Series(np.arange(len(sd)), index=sd).groupby([sd.year, sd.month]).transform("count") - 1 - day_in_month
in_win = (days_left <= 2) | (day_in_month <= 2)   # last 3 (T-3 close onward) + first 3
pos18 = in_win.astype(float).shift(0)             # deterministic calendar — no lookahead
r18 = spy_ret * pos18.shift(1)                    # position decided at prior close
# costs: entry+exit once per month
chg = pos18.diff().abs().fillna(0.0)
r18 = (r18 - chg.shift(0)*TC).dropna()
ser18 = r18[r18.index >= "1997-06-01"]
m18, w18 = judge("tom_spy", ser18)

# ═════════ #19 overnight-share cross-section ═════════
print("\n=== #19 overnight-share cross-section ===")
on_ret = (openp / close.shift(1) - 1)
in_ret = (close / openp - 1)
spread = (on_ret - in_ret).rolling(252).sum().shift(1)
rows = []; i = 300
while i + 21 < n:
    d = dates[i]
    s = spread.loc[d].dropna()
    if len(s) >= 50:
        picks = s.nlargest(10).index
        fwd = ret1d.iloc[i:i+21][picks]
        pr = fwd.mean(axis=1).fillna(0.0); pr.iloc[0] -= 2*TC
        for dt, x in pr.items(): rows.append((dt, float(x)))
        i += 21
    else:
        rows.append((d, 0.0)); i += 1
ser19 = pd.Series(dict(rows)).sort_index(); ser19 = ser19[~ser19.index.duplicated(keep="last")]
m19, w19 = judge("overnight_share", ser19, corr_gate={"F": 0.40})

# ═════════ #20 VRP-proxy SPY timing ═════════
print("\n=== #20 VRP-proxy SPY timing ===")
rv21 = (spy_ret.rolling(21).std()**2) * TD
vrp = ((vix/100.0)**2 - rv21).dropna()
pct = vrp.rolling(756).rank(pct=True).shift(1)
expo = pd.Series(np.where(pct >= 0.6, 1.0, np.where(pct >= 0.3, 0.5, 0.0)), index=pct.index)
r20 = (spy_ret * expo).dropna()
chg = expo.diff().abs().fillna(0.0)
r20 = (r20 - chg*TC).dropna()
ser20 = r20[r20.index >= "2000-01-01"]
m20 = metrics_from_returns(ser20.values, TD)
avg_expo = float(expo.reindex(ser20.index).mean())
bench = (spy_ret.reindex(ser20.index) * avg_expo).dropna()
mb = metrics_from_returns(bench.values, TD)
print(f"  VRP timing: Sharpe {m20['sharpe']:.3f} | CAGR {m20['cagr']:.1%} | MaxDD {m20['max_dd']:.1%} | avg expo {avg_expo:.0%}")
print(f"  matched-beta bench: Sharpe {mb['sharpe']:.3f} — "
      f"{'BEATS beta' if m20['sharpe'] > mb['sharpe'] + 0.1 else 'does NOT beat beta -> bench'}")
if m20["sharpe"] > mb["sharpe"] + 0.1:
    judge("vrp_timing", ser20)

# record any candidates that passed portfolio-contribution
for nm, ser, m, wp, paper in [
    ("book_gkm_volume", ser17, m17, w17, "Gervais-Kaniel-Mingelgrin 2001 JF (#36)"),
    ("book_tom_spy", ser18, m18, w18, "Etula et al. 2020 RFS (#79)"),
    ("book_overnight_share", ser19, m19, w19, "Lou-Polk-Skouras 2019 JFE (#33)"),
]:
    if wp is not None:
        record_performance(name=nm, dates=ser.dropna().index, returns=ser.dropna().values,
            params={}, data_period=f"{ser.index.min().date()}..{ser.index.max().date()}",
            periods_per_year=TD, extra={"cycle": "Cycle 10", "paper": paper})
        record_improvement(f"NEW BOOK passes portfolio-contribution: {nm}", paper,
            anchor_m, wp, [nm], "adoption via Phase-2 incubation, human-gated")
        print(f"recorded {nm}")
