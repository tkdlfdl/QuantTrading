"""
Cycle 21 — acquire the three registry-named data assets and test what they unlock.

A. FOMC calendar (scraped from federalreserve.gov historical pages, validated
   ~8 meetings/yr) -> #37 pre-FOMC drift (Lucca-Moench, daily-close version:
   long SPY close(T-1) -> close(T) on announcement days).
B. VIX term structure (free proxies ^VIX3M, ^VIX9D via yfinance) -> convexity
   v2: long UVXY ONLY in backwardation (VIX/VIX3M > 1 = positive roll yield) —
   fixes the carry problem that killed convexity v1.
C. Shares outstanding (yfinance get_shares_full, coverage-checked) ->
   Medhat-Schmeling with TRUE turnover where coverage allows + honest caveat.

All strategies face full gates: standalone, corr, champ delta @0.25, LW p.
"""
from __future__ import annotations
import sys, warnings, re, time, json
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tools.metrics import metrics_from_returns
from tools.significance import sharpe_delta_test
from live import engine as E, config as C

TD, TC, END = 252, 0.001, "2026-07-08"
t0 = time.time()

def load(n):
    df = pd.read_csv(f"strategies/performance/{n}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)
champ = load("portfolio_champ_d14")
d8, d14 = load("book_d_retest"), load("book_d14")
ju = d8.index.union(d14.index)
dblend = 0.5*d8.reindex(ju).fillna(0) + 0.5*d14.reindex(ju).fillna(0)
BASE = {"A": load("book_a_retest"), "C": load("book_c_overlap_cap"), "D": dblend, "F": load("book_f_retest")}
worst = champ.nsmallest(int(len(champ)*0.05)).index

def gates(name, s):
    ms = metrics_from_returns(s.dropna().values, TD)
    s19 = s.reindex(champ.index).fillna(0)
    cc = float(np.corrcoef(s19, champ)[0,1])
    sd = float(s.reindex(worst).fillna(0).mean())
    sb, ss = C.ALLOC_BOOKS, dict(C.ALLOC_SHARES)
    C.ALLOC_BOOKS = ["A","C","D","F","N"]; C.ALLOC_SHARES = {"D": 2.0, "N": 0.25}
    try:
        books = dict(BASE); books["N"] = s
        R = pd.DataFrame(books); R = R[(R.index >= "2019-01-02") & (R.index <= END)].fillna(0.0)
        ser = E.ivol_voltgt(R).dropna()
    finally:
        C.ALLOC_BOOKS, C.ALLOC_SHARES = sb, ss
    m = metrics_from_returns(ser.values, TD)
    r = sharpe_delta_test(ser, champ)
    print(f"  {name}: standalone {ms['sharpe']:.2f}/{ms['max_dd']:.0%} | corr {cc:+.2f} | stress {sd:+.2%}/d")
    print(f"    champ+N @0.25: {m['sharpe']:.3f} (delta {r['delta']:+.3f}, p {r['p_one_sided']:.3f})"
          f"{'  <-- SIGNIFICANT' if r['significant_p10'] else ''}")

daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)
spy = daily["SPY"].dropna(); spy_r = spy.pct_change()

# ══════════ A. FOMC calendar ══════════
print("=== A. FOMC calendar scrape ===")
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
dates = []
hdr = {"User-Agent": "Mozilla/5.0 (research)"}
for yr in range(1997, 2019):
    try:
        html = requests.get(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{yr}.htm",
                            headers=hdr, timeout=20).text
        for mo, dd in re.findall(rf"({MONTHS})\s+(\d+(?:-\d+)?)(?:,)?\s*(?:{yr})?", html):
            day = dd.split("-")[-1]
            try:
                dates.append(pd.Timestamp(f"{mo} {day}, {yr}"))
            except Exception:
                pass
    except Exception as e:
        print(f"  {yr}: fetch failed ({e})")
try:
    html = requests.get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
                        headers=hdr, timeout=20).text
    for yr in range(2019, 2027):
        for mo, dd in re.findall(rf"({MONTHS})\s+(\d+(?:-\d+)?)[^\d]{{0,20}}{yr}", html):
            day = dd.split("-")[-1]
            try:
                dates.append(pd.Timestamp(f"{mo} {day}, {yr}"))
            except Exception:
                pass
except Exception as e:
    print(f"  recent calendar fetch failed ({e})")
fomc = pd.Series(sorted(set(dates)))
per_yr = fomc.groupby(fomc.dt.year).size()
ok_years = per_yr[(per_yr >= 6) & (per_yr <= 14)]
print(f"  scraped {len(fomc)} candidate dates; years with plausible counts (6-14): "
      f"{len(ok_years)}/{len(per_yr)}")
if len(ok_years) >= 25:
    fomc[fomc.dt.year.isin(ok_years.index)].to_frame("date").to_csv("data/cache/fomc_dates.csv", index=False)
    print("  cached data/cache/fomc_dates.csv")
    fdates = set(pd.to_datetime(pd.read_csv("data/cache/fomc_dates.csv")["date"]).dt.normalize())
    # pre-FOMC drift: hold SPY close(T-1)->close(T) on announcement day T
    is_ann = pd.Series([d.normalize() in fdates for d in spy.index], index=spy.index)
    pos = is_ann.astype(float)                      # in market on announcement day
    s37 = (spy_r * pos - pos.diff().abs().fillna(0)*TC).dropna()
    s37 = s37[s37.index >= "1997-06-01"]
    print(f"  event days in sample: {int(pos.sum())}")
    gates("#37 pre-FOMC drift", s37)
else:
    print("  VALIDATION FAILED — too few plausible years; #37 stays data-blocked (no approximation)")

# ══════════ B. VIX term structure ══════════
print(f"\n=== B. VIX term structure ({time.time()-t0:.0f}s) ===")
vts = yf.download(["^VIX", "^VIX3M", "^VIX9D"], start="2002-01-01", interval="1d",
                  auto_adjust=False, progress=False)["Close"]
vts.index = pd.to_datetime(vts.index).tz_localize(None)
vts.to_parquet("data/cache/vix_term_structure.parquet")
cov3m = vts["^VIX3M"].dropna()
print(f"  ^VIX3M coverage: {cov3m.index.min().date()}..{cov3m.index.max().date()}")
uvxy = daily["UVXY"].dropna(); uvxy_r = uvxy.pct_change()
ratio = (vts["^VIX"]/vts["^VIX3M"]).reindex(daily.index).ffill()
backw = (ratio > 1.0).shift(1).fillna(False)
# carry check by regime
uj = uvxy_r.dropna().index
print(f"  UVXY drift | backwardation: {uvxy_r[backw.reindex(uj).fillna(False)].mean():+.2%}/d "
      f"({int(backw.reindex(uj).fillna(False).sum())} days) | contango: "
      f"{uvxy_r[~backw.reindex(uj).fillna(False).astype(bool)].mean():+.2%}/d")
pos_b = backw.astype(float).reindex(uj).fillna(0)
s_cx = (uvxy_r*pos_b - pos_b.diff().abs().fillna(0)*0.0025).dropna()
s_cx = s_cx[s_cx.index >= "2012-01-01"]
gates("convexity v2 (backwardation-gated UVXY)", s_cx)

# ══════════ C. shares outstanding ══════════
print(f"\n=== C. shares outstanding coverage probe ({time.time()-t0:.0f}s) ===")
hc = pd.read_parquet("data/cache/merged_hourly_close.parquet")
probe = sorted(set(hc.columns) - {"SPY"})[:40]
from concurrent.futures import ThreadPoolExecutor
def sh(t):
    try:
        s = yf.Ticker(t).get_shares_full(start="2018-01-01")
        return t, (str(s.index.min().date()) if s is not None and len(s) else None)
    except Exception:
        return t, None
with ThreadPoolExecutor(8) as ex:
    res = dict(ex.map(sh, probe))
have = {k: v for k, v in res.items() if v}
starts = sorted(have.values())
print(f"  probe 40 tickers: {len(have)} have share history; earliest starts: {starts[:3]}, "
      f"median {starts[len(starts)//2] if starts else 'n/a'}")
print("  -> verdict on MS retest feasibility printed above; full fetch only if coverage supports it")
print(f"\ntotal {time.time()-t0:.0f}s")
