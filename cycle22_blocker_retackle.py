"""
Cycle 22 — re-tackle the three registry-named data blockers.

A. FOMC calendar (#37): scrape v2 — parse per-year historical pages with
   meeting-anchored regex (last date of each meeting span = announcement day);
   validate 6-12 meetings/yr before caching. Unlocks pre-FOMC drift test.
B. VIX term structure: the UNUSED angle. UVXY-long is closed (bleeds in all
   regimes, #124). Open question: does front-end inversion (^VIX9D > ^VIX)
   add value as an EXTRA de-risk trigger on the champion overlay
   (tighten vol target 15% -> 10% on inversion days)?
C. Shares outstanding: extend the quiet-capitulation harvest — ordering was
   only validated on the D8 sleeve; test the same ordering on the D14 sleeve
   (full-quiet blend) with anchor discipline.

All vs the quiet-champion base (quiet-D8 + D14 blend, 2 shares, A/C/D/F).
"""
from __future__ import annotations
import sys, warnings, re, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import requests

import strategies.contrarian_bubble_hourly as CB
from tools.metrics import metrics_from_returns
from tools.significance import sharpe_delta_test
from tools.record import record_performance
from live import engine as E, config as C
from data.db.client import get_conn

TD, TC, END = 252, 0.001, "2026-07-08"
t0 = time.time()

def load(nm):
    df = pd.read_csv(f"strategies/performance/{nm}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)

champ = load("portfolio_champ_d14")
daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)
spy = daily["SPY"].dropna()
spy_r = spy.pct_change()

# ================= A. FOMC scrape v2 =================
print("=== A. FOMC scrape v2 (meeting-anchored parsing) ===")
MONTHS = ("January|February|March|April|May|June|July|August|September|"
          "October|November|December")
hdr = {"User-Agent": "Mozilla/5.0"}
dates = []
for yr in range(1997, 2021):
    try:
        html = requests.get(
            f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{yr}.htm",
            headers=hdr, timeout=20).text
        # headings like "January 28-29 Meeting" / "March 15 (unscheduled)" etc.
        # anchor on the word Meeting within 40 chars after the date span
        pat = rf"(({MONTHS})\s+\d+(?:\s*[-–/]\s*(?:({MONTHS})\s+)?\d+)?)[^<]{{0,40}}[Mm]eeting"
        for grp in re.findall(pat, html):
            span = grp[0]
            last = re.findall(rf"({MONTHS})\s+(\d+)", span)
            mo, dd = last[-1]
            if len(last) == 1 and ("-" in span or "–" in span or "/" in span):
                # "January 28-29" -> month from span start, day = trailing number
                tail = re.findall(r"(\d+)\s*$", span)
                if tail:
                    dd = tail[-1]
            try:
                dates.append(pd.Timestamp(f"{mo} {dd}, {yr}"))
            except Exception:
                pass
    except Exception as e:
        print(f"  {yr}: fetch failed ({e})")

try:
    html = requests.get(
        "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        headers=hdr, timeout=20).text
    secs = re.split(r"(20\d\d)\s+FOMC\s+[Mm]eetings", html)
    for k in range(1, len(secs) - 1, 2):
        yr = int(secs[k]); body = secs[k + 1]
        found = 0
        for mo, span in re.findall(rf"({MONTHS})[\s<>/a-z]*?(\d+(?:\s*[-–]\s*\d+)?)", body):
            if found >= 10:
                break
            dd = re.findall(r"\d+", span)[-1]
            try:
                dates.append(pd.Timestamp(f"{mo} {dd}, {yr}"))
                found += 1
            except Exception:
                pass
except Exception as e:
    print(f"  recent page failed ({e})")

fomc = pd.Series(sorted(set(dates)))
if len(fomc):
    per = fomc.groupby(fomc.dt.year).size()
    okyrs = per[(per >= 6) & (per <= 12)]
    print(f"  {len(fomc)} unique dates | years plausible (6-12/yr): {len(okyrs)}/{len(per)}")
    print("  per-year counts:", dict(per))
    if len(okyrs) >= 22:
        f2 = fomc[fomc.dt.year.isin(okyrs.index)]
        f2.to_frame("date").to_csv("data/cache/fomc_dates.csv", index=False)
        print(f"  CACHED data/cache/fomc_dates.csv ({len(f2)} dates)")
        fd = set(pd.to_datetime(f2).dt.normalize())
        is_ann = pd.Series([d.normalize() in fd for d in spy.index],
                           index=spy.index).astype(float)
        s37 = (spy_r * is_ann - is_ann.diff().abs().fillna(0) * TC).dropna()
        s37 = s37[s37.index >= "1997-06-01"]
        m37 = metrics_from_returns(s37.values, TD)
        ann_days = s37[is_ann.reindex(s37.index).astype(bool)]
        print(f"  #37 announcement-day SPY: Sharpe {m37['sharpe']:.2f} | "
              f"CAGR {m37['cagr']:.1%} | MaxDD {m37['max_dd']:.0%} | "
              f"{int(is_ann.sum())} event days | avg event-day {ann_days.mean():+.2%}")
    else:
        print("  VALIDATION FAILED AGAIN — #37 stays data-blocked (no approximation)")
else:
    print("  no dates extracted — #37 stays blocked")

# ================= build quiet-champion base =================
print(f"\n=== building quiet-champion base ({time.time()-t0:.0f}s) ===")
d8q, d14 = load("book_d8_quiet"), load("book_d14")
ju = d8q.index.union(d14.index)
dbl = 0.5 * d8q.reindex(ju).fillna(0) + 0.5 * d14.reindex(ju).fillna(0)
BASE = {"A": load("book_a_retest"), "C": load("book_c_overlap_cap"),
        "D": dbl, "F": load("book_f_retest")}

def build_port(books, alloc, shares):
    sb, ss = C.ALLOC_BOOKS, dict(C.ALLOC_SHARES)
    C.ALLOC_BOOKS = alloc; C.ALLOC_SHARES = shares
    try:
        R = pd.DataFrame(books)
        R = R[(R.index >= "2019-01-02") & (R.index <= END)].fillna(0.0)
        return E.ivol_voltgt(R).dropna()
    finally:
        C.ALLOC_BOOKS, C.ALLOC_SHARES = sb, ss

base_ser = build_port(BASE, ["A", "C", "D", "F"], {"D": 2.0})
mb = metrics_from_returns(base_ser.values, TD)
print(f"  quiet-champion base: Sharpe {mb['sharpe']:.3f} | MaxDD {mb['max_dd']:.1%}")

# ================= B. VIX 9D/30D inversion extra de-risk =================
print(f"\n=== B. VIX 9D inversion as extra overlay de-risk ({time.time()-t0:.0f}s) ===")
vts = pd.read_parquet("data/cache/vix_term_structure.parquet")
inv = (vts["^VIX9D"] > vts["^VIX"]).shift(1)
cov = vts["^VIX9D"].dropna()
print(f"  ^VIX9D coverage {cov.index.min().date()}.. | "
      f"inversion days {int(inv.fillna(False).sum())}")

# rebuild the pre-overlay stream manually (ivol + panic, no vol target),
# then compare 15%-flat target vs 15%/10% inversion-tightened target.
panic = ((spy.pct_change(504) < 0) &
         (spy_r.rolling(63).std() * np.sqrt(TD) >
          (spy_r.rolling(63).std() * np.sqrt(TD)).rolling(756).quantile(0.8)))
R = pd.DataFrame(BASE)
R = R[(R.index >= "2019-01-02") & (R.index <= END)].fillna(0.0)
panic = panic.shift(1).reindex(R.index).fillna(False)
cols = list(R.columns)
iA, iF, iD = cols.index("A"), cols.index("F"), cols.index("D")
port = np.zeros(len(R)); w = None
for t in range(len(R)):
    if w is None or t % 21 == 0:
        hist = R.iloc[max(0, t - 60):t]
        v = hist.std() * np.sqrt(TD)
        iv = np.array([(2.0 if c == "D" else 1.0) / v[c]
                       if np.isfinite(v[c]) and v[c] > 1e-9 else 0.0 for c in cols])
        w = iv / iv.sum() if iv.sum() else np.ones(len(cols)) / len(cols)
    wt = w.copy()
    if bool(panic.iloc[t]):
        freed = 0.5 * (wt[iA] + wt[iF])
        wt[iA] *= 0.5; wt[iF] *= 0.5; wt[iD] += freed
    port[t] = float(wt @ R.iloc[t].values)
pre = pd.Series(port, index=R.index)

rv = pre.rolling(20).std().shift(1) * np.sqrt(TD)
flat = (pre * (0.15 / rv).clip(upper=1.0).fillna(1.0)).dropna()
m_flat = metrics_from_returns(flat.values, TD)
anchor_delta = m_flat["sharpe"] - mb["sharpe"]
print(f"  ANCHOR (manual rebuild, flat 15%): {m_flat['sharpe']:.3f} "
      f"(vs engine {mb['sharpe']:.3f}, diff {anchor_delta:+.3f})")
if abs(anchor_delta) > 0.05:
    print("  ANCHOR FAILED — VOID experiment B")
else:
    inv_al = inv.reindex(R.index).fillna(False).astype(bool)
    for tightened in (0.10, 0.08, 0.12):
        tgt = pd.Series(np.where(inv_al.values, tightened, 0.15), index=R.index)
        ser_i = (pre * (tgt / rv).clip(upper=1.0).fillna(1.0)).dropna()
        r_i = sharpe_delta_test(ser_i, flat)
        m_i = metrics_from_returns(ser_i.values, TD)
        sig = "  <-- SIGNIFICANT" if r_i["significant_p10"] else ""
        print(f"  inversion->target {tightened:.0%}: Sharpe {m_i['sharpe']:.3f} "
              f"MaxDD {m_i['max_dd']:.1%} (delta {r_i['delta']:+.3f}, "
              f"p {r_i['p_one_sided']:.3f}){sig}")

# ================= C. quiet ordering on D14 =================
print(f"\n=== C. quiet ordering extended to D14 ({time.time()-t0:.0f}s) ===")
hc = pd.read_parquet("data/cache/merged_hourly_close.parquet")
hc.index = pd.to_datetime(hc.index)
ho = pd.read_parquet("data/cache/merged_hourly_open.parquet")
ho.index = pd.to_datetime(ho.index)
common = sorted((set(ho.columns) & set(hc.columns)) - {"SPY"})
hcc, hoo = hc[common], ho[common]
B = CB._bubble_matrix(hcc, 104)

# ANCHOR: plain B through patched path must reproduce official D14 baseline
_orig = CB._bubble_matrix
CB._bubble_matrix = lambda close, m_: B
anc, _, _ = CB.run_contrarian_bubble_hourly(
    hoo, hcc, ma_window_grid=[104], buy_threshold_grid=[0.8],
    hold_hours_grid=[14], top_n_grid=[20])
CB._bubble_matrix = _orig
anc = anc.dropna(); anc = anc[anc.index <= END]
m_anc = metrics_from_returns(anc.values, TD)
d14_off = d14[d14.index <= END]
m_off = metrics_from_returns(d14_off.dropna().values, TD)
print(f"  ANCHOR D14: patched {m_anc['sharpe']:.3f} vs official {m_off['sharpe']:.3f} "
      f"(diff {m_anc['sharpe']-m_off['sharpe']:+.3f})")
if abs(m_anc["sharpe"] - m_off["sharpe"]) > 0.05:
    print("  ANCHOR FAILED — VOID experiment C")
else:
    con = get_conn()
    vol_d = con.execute(
        "SELECT ts, symbol, volume FROM ohlcv WHERE interval='1d' "
        "AND ts >= '2018-06-01'").df()
    vol_d["ts"] = pd.to_datetime(vol_d["ts"])
    V = vol_d.pivot_table(index="ts", columns="symbol", values="volume").sort_index()
    shares = pd.read_parquet("data/cache/shares_outstanding.parquet")
    TO = V / shares.reindex(V.index).ffill()
    z = (TO.rolling(5).mean() / TO.rolling(60).mean()).shift(1)
    zl = np.log(z.clip(0.1, 10.0))
    f = zl.reindex(columns=common).reindex(hcc.index.normalize()).values
    rank = np.clip(np.nan_to_num(f / (np.nanstd(f) + 1e-9), nan=0.0), -1, 1)
    S = np.where(B < -0.8, -0.81 + rank * 0.09, 0.0).astype(np.float32)
    CB._bubble_matrix = lambda close, m_: S
    d14q, _, _ = CB.run_contrarian_bubble_hourly(
        hoo, hcc, ma_window_grid=[104], buy_threshold_grid=[0.8],
        hold_hours_grid=[14], top_n_grid=[20])
    CB._bubble_matrix = _orig
    d14q = d14q.dropna(); d14q = d14q[d14q.index <= END]
    m_q14 = metrics_from_returns(d14q.values, TD)
    print(f"  quiet-D14 standalone: Sharpe {m_q14['sharpe']:.3f} "
          f"MaxDD {m_q14['max_dd']:.1%} (official D14 {m_off['sharpe']:.3f})")
    ju2 = d8q.index.union(d14q.index)
    dbl_q = 0.5 * d8q.reindex(ju2).fillna(0) + 0.5 * d14q.reindex(ju2).fillna(0)
    ser_q = build_port({**BASE, "D": dbl_q}, ["A", "C", "D", "F"], {"D": 2.0})
    r_q = sharpe_delta_test(ser_q, base_ser)
    m_q = metrics_from_returns(ser_q.values, TD)
    sig = "  <-- SIGNIFICANT" if r_q["significant_p10"] else ""
    print(f"  full-quiet champ: Sharpe {m_q['sharpe']:.3f} MaxDD {m_q['max_dd']:.1%} "
          f"(delta vs quiet-D8-only {r_q['delta']:+.3f}, p {r_q['p_one_sided']:.3f}){sig}")
    if r_q["significant_p10"] and r_q["delta"] > 0:
        record_performance(
            name="book_d14_quiet", dates=d14q.index, returns=d14q.values,
            params={"ordering": "quiet 5/60", "hold": 14, "top_n": 20, "ma": 104},
            data_period=f"{d14q.index.min().date()}..{END}",
            periods_per_year=TD,
            extra={"note": "cycle22 blocker re-tackle C: quiet ordering on D14 sleeve"})
        print("  recorded book_d14_quiet")

print(f"\ntotal {time.time()-t0:.0f}s")
