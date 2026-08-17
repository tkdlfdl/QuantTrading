"""
Queue #47 — S&P announcement dates + the announcement->effective front-run.

DATA: press.spglobal.com index-news listing (s=2429) per year with &l=300:
titles + dates of "... Set to Join S&P 500 ..." releases. Cached to
data/cache/sp500_announcements.csv.

MATCH: each add event in sp500_changes.csv carries the security NAME (from
the pinned changes table); an announcement matches if its title contains the
name's distinctive prefix and its date falls in [effective-30d, effective-1d].

TEST (the only surviving index-rebal window): buy at the close of the first
trading day AFTER the announcement, exit at the effective-date close.
Cohorts: all matched adds; dual-index subset (NDX member within 1yr of event
per ndx_membership.parquet). Costs 0.1%/side. Full gates.
"""
from __future__ import annotations
import sys, warnings, time, re
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import requests
from pathlib import Path

from tools.metrics import metrics_from_returns
from tools.significance import sharpe_delta_test
from live import engine as E, config as C

TD, TC, END = 252, 0.001, "2026-07-08"
t0 = time.time()
HDR = {"User-Agent": "Mozilla/5.0"}
CACHE = Path("data/cache/sp500_announcements.csv")

# ══════════ 1. announcement titles per year ══════════
if CACHE.exists():
    ann = pd.read_csv(CACHE, parse_dates=["date"])
    print(f"cache hit: {len(ann)} announcements")
else:
    # flat archive pager: l=100 per page, o = offset (l>100 is capped;
    # year filter ignores l — both discovered by probing)
    rows = []
    o = 0
    while o < 6000:
        try:
            html = requests.get(
                f"https://press.spglobal.com/index.php?s=2429&l=100&o={o}",
                headers=HDR, timeout=60).text
        except Exception as e:
            print(f"  o={o}: fetch failed ({e}); stopping pager")
            break
        items = re.findall(
            r'wd_date">(\w{3} \d{1,2}, \d{4})<.{0,400}?wd_title"><a[^>]*>'
            r'([^<>]{15,300})</a>', html, flags=re.S)
        if not items:
            break
        for d_, title in items:
            if "S&P 500" in title and ("Set to Join" in title
                                       or "Join S&P 500" in title
                                       or "Replace" in title):
                rows.append((pd.to_datetime(d_), title.strip()))
        print(f"  o={o}: {len(items)} items, through {items[-1][0]} "
              f"({len(rows)} S&P-500 releases so far)", flush=True)
        if len(items) < 100:
            break
        o += 100
        time.sleep(0.5)
    ann = pd.DataFrame(rows, columns=["date", "title"]).drop_duplicates()
    if len(ann) < 100:
        print(f"VALIDATION FAILED ({len(ann)} releases <100) -> DATA-BLOCKED")
        sys.exit(1)
    ann.to_csv(CACHE, index=False)
    print(f"cached {len(ann)} announcements  ({time.time()-t0:.0f}s)")

# ══════════ 2. match to add events ══════════
import io
sp = pd.read_csv("data/cache/sp500_changes.csv", parse_dates=["date"])
# need security NAMES: re-pull the pinned changes table for name mapping
SP_OLDID = 1332941675
html = requests.get("https://en.wikipedia.org/w/index.php"
                    f"?title=List_of_S%26P_500_companies&oldid={SP_OLDID}",
                    headers=HDR, timeout=30).text
name_map = {}
for tb in pd.read_html(io.StringIO(html)):
    cols = ["|".join(map(str, c)) if isinstance(c, tuple) else str(c)
            for c in tb.columns]
    tb.columns = cols
    if any("added" in c.lower() and "ticker" in c.lower() for c in cols):
        tcol = [c for c in cols if "added" in c.lower() and "ticker" in c.lower()][0]
        scol = [c for c in cols if "added" in c.lower() and "security" in c.lower()][0]
        for _, r in tb.iterrows():
            t_, s_ = str(r[tcol]).strip(), str(r[scol]).strip()
            if t_ and t_.lower() != "nan" and s_ and s_.lower() != "nan":
                name_map[t_] = s_
        break

STOP = {"the", "inc", "inc.", "corp", "corp.", "company", "co", "co.",
        "group", "holdings", "technologies", "&"}
def prefix(name: str) -> str:
    words = [w for w in re.split(r"[\s,]+", name) if w.lower() not in STOP]
    return " ".join(words[:2]) if words else name

adds = sp[sp["action"] == "add"].copy()
adds["name"] = adds["ticker"].map(name_map)
adds = adds.dropna(subset=["name"])
matched = []
for _, r in adds.iterrows():
    pfx = prefix(r["name"])
    if len(pfx) < 3:
        continue
    win = ann[(ann["date"] >= r["date"] - pd.Timedelta(days=30))
              & (ann["date"] < r["date"])]
    hit = win[win["title"].str.contains(re.escape(pfx), case=False, na=False)]
    if len(hit):
        matched.append((r["ticker"], r["date"], hit["date"].max()))
mt = pd.DataFrame(matched, columns=["ticker", "eff", "ann"])
mt["lag"] = (mt["eff"] - mt["ann"]).dt.days
print(f"matched add events: {len(mt)}/{len(adds)} | "
      f"median announce->effective lag {mt['lag'].median():.0f}d  "
      f"({time.time()-t0:.0f}s)")
if len(mt) < 60:
    print("MATCH RATE TOO LOW (<60 events) -> report data quality, no test")
    sys.exit(0)

# ══════════ 3. front-run stream ══════════
daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)
stocks = [c for c in daily.columns if not c.startswith("^")]
px = daily[stocks].ffill(limit=5)
idx = px.index

ndx = pd.read_parquet("data/cache/ndx_membership.parquet")
ndx_by_year = {y: set(g["ticker"]) for y, g in ndx.groupby("year")}

def in_ndx(ticker: str, when: pd.Timestamp) -> bool:
    ys = [y for y in ndx_by_year if abs(y - when.year) <= 1]
    return any(ticker in ndx_by_year[y] for y in ys)

def stream(sub: pd.DataFrame) -> pd.Series:
    P = pd.DataFrame(0.0, index=idx,
                     columns=sorted(set(sub["ticker"]) & set(px.columns)))
    used = 0
    for _, r in sub.iterrows():
        s_ = r["ticker"]
        if s_ not in P.columns:
            continue
        a_ = idx.searchsorted(r["ann"]) + 1     # first close AFTER announcement
        e_ = idx.searchsorted(r["eff"])
        if e_ <= a_ or e_ >= len(idx):
            continue
        if px[s_].iloc[a_:e_ + 1].notna().sum() < 2:
            continue
        P.iloc[a_:e_ + 1, P.columns.get_loc(s_)] = 1.0
        used += 1
    ret_ = px[P.columns].pct_change().clip(-0.5, 0.5)
    w = P.div(P.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    s = ((ret_.fillna(0) * w.shift(1)).sum(axis=1)
         - w.diff().abs().sum(axis=1).fillna(0) * TC)
    print(f"    events used {used}; in-market {float((P.sum(axis=1)>0).mean()):.0%}")
    return s.dropna()


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
        "DU": load("book_du_live_spec")}


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

mt["dual"] = [in_ndx(r.ticker, r.ann) for r in mt.itertuples()]
for label, sub in (("ALL matched adds", mt), ("dual-index (in NDX)", mt[mt["dual"]])):
    print(f"  -- {label} ({len(sub)} events):")
    if len(sub) < 10:
        print("    too few events, skip")
        continue
    s = stream(sub)
    s = s[s.index <= END]
    first = s.ne(0).idxmax()
    s = s[s.index >= first]
    ms = metrics_from_returns(s.values, TD)
    rec = s[s.index >= "2015-01-01"]
    mr = metrics_from_returns(rec.values, TD)
    print(f"    standalone {s.index.min().date()}..: Sharpe {ms['sharpe']:.2f} "
          f"CAGR {ms['cagr']:.1%} MaxDD {ms['max_dd']:.0%} | 2015+ {mr['sharpe']:.2f}")
    ser = build_port({**FULL, "N": s},
                     ["A", "C", "D", "F", "G", "X", "DU", "N"],
                     {"D": 2.0, "G": .25, "X": .25, "DU": .25, "N": .25})
    r = sharpe_delta_test(ser, full_stack)
    m = metrics_from_returns(ser.values, TD)
    sig = "  <-- ADDS ON FULL STACK" if r["significant_p10"] and r["delta"] > 0 else ""
    print(f"    FULL-STACK +N @0.25: {m['sharpe']:.3f} (delta {r['delta']:+.3f}, "
          f"p {r['p_one_sided']:.3f}){sig}")
print(f"total {time.time()-t0:.0f}s")
