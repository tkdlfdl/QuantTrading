"""
Cycle 25 — index-rebalance strategies (S&P 500 membership changes).

DATA: Wikipedia "List of S&P 500 companies" -> "Selected changes" table
(date, added ticker, removed ticker). Cached to data/cache/sp500_changes.csv.
Validation gate: >= 15 usable years with >= 10 events/yr, else DATA-BLOCKED.

STRATEGIES (both sides of the same event, pre-registered):
 (i)  DELETION REBOUND: buy each removed name at the close of its effective
      date, hold 60 trading days, equal-weight across open events.
      Literature: Chen-Noronha-Singal; deletions overshoot on forced selling
      and revert — the side Greenwood-Sammon say SURVIVED post-2010.
 (ii) ADD EFFECT: buy each added name at effective-date close, hold 20d.
      Expected dead post-2010 (#121) — serves as the control.

SURVIVORSHIP CAVEAT (quantified, printed): deleted names that later delisted
have no yfinance history -> the tested deletion sample tilts toward
survivors. Coverage fraction is reported; if < 50% the verdict is capped at
"suggestive" regardless of numbers.

Gates: standalone, corr(champ), stress, full-stack marginal @0.25, LW p.
Costs 0.1%/side. No look-ahead: entries at effective-date CLOSE (the date in
the table), i.e. after the change is public.
"""
from __future__ import annotations
import sys, warnings, time, io
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import requests

from tools.metrics import metrics_from_returns
from tools.significance import sharpe_delta_test
from live import engine as E, config as C

TD, TC, END = 252, 0.001, "2026-07-08"
t0 = time.time()

# ══════════ 1. scrape + validate changes table ══════════
print("=== S&P 500 changes scrape ===")
from pathlib import Path
CACHE = Path("data/cache/sp500_changes.csv")
# The LIVE page dropped the changes table (2026); use the pinned Jan-2026
# revision, which still carries it (388 events, Effective Date + Reason).
SP_OLDID = 1332941675
if CACHE.exists():
    ch = pd.read_csv(CACHE, parse_dates=["date"])
    print(f"  cache hit: {len(ch)} change rows")
else:
    url = ("https://en.wikipedia.org/w/index.php"
           f"?title=List_of_S%26P_500_companies&oldid={SP_OLDID}")
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    tables = pd.read_html(io.StringIO(html))
    cand = None
    for tb in tables:
        cols = ["|".join(map(str, c)) if isinstance(c, tuple) else str(c)
                for c in tb.columns]
        joined = " ".join(cols).lower()
        if "added" in joined and "removed" in joined and "date" in joined:
            cand = tb
            cand.columns = cols
            break
    if cand is None:
        print("  changes table not found in pinned revision -> DATA-BLOCKED")
        sys.exit(1)
    date_c = [c for c in cand.columns if "date" in c.lower()][0]
    add_c = [c for c in cand.columns if "added" in c.lower() and "ticker" in c.lower()]
    rem_c = [c for c in cand.columns if "removed" in c.lower() and "ticker" in c.lower()]
    rows = []
    for _, r in cand.iterrows():
        try:
            d_ = pd.to_datetime(str(r[date_c]))
        except Exception:
            continue
        a_ = str(r[add_c[0]]).strip() if add_c else ""
        x_ = str(r[rem_c[0]]).strip() if rem_c else ""
        if a_ and a_.lower() != "nan":
            rows.append((d_, "add", a_))
        if x_ and x_.lower() != "nan":
            rows.append((d_, "remove", x_))
    ch = pd.DataFrame(rows, columns=["date", "action", "ticker"]).dropna()
    ch = ch[ch["ticker"].str.len() <= 6]
    ch.to_csv(CACHE, index=False)
    print(f"  scraped+cached {len(ch)} change rows (revision {SP_OLDID})")

per_yr = ch.groupby([ch["date"].dt.year, "action"]).size().unstack(fill_value=0)
usable = per_yr[(per_yr.get("add", 0) + per_yr.get("remove", 0)) >= 10]
print(f"  years with >=10 events: {len(usable)} "
      f"({int(usable.index.min())}..{int(usable.index.max())})")
if len(usable) < 15:
    print("  VALIDATION FAILED -> DATA-BLOCKED (no approximation)")
    sys.exit(1)
ch = ch[ch["date"].dt.year.isin(usable.index)]

# ══════════ 2. price coverage for event tickers ══════════
print(f"\n=== price coverage ({time.time()-t0:.0f}s) ===")
daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)
have = set(daily.columns)
ev_tickers = sorted(set(ch["ticker"]) - have)
import yfinance as yf
fetched = pd.DataFrame()
if ev_tickers:
    fetched = yf.download(ev_tickers, start="1998-01-01", interval="1d",
                          auto_adjust=True, progress=False, threads=True)["Close"]
    if isinstance(fetched, pd.Series):
        fetched = fetched.to_frame()
    fetched.index = pd.to_datetime(fetched.index).tz_localize(None)
px_all = pd.concat([daily, fetched], axis=1)
px_all = px_all.loc[:, ~px_all.columns.duplicated()]

rem = ch[ch["action"] == "remove"]
add = ch[ch["action"] == "add"]
rem_cov = rem["ticker"].isin(
    [c for c in px_all.columns if px_all[c].notna().sum() > 100]).mean()
add_cov = add["ticker"].isin(
    [c for c in px_all.columns if px_all[c].notna().sum() > 100]).mean()
print(f"  removals with usable prices: {rem_cov:.0%} of {len(rem)} | "
      f"adds: {add_cov:.0%} of {len(add)}")
capped = rem_cov < 0.5

# ══════════ 3. build event streams ══════════
def event_stream(events: pd.DataFrame, hold_days: int) -> pd.Series:
    idx = px_all.index
    pos = {}
    used = 0
    for _, r in events.iterrows():
        s_ = r["ticker"]
        if s_ not in px_all.columns:
            continue
        col = px_all[s_]
        loc = idx.searchsorted(r["date"])
        if loc >= len(idx) - 2:
            continue
        seg = col.iloc[loc:loc + hold_days + 1]
        if seg.notna().sum() < min(hold_days // 2, 10) or seg.iloc[0] != seg.iloc[0]:
            continue
        pos.setdefault(s_, []).append((loc, min(loc + hold_days, len(idx) - 1)))
        used += 1
    P = pd.DataFrame(0.0, index=idx, columns=sorted(pos))
    for s_, spans in pos.items():
        for a_, b_ in spans:
            P.iloc[a_:b_ + 1, P.columns.get_loc(s_)] = 1.0
    ret_ = px_all[P.columns].ffill(limit=5).pct_change().clip(-0.5, 0.5)
    w = P.div(P.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    strat = ((ret_.fillna(0) * w.shift(1)).sum(axis=1)
             - w.diff().abs().sum(axis=1).fillna(0) * TC)
    print(f"    events used: {used}; in-market "
          f"{float((P.sum(axis=1) > 0).mean()):.0%} of days")
    return strat.dropna()


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


def judge(name, stream):
    stream = stream[stream.index <= END]
    first = stream.ne(0).idxmax()
    stream = stream[stream.index >= first]
    ms = metrics_from_returns(stream.values, TD)
    s19 = stream.reindex(champ.index).fillna(0)
    cc = float(np.corrcoef(s19, champ)[0, 1])
    sd = float(stream.reindex(worst).fillna(0).mean())
    print(f"  {name}: standalone {stream.index.min().date()}..: "
          f"Sharpe {ms['sharpe']:.2f} CAGR {ms['cagr']:.1%} "
          f"MaxDD {ms['max_dd']:.0%} | corr(champ) {cc:+.2f} | stress {sd:+.2%}/d")
    # recent-decade slice (post-2015): is the effect alive NOW?
    rec = stream[stream.index >= "2015-01-01"]
    mr = metrics_from_returns(rec.values, TD)
    print(f"    2015+: Sharpe {mr['sharpe']:.2f} CAGR {mr['cagr']:.1%}")
    ser = build_port({**FULL, "N": stream},
                     ["A", "C", "D", "F", "G", "X", "DU", "N"],
                     {"D": 2.0, "G": .25, "X": .25, "DU": .25, "N": .25})
    r = sharpe_delta_test(ser, full_stack)
    m = metrics_from_returns(ser.values, TD)
    sig = "  <-- ADDS ON FULL STACK" if r["significant_p10"] and r["delta"] > 0 else ""
    print(f"    FULL-STACK +N @0.25: {m['sharpe']:.3f} (delta {r['delta']:+.3f}, "
          f"p {r['p_one_sided']:.3f}){sig}")


print(f"\n=== (i) deletion rebound (hold 60d) ({time.time()-t0:.0f}s) ===")
s_del = event_stream(rem, 60)
judge("deletion rebound", s_del)
if capped:
    print("    NOTE: coverage <50% -> verdict capped at SUGGESTIVE "
          "(survivor-tilted sample)")

print(f"\n=== (ii) add effect (hold 20d) ({time.time()-t0:.0f}s) ===")
s_add = event_stream(add, 20)
judge("add effect", s_add)

# ══════════ (iii) cross-index migration cohort (user idea) ══════════
# S&P adds that are ALSO Nasdaq-100 members: dual-index names face demand
# from both tracker pools. NDX membership taken from the pinned Jan-2026
# revision of the Nasdaq-100 article (live page's table is gone — the same
# breakage that hit the universe scraper). LIMITATION (stated, not hidden):
# membership is AS OF 2026-01, not as of each add date — this is a cohort
# diagnostic, not a claim about historical NDX membership at event time.
print(f"\n=== (iii) migration cohort: S&P adds x NDX membership "
      f"({time.time()-t0:.0f}s) ===")
ndx = None
try:
    r_ = requests.get("https://en.wikipedia.org/w/api.php", params=dict(
        action="query", titles="Nasdaq-100", prop="revisions", rvlimit=1,
        rvprop="ids", format="json", rvstart="2026-01-15T00:00:00Z"),
        headers={"User-Agent": "Mozilla/5.0"}, timeout=30).json()
    ndx_oldid = list(r_["query"]["pages"].values())[0]["revisions"][0]["revid"]
    html_n = requests.get(
        f"https://en.wikipedia.org/w/index.php?title=Nasdaq-100&oldid={ndx_oldid}",
        headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    for tb in pd.read_html(io.StringIO(html_n)):
        cols = [str(c).lower() for c in tb.columns]
        if any("ticker" in c or "symbol" in c for c in cols) and 80 <= len(tb) <= 120:
            tick_col = tb.columns[[i for i, c in enumerate(cols)
                                   if "ticker" in c or "symbol" in c][0]]
            ndx = set(tb[tick_col].astype(str).str.strip())
            break
except Exception as e:
    print(f"  NDX fetch failed ({e})")
if ndx:
    print(f"  NDX constituents (2026-01 revision): {len(ndx)}")
    add_ndx = add[add["ticker"].isin(ndx)]
    add_oth = add[~add["ticker"].isin(ndx)]
    print(f"  dual-index adds: {len(add_ndx)} | other adds: {len(add_oth)}")
    print("  -- dual-index cohort (hold 20d):")
    judge("adds in NDX", event_stream(add_ndx, 20))
    print("  -- non-NDX cohort (hold 20d):")
    judge("adds not in NDX", event_stream(add_oth, 20))
else:
    print("  NDX membership unavailable -> cohort split DATA-BLOCKED")
print(f"\ntotal {time.time()-t0:.0f}s")
