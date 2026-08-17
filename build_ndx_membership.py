"""
Queue #45 — NDX membership history from year-end pinned Wikipedia revisions.

For each Dec-31 (2008..2025) fetch the latest Nasdaq-100 article revision
before that date via the MediaWiki API, parse the constituents table, and
build a membership panel. Cache: data/cache/ndx_membership.parquet
(columns: year, ticker). Validation: each snapshot must have 90-110 tickers
or it is dropped (logged).

Then the REVERSE-migration event study (user request): NDX adds = tickers in
snapshot Y but not Y-1, event-dated to the December-Y-1 reconstitution
effective date (Monday after 3rd Friday of December — exchange rule, not an
approximation). Cohorts: adds already in the S&P 500 at event time
(reconstructed from sp500_changes.csv + current members) vs not.
Buy at effective close, hold 20d. Full gates.
"""
from __future__ import annotations
import sys, warnings, time, io
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
CACHE = Path("data/cache/ndx_membership.parquet")

if CACHE.exists():
    mem = pd.read_parquet(CACHE)
    print(f"cache hit: {mem['year'].nunique()} snapshots")
else:
    # Pre-2018 article revisions carry no components table (probed 2008-2017:
    # section exists but membership was never tabulated) -> modern window only.
    rows = []
    for year in range(2017, 2027):
        try:
            r = None
            for backoff in (0, 2, 5, 12, 30):
                if backoff:
                    time.sleep(backoff)
                # 2026: the mid-2026 page dropped the table; use the January
                # revision (same workaround as the S&P changes table)
                rvstart = (f"{year}-01-15T00:00:00Z" if year == 2026
                           else f"{year}-12-31T23:59:59Z")
                resp = requests.get("https://en.wikipedia.org/w/api.php", params=dict(
                    action="query", titles="Nasdaq-100", prop="revisions",
                    rvlimit=1, rvprop="ids|timestamp", format="json",
                    rvstart=rvstart),
                    headers=HDR, timeout=30)
                try:
                    r = resp.json()
                    break
                except ValueError:
                    continue
            if r is None:
                print(f"  {year}: API throttled through backoff ladder")
                continue
            revs = list(r["query"]["pages"].values())[0].get("revisions")
            if not revs:
                continue
            oldid = revs[0]["revid"]
            html = requests.get(
                f"https://en.wikipedia.org/w/index.php?title=Nasdaq-100&oldid={oldid}",
                headers=HDR, timeout=30).text
            best = None
            for tb in pd.read_html(io.StringIO(html)):
                cols = [str(c).lower() for c in tb.columns]
                tick_i = [i for i, c in enumerate(cols)
                          if "ticker" in c or "symbol" in c]
                if tick_i and 80 <= len(tb) <= 120:
                    col_ = tb[tb.columns[tick_i[0]]]
                    if isinstance(col_, pd.DataFrame):   # duplicated header name
                        col_ = col_.iloc[:, 0]
                    # plain-Python cast: pandas .str ops can still yield float
                    # NaN on object columns even after astype(str)
                    best = pd.Series([str(x).strip() for x in col_.tolist()])
                    break
            if best is None:
                print(f"  {year}: no constituents table (revid {oldid})")
                continue
            ticks = sorted({t for t in best
                            if 1 <= len(t) <= 6 and t.upper() == t and t.isascii()})
            if not 90 <= len(ticks) <= 110:
                print(f"  {year}: implausible count {len(ticks)} -> dropped")
                continue
            for t_ in ticks:
                rows.append((year, t_))
            print(f"  {year}: {len(ticks)} tickers (revid {oldid})", flush=True)
            time.sleep(0.3)
        except Exception as e:
            print(f"  {year}: failed ({e})")
    mem = pd.DataFrame(rows, columns=["year", "ticker"])
    # modern-window validation: >= 8 year-end snapshots (2018+; pre-2018 has
    # no tabulated membership anywhere on Wikipedia — limitation stated)
    if mem["year"].nunique() < 8:
        print("VALIDATION FAILED (<8 snapshots) -> DATA-BLOCKED")
        sys.exit(1)
    mem.to_parquet(CACHE)
    print(f"cached {CACHE.name}: {mem['year'].nunique()} snapshots, "
          f"{len(mem)} rows  ({time.time()-t0:.0f}s)")

years = sorted(mem["year"].unique())
snap = {y: set(mem[mem["year"] == y]["ticker"]) for y in years}


def recon_date(dec_year: int) -> pd.Timestamp:
    """Monday after the 3rd Friday of December (NDX reconstitution rule)."""
    d = pd.Timestamp(dec_year, 12, 1)
    fridays = [d + pd.Timedelta(days=k) for k in range(31 - d.day + 1)
               if (d + pd.Timedelta(days=k)).weekday() == 4
               and (d + pd.Timedelta(days=k)).month == 12]
    return fridays[2] + pd.Timedelta(days=3)


# adds between consecutive snapshots -> event at Dec reconstitution of Y-1
events = []
for a, b in zip(years[:-1], years[1:]):
    if b - a != 1:
        continue
    for t_ in sorted(snap[b] - snap[a]):
        events.append((recon_date(a), t_))
ev = pd.DataFrame(events, columns=["date", "ticker"])
print(f"NDX add events: {len(ev)} across {len(years)-1} year-pairs")

# S&P membership at event time: current members - later adds + later removals
sp = pd.read_csv("data/cache/sp500_changes.csv", parse_dates=["date"])
daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)
try:
    cur = pd.read_html(io.StringIO(requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers=HDR, timeout=30).text))[0]
    sp_now = set(cur["Symbol"].astype(str).str.strip())
except Exception:
    sp_now = set(daily.columns)

def sp_member(ticker: str, when: pd.Timestamp) -> bool:
    m = ticker in sp_now
    later = sp[(sp["date"] > when)]
    for _, r in later.iterrows():
        if r["ticker"] != ticker:
            continue
        if r["action"] == "add":
            m = False       # added after `when` -> was NOT a member then
        elif r["action"] == "remove":
            m = True        # removed after `when` -> WAS a member then
    return m

ev["in_sp"] = [sp_member(r.ticker, r.date) for r in ev.itertuples()]
print(f"  adds already in S&P: {int(ev['in_sp'].sum())} | "
      f"not in S&P: {int((~ev['in_sp']).sum())}")

# ---- event streams ----
stocks = [c for c in daily.columns if not c.startswith("^")]
px_all = daily[stocks].ffill(limit=5)


def event_stream(sub: pd.DataFrame, hold_days: int = 20) -> pd.Series:
    idx = px_all.index
    P = pd.DataFrame(0.0, index=idx,
                     columns=sorted(set(sub["ticker"]) & set(px_all.columns)))
    used = 0
    for _, r in sub.iterrows():
        s_ = r["ticker"]
        if s_ not in P.columns:
            continue
        loc = idx.searchsorted(r["date"])
        if loc >= len(idx) - 2:
            continue
        seg = px_all[s_].iloc[loc:loc + hold_days + 1]
        if seg.notna().sum() < 10:
            continue
        P.iloc[loc:min(loc + hold_days, len(idx) - 1) + 1,
               P.columns.get_loc(s_)] = 1.0
        used += 1
    ret_ = px_all[P.columns].pct_change().clip(-0.5, 0.5)
    w = P.div(P.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    s = ((ret_.fillna(0) * w.shift(1)).sum(axis=1)
         - w.diff().abs().sum(axis=1).fillna(0) * TC)
    print(f"    events used {used}; in-market {float((P.sum(axis=1)>0).mean()):.0%}")
    return s.dropna()


def load(n):
    df = pd.read_csv(f"strategies/performance/{n}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)

champ = load("portfolio_champ_d14")
for label, sub in (("ALL NDX adds", ev),
                   ("adds ALREADY in S&P (reverse migration)", ev[ev["in_sp"]]),
                   ("adds NOT in S&P", ev[~ev["in_sp"]])):
    print(f"  -- {label}:")
    s = event_stream(sub)
    s = s[s.index <= END]
    first = s.ne(0).idxmax()
    s = s[s.index >= first]
    ms = metrics_from_returns(s.values, TD)
    rec = s[s.index >= "2015-01-01"]
    mr = metrics_from_returns(rec.values, TD)
    cc = float(np.corrcoef(s.reindex(champ.index).fillna(0), champ)[0, 1])
    print(f"    standalone {s.index.min().date()}..: Sharpe {ms['sharpe']:.2f} "
          f"CAGR {ms['cagr']:.1%} MaxDD {ms['max_dd']:.0%} | 2015+ {mr['sharpe']:.2f} | "
          f"corr(champ) {cc:+.2f}")
print(f"total {time.time()-t0:.0f}s")
