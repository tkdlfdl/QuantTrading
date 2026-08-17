"""
Cycle 24 part 2 —
 (c-fix) sector weekly reversal, positional-assignment bug fixed (the v1 run
         used chained .iloc[][cols] which writes to a copy -> VOID, retest).
 (d)     FULL earnings-calendar acquisition (universe-wide, cached parquet)
         + Frazzini-Lamont earnings-announcement premium test:
         long each name close(T-2)->close(T+1) around its announcement,
         equal-weight across active events, 0.1%/side.
Gates: standalone, corr, stress, full-stack marginal @0.25, LW p.
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


# ══════════ (c-fix) sector weekly reversal ══════════
print(f"=== (c-fix) sector SPDR weekly reversal ===")
sect = pd.read_parquet("data/cache/sector_etf_close.parquet")
sect.index = pd.to_datetime(sect.index)
sret = sect.pct_change()
r5 = sect.pct_change(5).shift(1)
pos_s = pd.DataFrame(0.0, index=sect.index, columns=sect.columns)
dates = sect.index
i = 10
while i < len(dates) - 5:
    row = r5.iloc[i].dropna()
    if len(row) >= 8:
        losers = [sect.columns.get_loc(c) for c in row.nsmallest(2).index]
        pos_s.iloc[i + 1:i + 6, losers] = 0.5
        i += 5
    else:
        i += 1
sc_ = ((sret.fillna(0) * pos_s).sum(axis=1)
       - pos_s.diff().abs().sum(axis=1).fillna(0) * TC)
sc_ = sc_[sc_.index >= dates[15]].dropna()
judge("sector weekly reversal", sc_)

# also the momentum flavor (winners), same harness — pre-registered pair
pos_w = pd.DataFrame(0.0, index=sect.index, columns=sect.columns)
i = 10
while i < len(dates) - 5:
    row = r5.iloc[i].dropna()
    if len(row) >= 8:
        winners = [sect.columns.get_loc(c) for c in row.nlargest(2).index]
        pos_w.iloc[i + 1:i + 6, winners] = 0.5
        i += 5
    else:
        i += 1
sw_ = ((sret.fillna(0) * pos_w).sum(axis=1)
       - pos_w.diff().abs().sum(axis=1).fillna(0) * TC)
sw_ = sw_[sw_.index >= dates[15]].dropna()
judge("sector weekly momentum", sw_)

# ══════════ (d) full earnings acquisition ══════════
print(f"\n=== (d) earnings-calendar acquisition ({time.time()-t0:.0f}s) ===")
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)
stocks = [c for c in daily.columns
          if not c.startswith("^") and c not in {"SPY", "UVXY", "QQQ"}]

CACHE = Path("data/cache/earnings_dates.parquet")
if CACHE.exists():
    edf = pd.read_parquet(CACHE)
    print(f"  cache hit: {len(edf)} rows, {edf['symbol'].nunique()} tickers")
else:
    def fetch(t_):
        try:
            e = yf.Ticker(t_).get_earnings_dates(limit=120)
            if e is None or len(e) == 0:
                return None
            idx_ = pd.to_datetime(e.index).tz_localize(None).normalize()
            return pd.DataFrame({"symbol": t_, "date": idx_.unique()})
        except Exception:
            return None
    frames = []
    with ThreadPoolExecutor(8) as ex:
        for k, f in enumerate(ex.map(fetch, stocks)):
            if f is not None:
                frames.append(f)
    edf = pd.concat(frames, ignore_index=True)
    edf.to_parquet(CACHE)
    print(f"  fetched+cached: {len(edf)} rows, {edf['symbol'].nunique()} tickers "
          f"({time.time()-t0:.0f}s)")
per_yr = edf.groupby(edf["date"].dt.year).size()
years_ok = per_yr[per_yr > 200]
print(f"  events/yr>200 from {years_ok.index.min()} to {years_ok.index.max()}")

# ---- Frazzini-Lamont: hold close(T-2) -> close(T+1) around announcements ----
px = daily[stocks].ffill()
ret = px.pct_change()
jump = ret.abs().rolling(252).max()
pos = pd.DataFrame(0.0, index=px.index, columns=stocks)
tdx = {d: i for i, d in enumerate(px.index)}
n_ev = 0
start_yr = int(years_ok.index.min())
for _, row in edf.iterrows():
    d_, s_ = row["date"], row["symbol"]
    if d_.year < start_yr or s_ not in tdx and True:
        pass
    # announcement day T: nearest trading day >= d_
    loc = px.index.searchsorted(d_)
    if loc < 3 or loc >= len(px.index) - 2:
        continue
    if d_.year < start_yr:
        continue
    # windows: held on days T-1, T, T+1 (entered close T-2)
    if np.isfinite(jump.iloc[loc][s_]) and jump.iloc[loc][s_] > 1.0:
        continue
    pos.iloc[loc - 1:loc + 2, pos.columns.get_loc(s_)] = 1.0
    n_ev += 1
print(f"  events used: {n_ev}  ({time.time()-t0:.0f}s)")
w = pos.div(pos.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
sd_ = ((ret.fillna(0) * w.shift(1)).sum(axis=1)
       - w.diff().abs().sum(axis=1).fillna(0) * TC * 0.5)
sd_ = sd_[sd_.index >= f"{start_yr}-06-01"].dropna()
judge("earnings-announcement premium", sd_)
print(f"\ntotal {time.time()-t0:.0f}s")
