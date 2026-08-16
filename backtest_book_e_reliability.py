"""
Book E reliability check.

Replicates the EXACT recorded Book E strategy (reddit_sentiment_long_backtest.py,
locked live params ma=15 zw=40 mild=0.5 extreme=0.6 hold=8 top_n=5 min_ment=5):
bubble-proxy on a sentiment price-index; LONG capitulation (score < -extreme)
plus moderate-hype (mild < score <= extreme); cash otherwise.

Then re-scores it two ways to test whether the documented Sharpe 1.97 is an
artifact of methodology:

  1. ORIGINAL convention (per-trade block returns, RF=2%, TC=0.25% one-way,
     original window 2024-03-31..2026-06-01)  -> should reproduce ~1.97
  2. PROJECT-STANDARD convention (daily returns, Sharpe = mean*252/(std*sqrt252),
     rf=0, same window)                        -> conversion effect
  3. PROJECT-STANDARD on FULL 2019-2026 backfilled history -> earliest-data rule

Records the full-history daily series via tools/record.py.
"""
from __future__ import annotations
import numpy as np, pandas as pd, duckdb

from tools.metrics import sharpe_ratio, max_drawdown, metrics_from_returns
from tools.record import record_performance

P = dict(ma=15, zw=40, mild=0.5, extreme=0.6, hold=8, top_n=5, minm=5)  # locked live params
RF, TC, TD = 0.02, 0.0025, 252
OLD_START, OLD_END = "2024-03-31", "2026-06-01"

con = duckdb.connect("data/market_data.duckdb", read_only=True)
sd = con.execute("select date,symbol,weighted_compound,mention_count from sentiment_daily").df()
sd["date"] = pd.to_datetime(sd["date"])
sent_all = sd.pivot_table(index="date", columns="symbol", values="weighted_compound", aggfunc="mean")
ment_all = sd.pivot_table(index="date", columns="symbol", values="mention_count", aggfunc="sum").fillna(0)
prices = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
prices.index = pd.to_datetime(prices.index)


def build(window_start, window_end):
    s = sent_all.loc[(sent_all.index >= window_start) & (sent_all.index <= window_end)]
    cal = pd.bdate_range(s.index.min(), s.index.max())
    syms = [x for x in s.columns if x in prices.columns]
    sent = sent_all.reindex(cal)[syms]
    ment = ment_all.reindex(cal)[syms].fillna(0)
    px = prices.reindex(cal)[syms].ffill()
    ret = px.pct_change().fillna(0)
    cum = (ment > 0).cumsum()
    return cal, syms, sent, ment, ret, cum


def sidx(s, scale=0.05, base=100.0):
    return base * (1 + s.fillna(0).clip(-1, 1) * scale).cumprod()

def bub(p, ma, z):
    lp = np.log(p.replace(0, np.nan).ffill()); fair = p.rolling(ma).mean()
    res = lp - np.log(fair)
    zz = (res - res.rolling(z).mean()) / res.rolling(z).std()
    return np.tanh(zz / 2)


def signals(cal, syms, sent):
    b = pd.DataFrame(index=cal, columns=syms, dtype=float)
    for s in syms:
        b[s] = bub(sidx(sent[s]), P["ma"], P["zw"])
    return b.shift(1)


def run(cal, sig, cum, ret, daily: bool):
    """daily=False -> original per-block series; daily=True -> daily return series."""
    n = len(cal); warmup = P["ma"] + P["zw"] + 1
    i = warmup; block, dates_b = [], []
    daily_rows = []
    while i < n - P["hold"]:
        elig = cum.iloc[i-1]; elig = elig[elig >= P["minm"]].index
        sc = sig.iloc[i][elig].dropna()
        if sc.empty:
            i += 1; continue
        cap  = sc[sc < -P["extreme"]].nsmallest(P["top_n"])
        momo = sc[(sc > P["mild"]) & (sc <= P["extreme"])].nlargest(P["top_n"])
        longs = pd.concat([cap, momo]).drop_duplicates()
        if len(longs) == 0:
            block.append(RF/TD*P["hold"]); dates_b.append(cal[i])
            for j in range(i, i+P["hold"]):
                daily_rows.append((cal[j], RF/TD))
            i += P["hold"]; continue
        fwd = ret.iloc[i:i+P["hold"]][longs.index]
        block.append(fwd.mean(axis=1).sum() - TC); dates_b.append(cal[i])
        dr = fwd.mean(axis=1)
        dr.iloc[0] -= TC
        for d, r in dr.items():
            daily_rows.append((d, float(r)))
        i += P["hold"]
    if daily:
        ser = pd.Series(dict(daily_rows)).sort_index()
        return ser[~ser.index.duplicated(keep="last")]
    return pd.Series(block, index=pd.DatetimeIndex(dates_b))


def old_perf(s):
    yrs = (s.index[-1]-s.index[0]).days/365.25; tpy = len(s)/yrs
    w = (1+s).cumprod(); tot = w.iloc[-1]-1
    ann = (1+tot)**(1/yrs)-1
    sh = (s.mean()-RF/tpy)/s.std()*np.sqrt(tpy)
    dd = (w/w.cummax()-1).min()
    return dict(ann=ann, sharpe=sh, maxdd=dd, total=tot, trades=len(s))


def report_daily(tag, ser):
    m = metrics_from_returns(ser.values, TD)
    print(f"[{tag}] daily-Sharpe {m['sharpe']:.3f} | CAGR {m['cagr']:.1%} | "
          f"total {m['total_return']:.1%} | MaxDD {m['max_dd']:.1%} | days {m['n']}")
    for y, x in ser.groupby(ser.index.year):
        print(f"   {y}: ret {(1+x).prod()-1:+7.2%}  sharpe {sharpe_ratio(x.values,TD):5.2f}  "
              f"mdd {max_drawdown(x.values):7.2%}  days {len(x)}")
    return m


# ── 1) Reproduce original convention on original window ─────────────────────
cal, syms, sent, ment, ret, cum = build(OLD_START, OLD_END)
sig = signals(cal, syms, sent)
blk = run(cal, sig, cum, ret, daily=False)
op = old_perf(blk)
print(f"[1: ORIGINAL convention, {OLD_START}..{OLD_END}] "
      f"block-Sharpe {op['sharpe']:.3f} | ann {op['ann']:.1%} | MaxDD {op['maxdd']:.1%} | trades {op['trades']}")
print("    (documented Book E: Sharpe 1.968, ann 37.0%, MaxDD -12.2%, 63 trades)")

# ── 2) Same window, project-standard daily convention ───────────────────────
ser2 = run(cal, sig, cum, ret, daily=True)
print()
report_daily(f"2: DAILY convention, {OLD_START}..{OLD_END}", ser2)

# ── 3) Full backfilled history, daily convention ────────────────────────────
cal, syms, sent, ment, ret, cum = build("2019-01-01", "2026-06-01")
sig = signals(cal, syms, sent)
ser3 = run(cal, sig, cum, ret, daily=True)
print()
m3 = report_daily("3: DAILY convention, FULL 2019-2026", ser3)

record_performance(
    name="book_e_locked_params_full_history",
    dates=ser3.index, returns=ser3.values,
    params=P, data_period=f"{ser3.index.min().date()}..{ser3.index.max().date()}",
    periods_per_year=TD,
    extra={"note": "exact locked live Book E strategy re-scored with daily attribution "
                   "on full backfilled sentiment history; reliability check"})
print("\nrecorded -> strategies/performance/book_e_locked_params_full_history_*")
