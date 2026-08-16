"""
Book C (Intraday MR + momentum flip) — proper daily MARK-TO-MARKET attribution.

Replicates strategies/intraday_mean_reversion.py exactly (signals, phases,
costs, borrow) with locked params sigma=4.0, lookback=20d, flip_hold=3d, top5,
but distributes each trade's P&L across the days it is actually held instead of
lumping it on the entry date:

  exec day   : phase1 (1st bar o->c, faded) + phase2 (2nd bar open -> day close, flipped)
  middle days: day close -> day close, flipped direction
  exit day   : prev day close -> last bar close, flipped direction

Costs identical to engine: 0.1% per phase per position; 8%/yr borrow on short
holds (hourly for phase 1 shorts, daily for phase 2 shorts). Multiple concurrent
trades add (engine attributes additively too). Records as book_c_retest.
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from collections import defaultdict

from live.config import PARAMS
from tools.metrics import sharpe_ratio, max_drawdown, metrics_from_returns
from tools.record import record_performance

TD, TC, BORROW = 252, 0.001, 0.08
p = PARAMS["C"]
SIGMA, LB, FLIP, TOPN = p["sigma"], p["z_lookback_days"], p["flip_hold_days"], p["top_n"]

print("Loading panels...")
ho = pd.read_parquet("data/cache/merged_hourly_open.parquet")
hc = pd.read_parquet("data/cache/merged_hourly_close.parquet")
ho.index = pd.to_datetime(ho.index); hc.index = pd.to_datetime(hc.index)
daily_all = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily_all.index = pd.to_datetime(daily_all.index)
common = sorted(set(ho.columns) & set(hc.columns) & set(daily_all.columns))
ho, hc = ho[common], hc[common]
daily_ret = daily_all[common].pct_change()

# trading-day index of hourly bars (same as engine's _build_day_index)
day_idx = defaultdict(list)
for ts in ho.index:
    day_idx[ts.date()].append(ts)
days = sorted(day_idx.keys())
day_to_i = {d: i for i, d in enumerate(days)}
# daily close per engine convention = last hourly bar close of the day
day_close = pd.DataFrame({d: hc.loc[day_idx[d][-1]] for d in days}).T
day_close.index = pd.to_datetime(day_close.index)

roll_mean = daily_ret.rolling(LB).mean()
roll_std  = daily_ret.rolling(LB).std()

daily_borrow  = BORROW / TD
hourly_borrow = daily_borrow / 6.5
p2_hold_days  = FLIP if FLIP >= 1 else 0.5

pnl = defaultdict(float)          # date -> summed daily return
n_trades = 0

for sig_date, exec_date in zip(days[:-1], days[1:]):
    sig_ts = pd.Timestamp(sig_date)
    if sig_ts not in daily_ret.index:
        continue
    r, mu, sd = daily_ret.loc[sig_ts], roll_mean.loc[sig_ts], roll_std.loc[sig_ts]
    z = ((r - mu) / sd).where(sd > 0).dropna()
    longs  = z[z < -SIGMA].nsmallest(TOPN)
    shorts = z[z >  SIGMA].nlargest(TOPN)
    positions = {t: 1 for t in longs.index}
    positions.update({t: -1 for t in shorts.index})
    if not positions:
        continue
    bars = day_idx.get(exec_date, [])
    if len(bars) < 2:
        continue
    exec_i = day_to_i[exec_date]
    p2_exit_i = exec_i + FLIP if FLIP > 0 else exec_i
    if p2_exit_i >= len(days):
        continue
    p2_exit_day = days[p2_exit_i]
    exit_bars = day_idx.get(p2_exit_day, [])
    if not exit_bars:
        continue

    # per-day contributions for this trade: {date: [per-position returns]}
    contrib = defaultdict(list)
    held_days = [days[j] for j in range(exec_i, p2_exit_i + 1)]
    for tick, d0 in positions.items():
        try:
            ep1 = ho.at[bars[0], tick]; xp1 = hc.at[bars[0], tick]
            ep2 = ho.at[bars[1], tick]
            xp2 = hc.at[exit_bars[-1], tick]
            if any(pd.isna(v) or v <= 0 for v in (ep1, xp1, ep2)) or pd.isna(xp2):
                continue
        except KeyError:
            continue
        flip_dir = -d0
        # ── exec day: phase1 + phase2 to day close ──
        p1 = (xp1/ep1 - 1) * d0 - TC - (hourly_borrow if d0 == -1 else 0.0)
        dc0 = day_close.at[pd.Timestamp(exec_date), tick]
        if pd.isna(dc0) or dc0 <= 0:
            continue
        p2_day0 = (dc0/ep2 - 1) * flip_dir - TC - (daily_borrow if flip_dir == -1 else 0.0)
        contrib[exec_date].append(p1 + p2_day0)
        # ── middle + exit days ──
        prev_c = dc0
        for j, d in enumerate(held_days[1:], start=1):
            if d == p2_exit_day:
                cur = xp2
            else:
                cur = day_close.at[pd.Timestamp(d), tick]
            if pd.isna(cur) or pd.isna(prev_c) or prev_c <= 0:
                prev_c = cur
                continue
            dr = (cur/prev_c - 1) * flip_dir - (daily_borrow if flip_dir == -1 else 0.0)
            contrib[d].append(dr)
            prev_c = cur
    if contrib:
        n_trades += 1
        for d, lst in contrib.items():
            pnl[d] += float(np.mean(lst))

# full trading-day calendar series (0 on flat days)
ser = pd.Series({pd.Timestamp(d): pnl.get(d, 0.0) for d in days}).sort_index()
warm = LB + 1
ser = ser.iloc[warm:]

m = metrics_from_returns(ser.values, TD)
print(f"\nBOOK C [PROPER daily MTM] sigma={SIGMA} lb={LB} flip={FLIP} top{TOPN}")
print(f"  MTM    : Sharpe {m['sharpe']:.3f} | CAGR {m['cagr']:.1%} | "
      f"total {m['total_return']:.1%} | MaxDD {m['max_dd']:.1%} | trades {n_trades}")
print(f"  block  : Sharpe 0.853 | MaxDD -30.0%  (entry-date convention, prior retest)")
print(f"  docs   : Sharpe 0.927 | MaxDD -20.8%")
for y, x in ser.groupby(ser.index.year):
    print(f"   {y}: ret {(1+x).prod()-1:+8.2%}  sharpe {sharpe_ratio(x.values,TD):6.2f}  "
          f"mdd {max_drawdown(x.values):8.2%}")

record_performance(
    name="book_c_retest", dates=ser.index, returns=ser.values,
    params=dict(sigma=SIGMA, z_lookback_days=LB, flip_hold_days=FLIP, top_n=TOPN),
    data_period=f"{ser.index.min().date()}..{ser.index.max().date()}",
    periods_per_year=TD,
    extra={"convention": "PROPER daily MTM (rewrite)", "retest": "2026-08-15 all-books retest",
           "n_trades": n_trades})
print("recorded -> strategies/performance/book_c_retest_* (MTM, replaces block version)")
