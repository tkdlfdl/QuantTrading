"""
3-Strategy Portfolio Backtest
==============================
Strategies:
  1. Momentum + UVXY Hedge + Leverage  (daily, from 2015)
  2. Intraday Mean-Reversion + Flip    (daily, from 2019 via Alpaca+yfinance cache)
  3. QQQ Hourly Bubble Score           (daily, from 2020 via Alpaca+yfinance cache)

Allocation methods:
  A. Fixed-weight grid search  (Momentum / IntradayMR / QQQBubble, step=10%)
  B. Momentum allocation with QQQBubble as a fixed anchor weight
       - QQQBubble weight: fixed (grid over qqq_fixed_grid)
       - Remaining split dynamically between Momentum and IntradayMR
         proportional to their recent cumulative return (lookback window)
       - Transaction cost deducted on rebalance day (0.1% x one-way turnover)

Output: yearly Return / Sharpe / Max Drawdown per strategy + portfolio.

Usage:
  python run_3strategy_backtest.py
"""
import sys, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_n = [0]
def _save(*a, **k):
    _n[0] += 1
    p = f"results/3strat_chart_{_n[0]}.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    print(f"  [chart saved: {p}]")
plt.show = _save

sys.path.insert(0, ".")

import os, math
import numpy as np
import pandas as pd
import requests
from io import StringIO
from itertools import product
from datetime import datetime, timedelta
import yfinance as yf

from data.db.schema import init
from data.universe import get_universe
from data.intraday_loader import load_daily_close, load_hourly_bars
from strategies.momentum_bubble_hedge import run_momentum_bubble_hedge_and_low_bubble_leverage
from strategies.intraday_mean_reversion import run_intraday_mean_reversion
from strategies.qqq_bubble_hourly import run_qqq_bubble_hourly

os.makedirs("results", exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
MOM_START    = "2015-01-01"
TRADING_DAYS = 252
TC_RATE      = 0.001    # 0.1% per unit of one-way turnover on rebalance
WEIGHT_STEP  = 0.10     # fixed-weight grid step

MOM_PARAMS = dict(
    lookback=140, holding_period=40, LongShort_flag=True, top=5,
    bubble_indicator_grid=["QQQ", "SPY", "Momentum"],
    ma_window_grid=[120], z_window_grid=[240],
    hedge_bubble_entry_grid=[0.85], hedge_alloc_grid=[0.5], hedge_hold_days_grid=[40],
    low_bubble_entry_grid=[-0.88],
    momentum_extra_leverage_grid=[0.25],
    leverage_hold_days_grid=[50],
)

MR_PARAMS = dict(
    sigma_grid=[4.0],
    flip_hold_days_grid=[2],
    lookback_grid=[20],
    top_n_grid=[5],
    transaction_cost=0.001,
    short_borrow_rate=0.08,
)

QQQ_PARAMS = dict(
    ma_window_grid=[50],
    z_window_grid=[200],
    buy_threshold_grid=[0.9],
    short_threshold_grid=[0.95],
    hold_hours_grid=[24],
    transaction_cost=0.001,
    short_borrow_rate=0.08,
    enable_short=True,
)

# Momentum-allocation grid
DYN_QQQ_FIXED  = [0.0, 0.1, 0.2, 0.3]     # QQQBubble anchor weights
DYN_LOOKBACK   = [20, 40, 60, 120]
DYN_HOLD       = [5, 10, 20, 40]
DYN_MAX_ALLOC  = [0.5, 0.6, 0.7, 0.8, 1.0]  # cap on any single dynamic strategy


# ── Helpers ────────────────────────────────────────────────────────────────

def _sharpe(r, td=TRADING_DAYS):
    s = r.std(); return float(np.sqrt(td) * r.mean() / s) if s > 0 else np.nan

def _sortino(r, td=TRADING_DAYS):
    ds = r[r < 0].std(); return float(np.sqrt(td) * r.mean() / ds) if ds > 0 else np.nan

def _mdd(r):
    w = (1 + r).cumprod(); w = w / w.iloc[0]
    return float((w / w.cummax() - 1).min())

def _stats(r):
    w = (1 + r).cumprod(); w = w / w.iloc[0]
    return dict(Sharpe=_sharpe(r), Sortino=_sortino(r),
                Return=float(w.iloc[-1] - 1), Max_DD=_mdd(r))

def _weight_grid(names, step):
    vals = np.arange(0, 1 + step, step).round(2)
    return [dict(zip(names, c)) for c in product(vals, repeat=len(names))
            if abs(sum(c) - 1.0) < 1e-9]

def _apply_max_alloc(w_dict, max_alloc):
    """Iteratively cap any weight above max_alloc and redistribute excess."""
    if max_alloc >= 1.0:
        return w_dict
    w = dict(w_dict)
    for _ in range(len(w) + 1):
        capped  = {k: min(v, max_alloc) for k, v in w.items()}
        excess  = sum(w[k] - capped[k] for k in w)
        if excess < 1e-9:
            return capped
        free    = {k: v for k, v in capped.items() if v < max_alloc - 1e-9}
        f_total = sum(free.values())
        if f_total <= 0:
            return capped
        for k in free:
            capped[k] += excess * capped[k] / f_total
        w = capped
    return w

def _header(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}")


# ── Yearly breakdown helper ─────────────────────────────────────────────────

def yearly_table(ret_dict):
    rows = []
    all_years = sorted({y for r in ret_dict.values() for y in r.index.year.unique()})
    for yr in all_years:
        row = {"Year": yr}
        for name, r in ret_dict.items():
            yr_r = r[r.index.year == yr]
            if yr_r.empty:
                row[f"{name}_Ret"] = np.nan
                row[f"{name}_Sharpe"] = np.nan
                row[f"{name}_MDD"] = np.nan
            else:
                w = (1 + yr_r).cumprod(); w = w / w.iloc[0]
                row[f"{name}_Ret"]    = round(float(w.iloc[-1] - 1), 4)
                row[f"{name}_Sharpe"] = round(_sharpe(yr_r), 3)
                row[f"{name}_MDD"]    = round(float((w / w.cummax() - 1).min()), 4)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Year")


def print_yearly(df, names):
    for name in names:
        cols = [f"{name}_Ret", f"{name}_Sharpe", f"{name}_MDD"]
        present = [c for c in cols if c in df.columns]
        if not present: continue
        print(f"\n  [{name}]")
        sub = df[present].copy(); sub.columns = ["Return","Sharpe","MaxDD"]
        for yr, row in sub.iterrows():
            r  = f"{row['Return']:>+8.2%}" if pd.notna(row['Return'])  else "       N/A"
            sh = f"{row['Sharpe']:>7.3f}"  if pd.notna(row['Sharpe'])  else "    N/A"
            md = f"{row['MaxDD']:>8.2%}"   if pd.notna(row['MaxDD'])   else "     N/A"
            print(f"    {yr}:  Return={r}  Sharpe={sh}  MaxDD={md}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 1  -  Load individual strategy returns
# ══════════════════════════════════════════════════════════════════════════

init()
_header("STEP 1  -  Strategy returns")

# ── 1. Momentum (daily, 2015+) ─────────────────────────────────────────────
print("\n[1/3] Momentum strategy (from 2015)...")
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

def _scrape(url, col):
    try:
        html = session.get(url, timeout=20)
        for t in pd.read_html(StringIO(html.text)):
            if col in t.columns:
                return (t[col].dropna().astype(str).str.strip()
                              .str.replace(".", "-", regex=False).unique().tolist())
    except Exception:
        pass
    return []

sp500     = _scrape("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol")
nasdaq100 = _scrape("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker")
stock_list = list(set(nasdaq100 + sp500 + ["TMF","TLT"])) + [
    "QQQ","SPY","UVXY","^VIX","^FVX","^TNX"]

print(f"  Downloading {len(stock_list)} tickers from {MOM_START}...")
df_raw = yf.download(tickers=stock_list, start=MOM_START, progress=False, auto_adjust=True)
close  = df_raw.bfill().ffill().dropna(axis="columns")["Close"].copy()
mom_results = run_momentum_bubble_hedge_and_low_bubble_leverage({"Close": close}, **MOM_PARAMS)
mom_ret = mom_results[3]["Momentum_HighBubbleHedge_LowBubbleLeverage"].rename("Momentum")
print(f"  Momentum: {mom_ret.index[0].date()} - {mom_ret.index[-1].date()} "
      f"| {len(mom_ret)} days | Sharpe={_sharpe(mom_ret):.3f} | "
      f"Return={(1+mom_ret).prod()-1:+.1%}")

# ── 2. IntradayMR (hourly/daily, 2019+) ────────────────────────────────────
print("\n[2/3] Intraday Mean-Reversion strategy (from 2019)...")
universe    = get_universe()
daily_close = load_daily_close(universe, use_cache=True)
hourly_open, hourly_close_df = load_hourly_bars(universe, use_cache=True)
mr_raw, _, _ = run_intraday_mean_reversion(
    daily_close=daily_close,
    hourly_open=hourly_open,
    hourly_close=hourly_close_df,
    **MR_PARAMS,
)
mr_ret = mr_raw.rename("IntradayMR") if mr_raw is not None else pd.Series(dtype=float)
print(f"  IntradayMR: {mr_ret.index[0].date()} - {mr_ret.index[-1].date()} "
      f"| {len(mr_ret)} days | Sharpe={_sharpe(mr_ret):.3f} | "
      f"Return={(1+mr_ret).prod()-1:+.1%}")

# ── 3. QQQBubble (hourly, 2020+) ───────────────────────────────────────────
print("\n[3/3] QQQ Bubble strategy (from cached Alpaca data)...")
from pathlib import Path as _P
_qo = _P("data/cache/qqq_hourly_open.parquet")
_qc = _P("data/cache/qqq_hourly_close.parquet")
if _qo.exists():
    qqq_ho = pd.read_parquet(_qo).iloc[:,0]
    qqq_hc = pd.read_parquet(_qc).iloc[:,0]
    if qqq_ho.index.tz is not None:
        qqq_ho.index = qqq_ho.index.tz_localize(None)
        qqq_hc.index = qqq_hc.index.tz_localize(None)
    print(f"  Cache: {qqq_ho.index[0].date()} - {qqq_ho.index[-1].date()} "
          f"| {len(qqq_ho)} hourly bars")
else:
    start730 = (datetime.now() - timedelta(days=729)).strftime("%Y-%m-%d")
    raw = yf.download("QQQ", start=start730, interval="1h", progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    qqq_ho = raw["Open"].ffill()
    qqq_hc = raw["Close"].ffill()

qqq_raw, _, _ = run_qqq_bubble_hourly(hourly_open=qqq_ho, hourly_close=qqq_hc, **QQQ_PARAMS)
qqq_ret = qqq_raw.rename("QQQBubble") if qqq_raw is not None else pd.Series(dtype=float)
print(f"  QQQBubble: {qqq_ret.index[0].date()} - {qqq_ret.index[-1].date()} "
      f"| {len(qqq_ret)} days | Sharpe={_sharpe(qqq_ret):.3f} | "
      f"Return={(1+qqq_ret).prod()-1:+.1%}")

# ── Align to common intersection ───────────────────────────────────────────
idx = mom_ret.index.intersection(mr_ret.index).intersection(qqq_ret.index)
mom_c  = mom_ret.loc[idx]
mr_c   = mr_ret.loc[idx]
qqq_c  = qqq_ret.loc[idx]

print(f"\nCommon period: {idx[0].date()} to {idx[-1].date()} ({len(idx)} trading days)")
print(f"  Momentum   Sharpe={_sharpe(mom_c):>6.3f}  Return={(1+mom_c).prod()-1:>+8.2%}  MaxDD={_mdd(mom_c):>7.2%}")
print(f"  IntradayMR Sharpe={_sharpe(mr_c):>6.3f}  Return={(1+mr_c).prod()-1:>+8.2%}  MaxDD={_mdd(mr_c):>7.2%}")
print(f"  QQQBubble  Sharpe={_sharpe(qqq_c):>6.3f}  Return={(1+qqq_c).prod()-1:>+8.2%}  MaxDD={_mdd(qqq_c):>7.2%}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 2  -  Individual yearly breakdown (each strategy's full history)
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 2  -  Individual strategy yearly performance (full history)")
standalone = {"Momentum": mom_ret, "IntradayMR": mr_ret, "QQQBubble": qqq_ret}
yr_ind = yearly_table(standalone)
print_yearly(yr_ind, list(standalone.keys()))


# ══════════════════════════════════════════════════════════════════════════
# STEP 3  -  Option A: Fixed weight grid search
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 3  -  Option A: Fixed weight grid (step=10%)")

names  = ["Momentum", "IntradayMR", "QQQBubble"]
combos = _weight_grid(names, WEIGHT_STEP)
print(f"Testing {len(combos)} weight combinations...\n")

rows_A = []
for w in combos:
    port = mom_c * w["Momentum"] + mr_c * w["IntradayMR"] + qqq_c * w["QQQBubble"]
    s    = _stats(port)
    rows_A.append({**{f"w_{n}": w[n] for n in names}, **s})

dfA = pd.DataFrame(rows_A).sort_values("Sharpe", ascending=False)

print(f"{'w_Mom':>8} {'w_MR':>8} {'w_QQQ':>8} | {'Sharpe':>7} {'Sortino':>8} {'Return':>9} {'Max_DD':>8}")
print("-" * 62)
for _, r in dfA.head(20).iterrows():
    print(f"{r['w_Momentum']:>8.0%} {r['w_IntradayMR']:>8.0%} {r['w_QQQBubble']:>8.0%} | "
          f"{r['Sharpe']:>7.3f} {r['Sortino']:>8.3f} {r['Return']:>9.2%} {r['Max_DD']:>8.2%}")

bA = dfA.iloc[0]
bestA_ret = (mom_c * bA["w_Momentum"] + mr_c * bA["w_IntradayMR"] + qqq_c * bA["w_QQQBubble"])
bestA_ret = bestA_ret.rename("Portfolio_Fixed")
w_desc_A  = f"{bA['w_Momentum']:.0%} Mom + {bA['w_IntradayMR']:.0%} MR + {bA['w_QQQBubble']:.0%} QQQ"

print(f"\nBest fixed weights: {w_desc_A}")
print(f"  Sharpe={bA['Sharpe']:.3f}  Sortino={bA['Sortino']:.3f}  "
      f"Return={bA['Return']:+.1%}  MaxDD={bA['Max_DD']:.1%}")

_header("Option A  -  Yearly breakdown (best fixed weights)")
yr_A = yearly_table({"Portfolio_Fixed": bestA_ret})
print_yearly(yr_A, ["Portfolio_Fixed"])


# ══════════════════════════════════════════════════════════════════════════
# STEP 4  -  Option B: Momentum allocation (QQQBubble fixed anchor)
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 4  -  Option B: Momentum allocation (QQQBubble fixed, TC on rebalance)")

print(f"Transaction cost: {TC_RATE:.1%} per unit of one-way turnover\n")
print(f"Grid: {len(DYN_QQQ_FIXED)} qqq_fixed x {len(DYN_LOOKBACK)} lookback x "
      f"{len(DYN_HOLD)} hold x {len(DYN_MAX_ALLOC)} max_alloc = "
      f"{len(DYN_QQQ_FIXED)*len(DYN_LOOKBACK)*len(DYN_HOLD)*len(DYN_MAX_ALLOC)} combos\n")

ret_df = pd.DataFrame({"Momentum": mom_c, "IntradayMR": mr_c, "QQQBubble": qqq_c})

rows_B      = []
bestB_sharpe = -np.inf
bestB_ret    = None
bestB_params = None

for qqq_fixed, lookback, hold_period, max_alloc in product(
        DYN_QQQ_FIXED, DYN_LOOKBACK, DYN_HOLD, DYN_MAX_ALLOC):

    remaining = 1.0 - qqq_fixed
    if remaining < 0:
        continue

    dyn_names = ["Momentum", "IntradayMR"]
    port_rows  = []
    prev_w     = None   # weights from previous rebalance period

    for i in range(lookback, len(ret_df) - 1, hold_period):

        # ── Compute dynamic weights for Momentum and IntradayMR ────────────
        window   = ret_df[dyn_names].iloc[max(0, i - lookback): i]
        cum_ret  = (1 + window).prod() - 1
        pos      = cum_ret.clip(lower=0)
        total    = pos.sum()

        if total > 0:
            splits = (pos / total).to_dict()
        else:
            splits = {n: 1.0 / len(dyn_names) for n in dyn_names}

        # Scale to remaining fraction, then cap each at max_alloc of total
        raw_w = {n: splits[n] * remaining for n in dyn_names}
        raw_w["QQQBubble"] = qqq_fixed
        new_w = _apply_max_alloc(raw_w, max_alloc)

        # Ensure QQQBubble stays exactly at qqq_fixed (do not cap it)
        new_w["QQQBubble"] = qqq_fixed
        # Re-normalize the dynamic part so total = 1
        dyn_total = sum(new_w[n] for n in dyn_names)
        target_dyn = 1.0 - qqq_fixed
        if dyn_total > 0:
            scale = target_dyn / dyn_total
            for n in dyn_names:
                new_w[n] *= scale

        # ── Transaction cost (one-way turnover vs previous weights) ────────
        if prev_w is not None:
            turnover = sum(abs(new_w.get(k, 0) - prev_w.get(k, 0))
                           for k in set(list(new_w) + list(prev_w))) / 2.0
            rebal_cost = turnover * TC_RATE
        else:
            rebal_cost = 0.0

        # ── Apply weights for this holding period ───────────────────────────
        hold_end = min(i + hold_period, len(ret_df))
        for j, row_i in enumerate(range(i, hold_end)):
            date = ret_df.index[row_i]
            r    = sum(ret_df.loc[date, k] * v for k, v in new_w.items())
            if j == 0:
                r -= rebal_cost   # deduct TC on first day of new period
            port_rows.append({"date": date, "ret": r})

        prev_w = dict(new_w)

    if not port_rows:
        continue

    port_ret = pd.DataFrame(port_rows).set_index("date")["ret"]
    port_ret = port_ret[~port_ret.index.duplicated(keep="last")]

    wealth = (1 + port_ret).cumprod(); wealth = wealth / wealth.iloc[0]
    sh  = _sharpe(port_ret)
    so  = _sortino(port_ret)
    mdd = float((wealth / wealth.cummax() - 1).min())
    tot = float(wealth.iloc[-1] - 1)

    row = dict(qqq_fixed=qqq_fixed, lookback=lookback, hold_period=hold_period,
               max_alloc=max_alloc, Sharpe=sh, Sortino=so,
               Total_Return=tot, Max_DD=mdd)
    rows_B.append(row)

    if pd.notna(sh) and sh > bestB_sharpe:
        bestB_sharpe = sh
        bestB_ret    = port_ret.rename("Portfolio_MomAlloc")
        bestB_params = row

dfB = pd.DataFrame(rows_B).sort_values("Sharpe", ascending=False)

_header("Option B  -  Top 20 by Sharpe")
print(f"{'QQQ_Fix':>8} {'Lookback':>9} {'Hold':>6} {'MaxAlloc':>9} | "
      f"{'Sharpe':>7} {'Sortino':>8} {'Return':>9} {'Max_DD':>8}")
print("-" * 72)
for _, r in dfB.head(20).iterrows():
    print(f"{r['qqq_fixed']:>8.0%} {int(r['lookback']):>9} "
          f"{int(r['hold_period']):>6} {r['max_alloc']:>9.0%} | "
          f"{r['Sharpe']:>7.3f} {r['Sortino']:>8.3f} "
          f"{r['Total_Return']:>9.2%} {r['Max_DD']:>8.2%}")

p = bestB_params
print(f"\nBest momentum alloc: QQQ_fixed={p['qqq_fixed']:.0%}  "
      f"lookback={p['lookback']}d  hold={p['hold_period']}d  max_alloc={p['max_alloc']:.0%}")
print(f"  Sharpe={p['Sharpe']:.3f}  Sortino={p['Sortino']:.3f}  "
      f"Return={p['Total_Return']:+.1%}  MaxDD={p['Max_DD']:.1%}")

_header("Option B  -  Yearly breakdown (best momentum allocation)")
yr_B = yearly_table({"Portfolio_MomAlloc": bestB_ret})
print_yearly(yr_B, ["Portfolio_MomAlloc"])


# ══════════════════════════════════════════════════════════════════════════
# STEP 5  -  Full yearly comparison table
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 5  -  Full yearly comparison")

all_series = {
    "Momentum":         mom_ret,
    "IntradayMR":       mr_ret,
    "QQQBubble":        qqq_ret,
    "Portfolio_Fixed":  bestA_ret,
    "Portfolio_MomAlloc": bestB_ret,
}
yr_all   = yearly_table(all_series)
all_names = list(all_series.keys())

def _fmt_table(df, cols_suffix, fmt_fn):
    cols = [f"{n}{cols_suffix}" for n in all_names if f"{n}{cols_suffix}" in df.columns]
    sub  = df[cols].copy()
    sub.columns = [c.replace(cols_suffix,"") for c in sub.columns]
    return sub.map(lambda x: fmt_fn(x) if pd.notna(x) else "  N/A")

print("\n  YEARLY RETURN")
print(_fmt_table(yr_all, "_Ret",    lambda x: f"{x:>+7.2%}").to_string())
print("\n  YEARLY SHARPE")
print(_fmt_table(yr_all, "_Sharpe", lambda x: f"{x:>6.3f}").to_string())
print("\n  YEARLY MAX DRAWDOWN")
print(_fmt_table(yr_all, "_MDD",    lambda x: f"{x:>7.2%}").to_string())


# ══════════════════════════════════════════════════════════════════════════
# STEP 6  -  Full-period summary
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 6  -  Full-period summary")
rows_sum = []
for name, s in all_series.items():
    w = (1 + s).cumprod(); w = w / w.iloc[0]
    rows_sum.append({"Strategy": name,
                     "Start": str(s.index[0].date()), "End": str(s.index[-1].date()),
                     "Days": len(s), "Total Return": float(w.iloc[-1]-1),
                     "Sharpe": _sharpe(s), "Sortino": _sortino(s), "Max DD": _mdd(s)})
sumdf = pd.DataFrame(rows_sum).set_index("Strategy")
print(sumdf.to_string(float_format=lambda x: f"{x:+.3f}" if abs(x)<100 else f"{x:.1f}"))


# ══════════════════════════════════════════════════════════════════════════
# STEP 7  -  Charts
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 7  -  Generating charts")

colors = {"Momentum":"steelblue","IntradayMR":"seagreen","QQQBubble":"darkorange",
          "Portfolio_Fixed":"crimson","Portfolio_MomAlloc":"navy"}

# Chart 1  -  Cumulative wealth (common period)
fig, axes = plt.subplots(3, 1, figsize=(18,16),
                          gridspec_kw={"height_ratios":[3,2,1.5]})
ax = axes[0]
for name, s in all_series.items():
    w   = (1+s).cumprod(); w = w/w.iloc[0]
    lw  = 3 if "Portfolio" in name else 1.5
    ax.plot(w.index, w.values, label=name, color=colors.get(name,"gray"), linewidth=lw)
ax.set_title("3-Strategy Portfolio Backtest  -  Cumulative Wealth")
ax.set_ylabel("Wealth (rebased to 1)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.1f}x"))

# Chart 2  -  Annual bar chart for portfolios
ax2 = axes[1]
port_s  = {"Portfolio_Fixed": bestA_ret, "Portfolio_MomAlloc": bestB_ret}
all_yrs = sorted({y for s in port_s.values() for y in s.index.year.unique()})
np_p    = len(port_s)
width   = 0.7 / np_p
offsets = np.linspace(-(np_p-1)/2, (np_p-1)/2, np_p) * width
for j, (name, s) in enumerate(port_s.items()):
    rets = [float((1+s[s.index.year==yr]).prod()-1) if (s.index.year==yr).any() else 0.0
            for yr in all_yrs]
    ax2.bar(np.arange(len(all_yrs))+offsets[j], rets, width=width,
            label=name, color=colors.get(name,"gray"), alpha=0.85)
ax2.axhline(0, color="black", lw=0.8)
ax2.set_xticks(range(len(all_yrs))); ax2.set_xticklabels(all_yrs)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax2.set_title("Portfolio Annual Returns"); ax2.legend(fontsize=9); ax2.grid(True,alpha=0.3,axis="y")

# Chart 3  -  Drawdown
ax3 = axes[2]
for name, s in port_s.items():
    w = (1+s).cumprod(); w = w/w.iloc[0]; dd = w/w.cummax()-1
    ax3.fill_between(dd.index, dd.values, 0, alpha=0.4, color=colors.get(name,"gray"), label=name)
ax3.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax3.set_ylabel("Drawdown"); ax3.legend(fontsize=8); ax3.grid(True,alpha=0.3)
plt.tight_layout(); plt.show()

# Chart 2  -  Sharpe heat table (all strategies, yearly)
all_yrs_all = sorted({y for s in all_series.values() for y in s.index.year.unique()})
fig2, ax = plt.subplots(figsize=(max(10, len(all_yrs_all)*1.4), len(all_names)*0.9+2))
ax.axis("off")

def _cell_val(name, yr, metric):
    s = all_series[name]
    yr_s = s[s.index.year == yr]
    if yr_s.empty: return "N/A"
    if metric == "Return": return f"{float((1+yr_s).prod()-1):>+.1%}"
    if metric == "Sharpe": return f"{_sharpe(yr_s):.2f}"
    if metric == "MaxDD":  return f"{_mdd(yr_s):.1%}"

for metric, title in [("Return","Return"),("Sharpe","Sharpe"),("MaxDD","MaxDD")]:
    fig_t, ax_t = plt.subplots(figsize=(max(10, len(all_yrs_all)*1.4), len(all_names)*0.9+2))
    ax_t.axis("off")
    data = [[name] + [_cell_val(name, yr, metric) for yr in all_yrs_all] for name in all_names]
    tbl  = ax_t.table(cellText=data, colLabels=["Strategy"]+list(all_yrs_all),
                      cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.2, 1.9)
    for (ri, ci), cell in tbl.get_celld().items():
        if ri == 0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
        elif ci == 0:
            cell.set_facecolor("#ecf0f1"); cell.set_text_props(fontweight="bold")
        else:
            txt = cell.get_text().get_text().replace("+","").replace("%","")
            try:
                v = float(txt)
                if metric == "Sharpe":
                    if v > 1.5:   cell.set_facecolor("#2ecc71")
                    elif v > 0.5: cell.set_facecolor("#a8e6cf")
                    elif v > 0:   cell.set_facecolor("#ffeaa7")
                    elif v > -0.5:cell.set_facecolor("#fab1a0")
                    else:         cell.set_facecolor("#e17055")
                elif metric in ("Return","MaxDD"):
                    v2 = float(txt)/100
                    if metric == "MaxDD": v2 = -v2   # flip so more negative = worse
                    if v2 > 0.20:   cell.set_facecolor("#2ecc71")
                    elif v2 > 0.05: cell.set_facecolor("#a8e6cf")
                    elif v2 > 0:    cell.set_facecolor("#ffeaa7")
                    elif v2 > -0.10:cell.set_facecolor("#fab1a0")
                    else:           cell.set_facecolor("#e17055")
            except (ValueError, TypeError):
                cell.set_facecolor("#dfe6e9")
    ax_t.set_title(f"Yearly {title} by Strategy", pad=20, fontsize=13)
    plt.tight_layout(); plt.show()


# ══════════════════════════════════════════════════════════════════════════
# STEP 8  -  Save to Excel
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 8  -  Saving to Excel")
xl = "results/3strategy_backtest.xlsx"
try:
    with pd.ExcelWriter(xl, engine="openpyxl") as writer:
        sumdf.reset_index().to_excel(writer, sheet_name="Summary", index=False)

        ret_tbl = _fmt_table(yr_all,"_Ret",    lambda x: f"{x:+.2%}").reset_index()
        sh_tbl  = _fmt_table(yr_all,"_Sharpe", lambda x: f"{x:.3f}").reset_index()
        mdd_tbl = _fmt_table(yr_all,"_MDD",    lambda x: f"{x:.2%}").reset_index()
        ret_tbl.to_excel(writer, sheet_name="Yearly_Return",  index=False)
        sh_tbl .to_excel(writer, sheet_name="Yearly_Sharpe",  index=False)
        mdd_tbl.to_excel(writer, sheet_name="Yearly_MaxDD",   index=False)

        dfA.to_excel(writer, sheet_name="Grid_FixedWeight", index=False)
        dfB.to_excel(writer, sheet_name="Grid_MomAlloc",    index=False)

        daily = pd.DataFrame({k: v for k,v in all_series.items()})
        daily.index.name = "Date"
        daily.to_excel(writer, sheet_name="Daily_Returns")
    print(f"  Saved: {xl}")
except Exception as e:
    print(f"  Excel save failed: {e}")


_header("DONE")
print(f"\nOption A (best fixed):       {w_desc_A}")
print(f"  Sharpe={bA['Sharpe']:.3f}  Return={bA['Return']:+.1%}  MaxDD={bA['Max_DD']:.1%}")
print(f"\nOption B (best mom-alloc):   QQQ_fixed={p['qqq_fixed']:.0%}  "
      f"lookback={p['lookback']}d  hold={p['hold_period']}d  max_alloc={p['max_alloc']:.0%}")
print(f"  Sharpe={p['Sharpe']:.3f}  Return={p['Total_Return']:+.1%}  MaxDD={p['Max_DD']:.1%}")
print(f"\nTransaction cost: {TC_RATE:.1%}/unit of one-way turnover deducted on each rebalance day")
