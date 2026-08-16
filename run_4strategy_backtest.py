"""
4-Strategy Portfolio Backtest + Grid Search
============================================
Strategies:
  1. Momentum + UVXY Hedge + Leverage  (daily)
  2. Intraday Mean-Reversion + Flip    (1h → daily)
  3. QQQ Hourly Bubble Score           (long-only, 1h → daily)
  4. Reddit Sentiment Bubble           (daily; skipped if no sentiment data)

Grid search:
  Option A  -  Fixed weights (step=0.1, all available strategies)
  Option B  -  Leverage on IntradayMR (1x–3x, 12%/yr) + fixed-weight combine
  Option C  -  Dynamic momentum-based allocation (lookback x hold x max_alloc)

Yearly output:
  Return, Sharpe, Max Drawdown per year for each strategy + best portfolio.

Usage:
  python run_4strategy_backtest.py
"""
import sys, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_chart_n = [0]
def _save(*a, **k):
    _chart_n[0] += 1
    fname = f"results/backtest_chart_{_chart_n[0]}.png"
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    print(f"  [chart saved: {fname}]")
plt.show = _save

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import requests
from io import StringIO
from itertools import product
from datetime import datetime, timedelta
import yfinance as yf

from data.db.schema import init
from data.universe import get_universe, get_top_by_marketcap
from data.intraday_loader import load_daily_close, load_hourly_bars
from data.sentiment.aggregator import load_sentiment_panel
from strategies.momentum_bubble_hedge import run_momentum_bubble_hedge_and_low_bubble_leverage
from strategies.intraday_mean_reversion import run_intraday_mean_reversion
from strategies.qqq_bubble_hourly import run_qqq_bubble_hourly
from strategies.reddit_sentiment_bubble import run_reddit_sentiment_bubble
from portfolio.constructor import run_portfolio_allocation

import os
os.makedirs("results", exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
# Each strategy loads from its earliest available data:
#   Momentum  : 2015-01-01 (daily data, yfinance)
#   IntradayMR: 2019-01-02 (merged Alpaca+yfinance hourly cache)
#   QQQBubble : 2020-07-27 (Alpaca QQQ hourly cache)
#   Reddit    : whatever is in the sentiment DB
#
# Portfolio grid uses the common intersection of all active strategies.
MOM_START = "2015-01-01"
QQQ_CACHE_DIR = "data/cache"   # qqq_hourly_open/close.parquet live here

TRADING_DAYS = 252
LEVERAGE_COST_ANNUAL = 0.12   # 12%/yr borrowing cost for leveraged IntradayMR
LEVERAGE_GRID        = [1.0, 1.5, 2.0, 2.5, 3.0]
WEIGHT_STEP          = 0.2    # fixed-weight grid granularity

# Momentum: full grid search across bubble indicators
MOM_PARAMS = dict(
    lookback=140, holding_period=40, LongShort_flag=True, top=5,
    bubble_indicator_grid=["QQQ", "SPY", "Momentum"],
    ma_window_grid=[120], z_window_grid=[240],
    hedge_bubble_entry_grid=[0.85], hedge_alloc_grid=[0.5], hedge_hold_days_grid=[40],
    low_bubble_entry_grid=[-0.88],
    momentum_extra_leverage_grid=[0.25],
    leverage_hold_days_grid=[50],
)

# IntradayMR: use known best params (full grid over 7yr x 516 tickers is too slow)
MR_PARAMS = dict(
    sigma_grid=[4.0],
    flip_hold_days_grid=[2],
    lookback_grid=[20],
    top_n_grid=[5],
    transaction_cost=0.001,
    short_borrow_rate=0.08,
)

# QQQBubble: broader grid over the longer 2020+ window
QQQ_PARAMS = dict(
    ma_window_grid=[20, 50, 100],
    z_window_grid=[50, 100, 200],
    buy_threshold_grid=[0.5, 0.6, 0.7, 0.8, 0.9],
    short_threshold_grid=[0.85, 0.9, 0.92, 0.95, 0.97],
    hold_hours_grid=[1, 2, 4, 8, 24],
    transaction_cost=0.001,
    short_borrow_rate=0.08,
    enable_short=True,   # allow both long and short
)

RED_PARAMS = dict(
    holding_period_grid=[5],
    mild_threshold_grid=[0.5],
    extreme_threshold_grid=[0.9],
    top_n_grid=[5],
    ma_window_grid=[30],
    z_window_grid=[60],
    sentiment_scale_grid=[0.05],
    min_mentions=3,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _sharpe(r: pd.Series, td: int = TRADING_DAYS) -> float:
    s = r.std()
    return float(np.sqrt(td) * r.mean() / s) if s > 0 else np.nan

def _sortino(r: pd.Series, td: int = TRADING_DAYS) -> float:
    ds = r[r < 0].std()
    return float(np.sqrt(td) * r.mean() / ds) if ds > 0 else np.nan

def _mdd(r: pd.Series) -> float:
    w = (1 + r).cumprod(); w = w / w.iloc[0]
    return float((w / w.cummax() - 1).min())

def _stats(r: pd.Series) -> dict:
    w = (1 + r).cumprod(); w = w / w.iloc[0]
    return dict(Sharpe=_sharpe(r), Sortino=_sortino(r),
                Return=float(w.iloc[-1] - 1), Max_DD=_mdd(r))

def _weight_grid(names: list, step: float) -> list:
    vals = np.arange(0, 1 + step, step).round(2)
    return [dict(zip(names, c)) for c in product(vals, repeat=len(names))
            if abs(sum(c) - 1.0) < 1e-9]

def _apply_leverage(ret: pd.Series, lev: float) -> pd.Series:
    daily_cost = (lev - 1) * LEVERAGE_COST_ANNUAL / TRADING_DAYS
    return lev * ret - daily_cost

def _header(title: str):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


# ── Yearly breakdown ───────────────────────────────────────────────────────

def yearly_table(ret_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """Per-year Return / Sharpe / MaxDD for each series."""
    rows = []
    all_years = sorted(set(
        y for r in ret_dict.values() for y in r.index.year.unique()
    ))
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


def print_yearly(df: pd.DataFrame, strategies: list):
    """Pretty-print yearly table per strategy block."""
    for name in strategies:
        cols = [f"{name}_Ret", f"{name}_Sharpe", f"{name}_MDD"]
        present = [c for c in cols if c in df.columns]
        if not present:
            continue
        print(f"\n  [{name}]")
        sub = df[present].copy()
        sub.columns = ["Return", "Sharpe", "MaxDD"]
        for yr, row in sub.iterrows():
            r_str  = f"{row['Return']:>+8.2%}" if pd.notna(row['Return'])  else "       N/A"
            sh_str = f"{row['Sharpe']:>7.3f}"  if pd.notna(row['Sharpe'])  else "    N/A"
            md_str = f"{row['MaxDD']:>8.2%}"   if pd.notna(row['MaxDD'])   else "     N/A"
            print(f"    {yr}:  Return={r_str}  Sharpe={sh_str}  MaxDD={md_str}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 1  -  Load strategy returns
# ══════════════════════════════════════════════════════════════════════════

init()
_header("STEP 1  -  Strategy returns")

# 1. Momentum — daily data from 2015
print("\n[1/4] Momentum strategy...")
HEADERS = {"User-Agent": "Mozilla/5.0"}
session = requests.Session(); session.headers.update(HEADERS)

def _scrape(url, col):
    try:
        html = session.get(url, timeout=20)
        for t in pd.read_html(StringIO(html.text)):
            if col in t.columns:
                return t[col].dropna().astype(str).str.strip() \
                             .str.replace(".", "-", regex=False).unique().tolist()
    except Exception:
        pass
    return []

sp500     = _scrape("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol")
nasdaq100 = _scrape("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker")
base_tickers = list(set(nasdaq100 + sp500 + ["TMF", "TLT"]))
stock_list   = base_tickers + ["QQQ", "SPY", "UVXY", "^VIX", "^FVX",
                                "2Y", "US2Y", "DGS2", "10Y", "US10Y", "^TNX", "DGS10"]
print(f"  Downloading {len(stock_list)} tickers from {MOM_START}...")
df_mom = yf.download(tickers=stock_list, start=MOM_START, progress=False, auto_adjust=True)
close  = df_mom.bfill().ffill().dropna(axis="columns")["Close"].copy()
mom_results = run_momentum_bubble_hedge_and_low_bubble_leverage({"Close": close}, **MOM_PARAMS)
mom_ret = mom_results[3]["Momentum_HighBubbleHedge_LowBubbleLeverage"].rename("Momentum")
print(f"  Momentum: {len(mom_ret)} days  {mom_ret.index[0].date()} to {mom_ret.index[-1].date()}")
print(f"            Sharpe={_sharpe(mom_ret):.3f}  Return={(1+mom_ret).prod()-1:+.1%}")

# 2. Intraday MR
print("\n[2/4] Intraday Mean-Reversion strategy...")
universe    = get_universe()
daily_close = load_daily_close(universe, use_cache=True)
hourly_open, hourly_close_df = load_hourly_bars(universe, use_cache=True)
mr_ret_raw, mr_params, mr_grid = run_intraday_mean_reversion(
    daily_close=daily_close,
    hourly_open=hourly_open,
    hourly_close=hourly_close_df,
    **MR_PARAMS,
)
mr_ret = mr_ret_raw.rename("IntradayMR") if mr_ret_raw is not None else pd.Series(dtype=float)
if not mr_ret.empty:
    print(f"  IntradayMR: {len(mr_ret)} days  Sharpe={_sharpe(mr_ret):.3f}  "
          f"Return={(1+mr_ret).prod()-1:+.1%}")
else:
    print("  IntradayMR: no data returned  -  skipping")

# 3. QQQ Bubble — use cached Alpaca+yfinance hourly data (goes back to 2020-07-27)
print("\n[3/4] QQQ Hourly Bubble strategy...")
from pathlib import Path as _Path
_qqq_o_cache = _Path(QQQ_CACHE_DIR) / "qqq_hourly_open.parquet"
_qqq_c_cache = _Path(QQQ_CACHE_DIR) / "qqq_hourly_close.parquet"

if _qqq_o_cache.exists() and _qqq_c_cache.exists():
    qqq_ho_df = pd.read_parquet(_qqq_o_cache)
    qqq_hc_df = pd.read_parquet(_qqq_c_cache)
    # May be DataFrame with one column or a Series
    qqq_ho = qqq_ho_df.iloc[:, 0] if isinstance(qqq_ho_df, pd.DataFrame) else qqq_ho_df
    qqq_hc = qqq_hc_df.iloc[:, 0] if isinstance(qqq_hc_df, pd.DataFrame) else qqq_hc_df
    if qqq_ho.index.tz is not None:
        qqq_ho.index = qqq_ho.index.tz_localize(None)
        qqq_hc.index = qqq_hc.index.tz_localize(None)
    print(f"  QQQ cache: {qqq_ho.index[0].date()} to {qqq_ho.index[-1].date()} "
          f"({len(qqq_ho)} hourly bars)")
else:
    # Fallback: fresh yfinance download (last 730 days only)
    print("  QQQ cache not found — downloading from yfinance (730d)...")
    qqq_start = (datetime.now() - timedelta(days=729)).strftime("%Y-%m-%d")
    qqq_raw   = yf.download("QQQ", start=qqq_start, interval="1h",
                             progress=False, auto_adjust=True)
    if isinstance(qqq_raw.columns, pd.MultiIndex):
        qqq_raw.columns = qqq_raw.columns.get_level_values(0)
    if qqq_raw.index.tz is not None:
        qqq_raw.index = qqq_raw.index.tz_localize(None)
    qqq_ho = qqq_raw["Open"].ffill()
    qqq_hc = qqq_raw["Close"].ffill()

qqq_daily_raw, qqq_params, qqq_grid = run_qqq_bubble_hourly(
    hourly_open=qqq_ho, hourly_close=qqq_hc, **QQQ_PARAMS
)
qqq_ret = qqq_daily_raw.rename("QQQBubble") if qqq_daily_raw is not None else pd.Series(dtype=float)
if not qqq_ret.empty:
    print(f"  QQQBubble:  {len(qqq_ret)} days  {qqq_ret.index[0].date()} to {qqq_ret.index[-1].date()}")
    print(f"              Sharpe={_sharpe(qqq_ret):.3f}  Return={(1+qqq_ret).prod()-1:+.1%}")
else:
    print("  QQQBubble: no data  -  skipping")

# 4. Reddit Sentiment (optional) — loads all available data from DB
print("\n[4/4] Reddit Sentiment strategy...")
reddit_ret = pd.Series(dtype=float, name="Reddit")
try:
    top_universe = get_top_by_marketcap(universe, n=50)
    # Use earliest sentiment date available in DB
    sent_panel_full = load_sentiment_panel(top_universe)
    red_start = str(sent_panel_full.index[0].date()) if not sent_panel_full.empty else MOM_START
    price_raw    = yf.download(tickers=top_universe, start=red_start, progress=False, auto_adjust=True)
    price_panel  = price_raw["Close"].ffill().dropna(axis="columns")
    common_syms  = [s for s in sent_panel_full.columns if s in price_panel.columns]
    print(f"  Sentiment data: {red_start} to {str(sent_panel_full.index[-1].date()) if not sent_panel_full.empty else 'N/A'}")
    print(f"  Symbols with sentiment: {len(common_syms)}")

    if len(common_syms) >= 3 and len(sent_panel_full) >= 100:
        red_results = run_reddit_sentiment_bubble(
            sentiment_panel=sent_panel_full[common_syms],
            price_panel=price_panel[common_syms],
            transaction_cost=0.001,
            cash_rate=0.02,
            short_borrow_rate=0.08,
            **RED_PARAMS,
        )
        reddit_ret = red_results[3].rename("Reddit")
        print(f"  Reddit:     {len(reddit_ret)} days  {reddit_ret.index[0].date()} to {reddit_ret.index[-1].date()}")
        print(f"              Sharpe={_sharpe(reddit_ret):.3f}  Return={(1+reddit_ret).prod()-1:+.1%}")
    else:
        print("  Reddit: insufficient data  -  running without Reddit")
except Exception as e:
    print(f"  Reddit: error ({e})  -  running without Reddit")


# ── Standalone yearly tables (each strategy's full history) ────────────────
all_standalone = {}
for name, s in [("Momentum", mom_ret), ("IntradayMR", mr_ret),
                ("QQQBubble", qqq_ret), ("Reddit", reddit_ret)]:
    if not s.empty:
        all_standalone[name] = s

# ── Align strategies to common date range for portfolio ────────────────────
active = {k: v for k, v in all_standalone.items()}

# Common intersection (no artificial START cutoff — let data dictate)
idx = active[list(active.keys())[0]].index
for s in active.values():
    idx = idx.intersection(s.index)

active_common = {name: s.loc[idx] for name, s in active.items()}

strategy_names = list(active_common.keys())
print(f"\nActive strategies: {strategy_names}")
print(f"Common period: {idx[0].date()} to {idx[-1].date()} ({len(idx)} trading days)")
print()
for name, s in active.items():
    print(f"  {name:<14} Sharpe={_sharpe(s):>6.3f}  Sortino={_sortino(s):>6.3f}  "
          f"Return={(1+s).prod()-1:>+8.2%}  MaxDD={_mdd(s):>7.2%}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 2  -  Individual strategy yearly breakdown
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 2  -  Individual strategy yearly performance (full history)")
# Use each strategy's full individual history (not the common intersection)
yearly_df = yearly_table(all_standalone)
print_yearly(yearly_df, list(all_standalone.keys()))

_header("STEP 2b  -  Portfolio period individual stats (common intersection)")
for name, s in active_common.items():
    print(f"  {name:<14} Sharpe={_sharpe(s):>6.3f}  Sortino={_sortino(s):>6.3f}  "
          f"Return={(1+s).prod()-1:>+8.2%}  MaxDD={_mdd(s):>7.2%}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 3  -  Option A: Fixed weight grid search
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 3  -  Option A: Fixed weight grid search (step=20%)")

combos = _weight_grid(strategy_names, WEIGHT_STEP)
print(f"Testing {len(combos)} weight combinations...\n")

rows_A = []
for w in combos:
    port = sum(active_common[n] * w[n] for n in strategy_names)
    s    = _stats(port)
    rows_A.append({**{f"w_{n}": w[n] for n in strategy_names}, **s})

dfA = pd.DataFrame(rows_A).sort_values("Sharpe", ascending=False)

# Header
w_cols = " ".join(f"{'w_'+n:>12}" for n in strategy_names)
print(f"{w_cols} | {'Sharpe':>7} {'Sortino':>8} {'Return':>8} {'Max_DD':>8}")
print("-" * (len(strategy_names)*13 + 38))
for _, r in dfA.head(20).iterrows():
    w_str = " ".join(f"{r[f'w_{n}']:>12.0%}" for n in strategy_names)
    print(f"{w_str} | {r['Sharpe']:>7.3f} {r['Sortino']:>8.3f} "
          f"{r['Return']:>8.2%} {r['Max_DD']:>8.2%}")

bestA     = dfA.iloc[0]
bestA_ret = sum(active_common[n] * bestA[f"w_{n}"] for n in strategy_names)
bestA_ret = bestA_ret.rename("Portfolio_FixedW")

w_desc_A = " + ".join(f"{bestA[f'w_{n}']:.0%} {n}" for n in strategy_names)
print(f"\nBest fixed: {w_desc_A}")
print(f"  Sharpe={bestA['Sharpe']:.3f}  Sortino={bestA['Sortino']:.3f}  "
      f"Return={bestA['Return']:+.1%}  MaxDD={bestA['Max_DD']:.1%}")


# ── Yearly for best fixed portfolio ────────────────────────────────────────
_header("Option A  -  Best portfolio yearly breakdown")
yr_portA = yearly_table({"Portfolio_FixedW": bestA_ret})
print_yearly(yr_portA, ["Portfolio_FixedW"])


# ══════════════════════════════════════════════════════════════════════════
# STEP 4  -  Option B: Leverage on IntradayMR + fixed-weight combine
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 4  -  Option B: Leverage on IntradayMR (12%/yr) + fixed-weight combine")

rows_B   = []
bestB_sharpe = -np.inf
bestB_ret    = None
bestB_label  = ""

if "IntradayMR" in active_common:
    mr_series = active_common["IntradayMR"]
    for lev in LEVERAGE_GRID:
        lev_mr     = _apply_leverage(mr_series, lev)
        lev_active = {**active_common, "IntradayMR": lev_mr}
        lev_stats  = _stats(lev_mr)
        print(f"\n  Leveraged MR standalone ({lev:.1f}x): "
              f"Sharpe={lev_stats['Sharpe']:.3f}  "
              f"Return={lev_stats['Return']:+.1%}  Max_DD={lev_stats['Max_DD']:.1%}")

        for w in _weight_grid(strategy_names, WEIGHT_STEP):
            port  = sum(lev_active[n] * w[n] for n in strategy_names)
            s     = _stats(port)
            row   = {"leverage": lev, **{f"w_{n}": w[n] for n in strategy_names}, **s}
            rows_B.append(row)
            if pd.notna(s["Sharpe"]) and s["Sharpe"] > bestB_sharpe:
                bestB_sharpe = s["Sharpe"]
                bestB_ret    = port.rename("Portfolio_LevMR")
                bestB_label  = (f"lev={lev:.1f}x  " +
                                "  ".join(f"w_{n}={w[n]:.0%}" for n in strategy_names))

    dfB = pd.DataFrame(rows_B).sort_values("Sharpe", ascending=False)
    _header("Option B  -  Top 20 by Sharpe")
    lev_col = f"{'lev':>5}"
    w_cols  = " ".join(f"{'w_'+n:>12}" for n in strategy_names)
    print(f"{lev_col} {w_cols} | {'Sharpe':>7} {'Sortino':>8} {'Return':>8} {'Max_DD':>8}")
    print("-" * (len(strategy_names)*13 + 44))
    for _, r in dfB.head(20).iterrows():
        w_str = " ".join(f"{r[f'w_{n}']:>12.0%}" for n in strategy_names)
        print(f"{r['leverage']:>5.1f} {w_str} | {r['Sharpe']:>7.3f} {r['Sortino']:>8.3f} "
              f"{r['Return']:>8.2%} {r['Max_DD']:>8.2%}")

    bB = dfB.iloc[0]
    print(f"\nBest levered: {bestB_label}")
    print(f"  Sharpe={bB['Sharpe']:.3f}  Sortino={bB['Sortino']:.3f}  "
          f"Return={bB['Return']:+.1%}  MaxDD={bB['Max_DD']:.1%}")

    _header("Option B  -  Best portfolio yearly breakdown")
    yr_portB = yearly_table({"Portfolio_LevMR": bestB_ret})
    print_yearly(yr_portB, ["Portfolio_LevMR"])
else:
    print("  IntradayMR not available  -  skipping Option B")
    bestB_ret = None


# ══════════════════════════════════════════════════════════════════════════
# STEP 5  -  Option C: Dynamic momentum allocation grid
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 5  -  Option C: Dynamic momentum allocation")

DYN_LOOKBACK  = [20, 40, 60, 120]
DYN_HOLD      = [5, 10, 20, 40]
DYN_MAX_ALLOC = [0.5, 0.6, 0.7, 0.8, 1.0]

print(f"Grid: {len(DYN_LOOKBACK)} lookback x {len(DYN_HOLD)} hold x "
      f"{len(DYN_MAX_ALLOC)} max_alloc = "
      f"{len(DYN_LOOKBACK)*len(DYN_HOLD)*len(DYN_MAX_ALLOC)} combos\n")

(_, _, bestC_ret, bestC_wealth, bestC_weights,
 grid_C, bestC_params) = run_portfolio_allocation(
    returns_dict=active_common,
    method="momentum",
    lookback_grid=DYN_LOOKBACK,
    hold_period_grid=DYN_HOLD,
    max_alloc_grid=DYN_MAX_ALLOC,
)
bestC_ret = bestC_ret.rename("Portfolio_DynMom")

_header("Option C  -  Top 20 by Sharpe (dynamic momentum allocation)")
print(f"{'Lookback':>9} {'Hold':>6} {'MaxAlloc':>9} | "
      f"{'Sharpe':>7} {'Sortino':>8} {'Return':>9} {'Max_DD':>8}")
print("-" * 60)
for _, r in grid_C.head(20).iterrows():
    lb  = int(r['lookback'])   if r['lookback'] != '-'   else '-'
    hp  = int(r['hold_period']) if r['hold_period'] != '-' else '-'
    ma  = r.get('max_alloc', np.nan)
    ma_s = f"{float(ma):>9.0%}" if pd.notna(ma) and ma != '-' else "      -"
    print(f"{lb:>9} {hp:>6} {ma_s} | "
          f"{r['Sharpe']:>7.3f} {r['Sortino']:>8.3f} "
          f"{r['Total Return']:>9.2%} {r['Max DD']:>8.2%}")

p = bestC_params
print(f"\nBest params: lookback={p.get('lookback','-')}d  "
      f"hold={p.get('hold_period','-')}d  "
      f"max_alloc={float(p.get('max_alloc',1.0)):.0%}")
print(f"  Sharpe={p['Sharpe']:.3f}  Sortino={p['Sortino']:.3f}  "
      f"Return={p['Total Return']:+.1%}  MaxDD={p['Max DD']:.1%}")

_header("Option C  -  Best portfolio yearly breakdown")
yr_portC = yearly_table({"Portfolio_DynMom": bestC_ret})
print_yearly(yr_portC, ["Portfolio_DynMom"])


# ══════════════════════════════════════════════════════════════════════════
# STEP 6  -  Comprehensive yearly comparison table
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 6  -  Full yearly comparison (Return / Sharpe / MaxDD)")

# Individual strategies shown from their full history; portfolios from common period
all_series_for_yearly = dict(all_standalone)   # full individual histories
all_series_for_yearly["Portfolio_FixedW"]  = bestA_ret
if bestB_ret is not None:
    all_series_for_yearly["Portfolio_LevMR"] = bestB_ret
all_series_for_yearly["Portfolio_DynMom"] = bestC_ret

yr_all = yearly_table(all_series_for_yearly)
all_names = list(all_series_for_yearly.keys())

# Print a compact table grouped by metric
print("\n  YEARLY RETURN")
ret_cols = [f"{n}_Ret" for n in all_names if f"{n}_Ret" in yr_all.columns]
ret_df = yr_all[ret_cols].copy()
ret_df.columns = [c.replace("_Ret", "") for c in ret_df.columns]
_fmt = getattr(ret_df, "applymap", None) or getattr(ret_df, "map", None)
print(ret_df.map(lambda x: f"{x:+.2%}" if pd.notna(x) else "  N/A").to_string())

print("\n  YEARLY SHARPE")
sh_cols = [f"{n}_Sharpe" for n in all_names if f"{n}_Sharpe" in yr_all.columns]
sh_df = yr_all[sh_cols].copy()
sh_df.columns = [c.replace("_Sharpe", "") for c in sh_df.columns]
print(sh_df.map(lambda x: f"{x:>6.3f}" if pd.notna(x) else "   N/A").to_string())

print("\n  YEARLY MAX DRAWDOWN")
mdd_cols = [f"{n}_MDD" for n in all_names if f"{n}_MDD" in yr_all.columns]
mdd_df = yr_all[mdd_cols].copy()
mdd_df.columns = [c.replace("_MDD", "") for c in mdd_df.columns]
print(mdd_df.map(lambda x: f"{x:.2%}" if pd.notna(x) else "  N/A").to_string())


# ══════════════════════════════════════════════════════════════════════════
# STEP 7  -  Full-period summary table
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 7  -  Full-period summary")

rows_sum = []
for name, s in all_series_for_yearly.items():
    w   = (1 + s).cumprod(); w = w / w.iloc[0]
    yr0 = s.index[0].year
    yr1 = s.index[-1].year
    rows_sum.append({
        "Strategy":   name,
        "Start":      str(s.index[0].date()),
        "End":        str(s.index[-1].date()),
        "Days":       len(s),
        "Total Return": float(w.iloc[-1] - 1),
        "Sharpe":     _sharpe(s),
        "Sortino":    _sortino(s),
        "Max DD":     _mdd(s),
    })

summary_df = pd.DataFrame(rows_sum).set_index("Strategy")
print(summary_df.to_string(
    float_format=lambda x: f"{x:+.3f}" if abs(x) < 100 else f"{x:.1f}"
))


# ══════════════════════════════════════════════════════════════════════════
# STEP 8  -  Charts
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 8  -  Generating charts")

# Chart 1: Cumulative wealth  -  all strategies + portfolios
fig, axes = plt.subplots(3, 1, figsize=(18, 16),
                          gridspec_kw={"height_ratios": [3, 2, 1.5]})

color_map = {
    "Momentum":        "steelblue",
    "IntradayMR":      "seagreen",
    "QQQBubble":       "darkorange",
    "Reddit":          "mediumpurple",
    "Portfolio_FixedW":"crimson",
    "Portfolio_LevMR": "chocolate",
    "Portfolio_DynMom":"navy",
}

ax = axes[0]
for name, s in all_series_for_yearly.items():
    w   = (1 + s).cumprod(); w = w / w.iloc[0]
    lw  = 3 if "Portfolio" in name else 1.5
    col = color_map.get(name, "gray")
    ax.plot(w.index, w.values, label=name, color=col, linewidth=lw)
ax.set_title("4-Strategy Portfolio Backtest  -  Cumulative Wealth", fontsize=14)
ax.set_ylabel("Wealth (rebased to 1)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.4)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))

# Chart 2: Yearly returns heatmap-style bar chart
ax2 = axes[1]
port_series = {k: v for k, v in all_series_for_yearly.items() if "Portfolio" in k}
all_years   = sorted({y for s in port_series.values() for y in s.index.year.unique()})
n_ports     = len(port_series)
width       = 0.8 / n_ports
offsets     = np.linspace(-(n_ports-1)/2, (n_ports-1)/2, n_ports) * width

for j, (name, s) in enumerate(port_series.items()):
    rets = []
    for yr in all_years:
        yr_s = s[s.index.year == yr]
        rets.append(float((1+yr_s).prod()-1) if not yr_s.empty else 0.0)
    bars = ax2.bar(
        np.arange(len(all_years)) + offsets[j], rets,
        width=width, label=name, color=color_map.get(name, "gray"), alpha=0.85
    )
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_xticks(range(len(all_years)))
ax2.set_xticklabels(all_years)
ax2.set_ylabel("Annual Return")
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax2.set_title("Portfolio Annual Returns by Year")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis="y")

# Chart 3: Drawdown for best portfolio
ax3 = axes[2]
for name, s in port_series.items():
    w   = (1 + s).cumprod(); w = w / w.iloc[0]
    dd  = (w / w.cummax() - 1)
    ax3.fill_between(dd.index, dd.values, 0,
                     alpha=0.35, color=color_map.get(name, "gray"), label=name)
ax3.set_ylabel("Drawdown")
ax3.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Chart 2: Individual strategy + portfolio yearly Sharpe heat table
fig2, ax = plt.subplots(figsize=(max(12, len(all_years)*1.6), len(all_names)*0.8 + 2))
ax.axis("off")

cols_sh = ["Year"] + all_years
all_yr_sharpe = []
for name in all_names:
    row = [name]
    for yr in all_years:
        s = all_series_for_yearly[name]
        yr_s = s[s.index.year == yr]
        if yr_s.empty or yr_s.std() == 0:
            row.append("N/A")
        else:
            sh = _sharpe(yr_s)
            row.append(f"{sh:.2f}" if pd.notna(sh) else "N/A")
    all_yr_sharpe.append(row)

tbl = ax.table(
    cellText=all_yr_sharpe,
    colLabels=cols_sh,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 1.8)

# Color code cells
for (row_i, col_j), cell in tbl.get_celld().items():
    if row_i == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    elif col_j == 0:
        cell.set_facecolor("#ecf0f1")
        cell.set_text_props(fontweight="bold")
    else:
        try:
            v = float(cell.get_text().get_text())
            if v > 1.5:
                cell.set_facecolor("#2ecc71")
            elif v > 0.5:
                cell.set_facecolor("#a8e6cf")
            elif v > 0:
                cell.set_facecolor("#ffeaa7")
            elif v > -0.5:
                cell.set_facecolor("#fab1a0")
            else:
                cell.set_facecolor("#e17055")
        except (ValueError, TypeError):
            cell.set_facecolor("#dfe6e9")

ax.set_title("Yearly Sharpe by Strategy (green=good, red=bad)", pad=20, fontsize=13)
plt.tight_layout()
plt.show()

# Chart 3: Yearly return comparison table
fig3, ax = plt.subplots(figsize=(max(12, len(all_years)*1.6), len(all_names)*0.8 + 2))
ax.axis("off")

all_yr_ret = []
for name in all_names:
    row = [name]
    for yr in all_years:
        s = all_series_for_yearly[name]
        yr_s = s[s.index.year == yr]
        if yr_s.empty:
            row.append("N/A")
        else:
            ret = float((1+yr_s).prod()-1)
            row.append(f"{ret:+.1%}")
    all_yr_ret.append(row)

tbl2 = ax.table(
    cellText=all_yr_ret,
    colLabels=cols_sh,
    cellLoc="center",
    loc="center",
)
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(9)
tbl2.scale(1.2, 1.8)

for (row_i, col_j), cell in tbl2.get_celld().items():
    if row_i == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    elif col_j == 0:
        cell.set_facecolor("#ecf0f1")
        cell.set_text_props(fontweight="bold")
    else:
        txt = cell.get_text().get_text().replace("+","").replace("%","")
        try:
            v = float(txt) / 100
            if v > 0.30:
                cell.set_facecolor("#2ecc71")
            elif v > 0.10:
                cell.set_facecolor("#a8e6cf")
            elif v > 0:
                cell.set_facecolor("#ffeaa7")
            elif v > -0.10:
                cell.set_facecolor("#fab1a0")
            else:
                cell.set_facecolor("#e17055")
        except (ValueError, TypeError):
            cell.set_facecolor("#dfe6e9")

ax.set_title("Yearly Return by Strategy", pad=20, fontsize=13)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════
# STEP 9  -  Save to Excel
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 9  -  Saving results to Excel")

xl_path = "results/4strategy_backtest.xlsx"
try:
    with pd.ExcelWriter(xl_path, engine="openpyxl") as writer:

        # Sheet 1: Summary
        summary_df.reset_index().to_excel(writer, sheet_name="Summary", index=False)

        # Sheet 2: Yearly Return
        ret_df.reset_index().to_excel(writer, sheet_name="Yearly_Return", index=False)

        # Sheet 3: Yearly Sharpe
        sh_df.reset_index().to_excel(writer, sheet_name="Yearly_Sharpe", index=False)

        # Sheet 4: Yearly MaxDD
        mdd_df.reset_index().to_excel(writer, sheet_name="Yearly_MaxDD", index=False)

        # Sheet 5: Fixed weight grid
        dfA.to_excel(writer, sheet_name="Grid_FixedWeight", index=False)

        # Sheet 6: Leveraged MR grid
        if rows_B:
            dfB.to_excel(writer, sheet_name="Grid_LevMR", index=False)

        # Sheet 7: Dynamic momentum grid
        grid_C.to_excel(writer, sheet_name="Grid_DynMom", index=False)

        # Sheet 8: Daily returns
        daily_rets = pd.DataFrame(all_series_for_yearly)
        daily_rets.index.name = "Date"
        daily_rets.to_excel(writer, sheet_name="Daily_Returns")

    print(f"  Saved: {xl_path}")
except Exception as e:
    print(f"  Excel save failed: {e}")

_header("DONE")
print(f"\nActive strategies: {strategy_names}")
print(f"\nOption A (best fixed):   {w_desc_A}")
print(f"  Sharpe={bestA['Sharpe']:.3f}  Return={bestA['Return']:+.1%}  MaxDD={bestA['Max_DD']:.1%}")

if rows_B:
    bB = dfB.iloc[0]
    print(f"\nOption B (best levered):  lev={bB['leverage']:.1f}x  "
          + "  ".join(f"w_{n}={bB[f'w_{n}']:.0%}" for n in strategy_names))
    print(f"  Sharpe={bB['Sharpe']:.3f}  Return={bB['Return']:+.1%}  MaxDD={bB['Max_DD']:.1%}")

print(f"\nOption C (best dynamic):  lookback={p.get('lookback','-')}d  "
      f"hold={p.get('hold_period','-')}d  max_alloc={float(p.get('max_alloc',1.0)):.0%}")
print(f"  Sharpe={p['Sharpe']:.3f}  Return={p['Total Return']:+.1%}  MaxDD={p['Max DD']:.1%}")
print()
