"""
3-Strategy Combined Portfolio
==============================
Strategies:
  1. Momentum + UVXY Hedge + Leverage (daily)
  2. Reddit Sentiment Bubble (daily)
  3. Intraday Mean-Reversion + Flip (1h signal, best params)

Two portfolio options:
  Option 1 — Fixed weight grid search (step=0.2, all 3 strategies)
  Option 2 — Leverage on Intraday MR (grid 1-3x, 12%/yr cost),
             then fixed-weight combine with Momentum + Reddit

Usage: python run_combined_portfolio.py
"""
import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_n = [0]
def _save(*a, **k):
    _n[0] += 1
    plt.savefig(f"combined_portfolio_{_n[0]}.png", dpi=130, bbox_inches="tight")
    print(f"[Chart saved: combined_portfolio_{_n[0]}.png]")
plt.show = _save

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from itertools import product

from data.db.schema import init
from data.universe import get_universe
from data.intraday_loader import load_daily_close, load_hourly_bars
from strategies.intraday_mean_reversion import run_intraday_mean_reversion
from run_portfolio import get_momentum_returns, get_reddit_returns, START

# ── Intraday MR best params ────────────────────────────────────────────────
MR_PARAMS = dict(
    sigma_grid          = [4.0],
    flip_hold_days_grid = [2],
    lookback_grid       = [20],
    top_n_grid          = [5],
    transaction_cost    = 0.001,
    short_borrow_rate   = 0.08,
)

# ── Leverage grid for option 2 ─────────────────────────────────────────────
LEVERAGE_GRID      = [1.0, 1.5, 2.0, 2.5, 3.0]
LEVERAGE_COST_ANNUAL = 0.12          # 12% annual cost on borrowed amount
LEVERAGE_COST_DAILY  = LEVERAGE_COST_ANNUAL / 252

WEIGHT_STEP = 0.2    # fixed-weight grid granularity


# ── Helpers ────────────────────────────────────────────────────────────────

def _sharpe(r: pd.Series, td: int = 252) -> float:
    s = r.std(); return float(np.sqrt(td) * r.mean() / s) if s > 0 else np.nan

def _sortino(r: pd.Series, td: int = 252) -> float:
    ds = r[r < 0].std(); return float(np.sqrt(td) * r.mean() / ds) if ds > 0 else np.nan

def _stats(r: pd.Series) -> dict:
    w   = (1 + r).cumprod(); w = w / w.iloc[0]
    mdd = float((w / w.cummax() - 1).min())
    return dict(Sharpe=_sharpe(r), Sortino=_sortino(r),
                Return=float(w.iloc[-1]-1), Max_DD=mdd)

def _weight_grid(names: list, step: float) -> list[dict]:
    """All weight combos (sum=1) with given step for N strategies."""
    vals = np.arange(0, 1+step, step).round(2)
    combos = []
    for c in product(vals, repeat=len(names)):
        if abs(sum(c) - 1.0) < 1e-9:
            combos.append(dict(zip(names, c)))
    return combos

def _apply_leverage(ret: pd.Series, lev: float) -> pd.Series:
    """Scale returns by leverage and deduct daily borrowing cost."""
    daily_cost = (lev - 1) * LEVERAGE_COST_DAILY
    return lev * ret - daily_cost

def _print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ── Get strategy returns ───────────────────────────────────────────────────

init()
print("=" * 70)
print("STEP 1 — Loading strategy returns")
print("=" * 70)

print("\n[1/3] Momentum strategy...")
mom_ret = get_momentum_returns(START)

print("\n[2/3] Reddit Sentiment strategy...")
reddit_ret = get_reddit_returns(START)

print("\n[3/3] Intraday MR strategy (best params: σ=4, flip=2d, lb=20d)...")
universe = get_universe()
daily_close = load_daily_close(universe, use_cache=True)
hourly_open, hourly_close = load_hourly_bars(universe, use_cache=True)
mr_ret_raw, mr_params, _ = run_intraday_mean_reversion(
    daily_close=daily_close,
    hourly_open=hourly_open,
    hourly_close=hourly_close,
    **MR_PARAMS,
)
mr_ret = mr_ret_raw.rename("IntradayMR")

# Align to common date range
common_idx = mom_ret.index.intersection(reddit_ret.index).intersection(mr_ret.index)
mom_ret    = mom_ret.loc[common_idx]
reddit_ret = reddit_ret.loc[common_idx]
mr_ret     = mr_ret.loc[common_idx]

print(f"\nCommon period: {common_idx[0].date()} to {common_idx[-1].date()} ({len(common_idx)} days)")
print(f"  Momentum  Sharpe: {_sharpe(mom_ret):.3f}  Return: {(1+mom_ret).prod()-1:+.1%}")
print(f"  Reddit    Sharpe: {_sharpe(reddit_ret):.3f}  Return: {(1+reddit_ret).prod()-1:+.1%}")
print(f"  IntradayMR Sharpe: {_sharpe(mr_ret):.3f}  Return: {(1+mr_ret).prod()-1:+.1%}")


# ══════════════════════════════════════════════════════════════════════════
# OPTION 1 — Fixed-weight grid search across all 3 strategies
# ══════════════════════════════════════════════════════════════════════════

_print_header("OPTION 1: Fixed Weight Grid Search (step=20%)")

names   = ["Momentum", "Reddit", "IntradayMR"]
combos  = _weight_grid(names, WEIGHT_STEP)
print(f"Testing {len(combos)} weight combinations...\n")

rows_opt1 = []
for w in combos:
    port = (mom_ret    * w["Momentum"]
          + reddit_ret * w["Reddit"]
          + mr_ret     * w["IntradayMR"])
    s = _stats(port)
    rows_opt1.append({**w, **s})

df1 = pd.DataFrame(rows_opt1).sort_values("Sharpe", ascending=False)

print(f"{'w_Mom':>6} {'w_Red':>6} {'w_MR':>6} | "
      f"{'Sharpe':>7} {'Sortino':>8} {'Return':>8} {'Max_DD':>8}")
print("-" * 60)
for _, r in df1.head(15).iterrows():
    print(f"{r['Momentum']:>6.0%} {r['Reddit']:>6.0%} {r['IntradayMR']:>6.0%} | "
          f"{r['Sharpe']:>7.3f} {r['Sortino']:>8.3f} {r['Return']:>8.1%} {r['Max_DD']:>8.1%}")

best1 = df1.iloc[0]
best1_ret = (mom_ret    * best1["Momentum"]
           + reddit_ret * best1["Reddit"]
           + mr_ret     * best1["IntradayMR"])


# ══════════════════════════════════════════════════════════════════════════
# OPTION 2 — Leverage on IntradayMR (12%/yr cost) + fixed-weight combine
# ══════════════════════════════════════════════════════════════════════════

_print_header("OPTION 2: Leverage on IntradayMR (12%/yr) + Fixed-Weight Combine")

rows_opt2 = []
best2_sharpe = -np.inf
best2_ret    = None
best2_label  = ""

print(f"Leverage cost: {LEVERAGE_COST_ANNUAL:.0%}/yr = {LEVERAGE_COST_DAILY:.6f}/day\n")
print(f"{'Lev':>4} {'w_Mom':>6} {'w_Red':>6} {'w_MR':>6} | "
      f"{'Sharpe':>7} {'Sortino':>8} {'Return':>8} {'Max_DD':>8}")
print("-" * 65)

for lev in LEVERAGE_GRID:
    lev_mr = _apply_leverage(mr_ret, lev)
    lev_stats = _stats(lev_mr)
    print(f"\n  Leveraged MR standalone ({lev:.1f}x): "
          f"Sharpe={lev_stats['Sharpe']:.3f}  "
          f"Return={lev_stats['Return']:+.1%}  "
          f"Max_DD={lev_stats['Max_DD']:.1%}")

    for w in _weight_grid(["Momentum", "Reddit", "IntradayMR"], WEIGHT_STEP):
        port = (mom_ret    * w["Momentum"]
              + reddit_ret * w["Reddit"]
              + lev_mr     * w["IntradayMR"])
        s = _stats(port)
        rows_opt2.append({"leverage": lev, **w, **s})

        if pd.notna(s["Sharpe"]) and s["Sharpe"] > best2_sharpe:
            best2_sharpe = s["Sharpe"]
            best2_ret    = port.rename("Portfolio")
            best2_label  = (f"lev={lev:.1f}x  "
                           f"w_Mom={w['Momentum']:.0%}  "
                           f"w_Red={w['Reddit']:.0%}  "
                           f"w_MR={w['IntradayMR']:.0%}")

df2 = pd.DataFrame(rows_opt2).sort_values("Sharpe", ascending=False)

_print_header("OPTION 2 — Top 15 by Sharpe")
print(f"{'Lev':>4} {'w_Mom':>6} {'w_Red':>6} {'w_MR':>6} | "
      f"{'Sharpe':>7} {'Sortino':>8} {'Return':>8} {'Max_DD':>8}")
print("-" * 65)
for _, r in df2.head(15).iterrows():
    print(f"{r['leverage']:>4.1f} {r['Momentum']:>6.0%} {r['Reddit']:>6.0%} {r['IntradayMR']:>6.0%} | "
          f"{r['Sharpe']:>7.3f} {r['Sortino']:>8.3f} {r['Return']:>8.1%} {r['Max_DD']:>8.1%}")


# ══════════════════════════════════════════════════════════════════════════
# Chart: compare individual strategies + both portfolio options
# ══════════════════════════════════════════════════════════════════════════

all_series = {
    "Momentum":        mom_ret,
    "Reddit":          reddit_ret,
    "IntradayMR":      mr_ret,
    "Option1 (best fixed)": best1_ret,
    "Option2 (best levered)": best2_ret,
}
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                gridspec_kw={"height_ratios": [3, 1]})

styles = {"Momentum": ("steelblue",1.2), "Reddit": ("darkorange",1.2),
          "IntradayMR": ("green",1.2),
          "Option1 (best fixed)": ("purple",2.5),
          "Option2 (best levered)": ("crimson",2.5)}

for name, ret in all_series.items():
    if ret is None: continue
    w   = (1 + ret).cumprod(); w = w / w.iloc[0]
    col, lw = styles[name]
    ax1.plot(w.index, w.values, label=name, color=col, linewidth=lw)

ax1.set_title("3-Strategy Portfolio | Option 1: Fixed Weights | Option 2: Levered IntradayMR")
ax1.set_ylabel("Cumulative Wealth"); ax1.legend(fontsize=9); ax1.grid(True)

# Drawdown of best option 2
if best2_ret is not None:
    w2 = (1 + best2_ret).cumprod(); w2 = w2 / w2.iloc[0]
    dd = (w2 / w2.cummax() - 1)
    ax2.fill_between(dd.index, dd.values, 0, alpha=0.4, color="crimson",
                     label=f"Option2 DD ({best2_label})")
    ax2.set_ylabel("Drawdown"); ax2.legend(fontsize=8); ax2.grid(True)

plt.tight_layout()
plt.show()

# ── Summary ────────────────────────────────────────────────────────────────
_print_header("SUMMARY")
print(f"\nOption 1 best: {best1['Momentum']:.0%} Mom + {best1['Reddit']:.0%} Reddit + "
      f"{best1['IntradayMR']:.0%} IntradayMR")
print(f"  Sharpe={best1['Sharpe']:.3f}  Sortino={best1['Sortino']:.3f}  "
      f"Return={best1['Return']:+.1%}  Max_DD={best1['Max_DD']:.1%}")

b2 = df2.iloc[0]
print(f"\nOption 2 best: {b2['leverage']:.1f}x lev on MR | "
      f"{b2['Momentum']:.0%} Mom + {b2['Reddit']:.0%} Reddit + {b2['IntradayMR']:.0%} MR")
print(f"  Sharpe={b2['Sharpe']:.3f}  Sortino={b2['Sortino']:.3f}  "
      f"Return={b2['Return']:+.1%}  Max_DD={b2['Max_DD']:.1%}")

print(f"\nLeverage cost: 12%/yr × (leverage-1) deducted daily from leveraged MR returns")
