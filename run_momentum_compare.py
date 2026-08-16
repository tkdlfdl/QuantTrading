"""
Comparison: Hourly Momentum (Long-Only) vs Daily Momentum (Long-Only)
======================================================================
Strategy A -- Hourly Momentum (Long-Only)
  Universe  : NASDAQ100 + S&P500 (516 tickers)
  Signal    : 300-hour cumulative close-to-close return (best from grid)
  Entry     : Top-20 stocks, long at open of next hour
  Hold      : 20 hours
  TC        : 0.1% per trade (included in grid results)

Strategy B -- Daily Cross-Sectional Momentum (Long-Only)
  Universe  : NASDAQ100 + S&P500 (516 tickers, daily close cache)
  Signal    : 140-day cumulative return
  Entry     : Top-N stocks, equal weight, rebalance every 40 days
  Hold      : 40 trading days
  TC        : 0.1% per position that changes on each rebalance

Comparison period : common overlap of both strategies
Metrics           : Sharpe, Sortino, Total Return, Max DD, yearly breakdown
"""
import sys, warnings, os
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_n = [0]
def _save(*a, **k):
    _n[0] += 1; p = f"results/compare_chart_{_n[0]}.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); print(f"  [chart saved: {p}]", flush=True)
plt.show = _save

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from data.db.schema import init
from data.universe import get_universe

os.makedirs("results", exist_ok=True)

TRADING_DAYS = 252
TC_RATE      = 0.001   # 0.1% one-way


def _sharpe(r, td=TRADING_DAYS):
    s = r.std(); return float(np.sqrt(td)*r.mean()/s) if s > 0 else np.nan

def _sortino(r, td=TRADING_DAYS):
    ds = r[r<0].std(); return float(np.sqrt(td)*r.mean()/ds) if ds > 0 else np.nan

def _mdd(r):
    w = (1+r).cumprod(); w = w/w.iloc[0]
    return float((w/w.cummax()-1).min())

def _stats(r):
    w = (1+r).cumprod(); w = w/w.iloc[0]
    return {"Sharpe": _sharpe(r), "Sortino": _sortino(r),
            "Total Return": float(w.iloc[-1]-1), "Max DD": _mdd(r)}

def _header(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}", flush=True)

def print_yearly(name, r):
    print(f"\n  [{name}]")
    for yr, grp in r.groupby(r.index.year):
        if len(grp) < 10: continue
        w = (1+grp).cumprod(); w = w/w.iloc[0]
        print(f"    {yr}: Return={float(w.iloc[-1]-1):>+8.2%}  "
              f"Sharpe={_sharpe(grp):>6.3f}  MaxDD={float((w/w.cummax()-1).min()):>8.2%}  "
              f"Days={len(grp)}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 -- Load Strategy A (hourly momentum, long-only, best params)
# ══════════════════════════════════════════════════════════════════════════

init()
_header("Strategy A -- Hourly Momentum Long-Only (best grid params)")

xl_h   = pd.ExcelFile("results/hourly_momentum.xlsx")
dr_h   = pd.read_excel(xl_h, "Daily_Returns", index_col=0, parse_dates=True)
dr_h.index = pd.to_datetime(dr_h.index)
hourly_ret = dr_h["Long"].rename("HourlyMom_Long")

# Also load grid to confirm best params
grid_long = pd.read_excel(xl_h, "Grid_Long")
best_long = grid_long.dropna(subset=["Sharpe"]).iloc[0]
print(f"Best params: lookback={int(best_long['lookback'])}h  "
      f"hold={int(best_long['hold'])}h  top_n={int(best_long['top_n'])}")
print(f"Full period: {hourly_ret.index[0].date()} to {hourly_ret.index[-1].date()} "
      f"({len(hourly_ret)} days)")
s = _stats(hourly_ret)
print(f"Sharpe={s['Sharpe']:.3f}  Sortino={s['Sortino']:.3f}  "
      f"Return={s['Total Return']:+.1%}  MaxDD={s['Max DD']:.1%}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 -- Build Strategy B (daily momentum, lb=140d, hold=40d)
# ══════════════════════════════════════════════════════════════════════════

_header("Strategy B -- Daily Momentum (lb=140d, hold=40d, TC=0.1%)")

# Load daily close prices
print("Loading daily close cache...", flush=True)
from pathlib import Path
daily_close = pd.read_parquet(Path("data/cache/daily_close.parquet"))
daily_close.index = pd.to_datetime(daily_close.index)
print(f"Daily cache: {daily_close.index[0].date()} to {daily_close.index[-1].date()} "
      f"| {len(daily_close)} days x {len(daily_close.columns)} tickers")

LOOKBACK_D = 140   # trading days
HOLD_D     = 40    # trading days

# Run over a grid of top_n values so we can compare fairly
TOP_N_DAILY = [5, 10, 20, 30]

daily_grid  = []
best_d_ret  = None
best_d_sh   = -np.inf
best_d_tn   = None

for top_n in TOP_N_DAILY:
    # Compute 140-day momentum at each date
    ret_140  = daily_close.pct_change(LOOKBACK_D)   # close[t]/close[t-140] - 1
    port_rows = []
    prev_holdings = set()

    i = LOOKBACK_D
    while i < len(daily_close) - 1:
        date    = daily_close.index[i]
        mom_row = ret_140.iloc[i].dropna()

        # Select top-N by 140d momentum
        if len(mom_row) < top_n:
            i += HOLD_D; continue

        top_stocks = mom_row.nlargest(top_n).index.tolist()

        # Transaction cost: 0.1% on each position that enters OR exits
        entering = set(top_stocks) - prev_holdings
        exiting  = prev_holdings  - set(top_stocks)
        n_changes = len(entering) + len(exiting)
        tc_cost   = TC_RATE * n_changes / top_n   # per-portfolio TC

        # Hold period: next HOLD_D trading days
        hold_end = min(i + HOLD_D, len(daily_close) - 1)
        hold_dates = daily_close.index[i+1 : hold_end+1]

        # Equal-weight daily returns
        port_daily_ret = (daily_close[top_stocks]
                          .pct_change()
                          .iloc[i+1 : hold_end+1]
                          .mean(axis=1))

        # Deduct TC on first day of hold period
        if len(port_daily_ret) > 0:
            port_daily_ret.iloc[0] -= tc_cost

        for d, rv in zip(hold_dates, port_daily_ret):
            port_rows.append({"date": d, "ret": rv})

        prev_holdings = set(top_stocks)
        i += HOLD_D

    if not port_rows:
        continue

    port_ret = (pd.DataFrame(port_rows)
                  .set_index("date")["ret"]
                  .groupby(level=0).mean()   # deduplicate any overlapping dates
                  .reindex(daily_close.index[LOOKBACK_D:], fill_value=0.0)
                  .dropna())
    port_ret.name = f"DailyMom_n{top_n}"

    s_d = _stats(port_ret)
    row = {"top_n": top_n, "lookback": LOOKBACK_D, "hold": HOLD_D, **s_d}
    daily_grid.append(row)

    if pd.notna(s_d["Sharpe"]) and s_d["Sharpe"] > best_d_sh:
        best_d_sh  = s_d["Sharpe"]
        best_d_ret = port_ret.rename("DailyMom_LongOnly")
        best_d_tn  = top_n

grid_d_df = pd.DataFrame(daily_grid)
print(f"\nDaily momentum grid (lb={LOOKBACK_D}d, hold={HOLD_D}d):")
print(f"{'top_n':>7} | {'Sharpe':>7} {'Sortino':>8} {'Return':>9} {'Max_DD':>8}")
print("-" * 45)
for _, r in grid_d_df.iterrows():
    print(f"{int(r['top_n']):>7} | {r['Sharpe']:>7.3f} {r['Sortino']:>8.3f} "
          f"{r['Total Return']:>9.2%} {r['Max DD']:>8.2%}")

print(f"\nBest daily: top_n={best_d_tn}  Sharpe={best_d_sh:.3f}")
print(f"Full period: {best_d_ret.index[0].date()} to {best_d_ret.index[-1].date()} "
      f"({len(best_d_ret)} days)")


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 -- Align to common period and compare
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 3 -- Head-to-head comparison (common period)")

common = hourly_ret.index.intersection(best_d_ret.index)
h_ret  = hourly_ret.loc[common]
d_ret  = best_d_ret.loc[common]

print(f"Common period: {common[0].date()} to {common[-1].date()} ({len(common)} days)")
print()

strategies = {
    f"Hourly Momentum\n(lb={int(best_long['lookback'])}h hold={int(best_long['hold'])}h n={int(best_long['top_n'])} TC=0.1%)": h_ret,
    f"Daily Momentum\n(lb={LOOKBACK_D}d hold={HOLD_D}d n={best_d_tn} TC=0.1%)": d_ret,
}

# Full metrics table
print(f"{'Metric':<22} {'Hourly Mom':>14} {'Daily Mom':>14}")
print("-" * 52)
for metric in ["Sharpe", "Sortino", "Total Return", "Max DD"]:
    hv = _stats(h_ret)[metric]
    dv = _stats(d_ret)[metric]
    if metric in ("Total Return", "Max DD"):
        print(f"{metric:<22} {hv:>14.2%} {dv:>14.2%}")
    else:
        print(f"{metric:<22} {hv:>14.3f} {dv:>14.3f}")


# ── Yearly breakdown ───────────────────────────────────────────────────────
_header("Yearly breakdown (common period)")

all_yrs = sorted(set(h_ret.index.year.unique()) | set(d_ret.index.year.unique()))
print(f"\n{'Year':>6} | {'H_Return':>10} {'H_Sharpe':>10} {'H_MaxDD':>9} | "
      f"{'D_Return':>10} {'D_Sharpe':>10} {'D_MaxDD':>9}")
print("-" * 76)
for yr in all_yrs:
    hg = h_ret[h_ret.index.year == yr]
    dg = d_ret[d_ret.index.year == yr]
    if len(hg) < 5 or len(dg) < 5: continue
    hw  = (1+hg).cumprod(); hw=hw/hw.iloc[0]
    dw  = (1+dg).cumprod(); dw=dw/dw.iloc[0]
    print(f"{yr:>6} | {float(hw.iloc[-1]-1):>+10.2%} {_sharpe(hg):>10.3f} "
          f"{float((hw/hw.cummax()-1).min()):>9.2%} | "
          f"{float(dw.iloc[-1]-1):>+10.2%} {_sharpe(dg):>10.3f} "
          f"{float((dw/dw.cummax()-1).min()):>9.2%}")


# ── Also show full history for daily momentum ──────────────────────────────
_header("Daily Momentum -- full history (2018-2026)")
print_yearly("DailyMom_Full", best_d_ret)


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 -- Charts
# ══════════════════════════════════════════════════════════════════════════

_header("Charts")

colors = {"Hourly": "steelblue", "Daily": "crimson"}

# Chart 1: Cumulative wealth comparison (common period)
fig, axes = plt.subplots(3, 1, figsize=(18, 14),
                          gridspec_kw={"height_ratios": [3, 1.5, 1.5]})

ax1 = axes[0]
for label, s, col in [
    (f"Hourly Mom (lb={int(best_long['lookback'])}h hold={int(best_long['hold'])}h n={int(best_long['top_n'])})",
     h_ret, colors["Hourly"]),
    (f"Daily Mom (lb={LOOKBACK_D}d hold={HOLD_D}d n={best_d_tn})",
     d_ret, colors["Daily"]),
]:
    w = (1+s).cumprod(); w = w/w.iloc[0]
    ax1.plot(w.index, w.values, label=label, color=col, linewidth=2)

ax1.set_title("Hourly Momentum vs Daily Momentum (Long-Only, TC=0.1%)", fontsize=13)
ax1.set_ylabel("Cumulative Wealth (rebased to 1)")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.1f}x"))
ax1.legend(fontsize=10); ax1.grid(True, alpha=0.4)

# Drawdown
ax2 = axes[1]
for label, s, col in [("Hourly", h_ret, colors["Hourly"]),
                       ("Daily",  d_ret, colors["Daily"])]:
    w = (1+s).cumprod(); w=w/w.iloc[0]; dd = w/w.cummax()-1
    ax2.fill_between(dd.index, dd.values, 0, alpha=0.4, color=col, label=label)
ax2.axhline(0, color="black", lw=0.8)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax2.set_ylabel("Drawdown"); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.4)

# Rolling 60-day Sharpe
ax3 = axes[2]
for label, s, col in [("Hourly", h_ret, colors["Hourly"]),
                       ("Daily",  d_ret, colors["Daily"])]:
    roll = s.rolling(60).apply(
        lambda x: float(np.sqrt(252)*x.mean()/x.std()) if x.std()>0 else np.nan)
    ax3.plot(roll.index, roll.values, label=label, color=col, linewidth=1.5)
ax3.axhline(0, color="black", lw=0.8, linestyle="--")
ax3.set_ylabel("60-day Rolling Sharpe"); ax3.legend(fontsize=9); ax3.grid(True, alpha=0.4)

plt.tight_layout(); plt.show()

# Chart 2: Full daily momentum history (2018-2026)
fig2, axes2 = plt.subplots(2, 1, figsize=(18, 10),
                             gridspec_kw={"height_ratios": [3, 1]})
w_full = (1+best_d_ret).cumprod(); w_full = w_full/w_full.iloc[0]
axes2[0].plot(w_full.index, w_full.values, color=colors["Daily"], linewidth=1.8,
               label=f"Daily Momentum (lb={LOOKBACK_D}d hold={HOLD_D}d n={best_d_tn})")
axes2[0].set_title(f"Daily Momentum Full History (2018-2026) | "
                   f"Sharpe={_sharpe(best_d_ret):.3f}  "
                   f"Return={(1+best_d_ret).prod()-1:+.1%}  "
                   f"MaxDD={_mdd(best_d_ret):.1%}")
axes2[0].set_ylabel("Cumulative Wealth")
axes2[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.1f}x"))
axes2[0].legend(fontsize=10); axes2[0].grid(True, alpha=0.4)
dd_full = w_full/w_full.cummax()-1
axes2[1].fill_between(dd_full.index, dd_full.values, 0, alpha=0.5, color=colors["Daily"])
axes2[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
axes2[1].set_ylabel("Drawdown"); axes2[1].grid(True, alpha=0.4)
plt.tight_layout(); plt.show()

# Chart 3: Yearly bar chart side by side
yearly_data = {}
for yr in all_yrs:
    hg = h_ret[h_ret.index.year == yr]; dg = d_ret[d_ret.index.year == yr]
    if len(hg) < 5 or len(dg) < 5: continue
    hw = (1+hg).cumprod(); hw=hw/hw.iloc[0]
    dw = (1+dg).cumprod(); dw=dw/dw.iloc[0]
    yearly_data[yr] = {"Hourly": float(hw.iloc[-1]-1), "Daily": float(dw.iloc[-1]-1)}

ydf = pd.DataFrame(yearly_data).T
fig3, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(ydf)); width = 0.35
ax.bar(x-width/2, ydf["Hourly"], width, label="Hourly Mom", color=colors["Hourly"], alpha=0.85)
ax.bar(x+width/2, ydf["Daily"],  width, label="Daily Mom",  color=colors["Daily"],  alpha=0.85)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(ydf.index)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_title("Annual Return: Hourly vs Daily Momentum (common period)")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout(); plt.show()


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 -- Save
# ══════════════════════════════════════════════════════════════════════════

_header("Save to Excel")
xl = "results/momentum_comparison.xlsx"
try:
    with pd.ExcelWriter(xl, engine="openpyxl") as writer:
        # Summary
        rows = []
        for name, s, extra in [
            ("Hourly_Mom_LongOnly", h_ret,
             {"lookback": f"{int(best_long['lookback'])}h",
              "hold": f"{int(best_long['hold'])}h",
              "top_n": int(best_long["top_n"])}),
            ("Daily_Mom_LongOnly", d_ret,
             {"lookback": f"{LOOKBACK_D}d", "hold": f"{HOLD_D}d", "top_n": best_d_tn}),
            ("Daily_Mom_FullHist", best_d_ret,
             {"lookback": f"{LOOKBACK_D}d", "hold": f"{HOLD_D}d", "top_n": best_d_tn}),
        ]:
            w = (1+s).cumprod(); w=w/w.iloc[0]
            rows.append({"Strategy": name, "Start": str(s.index[0].date()),
                          "End": str(s.index[-1].date()), "Days": len(s),
                          "Total Return": float(w.iloc[-1]-1),
                          "Sharpe": _sharpe(s), "Sortino": _sortino(s),
                          "Max DD": _mdd(s), "TC": "0.1% per trade", **extra})
        pd.DataFrame(rows).to_excel(writer, sheet_name="Summary", index=False)

        # Yearly comparison
        yr_rows = []
        for yr in all_yrs:
            hg = h_ret[h_ret.index.year==yr]; dg = d_ret[d_ret.index.year==yr]
            dg_full = best_d_ret[best_d_ret.index.year==yr]
            row = {"Year": yr}
            for tag, g in [("Hourly",hg),("Daily_Common",dg),("Daily_Full",dg_full)]:
                if len(g) < 5:
                    row[f"{tag}_Ret"]=np.nan; row[f"{tag}_Sharpe"]=np.nan; row[f"{tag}_MDD"]=np.nan
                else:
                    w = (1+g).cumprod(); w=w/w.iloc[0]
                    row[f"{tag}_Ret"]   = float(w.iloc[-1]-1)
                    row[f"{tag}_Sharpe"] = _sharpe(g)
                    row[f"{tag}_MDD"]   = float((w/w.cummax()-1).min())
            yr_rows.append(row)
        pd.DataFrame(yr_rows).set_index("Year").to_excel(writer, sheet_name="Yearly")

        grid_d_df.to_excel(writer, sheet_name="Grid_DailyMom", index=False)

        dr_out = pd.DataFrame({"HourlyMom": h_ret, "DailyMom": d_ret,
                                "DailyMom_Full": best_d_ret})
        dr_out.to_excel(writer, sheet_name="Daily_Returns")

    print(f"  Saved: {xl}", flush=True)
except Exception as e:
    print(f"  Save failed: {e}", flush=True)


_header("FINAL COMPARISON SUMMARY")
print(f"\nCommon period: {common[0].date()} to {common[-1].date()} ({len(common)} days)")
print(f"\n{'Metric':<22} {'Hourly Mom':>16} {'Daily Mom':>16}")
print("-" * 56)
for metric in ["Sharpe","Sortino","Total Return","Max DD"]:
    hv = _stats(h_ret)[metric]; dv = _stats(d_ret)[metric]
    winner = "<-- better" if hv > dv else "         <-- better"
    fmt = ".3f" if metric not in ("Total Return","Max DD") else ".2%"
    print(f"{metric:<22} {hv:>16{fmt}} {dv:>16{fmt}}  {winner}")

print(f"\nDaily Momentum full history (2018-2026):")
print(f"  Sharpe={_sharpe(best_d_ret):.3f}  Return={(1+best_d_ret).prod()-1:+.1%}  "
      f"MaxDD={_mdd(best_d_ret):.1%}")
print(f"\nNote: TC=0.1% applied to BOTH strategies.")
print(f"  Hourly: 0.1% per trade (per-position entry/exit)")
print(f"  Daily:  0.1% per position that changes on each 40-day rebalance")
