"""
Hourly Cross-Sectional Momentum Backtest
==========================================
Universe : NASDAQ100 + S&P500 (merged Alpaca + yfinance hourly cache, 2019-2026)
Signal   : Cumulative close-to-close return over lookback hours
Long     : Top-N stocks by momentum -> LONG next hour, hold H hours
Short    : Bottom-N stocks by momentum -> SHORT next hour, hold H hours
Grids    : Long and short are searched INDEPENDENTLY then combined

Usage:
  python run_hourly_momentum.py
"""
import sys, warnings, os
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_n = [0]
def _save(*a, **k):
    _n[0] += 1; p = f"results/hourly_mom_chart_{_n[0]}.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); print(f"  [chart saved: {p}]", flush=True)
plt.show = _save

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from data.db.schema import init
from data.universe import get_universe
from data.intraday_loader import load_hourly_bars
from strategies.hourly_momentum import run_hourly_momentum, _sharpe, _sortino, _mdd

os.makedirs("results", exist_ok=True)

TRADING_DAYS = 252

# Grid centred on user request (120h lookback, 5h hold)
LOOKBACK_GRID = [60, 120, 200, 300]
HOLD_GRID     = [1, 2, 5, 10, 20]
TOP_N_GRID    = [5, 10, 20, 30]


def _header(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}", flush=True)

def yearly_stats(ret: pd.Series) -> None:
    for yr, grp in ret.groupby(ret.index.year):
        if len(grp) < 10: continue
        w = (1+grp).cumprod(); w=w/w.iloc[0]
        print(f"  {yr}: Return={float(w.iloc[-1]-1):>+8.2%}  "
              f"Sharpe={_sharpe(grp):>6.3f}  MaxDD={float((w/w.cummax()-1).min()):>8.2%}  "
              f"Days={len(grp)}")


# ══════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════

init()
_header("Load hourly data (merged Alpaca + yfinance cache)")

universe = get_universe()
hourly_open, hourly_close = load_hourly_bars(universe, use_cache=True)

print(f"Loaded: {len(hourly_open.columns)} tickers | "
      f"{len(hourly_open)} bars | "
      f"{hourly_open.index[0].date()} to {hourly_open.index[-1].date()}")


# ══════════════════════════════════════════════════════════════════════════
# Run grid search
# ══════════════════════════════════════════════════════════════════════════

_header("Grid search: long and short independently")
print(f"Lookback grid : {LOOKBACK_GRID} hours")
print(f"Hold grid     : {HOLD_GRID} hours")
print(f"Top-N grid    : {TOP_N_GRID}")
print(f"Total combos  : {len(LOOKBACK_GRID)*len(HOLD_GRID)*len(TOP_N_GRID)} per side\n")

results = run_hourly_momentum(
    hourly_open   = hourly_open,
    hourly_close  = hourly_close,
    lookback_grid = LOOKBACK_GRID,
    hold_grid     = HOLD_GRID,
    top_n_grid    = TOP_N_GRID,
)

long_grid    = results["long_grid"]
short_grid   = results["short_grid"]
best_long    = results["best_long"]
best_short   = results["best_short"]
best_long_p  = results["best_long_params"]
best_short_p = results["best_short_params"]
combined_ret = results["combined_ret"]
combo_grid   = results["combined_grid"]


# ══════════════════════════════════════════════════════════════════════════
# Print results
# ══════════════════════════════════════════════════════════════════════════

_header("LONG grid -- Top 20 by Sharpe")
print(f"{'Lookback':>9} {'Hold':>6} {'TopN':>6} | "
      f"{'Sharpe':>7} {'Sortino':>8} {'Return':>9} {'Max_DD':>8} {'nTrades':>8} {'WinRate':>8}")
print("-" * 72)
for _, r in long_grid.dropna(subset=["Sharpe"]).head(20).iterrows():
    print(f"{int(r['lookback']):>9} {int(r['hold']):>6} {int(r['top_n']):>6} | "
          f"{r['Sharpe']:>7.3f} {r['Sortino']:>8.3f} {r['Total_Return']:>9.2%} "
          f"{r['Max_DD']:>8.2%} {int(r['n_trades']):>8} {r['Win_Rate']:>8.2%}")

if best_long_p:
    print(f"\nBest LONG: lookback={best_long_p['lookback']}h  "
          f"hold={best_long_p['hold']}h  top_n={best_long_p['top_n']}")
    print(f"  Sharpe={best_long_p['Sharpe']:.3f}  "
          f"Return={best_long_p['Total_Return']:+.1%}  "
          f"MaxDD={best_long_p['Max_DD']:.1%}  WinRate={best_long_p['Win_Rate']:.1%}")

_header("LONG -- Yearly breakdown (best params)")
if best_long is not None:
    yearly_stats(best_long)


_header("SHORT grid -- Top 20 by Sharpe")
print(f"{'Lookback':>9} {'Hold':>6} {'TopN':>6} | "
      f"{'Sharpe':>7} {'Sortino':>8} {'Return':>9} {'Max_DD':>8} {'nTrades':>8} {'WinRate':>8}")
print("-" * 72)
for _, r in short_grid.dropna(subset=["Sharpe"]).head(20).iterrows():
    print(f"{int(r['lookback']):>9} {int(r['hold']):>6} {int(r['top_n']):>6} | "
          f"{r['Sharpe']:>7.3f} {r['Sortino']:>8.3f} {r['Total_Return']:>9.2%} "
          f"{r['Max_DD']:>8.2%} {int(r['n_trades']):>8} {r['Win_Rate']:>8.2%}")

if best_short_p:
    print(f"\nBest SHORT: lookback={best_short_p['lookback']}h  "
          f"hold={best_short_p['hold']}h  top_n={best_short_p['top_n']}")
    print(f"  Sharpe={best_short_p['Sharpe']:.3f}  "
          f"Return={best_short_p['Total_Return']:+.1%}  "
          f"MaxDD={best_short_p['Max_DD']:.1%}  WinRate={best_short_p['Win_Rate']:.1%}")

_header("SHORT -- Yearly breakdown (best params)")
if best_short is not None:
    yearly_stats(best_short)


_header("COMBINED (best long + best short, equal weight)")
if combined_ret is not None:
    s = combined_ret
    print(f"Period: {s.index[0].date()} to {s.index[-1].date()} ({len(s)} days)")
    print(f"Sharpe={_sharpe(s):.3f}  Sortino={_sortino(s):.3f}  "
          f"Return={(1+s).prod()-1:+.1%}  MaxDD={_mdd(s):.1%}")
    yearly_stats(s)


_header("COMBINED grid (top-5 long x top-5 short params)")
if not combo_grid.empty:
    print(f"{'LB_L':>5} {'H_L':>5} {'N_L':>5} | {'LB_S':>5} {'H_S':>5} {'N_S':>5} | "
          f"{'Sharpe':>7} {'Sortino':>8} {'Return':>9} {'Max_DD':>8}")
    print("-" * 70)
    for _, r in combo_grid.head(15).iterrows():
        print(f"{int(r['lb_long']):>5} {int(r['hold_long']):>5} {int(r['n_long']):>5} | "
              f"{int(r['lb_short']):>5} {int(r['hold_short']):>5} {int(r['n_short']):>5} | "
              f"{r['Sharpe']:>7.3f} {r['Sortino']:>8.3f} "
              f"{r['Total_Return']:>9.2%} {r['Max_DD']:>8.2%}")

    best_combo = combo_grid.iloc[0]
    print(f"\nBest combined:")
    print(f"  Long:  lb={int(best_combo['lb_long'])}h  hold={int(best_combo['hold_long'])}h  "
          f"n={int(best_combo['n_long'])}")
    print(f"  Short: lb={int(best_combo['lb_short'])}h  hold={int(best_combo['hold_short'])}h  "
          f"n={int(best_combo['n_short'])}")
    print(f"  Sharpe={best_combo['Sharpe']:.3f}  Return={best_combo['Total_Return']:+.1%}  "
          f"MaxDD={best_combo['Max_DD']:.1%}")


# ══════════════════════════════════════════════════════════════════════════
# Charts
# ══════════════════════════════════════════════════════════════════════════

_header("Charts")

series_to_plot = {}
if best_long   is not None: series_to_plot["Long"]     = best_long
if best_short  is not None: series_to_plot["Short"]    = best_short
if combined_ret is not None: series_to_plot["Combined"] = combined_ret

if series_to_plot:
    fig, axes = plt.subplots(3, 1, figsize=(18, 14),
                              gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    colors = {"Long": "steelblue", "Short": "crimson", "Combined": "seagreen"}

    ax1 = axes[0]
    for name, s in series_to_plot.items():
        w = (1+s).cumprod(); w = w/w.iloc[0]
        lw = 2.5 if name == "Combined" else 1.8
        ax1.plot(w.index, w.values, label=name, color=colors[name], linewidth=lw)
    ax1.set_title(
        f"Hourly Momentum | NASDAQ100 + S&P500 (516 tickers)\n"
        f"Long: lb={best_long_p['lookback']}h hold={best_long_p['hold']}h n={best_long_p['top_n']}  |  "
        f"Short: lb={best_short_p['lookback']}h hold={best_short_p['hold']}h n={best_short_p['top_n']}"
        if best_long_p and best_short_p else "Hourly Momentum")
    ax1.set_ylabel("Cumulative Wealth")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.4)

    # Drawdown
    ax2 = axes[1]
    for name, s in series_to_plot.items():
        w = (1+s).cumprod(); w = w/w.iloc[0]; dd = w/w.cummax()-1
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.35,
                          color=colors[name], label=name)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.set_ylabel("Drawdown"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.4)

    # Rolling 60-day Sharpe
    ax3 = axes[2]
    for name, s in series_to_plot.items():
        roll_sh = s.rolling(60).apply(
            lambda x: float(np.sqrt(252)*x.mean()/x.std()) if x.std() > 0 else np.nan)
        ax3.plot(roll_sh.index, roll_sh.values, label=name, color=colors[name], linewidth=1.5)
    ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax3.set_ylabel("60-day Rolling Sharpe"); ax3.legend(fontsize=8); ax3.grid(True, alpha=0.4)

    plt.tight_layout(); plt.show()

    # Heatmap: Sharpe by lookback x hold for long side
    if not long_grid.dropna(subset=["Sharpe"]).empty:
        for tn in TOP_N_GRID[:2]:
            sub = long_grid[long_grid["top_n"] == tn]
            pivot = sub.pivot_table(index="lookback", columns="hold", values="Sharpe")
            if pivot.empty: continue
            fig2, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=2)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{h}h" for h in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{lb}h" for lb in pivot.index])
            plt.colorbar(im, ax=ax, label="Sharpe")
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    v = pivot.values[i, j]
                    ax.text(j, i, f"{v:.2f}" if pd.notna(v) else "N/A",
                            ha="center", va="center", fontsize=10)
            ax.set_title(f"LONG Sharpe: lookback x hold  (top_n={tn})")
            ax.set_xlabel("Hold (hours)"); ax.set_ylabel("Lookback (hours)")
            plt.tight_layout(); plt.show()

    # Short heatmap
    if not short_grid.dropna(subset=["Sharpe"]).empty:
        for tn in TOP_N_GRID[:2]:
            sub = short_grid[short_grid["top_n"] == tn]
            pivot = sub.pivot_table(index="lookback", columns="hold", values="Sharpe")
            if pivot.empty: continue
            fig3, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=2)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{h}h" for h in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{lb}h" for lb in pivot.index])
            plt.colorbar(im, ax=ax, label="Sharpe")
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    v = pivot.values[i, j]
                    ax.text(j, i, f"{v:.2f}" if pd.notna(v) else "N/A",
                            ha="center", va="center", fontsize=10)
            ax.set_title(f"SHORT Sharpe: lookback x hold  (top_n={tn})")
            ax.set_xlabel("Hold (hours)"); ax.set_ylabel("Lookback (hours)")
            plt.tight_layout(); plt.show()


# ══════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════

_header("Save to Excel")
xl = "results/hourly_momentum.xlsx"
try:
    with pd.ExcelWriter(xl, engine="openpyxl") as writer:
        # Summary
        rows = []
        for side, p, s in [("Long", best_long_p, best_long),
                            ("Short", best_short_p, best_short),
                            ("Combined", None, combined_ret)]:
            if s is None: continue
            w = (1+s).cumprod(); w=w/w.iloc[0]
            rows.append({"Side": side,
                          "Start": str(s.index[0].date()), "End": str(s.index[-1].date()),
                          "Days": len(s), "Total Return": float(w.iloc[-1]-1),
                          "Sharpe": _sharpe(s), "Sortino": _sortino(s), "Max DD": _mdd(s),
                          **(p or {})})
        pd.DataFrame(rows).to_excel(writer, sheet_name="Summary", index=False)

        long_grid.to_excel(writer,  sheet_name="Grid_Long",  index=False)
        short_grid.to_excel(writer, sheet_name="Grid_Short", index=False)
        if not combo_grid.empty:
            combo_grid.to_excel(writer, sheet_name="Grid_Combined", index=False)

        # Daily returns
        dr = {}
        if best_long   is not None: dr["Long"]     = best_long
        if best_short  is not None: dr["Short"]    = best_short
        if combined_ret is not None: dr["Combined"] = combined_ret
        pd.DataFrame(dr).to_excel(writer, sheet_name="Daily_Returns")

    print(f"  Saved: {xl}", flush=True)
except Exception as e:
    print(f"  Save failed: {e}", flush=True)

_header("SUMMARY")
print(f"\nUniverse: {len(universe)} tickers (NASDAQ100 + S&P500)")
print(f"Hourly data: {hourly_open.index[0].date()} to {hourly_open.index[-1].date()}")
if best_long_p:
    print(f"\nBest LONG:  lb={best_long_p['lookback']}h  hold={best_long_p['hold']}h  "
          f"n={best_long_p['top_n']}  Sharpe={best_long_p['Sharpe']:.3f}  "
          f"Return={best_long_p['Total_Return']:+.1%}")
if best_short_p:
    print(f"Best SHORT: lb={best_short_p['lookback']}h  hold={best_short_p['hold']}h  "
          f"n={best_short_p['top_n']}  Sharpe={best_short_p['Sharpe']:.3f}  "
          f"Return={best_short_p['Total_Return']:+.1%}")
if combined_ret is not None:
    print(f"Combined:   Sharpe={_sharpe(combined_ret):.3f}  "
          f"Return={(1+combined_ret).prod()-1:+.1%}  MaxDD={_mdd(combined_ret):.1%}")
