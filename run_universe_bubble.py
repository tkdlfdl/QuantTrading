"""
Runner: Universe Hourly Bubble Strategy (Long/Short)
  Signal : bubble score on each stock's 1h bars
  Trade  : long bottom-N, short top-N by score, hold X hours
  Universe: SP500 + NASDAQ100 (~516 tickers)
"""
import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from pathlib import Path

from data.universe import get_universe
from data.intraday_loader import load_hourly_bars
from strategies.universe_bubble_hourly import run_universe_bubble_hourly

OUT_DIR = Path("results"); OUT_DIR.mkdir(exist_ok=True)

# ── Load data ───────────────────────────────────────────────────────────────
print("Loading universe hourly bars (cached)...")
universe = get_universe()
ho, hc   = load_hourly_bars(universe, use_cache=True)
print(f"Data: {hc.shape[1]} tickers × {hc.shape[0]} bars  "
      f"({hc.index[0].date()} → {hc.index[-1].date()})")

# ── Run strategy ───────────────────────────────────────────────────────────
best_ret, best_params, grid_df = run_universe_bubble_hourly(
    hourly_open  = ho,
    hourly_close = hc,
    ma_window_grid   = [20, 50, 100],
    z_window_grid    = [50, 100, 200],
    threshold_grid   = [0.5, 0.6, 0.7, 0.8, 0.9],
    hold_hours_grid  = [1, 2, 4, 8],
    top_n_grid       = [5, 10, 20],
    transaction_cost  = 0.001,
    short_borrow_rate = 0.08,
)

# ── Results ────────────────────────────────────────────────────────────────
S = "=" * 75
print(f"\n{S}\nTOP 20 COMBOS (by Sharpe)\n{S}")
cols = ["ma_window","z_window","threshold","hold_hours","top_n",
        "Sharpe","Sortino","Total_Return","Max_DD","n_trades","Win_Rate"]
print(grid_df[cols].head(20).to_string(index=False))

if best_params:
    print(f"\n{S}\nYEARLY BREAKDOWN (best params)\n{S}")
    for yr, grp in best_ret.groupby(best_ret.index.year):
        r   = float((1+grp).prod()-1)
        sh  = float(np.sqrt(252)*grp.mean()/grp.std()) if grp.std()>0 else np.nan
        mdd = float(((1+grp).cumprod()/((1+grp).cumprod().cummax())-1).min())
        nt  = int((grp!=0).sum())
        print(f"  {yr}: Return={r:>+8.2%}  Sharpe={sh:>6.3f}  "
              f"Max_DD={mdd:>8.2%}  Trade_days={nt}")

    # Chart
    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                              gridspec_kw={"height_ratios":[3,1]})
    wealth = (1+best_ret).cumprod(); wealth=wealth/wealth.iloc[0]
    axes[0].plot(wealth.index, wealth.values, color="steelblue", linewidth=2)
    axes[0].set_title(
        f"Universe Hourly Bubble L/S | "
        f"ma={best_params['ma_window']}h  z={best_params['z_window']}h  "
        f"thresh={best_params['threshold']}  hold={best_params['hold_hours']}h  "
        f"top_n={best_params['top_n']}  "
        f"Sharpe={best_params['Sharpe']:.3f}  "
        f"Return={best_params['Total_Return']:+.1%}  "
        f"Trades={best_params['n_trades']}",
        fontsize=10, fontweight="bold"
    )
    axes[0].set_ylabel("Cumulative Wealth"); axes[0].grid(True, alpha=0.4)
    dd = wealth/wealth.cummax()-1
    axes[1].fill_between(dd.index, dd.values*100, 0, alpha=0.4, color="red")
    axes[1].set_ylabel("Drawdown (%)"); axes[1].grid(True, alpha=0.4)
    plt.tight_layout()
    out = OUT_DIR / "universe_bubble_hourly.png"
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print(f"\n[Chart saved: {out}]")

grid_df.to_excel(OUT_DIR/"universe_bubble_grid.xlsx", index=False)
print(f"[Grid saved: results/universe_bubble_grid.xlsx]")
