"""
Runner: QQQ Hourly Bubble Score Strategy
  - Download 2 years of QQQ 1h bars
  - Apply bubble score, buy on extreme low, hold X hours
  - Grid: ma_window × z_window × threshold × hold_hours
"""
import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

from strategies.qqq_bubble_hourly import run_qqq_bubble_hourly

OUT_DIR = Path("results"); OUT_DIR.mkdir(exist_ok=True)

# ── Download QQQ 1h bars (last 730 days) ──────────────────────────────────
start = (datetime.now() - timedelta(days=729)).strftime("%Y-%m-%d")
print(f"Downloading QQQ 1h bars from {start}...")
raw = yf.download("QQQ", start=start, interval="1h",
                  progress=False, auto_adjust=True)

if raw.empty:
    print("ERROR: no data returned"); sys.exit(1)

# flatten any MultiIndex
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
if raw.index.tz is not None:
    raw.index = raw.index.tz_localize(None)

ho = raw["Open"].ffill()
hc = raw["Close"].ffill()
print(f"Bars: {len(hc)}  |  {hc.index[0].date()} → {hc.index[-1].date()}")

# ── Run strategy ───────────────────────────────────────────────────────────
best_ret, best_params, grid_df = run_qqq_bubble_hourly(
    hourly_open  = ho,
    hourly_close = hc,
    ma_window_grid  = [20, 50, 100],
    z_window_grid   = [50, 100, 200],
    threshold_grid  = [0.5, 0.6, 0.7, 0.8, 0.9],
    hold_hours_grid = [1, 2, 4, 8, 24],
    transaction_cost = 0.001,
)

# ── Print results ──────────────────────────────────────────────────────────
S = "=" * 72
print(f"\n{S}\nTOP 20 PARAMETER COMBOS (by Sharpe)\n{S}")
cols = ["ma_window","z_window","threshold","hold_hours",
        "Sharpe","Sortino","Total_Return","Max_DD","n_trades","Win_Rate"]
print(grid_df[cols].head(20).to_string(index=False))

# QQQ buy-and-hold benchmark
bh_ret = hc.pct_change().dropna()
bh_daily = bh_ret.resample("B").apply(lambda r: (1+r).prod()-1)
bh_wealth = (1+bh_daily).cumprod(); bh_wealth = bh_wealth/bh_wealth.iloc[0]
bh_sh  = float(np.sqrt(252)*bh_daily.mean()/bh_daily.std())
bh_tot = float(bh_wealth.iloc[-1]-1)
bh_mdd = float((bh_wealth/bh_wealth.cummax()-1).min())

print(f"\n{S}\nBENCHMARK: QQQ Buy-and-Hold\n{S}")
print(f"  Sharpe={bh_sh:.3f}  Return={bh_tot:+.1%}  Max_DD={bh_mdd:.1%}")

if best_params:
    print(f"\n{S}\nSTRATEGY vs BENCHMARK\n{S}")
    print(f"  {'':30} {'Sharpe':>7} {'Return':>8} {'Max_DD':>8} {'Trades':>7}")
    print(f"  {'QQQ Buy-and-Hold':30} {bh_sh:>7.3f} {bh_tot:>+8.1%} {bh_mdd:>8.1%} {'N/A':>7}")
    print(f"  {'QQQ Bubble Strategy':30} {best_params['Sharpe']:>7.3f} "
          f"{best_params['Total_Return']:>+8.1%} {best_params['Max_DD']:>8.1%} "
          f"{best_params['n_trades']:>7}")

print(f"\n{S}\nYEARLY BREAKDOWN (best params)\n{S}")
if best_ret is not None:
    for yr, grp in best_ret.groupby(best_ret.index.year):
        r   = float((1+grp).prod()-1)
        sh  = float(np.sqrt(252)*grp.mean()/grp.std()) if grp.std()>0 else np.nan
        mdd = float(((1+grp).cumprod()/((1+grp).cumprod().cummax())-1).min())
        ntrades = int((grp!=0).sum())
        print(f"  {yr}: Return={r:>+8.2%}  Sharpe={sh:>6.3f}  Max_DD={mdd:>8.2%}  "
              f"Signal_days={ntrades}")

# ── Chart ──────────────────────────────────────────────────────────────────
if best_ret is not None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                              gridspec_kw={"height_ratios": [3, 1.5, 1]})

    # Wealth
    strat_w = (1+best_ret).cumprod(); strat_w = strat_w/strat_w.iloc[0]
    common  = strat_w.index.intersection(bh_wealth.index)
    axes[0].plot(common, strat_w.loc[common], color="steelblue",
                 linewidth=2, label="QQQ Bubble Strategy")
    axes[0].plot(common, bh_wealth.loc[common], color="gray",
                 linewidth=1.2, linestyle="--", label="QQQ Buy-and-Hold")
    axes[0].set_title(
        f"QQQ Hourly Bubble | ma={best_params['ma_window']}h  "
        f"z={best_params['z_window']}h  thresh=-{best_params['threshold']}  "
        f"hold={best_params['hold_hours']}h  "
        f"Sharpe={best_params['Sharpe']:.3f}  "
        f"Return={best_params['Total_Return']:+.1%}  "
        f"Trades={best_params['n_trades']}"
    )
    axes[0].set_ylabel("Cumulative Wealth"); axes[0].legend(); axes[0].grid(True, alpha=0.4)

    # Bubble score (best ma+z combo)
    from strategies.qqq_bubble_hourly import calculate_bubble_score
    score = calculate_bubble_score(hc, best_params["ma_window"], best_params["z_window"])
    axes[1].plot(score.index, score.values, color="darkorange", linewidth=0.8, alpha=0.8)
    axes[1].axhline(-best_params["threshold"], color="green", linewidth=1.2,
                    linestyle="--", label=f"Entry: score < -{best_params['threshold']}")
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_ylabel("Bubble Score"); axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.4)
    axes[1].set_ylim(-1.1, 1.1)

    # Drawdown
    dd = strat_w/strat_w.cummax()-1
    axes[2].fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
    axes[2].set_ylabel("Drawdown"); axes[2].grid(True, alpha=0.4)

    plt.tight_layout()
    out = OUT_DIR / "qqq_bubble_hourly.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n[Chart saved: {out}]")

# ── Save grid ──────────────────────────────────────────────────────────────
grid_out = OUT_DIR / "qqq_bubble_grid.xlsx"
grid_df.to_excel(grid_out, index=False)
print(f"[Grid saved: {grid_out}]")
