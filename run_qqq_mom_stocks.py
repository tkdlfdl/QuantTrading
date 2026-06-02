"""
Runner: QQQ Signal → Momentum Stock Strategy
  Signal : QQQ hourly bubble score extreme low
  Trades : buy top-N momentum stocks from SP500+NASDAQ100 universe
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

from data.universe import get_universe
from data.intraday_loader import load_hourly_bars
from strategies.qqq_signal_momentum_stocks import run_qqq_signal_momentum_stocks
from strategies.qqq_bubble_hourly import calculate_bubble_score

OUT_DIR = Path("results"); OUT_DIR.mkdir(exist_ok=True)

# ── Download QQQ 1h bars ────────────────────────────────────────────────────
start = (datetime.now() - timedelta(days=729)).strftime("%Y-%m-%d")
print(f"Downloading QQQ 1h bars from {start}...")
raw = yf.download("QQQ", start=start, interval="1h",
                  progress=False, auto_adjust=True)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
if raw.index.tz is not None:
    raw.index = raw.index.tz_localize(None)
qqq_o = raw["Open"].ffill()
qqq_c = raw["Close"].ffill()
print(f"QQQ bars: {len(qqq_c)}  |  {qqq_c.index[0].date()} → {qqq_c.index[-1].date()}")

# ── Load universe 1h bars (cached) ─────────────────────────────────────────
print("\nLoading universe hourly bars (SP500+NASDAQ100)...")
universe = get_universe()
stk_o, stk_c = load_hourly_bars(universe, use_cache=True)
print(f"Universe: {stk_c.shape[1]} tickers, {stk_c.shape[0]} bars")

# ── Run strategy ───────────────────────────────────────────────────────────
best_ret, best_params, grid_df = run_qqq_signal_momentum_stocks(
    qqq_open   = qqq_o,
    qqq_close  = qqq_c,
    stock_open  = stk_o,
    stock_close = stk_c,
    ma_window_grid               = [50, 100],
    z_window_grid                = [100, 200],
    threshold_grid               = [0.7, 0.8, 0.9],
    hold_hours_grid              = [1, 2, 4, 8],
    momentum_lookback_hours_grid = [6, 24, 48],
    top_n_grid                   = [5, 10, 20],
    transaction_cost             = 0.001,
)

# ── Benchmarks ─────────────────────────────────────────────────────────────
qqq_daily = qqq_c.pct_change().dropna().resample("B").apply(lambda r: (1+r).prod()-1)
qqq_w     = (1+qqq_daily).cumprod(); qqq_w = qqq_w/qqq_w.iloc[0]
qqq_sh    = float(np.sqrt(252)*qqq_daily.mean()/qqq_daily.std())
qqq_tot   = float(qqq_w.iloc[-1]-1)
qqq_mdd   = float((qqq_w/qqq_w.cummax()-1).min())

S = "=" * 72
print(f"\n{S}\nTOP 20 COMBOS (by Sharpe)\n{S}")
cols = ["ma_window","z_window","threshold","hold_hours","mom_lookback","top_n",
        "Sharpe","Sortino","Total_Return","Max_DD","n_trades","Win_Rate"]
print(grid_df[cols].head(20).to_string(index=False))

print(f"\n{S}\nBENCHMARK: QQQ Buy-and-Hold\n{S}")
print(f"  Sharpe={qqq_sh:.3f}  Return={qqq_tot:+.1%}  Max_DD={qqq_mdd:.1%}")

if best_params:
    print(f"\n{S}\nSTRATEGY vs BENCHMARK\n{S}")
    print(f"  {'':35} {'Sharpe':>7} {'Return':>8} {'Max_DD':>8} {'Trades':>7}")
    print(f"  {'QQQ Buy-and-Hold':35} {qqq_sh:>7.3f} {qqq_tot:>+8.1%} {qqq_mdd:>8.1%}    N/A")
    print(f"  {'QQQ Signal → Momentum Stocks':35} {best_params['Sharpe']:>7.3f} "
          f"{best_params['Total_Return']:>+8.1%} {best_params['Max_DD']:>8.1%} "
          f"{best_params['n_trades']:>7}")

    print(f"\n{S}\nYEARLY BREAKDOWN (best params)\n{S}")
    for yr, grp in best_ret.groupby(best_ret.index.year):
        r   = float((1+grp).prod()-1)
        sh  = float(np.sqrt(252)*grp.mean()/grp.std()) if grp.std()>0 else np.nan
        mdd = float(((1+grp).cumprod()/((1+grp).cumprod().cummax())-1).min())
        nt  = int((grp!=0).sum())
        print(f"  {yr}: Return={r:>+8.2%}  Sharpe={sh:>6.3f}  Max_DD={mdd:>8.2%}  "
              f"Signal_days={nt}")

# ── Chart ──────────────────────────────────────────────────────────────────
if best_ret is not None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                              gridspec_kw={"height_ratios": [3, 1.5, 1]})

    strat_w = (1+best_ret).cumprod(); strat_w = strat_w/strat_w.iloc[0]
    common  = strat_w.index.intersection(qqq_w.index)
    axes[0].plot(common, strat_w.loc[common], color="steelblue", linewidth=2,
                 label=f"QQQ Signal → Top-{best_params['top_n']} Momentum Stocks")
    axes[0].plot(common, qqq_w.loc[common], color="gray", linewidth=1.2,
                 linestyle="--", label="QQQ Buy-and-Hold")
    axes[0].set_title(
        f"QQQ Signal → Momentum Stocks | "
        f"ma={best_params['ma_window']}h  z={best_params['z_window']}h  "
        f"thresh=-{best_params['threshold']}  hold={best_params['hold_hours']}h  "
        f"mom_lb={best_params['mom_lookback']}h  top_n={best_params['top_n']}  "
        f"Sharpe={best_params['Sharpe']:.3f}  "
        f"Return={best_params['Total_Return']:+.1%}  "
        f"Trades={best_params['n_trades']}"
    )
    axes[0].set_ylabel("Cumulative Wealth"); axes[0].legend(); axes[0].grid(True, alpha=0.4)

    # QQQ bubble score
    score = calculate_bubble_score(qqq_c, best_params["ma_window"], best_params["z_window"])
    axes[1].plot(score.index, score.values, color="darkorange", linewidth=0.8, alpha=0.8)
    axes[1].axhline(-best_params["threshold"], color="green", linewidth=1.2,
                    linestyle="--", label=f"Entry: QQQ score < -{best_params['threshold']}")
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_ylabel("QQQ Bubble Score"); axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.4); axes[1].set_ylim(-1.1, 1.1)

    dd = strat_w/strat_w.cummax()-1
    axes[2].fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
    axes[2].set_ylabel("Drawdown"); axes[2].grid(True, alpha=0.4)

    plt.tight_layout()
    out = OUT_DIR / "qqq_mom_stocks.png"
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print(f"\n[Chart saved: {out}]")

grid_df.to_excel(OUT_DIR / "qqq_mom_stocks_grid.xlsx", index=False)
print(f"[Grid saved: results/qqq_mom_stocks_grid.xlsx]")
