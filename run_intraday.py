"""
Runner: Intraday Mean-Reversion Strategy
  Signal  : daily return z-score vs rolling window (long history)
  Execute : 1h bars — enter at market open, exit after X hours
  Universe: NASDAQ 100 + S&P 500 (top 50 by market cap)
"""
import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from data.db.schema import init
from data.universe import get_universe, get_top_by_marketcap
from data.intraday_loader import load_daily_close, load_hourly_bars
from strategies.intraday_mean_reversion import run_intraday_mean_reversion

import pandas as pd
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────
DAILY_START  = "2018-01-01"   # long history for lookback window
TOP_N_UNIV   = 50             # limit universe to top 50 by mktcap

SIGMA_GRID      = [1.5, 2.0, 2.5, 3.0]
HOLD_HOURS_GRID = [1, 2, 4, 8]
LOOKBACK_GRID   = [20, 60, 120]
TOP_N_GRID      = [5, 10, 20]
TRANS_COST      = 0.001       # 0.1% per trade (round-trip)

# ── Universe ────────────────────────────────────────────────────────────────
init()
print("Building universe (NASDAQ100 + S&P500, top 50 by market cap)...")
universe = get_top_by_marketcap(get_universe(), n=TOP_N_UNIV)
print(f"Universe: {len(universe)} tickers\n")

# ── Data ────────────────────────────────────────────────────────────────────
daily_close          = load_daily_close(universe, start=DAILY_START, use_cache=True)
hourly_open, hourly_close = load_hourly_bars(universe, use_cache=True)

print(f"\nDaily close : {len(daily_close)} days × {len(daily_close.columns)} tickers")
print(f"Hourly open : {len(hourly_open)} bars × {len(hourly_open.columns)} tickers\n")

# ── Strategy ────────────────────────────────────────────────────────────────
best_ret, best_params, grid_df = run_intraday_mean_reversion(
    daily_close=daily_close,
    hourly_open=hourly_open,
    hourly_close=hourly_close,
    sigma_grid=SIGMA_GRID,
    hold_hours_grid=HOLD_HOURS_GRID,
    lookback_grid=LOOKBACK_GRID,
    top_n_grid=TOP_N_GRID,
    transaction_cost=TRANS_COST,
)

# ── Results ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TOP 15 PARAMETER COMBOS (by Sharpe)")
print("=" * 65)
print(grid_df.head(15).to_string(index=False))

print("\n" + "=" * 65)
print("YEARLY PERFORMANCE (best params)")
print("=" * 65)
if best_ret is not None:
    yearly = best_ret.groupby(best_ret.index.year).apply(
        lambda r: pd.Series({
            "Return":   float((1 + r).prod() - 1),
            "Sharpe":   float(np.sqrt(252) * r.mean() / r.std()) if r.std() > 0 else np.nan,
            "Max DD":   float((((1 + r).cumprod() / (1 + r).cumprod().cummax()) - 1).min()),
            "N trades": int((r != 0).sum()),
        })
    )
    print(yearly.to_string())

# ── Chart ───────────────────────────────────────────────────────────────────
if best_ret is not None:
    wealth = (1 + best_ret).cumprod()
    wealth = wealth / wealth.iloc[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(wealth.index, wealth.values, linewidth=1.5, color="steelblue")
    ax1.set_title(
        f"Intraday Mean-Reversion | σ={best_params['sigma']} | "
        f"hold={best_params['hold_hours']}h | lookback={best_params['lookback']}d | "
        f"top_n={best_params['top_n']} | "
        f"Sharpe={best_params['Sharpe']:.3f} | "
        f"Return={best_params['Total_Return']:+.1%}"
    )
    ax1.set_ylabel("Cumulative Wealth")
    ax1.grid(True)

    dd = (wealth / wealth.cummax() - 1)
    ax2.fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
    ax2.set_ylabel("Drawdown")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("intraday_mr_result.png", dpi=130, bbox_inches="tight")
    print("\n[Chart saved: intraday_mr_result.png]")
    plt.close()
