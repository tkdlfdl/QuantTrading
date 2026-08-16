"""
INTRADAY MEAN REVERSION — HOURLY BACKTEST
==========================================
Runs the IntradayMR strategy from strategies/intraday_mean_reversion.py
on the full hourly dataset (2019-2026, 7.4 years).

Phase 1: Mean-reversion for 1 hour after daily abnormal return
Phase 2: Momentum flip for flip_hold_days after that
"""

import sys, warnings, os
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strategies.intraday_mean_reversion import run_intraday_mean_reversion

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
print("Loading data...")

daily_close = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
hourly_close = pd.read_parquet("data/cache/merged_hourly_close.parquet")
hourly_open  = pd.read_parquet("data/cache/merged_hourly_open.parquet")

# Normalize timestamps to hour-floor so open/close bars align
hourly_close.index = hourly_close.index.floor("h")
hourly_open.index  = hourly_open.index.floor("h")
hourly_close = hourly_close[~hourly_close.index.duplicated(keep="last")]
hourly_open  = hourly_open[~hourly_open.index.duplicated(keep="last")]

# Keep S&P500 + NASDAQ100 tickers in common across all three datasets
daily_cols  = set(daily_close.columns)
hourly_cols = set(hourly_close.columns) & set(hourly_open.columns)
common_cols = sorted(daily_cols & hourly_cols)

# Quality filter: drop stocks with >30% NaN in hourly
valid_mask  = (hourly_close[common_cols].isna().mean() < 0.30) & \
              (hourly_open[common_cols].isna().mean() < 0.30)
good_cols   = [c for c in common_cols if valid_mask[c]]

daily_close  = daily_close[good_cols].ffill()
hourly_close = hourly_close[good_cols].ffill()
hourly_open  = hourly_open[good_cols].ffill()

# Trim to the hourly data window
start = hourly_close.index[0].date()
end   = hourly_close.index[-1].date()
daily_close = daily_close.loc[str(start):str(end)]

years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25

print(f"Daily:         {daily_close.shape}  ({daily_close.index[0].date()} to {daily_close.index[-1].date()})")
print(f"Hourly close:  {hourly_close.shape}  ({hourly_close.index[0]} to {hourly_close.index[-1]})")
print(f"Hourly open:   {hourly_open.shape}")
print(f"Tickers:       {len(good_cols)}")
print(f"Test period:   {years:.2f} years")

# ─────────────────────────────────────────────────────────────
# RUN STRATEGY
# ─────────────────────────────────────────────────────────────
print("\nRunning IntradayMR grid search...")

best_ret, best_params, grid_df = run_intraday_mean_reversion(
    daily_close  = daily_close,
    hourly_open  = hourly_open,
    hourly_close = hourly_close,
    sigma_grid          = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
    flip_hold_days_grid = [0, 1, 2, 3, 4, 5],
    lookback_grid       = [10, 20, 40, 60, 120],
    top_n_grid          = [5, 10, 20],
    transaction_cost    = 0.001,
    short_borrow_rate   = 0.08,
)

# ─────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*110}")
print("GRID SEARCH RESULTS - TOP 20 (by Sharpe)")
print(f"{'='*110}")
fmt = {"Sharpe":"{:.4f}".format,"Sortino":"{:.4f}".format,
       "Total_Return":"{:.2%}".format,"Max_DD":"{:.2%}".format}
print(grid_df.head(20).to_string(index=False, formatters=fmt))

print(f"\nPositive Sharpe: {(grid_df.Sharpe > 0).sum()}/{len(grid_df)}")
print(f"Sharpe > 0.5:    {(grid_df.Sharpe > 0.5).sum()}/{len(grid_df)}")
print(f"Sharpe > 1.0:    {(grid_df.Sharpe > 1.0).sum()}/{len(grid_df)}")

# ─────────────────────────────────────────────────────────────
# PARAMETER SENSITIVITY
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*110}")
print("PARAMETER SENSITIVITY")
print(f"{'='*110}")
for col, label in [("sigma","Sigma"), ("lookback","Lookback (d)"),
                   ("flip_hold_days","Flip Hold Days"), ("top_n","Top-N")]:
    g = grid_df.groupby(col)["Sharpe"].agg(["mean","max","count"]).reset_index()
    g.columns = [label, "Avg Sharpe", "Best Sharpe", "N"]
    print(f"\n{g.to_string(index=False)}")

# ─────────────────────────────────────────────────────────────
# YEARLY BREAKDOWN
# ─────────────────────────────────────────────────────────────
if best_ret is not None:
    wealth = (1 + best_ret).cumprod()
    total_ret = wealth.iloc[-1] - 1
    ann_ret   = (1 + total_ret) ** (1 / years) - 1

    # Correct Sharpe using daily returns annualised at 252
    sh = best_ret.mean() / best_ret.std() * np.sqrt(252) if best_ret.std() > 0 else 0
    dn = best_ret[best_ret < 0].std(ddof=0)
    so = best_ret.mean() / dn * np.sqrt(252) if dn > 0 else 0
    dd = (wealth / wealth.cummax() - 1).min()

    print(f"\n{'='*110}")
    print("OVERALL PERFORMANCE  (Best Parameters)")
    print(f"{'='*110}")
    print(f"  Total Return:   {total_ret:.2%}")
    print(f"  Annual Return:  {ann_ret:.2%}")
    print(f"  Sharpe Ratio:   {sh:.4f}")
    print(f"  Sortino Ratio:  {so:.4f}")
    print(f"  Max Drawdown:   {dd:.2%}")
    print(f"  Trading Days:   {len(best_ret)}")
    print(f"  Active Trades:  {int((best_ret != 0).sum())}")
    print(f"  Win Rate:       {(best_ret > 0).sum() / max((best_ret != 0).sum(), 1):.1%}")

    print(f"\n{'='*110}")
    print("YEARLY BREAKDOWN  (Best Parameters)")
    print(f"{'='*110}")
    print(f"  {'Year':<6}  {'Return':>9}  {'Ann Ret':>9}  {'Sharpe':>8}  {'Sortino':>9}  {'MaxDD':>9}  {'Trades':>7}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*7}")

    yearly_rows = []
    for yr in sorted(best_ret.index.year.unique()):
        ys = best_ret[best_ret.index.year == yr]
        yw = wealth[wealth.index.year == yr]
        if len(ys) == 0: continue

        ytr  = (1 + ys).prod() - 1
        ysh  = ys.mean() / ys.std() * np.sqrt(252) if ys.std() > 0 else 0
        ydn  = ys[ys < 0].std(ddof=0)
        yso  = ys.mean() / ydn * np.sqrt(252) if ydn > 0 else 0
        ydd  = (yw / yw.cummax() - 1).min()
        act  = int((ys != 0).sum())

        # Annual return for the year
        yr_days = len(ys)
        yar = (1 + ytr) ** (252 / yr_days) - 1 if ytr > -1 else -1

        print(f"  {yr:<6}  {ytr:>9.2%}  {yar:>9.2%}  {ysh:>8.3f}  {yso:>9.3f}  {ydd:>9.2%}  {act:>7}")
        yearly_rows.append(dict(Year=yr, Return=ytr, Ann_Ret=yar,
                                Sharpe=ysh, Sortino=yso, MaxDD=ydd, Trades=act))

    # ─────────────────────────────────────────────────────────
    # PLOTS
    # ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Intraday Mean Reversion Strategy\n"
                 f"Best: sigma={best_params['sigma']}, lookback={best_params['lookback']}d, "
                 f"flip_hold={best_params['flip_hold_days']}d, top_n={best_params['top_n']}",
                 fontsize=12, fontweight="bold")

    # Wealth
    axes[0,0].plot(wealth.index, wealth.values, lw=2, color="steelblue")
    axes[0,0].set_title(f"Cumulative Wealth  (Ann={ann_ret:.1%}, Sharpe={sh:.3f})", fontweight="bold")
    axes[0,0].set_ylabel("Wealth Multiple"); axes[0,0].grid(True, alpha=0.3)

    # Drawdown
    dd_series = wealth / wealth.cummax() - 1
    axes[0,1].fill_between(dd_series.index, dd_series.values, 0, alpha=0.6, color="red")
    axes[0,1].set_title("Drawdown", fontweight="bold")
    axes[0,1].set_ylabel("Drawdown"); axes[0,1].grid(True, alpha=0.3)

    # Yearly returns
    if yearly_rows:
        ydf  = pd.DataFrame(yearly_rows)
        clrs = ["green" if r >= 0 else "red" for r in ydf["Return"]]
        axes[1,0].bar(ydf["Year"].astype(str), ydf["Return"]*100, color=clrs, alpha=0.7)
        axes[1,0].axhline(0, color="black", lw=0.8)
        axes[1,0].set_title("Yearly Returns (%)", fontweight="bold")
        axes[1,0].set_ylabel("Return (%)"); axes[1,0].grid(True, alpha=0.3, axis="y")

        # Yearly Sharpe
        sh_clrs = ["steelblue" if s >= 0 else "tomato" for s in ydf["Sharpe"]]
        axes[1,1].bar(ydf["Year"].astype(str), ydf["Sharpe"], color=sh_clrs, alpha=0.7)
        axes[1,1].axhline(0, color="black", lw=0.8)
        axes[1,1].set_title("Yearly Sharpe Ratios", fontweight="bold")
        axes[1,1].set_ylabel("Sharpe"); axes[1,1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/intraday_mr_hourly_backtest.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: results/intraday_mr_hourly_backtest.png")

    grid_df.to_csv("results/intraday_mr_hourly_grid.csv", index=False)
    print(f"Saved: results/intraday_mr_hourly_grid.csv")
    pd.DataFrame(yearly_rows).to_csv("results/intraday_mr_yearly.csv", index=False)
    print(f"Saved: results/intraday_mr_yearly.csv")

print("\nDONE")
