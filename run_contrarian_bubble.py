"""
Runner: Contrarian Bubble Strategy (Long-Only, Hourly)
  Signal : bubble < -threshold (extreme oversold per-stock)
  Trade  : buy top-N most oversold, hold X hours, exit at close
  Universe: SP500 + NASDAQ100 (~515 tickers)
  Costs  : 0.1% TC per trade

Usage:
  python run_contrarian_bubble.py
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from pathlib import Path

from data.universe import get_universe
from strategies.contrarian_bubble_hourly import run_contrarian_bubble_hourly

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

# ── Load merged data (Alpaca 2019-2024 + yfinance 2024-2026) ───────────────
print("Loading merged hourly bars (Alpaca + yfinance)...")
universe = set(get_universe())
ho = pd.read_parquet("data/cache/merged_hourly_open.parquet")
hc = pd.read_parquet("data/cache/merged_hourly_close.parquet")
# Normalize index
for df in (ho, hc):
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.floor("h")
# Filter to universe tickers only
valid = [c for c in hc.columns if c in universe]
ho = ho[valid].ffill()
hc = hc[valid].ffill()
print(f"Data: {hc.shape[1]} tickers x {hc.shape[0]} bars  "
      f"({hc.index[0].date()} -> {hc.index[-1].date()})")

# ── Run strategy with grid search ──────────────────────────────────────────
best_ret, best_params, grid_df = run_contrarian_bubble_hourly(
    hourly_open=ho,
    hourly_close=hc,
    ma_window_grid=[52, 104, 156, 208],
    buy_threshold_grid=[0.9, 0.8, 0.7, 0.6, 0.5],
    hold_hours_grid=[4, 8, 13, 26, 52],
    top_n_grid=[5, 10, 20],
    transaction_cost=0.001,
)

# ── Print results ──────────────────────────────────────────────────────────
S = "=" * 80
print(f"\n{S}")
print("TOP 20 COMBOS (by Sharpe) - Contrarian Bubble Long-Only")
print(S)
cols = ["ma_window", "buy_threshold", "hold_hours", "top_n",
        "Sharpe", "Sortino", "Total_Return", "Max_DD", "n_trades", "Win_Rate"]
print(grid_df[cols].head(20).to_string(index=False))

print(f"\nPositive Sharpe:  {(grid_df.Sharpe > 0).sum()}/{len(grid_df)}")
print(f"Sharpe > 0.5:     {(grid_df.Sharpe > 0.5).sum()}/{len(grid_df)}")
print(f"Sharpe > 1.0:     {(grid_df.Sharpe > 1.0).sum()}/{len(grid_df)}")
print(f"Sharpe > 2.0:     {(grid_df.Sharpe > 2.0).sum()}/{len(grid_df)}")

# ── Parameter sensitivity ──────────────────────────────────────────────────
print(f"\n{S}\nPARAMETER SENSITIVITY\n{S}")
for col, label in [
    ("ma_window", "MA Window (h)"),
    ("buy_threshold", "Buy Threshold"),
    ("hold_hours", "Hold (h)"),
    ("top_n", "Top-N"),
]:
    g = (grid_df.dropna(subset=["Sharpe"])
         .groupby(col)["Sharpe"]
         .agg(["mean", "max", "count"])
         .reset_index())
    g.columns = [label, "Avg Sharpe", "Best Sharpe", "N"]
    print(f"\n{g.to_string(index=False)}")

# ── Yearly breakdown ───────────────────────────────────────────────────────
if best_params and best_ret is not None:
    print(f"\n{S}\nYEARLY BREAKDOWN (best params)\n{S}")
    print(f"  MA={best_params['ma_window']}h  "
          f"thresh={best_params['buy_threshold']}  "
          f"hold={best_params['hold_hours']}h  "
          f"top_n={best_params['top_n']}\n")
    print(f"  {'Year':<6}  {'Return':>9}  {'Ann Ret':>9}  {'Sharpe':>8}  "
          f"{'Max DD':>9}  {'Trade Days':>10}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*10}")

    yearly_rows = []
    for yr, grp in best_ret.groupby(best_ret.index.year):
        if grp.empty:
            continue
        tr = float((1 + grp).prod() - 1)
        yrs = len(grp) / 252
        ar = float((1 + tr) ** (1 / max(yrs, 0.01)) - 1) if tr > -1 else -1.0
        std = grp.std()
        sh = float(np.sqrt(252) * grp.mean() / std) if std > 0 else np.nan
        w = (1 + grp).cumprod()
        mdd = float((w / w.cummax() - 1).min())
        nt = int((grp != 0).sum())
        print(f"  {yr:<6}  {tr:>9.2%}  {ar:>9.2%}  {sh:>8.3f}  {mdd:>9.2%}  {nt:>10}")
        yearly_rows.append(dict(Year=yr, Return=tr, Ann_Ret=ar,
                                Sharpe=sh, Max_DD=mdd, Trade_Days=nt))

    # ── Chart ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Contrarian Bubble Strategy (Long-Only) — S&P500+NASDAQ100\n"
        f"MA={best_params['ma_window']}h  |  bubble<-{best_params['buy_threshold']}  |  "
        f"hold={best_params['hold_hours']}h  |  top-{best_params['top_n']}\n"
        f"Ann={best_params['Total_Return']:.1%}  Sharpe={best_params['Sharpe']:.3f}  "
        f"MaxDD={best_params['Max_DD']:.2%}  WinRate={best_params['Win_Rate']:.0%}",
        fontsize=11, fontweight="bold"
    )

    # Wealth curve
    wealth = (1 + best_ret).cumprod()
    wealth = wealth / wealth.iloc[0]
    axes[0, 0].plot(wealth.index, wealth.values, lw=2, color="steelblue")
    axes[0, 0].set_title("Cumulative Wealth", fontweight="bold")
    axes[0, 0].set_ylabel("Wealth Multiple")
    axes[0, 0].grid(True, alpha=0.3)

    # Drawdown
    dd = wealth / wealth.cummax() - 1
    axes[0, 1].fill_between(dd.index, dd.values * 100, 0, alpha=0.6, color="red")
    axes[0, 1].set_title(f"Drawdown (Max={best_params['Max_DD']:.2%})", fontweight="bold")
    axes[0, 1].set_ylabel("Drawdown (%)")
    axes[0, 1].grid(True, alpha=0.3)

    # Yearly returns bar chart
    if yearly_rows:
        ydf = pd.DataFrame(yearly_rows)
        clrs = ["steelblue" if r >= 0 else "tomato" for r in ydf["Return"]]
        axes[1, 0].bar(ydf["Year"].astype(str), ydf["Return"] * 100, color=clrs, alpha=0.8)
        axes[1, 0].axhline(0, color="black", lw=0.8)
        axes[1, 0].set_title("Yearly Returns (%)", fontweight="bold")
        axes[1, 0].set_ylabel("Return (%)")
        axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Rolling 30-day Sharpe
    roll_sh = (best_ret.rolling(30).mean() / best_ret.rolling(30).std() * np.sqrt(252))
    axes[1, 1].plot(roll_sh.index, roll_sh.values, lw=1.5, color="purple")
    axes[1, 1].axhline(0, color="black", lw=0.8)
    axes[1, 1].set_title("Rolling 30-Day Sharpe", fontweight="bold")
    axes[1, 1].set_ylabel("Sharpe")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = OUT_DIR / "contrarian_bubble_hourly.png"
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n[Chart saved: {out_png}]")

# ── Save grid ──────────────────────────────────────────────────────────────
out_xlsx = OUT_DIR / "contrarian_bubble_hourly_grid.xlsx"
grid_df.to_excel(out_xlsx, index=False)
print(f"[Grid saved: {out_xlsx}]")
