"""
Runner: Contrarian Bubble Long/Short Strategy
  LONG  : buy top-N most oversold  (bubble < -buy_threshold)
  SHORT : sell top-N most overbought (bubble > +short_threshold)
  Universe: SP500 + NASDAQ100 (515 tickers)
  Data: Alpaca 2019-2024 + yfinance 2024-2026 (merged)

Usage:
  python run_contrarian_bubble_ls.py
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
from strategies.contrarian_bubble_ls_hourly import run_contrarian_bubble_ls_hourly

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

# ── Load merged data ────────────────────────────────────────────────────────
print("Loading merged hourly bars (Alpaca 2019-2024 + yfinance 2024-2026)...")
universe = set(get_universe())
ho = pd.read_parquet("data/cache/merged_hourly_open.parquet")
hc = pd.read_parquet("data/cache/merged_hourly_close.parquet")
for df in (ho, hc):
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.floor("h")
valid = [c for c in hc.columns if c in universe]
ho = ho[valid].ffill()
hc = hc[valid].ffill()
print(f"Data: {hc.shape[1]} tickers x {hc.shape[0]} bars  "
      f"({hc.index[0].date()} -> {hc.index[-1].date()})")

# ── Run grid search ────────────────────────────────────────────────────────
best, grid_df = run_contrarian_bubble_ls_hourly(
    hourly_open=ho,
    hourly_close=hc,
    ma_window_grid=[52, 104, 156],
    buy_threshold_grid=[0.8, 0.9],
    short_threshold_grid=[0.8, 0.85, 0.9, 0.95],
    hold_hours_grid=[8, 13],
    top_n_grid=[10, 20],
    transaction_cost=0.001,
    short_borrow_rate=0.08,
)

# ── Print grid top results ─────────────────────────────────────────────────
S = "=" * 95
print(f"\n{S}")
print("TOP 20 COMBOS (by Combined Sharpe) - Contrarian Bubble Long/Short")
print(S)
cols = ["ma_window", "buy_threshold", "short_threshold", "hold_hours", "top_n",
        "Long_Sharpe", "Short_Sharpe", "Combined_Sharpe",
        "Long_Return", "Short_Return", "Combined_Return",
        "Long_MaxDD", "Short_MaxDD", "Combined_MaxDD"]
print(grid_df[cols].head(20).to_string(index=False))

print(f"\nPositive Combined Sharpe: {(grid_df.Combined_Sharpe > 0).sum()}/{len(grid_df)}")
print(f"Combined Sharpe > 1.0:   {(grid_df.Combined_Sharpe > 1.0).sum()}/{len(grid_df)}")
print(f"Combined Sharpe > 2.0:   {(grid_df.Combined_Sharpe > 2.0).sum()}/{len(grid_df)}")
print(f"Short Sharpe > 0:        {(grid_df.Short_Sharpe > 0).sum()}/{len(grid_df)}")
print(f"Short Sharpe > 1.0:      {(grid_df.Short_Sharpe > 1.0).sum()}/{len(grid_df)}")

# ── Decomposition report ───────────────────────────────────────────────────
if best:
    p = best["params"]
    lm = best["long_metrics"]
    sm = best["short_metrics"]
    cm = best["combined_metrics"]

    print(f"\n{S}")
    print("BEST PARAMS DECOMPOSITION")
    print(S)
    print(f"  MA={p['ma_window']}h  buy_thresh={p['buy_threshold']}  "
          f"short_thresh={p['short_threshold']}  hold={p['hold_hours']}h  top_n={p['top_n']}\n")

    print(f"  {'Metric':<18} {'Long':>12} {'Short':>12} {'Combined':>12}")
    print(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*12}")
    for key, label in [
        ("Sharpe", "Sharpe"),
        ("Sortino", "Sortino"),
        ("Total_Return", "Total Return"),
        ("Max_DD", "Max Drawdown"),
        ("Win_Rate", "Win Rate"),
        ("Active_Days", "Active Days"),
    ]:
        lv = lm.get(key, np.nan)
        sv = sm.get(key, np.nan)
        cv = cm.get(key, np.nan)
        if key in ("Total_Return", "Max_DD", "Win_Rate"):
            fmt = lambda v: f"{v:.2%}" if pd.notna(v) else "N/A"
        elif key == "Active_Days":
            fmt = lambda v: f"{int(v)}" if pd.notna(v) else "N/A"
        else:
            fmt = lambda v: f"{v:.4f}" if pd.notna(v) else "N/A"
        print(f"  {label:<18} {fmt(lv):>12} {fmt(sv):>12} {fmt(cv):>12}")

    # ── Yearly breakdown ───────────────────────────────────────────────────
    print(f"\n{S}")
    print("YEARLY BREAKDOWN")
    print(S)
    print(f"  {'Year':<6}  {'Long':>9}  {'Short':>9}  {'Combined':>10}  "
          f"{'L-Sharpe':>9}  {'S-Sharpe':>9}  {'C-Sharpe':>9}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*9}")

    yearly_rows = []
    for yr in sorted(best["combined"].index.year.unique()):
        sl = best["long"][best["long"].index.year == yr]
        ss = best["short"][best["short"].index.year == yr]
        sc = best["combined"][best["combined"].index.year == yr]
        if sc.empty:
            continue
        rl = float((1 + sl).prod() - 1)
        rs = float((1 + ss).prod() - 1)
        rc = float((1 + sc).prod() - 1)
        shl = float(np.sqrt(252) * sl.mean() / sl.std()) if sl.std() > 0 else np.nan
        shs = float(np.sqrt(252) * ss.mean() / ss.std()) if ss.std() > 0 else np.nan
        shc = float(np.sqrt(252) * sc.mean() / sc.std()) if sc.std() > 0 else np.nan
        print(f"  {yr:<6}  {rl:>9.2%}  {rs:>9.2%}  {rc:>10.2%}  "
              f"{shl:>9.3f}  {shs:>9.3f}  {shc:>9.3f}")
        yearly_rows.append(dict(Year=yr, Long=rl, Short=rs, Combined=rc,
                                Long_Sharpe=shl, Short_Sharpe=shs, Combined_Sharpe=shc))

    # ── Chart ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Contrarian Bubble Long/Short | MA={p['ma_window']}h  "
        f"buy<-{p['buy_threshold']}  short>{p['short_threshold']}  "
        f"hold={p['hold_hours']}h  top-{p['top_n']}\n"
        f"Long: Sharpe={lm.get('Sharpe', np.nan):.3f}  "
        f"Short: Sharpe={sm.get('Sharpe', np.nan):.3f}  "
        f"Combined: Sharpe={cm.get('Sharpe', np.nan):.3f}",
        fontsize=11, fontweight="bold"
    )

    # Wealth curves
    wl = (1 + best["long"]).cumprod()
    ws = (1 + best["short"]).cumprod()
    wc = (1 + best["combined"]).cumprod()

    axes[0, 0].plot(wl.index, wl.values, lw=2, color="steelblue", label="Long")
    axes[0, 0].plot(ws.index, ws.values, lw=2, color="tomato", label="Short")
    axes[0, 0].plot(wc.index, wc.values, lw=2, color="green", label="Combined", ls="--")
    axes[0, 0].set_title("Cumulative Wealth", fontweight="bold")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    # Drawdowns
    for w, color, label in [(wl, "steelblue", "Long"),
                             (ws, "tomato", "Short"),
                             (wc, "green", "Combined")]:
        dd = w / w.cummax() - 1
        axes[0, 1].plot(dd.index, dd.values * 100, lw=1.5, color=color, label=label)
    axes[0, 1].set_title("Drawdown (%)", fontweight="bold")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    # Yearly bar chart
    if yearly_rows:
        ydf = pd.DataFrame(yearly_rows)
        x = np.arange(len(ydf))
        w = 0.25
        axes[0, 2].bar(x - w, ydf["Long"] * 100, w, color="steelblue", alpha=0.8, label="Long")
        axes[0, 2].bar(x, ydf["Short"] * 100, w, color="tomato", alpha=0.8, label="Short")
        axes[0, 2].bar(x + w, ydf["Combined"] * 100, w, color="green", alpha=0.8, label="Combined")
        axes[0, 2].axhline(0, color="black", lw=0.8)
        axes[0, 2].set_xticks(x); axes[0, 2].set_xticklabels(ydf["Year"].astype(str))
        axes[0, 2].set_title("Yearly Returns (%)", fontweight="bold")
        axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3, axis="y")

        # Yearly Sharpe
        axes[1, 0].plot(ydf["Year"], ydf["Long_Sharpe"], marker="o", color="steelblue", label="Long")
        axes[1, 0].plot(ydf["Year"], ydf["Short_Sharpe"], marker="s", color="tomato", label="Short")
        axes[1, 0].plot(ydf["Year"], ydf["Combined_Sharpe"], marker="^", color="green", label="Combined")
        axes[1, 0].axhline(0, color="black", lw=0.8)
        axes[1, 0].set_title("Yearly Sharpe", fontweight="bold")
        axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    # Short threshold sensitivity
    st_sens = (grid_df.dropna(subset=["Short_Sharpe"])
               .groupby("short_threshold")["Short_Sharpe"]
               .agg(["mean", "max"]).reset_index())
    axes[1, 1].plot(st_sens["short_threshold"], st_sens["mean"],
                    marker="o", color="tomato", label="Avg Short Sharpe")
    axes[1, 1].plot(st_sens["short_threshold"], st_sens["max"],
                    marker="s", ls="--", color="darkred", label="Best Short Sharpe")
    axes[1, 1].axhline(0, color="black", lw=0.8)
    axes[1, 1].set_title("Short Threshold vs Short Sharpe", fontweight="bold")
    axes[1, 1].set_xlabel("Short Threshold")
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    # Rolling 30-day Sharpe comparison
    for s, color, label in [(best["long"], "steelblue", "Long"),
                             (best["short"], "tomato", "Short"),
                             (best["combined"], "green", "Combined")]:
        roll = s.rolling(30).mean() / s.rolling(30).std() * np.sqrt(252)
        axes[1, 2].plot(roll.index, roll.values, lw=1.2, color=color, label=label, alpha=0.8)
    axes[1, 2].axhline(0, color="black", lw=0.8)
    axes[1, 2].set_title("Rolling 30-Day Sharpe", fontweight="bold")
    axes[1, 2].legend(); axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = OUT_DIR / "contrarian_bubble_ls_hourly.png"
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n[Chart saved: {out_png}]")

out_xlsx = OUT_DIR / "contrarian_bubble_ls_grid.xlsx"
grid_df.to_excel(out_xlsx, index=False)
print(f"[Grid saved: {out_xlsx}]")
