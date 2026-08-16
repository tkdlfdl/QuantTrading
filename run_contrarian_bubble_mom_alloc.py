"""
Contrarian Bubble L/S with Momentum-Based Dynamic Allocation
=============================================================
Step 1: Run L/S backtest (fixed best params) to get raw long/short daily series.
Step 2: Grid-search momentum allocation rules on top:
  - lookback_days : rolling window to measure each side's momentum
  - rebalance_days: how many days to hold allocation before recalculating

Allocation rule (at each rebalance point):
  roll_long  = cumulative return of long side over last lookback_days
  roll_short = cumulative return of short side over last lookback_days
  w_long     = max(0, roll_long)  / (max(0, roll_long) + max(0, roll_short))
  w_short    = max(0, roll_short) / (max(0, roll_long) + max(0, roll_short))
  If both negative  -> go FLAT (w_long = w_short = 0)
  Each side can have 0-100% weight; they always sum to 100% (or 0 if flat).

Grid:
  lookback_days  : 5, 10, 20, 30, 60, 90, 120  (days -> weeks -> months)
  rebalance_days : 1, 5, 10, 20, 60             (daily -> weekly -> monthly -> quarterly)
  Total          : 7 x 5 = 35 combos

Usage:
  python run_contrarian_bubble_mom_alloc.py
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product as iproduct

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from pathlib import Path

from data.universe import get_universe
from strategies.contrarian_bubble_ls_hourly import run_contrarian_bubble_ls_hourly

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

TRADING_DAYS = 252

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

# ── Step 1: L/S backtest with best params ──────────────────────────────────
# Best from prior grid: MA=156h, buy=0.8, short=0.8, hold=8h, top_n=20
print("\nStep 1: Running L/S backtest with best params...")
best_ls, _ = run_contrarian_bubble_ls_hourly(
    hourly_open=ho,
    hourly_close=hc,
    ma_window_grid=[156],
    buy_threshold_grid=[0.8],
    short_threshold_grid=[0.8],
    hold_hours_grid=[8],
    top_n_grid=[20],
    transaction_cost=0.001,
    short_borrow_rate=0.08,
)

if best_ls is None:
    print("ERROR: L/S backtest returned no results.")
    sys.exit(1)

s_long  = best_ls["long"]    # daily return series
s_short = best_ls["short"]   # daily return series
s_fixed = best_ls["combined"] # fixed 50/50 baseline

print(f"Long  series: {len(s_long)} days  "
      f"({s_long.index[0].date()} -> {s_long.index[-1].date()})")
print(f"Short series: {len(s_short)} days")

# ── Step 2: Momentum allocation engine ────────────────────────────────────
def momentum_alloc(s_l: pd.Series, s_s: pd.Series,
                   lookback: int, rebalance: int) -> pd.Series:
    """
    Dynamic long/short allocation based on rolling lookback-day momentum.
    Returns daily portfolio return series.
    """
    n = len(s_l)
    l_vals = s_l.values
    s_vals = s_s.values
    w_long  = np.zeros(n)
    w_short = np.zeros(n)

    current_wl = 0.5
    current_ws = 0.5

    for i in range(n):
        # Recompute allocation every `rebalance` days once we have enough lookback
        if i >= lookback and i % rebalance == 0:
            roll_l = float(np.prod(1 + l_vals[i - lookback:i]) - 1)
            roll_s = float(np.prod(1 + s_vals[i - lookback:i]) - 1)
            pos_l = max(0.0, roll_l)
            pos_s = max(0.0, roll_s)
            total = pos_l + pos_s
            if total > 1e-9:
                current_wl = pos_l / total
                current_ws = pos_s / total
            else:
                current_wl = 0.0   # both sides negative -> flat
                current_ws = 0.0

        w_long[i]  = current_wl
        w_short[i] = current_ws

    port = w_long * l_vals + w_short * s_vals
    return pd.Series(port, index=s_l.index, name=f"MomAlloc_lb{lookback}_rb{rebalance}")


def metrics(s: pd.Series) -> dict:
    wealth = (1 + s).cumprod()
    std = s.std()
    sh  = float(np.sqrt(TRADING_DAYS) * s.mean() / std) if std > 0 else np.nan
    ds  = s[s < 0].std(ddof=0)
    so  = float(np.sqrt(TRADING_DAYS) * s.mean() / ds)  if ds  > 0 else np.nan
    mdd = float((wealth / wealth.cummax() - 1).min())
    tr  = float(wealth.iloc[-1] - 1)
    yrs = len(s) / TRADING_DAYS
    cagr = float((1 + tr) ** (1 / yrs) - 1) if tr > -1 and yrs > 0 else np.nan
    act  = s[s != 0]
    wr   = float((act > 0).mean()) if len(act) > 0 else np.nan
    return dict(Sharpe=sh, Sortino=so, CAGR=cagr, Total_Return=tr,
                Max_DD=mdd, Win_Rate=wr)


# ── Grid search ────────────────────────────────────────────────────────────
LOOKBACK_GRID   = [5, 10, 20, 30, 60, 90, 120]
REBALANCE_GRID  = [1, 5, 10, 20, 60]

print(f"\nStep 2: Momentum allocation grid search "
      f"({len(LOOKBACK_GRID)} x {len(REBALANCE_GRID)} = "
      f"{len(LOOKBACK_GRID)*len(REBALANCE_GRID)} combos)...")

results = []
best_sharpe = -np.inf
best_series = None
best_params = None

for lb, rb in iproduct(LOOKBACK_GRID, REBALANCE_GRID):
    s_port = momentum_alloc(s_long, s_short, lb, rb)
    m = metrics(s_port)
    row = dict(lookback_days=lb, rebalance_days=rb, **m)
    results.append(row)
    if pd.notna(m["Sharpe"]) and m["Sharpe"] > best_sharpe:
        best_sharpe = m["Sharpe"]
        best_series = s_port
        best_params = row

grid_df = pd.DataFrame(results).sort_values("Sharpe", ascending=False)

# Baselines
m_long   = metrics(s_long)
m_short  = metrics(s_short)
m_fixed  = metrics(s_fixed)

# ── Print results ──────────────────────────────────────────────────────────
S = "=" * 80
print(f"\n{S}")
print("BASELINES")
print(S)
print(f"  {'Strategy':<22} {'Sharpe':>8} {'CAGR':>8} {'Total Ret':>10} {'Max DD':>9} {'Win Rate':>9}")
print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*10} {'-'*9} {'-'*9}")
for label, m in [("Long-only", m_long), ("Short-only", m_short), ("Fixed 50/50", m_fixed)]:
    print(f"  {label:<22} {m['Sharpe']:>8.4f} {m['CAGR']:>8.2%} "
          f"{m['Total_Return']:>10.2%} {m['Max_DD']:>9.2%} {m['Win_Rate']:>9.2%}")

print(f"\n{S}")
print("TOP 20 MOMENTUM ALLOCATION COMBOS (by Sharpe)")
print(S)
print(f"  {'lookback':>8} {'rebalance':>10} {'Sharpe':>8} {'CAGR':>8} "
      f"{'Total Ret':>10} {'Max DD':>9} {'Win Rate':>9}")
print(f"  {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*9} {'-'*9}")
for _, row in grid_df.head(20).iterrows():
    print(f"  {int(row.lookback_days):>8}d {int(row.rebalance_days):>9}d "
          f"{row.Sharpe:>8.4f} {row.CAGR:>8.2%} "
          f"{row.Total_Return:>10.2%} {row.Max_DD:>9.2%} {row.Win_Rate:>9.2%}")

# ── Sensitivity ─────────────────────────────────────────────────────────────
print(f"\n{S}\nPARAMETER SENSITIVITY\n{S}")

print("\n  Lookback (days) -> Avg/Best Sharpe:")
for lb, grp in grid_df.groupby("lookback_days"):
    label = f"{lb}d"
    if lb >= 20:
        label += f" (~{lb//5}wk)" if lb < 60 else f" (~{lb//20}mo)"
    print(f"    {label:<12}  avg={grp.Sharpe.mean():.4f}  best={grp.Sharpe.max():.4f}")

print("\n  Rebalance (days) -> Avg/Best Sharpe:")
for rb, grp in grid_df.groupby("rebalance_days"):
    label = f"{rb}d"
    if rb == 5:   label += " (weekly)"
    if rb == 20:  label += " (monthly)"
    if rb == 60:  label += " (quarterly)"
    print(f"    {label:<20}  avg={grp.Sharpe.mean():.4f}  best={grp.Sharpe.max():.4f}")

# ── Best combo yearly breakdown ─────────────────────────────────────────────
if best_series is not None:
    print(f"\n{S}")
    print(f"BEST COMBO YEARLY BREAKDOWN  "
          f"(lookback={int(best_params['lookback_days'])}d  "
          f"rebalance={int(best_params['rebalance_days'])}d)")
    print(S)
    print(f"  {'Year':<6}  {'MomAlloc':>10}  {'Fixed50/50':>10}  "
          f"{'Long-only':>10}  {'Sharpe(MA)':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for yr in sorted(best_series.index.year.unique()):
        sm = best_series[best_series.index.year == yr]
        sf = s_fixed[s_fixed.index.year == yr]
        sl = s_long[s_long.index.year == yr]
        rm = float((1 + sm).prod() - 1)
        rf = float((1 + sf).prod() - 1)
        rl = float((1 + sl).prod() - 1)
        sh = float(np.sqrt(252) * sm.mean() / sm.std()) if sm.std() > 0 else np.nan
        print(f"  {yr:<6}  {rm:>10.2%}  {rf:>10.2%}  {rl:>10.2%}  {sh:>10.3f}")

# ── Chart ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    f"Contrarian Bubble L/S -- Momentum Allocation\n"
    f"MA=156h | buy<-0.8 | short>+0.8 | hold=8h | top-20\n"
    f"Best alloc: lookback={int(best_params['lookback_days'])}d  "
    f"rebalance={int(best_params['rebalance_days'])}d  "
    f"Sharpe={best_params['Sharpe']:.3f}  CAGR={best_params['CAGR']:.1%}  "
    f"MaxDD={best_params['Max_DD']:.2%}",
    fontsize=11, fontweight="bold"
)

# Wealth curves
wl  = (1 + s_long).cumprod()
ws  = (1 + s_short).cumprod()
wf  = (1 + s_fixed).cumprod()
wm  = (1 + best_series).cumprod()
axes[0, 0].plot(wl.index, wl.values,  lw=1.5, color="steelblue", label="Long-only",  alpha=0.7)
axes[0, 0].plot(ws.index, ws.values,  lw=1.5, color="tomato",    label="Short-only", alpha=0.7)
axes[0, 0].plot(wf.index, wf.values,  lw=1.5, color="gray",      label="Fixed 50/50",ls="--", alpha=0.7)
axes[0, 0].plot(wm.index, wm.values,  lw=2.5, color="green",     label=f"MomAlloc (best)")
axes[0, 0].set_title("Cumulative Wealth", fontweight="bold")
axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3)

# Drawdown comparison
for w, color, label, ls in [
    (wl, "steelblue", "Long-only", "-"),
    (wf, "gray", "Fixed 50/50", "--"),
    (wm, "green", "MomAlloc", "-"),
]:
    dd = w / w.cummax() - 1
    axes[0, 1].plot(dd.index, dd.values * 100, lw=1.5, color=color, label=label, ls=ls)
axes[0, 1].set_title("Drawdown (%)", fontweight="bold")
axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.3)

# Lookback sensitivity heatmap-style
lb_avg = grid_df.groupby("lookback_days")["Sharpe"].mean()
lb_best = grid_df.groupby("lookback_days")["Sharpe"].max()
axes[0, 2].bar([str(x) + "d" for x in lb_avg.index], lb_avg.values,
               color="steelblue", alpha=0.6, label="Avg Sharpe")
axes[0, 2].plot([str(x) + "d" for x in lb_best.index], lb_best.values,
                marker="o", color="darkblue", lw=2, label="Best Sharpe")
axes[0, 2].axhline(m_fixed["Sharpe"], color="gray", ls="--", lw=1, label="Fixed 50/50")
axes[0, 2].set_title("Lookback Sensitivity", fontweight="bold")
axes[0, 2].legend(fontsize=8); axes[0, 2].grid(True, alpha=0.3, axis="y")

# Rebalance sensitivity
rb_avg = grid_df.groupby("rebalance_days")["Sharpe"].mean()
rb_best = grid_df.groupby("rebalance_days")["Sharpe"].max()
rb_labels = ["1d\n(daily)", "5d\n(weekly)", "10d\n(bi-wk)", "20d\n(monthly)", "60d\n(qtrly)"]
axes[1, 0].bar(rb_labels, rb_avg.values, color="tomato", alpha=0.6, label="Avg Sharpe")
axes[1, 0].plot(rb_labels, rb_best.values, marker="s", color="darkred", lw=2, label="Best Sharpe")
axes[1, 0].axhline(m_fixed["Sharpe"], color="gray", ls="--", lw=1, label="Fixed 50/50")
axes[1, 0].set_title("Rebalance Frequency Sensitivity", fontweight="bold")
axes[1, 0].legend(fontsize=8); axes[1, 0].grid(True, alpha=0.3, axis="y")

# Heatmap of Sharpe: lookback x rebalance
pivot = grid_df.pivot(index="lookback_days", columns="rebalance_days", values="Sharpe")
im = axes[1, 1].imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                        vmin=grid_df.Sharpe.quantile(0.1),
                        vmax=grid_df.Sharpe.quantile(0.9))
axes[1, 1].set_xticks(range(len(pivot.columns)))
axes[1, 1].set_xticklabels([f"{c}d" for c in pivot.columns])
axes[1, 1].set_yticks(range(len(pivot.index)))
axes[1, 1].set_yticklabels([f"{r}d" for r in pivot.index])
axes[1, 1].set_xlabel("Rebalance Days"); axes[1, 1].set_ylabel("Lookback Days")
axes[1, 1].set_title("Sharpe Heatmap (Lookback x Rebalance)", fontweight="bold")
plt.colorbar(im, ax=axes[1, 1])
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        axes[1, 1].text(j, i, f"{pivot.values[i,j]:.2f}",
                        ha="center", va="center", fontsize=7, color="black")

# Dynamic weight over time (best combo)
lb_b = int(best_params["lookback_days"])
rb_b = int(best_params["rebalance_days"])
# Recompute weights for plotting
wt_long  = np.zeros(len(s_long))
wt_short = np.zeros(len(s_long))
cur_wl = 0.5; cur_ws = 0.5
l_v = s_long.values; s_v = s_short.values
for i in range(len(s_long)):
    if i >= lb_b and i % rb_b == 0:
        rl = float(np.prod(1 + l_v[i-lb_b:i]) - 1)
        rs = float(np.prod(1 + s_v[i-lb_b:i]) - 1)
        pl = max(0.0, rl); ps = max(0.0, rs); tot = pl + ps
        if tot > 1e-9:
            cur_wl = pl / tot; cur_ws = ps / tot
        else:
            cur_wl = 0.0; cur_ws = 0.0
    wt_long[i] = cur_wl; wt_short[i] = cur_ws
axes[1, 2].stackplot(s_long.index,
                     [wt_long * 100, wt_short * 100],
                     labels=["Long %", "Short %"],
                     colors=["steelblue", "tomato"], alpha=0.7)
axes[1, 2].set_title(f"Dynamic Allocation Over Time\n"
                     f"(lb={lb_b}d, rb={rb_b}d)", fontweight="bold")
axes[1, 2].set_ylabel("Allocation (%)")
axes[1, 2].legend(fontsize=8); axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
out_png = OUT_DIR / "contrarian_bubble_mom_alloc.png"
plt.savefig(out_png, dpi=130, bbox_inches="tight")
plt.close()
print(f"\n[Chart saved: {out_png}]")

grid_df.to_excel(OUT_DIR / "contrarian_bubble_mom_alloc_grid.xlsx", index=False)
print(f"[Grid saved: results/contrarian_bubble_mom_alloc_grid.xlsx]")
