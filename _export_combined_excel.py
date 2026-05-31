"""
Export comprehensive Excel report for the 3-strategy dynamic portfolio.
Best params: lookback=40d, hold=10d, max_alloc=50%

Sheets:
  1. Daily_Data          — daily weights, returns, cumulative wealth, drawdown
  2. Performance_Summary — Sharpe, Sortino, return, max DD, win rate per strategy
  3. Yearly_Stats        — return, Sharpe, max DD by year × strategy
  4. Rebalancing_History — weight at each rebalance date + period return
  5. Drawdown_Detail     — underwater curve per strategy + portfolio
  6. Grid_Results        — all 80 dynamic allocation combos ranked by Sharpe
  7. Charts              — embedded images: wealth, allocation, drawdown, yearly bars
"""
import sys, tempfile, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from pathlib import Path
from openpyxl.drawing.image import Image as XLImage

from data.db.schema import init
from data.universe import get_universe
from data.intraday_loader import load_daily_close, load_hourly_bars
from strategies.intraday_mean_reversion import run_intraday_mean_reversion
from portfolio.constructor import run_portfolio_allocation
from run_portfolio import get_momentum_returns, get_reddit_returns, START

OUT_DIR  = Path("results"); OUT_DIR.mkdir(exist_ok=True)
OUT_FILE = OUT_DIR / "combined_portfolio_report.xlsx"
TMP_DIR  = Path(tempfile.mkdtemp())

COLORS = {
    "Momentum":  "#1f77b4",
    "Reddit":    "#ff7f0e",
    "IntradayMR":"#2ca02c",
    "Portfolio": "#d62728",
}

# ── Helpers ────────────────────────────────────────────────────────────────
def cum_wealth(r):
    w = (1 + r).cumprod(); return w / w.iloc[0]

def drawdown(r):
    w = cum_wealth(r); return w / w.cummax() - 1

def save_fig(name: str, fig) -> str:
    path = str(TMP_DIR / f"{name}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path

def embed(ws, img_path: str, cell: str, width_px=900, height_px=500):
    img = XLImage(img_path)
    img.width  = width_px
    img.height = height_px
    ws.add_image(img, cell)

# ── Load strategy returns ──────────────────────────────────────────────────
init()
print("Loading strategies...")

print("  [1/3] Momentum...")
mom_ret = get_momentum_returns(START)

print("  [2/3] Reddit Sentiment...")
reddit_ret = get_reddit_returns(START)

print("  [3/3] Intraday MR...")
universe = get_universe()
daily_close = load_daily_close(universe, use_cache=True)
hourly_open, hourly_close = load_hourly_bars(universe, use_cache=True)
mr_ret_raw, _, _ = run_intraday_mean_reversion(
    daily_close=daily_close, hourly_open=hourly_open, hourly_close=hourly_close,
    sigma_grid=[4.0], flip_hold_days_grid=[2], lookback_grid=[20],
    top_n_grid=[5], transaction_cost=0.001, short_borrow_rate=0.08,
)
mr_ret = mr_ret_raw.rename("IntradayMR")

idx = mom_ret.index.intersection(reddit_ret.index).intersection(mr_ret.index)
mom_ret    = mom_ret.loc[idx]
reddit_ret = reddit_ret.loc[idx]
mr_ret     = mr_ret.loc[idx]
print(f"Common period: {idx[0].date()} → {idx[-1].date()} ({len(idx)} days)")

# ── Dynamic portfolio ──────────────────────────────────────────────────────
print("\nRunning dynamic portfolio (lookback=40, hold=10, max_alloc=50%)...")
(_, _, best_ret, best_wealth, best_weights_ts, grid_df, best_params) = \
    run_portfolio_allocation(
        returns_dict={"Momentum": mom_ret, "Reddit": reddit_ret, "IntradayMR": mr_ret},
        method="momentum",
        lookback_grid=[40], hold_period_grid=[10], max_alloc_grid=[0.5],
    )
port_ret = best_ret.rename("Portfolio")

strat_names = ["Momentum", "Reddit", "IntradayMR"]
strategies  = {"Momentum": mom_ret, "Reddit": reddit_ret,
               "IntradayMR": mr_ret, "Portfolio": port_ret}

# Daily weights (forward-fill from rebalancing dates)
daily_weights = pd.DataFrame(index=idx, columns=strat_names, dtype=float)
if best_weights_ts is not None:
    for col in strat_names:
        if col in best_weights_ts.columns:
            for dt, row in best_weights_ts.iterrows():
                daily_weights.loc[daily_weights.index >= dt, col] = float(row[col])
daily_weights = daily_weights.ffill().fillna(1/3)

# ── Performance stats ──────────────────────────────────────────────────────
def perf_stats(name, ret):
    w = cum_wealth(ret); mdd = float((w / w.cummax() - 1).min())
    sh  = float(np.sqrt(252)*ret.mean()/ret.std()) if ret.std()>0 else np.nan
    so  = float(np.sqrt(252)*ret.mean()/ret[ret<0].std()) if ret[ret<0].std()>0 else np.nan
    n_yr = len(ret)/252
    ann = float((1+ret).prod()**(1/n_yr)-1) if n_yr>0 else np.nan
    return {"Strategy": name, "Total Return": float(w.iloc[-1]-1),
            "Annualized Return": ann, "Sharpe Ratio": sh, "Sortino Ratio": so,
            "Max Drawdown": mdd, "Avg Daily Return": float(ret.mean()),
            "Daily Std": float(ret.std()), "Win Rate": float((ret>0).mean()),
            "Best Day": float(ret.max()), "Worst Day": float(ret.min()),
            "Positive Days": int((ret>0).sum()), "Negative Days": int((ret<0).sum()),
            "Total Days": len(ret)}

perf_df = pd.DataFrame([perf_stats(n,r) for n,r in strategies.items()]).set_index("Strategy")

# Yearly stats
yearly_rows = []
for name, ret in strategies.items():
    for yr, grp in ret.groupby(ret.index.year):
        w  = cum_wealth(grp); mdd = float((w/w.cummax()-1).min())
        sh = float(np.sqrt(252)*grp.mean()/grp.std()) if grp.std()>0 else np.nan
        so_s = grp[grp<0].std()
        so = float(np.sqrt(252)*grp.mean()/so_s) if so_s>0 else np.nan
        yearly_rows.append({"Strategy":name,"Year":yr,
            "Return":float((1+grp).prod()-1),"Sharpe":sh,"Sortino":so,
            "Max Drawdown":mdd,"Trading Days":len(grp),"Win Rate":float((grp>0).mean())})
yearly_df = pd.DataFrame(yearly_rows)

# Daily data sheet
daily = pd.DataFrame(index=idx); daily.index.name="Date"
for s in strat_names: daily[f"Weight_{s}(%)"] = daily_weights[s]*100
for name, ret in strategies.items():
    r = ret.reindex(idx).fillna(0)
    daily[f"Return_{name}(%)"]    = r*100
    daily[f"CumWealth_{name}"]    = cum_wealth(r)
    daily[f"Drawdown_{name}(%)"]  = drawdown(r)*100

# Rebalancing history
if best_weights_ts is not None and isinstance(best_weights_ts.index, pd.DatetimeIndex):
    reb_df = best_weights_ts.copy(); reb_df.index.name="Rebalance_Date"
    for col in strat_names:
        if col not in reb_df.columns: reb_df[col] = np.nan
    period_rets = []
    for i, dt in enumerate(reb_df.index):
        nxt = reb_df.index[i+1] if i+1<len(reb_df.index) else idx[-1]
        pr  = port_ret.loc[(port_ret.index>=dt)&(port_ret.index<nxt)]
        period_rets.append(float((1+pr).prod()-1) if len(pr)>0 else 0.0)
    reb_df["Period_Return(%)"] = [r*100 for r in period_rets]
    for col in strat_names: reb_df[col] = reb_df[col]*100
else:
    reb_df = pd.DataFrame()

# Drawdown detail
dd_df = pd.DataFrame(index=idx); dd_df.index.name="Date"
for name, ret in strategies.items():
    dd_df[f"DD_{name}(%)"] = drawdown(ret.reindex(idx).fillna(0))*100

# ══════════════════════════════════════════════════════════════════════════
# BUILD CHARTS
# ══════════════════════════════════════════════════════════════════════════
print("\nGenerating charts...")

# Chart 1: Cumulative wealth
fig, ax = plt.subplots(figsize=(14, 6))
for name, ret in strategies.items():
    w  = cum_wealth(ret.reindex(idx).fillna(0))
    lw = 2.5 if name == "Portfolio" else 1.5
    ax.plot(w.index, w.values, label=name, color=COLORS[name], linewidth=lw)
ax.set_title("Cumulative Wealth — All Strategies + Portfolio (lookback=40d, hold=10d, max_alloc=50%)",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Wealth (1 = start)"); ax.legend(fontsize=10); ax.grid(True, alpha=0.4)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.1f}x"))
fig.tight_layout()
chart1 = save_fig("chart_wealth", fig)

# Chart 2: Dynamic allocation over time (stacked area)
fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios":[2,1]})
ax1, ax2 = axes
ax1.stackplot(daily_weights.index,
              daily_weights["Momentum"]*100,
              daily_weights["Reddit"]*100,
              daily_weights["IntradayMR"]*100,
              labels=strat_names,
              colors=[COLORS["Momentum"], COLORS["Reddit"], COLORS["IntradayMR"]],
              alpha=0.8)
ax1.set_title("Dynamic Allocation Over Time (%)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Allocation (%)"); ax1.set_ylim(0, 100); ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.4)
# Per-strategy weight line
for s in strat_names:
    ax2.plot(daily_weights.index, daily_weights[s]*100,
             label=s, color=COLORS[s], linewidth=1.2)
ax2.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="50% cap")
ax2.set_ylabel("Weight (%)"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.4)
fig.tight_layout()
chart2 = save_fig("chart_allocation", fig)

# Chart 3: Drawdown comparison
fig, ax = plt.subplots(figsize=(14, 5))
for name, ret in strategies.items():
    dd = drawdown(ret.reindex(idx).fillna(0))*100
    lw = 2.5 if name == "Portfolio" else 1.2
    ax.plot(dd.index, dd.values, label=name, color=COLORS[name], linewidth=lw)
_port_dd = drawdown(port_ret.reindex(idx).fillna(0)) * 100
ax.fill_between(_port_dd.index, _port_dd.values, 0, alpha=0.15, color=COLORS["Portfolio"])
ax.set_title("Drawdown Comparison (%)", fontsize=12, fontweight="bold")
ax.set_ylabel("Drawdown (%)"); ax.legend(fontsize=10); ax.grid(True, alpha=0.4)
fig.tight_layout()
chart3 = save_fig("chart_drawdown", fig)

# Chart 4: Yearly return bar chart
fig, ax = plt.subplots(figsize=(12, 5))
yr_pivot = yearly_df.pivot(index="Year", columns="Strategy", values="Return")*100
yr_pivot = yr_pivot[["Momentum","Reddit","IntradayMR","Portfolio"]]
x = np.arange(len(yr_pivot))
w_bar = 0.2
for i, col in enumerate(yr_pivot.columns):
    ax.bar(x + i*w_bar, yr_pivot[col], width=w_bar,
           label=col, color=COLORS[col], alpha=0.85)
ax.set_xticks(x + w_bar*1.5)
ax.set_xticklabels([str(y) for y in yr_pivot.index])
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Yearly Returns by Strategy (%)", fontsize=12, fontweight="bold")
ax.set_ylabel("Return (%)"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
chart4 = save_fig("chart_yearly_returns", fig)

# Chart 5: Yearly Sharpe comparison
fig, ax = plt.subplots(figsize=(12, 5))
yr_sh = yearly_df.pivot(index="Year", columns="Strategy", values="Sharpe")
yr_sh = yr_sh[["Momentum","Reddit","IntradayMR","Portfolio"]]
for i, col in enumerate(yr_sh.columns):
    ax.bar(x + i*w_bar, yr_sh[col], width=w_bar,
           label=col, color=COLORS[col], alpha=0.85)
ax.set_xticks(x + w_bar*1.5)
ax.set_xticklabels([str(y) for y in yr_sh.index])
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Yearly Sharpe Ratio by Strategy", fontsize=12, fontweight="bold")
ax.set_ylabel("Sharpe Ratio"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
chart5 = save_fig("chart_yearly_sharpe", fig)

# Chart 6: Yearly max drawdown
fig, ax = plt.subplots(figsize=(12, 5))
yr_dd = yearly_df.pivot(index="Year", columns="Strategy", values="Max Drawdown")*100
yr_dd = yr_dd[["Momentum","Reddit","IntradayMR","Portfolio"]]
for i, col in enumerate(yr_dd.columns):
    ax.bar(x + i*w_bar, yr_dd[col], width=w_bar,
           label=col, color=COLORS[col], alpha=0.85)
ax.set_xticks(x + w_bar*1.5)
ax.set_xticklabels([str(y) for y in yr_dd.index])
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Yearly Max Drawdown by Strategy (%)", fontsize=12, fontweight="bold")
ax.set_ylabel("Max Drawdown (%)"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
chart6 = save_fig("chart_yearly_dd", fig)

# Chart 7: Rolling 60-day Sharpe
fig, ax = plt.subplots(figsize=(14, 5))
for name, ret in strategies.items():
    rs = ret.reindex(idx).fillna(0).rolling(60)
    roll_sh = np.sqrt(252) * rs.mean() / rs.std()
    lw = 2.5 if name == "Portfolio" else 1.2
    ax.plot(roll_sh.index, roll_sh.values, label=name, color=COLORS[name], linewidth=lw)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Rolling 60-Day Sharpe Ratio", fontsize=12, fontweight="bold")
ax.set_ylabel("Sharpe"); ax.legend(fontsize=10); ax.grid(True, alpha=0.4)
fig.tight_layout()
chart7 = save_fig("chart_rolling_sharpe", fig)

# ══════════════════════════════════════════════════════════════════════════
# WRITE EXCEL
# ══════════════════════════════════════════════════════════════════════════
print(f"\nWriting Excel → {OUT_FILE}")

with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
    wb = writer.book

    # Sheet 1: Daily Data
    daily.to_excel(writer, sheet_name="Daily_Data")
    ws = writer.sheets["Daily_Data"]
    ws.column_dimensions["A"].width = 13
    for c in "BCDEFGHIJKLMNOPQRSTUVWXYZ": ws.column_dimensions[c].width = 16

    # Sheet 2: Performance Summary
    perf_df.to_excel(writer, sheet_name="Performance_Summary")
    ws2 = writer.sheets["Performance_Summary"]
    ws2.column_dimensions["A"].width = 15
    for c in "BCDEFGHIJKLMN": ws2.column_dimensions[c].width = 20
    # Embed wealth chart next to table
    embed(ws2, chart1, "A8",  width_px=900, height_px=400)
    embed(ws2, chart7, "A32", width_px=900, height_px=350)

    # Sheet 3: Yearly Stats
    yearly_df.to_excel(writer, sheet_name="Yearly_Stats", index=False)
    ws3 = writer.sheets["Yearly_Stats"]
    for c in "ABCDEFGH": ws3.column_dimensions[c].width = 15
    # Embed yearly charts
    n_rows = len(yearly_df) + 3
    embed(ws3, chart4, f"A{n_rows+1}",  width_px=900, height_px=370)
    embed(ws3, chart5, f"A{n_rows+25}", width_px=900, height_px=370)
    embed(ws3, chart6, f"A{n_rows+49}", width_px=900, height_px=370)

    # Sheet 4: Rebalancing History
    if not reb_df.empty:
        reb_df.to_excel(writer, sheet_name="Rebalancing_History")
        ws4 = writer.sheets["Rebalancing_History"]
        ws4.column_dimensions["A"].width = 20
        for c in "BCDE": ws4.column_dimensions[c].width = 16
        # Embed allocation chart
        n_reb = len(reb_df) + 3
        embed(ws4, chart2, f"A{n_reb+2}", width_px=950, height_px=500)

    # Sheet 5: Drawdown Detail
    dd_df.to_excel(writer, sheet_name="Drawdown_Detail")
    ws5 = writer.sheets["Drawdown_Detail"]
    ws5.column_dimensions["A"].width = 13
    for c in "BCDE": ws5.column_dimensions[c].width = 18
    embed(ws5, chart3, f"A{len(dd_df)+3}", width_px=950, height_px=400)

    # Sheet 6: Grid Results
    if grid_df is not None and not grid_df.empty:
        grid_df.to_excel(writer, sheet_name="Grid_Results", index=False)
        ws6 = writer.sheets["Grid_Results"]
        for c in "ABCDEFGHIJ": ws6.column_dimensions[c].width = 14

# Cleanup temp PNGs
for f in TMP_DIR.iterdir():
    try: f.unlink()
    except: pass
TMP_DIR.rmdir()

# ── Print summary ──────────────────────────────────────────────────────────
print("\nPerformance Summary:")
print(perf_df[["Total Return","Annualized Return","Sharpe Ratio",
               "Sortino Ratio","Max Drawdown","Win Rate"]].to_string())

print("\nYearly Stats — Portfolio:")
print(yearly_df[yearly_df["Strategy"]=="Portfolio"][
    ["Year","Return","Sharpe","Max Drawdown","Trading Days"]].to_string(index=False))

print(f"\nSaved → {OUT_FILE.resolve()}")
print("7 charts embedded across sheets.")
