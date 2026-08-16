"""
QQQ Bubble - Extensive Grid Search with Yearly Breakdown
=========================================================
Large grid search to find optimal bubble detection parameters
Shows: Sharpe, Max Drawdown, Return by year and overall
"""
import sys, warnings, os
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_n = [0]
def _save(*a, **k):
    _n[0] += 1; p = f"results/qqq_bubble_grid_chart_{_n[0]}.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); print(f"  [chart saved: {p}]", flush=True)
plt.show = _save

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

os.makedirs("results", exist_ok=True)

TRADING_DAYS = 252
TC = 0.0025  # 0.25% one-way

def _sharpe(r, td=TRADING_DAYS):
    s = r.std(); return float(np.sqrt(td)*r.mean()/s) if s > 0 else np.nan

def _sortino(r, td=TRADING_DAYS):
    ds = r[r<0].std(); return float(np.sqrt(td)*r.mean()/ds) if ds > 0 else np.nan

def _mdd(r):
    w = (1+r).cumprod(); w = w/w.iloc[0]
    return float((w/w.cummax()-1).min())

def _stats(r):
    if len(r) < 5: return {"Sharpe": np.nan, "Sortino": np.nan,
                           "Total Return": np.nan, "Max DD": np.nan}
    w = (1+r).cumprod(); w = w/w.iloc[0]
    return {"Sharpe": _sharpe(r), "Sortino": _sortino(r),
            "Total Return": float(w.iloc[-1]-1), "Max DD": _mdd(r)}

def _header(t):
    print(f"\n{'='*90}\n{t}\n{'='*90}", flush=True)


_header("QQQ BUBBLE - EXTENSIVE GRID SEARCH (2018-2026)")

# Load QQQ proxy
print("Loading data...")
daily_close = pd.read_parquet(Path("data/cache/daily_close.parquet"))
daily_close.index = pd.to_datetime(daily_close.index)

nasdaq100 = ["AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AVGO", "ASML"]
available = [t for t in nasdaq100 if t in daily_close.columns]
qqq_prices = daily_close[available].mean(axis=1)

print(f"QQQ Proxy: {qqq_prices.index[0].date()} to {qqq_prices.index[-1].date()} ({len(qqq_prices)} days)")

# EXTENSIVE GRID
MA_WINDOWS = [10, 15, 20, 30, 50, 75, 100, 150, 200]
Z_WINDOWS = [30, 50, 75, 100, 150, 200, 250]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
HOLD_DAYS = 40

total_combos = len(MA_WINDOWS) * len(Z_WINDOWS) * len(THRESHOLDS)
print(f"\nGrid: {len(MA_WINDOWS)} MA x {len(Z_WINDOWS)} Z x {len(THRESHOLDS)} Thresh = {total_combos} combos")

grid_results = []
best_ret = None
best_sh = -np.inf
best_params = None

combo = 0
for ma_win in MA_WINDOWS:
    for z_win in Z_WINDOWS:
        for thresh in THRESHOLDS:
            combo += 1
            if combo % 50 == 0:
                print(f"  {combo}/{total_combos}...")

            # Compute bubble score
            log_c = np.log(qqq_prices.replace(0, np.nan).ffill())
            fair = qqq_prices.rolling(ma_win).mean()
            residual = log_c - np.log(fair)
            z = ((residual - residual.rolling(z_win).mean()) /
                 residual.rolling(z_win).std())
            bubble_score = np.tanh(z / 2)

            # Generate signals
            trade_rets = []
            i = max(ma_win, z_win)
            while i < len(qqq_prices) - HOLD_DAYS:
                score = bubble_score.iloc[i]

                if pd.isna(score) or abs(score) < thresh:
                    i += 1
                    continue

                if score < -thresh:  # Long signal
                    entry_price = qqq_prices.iloc[i]
                    exit_price = qqq_prices.iloc[i + HOLD_DAYS]

                    if entry_price > 0 and exit_price > 0:
                        raw_ret = (exit_price / entry_price - 1)
                        net_ret = raw_ret - TC
                        trade_rets.append({
                            "date": qqq_prices.index[i],
                            "ret": net_ret,
                        })

                i += HOLD_DAYS

            if len(trade_rets) < 3:
                grid_results.append({
                    "ma": ma_win, "z": z_win, "thresh": thresh,
                    "Sharpe": np.nan, "Sortino": np.nan, "Return": np.nan, "MaxDD": np.nan,
                    "n_trades": len(trade_rets),
                })
                continue

            # Daily returns
            tdf = pd.DataFrame(trade_rets)
            daily_ret = (tdf.assign(dt=pd.to_datetime(tdf["date"]).dt.normalize())
                            .groupby("dt")["ret"].mean()
                            .reindex(pd.date_range(tdf["date"].min(),
                                                   qqq_prices.index[-1], freq="B"),
                                     fill_value=0.0))

            s = _stats(daily_ret)
            grid_results.append({
                "ma": ma_win, "z": z_win, "thresh": thresh,
                "Sharpe": s["Sharpe"], "Sortino": s["Sortino"],
                "Return": s["Total Return"], "MaxDD": s["Max DD"],
                "n_trades": len(trade_rets),
            })

            if pd.notna(s["Sharpe"]) and s["Sharpe"] > best_sh:
                best_sh = s["Sharpe"]
                best_ret = daily_ret.rename("QQQ_Bubble_Best")
                best_params = {"ma": ma_win, "z": z_win, "thresh": thresh, "n_trades": len(trade_rets)}

grid_df = pd.DataFrame(grid_results).sort_values("Sharpe", ascending=False)

print(f"\nExtensive Grid Complete!")
print(f"Valid results: {len(grid_df[grid_df['Sharpe'].notna()])}/{total_combos}")


_header("TOP 20 RESULTS")

print(f"\n{'MA':>4} {'Z':>5} {'Thr':>5} | {'Sharpe':>8} {'Sortino':>9} {'Return':>10} {'MaxDD':>8} {'Trades':>7}")
print("-" * 70)
for i, (_, row) in enumerate(grid_df.head(20).iterrows()):
    if pd.isna(row['Sharpe']):
        continue
    print(f"{int(row['ma']):>4} {int(row['z']):>5} {row['thresh']:>5.1f} | "
          f"{row['Sharpe']:>8.3f} {row['Sortino']:>9.3f} {row['Return']:>10.2%} "
          f"{row['MaxDD']:>8.2%} {int(row['n_trades']):>7}")

if best_params:
    s_best = _stats(best_ret)
    print(f"\n{'='*70}")
    print(f"BEST: MA={best_params['ma']}d, Z={best_params['z']}d, Threshold={best_params['thresh']}")
    print(f"Sharpe={s_best['Sharpe']:.4f} | Return={s_best['Total Return']:+.2%} | MaxDD={s_best['Max DD']:.2%} | Trades={best_params['n_trades']}")
    print(f"Period: {best_ret.index[0].date()} to {best_ret.index[-1].date()}")


_header("YEARLY BREAKDOWN - BEST STRATEGY")

# Compute yearly metrics for best strategy
years = sorted(best_ret.index.year.unique())
yearly_rows = []

print(f"\n{'Year':>6} | {'Return':>10} {'Sharpe':>10} {'Sortino':>10} {'MaxDD':>9} {'Days':>6}")
print("-" * 68)

for yr in years:
    yr_ret = best_ret[best_ret.index.year == yr]
    if len(yr_ret) < 5:
        print(f"{yr:>6} | {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>9} {len(yr_ret):>6}")
        continue

    w = (1+yr_ret).cumprod(); w = w/w.iloc[0]
    ret = float(w.iloc[-1]-1)
    sh = _sharpe(yr_ret)
    so = _sortino(yr_ret)
    dd = float((w/w.cummax()-1).min())

    yearly_rows.append({
        "Year": yr, "Return": ret, "Sharpe": sh, "Sortino": so, "MaxDD": dd, "Days": len(yr_ret)
    })

    print(f"{yr:>6} | {ret:>10.2%} {sh:>10.3f} {so:>10.3f} {dd:>9.2%} {len(yr_ret):>6}")

yearly_df = pd.DataFrame(yearly_rows)

# Full period stats
print(f"\n{'FULL':>6} | {s_best['Total Return']:>10.2%} {s_best['Sharpe']:>10.3f} "
      f"{s_best['Sortino']:>10.3f} {s_best['Max DD']:>9.2%} {len(best_ret):>6}")


_header("COMPARISON TABLE: Top 5 Strategies (Yearly Performance)")

top5 = grid_df.dropna(subset=['Sharpe']).head(5)

for idx, (_, row) in enumerate(top5.iterrows(), 1):
    ma, z, thresh = int(row['ma']), int(row['z']), row['thresh']

    # Recalculate daily returns for this combo
    log_c = np.log(qqq_prices.replace(0, np.nan).ffill())
    fair = qqq_prices.rolling(ma).mean()
    residual = log_c - np.log(fair)
    z_score = ((residual - residual.rolling(z).mean()) /
               residual.rolling(z).std())
    bubble_score = np.tanh(z_score / 2)

    trade_rets = []
    i = max(ma, z)
    while i < len(qqq_prices) - HOLD_DAYS:
        score = bubble_score.iloc[i]
        if pd.isna(score) or abs(score) < thresh:
            i += 1
            continue
        if score < -thresh:
            entry_price = qqq_prices.iloc[i]
            exit_price = qqq_prices.iloc[i + HOLD_DAYS]
            if entry_price > 0 and exit_price > 0:
                raw_ret = (exit_price / entry_price - 1)
                net_ret = raw_ret - TC
                trade_rets.append({"date": qqq_prices.index[i], "ret": net_ret})
        i += HOLD_DAYS

    if len(trade_rets) >= 3:
        tdf = pd.DataFrame(trade_rets)
        strat_ret = (tdf.assign(dt=pd.to_datetime(tdf["date"]).dt.normalize())
                        .groupby("dt")["ret"].mean()
                        .reindex(pd.date_range(tdf["date"].min(),
                                               qqq_prices.index[-1], freq="B"),
                                 fill_value=0.0))

        print(f"\n({idx}) MA={ma}d, Z={z}d, Thresh={thresh:.1f} | Sharpe={row['Sharpe']:.3f}, Return={row['Return']:+.1%}, MaxDD={row['MaxDD']:.1%}")
        print(f"     {'Year':>6} | {'Return':>10} {'Sharpe':>10} {'MaxDD':>9}")
        print(f"     {'-'*44}")

        for yr in sorted(strat_ret.index.year.unique()):
            yr_ret = strat_ret[strat_ret.index.year == yr]
            if len(yr_ret) < 5:
                continue
            w = (1+yr_ret).cumprod(); w = w/w.iloc[0]
            ret = float(w.iloc[-1]-1)
            sh = _sharpe(yr_ret)
            dd = float((w/w.cummax()-1).min())
            print(f"     {yr:>6} | {ret:>10.2%} {sh:>10.3f} {dd:>9.2%}")


_header("CHARTS")

# Chart 1: Sharpe vs Return scatter
fig, ax = plt.subplots(figsize=(14, 8))
valid_grid = grid_df[grid_df['Sharpe'].notna()]
scatter = ax.scatter(valid_grid['Return'], valid_grid['Sharpe'],
                    c=valid_grid['n_trades'], s=100, alpha=0.6, cmap='viridis')
ax.set_xlabel("Total Return", fontsize=11)
ax.set_ylabel("Sharpe Ratio", fontsize=11)
ax.set_title("QQQ Bubble Grid Search: Sharpe vs Return (colored by trade count)", fontsize=12, fontweight="bold")
cb = plt.colorbar(scatter, ax=ax)
cb.set_label("# Trades")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Chart 2: Top 5 strategies cumulative wealth
fig, ax = plt.subplots(figsize=(16, 8))
colors = ['steelblue', 'orange', 'green', 'crimson', 'purple']

for idx, (_, row) in enumerate(top5.head(5).iterrows()):
    ma, z, thresh = int(row['ma']), int(row['z']), row['thresh']

    log_c = np.log(qqq_prices.replace(0, np.nan).ffill())
    fair = qqq_prices.rolling(ma).mean()
    residual = log_c - np.log(fair)
    z_score = ((residual - residual.rolling(z).mean()) /
               residual.rolling(z).std())
    bubble_score = np.tanh(z_score / 2)

    trade_rets = []
    i = max(ma, z)
    while i < len(qqq_prices) - HOLD_DAYS:
        score = bubble_score.iloc[i]
        if pd.isna(score) or abs(score) < thresh:
            i += 1
            continue
        if score < -thresh:
            entry_price = qqq_prices.iloc[i]
            exit_price = qqq_prices.iloc[i + HOLD_DAYS]
            if entry_price > 0 and exit_price > 0:
                net_ret = (exit_price / entry_price - 1) - TC
                trade_rets.append({"date": qqq_prices.index[i], "ret": net_ret})
        i += HOLD_DAYS

    tdf = pd.DataFrame(trade_rets)
    strat_ret = (tdf.assign(dt=pd.to_datetime(tdf["date"]).dt.normalize())
                    .groupby("dt")["ret"].mean()
                    .reindex(pd.date_range(tdf["date"].min(),
                                           qqq_prices.index[-1], freq="B"),
                             fill_value=0.0))

    w = (1+strat_ret).cumprod(); w = w/w.iloc[0]
    sh = _sharpe(strat_ret)
    ax.plot(w.index, w.values, label=f"MA={ma}, Z={z}, T={thresh:.1f} (Sh={sh:.3f})",
            color=colors[idx], linewidth=2, alpha=0.8)

ax.set_title("Top 5 QQQ Bubble Strategies - Cumulative Wealth", fontsize=12, fontweight="bold")
ax.set_ylabel("Cumulative Wealth")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.1f}x"))
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Chart 3: Yearly Sharpe
if len(yearly_df) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(yearly_df['Year'], yearly_df['Sharpe'], color='steelblue', alpha=0.7, width=0.6)
    ax.axhline(y=s_best['Sharpe'], color='red', linestyle='--', label=f"Overall: {s_best['Sharpe']:.3f}", linewidth=2)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Sharpe Ratio", fontsize=11)
    ax.set_title(f"QQQ Bubble (Best: MA={best_params['ma']}d, Z={best_params['z']}d, T={best_params['thresh']}) - Yearly Sharpe", fontsize=12, fontweight="bold")
    ax.set_xticks(yearly_df['Year'])
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


_header("SAVE RESULTS")

xl = "results/qqq_bubble_extensive_grid.xlsx"
try:
    with pd.ExcelWriter(xl, engine="openpyxl") as writer:
        # Full grid
        grid_df.to_excel(writer, sheet_name="Full_Grid", index=False)

        # Top 20
        top20 = grid_df.dropna(subset=['Sharpe']).head(20)
        top20.to_excel(writer, sheet_name="Top_20", index=False)

        # Yearly breakdown
        yearly_df.to_excel(writer, sheet_name="Yearly_Breakdown", index=False)

        # Summary
        summary = pd.DataFrame({
            "Parameter": ["Best MA", "Best Z", "Best Threshold", "Best Sharpe", "Best Return", "Best MaxDD", "Best Trades"],
            "Value": [best_params['ma'], best_params['z'], best_params['thresh'],
                     f"{s_best['Sharpe']:.4f}", f"{s_best['Total Return']:+.2%}", f"{s_best['Max DD']:.2%}",
                     best_params['n_trades']]
        })
        summary.to_excel(writer, sheet_name="Summary", index=False)

        # Daily returns
        best_ret.to_excel(writer, sheet_name="Daily_Returns")

    print(f"Saved: {xl}")
except Exception as e:
    print(f"Save error: {e}")


_header("FINAL SUMMARY")

print(f"""
QQQ BUBBLE - EXTENSIVE GRID SEARCH RESULTS

Grid Size: {total_combos} combinations tested
  MA Windows:  {len(MA_WINDOWS)} values {MA_WINDOWS}
  Z Windows:   {len(Z_WINDOWS)} values {Z_WINDOWS}
  Thresholds:  {len(THRESHOLDS)} values {THRESHOLDS}

BEST STRATEGY:
  Parameters:  MA={best_params['ma']}d, Z={best_params['z']}d, Threshold={best_params['thresh']}
  Hold Period: {HOLD_DAYS} days
  Trades:      {best_params['n_trades']} total over 8 years (avg {best_params['n_trades']/8:.1f} per year)

  Overall Performance (2018-2026):
    Sharpe:       {s_best['Sharpe']:.4f}
    Sortino:      {s_best['Sortino']:.4f}
    Return:       {s_best['Total Return']:+.2%}
    Max DD:       {s_best['Max DD']:.2%}

  Yearly Breakdown:
    Best Year:    {yearly_df.loc[yearly_df['Sharpe'].idxmax(), 'Year']:.0f} (Sharpe {yearly_df['Sharpe'].max():.3f})
    Worst Year:   {yearly_df.loc[yearly_df['Sharpe'].idxmin(), 'Year']:.0f} (Sharpe {yearly_df['Sharpe'].min():.3f})

Results: results/qqq_bubble_extensive_grid.xlsx
Charts:  results/qqq_bubble_grid_chart_*.png
""")

EOF
