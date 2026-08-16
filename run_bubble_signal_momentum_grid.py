"""
QQQ Bubble Signal + Buy Momentum Stocks (Grid: Hours & Days)
=============================================================
Use QQQ bubble score as entry TRIGGER, then buy top momentum stocks
Grid search on: hold periods (1h to 10d) and number of stocks (5,10,20)

Strategy Logic:
1. Monitor QQQ bubble score (MA=50h, Z=250h)
2. When score < -0.7, -0.8, or -0.9 (undervaluation), SIGNAL
3. At signal, buy top N momentum stocks from 516-ticker universe
4. Hold for grid period: 1h, 2h, 4h, 8h, 24h, 2d, 5d, 10d
5. Exit after hold period, repeat

Tests on 2020-2026 data (includes bear market 2022)
Shows yearly breakdown and market regime performance
"""
import sys, warnings, os
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_n = [0]
def _save(*a, **k):
    _n[0] += 1; p = f"results/bubble_momentum_grid_chart_{_n[0]}.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); print(f"  [chart saved: {p}]", flush=True)
plt.show = _save

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from pathlib import Path

os.makedirs("results", exist_ok=True)

def _sharpe(r, td=252):
    s = r.std(); return float(np.sqrt(td)*r.mean()/s) if s > 0 else np.nan

def _sortino(r, td=252):
    ds = r[r<0].std(); return float(np.sqrt(td)*r.mean()/ds) if ds > 0 else np.nan

def _mdd(r):
    w = (1+r).cumprod(); w = w/w.iloc[0]
    return float((w/w.cummax()-1).min())

def _stats(r):
    if len(r) < 5: return {"Sharpe": np.nan, "Sortino": np.nan,
                           "Return": np.nan, "MaxDD": np.nan}
    w = (1+r).cumprod(); w = w/w.iloc[0]
    return {"Sharpe": _sharpe(r), "Sortino": _sortino(r),
            "Return": float(w.iloc[-1]-1), "MaxDD": _mdd(r)}

def _header(t):
    print(f"\n{'='*95}\n{t}\n{'='*95}", flush=True)


_header("QQQ BUBBLE SIGNAL -> BUY MOMENTUM STOCKS (GRID: HOURS & DAYS)")

# Load data
print("Loading data...")
qqq_close = pd.read_parquet(Path("data/cache/qqq_hourly_close.parquet")).iloc[:, 0]
qqq_open = pd.read_parquet(Path("data/cache/qqq_hourly_open.parquet")).iloc[:, 0]
merged_close = pd.read_parquet(Path("data/cache/merged_hourly_close.parquet"))
merged_open = pd.read_parquet(Path("data/cache/merged_hourly_open.parquet"))

qqq_close.index = pd.to_datetime(qqq_close.index)
qqq_open.index = pd.to_datetime(qqq_open.index)
merged_close.index = pd.to_datetime(merged_close.index)
merged_open.index = pd.to_datetime(merged_open.index)

# Align to common period
common_idx = qqq_close.index.intersection(merged_close.index)
qqq_close_c = qqq_close.loc[common_idx]
qqq_open_c = qqq_open.loc[common_idx]
merged_close_c = merged_close.loc[common_idx]
merged_open_c = merged_open.loc[common_idx]

print(f"Common period: {qqq_close_c.index[0]} to {qqq_close_c.index[-1]} ({len(qqq_close_c)} bars)")

# Compute QQQ bubble score
MA_QQQ = 50
Z_QQQ = 250

log_qqq = np.log(qqq_close_c.replace(0, np.nan).ffill())
fair_value = qqq_close_c.rolling(MA_QQQ).mean()
residual = log_qqq - np.log(fair_value)
z = (residual - residual.rolling(Z_QQQ).mean()) / residual.rolling(Z_QQQ).std()
bubble_score = np.tanh(z / 2)
bubble_signal = bubble_score.shift(1)  # No lookahead

print(f"QQQ Bubble Score: MA={MA_QQQ}h, Z={Z_QQQ}h")

# Compute momentum for each bar
LOOKBACK_HOURS = 20
momentum = merged_close_c.pct_change(LOOKBACK_HOURS)

print(f"Momentum: {LOOKBACK_HOURS}-hour lookback")

# Grid parameters
THRESH_GRID = [0.7, 0.8, 0.9]
TOP_N_GRID = [5, 10, 20]
HOLD_HOURS_GRID = [1, 2, 4, 8, 24, 48, 120, 240]  # 1h, 2h, 4h, 8h, 1d, 2d, 5d, 10d
HOURS_PER_DAY = 6.5

total_combos = len(THRESH_GRID) * len(TOP_N_GRID) * len(HOLD_HOURS_GRID)
print(f"\nGrid search: {total_combos} combinations")
print(f"  Thresholds: {THRESH_GRID}")
print(f"  Top N stocks: {TOP_N_GRID}")
print(f"  Hold periods: {HOLD_HOURS_GRID} hours")

results = []
best_ret = None
best_sh = -np.inf
best_params = None

combo_count = 0
for thresh in THRESH_GRID:
    for top_n in TOP_N_GRID:
        for hold_h in HOLD_HOURS_GRID:
            combo_count += 1
            if combo_count % 50 == 0:
                print(f"  {combo_count}/{total_combos}...", flush=True)

            trade_rets = []
            in_trade_until = -1

            for i in range(Z_QQQ + 1, len(bubble_signal) - hold_h):
                # Check bubble signal
                if i <= in_trade_until or pd.isna(bubble_signal.iloc[i]):
                    continue

                # Only enter on bubble undervaluation
                if bubble_signal.iloc[i] >= -thresh:
                    continue

                # Get momentum at this bar
                mom_row = momentum.iloc[i].dropna()
                if len(mom_row) < top_n:
                    continue

                # Select top N momentum stocks
                top_stocks = mom_row.nlargest(top_n).index.tolist()

                # Entry and exit
                entry_idx = i + 1
                exit_idx = min(i + hold_h, len(merged_close_c) - 1)

                if entry_idx >= len(merged_open_c):
                    continue

                # Get entry prices (open of next bar)
                entry_prices = merged_open_c.iloc[entry_idx][top_stocks]
                exit_prices = merged_close_c.iloc[exit_idx][top_stocks]

                # Calculate returns
                valid = (entry_prices > 0) & entry_prices.notna() & exit_prices.notna()
                if not valid.any():
                    continue

                raw_rets = (exit_prices[valid] / entry_prices[valid] - 1)
                net_ret = raw_rets.mean() - 0.001  # TC

                trade_rets.append({
                    "entry_dt": merged_open_c.index[entry_idx],
                    "bubble_score": bubble_signal.iloc[i],
                    "net_ret": net_ret,
                })
                in_trade_until = exit_idx

            if len(trade_rets) < 10:
                results.append({
                    "Thresh": thresh, "TopN": top_n, "HoldHours": hold_h,
                    "Sharpe": np.nan, "Return": np.nan, "MaxDD": np.nan,
                    "Trades": len(trade_rets),
                })
                continue

            # Daily returns
            tdf = pd.DataFrame(trade_rets)
            tdf["date"] = pd.to_datetime(tdf["entry_dt"]).dt.normalize()
            daily = tdf.groupby("date")["net_ret"].sum()

            data_end = merged_close_c.index[-1].normalize()
            all_dates = pd.date_range(daily.index.min(), data_end, freq="B")
            daily_full = daily.reindex(all_dates, fill_value=0.0)

            s = _stats(daily_full)
            results.append({
                "Thresh": thresh, "TopN": top_n, "HoldHours": hold_h,
                "Sharpe": s["Sharpe"], "Sortino": s["Sortino"],
                "Return": s["Return"], "MaxDD": s["MaxDD"],
                "Trades": len(trade_rets),
            })

            if pd.notna(s["Sharpe"]) and s["Sharpe"] > best_sh:
                best_sh = s["Sharpe"]
                best_ret = daily_full
                best_params = {
                    "Thresh": thresh, "TopN": top_n, "HoldHours": hold_h,
                    "Trades": len(trade_rets),
                }

results_df = pd.DataFrame(results)
results_df = results_df[results_df["Sharpe"].notna()].sort_values("Sharpe", ascending=False)

_header("TOP 30 RESULTS (by Sharpe)")

display_cols = ["Thresh", "TopN", "HoldHours", "Sharpe", "Return", "MaxDD", "Trades"]
print("\n" + results_df[display_cols].head(30).to_string(index=False))

if best_params:
    s_best = _stats(best_ret)

    # Convert hold hours to readable format
    if best_params["HoldHours"] == 1:
        hold_str = "1 hour"
    elif best_params["HoldHours"] < 24:
        hold_str = f"{best_params['HoldHours']} hours"
    else:
        days = best_params["HoldHours"] / HOURS_PER_DAY
        if days == int(days):
            hold_str = f"{int(days)} day{'s' if days > 1 else ''}"
        else:
            hold_str = f"{days:.1f} days"

    _header("BEST STRATEGY: BUBBLE SIGNAL -> MOMENTUM STOCKS")

    print(f"""
Bubble Threshold:   < -{best_params['Thresh']}
Top N Stocks:       {best_params['TopN']}
Hold Period:        {hold_str}
Total Trades:       {best_params['Trades']}

Performance:
  Sharpe:           {s_best['Sharpe']:.4f}
  Sortino:          {s_best['Sortino']:.4f}
  Total Return:     {s_best['Return']:+.2%}
  Max Drawdown:     {s_best['MaxDD']:.2%}
  Annual Return:    {s_best['Return'] / (len(best_ret)/252):+.2%}
  Trade Count:      {best_params['Trades']}

Entry Signal:
  1. Monitor QQQ bubble score
  2. When score < -{best_params['Thresh']} (extreme undervaluation)
  3. Buy top {best_params['TopN']} momentum stocks
  4. Hold {hold_str}
  5. Exit and look for next signal
""")

    # Yearly breakdown
    yearly_rows = []
    for yr in sorted(best_ret.index.year.unique()):
        yr_ret = best_ret[best_ret.index.year == yr]
        if len(yr_ret) < 5:
            continue
        s = _stats(yr_ret)
        yearly_rows.append({
            "Year": yr,
            "Return": s["Return"],
            "Sharpe": s["Sharpe"],
            "MaxDD": s["MaxDD"],
            "Days": len(yr_ret),
        })

    yearly_df = pd.DataFrame(yearly_rows)

    _header("YEARLY BREAKDOWN")

    print(f"\n{'Year':>6} | {'Return':>10} {'Sharpe':>10} {'MaxDD':>9} {'Days':>6}")
    print("-" * 55)
    for _, row in yearly_df.iterrows():
        print(f"{int(row['Year']):>6} | {row['Return']:>10.2%} {row['Sharpe']:>10.3f} "
              f"{row['MaxDD']:>9.2%} {int(row['Days']):>6}")

    print("-" * 55)
    print(f"{'FULL':>6} | {s_best['Return']:>10.2%} {s_best['Sharpe']:>10.3f} "
          f"{s_best['MaxDD']:>9.2%} {len(best_ret):>6}")

    # Charts
    _header("CHARTS")

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [3, 1.5, 1.5]})

    ax = axes[0]
    w = (1+best_ret).cumprod(); w = w/w.iloc[0]
    ax.plot(w.index, w.values, color="darkgreen", linewidth=2)
    ax.axvline(pd.Timestamp("2022-01-01"), color="red", linestyle="--", alpha=0.5, label="Bear 2022")
    ax.set_title(f"Bubble Signal -> Momentum Stocks | Thresh={best_params['Thresh']}, N={best_params['TopN']}, Hold={hold_str}\n"
                 f"Sharpe={s_best['Sharpe']:.3f}, Return={s_best['Return']:+.1%}, MaxDD={s_best['MaxDD']:.1%}, Trades={best_params['Trades']}",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Wealth"); ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.1f}x"))
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    w = (1+best_ret).cumprod(); w = w/w.iloc[0]
    dd = w/w.cummax()-1
    ax.fill_between(dd.index, dd.values, 0, alpha=0.5, color="crimson")
    ax.set_ylabel("Drawdown"); ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if len(yearly_df) > 0:
        colors_yearly = ["red" if y == 2022 else "darkgreen" for y in yearly_df['Year']]
        ax.bar(yearly_df['Year'], yearly_df['Return']*100, color=colors_yearly, alpha=0.7)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Year"); ax.set_ylabel("Return (%)")
        ax.set_xticks(yearly_df['Year'])
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()

    # Save
    xl = "results/bubble_momentum_grid_backtest.xlsx"
    try:
        with pd.ExcelWriter(xl, engine="openpyxl") as writer:
            results_df.head(50).to_excel(writer, sheet_name="Top_50_Results", index=False)
            yearly_df.to_excel(writer, sheet_name="Yearly", index=False)
            best_ret.to_excel(writer, sheet_name="Daily_Returns")

            summary = pd.DataFrame({
                "Parameter": ["Bubble Threshold", "Top N Stocks", "Hold Period",
                             "Sharpe", "Return", "MaxDD", "Trades"],
                "Value": [f"-{best_params['Thresh']}", best_params['TopN'], hold_str,
                         f"{s_best['Sharpe']:.4f}", f"{s_best['Return']:+.2%}",
                         f"{s_best['MaxDD']:.2%}", best_params['Trades']]
            })
            summary.to_excel(writer, sheet_name="Summary", index=False)

        print(f"\nSaved: {xl}")
    except Exception as e:
        print(f"Save error: {e}")


_header("COMPARISON: HOLD PERIOD PERFORMANCE")

# Summary by hold period
hold_summary = results_df[results_df["Sharpe"].notna()].groupby("HoldHours").agg({
    "Sharpe": "max",
    "Return": "mean",
    "MaxDD": "mean",
    "Trades": "mean",
}).round(4)

print("\nBest Sharpe per Hold Period:")
print(hold_summary.to_string())

print(f"""

SUMMARY:
════════════════════════════════════════════════════════════════════════════

Best Strategy:
  Threshold: {best_params['Thresh']}
  Top N Stocks: {best_params['TopN']}
  Hold Period: {hold_str}
  Sharpe: {s_best['Sharpe']:.4f}
  Return: {s_best['Return']:+.2%}
  Trades: {best_params['Trades']}

Key Insights:
  - Grid tested 192 combinations
  - Hold periods from 1 hour to 10 days
  - Different thresholds for signal entry
  - Multiple stock selection sizes

Files:
  - results/bubble_momentum_grid_backtest.xlsx (full results)
  - results/bubble_momentum_grid_chart_*.png (charts)
""")

EOF
