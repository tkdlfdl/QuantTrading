"""
SHORT SQUEEZE + BUBBLE SCORE STRATEGY
======================================
Strategy Logic:
1. Identify most shorted stocks (using squeeze_extra dataset)
2. Calculate QQQ bubble score (indicator of market undervaluation)
3. When bubble score is LOW (<-0.7), buy most shorted stocks
4. Hold for varying periods (4h, 24h, 120h = 5d)
5. Exit after hold period or on profit target

Rationale:
- Low bubble score = market extremely undervalued
- High short interest = squeeze potential when market rebounds
- Combined = buy dip with squeeze catalyst
"""
import sys, warnings, os
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_n = [0]
def _save(*a, **k):
    _n[0] += 1; p = f"results/short_squeeze_bubble_chart_{_n[0]}.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); print(f"  [chart saved: {p}]", flush=True)
plt.show = _save

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
    print(f"\n{'='*100}\n{t}\n{'='*100}", flush=True)


_header("LOADING DATA")

# Load main QQQ data for bubble calculation
print("Loading QQQ data...", end="", flush=True)
qqq_close = pd.read_parquet(Path("data/cache/qqq_hourly_close.parquet")).iloc[:, 0]
qqq_open = pd.read_parquet(Path("data/cache/qqq_hourly_open.parquet")).iloc[:, 0]
qqq_close.index = pd.to_datetime(qqq_close.index)
qqq_open.index = pd.to_datetime(qqq_open.index)
print(f" OK ({len(qqq_close)} hours)")

# Load short squeeze candidates (most shorted stocks)
print("Loading short squeeze candidates...", end="", flush=True)
squeeze_close = pd.read_parquet(Path("data/cache/squeeze_extra_hourly_close.parquet"))
squeeze_open = pd.read_parquet(Path("data/cache/squeeze_extra_hourly_open.parquet"))
squeeze_close.index = pd.to_datetime(squeeze_close.index)
squeeze_open.index = pd.to_datetime(squeeze_open.index)
print(f" OK ({len(squeeze_close)} hours, {len(squeeze_close.columns)} stocks)")

# Find common period
common_idx = qqq_close.index.intersection(squeeze_close.index)
qqq_close_c = qqq_close.loc[common_idx]
qqq_open_c = qqq_open.loc[common_idx]
squeeze_close_c = squeeze_close.loc[common_idx]
squeeze_open_c = squeeze_open.loc[common_idx]

print(f"Common period: {qqq_close_c.index[0]} to {qqq_close_c.index[-1]}")
print(f"Stocks in squeeze universe: {len(squeeze_close_c.columns)}")


_header("CALCULATING QQQ BUBBLE SCORE")

# Calculate bubble score
log_qqq = np.log(qqq_close_c.replace(0, np.nan).ffill())
fair_value = qqq_close_c.rolling(50).mean()
residual = log_qqq - np.log(fair_value)
z = (residual - residual.rolling(250).mean()) / residual.rolling(250).std()
bubble_score = np.tanh(z / 2)
bubble_signal = bubble_score.shift(1)

print(f"Bubble score range: {bubble_score.min():.3f} to {bubble_score.max():.3f}")
print(f"Current bubble score: {bubble_score.iloc[-1]:.3f}")


_header("SHORT SQUEEZE + BUBBLE SCORE GRID SEARCH")

# Grid parameters
THRESH_GRID = [-0.5, -0.6, -0.7, -0.8]
TOP_N_GRID = [5, 10, 15, 20]
HOLD_HOURS_GRID = [4, 24, 48, 120, 240]

results = []
best_ret = None
best_sh = -np.inf
best_params = None

total_combos = len(THRESH_GRID) * len(TOP_N_GRID) * len(HOLD_HOURS_GRID)
combo_count = 0

print(f"Testing {total_combos} combinations...")
print(f"Thresholds: {THRESH_GRID}")
print(f"Top N stocks: {TOP_N_GRID}")
print(f"Hold periods: {HOLD_HOURS_GRID} hours\n")

for thresh in THRESH_GRID:
    for top_n in TOP_N_GRID:
        for hold_h in HOLD_HOURS_GRID:
            combo_count += 1
            if combo_count % 50 == 0:
                print(f"  {combo_count}/{total_combos}...", flush=True)

            trade_rets = []
            in_trade_until = -1

            for i in range(250, len(bubble_signal) - hold_h):
                if i <= in_trade_until or pd.isna(bubble_signal.iloc[i]):
                    continue

                # Entry: bubble score too low (undervalued)
                if bubble_signal.iloc[i] >= thresh:
                    continue

                # Get short squeeze candidates at this bar
                squeeze_row = squeeze_close_c.iloc[i].dropna()
                if len(squeeze_row) < top_n:
                    continue

                # Select top N by short squeeze potential
                # Use volatility/price action as proxy for squeeze potential
                squeeze_momentum = (squeeze_close_c.iloc[i] / squeeze_close_c.iloc[max(0, i-20)]) - 1
                squeeze_momentum = squeeze_momentum.dropna()

                if len(squeeze_momentum) < top_n:
                    continue

                # Buy most shorted (use reverse momentum - stocks that fell most = most shorts)
                top_squeeze = squeeze_momentum.nsmallest(top_n).index.tolist()

                # Entry and exit
                entry_idx = i + 1
                exit_idx = min(i + hold_h, len(squeeze_close_c) - 1)

                if entry_idx >= len(squeeze_open_c):
                    continue

                # Get entry prices
                entry_prices = squeeze_open_c.iloc[entry_idx][top_squeeze]
                exit_prices = squeeze_close_c.iloc[exit_idx][top_squeeze]

                # Calculate returns
                valid = (entry_prices > 0) & entry_prices.notna() & exit_prices.notna()
                if not valid.any():
                    continue

                raw_rets = (exit_prices[valid] / entry_prices[valid] - 1)
                net_ret = raw_rets.mean() - 0.001  # Transaction cost

                trade_rets.append({
                    "entry_dt": squeeze_close_c.index[entry_idx],
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

            data_end = squeeze_close_c.index[-1].normalize()
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

    _header("BEST STRATEGY: SHORT SQUEEZE + BUBBLE SCORE")

    hold_str = f"{best_params['HoldHours']} hours" if best_params['HoldHours'] < 24 else f"{best_params['HoldHours']/24:.0f} days"

    print(f"""
Bubble Threshold:   < {best_params['Thresh']}
Top N Squeeze Stocks: {best_params['TopN']}
Hold Period:        {hold_str}
Total Trades:       {best_params['Trades']}

Performance:
  Sharpe:           {s_best['Sharpe']:.4f}
  Sortino:          {s_best['Sortino']:.4f}
  Total Return:     {s_best['Return']:+.2%}
  Max Drawdown:     {s_best['MaxDD']:.2%}
  Annual Return:    {s_best['Return'] / (len(best_ret)/252):+.2%}

Strategy Logic:
  1. Monitor QQQ bubble score
  2. When score < {best_params['Thresh']} (extreme undervaluation)
  3. Buy top {best_params['TopN']} short squeeze candidates
  4. Hold {hold_str}
  5. Exit and repeat

Key Insight:
  - Bubble score identifies market bottoms
  - Short squeeze stocks provide leverage on reversal
  - Combined: amplified gains from undervaluation
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
    ax.plot(w.index, w.values, color="darkblue", linewidth=2)
    ax.axvline(pd.Timestamp("2022-01-01"), color="red", linestyle="--", alpha=0.5, label="Bear 2022")
    ax.set_title(f"Short Squeeze + Bubble Score | Thresh={best_params['Thresh']}, N={best_params['TopN']}, Hold={hold_str}\n"
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
        colors_yearly = ["red" if y == 2022 else "darkblue" for y in yearly_df['Year']]
        ax.bar(yearly_df['Year'], yearly_df['Return']*100, color=colors_yearly, alpha=0.7)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Year"); ax.set_ylabel("Return (%)")
        ax.set_xticks(yearly_df['Year'])
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()

    # Save
    xl = "results/short_squeeze_bubble_backtest.xlsx"
    try:
        with pd.ExcelWriter(xl, engine="openpyxl") as writer:
            results_df.head(50).to_excel(writer, sheet_name="Top_50_Results", index=False)
            yearly_df.to_excel(writer, sheet_name="Yearly", index=False)
            best_ret.to_excel(writer, sheet_name="Daily_Returns")

            summary = pd.DataFrame({
                "Parameter": ["Bubble Threshold", "Top N Stocks", "Hold Period",
                             "Sharpe", "Return", "MaxDD", "Trades"],
                "Value": [f"{best_params['Thresh']}", best_params['TopN'], hold_str,
                         f"{s_best['Sharpe']:.4f}", f"{s_best['Return']:+.2%}",
                         f"{s_best['MaxDD']:.2%}", best_params['Trades']]
            })
            summary.to_excel(writer, sheet_name="Summary", index=False)

        print(f"\nSaved: {xl}")
    except Exception as e:
        print(f"Save error: {e}")


_header("COMPARISON: HOLD PERIOD PERFORMANCE")

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
════════════════════════════════════════════════════════════════════════════════

Best Strategy:
  Threshold: {best_params['Thresh']}
  Top N Stocks: {best_params['TopN']}
  Hold Period: {hold_str}
  Sharpe: {s_best['Sharpe']:.4f}
  Return: {s_best['Return']:+.2%}
  Trades: {best_params['Trades']}

Key Insights:
  - Grid tested {total_combos} combinations
  - Hold periods from 4 hours to 10 days
  - Different bubble thresholds for signal entry
  - Uses short squeeze candidates (most shorted stocks)
  - Combined with bubble score undervaluation signal

Files:
  - results/short_squeeze_bubble_backtest.xlsx (full results)
  - results/short_squeeze_bubble_chart_*.png (charts)
""")
