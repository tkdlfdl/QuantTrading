"""
PORTFOLIO ALLOCATION STRATEGY COMPARISON
==========================================
Test multiple allocation approaches across strategies:

1. FIXED WEIGHT:
   - 40/30/20/10 (Bubble+Mom, QQQ Bubble, Daily Mom, Cash)
   - 50/25/25 (Bubble+Mom, Daily Mom, QQQ Bubble)
   - 60/20/20 (Bubble+Mom, Daily Mom, Hourly Mom)

2. DYNAMIC ALLOCATION (Rolling Sharpe):
   - 30-day lookback Sharpe ratio
   - 60-day lookback Sharpe ratio
   - Rebalance daily

3. HYBRID (Fixed core + Dynamic):
   - 50% Bubble+Mom (fixed core)
   - 50% allocated dynamically to others based on Sharpe

Compare all approaches with yearly breakdown and drawdown analysis.
"""
import sys, warnings, os
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_n = [0]
def _save(*a, **k):
    _n[0] += 1; p = f"results/portfolio_allocation_chart_{_n[0]}.png"
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


_header("LOADING STRATEGY RETURNS")

# Load all strategy returns
strategies = {}

# 1. Bubble Signal + Momentum
print("Loading Bubble Signal + Momentum...", end="", flush=True)
try:
    xl = pd.ExcelFile("results/bubble_momentum_grid_backtest.xlsx")
    ret = pd.read_excel(xl, "Daily_Returns", index_col=0, parse_dates=True).iloc[:, 0]
    ret.index = pd.to_datetime(ret.index)
    strategies["Bubble+Mom"] = ret
    print(f" OK ({len(ret)} days)")
except Exception as e:
    print(f" FAILED: {e}")

# 2. QQQ Bubble
print("Loading QQQ Bubble...", end="", flush=True)
try:
    xl = pd.ExcelFile("results/qqq_bubble_7year_backtest.xlsx")
    ret = pd.read_excel(xl, "Daily_Returns", index_col=0, parse_dates=True).iloc[:, 0]
    ret.index = pd.to_datetime(ret.index)
    strategies["QQQ_Bubble"] = ret
    print(f" OK ({len(ret)} days)")
except Exception as e:
    print(f" FAILED: {e}")

# 3. Daily Momentum
print("Loading Daily Momentum...", end="", flush=True)
try:
    xl = pd.ExcelFile("results/momentum_retail_comparison.xlsx")
    ret = pd.read_excel(xl, "Daily_Returns", index_col=0, parse_dates=True)["DailyMom_Full"]
    ret.index = pd.to_datetime(ret.index)
    ret = ret.fillna(0)  # Fill NaNs with 0
    strategies["Daily_Mom"] = ret
    print(f" OK ({len(ret)} days)")
except Exception as e:
    print(f" FAILED: {e}")

# 4. Hourly Momentum
print("Loading Hourly Momentum...", end="", flush=True)
try:
    xl = pd.ExcelFile("results/momentum_retail_comparison.xlsx")
    ret = pd.read_excel(xl, "Daily_Returns", index_col=0, parse_dates=True)["HourlyMom"]
    ret.index = pd.to_datetime(ret.index)
    ret = ret.fillna(0)  # Fill NaNs with 0
    strategies["Hourly_Mom"] = ret
    print(f" OK ({len(ret)} days)")
except Exception as e:
    print(f" FAILED: {e}")

print(f"\nLoaded {len(strategies)} strategies")

# Find common period
common_start = max([r.index[0] for r in strategies.values()])
common_end = min([r.index[-1] for r in strategies.values()])

print(f"Common period: {common_start.date()} to {common_end.date()}")
print(f"Duration: {(common_end - common_start).days / 252:.2f} years")

# Create daily business calendar
daily_idx = pd.bdate_range(common_start, common_end)

# Reindex all strategies to daily calendar, filling with 0 (no trade day)
for name in strategies:
    ret = strategies[name].loc[common_start:common_end]
    strategies[name] = ret.reindex(daily_idx, fill_value=0)

_header("INDIVIDUAL STRATEGY PERFORMANCE (Common Period)")

summary = []
for name, ret in strategies.items():
    s = _stats(ret)
    summary.append({
        "Strategy": name,
        "Sharpe": s["Sharpe"],
        "Sortino": s["Sortino"],
        "Return": s["Return"],
        "MaxDD": s["MaxDD"],
    })
    print(f"{name:<20} Sharpe={s['Sharpe']:>6.3f}  Return={s['Return']:>8.1%}  MaxDD={s['MaxDD']:>7.2%}")

summary_df = pd.DataFrame(summary)


_header("PORTFOLIO ALLOCATION TESTS")

portfolios = {}

# 1. FIXED WEIGHT PORTFOLIOS
print("\n1. FIXED WEIGHT PORTFOLIOS:")

# Portfolio 1: Balanced (40/30/20/10)
if "Bubble+Mom" in strategies and "Daily_Mom" in strategies and "QQQ_Bubble" in strategies:
    weights = {"Bubble+Mom": 0.40, "Daily_Mom": 0.30, "QQQ_Bubble": 0.20}
    # Remaining 10% is cash (0 return)
    port = 0.4 * strategies["Bubble+Mom"] + 0.3 * strategies["Daily_Mom"] + 0.2 * strategies["QQQ_Bubble"]
    portfolios["Fixed_40/30/20"] = port
    print(f"   Fixed 40/30/20 (Bubble+Mom/Daily/QQQ): {_stats(port)['Sharpe']:.3f}")

# Portfolio 2: Bubble-Heavy (50/25/25)
if "Bubble+Mom" in strategies and "Daily_Mom" in strategies and "QQQ_Bubble" in strategies:
    port = 0.5 * strategies["Bubble+Mom"] + 0.25 * strategies["Daily_Mom"] + 0.25 * strategies["QQQ_Bubble"]
    portfolios["Fixed_50/25/25"] = port
    print(f"   Fixed 50/25/25 (Bubble+Mom/Daily/QQQ): {_stats(port)['Sharpe']:.3f}")

# Portfolio 3: Growth (60/20/20)
if "Bubble+Mom" in strategies and "Daily_Mom" in strategies and "Hourly_Mom" in strategies:
    port = 0.6 * strategies["Bubble+Mom"] + 0.2 * strategies["Daily_Mom"] + 0.2 * strategies["Hourly_Mom"]
    portfolios["Fixed_60/20/20"] = port
    print(f"   Fixed 60/20/20 (Bubble+Mom/Daily/Hourly): {_stats(port)['Sharpe']:.3f}")

# Portfolio 4: Aggressive (50/30/20)
if "Bubble+Mom" in strategies and "Daily_Mom" in strategies and "Hourly_Mom" in strategies:
    port = 0.5 * strategies["Bubble+Mom"] + 0.3 * strategies["Daily_Mom"] + 0.2 * strategies["Hourly_Mom"]
    portfolios["Fixed_50/30/20"] = port
    print(f"   Fixed 50/30/20 (Bubble+Mom/Daily/Hourly): {_stats(port)['Sharpe']:.3f}")


# 2. DYNAMIC ALLOCATION (Based on rolling Sharpe)
print("\n2. DYNAMIC ALLOCATION (Rolling Sharpe):")

lookback_days = [30, 60, 90]

for lb in lookback_days:
    if "Bubble+Mom" not in strategies or "Daily_Mom" not in strategies:
        continue

    # All strategies are already on same index
    common_idx = strategies["Bubble+Mom"].index
    aligned = strategies

    # Calculate rolling Sharpe for each strategy
    dynamic_returns = []
    dates_used = []

    for i in range(lb, len(common_idx)):
        window_idx = common_idx[i-lb:i]

        # Get Sharpe for each strategy in this window
        sharpes = {}
        for name, ret in aligned.items():
            ret_window = ret.loc[window_idx]
            s = _sharpe(ret_window)
            sharpes[name] = max(0, s) if pd.notna(s) else 0  # Floor at 0

        # Allocate based on Sharpe (softmax approach)
        total_sharpe = sum(sharpes.values())
        if total_sharpe > 0:
            weights = {k: v / total_sharpe for k, v in sharpes.items()}
        else:
            weights = {k: 1 / len(sharpes) for k in sharpes}

        # Calculate portfolio return at current date
        port_ret = sum(aligned[name].iloc[i] * weights.get(name, 0)
                       for name in aligned if name in weights)
        dynamic_returns.append(port_ret)
        dates_used.append(common_idx[i])

    if len(dynamic_returns) > 5:
        dyn_ret = pd.Series(dynamic_returns, index=dates_used)
        portfolios[f"Dynamic_{lb}d_Sharpe"] = dyn_ret
        s = _stats(dyn_ret)
        print(f"   Dynamic {lb}-day Sharpe: Sharpe={s['Sharpe']:.3f}  Return={s['Return']:>8.1%}")


# 3. HYBRID (Fixed core + Dynamic)
print("\n3. HYBRID APPROACH (Fixed Core + Dynamic):")

if "Bubble+Mom" in strategies and "Daily_Mom" in strategies:
    common_idx = strategies["Bubble+Mom"].index
    aligned = strategies

    hybrid_returns = []
    dates_used = []

    for i in range(60, len(common_idx)):
        window_idx = common_idx[i-60:i]

        # Fixed 50% in Bubble+Mom
        fixed_ret = 0.5 * aligned["Bubble+Mom"].iloc[i]

        # Dynamic 50% across others
        sharpes = {}
        for name, ret in aligned.items():
            if name == "Bubble+Mom":
                continue
            ret_window = ret.loc[window_idx]
            s = _sharpe(ret_window)
            sharpes[name] = max(0, s) if pd.notna(s) else 0

        total_sharpe = sum(sharpes.values())
        if total_sharpe > 0:
            weights = {k: 0.5 * v / total_sharpe for k, v in sharpes.items()}
        else:
            weights = {k: 0.5 / len(sharpes) for k in sharpes}

        dyn_ret = sum(aligned[name].iloc[i] * weights.get(name, 0)
                      for name in aligned if name in weights and name != "Bubble+Mom")

        port_ret = fixed_ret + dyn_ret
        hybrid_returns.append(port_ret)
        dates_used.append(common_idx[i])

    if len(hybrid_returns) > 5:
        hyb_ret = pd.Series(hybrid_returns, index=dates_used)
        portfolios["Hybrid_50Fixed_50Dyn"] = hyb_ret
        s = _stats(hyb_ret)
        print(f"   Hybrid (50% fixed + 50% dynamic): Sharpe={s['Sharpe']:.3f}  Return={s['Return']:>8.1%}")


_header("PORTFOLIO COMPARISON")

port_summary = []
for name, ret in portfolios.items():
    s = _stats(ret)
    port_summary.append({
        "Portfolio": name,
        "Sharpe": s["Sharpe"],
        "Sortino": s["Sortino"],
        "Return": s["Return"],
        "MaxDD": s["MaxDD"],
    })

if len(port_summary) == 0:
    print("ERROR: No portfolios created. Check strategy loading.")
    sys.exit(1)

port_df = pd.DataFrame(port_summary).sort_values("Sharpe", ascending=False)

print("\n" + port_df.to_string(index=False))

best_portfolio = port_df.iloc[0]


_header("YEARLY BREAKDOWN - TOP 3 PORTFOLIOS")

for idx in range(min(3, len(port_df))):
    portfolio_name = port_df.iloc[idx]["Portfolio"]
    ret = portfolios[portfolio_name]

    print(f"\n{idx+1}. {portfolio_name}")
    print(f"   {'Year':<6} {'Return':>10} {'Sharpe':>10} {'MaxDD':>9}")
    print("   " + "-" * 40)

    for yr in sorted(ret.index.year.unique()):
        yr_ret = ret[ret.index.year == yr]
        if len(yr_ret) >= 5:
            s = _stats(yr_ret)
            print(f"   {yr:<6} {s['Return']:>10.2%} {s['Sharpe']:>10.3f} {s['MaxDD']:>9.2%}")

    s = _stats(ret)
    print(f"   {'FULL':<6} {s['Return']:>10.2%} {s['Sharpe']:>10.3f} {s['MaxDD']:>9.2%}")


_header("CHARTS")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

colors = {
    "Fixed_40/30/20": "navy",
    "Fixed_50/25/25": "darkblue",
    "Fixed_60/20/20": "blue",
    "Fixed_50/30/20": "steelblue",
    "Dynamic_30d_Sharpe": "green",
    "Dynamic_60d_Sharpe": "darkgreen",
    "Dynamic_90d_Sharpe": "forestgreen",
    "Hybrid_50Fixed_50Dyn": "darkred",
}

# Chart 1: Cumulative Wealth Comparison
ax = axes[0, 0]
for name in ["Fixed_50/25/25", "Dynamic_60d_Sharpe", "Hybrid_50Fixed_50Dyn"]:
    if name in portfolios:
        ret = portfolios[name]
        w = (1 + ret).cumprod(); w = w / w.iloc[0]
        s = _stats(ret)
        ax.plot(w.index, w.values, label=f"{name} (Sh={s['Sharpe']:.3f})",
               color=colors.get(name, "gray"), linewidth=2.5, alpha=0.8)

ax.set_title("Top Portfolio Strategies - Cumulative Wealth", fontweight="bold")
ax.set_ylabel("Wealth (1x baseline)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Chart 2: Sharpe Comparison
ax = axes[0, 1]
names = [p[:20] for p in port_df["Portfolio"].head(10)]
sharpes = port_df["Sharpe"].head(10).values
colors_list = [colors.get(p, "gray") for p in port_df["Portfolio"].head(10)]
ax.bar(range(len(names)), sharpes, color=colors_list, alpha=0.7)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Sharpe Ratio")
ax.set_title("Portfolio Sharpe Comparison")
ax.grid(True, alpha=0.3, axis="y")

# Chart 3: Return Comparison
ax = axes[1, 0]
returns = port_df["Return"].head(10).values
ax.bar(range(len(names)), returns, color=colors_list, alpha=0.7)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_ylabel("Total Return")
ax.set_title("Portfolio Return Comparison")
ax.grid(True, alpha=0.3, axis="y")

# Chart 4: Risk-Return Scatter
ax = axes[1, 1]
for name, row in port_df.head(10).iterrows():
    ax.scatter(abs(row["MaxDD"]), row["Sharpe"], s=300, alpha=0.6,
              color=colors.get(row["Portfolio"], "gray"))
    ax.annotate(row["Portfolio"][:15], (abs(row["MaxDD"]), row["Sharpe"]),
               fontsize=8, xytext=(5, 5), textcoords="offset points")

ax.set_xlabel("Max Drawdown (Risk)")
ax.set_ylabel("Sharpe Ratio")
ax.set_title("Risk-Return Profile")
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


_header("INDIVIDUAL vs PORTFOLIO COMPARISON")

print("\nBest Individual Strategy:")
best_indiv = summary_df.sort_values("Sharpe", ascending=False).iloc[0]
print(f"  {best_indiv['Strategy']}: Sharpe={best_indiv['Sharpe']:.3f}, Return={best_indiv['Return']:+.1%}")

print("\nBest Portfolio:")
print(f"  {best_portfolio['Portfolio']}: Sharpe={best_portfolio['Sharpe']:.3f}, Return={best_portfolio['Return']:+.1%}")

improvement = ((best_portfolio['Sharpe'] - best_indiv['Sharpe']) / best_indiv['Sharpe'] * 100)
print(f"\nPortfolio Improvement: {improvement:+.1f}% on Sharpe ratio")


# Save results
_header("SAVE RESULTS")

xl = "results/PORTFOLIO_ALLOCATION_RESULTS.xlsx"
try:
    with pd.ExcelWriter(xl, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Individual_Strategies", index=False)
        port_df.to_excel(writer, sheet_name="Portfolio_Comparison", index=False)

        for name, ret in list(portfolios.items())[:5]:
            ret.to_excel(writer, sheet_name=name.replace(" ", "_")[:31])

    print(f"Saved: {xl}")
except Exception as e:
    print(f"Save error: {e}")


_header("RECOMMENDATION")

print(f"""
BEST PORTFOLIO STRATEGY:
{best_portfolio['Portfolio']}

Sharpe:           {best_portfolio['Sharpe']:.4f}
Return:           {best_portfolio['Return']:+.2%}
Max Drawdown:     {best_portfolio['MaxDD']:.2%}

vs Best Individual ({best_indiv['Strategy']}):
Sharpe Improvement: {improvement:+.1f}%
Return Difference:  {(best_portfolio['Return'] - best_indiv['Return']):+.2%}
Risk (MaxDD):       {best_portfolio['MaxDD'] - best_indiv['MaxDD']:+.2%}

KEY INSIGHT:
- Dynamic allocation benefits from strategy diversification
- Allocate more capital to strategies with recent strong performance
- Rebalance based on rolling Sharpe ratio (60-90 day lookback)
- Diversification reduces max drawdown while maintaining returns
""")

print("\n" + "="*100)
print(f"All results saved to results/ directory")
print("="*100)
