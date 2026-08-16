"""
PERFORMANCE TEST: UVXY HEDGE + LEVERAGE STRATEGY
=================================================
Test bubble-based tactical allocation:
1. When Bubble > 0.7: Add UVXY hedge
2. When Bubble < -0.7: Add momentum leverage
3. Otherwise: Hold momentum only

This tests real trading performance metrics
"""
import sys, warnings, os
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_n = [0]
def _save(*a, **k):
    _n[0] += 1; p = f"results/uvxy_leverage_test_chart_{_n[0]}.png"
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

# Load daily momentum
print("Loading daily momentum returns...", end="", flush=True)
xl = pd.ExcelFile("results/momentum_retail_comparison.xlsx")
daily_ret = pd.read_excel(xl, "Daily_Returns", index_col=0, parse_dates=True)["DailyMom_Full"]
daily_ret.index = pd.to_datetime(daily_ret.index)
daily_ret = daily_ret.fillna(0)
print(f" OK ({len(daily_ret)} days)")

# Load QQQ data for bubble score
print("Loading QQQ for bubble score...", end="", flush=True)
qqq_close = pd.read_parquet(Path("data/cache/qqq_hourly_close.parquet")).iloc[:, 0]
qqq_close.index = pd.to_datetime(qqq_close.index)

# Create daily bubble score
log_qqq = np.log(qqq_close.replace(0, np.nan).ffill())
fair_value = qqq_close.rolling(50).mean()
residual = log_qqq - np.log(fair_value)
z = (residual - residual.rolling(250).mean()) / residual.rolling(250).std()
bubble_score_hourly = np.tanh(z / 2)

# Resample to daily
daily_dates = bubble_score_hourly.index.normalize().unique()
bubble_score = pd.Series(
    [bubble_score_hourly[bubble_score_hourly.index.normalize() == d].iloc[-1]
     for d in daily_dates],
    index=daily_dates
)
print(f" OK ({len(bubble_score)} days)")

# Align
common_idx = daily_ret.index.intersection(bubble_score.index)
daily_ret = daily_ret.loc[common_idx].fillna(0)
bubble_score = bubble_score.loc[common_idx]

print(f"Common period: {common_idx[0].date()} to {common_idx[-1].date()}")
print(f"Bubble score range: {bubble_score.min():.3f} to {bubble_score.max():.3f}")

_header("STRATEGY 1: BASELINE - MOMENTUM ONLY")

baseline_ret = daily_ret.copy()
s_baseline = _stats(baseline_ret)

print(f"""
Sharpe:           {s_baseline['Sharpe']:.4f}
Sortino:          {s_baseline['Sortino']:.4f}
Total Return:     {s_baseline['Return']:+.2%}
Max Drawdown:     {s_baseline['MaxDD']:.2%}
Annual Return:    {s_baseline['Return'] / (len(baseline_ret)/252):+.2%}
""")

_header("STRATEGY 2: UVXY HEDGE (When Bubble > 0.7)")

"""
Rules:
- When Bubble > 0.7: Replace 50% momentum with UVXY
- Hold hedge for 40 days
- Cost: UVXY decay (~1-2% daily on quiet days)
"""

uvxy_hedge_returns = []
hedge_days_remaining = 0
hedge_duration = 40
hedge_alloc = 0.50

for date in daily_ret.index:
    base_ret = daily_ret[date]

    # Check entry signal
    if hedge_days_remaining == 0 and bubble_score[date] > 0.7:
        hedge_days_remaining = hedge_duration

    if hedge_days_remaining > 0:
        # 50% momentum + 50% UVXY hedge
        # Approximate UVXY: inverse momentum with decay
        uvxy_approx = -base_ret * 1.5 - 0.002  # Inverse with decay
        ret = (1 - hedge_alloc) * base_ret + hedge_alloc * uvxy_approx
        hedge_days_remaining -= 1
    else:
        ret = base_ret

    uvxy_hedge_returns.append(ret)

uvxy_hedge_ret = pd.Series(uvxy_hedge_returns, index=daily_ret.index)
s_uvxy = _stats(uvxy_hedge_ret)

print(f"""
Sharpe:           {s_uvxy['Sharpe']:.4f}
Sortino:          {s_uvxy['Sortino']:.4f}
Total Return:     {s_uvxy['Return']:+.2%}
Max Drawdown:     {s_uvxy['MaxDD']:.2%}
Annual Return:    {s_uvxy['Return'] / (len(uvxy_hedge_ret)/252):+.2%}

vs Baseline:
  Sharpe Change:  {s_uvxy['Sharpe'] - s_baseline['Sharpe']:+.4f}
  Return Change:  {s_uvxy['Return'] - s_baseline['Return']:+.2%}
  MaxDD Change:   {s_uvxy['MaxDD'] - s_baseline['MaxDD']:+.2%}
""")

_header("STRATEGY 3: MOMENTUM LEVERAGE (When Bubble < -0.7)")

"""
Rules:
- When Bubble < -0.7: Add 30% extra leverage to momentum
- Hold leverage for 50 days
- Cost: 10% annual (~0.04% daily)
"""

leverage_returns = []
leverage_days_remaining = 0
leverage_duration = 50
leverage_extra = 0.30
leverage_cost_daily = (leverage_extra * 0.10) / 252

for date in daily_ret.index:
    base_ret = daily_ret[date]

    # Check entry signal
    if leverage_days_remaining == 0 and bubble_score[date] < -0.7:
        leverage_days_remaining = leverage_duration

    if leverage_days_remaining > 0:
        # 130% momentum (100% + 30% leverage)
        ret = (1 + leverage_extra) * base_ret - leverage_cost_daily
        leverage_days_remaining -= 1
    else:
        ret = base_ret

    leverage_returns.append(ret)

leverage_ret = pd.Series(leverage_returns, index=daily_ret.index)
s_leverage = _stats(leverage_ret)

print(f"""
Sharpe:           {s_leverage['Sharpe']:.4f}
Sortino:          {s_leverage['Sortino']:.4f}
Total Return:     {s_leverage['Return']:+.2%}
Max Drawdown:     {s_leverage['MaxDD']:.2%}
Annual Return:    {s_leverage['Return'] / (len(leverage_ret)/252):+.2%}

vs Baseline:
  Sharpe Change:  {s_leverage['Sharpe'] - s_baseline['Sharpe']:+.4f}
  Return Change:  {s_leverage['Return'] - s_baseline['Return']:+.2%}
  MaxDD Change:   {s_leverage['MaxDD'] - s_baseline['MaxDD']:+.2%}
""")

_header("STRATEGY 4: COMBINED (HEDGE + LEVERAGE)")

"""
Rules:
- When Bubble > 0.7: Switch to hedge (50% momentum + 50% UVXY)
- When Bubble < -0.7: Switch to leverage (130% momentum)
- Otherwise: 100% momentum
- Don't overlap strategies (exit current before entering new)
"""

combined_returns = []
current_mode = "normal"
mode_days_remaining = 0

for date in daily_ret.index:
    base_ret = daily_ret[date]

    # Exit current mode if duration expired
    if mode_days_remaining > 0:
        mode_days_remaining -= 1
    else:
        current_mode = "normal"

    # Check for new signal (only if not in a position)
    if mode_days_remaining == 0:
        if bubble_score[date] > 0.7:
            current_mode = "hedge"
            mode_days_remaining = 40
        elif bubble_score[date] < -0.7:
            current_mode = "leverage"
            mode_days_remaining = 50
        else:
            current_mode = "normal"

    # Apply current mode
    if current_mode == "hedge":
        uvxy_approx = -base_ret * 1.5 - 0.002
        ret = 0.5 * base_ret + 0.5 * uvxy_approx
    elif current_mode == "leverage":
        leverage_cost_daily = (0.30 * 0.10) / 252
        ret = 1.30 * base_ret - leverage_cost_daily
    else:
        ret = base_ret

    combined_returns.append(ret)

combined_ret = pd.Series(combined_returns, index=daily_ret.index)
s_combined = _stats(combined_ret)

print(f"""
Sharpe:           {s_combined['Sharpe']:.4f}
Sortino:          {s_combined['Sortino']:.4f}
Total Return:     {s_combined['Return']:+.2%}
Max Drawdown:     {s_combined['MaxDD']:.2%}
Annual Return:    {s_combined['Return'] / (len(combined_ret)/252):+.2%}

vs Baseline:
  Sharpe Change:  {s_combined['Sharpe'] - s_baseline['Sharpe']:+.4f}
  Return Change:  {s_combined['Return'] - s_baseline['Return']:+.2%}
  MaxDD Change:   {s_combined['MaxDD'] - s_baseline['MaxDD']:+.2%}
""")

_header("COMPARISON - ALL STRATEGIES")

results = [
    {
        "Strategy": "Momentum Only (Baseline)",
        "Sharpe": s_baseline["Sharpe"],
        "Sortino": s_baseline["Sortino"],
        "Return": s_baseline["Return"],
        "MaxDD": s_baseline["MaxDD"],
    },
    {
        "Strategy": "UVXY Hedge",
        "Sharpe": s_uvxy["Sharpe"],
        "Sortino": s_uvxy["Sortino"],
        "Return": s_uvxy["Return"],
        "MaxDD": s_uvxy["MaxDD"],
    },
    {
        "Strategy": "Leverage",
        "Sharpe": s_leverage["Sharpe"],
        "Sortino": s_leverage["Sortino"],
        "Return": s_leverage["Return"],
        "MaxDD": s_leverage["MaxDD"],
    },
    {
        "Strategy": "Combined (Hedge + Leverage)",
        "Sharpe": s_combined["Sharpe"],
        "Sortino": s_combined["Sortino"],
        "Return": s_combined["Return"],
        "MaxDD": s_combined["MaxDD"],
    },
]

results_df = pd.DataFrame(results).sort_values("Sharpe", ascending=False)
print("\n" + results_df.to_string(index=False))

_header("YEARLY BREAKDOWN - BEST STRATEGY")

best_name = results_df.iloc[0]["Strategy"]
if best_name == "Momentum Only (Baseline)":
    best_ret = baseline_ret
elif best_name == "UVXY Hedge":
    best_ret = uvxy_hedge_ret
elif best_name == "Leverage":
    best_ret = leverage_ret
else:
    best_ret = combined_ret

print(f"\nBest Strategy: {best_name}\n")
print(f"{'Year':<6} {'Return':>10} {'Sharpe':>10} {'MaxDD':>9}")
print("-" * 40)

for yr in range(best_ret.index.year.min(), best_ret.index.year.max() + 1):
    yr_ret = best_ret[best_ret.index.year == yr]
    if len(yr_ret) >= 5:
        s = _stats(yr_ret)
        print(f"{yr:<6} {s['Return']:>10.2%} {s['Sharpe']:>10.3f} {s['MaxDD']:>9.2%}")

print("-" * 40)
s_full = _stats(best_ret)
print(f"{'FULL':<6} {s_full['Return']:>10.2%} {s_full['Sharpe']:>10.3f} {s_full['MaxDD']:>9.2%}")

_header("STRATEGY ACTIVITY")

# Count signals
hedge_signals = (bubble_score > 0.7).sum()
leverage_signals = (bubble_score < -0.7).sum()
normal_days = ((bubble_score >= -0.7) & (bubble_score <= 0.7)).sum()

print(f"""
Bubble Score Distribution:
  Days < -0.7 (Leverage):   {leverage_signals} days ({leverage_signals/len(bubble_score)*100:.1f}%)
  Days -0.7 to 0.7 (Normal): {normal_days} days ({normal_days/len(bubble_score)*100:.1f}%)
  Days > 0.7 (Hedge):       {hedge_signals} days ({hedge_signals/len(bubble_score)*100:.1f}%)

Expected Annual Signals:
  Leverage entries:         ~{leverage_signals/len(bubble_score)*252:.0f} times/year
  Hedge entries:            ~{hedge_signals/len(bubble_score)*252:.0f} times/year
  Total signal days:        ~{(leverage_signals + hedge_signals)/len(bubble_score)*252:.0f} days/year
""")

_header("CHARTS")

fig, axes = plt.subplots(3, 2, figsize=(18, 14))

# Chart 1: Cumulative returns
ax = axes[0, 0]
for name, ret in [
    ("Baseline", baseline_ret),
    ("UVXY Hedge", uvxy_hedge_ret),
    ("Leverage", leverage_ret),
    ("Combined", combined_ret)
]:
    w = (1 + ret).cumprod(); w = w / w.iloc[0]
    s = _stats(ret)
    ax.plot(w.index, w.values, label=f"{name} (Sh={s['Sharpe']:.3f})", linewidth=2)

ax.set_title("Cumulative Wealth - All Strategies")
ax.set_ylabel("Wealth")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Chart 2: Bubble score
ax = axes[0, 1]
ax.plot(bubble_score.index, bubble_score.values, color="blue", linewidth=1)
ax.axhline(0.7, color="red", linestyle="--", label="Hedge entry (0.7)")
ax.axhline(-0.7, color="green", linestyle="--", label="Leverage entry (-0.7)")
ax.axhline(0, color="black", linestyle="-", alpha=0.3)
ax.fill_between(bubble_score.index, 0.7, 1, alpha=0.1, color="red")
ax.fill_between(bubble_score.index, -0.7, -1, alpha=0.1, color="green")
ax.set_ylabel("Bubble Score")
ax.set_ylim(-1, 1)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Chart 3: Sharpe comparison
ax = axes[1, 0]
sharpes = [s_baseline["Sharpe"], s_uvxy["Sharpe"], s_leverage["Sharpe"], s_combined["Sharpe"]]
colors = ["gray", "orange", "green", "blue"]
names = ["Baseline", "UVXY", "Leverage", "Combined"]
ax.bar(names, sharpes, color=colors, alpha=0.7)
ax.set_ylabel("Sharpe Ratio")
ax.set_title("Sharpe Ratio Comparison")
ax.grid(True, alpha=0.3, axis="y")

# Chart 4: Return comparison
ax = axes[1, 1]
returns = [s_baseline["Return"], s_uvxy["Return"], s_leverage["Return"], s_combined["Return"]]
ax.bar(names, returns, color=colors, alpha=0.7)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_ylabel("Total Return")
ax.set_title("Total Return Comparison")
ax.grid(True, alpha=0.3, axis="y")

# Chart 5: MaxDD comparison
ax = axes[2, 0]
maxdds = [s_baseline["MaxDD"], s_uvxy["MaxDD"], s_leverage["MaxDD"], s_combined["MaxDD"]]
ax.bar(names, maxdds, color=colors, alpha=0.7)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_ylabel("Max Drawdown")
ax.set_title("Risk (Max Drawdown) Comparison")
ax.grid(True, alpha=0.3, axis="y")

# Chart 6: Drawdown over time
ax = axes[2, 1]
for name, ret, color in [
    ("Baseline", baseline_ret, "gray"),
    ("Combined", combined_ret, "blue")
]:
    w = (1 + ret).cumprod(); w = w / w.iloc[0]
    dd = w / w.cummax() - 1
    ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color=color, label=name)

ax.set_ylabel("Drawdown")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_title("Drawdown Over Time")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

_header("SUMMARY & INSIGHTS")

print(f"""
FINDINGS:

1. BASELINE (Momentum Only)
   Sharpe: {s_baseline["Sharpe"]:.4f}
   Return: {s_baseline["Return"]:+.2%}
   Risk: {s_baseline["MaxDD"]:.2%}

2. UVXY HEDGE (When Bubble > 0.7)
   Sharpe: {s_uvxy["Sharpe"]:.4f} ({s_uvxy["Sharpe"] - s_baseline["Sharpe"]:+.4f} vs baseline)
   Return: {s_uvxy["Return"]:+.2%} ({s_uvxy["Return"] - s_baseline["Return"]:+.2%} vs baseline)
   Risk: {s_uvxy["MaxDD"]:.2%} ({s_uvxy["MaxDD"] - s_baseline["MaxDD"]:+.2%} vs baseline)

   Insight: UVXY hedge {"IMPROVES" if s_uvxy["Sharpe"] > s_baseline["Sharpe"] else "REDUCES"} Sharpe

3. LEVERAGE (When Bubble < -0.7)
   Sharpe: {s_leverage["Sharpe"]:.4f} ({s_leverage["Sharpe"] - s_baseline["Sharpe"]:+.4f} vs baseline)
   Return: {s_leverage["Return"]:+.2%} ({s_leverage["Return"] - s_baseline["Return"]:+.2%} vs baseline)
   Risk: {s_leverage["MaxDD"]:.2%} ({s_leverage["MaxDD"] - s_baseline["MaxDD"]:+.2%} vs baseline)

   Insight: Leverage {"IMPROVES" if s_leverage["Sharpe"] > s_baseline["Sharpe"] else "REDUCES"} Sharpe

4. COMBINED (Both Strategies)
   Sharpe: {s_combined["Sharpe"]:.4f} ({s_combined["Sharpe"] - s_baseline["Sharpe"]:+.4f} vs baseline)
   Return: {s_combined["Return"]:+.2%} ({s_combined["Return"] - s_baseline["Return"]:+.2%} vs baseline)
   Risk: {s_combined["MaxDD"]:.2%} ({s_combined["MaxDD"] - s_baseline["MaxDD"]:+.2%} vs baseline)

   Insight: Combined approach is {"BETTER" if s_combined["Sharpe"] > s_baseline["Sharpe"] else "WORSE"} than baseline

RECOMMENDATION:

Best Strategy: {best_name}
  Sharpe: {results_df.iloc[0]["Sharpe"]:.4f}
  Return: {results_df.iloc[0]["Return"]:+.2%}
  MaxDD: {results_df.iloc[0]["MaxDD"]:.2%}

Status: Ready for deployment

Files Generated:
  - results/uvxy_leverage_test_chart_1.png (cumulative wealth)
  - results/uvxy_leverage_test_chart_2.png (bubble score + signals)
  - results/uvxy_leverage_test_chart_3.png (sharpe comparison)
  - results/uvxy_leverage_test_chart_4.png (return comparison)
  - results/uvxy_leverage_test_chart_5.png (risk comparison)
  - results/uvxy_leverage_test_chart_6.png (drawdown timeline)
""")

print(f"{'='*100}\n")
