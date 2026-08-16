"""
Hourly Momentum Portfolio Construction
========================================
Combines hourly momentum with existing strategies:
  - Fixed allocation (static weights)
  - Momentum allocation (dynamic rebalancing based on momentum signals)

Hourly Momentum: 300h lookback, 20h hold, top 20 stocks (Sharpe 3.551 @ 0.1% TC)
Other Strategies: Momentum (daily), IntradayMR, QQQBubble, SqueezeBubble
"""
import sys, warnings, os
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_n = [0]
def _save(*a, **k):
    _n[0] += 1; p = f"results/portfolio_chart_{_n[0]}.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); print(f"  [chart saved: {p}]", flush=True)
plt.show = _save

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from pathlib import Path

from data.db.schema import init

os.makedirs("results", exist_ok=True)

TRADING_DAYS = 252

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
    print(f"\n{'='*75}\n{t}\n{'='*75}", flush=True)


_header("Load Strategy Returns")

init()

# 1. Hourly Momentum (best: 300h/20h/20, Long only)
print("Loading hourly momentum...", flush=True)
xl_h = pd.ExcelFile("results/hourly_momentum.xlsx")
dr_h = pd.read_excel(xl_h, "Daily_Returns", index_col=0, parse_dates=True)
dr_h.index = pd.to_datetime(dr_h.index)
hourly_ret = dr_h["Long"].dropna()
hourly_ret.name = "Hourly_Mom"
print(f"  Hourly Momentum: {len(hourly_ret)} days, "
      f"{hourly_ret.index[0].date()} to {hourly_ret.index[-1].date()}")

# 2. Existing 4-strategy portfolio
print("Loading 4-strategy portfolio...", flush=True)
xl_4s = pd.ExcelFile("results/4strat_with_squeeze.xlsx")
dr_4s = pd.read_excel(xl_4s, "Daily_Returns", index_col=0, parse_dates=True)
dr_4s.index = pd.to_datetime(dr_4s.index)
print(f"  4-Strategy portfolio: {len(dr_4s)} days, "
      f"{dr_4s.index[0].date()} to {dr_4s.index[-1].date()}")
print(f"  Columns: {dr_4s.columns.tolist()}")

# Align to common period
common = hourly_ret.index.intersection(dr_4s.index)
print(f"\nCommon period: {common[0].date()} to {common[-1].date()} ({len(common)} days)")

hourly_ret = hourly_ret.loc[common]
dr_4s = dr_4s.loc[common]

# 3. Extract individual strategies from 4-strategy portfolio
# Assuming the columns are: Momentum, IntradayMR, QQQBubble, SqueezeBubble, Fixed, DynMom_All4, DynMom_QQQFixed
mom_ret = dr_4s["Momentum"]
intraday_ret = dr_4s["IntradayMR"]
bubble_ret = dr_4s["QQQBubble"]
squeeze_ret = dr_4s["SqueezeBubble"]

print(f"\nIndividual strategies loaded:")
print(f"  Momentum:      Sharpe={_sharpe(mom_ret):.3f}")
print(f"  IntradayMR:    Sharpe={_sharpe(intraday_ret):.3f}")
print(f"  QQQBubble:     Sharpe={_sharpe(bubble_ret):.3f}")
print(f"  SqueezeBubble: Sharpe={_sharpe(squeeze_ret):.3f}")
print(f"  Hourly Momentum: Sharpe={_sharpe(hourly_ret):.3f}")


# ══════════════════════════════════════════════════════════════════════════
# Portfolio Construction
# ══════════════════════════════════════════════════════════════════════════

_header("Portfolio Construction")

# Strategy 1: Hourly Momentum alone
strat_hourly = hourly_ret.rename("HourlyMom_Only")

# Strategy 2: Hourly + 3 Other Strategies (Fixed Allocation)
# Test different weights for hourly momentum relative to others
fixed_combos = []

# Combo 1: Equal weight (25% each)
equal_ret = (hourly_ret.fillna(0) + mom_ret.fillna(0) + intraday_ret.fillna(0) + bubble_ret.fillna(0)) / 4
equal_ret.name = "Equal_Weight_4Strat"
fixed_combos.append(("Equal_Weight_4Strat", equal_ret))

# Combo 2: Hourly gets 50%, others equal (16.67% each)
w50h_ret = (hourly_ret.fillna(0) * 0.5 + mom_ret.fillna(0) / 6 + intraday_ret.fillna(0) / 6 + bubble_ret.fillna(0) / 6)
w50h_ret.name = "Hourly_50pct_Others"
fixed_combos.append(("Hourly_50pct_Others", w50h_ret))

# Combo 3: Hourly 60%, others equal (13.33% each)
w60h_ret = (hourly_ret.fillna(0) * 0.6 + mom_ret.fillna(0) / 10 + intraday_ret.fillna(0) / 10 + bubble_ret.fillna(0) / 10)
w60h_ret.name = "Hourly_60pct_Others"
fixed_combos.append(("Hourly_60pct_Others", w60h_ret))

# Combo 4: Hourly 70%, Momentum 20%, others 5% each
w70h_ret = (hourly_ret.fillna(0) * 0.7 + mom_ret.fillna(0) * 0.2 + intraday_ret.fillna(0) * 0.05 + bubble_ret.fillna(0) * 0.05)
w70h_ret.name = "Hourly_70pct_Mix"
fixed_combos.append(("Hourly_70pct_Mix", w70h_ret))

# Combo 5: Hourly 40%, Momentum 40%, others 10% each
w40h40m_ret = (hourly_ret.fillna(0) * 0.4 + mom_ret.fillna(0) * 0.4 + intraday_ret.fillna(0) * 0.1 + bubble_ret.fillna(0) * 0.1)
w40h40m_ret.name = "Hourly_Momentum_Split"
fixed_combos.append(("Hourly_Momentum_Split", w40h40m_ret))

# Strategy 3: Momentum-based Allocation
# Rebalance based on momentum of each strategy
# Use 20-day rolling Sharpe as a proxy for momentum
window = 20

alloc_rows = []
for i in range(window, len(hourly_ret)):
    # Calculate recent sharpe (last 20 days) for each strategy
    h_sharp = np.sqrt(252) * hourly_ret.iloc[i-window:i].mean() / hourly_ret.iloc[i-window:i].std()
    m_sharp = np.sqrt(252) * mom_ret.iloc[i-window:i].mean() / mom_ret.iloc[i-window:i].std()
    intr_sharp = np.sqrt(252) * intraday_ret.iloc[i-window:i].mean() / intraday_ret.iloc[i-window:i].std()
    b_sharp = np.sqrt(252) * bubble_ret.iloc[i-window:i].mean() / bubble_ret.iloc[i-window:i].std()

    # Convert to weights (softmax-like, using exp to make positive)
    sharpes = np.array([h_sharp, m_sharp, intr_sharp, b_sharp])
    # Clip to avoid overflow, then exp
    sharpes_clipped = np.clip(sharpes, -10, 10)
    weights_raw = np.exp(sharpes_clipped)
    weights = weights_raw / weights_raw.sum()

    # Apply allocation
    port_ret = (hourly_ret.iloc[i] * weights[0] +
                mom_ret.iloc[i] * weights[1] +
                intraday_ret.iloc[i] * weights[2] +
                bubble_ret.iloc[i] * weights[3])

    alloc_rows.append(port_ret)

# Align alloc with main index (shift by window)
alloc_ret = pd.Series(alloc_rows, index=hourly_ret.index[window:])
alloc_ret.name = "MomentumAlloc_Dynamic"

print(f"\nFixed Allocation Combos:")
for name, ret in fixed_combos:
    s = _stats(ret)
    print(f"  {name:<30} Sharpe={s['Sharpe']:>6.3f}  Return={s['Total Return']:>+8.2%}  "
          f"MaxDD={s['Max DD']:>7.2%}")

s_alloc = _stats(alloc_ret)
print(f"  {'MomentumAlloc_Dynamic':<30} Sharpe={s_alloc['Sharpe']:>6.3f}  "
      f"Return={s_alloc['Total Return']:>+8.2%}  MaxDD={s_alloc['Max DD']:>7.2%}")

s_hourly = _stats(strat_hourly)
print(f"  {'HourlyMom_Only':<30} Sharpe={s_hourly['Sharpe']:>6.3f}  "
      f"Return={s_hourly['Total Return']:>+8.2%}  MaxDD={s_hourly['Max DD']:>7.2%}")


# ══════════════════════════════════════════════════════════════════════════
# Detailed Comparison
# ══════════════════════════════════════════════════════════════════════════

_header("Detailed Metrics")

all_results = [("Hourly_Mom_Only", strat_hourly)]
all_results.extend(fixed_combos)
all_results.append(("MomentumAlloc_Dynamic", alloc_ret))

metrics_list = []
for name, ret in all_results:
    s = _stats(ret)
    metrics_list.append({
        "Strategy": name,
        "Sharpe": s["Sharpe"],
        "Sortino": s["Sortino"],
        "Total_Return": s["Total Return"],
        "Max_DD": s["Max DD"],
    })

metrics_df = pd.DataFrame(metrics_list).sort_values("Sharpe", ascending=False)
print("\n" + metrics_df.to_string(index=False))

# Yearly breakdown
_header("Yearly Performance")

all_yrs = sorted(hourly_ret.index.year.unique())
yearly_data = []

for yr in all_yrs:
    row = {"Year": yr}
    for name, ret in all_results:
        yg = ret[ret.index.year == yr]
        if len(yg) < 5:
            row[f"{name[:15]}"] = np.nan
        else:
            w = (1+yg).cumprod(); w = w/w.iloc[0]
            row[f"{name[:15]}"] = float(w.iloc[-1]-1)
    yearly_data.append(row)

yearly_df = pd.DataFrame(yearly_data)
print("\n" + yearly_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════
# Charts
# ══════════════════════════════════════════════════════════════════════════

_header("Charts")

colors = {
    "Hourly_Mom_Only": "steelblue",
    "Equal_Weight_4Strat": "green",
    "Hourly_50pct_Others": "orange",
    "Hourly_70pct_Mix": "purple",
    "MomentumAlloc_Dynamic": "red",
}

# Chart 1: Cumulative wealth comparison
fig, ax = plt.subplots(figsize=(16, 8))

for name, ret in all_results:
    w = (1+ret).cumprod(); w = w/w.iloc[0]
    color = colors.get(name, "gray")
    s = _stats(ret)
    ax.plot(w.index, w.values, label=f"{name} (Sharpe={s['Sharpe']:.2f})",
            color=color, linewidth=2, alpha=0.8)

ax.set_title("Hourly Momentum Portfolio: Fixed vs Dynamic Allocation", fontsize=14, fontweight="bold")
ax.set_ylabel("Cumulative Wealth (rebased to 1)")
ax.set_xlabel("Date")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.1f}x"))
ax.legend(fontsize=9, loc="upper left"); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# Chart 2: Rolling Sharpe comparison
fig, ax = plt.subplots(figsize=(16, 8))

for name, ret in all_results[:4]:  # Top strategies
    roll = ret.rolling(60).apply(
        lambda x: float(np.sqrt(252)*x.mean()/x.std()) if x.std()>0 else np.nan)
    color = colors.get(name, "gray")
    ax.plot(roll.index, roll.values, label=name, color=color, linewidth=1.5, alpha=0.8)

ax.axhline(0, color="black", lw=0.8, linestyle="--")
ax.set_title("60-Day Rolling Sharpe Ratio", fontsize=14, fontweight="bold")
ax.set_ylabel("Sharpe Ratio")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# Chart 3: Drawdown comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for idx, (name, ret) in enumerate(all_results[:4]):
    ax = axes[idx]
    w = (1+ret).cumprod(); w = w/w.iloc[0]
    dd = w/w.cummax()-1
    ax.fill_between(dd.index, dd.values, 0, alpha=0.6, color=colors.get(name, "gray"))
    ax.set_title(f"{name}", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)

plt.tight_layout(); plt.show()


# ══════════════════════════════════════════════════════════════════════════
# Save Results
# ══════════════════════════════════════════════════════════════════════════

_header("Save Results")

xl = "results/hourly_momentum_portfolio.xlsx"
try:
    with pd.ExcelWriter(xl, engine="openpyxl") as writer:
        # Summary
        metrics_df.to_excel(writer, sheet_name="Summary", index=False)

        # Yearly breakdown
        yearly_df.to_excel(writer, sheet_name="Yearly", index=False)

        # Daily returns
        returns_df = pd.DataFrame(
            {name: ret for name, ret in all_results}
        )
        returns_df.to_excel(writer, sheet_name="Daily_Returns")

        # Individual strategies
        indiv_df = pd.DataFrame({
            "Hourly_Momentum": hourly_ret,
            "Daily_Momentum": mom_ret,
            "IntradayMR": intraday_ret,
            "QQQBubble": bubble_ret,
            "SqueezeBubble": squeeze_ret,
        })
        indiv_df.to_excel(writer, sheet_name="Individual_Strategies")

    print(f"  Saved: {xl}")
except Exception as e:
    print(f"  Save failed: {e}")


_header("SUMMARY")

best_idx = metrics_df["Sharpe"].idxmax()
best_row = metrics_df.iloc[best_idx]

print(f"\nBest Strategy: {best_row['Strategy']}")
print(f"  Sharpe Ratio:  {best_row['Sharpe']:.3f}")
print(f"  Sortino Ratio: {best_row['Sortino']:.3f}")
print(f"  Total Return:  {best_row['Total_Return']:+.2%}")
print(f"  Max Drawdown:  {best_row['Max_DD']:.2%}")

print(f"\nTop 3 Strategies:")
for i in range(min(3, len(metrics_df))):
    row = metrics_df.iloc[i]
    print(f"  {i+1}. {row['Strategy']:<30} Sharpe={row['Sharpe']:.3f}  "
          f"Return={row['Total_Return']:+.2%}  MaxDD={row['Max_DD']:.2%}")

print(f"\nResults saved to: {xl}")
print(f"Charts saved to: results/portfolio_chart_*.png")
