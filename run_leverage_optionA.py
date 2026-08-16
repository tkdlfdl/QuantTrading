"""
Leveraged Portfolio: Momentum + QQQBubble + Reddit + SqueezeBubble
===================================================================
Applies leverage to the fixed-weight portfolio (Option A).

Leverage mechanics:
  levered_ret[t] = L * port_ret[t]  -  (L-1) * leverage_cost / 252
  cost = 10%/yr on borrowed amount = (L-1) of capital

Grid:
  Part 1 -- Fixed best weights x leverage grid  (show effect at fixed weights)
  Part 2 -- Joint grid: leverage x all weight combos  (find best levered allocation)

Usage:
  python run_leverage_optionA.py
"""
import os, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_n = [0]
def _save(*a, **k):
    _n[0] += 1
    p = f"results/lev_optA_chart_{_n[0]}.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    print(f"  [chart saved: {p}]", flush=True)
plt.show = _save

import numpy as np
import pandas as pd
from itertools import product

os.makedirs("results", exist_ok=True)

TRADING_DAYS     = 252
LEVERAGE_COST    = 0.10          # 10%/yr on borrowed capital
LEV_COST_DAILY   = LEVERAGE_COST / TRADING_DAYS
LEVERAGE_GRID    = [1.0, 1.5, 2.0, 2.5, 3.0]
WEIGHT_STEP      = 0.10


# ── Helpers ────────────────────────────────────────────────────────────────

def _sharpe(r, td=TRADING_DAYS):
    s = r.std(); return float(np.sqrt(td)*r.mean()/s) if s > 0 else np.nan

def _sortino(r, td=TRADING_DAYS):
    ds = r[r<0].std(); return float(np.sqrt(td)*r.mean()/ds) if ds > 0 else np.nan

def _mdd(r):
    w = (1+r).cumprod(); w = w/w.iloc[0]
    return float((w/w.cummax()-1).min())

def _stats(r):
    w = (1+r).cumprod(); w = w/w.iloc[0]
    return dict(Sharpe=_sharpe(r), Sortino=_sortino(r),
                Return=float(w.iloc[-1]-1), Max_DD=_mdd(r))

def _lever(ret: pd.Series, lev: float) -> pd.Series:
    """Scale returns by leverage and subtract daily borrowing cost."""
    return lev * ret - (lev - 1) * LEV_COST_DAILY

def _weight_grid(names, step):
    vals = np.arange(0, 1+step, step).round(2)
    return [dict(zip(names, c)) for c in product(vals, repeat=len(names))
            if abs(sum(c)-1.0) < 1e-9]

def _header(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}", flush=True)

def yearly_table(ret_dict):
    rows = []
    all_years = sorted({y for r in ret_dict.values() for y in r.index.year.unique()})
    for yr in all_years:
        row = {"Year": yr}
        for name, r in ret_dict.items():
            yr_r = r[r.index.year == yr]
            if yr_r.empty:
                row[f"{name}_Ret"] = np.nan; row[f"{name}_Sharpe"] = np.nan; row[f"{name}_MDD"] = np.nan
            else:
                w = (1+yr_r).cumprod(); w = w/w.iloc[0]
                row[f"{name}_Ret"]    = round(float(w.iloc[-1]-1), 4)
                row[f"{name}_Sharpe"] = round(_sharpe(yr_r), 3)
                row[f"{name}_MDD"]    = round(float((w/w.cummax()-1).min()), 4)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Year")

def print_yearly(df, names):
    for name in names:
        cols = [f"{name}_Ret", f"{name}_Sharpe", f"{name}_MDD"]
        present = [c for c in cols if c in df.columns]
        if not present: continue
        print(f"\n  [{name}]")
        sub = df[present].copy(); sub.columns = ["Return","Sharpe","MaxDD"]
        for yr, row in sub.iterrows():
            r  = f"{row['Return']:>+8.2%}" if pd.notna(row['Return'])  else "       N/A"
            sh = f"{row['Sharpe']:>7.3f}"  if pd.notna(row['Sharpe'])  else "    N/A"
            md = f"{row['MaxDD']:>8.2%}"   if pd.notna(row['MaxDD'])   else "     N/A"
            print(f"    {yr}:  Return={r}  Sharpe={sh}  MaxDD={md}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 -- Load returns
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 1 -- Load daily returns")

xl = pd.ExcelFile("results/portfolio_mqrs.xlsx")
dr = pd.read_excel(xl, "Daily_Returns", index_col=0, parse_dates=True)
dr.index = pd.to_datetime(dr.index)

names    = ["Momentum", "QQQBubble", "Reddit", "SqueezeBubble"]
ret_df   = dr[names].fillna(0)
n_days   = len(ret_df)

print(f"Period: {ret_df.index[0].date()} to {ret_df.index[-1].date()} ({n_days} days)")
for name in names:
    s = ret_df[name]
    print(f"  {name:<16} Sharpe={_sharpe(s):>6.3f}  "
          f"Return={(1+s).prod()-1:>+8.2%}  MaxDD={_mdd(s):>7.2%}")

# Unlevered Option A best weights (from previous run)
BEST_W = {"Momentum": 0.20, "QQQBubble": 0.50, "Reddit": 0.10, "SqueezeBubble": 0.20}
port_1x = sum(ret_df[n]*BEST_W[n] for n in names).rename("Portfolio_1x")
s1 = _stats(port_1x)
print(f"\nUnlevered Option A (20% Mom + 50% QQQ + 10% Reddit + 20% Squeeze):")
print(f"  Sharpe={s1['Sharpe']:.3f}  Sortino={s1['Sortino']:.3f}  "
      f"Return={s1['Return']:+.1%}  MaxDD={s1['Max_DD']:.1%}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 -- Part 1: Effect of leverage on the best fixed weights
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 2 -- Effect of leverage on fixed best weights (20/50/10/20)")
print(f"Leverage cost: {LEVERAGE_COST:.0%}/yr  =  {LEV_COST_DAILY:.6f}/day on borrowed capital\n")

lev_series = {}
print(f"{'Leverage':>10} {'Ann.Cost':>10} {'Sharpe':>8} {'Sortino':>8} "
      f"{'Return':>9} {'Max_DD':>8}")
print("-" * 62)
for lev in LEVERAGE_GRID:
    lev_ret       = _lever(port_1x, lev)
    s             = _stats(lev_ret)
    ann_cost      = (lev - 1) * LEVERAGE_COST
    lev_series[f"{lev:.1f}x"] = lev_ret.rename(f"Lev_{lev:.1f}x")
    print(f"{lev:>9.1f}x {ann_cost:>9.1%}  {s['Sharpe']:>8.3f} {s['Sortino']:>8.3f} "
          f"{s['Return']:>9.2%} {s['Max_DD']:>8.2%}")

_header("Yearly breakdown by leverage level")
yr_lev = yearly_table(lev_series)
for yr in yr_lev.index:
    row_parts = []
    for lev in LEVERAGE_GRID:
        key = f"{lev:.1f}x"
        r   = yr_lev.loc[yr, f"{key}_Ret"]
        sh  = yr_lev.loc[yr, f"{key}_Sharpe"]
        md  = yr_lev.loc[yr, f"{key}_MDD"]
        row_parts.append(
            f"{key}: {r:>+7.2%} Sh={sh:.2f} DD={md:.2%}" if pd.notna(r) else f"{key}: N/A"
        )
    print(f"  {yr}:  " + "  |  ".join(row_parts))


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 -- Part 2: Joint grid (leverage x weights)
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 3 -- Joint grid: leverage x weights (best at each leverage level)")

combos = _weight_grid(names, WEIGHT_STEP)
print(f"{len(LEVERAGE_GRID)} leverage levels x {len(combos)} weight combos "
      f"= {len(LEVERAGE_GRID)*len(combos)} total combinations\n", flush=True)

joint_results = []
best_per_lev  = {}       # lev -> best row
best_overall_sharpe = -np.inf
best_overall_ret    = None
best_overall_params = None

for lev in LEVERAGE_GRID:
    best_sharpe = -np.inf
    best_row    = None
    best_ret_l  = None

    for w in combos:
        port  = sum(ret_df[n]*w[n] for n in names)
        lport = _lever(port, lev)
        s     = _stats(lport)
        row   = {"leverage": lev, **{f"w_{n}": w[n] for n in names}, **s}
        joint_results.append(row)

        if pd.notna(s["Sharpe"]) and s["Sharpe"] > best_sharpe:
            best_sharpe = s["Sharpe"]
            best_row    = row
            best_ret_l  = lport.rename(f"BestW_Lev{lev:.1f}x")

    best_per_lev[lev] = (best_row, best_ret_l)

    if pd.notna(best_sharpe) and best_sharpe > best_overall_sharpe:
        best_overall_sharpe = best_sharpe
        best_overall_ret    = best_ret_l.rename("BestLev")
        best_overall_params = best_row

joint_df = pd.DataFrame(joint_results).sort_values("Sharpe", ascending=False)

# Print best weights at each leverage level
print(f"{'Lev':>5} {'AnnCost':>8} | {'w_Mom':>7} {'w_QQQ':>7} {'w_Red':>7} {'w_Sq':>7} | "
      f"{'Sharpe':>7} {'Sortino':>8} {'Return':>9} {'Max_DD':>8}")
print("-" * 85)
for lev in LEVERAGE_GRID:
    r, _ = best_per_lev[lev]
    print(f"{lev:>4.1f}x {(lev-1)*LEVERAGE_COST:>7.1%} | "
          f"{r['w_Momentum']:>7.0%} {r['w_QQQBubble']:>7.0%} "
          f"{r['w_Reddit']:>7.0%} {r['w_SqueezeBubble']:>7.0%} | "
          f"{r['Sharpe']:>7.3f} {r['Sortino']:>8.3f} "
          f"{r['Return']:>9.2%} {r['Max_DD']:>8.2%}")

p = best_overall_params
print(f"\nBest overall: {p['leverage']:.1f}x leverage")
print(f"  Weights: {p['w_Momentum']:.0%} Mom + {p['w_QQQBubble']:.0%} QQQ + "
      f"{p['w_Reddit']:.0%} Reddit + {p['w_SqueezeBubble']:.0%} Squeeze")
print(f"  Sharpe={p['Sharpe']:.3f}  Sortino={p['Sortino']:.3f}  "
      f"Return={p['Return']:+.1%}  MaxDD={p['Max_DD']:.1%}")

# Yearly breakdown for best at each leverage level
_header("Yearly breakdown -- best weights per leverage level")
best_lev_series = {f"{lev:.1f}x": s for lev, (_, s) in best_per_lev.items()}
yr_joint = yearly_table(best_lev_series)
for yr in yr_joint.index:
    row_parts = []
    for lev in LEVERAGE_GRID:
        key = f"{lev:.1f}x"
        r   = yr_joint.loc[yr, f"{key}_Ret"]
        sh  = yr_joint.loc[yr, f"{key}_Sharpe"]
        md  = yr_joint.loc[yr, f"{key}_MDD"]
        row_parts.append(
            f"{key}: {r:>+7.2%} Sh={sh:.2f} DD={md:.2%}" if pd.notna(r) else f"{key}: N/A"
        )
    print(f"  {yr}:  " + "  |  ".join(row_parts))


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 -- Top 20 combos from joint grid
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 4 -- Top 20 overall by Sharpe (joint grid)")
print(f"{'Lev':>5} {'AnnCost':>8} | {'w_Mom':>7} {'w_QQQ':>7} {'w_Red':>7} {'w_Sq':>7} | "
      f"{'Sharpe':>7} {'Sortino':>8} {'Return':>9} {'Max_DD':>8}")
print("-" * 85)
for _, r in joint_df.head(20).iterrows():
    print(f"{r['leverage']:>4.1f}x {(r['leverage']-1)*LEVERAGE_COST:>7.1%} | "
          f"{r['w_Momentum']:>7.0%} {r['w_QQQBubble']:>7.0%} "
          f"{r['w_Reddit']:>7.0%} {r['w_SqueezeBubble']:>7.0%} | "
          f"{r['Sharpe']:>7.3f} {r['Sortino']:>8.3f} "
          f"{r['Return']:>9.2%} {r['Max_DD']:>8.2%}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 -- Charts
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 5 -- Charts")

# Chart 1: Cumulative wealth -- fixed weights at different leverage levels
fig, axes = plt.subplots(3, 1, figsize=(18, 16),
                          gridspec_kw={"height_ratios": [3, 1.5, 1.5]})

cmap = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(LEVERAGE_GRID)))
ax = axes[0]
for (key, s), col in zip(lev_series.items(), cmap):
    w  = (1+s).cumprod(); w = w/w.iloc[0]
    lw = 2.5 if key == "1.0x" else 1.8
    ax.plot(w.index, w.values, label=key, color=col, linewidth=lw)
ax.set_title(f"Fixed Weights (20% Mom + 50% QQQ + 10% Reddit + 20% Squeeze) | "
             f"Leverage cost: {LEVERAGE_COST:.0%}/yr", fontsize=11)
ax.set_ylabel("Cumulative Wealth")
ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.1f}x"))

# Drawdown for each leverage level
ax2 = axes[1]
for (key, s), col in zip(lev_series.items(), cmap):
    w  = (1+s).cumprod(); w = w/w.iloc[0]; dd = w/w.cummax()-1
    ax2.plot(dd.index, dd.values, label=key, color=col, linewidth=1.5)
ax2.axhline(0, color="black", lw=0.8)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax2.set_ylabel("Drawdown"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.4)

# Sharpe vs leverage (fixed weights vs optimal weights)
ax3 = axes[2]
fixed_sharpes = [_stats(_lever(port_1x, lev))["Sharpe"] for lev in LEVERAGE_GRID]
optim_sharpes = [best_per_lev[lev][0]["Sharpe"] for lev in LEVERAGE_GRID]
ax3.plot(LEVERAGE_GRID, fixed_sharpes, "o-", color="steelblue",
         linewidth=2, markersize=7, label="Fixed weights (20/50/10/20)")
ax3.plot(LEVERAGE_GRID, optim_sharpes, "s--", color="crimson",
         linewidth=2, markersize=7, label="Optimal weights per leverage")
ax3.set_xlabel("Leverage"); ax3.set_ylabel("Sharpe Ratio")
ax3.set_title("Sharpe vs Leverage Level")
ax3.legend(fontsize=9); ax3.grid(True, alpha=0.4)
ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.1f}x"))

plt.tight_layout(); plt.show()

# Chart 2: Yearly return grouped bars -- best weights at each leverage level
yr_best_series = {f"{lev:.1f}x_Best": s for lev, (_, s) in best_per_lev.items()}
yr_data = yearly_table(yr_best_series)
all_yrs = list(yr_data.index)
n_l     = len(LEVERAGE_GRID)
width   = 0.7/n_l
offsets = np.linspace(-(n_l-1)/2,(n_l-1)/2,n_l)*width

fig2, ax = plt.subplots(figsize=(12, 6))
for j,(lev,col) in enumerate(zip(LEVERAGE_GRID, cmap)):
    key  = f"{lev:.1f}x_Best"
    rets = [float(yr_data.loc[yr, f"{key}_Ret"]) if pd.notna(yr_data.loc[yr, f"{key}_Ret"]) else 0
            for yr in all_yrs]
    ax.bar(np.arange(len(all_yrs))+offsets[j], rets, width=width,
           label=f"{lev:.1f}x (best weights)", color=col, alpha=0.85)

ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(len(all_yrs))); ax.set_xticklabels(all_yrs)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_title("Annual Return by Leverage Level (optimal weights)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout(); plt.show()

# Chart 3: Heatmap -- Sharpe at 2x leverage: w_Momentum x w_SqueezeBubble
lev2_df = joint_df[joint_df["leverage"] == 2.0]
pivot   = lev2_df.pivot_table(index="w_SqueezeBubble", columns="w_Momentum",
                               values="Sharpe", aggfunc="max")
if not pivot.empty:
    fig3, ax3h = plt.subplots(figsize=(10, 6))
    im = ax3h.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=5)
    ax3h.set_xticks(range(len(pivot.columns)))
    ax3h.set_xticklabels([f"{x:.0%}" for x in pivot.columns])
    ax3h.set_yticks(range(len(pivot.index)))
    ax3h.set_yticklabels([f"{x:.0%}" for x in pivot.index])
    plt.colorbar(im, ax=ax3h, label="Sharpe (max over QQQ/Reddit weights)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            ax3h.text(j, i, f"{v:.2f}" if pd.notna(v) else "N/A",
                      ha="center", va="center", fontsize=9)
    ax3h.set_xlabel("w_Momentum"); ax3h.set_ylabel("w_SqueezeBubble")
    ax3h.set_title("Sharpe at 2.0x Leverage: SqueezeBubble x Momentum weights")
    plt.tight_layout(); plt.show()


# ══════════════════════════════════════════════════════════════════════════
# STEP 6 -- Save
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 6 -- Save to Excel")
xl_out = "results/leveraged_optionA.xlsx"
try:
    with pd.ExcelWriter(xl_out, engine="openpyxl") as writer:

        # Summary: best weights at each leverage level
        sum_rows = []
        for lev in LEVERAGE_GRID:
            r, s = best_per_lev[lev]
            sum_rows.append({
                "Leverage": f"{lev:.1f}x",
                "Ann_BorrowCost": f"{(lev-1)*LEVERAGE_COST:.0%}",
                "w_Momentum":      r["w_Momentum"],
                "w_QQQBubble":     r["w_QQQBubble"],
                "w_Reddit":        r["w_Reddit"],
                "w_SqueezeBubble": r["w_SqueezeBubble"],
                "Sharpe":          round(r["Sharpe"],3),
                "Sortino":         round(r["Sortino"],3),
                "Total Return":    f"{r['Return']:+.2%}",
                "Max DD":          f"{r['Max_DD']:.2%}",
            })
        pd.DataFrame(sum_rows).to_excel(writer, sheet_name="Best_Per_Leverage", index=False)

        # Fixed weights at different leverage levels
        fixed_rows = []
        for lev in LEVERAGE_GRID:
            lret = _lever(port_1x, lev)
            s    = _stats(lret)
            fixed_rows.append({
                "Leverage": f"{lev:.1f}x",
                "Ann_BorrowCost": f"{(lev-1)*LEVERAGE_COST:.0%}",
                "Weights": "20% Mom + 50% QQQ + 10% Reddit + 20% Squeeze",
                "Sharpe":  round(s["Sharpe"],3),
                "Sortino": round(s["Sortino"],3),
                "Return":  f"{s['Return']:+.2%}",
                "Max_DD":  f"{s['Max_DD']:.2%}",
            })
        pd.DataFrame(fixed_rows).to_excel(writer, sheet_name="Fixed_Weights_Levered", index=False)

        # Yearly breakdown (fixed weights x leverage)
        yr_out = yearly_table(lev_series)
        yr_out.reset_index().to_excel(writer, sheet_name="Yearly_FixedW", index=False)

        # Yearly breakdown (best weights x leverage)
        yr_best_out = yearly_table(yr_best_series)
        yr_best_out.reset_index().to_excel(writer, sheet_name="Yearly_BestW", index=False)

        # Full joint grid
        joint_df.to_excel(writer, sheet_name="Joint_Grid", index=False)

        # Daily returns for all leverage levels (fixed weights)
        pd.DataFrame({k: s for k, s in lev_series.items()}).to_excel(
            writer, sheet_name="Daily_Returns_FixedW")

    print(f"  Saved: {xl_out}", flush=True)
except Exception as e:
    print(f"  Excel save failed: {e}", flush=True)


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════

_header("SUMMARY")
print(f"\nLeverage cost: {LEVERAGE_COST:.0%}/yr on borrowed capital\n")
print(f"  -- Fixed weights (20% Mom + 50% QQQ + 10% Reddit + 20% Squeeze) --")
print(f"{'Leverage':>10} {'AnnCost':>9} | {'Sharpe':>7} {'Sortino':>8} {'Return':>9} {'MaxDD':>8}")
print("-" * 60)
for lev in LEVERAGE_GRID:
    s = _stats(_lever(port_1x, lev))
    print(f"{lev:>9.1f}x {(lev-1)*LEVERAGE_COST:>8.1%} | "
          f"{s['Sharpe']:>7.3f} {s['Sortino']:>8.3f} "
          f"{s['Return']:>9.2%} {s['Max_DD']:>8.2%}")

print(f"\n  -- Optimal weights at each leverage level --")
print(f"{'Leverage':>10} {'AnnCost':>9} | {'w_Mom':>7} {'w_QQQ':>7} {'w_Red':>7} {'w_Sq':>7} | {'Sharpe':>7} {'Return':>9} {'MaxDD':>8}")
print("-" * 88)
for lev in LEVERAGE_GRID:
    r, _ = best_per_lev[lev]
    print(f"{lev:>9.1f}x {(lev-1)*LEVERAGE_COST:>8.1%} | "
          f"{r['w_Momentum']:>7.0%} {r['w_QQQBubble']:>7.0%} "
          f"{r['w_Reddit']:>7.0%} {r['w_SqueezeBubble']:>7.0%} | "
          f"{r['Sharpe']:>7.3f} {r['Return']:>9.2%} {r['Max_DD']:>8.2%}")
