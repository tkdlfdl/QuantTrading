"""
Portfolio: Momentum + QQQBubble + SqueezeBubble  (no Reddit)
=============================================================
Loads pre-computed daily returns from saved Excel files.
Common period: Jun 2025 - Jun 2026 (~246 days, limited by SqueezeBubble).

Covers:
  A. Fixed weight grid  (step=10%)
  B. Momentum allocation -- all 3 dynamic, TC on rebalance
  C. Leverage on best fixed-weight portfolio (10%/yr borrow cost)

Usage:
  python run_3strat_mom_qqq_squeeze.py
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
    p = f"results/mqs_chart_{_n[0]}.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    print(f"  [chart saved: {p}]", flush=True)
plt.show = _save

import numpy as np
import pandas as pd
from itertools import product

os.makedirs("results", exist_ok=True)

TRADING_DAYS  = 252
TC_RATE       = 0.001          # 0.1% per unit of one-way turnover
LEVERAGE_COST = 0.10           # 10%/yr on borrowed capital
LEV_DAILY     = LEVERAGE_COST / TRADING_DAYS
WEIGHT_STEP   = 0.10
LEVERAGE_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]

DYN_LOOKBACK  = [10, 20, 40, 60]
DYN_HOLD      = [5, 10, 20, 40]
DYN_MAX_ALLOC = [0.4, 0.5, 0.6, 0.8, 1.0]


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

def _lever(ret, lev):
    return lev * ret - (lev - 1) * LEV_DAILY

def _weight_grid(names, step):
    vals = np.arange(0, 1+step, step).round(2)
    return [dict(zip(names, c)) for c in product(vals, repeat=len(names))
            if abs(sum(c)-1.0) < 1e-9]

def _apply_max_alloc(w_dict, max_alloc):
    if max_alloc >= 1.0: return w_dict
    w = dict(w_dict)
    for _ in range(len(w)+1):
        capped = {k: min(v, max_alloc) for k, v in w.items()}
        excess = sum(w[k]-capped[k] for k in w)
        if excess < 1e-9: return capped
        free = {k: v for k, v in capped.items() if v < max_alloc-1e-9}
        ft   = sum(free.values())
        if ft <= 0: return capped
        for k in free: capped[k] += excess*capped[k]/ft
        w = capped
    return w

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

def run_momentum_alloc(ret_df, names, lookback, hold_period, max_alloc, tc_rate=TC_RATE):
    port_rows = []
    prev_w    = None
    for i in range(lookback, len(ret_df)-1, hold_period):
        window  = ret_df[names].iloc[max(0, i-lookback):i]
        cum_ret = (1+window).prod()-1
        pos     = cum_ret.clip(lower=0); total = pos.sum()
        raw     = (pos/total).to_dict() if total > 0 else {n: 1/len(names) for n in names}
        new_w   = _apply_max_alloc(raw, max_alloc)
        if prev_w is not None:
            turnover   = sum(abs(new_w.get(k,0)-prev_w.get(k,0)) for k in set(list(new_w)+list(prev_w)))/2
            rebal_cost = turnover * tc_rate
        else:
            rebal_cost = 0.0
        hold_end = min(i+hold_period, len(ret_df))
        for j, ri in enumerate(range(i, hold_end)):
            date = ret_df.index[ri]
            r    = sum(ret_df.loc[date, k]*v for k,v in new_w.items())
            if j == 0: r -= rebal_cost
            port_rows.append({"date": date, "ret": r})
        prev_w = dict(new_w)
    if not port_rows: return None
    s = pd.DataFrame(port_rows).set_index("date")["ret"]
    return s[~s.index.duplicated(keep="last")]


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 -- Load returns
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 1 -- Load daily returns")

dr3  = pd.read_excel("results/3strategy_backtest.xlsx",
                     sheet_name="Daily_Returns", index_col=0, parse_dates=True)
dr3.index = pd.to_datetime(dr3.index)

drsq = (pd.read_excel("results/short_squeeze_bubble.xlsx",
                      sheet_name="Daily_Returns", parse_dates=["Date"])
          .set_index("Date")["Return"].rename("SqueezeBubble"))

common = (dr3["Momentum"].index
          .intersection(dr3["QQQBubble"].index)
          .intersection(drsq.index))

names  = ["Momentum", "QQQBubble", "SqueezeBubble"]
ret_df = pd.DataFrame({
    "Momentum":      dr3.loc[common, "Momentum"],
    "QQQBubble":     dr3.loc[common, "QQQBubble"],
    "SqueezeBubble": drsq.loc[common],
}).fillna(0)

print(f"Period: {common[0].date()} to {common[-1].date()} ({len(common)} days)")
print()
for name in names:
    s = ret_df[name]
    print(f"  {name:<16} Sharpe={_sharpe(s):>6.3f}  Sortino={_sortino(s):>6.3f}  "
          f"Return={(1+s).prod()-1:>+8.2%}  MaxDD={_mdd(s):>7.2%}")

_header("Individual yearly performance")
print_yearly(yearly_table(ret_df[names].to_dict("series")), names)


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 -- Fixed weight grid
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 2 -- Fixed weight grid (step=10%)")

combos = _weight_grid(names, WEIGHT_STEP)
print(f"Testing {len(combos)} combinations...\n", flush=True)

rows_A = []
for w in combos:
    port = sum(ret_df[n]*w[n] for n in names)
    s    = _stats(port)
    rows_A.append({**{f"w_{n}": w[n] for n in names}, **s})

dfA = pd.DataFrame(rows_A).sort_values("Sharpe", ascending=False)

print(f"{'w_Mom':>7} {'w_QQQ':>7} {'w_Sq':>7} | "
      f"{'Sharpe':>7} {'Sortino':>8} {'Return':>9} {'Max_DD':>8}")
print("-" * 57)
for _, r in dfA.head(20).iterrows():
    print(f"{r['w_Momentum']:>7.0%} {r['w_QQQBubble']:>7.0%} "
          f"{r['w_SqueezeBubble']:>7.0%} | "
          f"{r['Sharpe']:>7.3f} {r['Sortino']:>8.3f} "
          f"{r['Return']:>9.2%} {r['Max_DD']:>8.2%}")

bA = dfA.iloc[0]
bestA_ret = sum(ret_df[n]*bA[f"w_{n}"] for n in names).rename("Fixed")
w_desc_A  = " + ".join(f"{bA[f'w_{n}']:.0%} {n}" for n in names if bA[f"w_{n}"] > 0)
print(f"\nBest fixed: {w_desc_A}")
print(f"  Sharpe={bA['Sharpe']:.3f}  Sortino={bA['Sortino']:.3f}  "
      f"Return={bA['Return']:+.1%}  MaxDD={bA['Max_DD']:.1%}")

_header("Option A yearly")
print_yearly(yearly_table({"Fixed": bestA_ret}), ["Fixed"])


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 -- Momentum allocation (all 3, TC on rebalance)
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 3 -- Momentum allocation (all 3 dynamic, TC on rebalance)")

total_B = len(DYN_LOOKBACK)*len(DYN_HOLD)*len(DYN_MAX_ALLOC)
print(f"Grid: {len(DYN_LOOKBACK)} lb x {len(DYN_HOLD)} hold x {len(DYN_MAX_ALLOC)} max = {total_B} combos")
print(f"TC: {TC_RATE:.1%} per unit of one-way turnover\n", flush=True)

rows_B = []; bestB_sharpe = -np.inf; bestB_ret = None; bestB_params = None

for lb, hold, ma in product(DYN_LOOKBACK, DYN_HOLD, DYN_MAX_ALLOC):
    if lb >= len(ret_df): continue
    port = run_momentum_alloc(ret_df, names, lb, hold, ma)
    if port is None: continue
    s   = _stats(port)
    row = dict(lookback=lb, hold=hold, max_alloc=ma, **s)
    rows_B.append(row)
    if pd.notna(s["Sharpe"]) and s["Sharpe"] > bestB_sharpe:
        bestB_sharpe = s["Sharpe"]; bestB_ret = port.rename("DynMom"); bestB_params = row

dfB = pd.DataFrame(rows_B).sort_values("Sharpe", ascending=False)

print(f"{'Lookback':>9} {'Hold':>6} {'MaxAlloc':>9} | "
      f"{'Sharpe':>7} {'Sortino':>8} {'Return':>9} {'Max_DD':>8}")
print("-" * 58)
for _, r in dfB.head(20).iterrows():
    print(f"{int(r['lookback']):>9} {int(r['hold']):>6} {r['max_alloc']:>9.0%} | "
          f"{r['Sharpe']:>7.3f} {r['Sortino']:>8.3f} "
          f"{r['Return']:>9.2%} {r['Max_DD']:>8.2%}")

pB = bestB_params
print(f"\nBest: lb={pB['lookback']}d  hold={pB['hold']}d  max={pB['max_alloc']:.0%}")
print(f"  Sharpe={pB['Sharpe']:.3f}  Return={pB['Return']:+.1%}  MaxDD={pB['Max_DD']:.1%}")

_header("Momentum allocation yearly")
print_yearly(yearly_table({"DynMom": bestB_ret}), ["DynMom"])


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 -- Leverage on fixed-weight portfolio (joint grid: lev x weights)
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 4 -- Leverage analysis (10%/yr borrow cost)")
print(f"Leverage cost: {LEVERAGE_COST:.0%}/yr = {LEV_DAILY:.6f}/day\n")

# Part A: Fixed best weights at different leverage levels
print("-- Fixed best weights --")
print(f"{'Leverage':>10} {'AnnCost':>9} | {'Sharpe':>7} {'Sortino':>8} {'Return':>9} {'Max_DD':>8}")
print("-" * 60)
lev_series = {}
for lev in LEVERAGE_GRID:
    lret = _lever(bestA_ret, lev).rename(f"{lev:.1f}x")
    s    = _stats(lret)
    lev_series[f"{lev:.1f}x"] = lret
    print(f"{lev:>9.1f}x {(lev-1)*LEVERAGE_COST:>8.1%} | "
          f"{s['Sharpe']:>7.3f} {s['Sortino']:>8.3f} "
          f"{s['Return']:>9.2%} {s['Max_DD']:>8.2%}")

# Part B: Joint grid - leverage x weights
print(f"\n-- Joint grid: {len(LEVERAGE_GRID)} lev x {len(combos)} weight = "
      f"{len(LEVERAGE_GRID)*len(combos)} combos --\n", flush=True)

joint_rows = []
best_per_lev = {}
best_overall_sharpe = -np.inf; best_overall = None; best_overall_params = None

for lev in LEVERAGE_GRID:
    best_sh = -np.inf; best_row = None; best_ret_l = None
    for w in combos:
        port  = sum(ret_df[n]*w[n] for n in names)
        lport = _lever(port, lev)
        s     = _stats(lport)
        row   = {"leverage": lev, **{f"w_{n}": w[n] for n in names}, **s}
        joint_rows.append(row)
        if pd.notna(s["Sharpe"]) and s["Sharpe"] > best_sh:
            best_sh = s["Sharpe"]
            best_row = row
            best_ret_l = lport.rename(f"BestW_{lev:.1f}x")
    best_per_lev[lev] = (best_row, best_ret_l)
    if pd.notna(best_sh) and best_sh > best_overall_sharpe:
        best_overall_sharpe = best_sh; best_overall = best_ret_l; best_overall_params = best_row

joint_df = pd.DataFrame(joint_rows).sort_values("Sharpe", ascending=False)

print(f"{'Lev':>5} {'AnnCost':>8} | {'w_Mom':>7} {'w_QQQ':>7} {'w_Sq':>7} | "
      f"{'Sharpe':>7} {'Sortino':>8} {'Return':>9} {'Max_DD':>8}")
print("-" * 75)
for lev in LEVERAGE_GRID:
    r, _ = best_per_lev[lev]
    print(f"{lev:>4.1f}x {(lev-1)*LEVERAGE_COST:>7.1%} | "
          f"{r['w_Momentum']:>7.0%} {r['w_QQQBubble']:>7.0%} "
          f"{r['w_SqueezeBubble']:>7.0%} | "
          f"{r['Sharpe']:>7.3f} {r['Sortino']:>8.3f} "
          f"{r['Return']:>9.2%} {r['Max_DD']:>8.2%}")

_header("Top 20 joint grid results (lev x weights)")
print(f"{'Lev':>5} {'AnnCost':>8} | {'w_Mom':>7} {'w_QQQ':>7} {'w_Sq':>7} | "
      f"{'Sharpe':>7} {'Sortino':>8} {'Return':>9} {'Max_DD':>8}")
print("-" * 75)
for _, r in joint_df.head(20).iterrows():
    print(f"{r['leverage']:>4.1f}x {(r['leverage']-1)*LEVERAGE_COST:>7.1%} | "
          f"{r['w_Momentum']:>7.0%} {r['w_QQQBubble']:>7.0%} "
          f"{r['w_SqueezeBubble']:>7.0%} | "
          f"{r['Sharpe']:>7.3f} {r['Sortino']:>8.3f} "
          f"{r['Return']:>9.2%} {r['Max_DD']:>8.2%}")

_header("Yearly breakdown: fixed weights at each leverage level")
yr_lev = yearly_table(lev_series)
for yr in yr_lev.index:
    parts = []
    for lev in LEVERAGE_GRID:
        k  = f"{lev:.1f}x"
        rv = yr_lev.loc[yr, f"{k}_Ret"]
        sh = yr_lev.loc[yr, f"{k}_Sharpe"]
        md = yr_lev.loc[yr, f"{k}_MDD"]
        parts.append(f"{k}: {rv:>+7.2%} Sh={sh:.2f} DD={md:.2%}" if pd.notna(rv) else f"{k}: N/A")
    print(f"  {yr}:  " + "  |  ".join(parts))

_header("Yearly breakdown: best weights at each leverage level")
yr_best = yearly_table({f"{lev:.1f}x_best": s for lev,(_, s) in best_per_lev.items()})
for yr in yr_best.index:
    parts = []
    for lev in LEVERAGE_GRID:
        k  = f"{lev:.1f}x_best"
        rv = yr_best.loc[yr, f"{k}_Ret"]
        sh = yr_best.loc[yr, f"{k}_Sharpe"]
        md = yr_best.loc[yr, f"{k}_MDD"]
        parts.append(f"{lev:.1f}x: {rv:>+7.2%} Sh={sh:.2f} DD={md:.2%}" if pd.notna(rv) else f"{lev:.1f}x: N/A")
    print(f"  {yr}:  " + "  |  ".join(parts))


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 -- Full yearly comparison
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 5 -- Full yearly comparison")
all_series = {
    "Momentum":      ret_df["Momentum"],
    "QQQBubble":     ret_df["QQQBubble"],
    "SqueezeBubble": ret_df["SqueezeBubble"],
    "Fixed_1x":      bestA_ret,
    "Fixed_2x":      lev_series["2.0x"],
    "Fixed_3x":      lev_series["3.0x"],
    "DynMom_1x":     bestB_ret,
}
yr_all = yearly_table(all_series)
all_names_plot = list(all_series.keys())

def _fmt(df, sfx, fn):
    cols = [f"{n}{sfx}" for n in all_names_plot if f"{n}{sfx}" in df.columns]
    sub  = df[cols].copy(); sub.columns = [c.replace(sfx,"") for c in sub.columns]
    return sub.map(lambda x: fn(x) if pd.notna(x) else "N/A")

print("\n  YEARLY RETURN")
print(_fmt(yr_all,"_Ret",    lambda x: f"{x:>+7.2%}").to_string())
print("\n  YEARLY SHARPE")
print(_fmt(yr_all,"_Sharpe", lambda x: f"{x:>6.3f}").to_string())
print("\n  YEARLY MAX DRAWDOWN")
print(_fmt(yr_all,"_MDD",    lambda x: f"{x:>7.2%}").to_string())

_header("Full-period summary")
rows_s = []
for name, s in all_series.items():
    if s is None: continue
    w = (1+s).cumprod(); w = w/w.iloc[0]
    rows_s.append({"Strategy": name, "Start": str(s.index[0].date()),
                   "End": str(s.index[-1].date()), "Days": len(s),
                   "Total Return": float(w.iloc[-1]-1),
                   "Sharpe": _sharpe(s), "Sortino": _sortino(s), "Max DD": _mdd(s)})
sumdf = pd.DataFrame(rows_s).set_index("Strategy")
print(sumdf.to_string(float_format=lambda x: f"{x:+.3f}" if abs(x)<100 else f"{x:.1f}"))


# ══════════════════════════════════════════════════════════════════════════
# STEP 6 -- Charts
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 6 -- Charts")

colors = {"Momentum":"steelblue","QQQBubble":"darkorange","SqueezeBubble":"seagreen",
          "Fixed_1x":"crimson","Fixed_2x":"crimson","Fixed_3x":"crimson",
          "DynMom_1x":"navy"}

# Chart 1: cumulative wealth -- strategies + levered portfolios
fig, axes = plt.subplots(3, 1, figsize=(18,16), gridspec_kw={"height_ratios":[3,1.5,1.5]})

ax = axes[0]
for name, s in [("Momentum",ret_df["Momentum"]),("QQQBubble",ret_df["QQQBubble"]),
                ("SqueezeBubble",ret_df["SqueezeBubble"]),("DynMom_1x",bestB_ret)]:
    w = (1+s).cumprod(); w = w/w.iloc[0]
    ax.plot(w.index, w.values, label=name, color=colors[name], linewidth=1.5)

cmap_lev = ["#fee8e7","#fc9f92","#e32929"]
for (lev, label, col) in [(1.0,"Fixed_1x",cmap_lev[0]),
                           (2.0,"Fixed_2x",cmap_lev[1]),
                           (3.0,"Fixed_3x",cmap_lev[2])]:
    s = lev_series[f"{lev:.1f}x"]
    w = (1+s).cumprod(); w = w/w.iloc[0]
    ax.plot(w.index, w.values, label=f"Fixed {lev:.1f}x ({(lev-1)*LEVERAGE_COST:.0%}/yr cost)",
            color=col, linewidth=2.5, linestyle="--" if lev > 1 else "-")

ax.set_title("Momentum + QQQBubble + SqueezeBubble | Fixed weights + Leverage")
ax.set_ylabel("Cumulative Wealth")
ax.legend(fontsize=9); ax.grid(True,alpha=0.4)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.1f}x"))

# Drawdown
ax2 = axes[1]
for lev, col in zip(LEVERAGE_GRID, ["#2ecc71","#f39c12","#e67e22","#e74c3c","#c0392b"]):
    s = lev_series[f"{lev:.1f}x"]
    w = (1+s).cumprod(); w = w/w.iloc[0]; dd = w/w.cummax()-1
    ax2.plot(dd.index, dd.values, label=f"{lev:.1f}x", color=col, linewidth=1.5)
ax2.axhline(0,color="black",lw=0.8)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax2.set_ylabel("Drawdown"); ax2.legend(fontsize=8); ax2.grid(True,alpha=0.4)

# Sharpe vs leverage
ax3 = axes[2]
fixed_sharpes  = [_stats(_lever(bestA_ret,lev))["Sharpe"] for lev in LEVERAGE_GRID]
optim_sharpes  = [best_per_lev[lev][0]["Sharpe"] for lev in LEVERAGE_GRID]
dynmom_sharpes = [_stats(_lever(bestB_ret,lev))["Sharpe"] for lev in LEVERAGE_GRID]

ax3.plot(LEVERAGE_GRID, fixed_sharpes,  "o-",  color="crimson",  linewidth=2, markersize=7, label="Fixed best weights")
ax3.plot(LEVERAGE_GRID, optim_sharpes,  "s--", color="purple",   linewidth=2, markersize=7, label="Optimal weights per leverage")
ax3.plot(LEVERAGE_GRID, dynmom_sharpes, "^:",  color="navy",     linewidth=2, markersize=7, label="DynMom alloc (levered)")
ax3.set_xlabel("Leverage"); ax3.set_ylabel("Sharpe")
ax3.set_title("Sharpe vs Leverage (10%/yr borrow cost)")
ax3.legend(fontsize=9); ax3.grid(True,alpha=0.4)
ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.1f}x"))
plt.tight_layout(); plt.show()

# Chart 2: weight heatmap at 2x leverage
lev2 = joint_df[joint_df["leverage"]==2.0]
pivot= lev2.pivot_table(index="w_SqueezeBubble",columns="w_Momentum",values="Sharpe",aggfunc="max")
if not pivot.empty:
    fig2, ax = plt.subplots(figsize=(10,6))
    im = ax.imshow(pivot.values,aspect="auto",cmap="RdYlGn",vmin=0,vmax=5)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.0%}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{x:.0%}" for x in pivot.index])
    plt.colorbar(im,ax=ax,label="Sharpe (max over QQQ weight)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i,j]
            ax.text(j,i,f"{v:.2f}" if pd.notna(v) else "N/A",ha="center",va="center",fontsize=9)
    ax.set_xlabel("w_Momentum"); ax.set_ylabel("w_SqueezeBubble")
    ax.set_title("Sharpe at 2.0x Leverage: SqueezeBubble x Momentum weights")
    plt.tight_layout(); plt.show()


# ══════════════════════════════════════════════════════════════════════════
# STEP 7 -- Save
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 7 -- Save to Excel")
xl = "results/mqs_portfolio.xlsx"
try:
    with pd.ExcelWriter(xl, engine="openpyxl") as writer:
        sumdf.reset_index().to_excel(writer, sheet_name="Summary", index=False)
        _fmt(yr_all,"_Ret",    lambda x: f"{x:+.2%}").reset_index().to_excel(writer, sheet_name="Yearly_Return", index=False)
        _fmt(yr_all,"_Sharpe", lambda x: f"{x:.3f}").reset_index().to_excel(writer, sheet_name="Yearly_Sharpe",  index=False)
        _fmt(yr_all,"_MDD",    lambda x: f"{x:.2%}").reset_index().to_excel(writer, sheet_name="Yearly_MaxDD",   index=False)
        dfA.to_excel(writer, sheet_name="Grid_Fixed",    index=False)
        dfB.to_excel(writer, sheet_name="Grid_DynMom",   index=False)
        joint_df.to_excel(writer, sheet_name="Grid_Leverage", index=False)

        # Best weights per leverage
        bpl = [{"Leverage":f"{lev:.1f}x","AnnCost":f"{(lev-1)*LEVERAGE_COST:.0%}",
                "w_Mom":r["w_Momentum"],"w_QQQ":r["w_QQQBubble"],"w_Sq":r["w_SqueezeBubble"],
                "Sharpe":r["Sharpe"],"Sortino":r["Sortino"],
                "Return":f"{r['Return']:+.2%}","MaxDD":f"{r['Max_DD']:.2%}"}
               for lev,(r,_) in best_per_lev.items()]
        pd.DataFrame(bpl).to_excel(writer, sheet_name="Best_Per_Leverage", index=False)

        pd.DataFrame(all_series).to_excel(writer, sheet_name="Daily_Returns")
    print(f"  Saved: {xl}", flush=True)
except Exception as e:
    print(f"  Excel save failed: {e}", flush=True)


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════

_header("FINAL SUMMARY")
print(f"\nPortfolio: Momentum + QQQBubble + SqueezeBubble")
print(f"Period: {common[0].date()} to {common[-1].date()} ({len(common)} days)")
print(f"\nOption A (best fixed):   {w_desc_A}")
print(f"  Sharpe={bA['Sharpe']:.3f}  Return={bA['Return']:+.1%}  MaxDD={bA['Max_DD']:.1%}")
print(f"\nOption B (dynamic):      lb={pB['lookback']}d  hold={pB['hold']}d  max={pB['max_alloc']:.0%}")
print(f"  Sharpe={pB['Sharpe']:.3f}  Return={pB['Return']:+.1%}  MaxDD={pB['Max_DD']:.1%}")
print(f"\nLeverage on fixed best weights ({LEVERAGE_COST:.0%}/yr cost):")
print(f"{'Lev':>7} | {'Sharpe':>7} {'Return':>9} {'MaxDD':>8}")
for lev in LEVERAGE_GRID:
    s = _stats(lev_series[f"{lev:.1f}x"])
    print(f"{lev:>6.1f}x | {s['Sharpe']:>7.3f} {s['Return']:>9.2%} {s['Max_DD']:>8.2%}")
print(f"\nBest levered (joint grid): {best_overall_params['leverage']:.1f}x")
print(f"  Weights: {best_overall_params['w_Momentum']:.0%} Mom + {best_overall_params['w_QQQBubble']:.0%} QQQ + {best_overall_params['w_SqueezeBubble']:.0%} Squeeze")
print(f"  Sharpe={best_overall_params['Sharpe']:.3f}  Return={best_overall_params['Return']:+.1%}  MaxDD={best_overall_params['Max_DD']:.1%}")
