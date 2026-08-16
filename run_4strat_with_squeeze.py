"""
4-Strategy Portfolio: Momentum + IntradayMR + QQQBubble + SqueezeBubble
========================================================================
Loads pre-computed daily returns from Excel files (no re-running needed).
Common period: Jun 2025 - Jun 2026 (~246 trading days, limited by SqueezeBubble).

Allocation options:
  A. Fixed weight grid  (step=10%, all 4 strategies)
  B. Momentum allocation -- dynamic across all 4, TC deducted on rebalance
  C. Momentum allocation -- QQQBubble fixed anchor, rest dynamic + TC

Usage:
  python run_4strat_with_squeeze.py
"""
import sys, warnings, os
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_n = [0]
def _save(*a, **k):
    _n[0] += 1
    p = f"results/4strat_squeeze_chart_{_n[0]}.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    print(f"  [chart saved: {p}]", flush=True)
plt.show = _save

import numpy as np
import pandas as pd
from itertools import product

os.makedirs("results", exist_ok=True)

TRADING_DAYS = 252
TC_RATE      = 0.001   # 0.1% per unit of one-way turnover on rebalance
WEIGHT_STEP  = 0.10    # fixed-weight grid step

DYN_LOOKBACK   = [10, 20, 40, 60]
DYN_HOLD       = [5, 10, 20, 40]
DYN_MAX_ALLOC  = [0.4, 0.5, 0.6, 0.8, 1.0]
QQQ_FIXED_GRID = [0.0, 0.1, 0.2, 0.3]   # Option C: QQQ anchor weight


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

def _weight_grid(names, step):
    vals = np.arange(0, 1+step, step).round(2)
    return [dict(zip(names, c)) for c in product(vals, repeat=len(names))
            if abs(sum(c)-1.0) < 1e-9]

def _apply_max_alloc(w_dict, max_alloc):
    if max_alloc >= 1.0:
        return w_dict
    w = dict(w_dict)
    for _ in range(len(w)+1):
        capped = {k: min(v, max_alloc) for k, v in w.items()}
        excess = sum(w[k]-capped[k] for k in w)
        if excess < 1e-9: return capped
        free   = {k: v for k, v in capped.items() if v < max_alloc-1e-9}
        ft     = sum(free.values())
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

def _momentum_weights(ret_df, names, lookback, i, max_alloc):
    window  = ret_df[names].iloc[max(0, i-lookback):i]
    cum_ret = (1+window).prod()-1
    pos     = cum_ret.clip(lower=0)
    total   = pos.sum()
    if total > 0:
        raw = (pos/total).to_dict()
    else:
        raw = {n: 1/len(names) for n in names}
    return _apply_max_alloc(raw, max_alloc)


def run_momentum_alloc(ret_df, names, lookback, hold_period, max_alloc,
                       qqq_fixed=0.0, tc_rate=TC_RATE):
    """
    Dynamic momentum allocation.
    If qqq_fixed > 0: QQQBubble is anchored at that weight; remainder
    is distributed dynamically among the other strategies.
    TC is deducted proportional to one-way turnover at each rebalance.
    """
    dyn_names = [n for n in names if n != "QQQBubble"] if qqq_fixed > 0 else names
    remaining = 1.0 - qqq_fixed

    port_rows  = []
    prev_w     = None

    for i in range(lookback, len(ret_df)-1, hold_period):
        # Dynamic weights for non-QQQ strategies
        dyn_w  = _momentum_weights(ret_df, dyn_names, lookback, i, max_alloc)
        # Scale to remaining fraction
        new_w  = {n: dyn_w[n]*remaining for n in dyn_names}
        if qqq_fixed > 0:
            new_w["QQQBubble"] = qqq_fixed

        # Transaction cost: one-way turnover vs previous weights
        if prev_w is not None:
            all_keys = set(list(new_w)+list(prev_w))
            turnover = sum(abs(new_w.get(k,0)-prev_w.get(k,0)) for k in all_keys)/2
            rebal_cost = turnover * tc_rate
        else:
            rebal_cost = 0.0

        hold_end = min(i+hold_period, len(ret_df))
        for j, ri in enumerate(range(i, hold_end)):
            date = ret_df.index[ri]
            r    = sum(ret_df.loc[date, k]*v for k,v in new_w.items())
            if j == 0:
                r -= rebal_cost
            port_rows.append({"date": date, "ret": r})

        prev_w = dict(new_w)

    if not port_rows:
        return None

    port_ret = pd.DataFrame(port_rows).set_index("date")["ret"]
    return port_ret[~port_ret.index.duplicated(keep="last")]


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 -- Load returns from Excel files
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 1 -- Load daily returns")

# 3-strategy returns (Momentum, IntradayMR, QQQBubble)
xl3 = pd.ExcelFile("results/3strategy_backtest.xlsx")
dr3 = pd.read_excel(xl3, "Daily_Returns", index_col=0, parse_dates=True)
dr3.index = pd.to_datetime(dr3.index)

mom_ret_full = dr3["Momentum"]
mr_ret_full  = dr3["IntradayMR"]
qqq_ret_full = dr3["QQQBubble"]

# SqueezeBubble returns
xlsq  = pd.ExcelFile("results/short_squeeze_bubble.xlsx")
drsq  = pd.read_excel(xlsq, "Daily_Returns", parse_dates=["Date"]).set_index("Date")
sq_ret_full = drsq["Return"].rename("SqueezeBubble")

# Align to common period (limited by SqueezeBubble)
common = (mom_ret_full.index
          .intersection(mr_ret_full.index)
          .intersection(qqq_ret_full.index)
          .intersection(sq_ret_full.index))

mom_ret = mom_ret_full.loc[common]
mr_ret  = mr_ret_full.loc[common]
qqq_ret = qqq_ret_full.loc[common]
sq_ret  = sq_ret_full.loc[common]

names = ["Momentum", "IntradayMR", "QQQBubble", "SqueezeBubble"]
ret_df = pd.DataFrame({
    "Momentum":      mom_ret,
    "IntradayMR":    mr_ret,
    "QQQBubble":     qqq_ret,
    "SqueezeBubble": sq_ret,
}).fillna(0)   # data gaps (e.g. last day missing for IntradayMR) treated as flat

print(f"Common period: {common[0].date()} to {common[-1].date()} ({len(common)} trading days)")
print()
for name in names:
    s = ret_df[name]
    print(f"  {name:<16} Sharpe={_sharpe(s):>6.3f}  Sortino={_sortino(s):>6.3f}  "
          f"Return={(1+s).prod()-1:>+8.2%}  MaxDD={_mdd(s):>7.2%}")

_header("Individual strategy yearly performance")
print_yearly(yearly_table(ret_df[names].to_dict("series")), names)


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 -- Option A: Fixed weight grid
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 2 -- Option A: Fixed weight grid (step=10%)")

combos = _weight_grid(names, WEIGHT_STEP)
print(f"Testing {len(combos)} combinations...\n", flush=True)

rows_A = []
for w in combos:
    port = sum(ret_df[n]*w[n] for n in names)
    s    = _stats(port)
    rows_A.append({**{f"w_{n}": w[n] for n in names}, **s})

dfA = pd.DataFrame(rows_A).sort_values("Sharpe", ascending=False)

print(f"{'w_Mom':>7} {'w_MR':>7} {'w_QQQ':>7} {'w_Sq':>7} | "
      f"{'Sharpe':>7} {'Sortino':>8} {'Return':>9} {'Max_DD':>8}")
print("-" * 70)
for _, r in dfA.head(20).iterrows():
    print(f"{r['w_Momentum']:>7.0%} {r['w_IntradayMR']:>7.0%} "
          f"{r['w_QQQBubble']:>7.0%} {r['w_SqueezeBubble']:>7.0%} | "
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
# STEP 3 -- Option B: Momentum allocation (all 4 dynamic, with TC)
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 3 -- Option B: Momentum allocation (all 4 dynamic, TC on rebalance)")

total_B = len(DYN_LOOKBACK)*len(DYN_HOLD)*len(DYN_MAX_ALLOC)
print(f"Grid: {len(DYN_LOOKBACK)} lb x {len(DYN_HOLD)} hold x {len(DYN_MAX_ALLOC)} max = {total_B} combos")
print(f"TC: {TC_RATE:.1%} per unit of one-way turnover\n", flush=True)

rows_B       = []
bestB_sharpe = -np.inf
bestB_ret    = None
bestB_params = None

for lb, hold, ma in product(DYN_LOOKBACK, DYN_HOLD, DYN_MAX_ALLOC):
    if lb >= len(ret_df): continue
    port = run_momentum_alloc(ret_df, names, lb, hold, ma, qqq_fixed=0.0)
    if port is None: continue
    s   = _stats(port)
    row = dict(lookback=lb, hold=hold, max_alloc=ma, **s)
    rows_B.append(row)
    if pd.notna(s["Sharpe"]) and s["Sharpe"] > bestB_sharpe:
        bestB_sharpe = s["Sharpe"]
        bestB_ret    = port.rename("DynMom_All4")
        bestB_params = row

dfB = pd.DataFrame(rows_B).sort_values("Sharpe", ascending=False)

print(f"{'Lookback':>9} {'Hold':>6} {'MaxAlloc':>9} | "
      f"{'Sharpe':>7} {'Sortino':>8} {'Return':>9} {'Max_DD':>8}")
print("-" * 60)
for _, r in dfB.head(20).iterrows():
    print(f"{int(r['lookback']):>9} {int(r['hold']):>6} {r['max_alloc']:>9.0%} | "
          f"{r['Sharpe']:>7.3f} {r['Sortino']:>8.3f} "
          f"{r['Return']:>9.2%} {r['Max_DD']:>8.2%}")

pB = bestB_params
print(f"\nBest B: lb={pB['lookback']}d  hold={pB['hold']}d  max={pB['max_alloc']:.0%}")
print(f"  Sharpe={pB['Sharpe']:.3f}  Return={pB['Return']:+.1%}  MaxDD={pB['Max_DD']:.1%}")

_header("Option B yearly")
print_yearly(yearly_table({"DynMom_All4": bestB_ret}), ["DynMom_All4"])


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 -- Option C: Momentum allocation, QQQBubble fixed anchor
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 4 -- Option C: Momentum allocation, QQQBubble fixed + TC")

total_C = len(QQQ_FIXED_GRID)*len(DYN_LOOKBACK)*len(DYN_HOLD)*len(DYN_MAX_ALLOC)
print(f"Grid: {len(QQQ_FIXED_GRID)} qqq_fixed x {len(DYN_LOOKBACK)} lb x "
      f"{len(DYN_HOLD)} hold x {len(DYN_MAX_ALLOC)} max = {total_C} combos\n", flush=True)

rows_C       = []
bestC_sharpe = -np.inf
bestC_ret    = None
bestC_params = None

for qf, lb, hold, ma in product(QQQ_FIXED_GRID, DYN_LOOKBACK, DYN_HOLD, DYN_MAX_ALLOC):
    if lb >= len(ret_df): continue
    port = run_momentum_alloc(ret_df, names, lb, hold, ma, qqq_fixed=qf)
    if port is None: continue
    s   = _stats(port)
    row = dict(qqq_fixed=qf, lookback=lb, hold=hold, max_alloc=ma, **s)
    rows_C.append(row)
    if pd.notna(s["Sharpe"]) and s["Sharpe"] > bestC_sharpe:
        bestC_sharpe = s["Sharpe"]
        bestC_ret    = port.rename("DynMom_QQQFixed")
        bestC_params = row

dfC = pd.DataFrame(rows_C).sort_values("Sharpe", ascending=False)

print(f"{'QQQ_Fix':>8} {'Lookback':>9} {'Hold':>6} {'MaxAlloc':>9} | "
      f"{'Sharpe':>7} {'Sortino':>8} {'Return':>9} {'Max_DD':>8}")
print("-" * 70)
for _, r in dfC.head(20).iterrows():
    print(f"{r['qqq_fixed']:>8.0%} {int(r['lookback']):>9} {int(r['hold']):>6} "
          f"{r['max_alloc']:>9.0%} | "
          f"{r['Sharpe']:>7.3f} {r['Sortino']:>8.3f} "
          f"{r['Return']:>9.2%} {r['Max_DD']:>8.2%}")

pC = bestC_params
print(f"\nBest C: qqq_fixed={pC['qqq_fixed']:.0%}  lb={pC['lookback']}d  "
      f"hold={pC['hold']}d  max={pC['max_alloc']:.0%}")
print(f"  Sharpe={pC['Sharpe']:.3f}  Return={pC['Return']:+.1%}  MaxDD={pC['Max_DD']:.1%}")

_header("Option C yearly")
print_yearly(yearly_table({"DynMom_QQQFixed": bestC_ret}), ["DynMom_QQQFixed"])


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 -- Full yearly comparison
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 5 -- Full yearly comparison")

all_series = {
    "Momentum":        mom_ret,
    "IntradayMR":      mr_ret,
    "QQQBubble":       qqq_ret,
    "SqueezeBubble":   sq_ret,
    "Fixed":           bestA_ret,
    "DynMom_All4":     bestB_ret,
    "DynMom_QQQFixed": bestC_ret,
}

yr_all    = yearly_table(all_series)
all_names = list(all_series.keys())

def _fmt(df, sfx, fn):
    cols = [f"{n}{sfx}" for n in all_names if f"{n}{sfx}" in df.columns]
    sub  = df[cols].copy(); sub.columns = [c.replace(sfx,"") for c in sub.columns]
    return sub.map(lambda x: fn(x) if pd.notna(x) else "N/A")

print("\n  YEARLY RETURN")
print(_fmt(yr_all, "_Ret",    lambda x: f"{x:>+7.2%}").to_string())
print("\n  YEARLY SHARPE")
print(_fmt(yr_all, "_Sharpe", lambda x: f"{x:>6.3f}").to_string())
print("\n  YEARLY MAX DRAWDOWN")
print(_fmt(yr_all, "_MDD",    lambda x: f"{x:>7.2%}").to_string())


# ── Full-period summary ─────────────────────────────────────────────────────
_header("STEP 6 -- Full-period summary")

rows_sum = []
for name, s in all_series.items():
    if s is None: continue
    w = (1+s).cumprod(); w = w/w.iloc[0]
    rows_sum.append({"Strategy": name,
                     "Start": str(s.index[0].date()), "End": str(s.index[-1].date()),
                     "Days": len(s), "Total Return": float(w.iloc[-1]-1),
                     "Sharpe": _sharpe(s), "Sortino": _sortino(s), "Max DD": _mdd(s)})

sumdf = pd.DataFrame(rows_sum).set_index("Strategy")
print(sumdf.to_string(float_format=lambda x: f"{x:+.3f}" if abs(x)<100 else f"{x:.1f}"))


# ══════════════════════════════════════════════════════════════════════════
# STEP 7 -- Charts
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 7 -- Charts")

colors = {"Momentum":"steelblue","IntradayMR":"seagreen","QQQBubble":"darkorange",
          "SqueezeBubble":"mediumpurple",
          "Fixed":"crimson","DynMom_All4":"navy","DynMom_QQQFixed":"saddlebrown"}

# Chart 1: Cumulative wealth
fig, axes = plt.subplots(3, 1, figsize=(18, 16),
                          gridspec_kw={"height_ratios": [3, 2, 1.5]})
ax = axes[0]
for name, s in all_series.items():
    if s is None: continue
    w   = (1+s).cumprod(); w = w/w.iloc[0]
    lw  = 2.5 if name in ("Fixed","DynMom_All4","DynMom_QQQFixed") else 1.3
    ax.plot(w.index, w.values, label=name, color=colors.get(name,"gray"), linewidth=lw)
ax.set_title("4-Strategy Portfolio (Momentum + IntradayMR + QQQBubble + SqueezeBubble)")
ax.set_ylabel("Cumulative Wealth")
ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.1f}x"))

# Chart 2: Annual return bars for portfolios only
port_s = {"Fixed": bestA_ret, "DynMom_All4": bestB_ret, "DynMom_QQQFixed": bestC_ret}
all_yrs = sorted({y for s in port_s.values() if s is not None for y in s.index.year.unique()})
np_p = sum(1 for s in port_s.values() if s is not None)
width = 0.7/np_p; offsets = np.linspace(-(np_p-1)/2,(np_p-1)/2,np_p)*width
ax2 = axes[1]
for j,(name,s) in enumerate(port_s.items()):
    if s is None: continue
    rets = [float((1+s[s.index.year==yr]).prod()-1) if (s.index.year==yr).any() else 0 for yr in all_yrs]
    ax2.bar(np.arange(len(all_yrs))+offsets[j], rets, width=width,
            label=name, color=colors.get(name,"gray"), alpha=0.85)
ax2.axhline(0,color="black",lw=0.8)
ax2.set_xticks(range(len(all_yrs))); ax2.set_xticklabels(all_yrs)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax2.set_title("Portfolio Annual Returns"); ax2.legend(fontsize=9); ax2.grid(True,alpha=0.3,axis="y")

# Chart 3: Drawdown
ax3 = axes[2]
for name, s in port_s.items():
    if s is None: continue
    w = (1+s).cumprod(); w = w/w.iloc[0]; dd = w/w.cummax()-1
    ax3.fill_between(dd.index, dd.values, 0, alpha=0.35, color=colors.get(name,"gray"), label=name)
ax3.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax3.set_ylabel("Drawdown"); ax3.legend(fontsize=8); ax3.grid(True,alpha=0.3)
plt.tight_layout(); plt.show()

# Chart 2: Sharpe heatmap -- fixed grid by w_SqueezeBubble x w_Momentum
pivot_fixed = dfA.pivot_table(index="w_SqueezeBubble", columns="w_Momentum",
                               values="Sharpe", aggfunc="max")
if not pivot_fixed.empty:
    fig2, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot_fixed.values, aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=3)
    ax.set_xticks(range(len(pivot_fixed.columns)))
    ax.set_xticklabels([f"{x:.0%}" for x in pivot_fixed.columns])
    ax.set_yticks(range(len(pivot_fixed.index)))
    ax.set_yticklabels([f"{x:.0%}" for x in pivot_fixed.index])
    plt.colorbar(im, ax=ax, label="Sharpe (max over MR/QQQ weights)")
    for i in range(len(pivot_fixed.index)):
        for j in range(len(pivot_fixed.columns)):
            v = pivot_fixed.values[i,j]
            ax.text(j, i, f"{v:.2f}" if pd.notna(v) else "N/A",
                    ha="center", va="center", fontsize=9)
    ax.set_xlabel("w_Momentum"); ax.set_ylabel("w_SqueezeBubble")
    ax.set_title("Fixed Grid Sharpe: SqueezeBubble weight x Momentum weight")
    plt.tight_layout(); plt.show()


# ══════════════════════════════════════════════════════════════════════════
# STEP 8 -- Save to Excel
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 8 -- Save to Excel")
xl = "results/4strat_with_squeeze.xlsx"
try:
    with pd.ExcelWriter(xl, engine="openpyxl") as writer:
        sumdf.reset_index().to_excel(writer, sheet_name="Summary", index=False)

        _fmt(yr_all,"_Ret",   lambda x: f"{x:+.2%}").reset_index().to_excel(
            writer, sheet_name="Yearly_Return",  index=False)
        _fmt(yr_all,"_Sharpe",lambda x: f"{x:.3f}").reset_index().to_excel(
            writer, sheet_name="Yearly_Sharpe",  index=False)
        _fmt(yr_all,"_MDD",   lambda x: f"{x:.2%}").reset_index().to_excel(
            writer, sheet_name="Yearly_MaxDD",   index=False)

        dfA.to_excel(writer, sheet_name="Grid_Fixed",     index=False)
        dfB.to_excel(writer, sheet_name="Grid_DynAll4",   index=False)
        dfC.to_excel(writer, sheet_name="Grid_DynQQQFix", index=False)

        pd.DataFrame(all_series).to_excel(writer, sheet_name="Daily_Returns")
    print(f"  Saved: {xl}", flush=True)
except Exception as e:
    print(f"  Excel save failed: {e}", flush=True)


_header("DONE")
print(f"\nCommon period: {common[0].date()} to {common[-1].date()} ({len(common)} days)")
print(f"\nOption A (best fixed):        {w_desc_A}")
print(f"  Sharpe={bA['Sharpe']:.3f}  Return={bA['Return']:+.1%}  MaxDD={bA['Max_DD']:.1%}")
print(f"\nOption B (all-4 dynamic):     lb={pB['lookback']}d  hold={pB['hold']}d  max={pB['max_alloc']:.0%}")
print(f"  Sharpe={pB['Sharpe']:.3f}  Return={pB['Return']:+.1%}  MaxDD={pB['Max_DD']:.1%}")
print(f"\nOption C (QQQ={pC['qqq_fixed']:.0%} fixed, rest dynamic): lb={pC['lookback']}d  hold={pC['hold']}d  max={pC['max_alloc']:.0%}")
print(f"  Sharpe={pC['Sharpe']:.3f}  Return={pC['Return']:+.1%}  MaxDD={pC['Max_DD']:.1%}")
print(f"\nTC applied on rebalance: {TC_RATE:.1%} per unit of one-way turnover")
