"""
Combined Portfolio: Book D + Book C + Book B
=============================================
Strategies:
  Book D: Contrarian Bubble   MA=104h, thresh=0.8, hold=8h,  top_n=20  | 2019-2026
  Book C: Intraday MR         sigma=4.0, lb=20d, flip=3d, top_n=5      | 2019-2026
  Book B: QQQ Bubble          MA=200h, Z=100h, buy=0.8, hold=24h       | 2020-2026

Portfolios:
  1. Fixed Equal Weight  : equal weight among available strategies (D+C from 2019,
                           D+C+B from 2020)
  2. Fixed D+C only      : 50% D + 50% C throughout (ignores B)
  3. Momentum Allocation : proportional to rolling return, max 50% per strategy,
                           grid search over lookback x rebalance
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import json
from pathlib import Path
from itertools import product

# -- Strategy modules --------------------------------------------------------
from strategies.contrarian_bubble_hourly import run_contrarian_bubble_hourly
from strategies.intraday_mean_reversion  import run_intraday_mean_reversion
from strategies.qqq_bubble_hourly        import run_qqq_bubble_hourly

TRADING_DAYS = 252
OUT_DIR = Path("results"); OUT_DIR.mkdir(exist_ok=True)


def _sharpe(s):
    std = s.std()
    return float(np.sqrt(TRADING_DAYS) * s.mean() / std) if std > 0 else np.nan

def _mdd(s):
    w = (1 + s).cumprod()
    return float((w / w.cummax() - 1).min())

def _cagr(s):
    w = (1 + s).cumprod()
    years = len(s) / TRADING_DAYS
    return float(w.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan

def _stats(s):
    ds = s[s < 0].std()
    so = float(np.sqrt(TRADING_DAYS) * s.mean() / ds) if ds > 0 else np.nan
    wr = float((s[s != 0] > 0).mean()) if (s != 0).any() else np.nan
    return dict(Sharpe=_sharpe(s), Sortino=so, CAGR=_cagr(s),
                Total_Return=float((1+s).cumprod().iloc[-1]-1),
                Max_DD=_mdd(s), Win_Rate=wr)


# ===========================================================================
# STEP 1: Load data
# ===========================================================================
print("=" * 72)
print("STEP 1: Loading data")
print("=" * 72)

# Merged hourly (2019-2026)
hc_all = pd.read_parquet("data/cache/merged_hourly_close.parquet")
ho_all = pd.read_parquet("data/cache/merged_hourly_open.parquet")
hc_all.index = hc_all.index.floor("h")
ho_all.index = ho_all.index.floor("h")
hc_all = hc_all[~hc_all.index.duplicated(keep="last")]
ho_all = ho_all[~ho_all.index.duplicated(keep="last")]

# Daily close (for Book C)
daily_all = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")

# QQQ hourly cache (2020-2026, Alpaca+yfinance)
qqq_cache = pd.read_parquet("data/cache/qqq_hourly.parquet")
qqq_hc = qqq_cache["close"].ffill()
qqq_ho = qqq_cache["open"].ffill()

# Common tickers for Books C+D
common = sorted(set(hc_all.columns) & set(ho_all.columns) & set(daily_all.columns))
valid  = [c for c in common
          if hc_all[c].isna().mean() < 0.30 and ho_all[c].isna().mean() < 0.30]
hc = hc_all[valid].ffill()
ho = ho_all[valid].ffill()
dc = daily_all[valid].ffill()
dc = dc.loc[str(hc.index[0].date()):str(hc.index[-1].date())]

print(f"Hourly universe : {len(valid)} tickers  {hc.index[0].date()} -> {hc.index[-1].date()}")
print(f"QQQ hourly      : {len(qqq_hc)} bars  {qqq_hc.index[0].date()} -> {qqq_hc.index[-1].date()}")


# ===========================================================================
# STEP 2: Run each strategy (single best-param combo)
# ===========================================================================
print("\n" + "=" * 72)
print("STEP 2: Running strategies (best params, single combo each)")
print("=" * 72)

# Book D - Contrarian Bubble
print("\n[D] Contrarian Bubble MA=104h, thresh=0.8, hold=8h, top_n=20...")
ret_D_raw, _, _ = run_contrarian_bubble_hourly(
    hourly_open=ho, hourly_close=hc,
    ma_window_grid=[104], buy_threshold_grid=[0.8],
    hold_hours_grid=[8], top_n_grid=[20],
)
ret_D = ret_D_raw.rename("BookD")
print(f"   Book D: Sharpe={_sharpe(ret_D):.3f}  CAGR={_cagr(ret_D):.1%}  MaxDD={_mdd(ret_D):.1%}")

# Book C - Intraday MR
print("\n[C] Intraday MR sigma=4.0, lookback=20d, flip=3d, top_n=5...")
ret_C_raw, _, _ = run_intraday_mean_reversion(
    daily_close=dc, hourly_open=ho, hourly_close=hc,
    sigma_grid=[4.0], flip_hold_days_grid=[3],
    lookback_grid=[20], top_n_grid=[5],
    transaction_cost=0.001, short_borrow_rate=0.08,
)
ret_C = ret_C_raw.rename("BookC")
print(f"   Book C: Sharpe={_sharpe(ret_C):.3f}  CAGR={_cagr(ret_C):.1%}  MaxDD={_mdd(ret_C):.1%}")

# Book B - QQQ Bubble
print("\n[B] QQQ Bubble MA=200h, Z=100h, buy=0.8, hold=24h...")
ret_B_raw, _, _ = run_qqq_bubble_hourly(
    hourly_open=qqq_ho, hourly_close=qqq_hc,
    ma_window_grid=[200], z_window_grid=[100],
    buy_threshold_grid=[0.8], short_threshold_grid=[0.95],
    hold_hours_grid=[24], transaction_cost=0.001,
    short_borrow_rate=0.08, enable_short=False,
)
ret_B = ret_B_raw.rename("BookB")
print(f"   Book B: Sharpe={_sharpe(ret_B):.3f}  CAGR={_cagr(ret_B):.1%}  MaxDD={_mdd(ret_B):.1%}")


# ===========================================================================
# STEP 3: Align to common index
# ===========================================================================
print("\n" + "=" * 72)
print("STEP 3: Aligning date ranges")
print("=" * 72)

# Full index = union of D and C (both 2019-2026); B starts later
dc_idx = ret_D.index.union(ret_C.index)
full_idx = pd.date_range(dc_idx.min(), dc_idx.max(), freq="B")

ret_D = ret_D.reindex(full_idx, fill_value=0.0)
ret_C = ret_C.reindex(full_idx, fill_value=0.0)
ret_B = ret_B.reindex(full_idx, fill_value=0.0)   # 0 before B's start (not active)

# Mark where B is truly active (has data)
b_start = qqq_hc.index[0].normalize()
b_active = full_idx >= b_start

rets_df = pd.DataFrame({"D": ret_D, "C": ret_C, "B": ret_B}, index=full_idx)
years = len(full_idx) / TRADING_DAYS
print(f"Portfolio period: {full_idx[0].date()} -> {full_idx[-1].date()}  ({years:.2f} yr)")
print(f"Book B active from: {b_start.date()}")


# ===========================================================================
# STEP 4: Fixed weight portfolios
# ===========================================================================
print("\n" + "=" * 72)
print("STEP 4: Fixed weight portfolios")
print("=" * 72)

S = "=" * 72

# Fixed 1: D+C only (50/50) throughout
port_DC = (0.5 * ret_D + 0.5 * ret_C).rename("Fixed_DC")

# Fixed 2: Equal weight among available
# 2019 to B start: 50/50 D+C; after B start: 33.3/33.3/33.3
port_EW = pd.Series(0.0, index=full_idx, name="Fixed_EW")
port_EW[~b_active] = (0.5 * ret_D[~b_active] + 0.5 * ret_C[~b_active])
port_EW[b_active]  = ((ret_D[b_active] + ret_C[b_active] + ret_B[b_active]) / 3)

for name, port in [("Fixed 50%D+50%C", port_DC),
                   ("Fixed EqualWeight (D+C then D+C+B)", port_EW)]:
    st = _stats(port)
    print(f"\n  {name}")
    print(f"    Sharpe={st['Sharpe']:.3f}  CAGR={st['CAGR']:.1%}  "
          f"Total={st['Total_Return']:+.1%}  MaxDD={st['Max_DD']:.1%}")


# ===========================================================================
# STEP 5: Momentum allocation (max 50% per strategy)
# ===========================================================================
print("\n" + S)
print("STEP 5: Momentum allocation grid search (max 50% cap per strategy)")
print(S)

MAX_ALLOC   = 0.50
LOOKBACK_GRID   = [20, 30, 60, 90, 120]   # days
REBALANCE_GRID  = [1, 5, 10, 20, 60]      # days

n = len(full_idx)
D_vals = ret_D.values
C_vals = ret_C.values
B_vals = ret_B.values


def run_mom_alloc(lookback: int, rebalance: int) -> pd.Series:
    port = np.zeros(n)
    wD = wC = wB = 0.0

    for i in range(n):
        # Recompute weights at rebalance points (after warmup)
        if i >= lookback and i % rebalance == 0:
            roll_D = float(np.prod(1 + D_vals[i-lookback:i]) - 1)
            roll_C = float(np.prod(1 + C_vals[i-lookback:i]) - 1)

            # B only active after its start date
            use_B = b_active[i] and (i - lookback) >= 0 and b_active[i - lookback]
            roll_B = float(np.prod(1 + B_vals[i-lookback:i]) - 1) if use_B else 0.0

            # Positive momentum only
            pos_D = max(0.0, roll_D)
            pos_C = max(0.0, roll_C)
            pos_B = max(0.0, roll_B) if use_B else 0.0
            total = pos_D + pos_C + pos_B

            if total > 1e-9:
                # Raw proportional weights
                rD = pos_D / total
                rC = pos_C / total
                rB = pos_B / total

                # Iterative capping at MAX_ALLOC (max 3 iterations sufficient)
                for _ in range(5):
                    excess_D = max(0.0, rD - MAX_ALLOC)
                    excess_C = max(0.0, rC - MAX_ALLOC)
                    excess_B = max(0.0, rB - MAX_ALLOC)
                    total_excess = excess_D + excess_C + excess_B
                    if total_excess < 1e-9:
                        break
                    rD = min(rD, MAX_ALLOC)
                    rC = min(rC, MAX_ALLOC)
                    rB = min(rB, MAX_ALLOC)
                    # Redistribute excess to uncapped strategies
                    uncapped = (rD < MAX_ALLOC) + (rC < MAX_ALLOC) + (rB < MAX_ALLOC)
                    if uncapped > 0:
                        each = total_excess / uncapped
                        if rD < MAX_ALLOC: rD = min(rD + each, MAX_ALLOC)
                        if rC < MAX_ALLOC: rC = min(rC + each, MAX_ALLOC)
                        if rB < MAX_ALLOC: rB = min(rB + each, MAX_ALLOC)

                wD, wC, wB = rD, rC, rB
            else:
                # All negative ? equal weight among active strategies
                n_active = 2 + (1 if use_B else 0)
                wD = wC = 1.0 / n_active
                wB = 1.0 / n_active if use_B else 0.0
        elif i < lookback:
            # Before warmup: equal weight among active
            use_B_now = b_active[i]
            n_active = 2 + (1 if use_B_now else 0)
            wD = wC = 1.0 / n_active
            wB = 1.0 / n_active if use_B_now else 0.0

        port[i] = wD * D_vals[i] + wC * C_vals[i] + wB * B_vals[i]

    return pd.Series(port, index=full_idx)


print(f"Grid: {len(LOOKBACK_GRID)} lookback x {len(REBALANCE_GRID)} rebalance = "
      f"{len(LOOKBACK_GRID)*len(REBALANCE_GRID)} combos")

mom_results = []
best_sh = -np.inf
best_port = None
best_params = None

for lb, rb in product(LOOKBACK_GRID, REBALANCE_GRID):
    port = run_mom_alloc(lb, rb)
    st   = _stats(port)
    mom_results.append(dict(lookback=lb, rebalance=rb, **st))
    if pd.notna(st["Sharpe"]) and st["Sharpe"] > best_sh:
        best_sh     = st["Sharpe"]
        best_port   = port.rename("MomAlloc_Best")
        best_params = dict(lookback=lb, rebalance=rb)

mom_df = pd.DataFrame(mom_results).sort_values("Sharpe", ascending=False)

print(f"\n{'Lookback':>9} {'Rebal':>6} {'Sharpe':>8} {'CAGR':>7} {'Total':>8} {'MaxDD':>8}")
print("-" * 56)
for _, r in mom_df.iterrows():
    print(f"{int(r['lookback']):>9} {int(r['rebalance']):>6} "
          f"{r['Sharpe']:>8.3f} {r['CAGR']:>7.1%} "
          f"{r['Total_Return']:>+8.1%} {r['Max_DD']:>8.1%}")

print(f"\nSensitivity - Lookback:")
for lb, grp in mom_df.groupby("lookback"):
    print(f"  {lb:>4}d  avg={grp['Sharpe'].mean():.3f}  best={grp['Sharpe'].max():.3f}")

print(f"\nSensitivity - Rebalance:")
for rb, grp in mom_df.groupby("rebalance"):
    print(f"  {rb:>4}d  avg={grp['Sharpe'].mean():.3f}  best={grp['Sharpe'].max():.3f}")


# ===========================================================================
# STEP 6: Summary comparison + yearly breakdown
# ===========================================================================
print("\n" + S)
print("STEP 6: Summary")
print(S)

portfolios = {
    "Book D only":        ret_D,
    "Book C only":        ret_C,
    "Book B only":        ret_B.where(b_active, 0.0),
    "Fixed 50/50 D+C":   port_DC,
    "Fixed EW (D+C+B)":  port_EW,
    f"MomAlloc lb={best_params['lookback']}d rb={best_params['rebalance']}d": best_port,
}

print(f"\n  {'Strategy':<32} {'Sharpe':>7} {'CAGR':>7} {'Total':>8} {'MaxDD':>8}")
print(f"  {'-'*32} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")
for name, port in portfolios.items():
    st = _stats(port)
    print(f"  {name:<32} {st['Sharpe']:>7.3f} {st['CAGR']:>7.1%} "
          f"{st['Total_Return']:>+8.1%} {st['Max_DD']:>8.1%}")

# Yearly breakdown for best mom-alloc
print(f"\n{S}")
print(f"YEARLY BREAKDOWN - MomAlloc (lb={best_params['lookback']}d, rb={best_params['rebalance']}d)  vs  Fixed DC  vs  Book D")
print(S)
print(f"  {'Year':>6} {'MomAlloc':>10} {'Fixed DC':>10} {'Book D':>8} {'Book C':>8} {'Book B':>8}")
print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

for yr in sorted(best_port.index.year.unique()):
    def yr_ret(s): return float((1 + s[s.index.year == yr]).prod() - 1)
    mA = yr_ret(best_port); dc50 = yr_ret(port_DC)
    bD = yr_ret(ret_D);     bC = yr_ret(ret_C);  bB = yr_ret(ret_B)
    flag = " <- B starts" if yr == b_start.year else ""
    print(f"  {yr:>6} {mA:>+10.2%} {dc50:>+10.2%} {bD:>+8.2%} {bC:>+8.2%} {bB:>+8.2%}{flag}")


# ===========================================================================
# STEP 7: Charts
# ===========================================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                          gridspec_kw={"height_ratios": [3, 1.5, 1.5]})

colors = {"Book D only": "steelblue", "Book C only": "darkorange",
          "Book B only": "purple", "Fixed 50/50 D+C": "gray",
          "Fixed EW (D+C+B)": "green"}
best_label = f"MomAlloc lb={best_params['lookback']}d rb={best_params['rebalance']}d"

for name, port in portfolios.items():
    w = (1 + port).cumprod(); w /= w.iloc[0]
    lw = 2.5 if "MomAlloc" in name else (1.8 if "Fixed EW" in name else 1.0)
    ls = "-" if ("MomAlloc" in name or "Fixed" in name) else "--"
    col = colors.get(name, "crimson")
    axes[0].plot(w.index, w.values, label=name, lw=lw, ls=ls, color=col)

st_best = _stats(best_port)
axes[0].set_title(
    f"Portfolio: Book D + C + B | MomAlloc best: lb={best_params['lookback']}d "
    f"rb={best_params['rebalance']}d | Sharpe={st_best['Sharpe']:.3f}  "
    f"CAGR={st_best['CAGR']:.1%}  MaxDD={st_best['Max_DD']:.1%}",
    fontsize=10
)
axes[0].set_ylabel("Cumulative Wealth")
axes[0].legend(fontsize=8, ncol=2); axes[0].grid(True, alpha=0.3)

# Drawdown
dd = (1 + best_port).cumprod()
dd = dd / dd.cummax() - 1
axes[1].fill_between(dd.index, dd.values, 0, alpha=0.5, color="crimson",
                     label=f"MomAlloc MaxDD={st_best['Max_DD']:.1%}")
dd_dc = (1 + port_DC).cumprod(); dd_dc = dd_dc / dd_dc.cummax() - 1
axes[1].plot(dd_dc.index, dd_dc.values, color="gray", lw=1.0, label="Fixed DC")
axes[1].set_ylabel("Drawdown"); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

# Yearly bar chart
years_list = sorted(best_port.index.year.unique())
ma_yr  = [float((1 + best_port[best_port.index.year==y]).prod()-1) for y in years_list]
dc_yr  = [float((1 + port_DC[port_DC.index.year==y]).prod()-1) for y in years_list]
x = np.arange(len(years_list)); w = 0.35
axes[2].bar(x - w/2, [r*100 for r in ma_yr], w, label="MomAlloc", color="crimson", alpha=0.7)
axes[2].bar(x + w/2, [r*100 for r in dc_yr], w, label="Fixed DC",  color="gray",   alpha=0.6)
axes[2].axhline(0, color="black", lw=0.8)
axes[2].set_xticks(x); axes[2].set_xticklabels([str(y) for y in years_list])
axes[2].set_ylabel("Return (%)"); axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out = OUT_DIR / "portfolio_dbc.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
plt.close()
print(f"\n[Chart saved: {out}]")

# Save grid
mom_df.to_excel(OUT_DIR / "portfolio_dbc_grid.xlsx", index=False)
print(f"[Grid saved: {OUT_DIR / 'portfolio_dbc_grid.xlsx'}]")

print("\nDONE")
