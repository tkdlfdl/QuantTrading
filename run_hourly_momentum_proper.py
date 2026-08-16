"""
Hourly Cross-Sectional Momentum -- Proper Backtest (2019-2026)
==============================================================
Fixes two critical flaws in the previous hourly_momentum.xlsx run:

  1. OVERLAP LEVERAGE: previous code entered a new batch every hour while
     prior holds were still open, creating ~3x implicit leverage.
     FIX: non-overlapping holds -- rebalance only after current hold closes.

  2. SHORT DATA WINDOW: previous run covered Aug-2024 to Jun-2026 (1.9yr),
     missing 2019-2023 entirely.
     FIX: full 2019-2026 period using the merged Alpaca+yfinance hourly cache.

Daily returns are computed bar-by-bar (entry bar: open->close, subsequent
bars: close->close), then compounded within each trading day.

Comparison: vs Book A (Daily Momentum + 1.25x Leverage + UVXY hedge).

Grid: lookback=[40,80,120,200,300]h x hold=[4,8,13,20,40]h x top_n=[5,10,20]
      = 75 combos (long-only)
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

TRADING_DAYS = 252
TC_ONE_WAY   = 0.001   # 0.1% each way -> 0.2% round-trip
OUT_DIR      = Path("results"); OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _sharpe(s):
    std = s.std()
    return float(np.sqrt(TRADING_DAYS) * s.mean() / std) if std > 0 else np.nan

def _mdd(s):
    w = (1 + s).cumprod()
    return float((w / w.cummax() - 1).min())

def _cagr(s):
    w  = (1 + s).cumprod()
    yr = len(s) / TRADING_DAYS
    return float(w.iloc[-1] ** (1 / yr) - 1) if yr > 0 else np.nan

def _stats(s):
    ds = s[s < 0].std()
    so = float(np.sqrt(TRADING_DAYS) * s.mean() / ds) if ds > 0 else np.nan
    wr = float((s[s != 0] > 0).mean()) if (s != 0).any() else np.nan
    return dict(Sharpe=_sharpe(s), Sortino=so, CAGR=_cagr(s),
                Total_Return=float((1+s).cumprod().iloc[-1]-1),
                Max_DD=_mdd(s), Win_Rate=wr)


# ---------------------------------------------------------------------------
# Book A (Daily Momentum + Leverage + UVXY) -- inline replica of live/settle.py
# ---------------------------------------------------------------------------
A_PARAMS = dict(
    lookback_days=140, rebalance_days=40, top_n=5,
    bubble_ma_days=120, bubble_z_days=240,
    lev_threshold=-0.88, lev_mult=0.25, lev_hold_days=50,
    hedge_threshold=0.85, hedge_alloc=0.50, hedge_hold_days=40,
    tc_per_cycle=0.010, lev_cost_ann=0.10,
)

def _bubble(equity, ma_days, z_days):
    lp   = np.log(np.maximum(equity, 1e-9))
    fair = lp.rolling(ma_days).mean()
    r    = lp - fair
    z    = (r - r.rolling(z_days).mean()) / r.rolling(z_days).std()
    return np.tanh(z / 2)

def run_book_a(daily_close):
    p        = A_PARAMS
    close    = daily_close.copy()
    lookback = p["lookback_days"]; holding = p["rebalance_days"]; top = p["top_n"]
    ret_d    = close.pct_change().ffill().fillna(0)
    ret_mom  = close.pct_change(lookback).ffill().fillna(0)

    rows = []
    for i in range(lookback + 1, len(ret_mom), holding):
        ranked = np.argsort(ret_mom.iloc[i-1:i].rank(axis=1, ascending=False).values[0])
        for j in range(i, min(i + holding, len(ret_mom))):
            date  = ret_d.index[j]
            ls    = np.sign(ret_mom.iloc[:, ranked[:top]].iloc[i-1:i]).abs()
            lr    = ls.mul(np.array(ret_d.iloc[:, ranked[:top]].iloc[j:j+1])[0])
            lret  = lr.values.mean() * top
            mom_r = lret / top - p["tc_per_cycle"] / holding
            h_ret = 0.0
            if "UVXY" in close.columns and pd.notna(close.loc[date, "UVXY"]):
                h_ret = ret_d.loc[date, "UVXY"]
            elif "^VIX" in close.columns and pd.notna(close.loc[date, "^VIX"]):
                v     = ret_d.loc[date, "^VIX"]
                h_ret = (2.0*v - 0.002 - 0.25*v**2
                         if date < pd.Timestamp("2018-02-28")
                         else 1.5*v - 0.0015 - 0.25*v**2)
            rows.append({"Date": date, "Momentum": mom_r, "Hedge": h_ret})

    dfA    = pd.DataFrame(rows).set_index("Date").dropna()
    base_w = (1 + dfA).cumprod() / (1 + dfA).cumprod().iloc[0]
    bub    = _bubble(base_w["Momentum"], p["bubble_ma_days"], p["bubble_z_days"])
    h_sig  = (bub > p["hedge_threshold"]).shift(1).fillna(False)
    l_sig  = (bub < p["lev_threshold"]).shift(1).fillna(False)
    lev_c  = p["lev_cost_ann"] / TRADING_DAYS

    out = []; h_rem = l_rem = 0
    for date in dfA.index:
        if h_rem == 0 and h_sig.loc[date]: h_rem = p["hedge_hold_days"]
        if l_rem == 0 and l_sig.loc[date]: l_rem = p["lev_hold_days"]
        base = dfA.loc[date, "Momentum"]
        if h_rem > 0:
            r = (1 - p["hedge_alloc"])*base + p["hedge_alloc"]*dfA.loc[date, "Hedge"]
            h_rem -= 1
        elif l_rem > 0:
            r = base + p["lev_mult"]*base - p["lev_mult"]*lev_c
            l_rem -= 1
        else:
            r = base
        out.append(r)

    return pd.Series(out, index=dfA.index)


# ---------------------------------------------------------------------------
# Hourly momentum -- non-overlapping, vectorised
# ---------------------------------------------------------------------------
def run_hourly_mom_combo(hc_np, ho_np, bar_ts, mom_np, lb, hold, top_n):
    """
    Non-overlapping: signal at bar i -> entry open bar i+1 -> exit close bar i+hold.
    Next signal at bar i+hold (step = hold, no gap, no overlap).

    Returns daily pd.Series (compounded hourly returns within each calendar day).
    """
    n, n_tick = hc_np.shape

    # --- Build position matrix H[bar, ticker] = weight if held ---
    H          = np.zeros((n, n_tick), dtype=np.float32)
    entry_mask = np.zeros(n, dtype=bool)
    exit_mask  = np.zeros(n, dtype=bool)
    n_trades   = 0

    i = lb
    while i + hold < n:
        row   = mom_np[i]
        valid = np.where(np.isfinite(row))[0]
        if len(valid) >= top_n:
            top_idx = valid[np.argsort(row[valid])[-top_n:]]
            eb = i + 1
            xb = i + hold
            H[eb:xb+1, top_idx] = 1.0 / top_n
            entry_mask[eb] = True
            exit_mask[xb]  = True
            n_trades += 1
        i += hold

    # --- Bar-level returns ---
    # Entry bar: open to close;  subsequent bars: close to close
    with np.errstate(divide="ignore", invalid="ignore"):
        o2c = np.where((ho_np > 0) & np.isfinite(ho_np) & np.isfinite(hc_np),
                       hc_np / ho_np - 1, 0.0)
        c_prev = np.vstack([hc_np[:1], hc_np[:-1]])
        c2c    = np.where((c_prev > 0) & np.isfinite(c_prev) & np.isfinite(hc_np),
                          hc_np / c_prev - 1, 0.0)

    # Weighted bar return for each bar
    bar_ret = np.where(entry_mask[:, None], o2c, c2c)   # shape [n, n_tick]
    port_h  = (H * bar_ret).sum(axis=1)                 # shape [n]

    # TC deductions
    tc_adj = np.zeros(n)
    tc_adj[entry_mask] -= TC_ONE_WAY
    tc_adj[exit_mask]  -= TC_ONE_WAY
    port_h += tc_adj

    # Zero out bars with no position
    port_h[H.sum(axis=1) == 0] = 0.0

    # --- Convert hourly -> daily (compound within each calendar day) ---
    s     = pd.Series(port_h.astype(float), index=bar_ts)
    daily = s.groupby(s.index.normalize()).apply(lambda g: float((1+g).prod() - 1))
    daily.index = pd.to_datetime(daily.index)

    return daily, n_trades


# ===========================================================================
# STEP 1: Load data
# ===========================================================================
SEP = "=" * 72
print(SEP)
print("STEP 1: Loading data")
print(SEP)

hc_all = pd.read_parquet("data/cache/merged_hourly_close.parquet")
ho_all = pd.read_parquet("data/cache/merged_hourly_open.parquet")
hc_all.index = hc_all.index.floor("h")
ho_all.index = ho_all.index.floor("h")
hc_all = hc_all[~hc_all.index.duplicated(keep="last")]
ho_all = ho_all[~ho_all.index.duplicated(keep="last")]

daily_all = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")

# Same 404-ticker filter as Books D and C
common = sorted(set(hc_all.columns) & set(ho_all.columns) & set(daily_all.columns))
valid  = [c for c in common
          if hc_all[c].isna().mean() < 0.30 and ho_all[c].isna().mean() < 0.30]
hc = hc_all[valid].ffill()
ho = ho_all[valid].ffill()

print(f"Hourly universe : {len(valid)} tickers  "
      f"{hc.index[0].date()} -> {hc.index[-1].date()}")
print(f"Hourly bars     : {len(hc)}")

hc_np  = hc.values.astype(np.float32)
ho_np  = ho.values.astype(np.float32)
bar_ts = hc.index
n, n_tick = hc_np.shape


# ===========================================================================
# STEP 2: Book A (Daily Momentum baseline)
# ===========================================================================
print("\n" + SEP)
print("STEP 2: Book A -- Daily Momentum + 1.25x Lev + UVXY (2019-2026 slice)")
print(SEP)

ret_A_full = run_book_a(daily_all)
# Slice to hourly portfolio window
start_dt = pd.Timestamp(hc.index[0].date())
end_dt   = pd.Timestamp(hc.index[-1].date())
ret_A    = ret_A_full.loc[start_dt:end_dt]

stA = _stats(ret_A)
print(f"  Sharpe : {stA['Sharpe']:.3f}")
print(f"  CAGR   : {stA['CAGR']:.1%}")
print(f"  Total  : {stA['Total_Return']:+.1%}")
print(f"  MaxDD  : {stA['Max_DD']:.1%}")


# ===========================================================================
# STEP 3: Hourly momentum grid search (non-overlapping, 2019-2026)
# ===========================================================================
print("\n" + SEP)
print("STEP 3: Hourly momentum grid search (non-overlapping, long-only)")
print(SEP)

LOOKBACK_GRID = [20, 40, 80, 120, 200, 300, 500, 750, 1000]
HOLD_GRID     = [4, 8, 13, 20, 40, 80, 120, 200]
TOP_N_GRID    = [5, 10, 20]
total_combos  = len(LOOKBACK_GRID) * len(HOLD_GRID) * len(TOP_N_GRID)
print(f"Grid: {len(LOOKBACK_GRID)} lookback x {len(HOLD_GRID)} hold x "
      f"{len(TOP_N_GRID)} top_n = {total_combos} combos")

# Precompute momentum matrices for each lookback (avoid recomputing per combo)
print("Pre-computing momentum matrices...")
mom_cache = {}
for lb in LOOKBACK_GRID:
    mom = np.full((n, n_tick), np.nan, dtype=np.float32)
    for i in range(lb, n):
        prev = hc_np[i - lb]
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.where(prev > 0, hc_np[i] / prev - 1, np.nan)
        mom[i] = m
    mom_cache[lb] = mom
    print(f"  lb={lb:>3}h done")

results     = []
best_sh     = -np.inf
best_ret    = None
best_params = None
combo_n     = 0

for lb, hold, top_n in product(LOOKBACK_GRID, HOLD_GRID, TOP_N_GRID):
    combo_n += 1
    daily, n_trades = run_hourly_mom_combo(
        hc_np, ho_np, bar_ts, mom_cache[lb], lb, hold, top_n
    )
    st = _stats(daily)
    results.append(dict(lookback=lb, hold=hold, top_n=top_n,
                        n_trades=n_trades, **st))
    if pd.notna(st["Sharpe"]) and st["Sharpe"] > best_sh:
        best_sh     = st["Sharpe"]
        best_ret    = daily.rename(f"HMom_lb{lb}h_h{hold}h_n{top_n}")
        best_params = dict(lookback=lb, hold=hold, top_n=top_n)
    if combo_n % 15 == 0:
        print(f"  {combo_n}/{total_combos} combos done...")

grid_df = pd.DataFrame(results).sort_values("Sharpe", ascending=False)

print(f"\n{'Lookback':>9} {'Hold':>5} {'TopN':>5} {'Sharpe':>8} "
      f"{'CAGR':>7} {'Total':>8} {'MaxDD':>8} {'Trades':>7}")
print("-" * 70)
for _, r in grid_df.head(20).iterrows():
    print(f"{int(r['lookback']):>9} {int(r['hold']):>5} {int(r['top_n']):>5} "
          f"{r['Sharpe']:>8.3f} {r['CAGR']:>7.1%} "
          f"{r['Total_Return']:>+8.1%} {r['Max_DD']:>8.1%} {int(r['n_trades']):>7}")

print(f"\nSensitivity -- Lookback:")
for lb, grp in grid_df.groupby("lookback"):
    print(f"  {lb:>4}h  avg={grp['Sharpe'].mean():.3f}  best={grp['Sharpe'].max():.3f}")

print(f"\nSensitivity -- Hold:")
for hold, grp in grid_df.groupby("hold"):
    print(f"  {hold:>4}h  avg={grp['Sharpe'].mean():.3f}  best={grp['Sharpe'].max():.3f}")

print(f"\nSensitivity -- TopN:")
for top_n, grp in grid_df.groupby("top_n"):
    print(f"  top{top_n:>3}  avg={grp['Sharpe'].mean():.3f}  best={grp['Sharpe'].max():.3f}")

grid_df.to_excel(OUT_DIR / "hourly_momentum_explore_grid.xlsx", index=False)
print(f"[Grid saved: {OUT_DIR / 'hourly_momentum_explore_grid.xlsx'}]")


# ===========================================================================
# STEP 4: Robustness
# ===========================================================================
print("\n" + SEP)
print("STEP 4: Robustness")
print(SEP)
pos = (grid_df["Sharpe"] > 0).sum()
g1  = (grid_df["Sharpe"] > 1).sum()
g2  = (grid_df["Sharpe"] > 2).sum()
print(f"  Positive Sharpe : {pos}/{total_combos}  ({100*pos/total_combos:.0f}%)")
print(f"  Sharpe > 1.0    : {g1}/{total_combos}   ({100*g1/total_combos:.0f}%)")
print(f"  Sharpe > 2.0    : {g2}/{total_combos}   ({100*g2/total_combos:.0f}%)")


# ===========================================================================
# STEP 5: Summary comparison vs Book A
# ===========================================================================
print("\n" + SEP)
print("STEP 5: Summary -- Hourly Momentum (best) vs Book A (Daily Momentum)")
print(SEP)

best_lb   = best_params["lookback"]
best_hold = best_params["hold"]
best_n    = best_params["top_n"]

print(f"\n  Best hourly params: lb={best_lb}h  hold={best_hold}h  top_n={best_n}")
print(f"\n  {'Strategy':<36} {'Sharpe':>7} {'CAGR':>7} {'Total':>9} {'MaxDD':>8}")
print(f"  {'-'*36} {'-'*7} {'-'*7} {'-'*9} {'-'*8}")

# Align both to common dates
idx = ret_A.index.intersection(best_ret.index)
rA_al = ret_A.reindex(idx, fill_value=0.0)
rH_al = best_ret.reindex(idx, fill_value=0.0)

comparisons = [
    (f"Book A  Daily Mom (lb=140d, top5)", rA_al),
    (f"Hourly Mom (lb={best_lb}h, hold={best_hold}h, top{best_n})", rH_al),
]

for name, r in comparisons:
    st = _stats(r)
    print(f"  {name:<36} {st['Sharpe']:>7.3f} {st['CAGR']:>7.1%} "
          f"{st['Total_Return']:>+9.1%} {st['Max_DD']:>8.1%}")


# ===========================================================================
# STEP 6: Yearly breakdown
# ===========================================================================
print("\n" + SEP)
print("STEP 6: Yearly breakdown")
print(SEP)
print(f"  {'Year':>6} {'HourlyMom':>10} {'BookA':>8}  |  note")
print(f"  {'-'*6} {'-'*10} {'-'*8}")

for yr in sorted(rH_al.index.year.unique()):
    def yr_ret(s): return float((1 + s[s.index.year == yr]).prod() - 1)
    hm = yr_ret(rH_al); ba = yr_ret(rA_al)
    print(f"  {yr:>6} {hm:>+10.2%} {ba:>+8.2%}")


# ===========================================================================
# STEP 7: Charts
# ===========================================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                          gridspec_kw={"height_ratios": [3, 1.5, 1.5]})

# Wealth curve
w_A = (1 + rA_al).cumprod(); w_A /= w_A.iloc[0]
w_H = (1 + rH_al).cumprod(); w_H /= w_H.iloc[0]

axes[0].plot(w_A.index, w_A.values, label="Book A  Daily Mom (1.25x lev)",
             color="gold", lw=2.0)
axes[0].plot(w_H.index, w_H.values,
             label=f"Hourly Mom lb={best_lb}h hold={best_hold}h top{best_n}",
             color="steelblue", lw=2.0)
stH = _stats(rH_al)
axes[0].set_title(
    f"Hourly Momentum (proper, non-overlapping) vs Book A  |  "
    f"H-Mom: Sharpe={stH['Sharpe']:.2f} CAGR={stH['CAGR']:.0%} MaxDD={stH['Max_DD']:.1%}  |  "
    f"BookA: Sharpe={stA['Sharpe']:.2f} CAGR={stA['CAGR']:.0%} MaxDD={stA['Max_DD']:.1%}",
    fontsize=9
)
axes[0].set_ylabel("Cumulative Wealth")
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

# Drawdown
for s, lbl, col in [(rH_al, "Hourly Mom", "steelblue"),
                     (rA_al, "Book A",     "gold")]:
    dd = (1+s).cumprod(); dd = dd/dd.cummax()-1
    axes[1].fill_between(dd.index, dd.values, 0, alpha=0.35, color=col, label=lbl)
axes[1].set_ylabel("Drawdown"); axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

# Yearly bar chart
years_list = sorted(rH_al.index.year.unique())
hm_yr = [float((1+rH_al[rH_al.index.year==y]).prod()-1) for y in years_list]
ba_yr = [float((1+rA_al[rA_al.index.year==y]).prod()-1) for y in years_list]
x = np.arange(len(years_list)); bw = 0.35
axes[2].bar(x-bw/2, [r*100 for r in hm_yr], bw, label="Hourly Mom", color="steelblue", alpha=0.8)
axes[2].bar(x+bw/2, [r*100 for r in ba_yr], bw, label="Book A",     color="gold",      alpha=0.7)
axes[2].axhline(0, color="black", lw=0.8)
axes[2].set_xticks(x); axes[2].set_xticklabels([str(y) for y in years_list])
axes[2].set_ylabel("Return (%)"); axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out = OUT_DIR / "hourly_momentum_explore.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
plt.close()
print(f"\n[Chart saved: {out}]")
print("\nDONE")
