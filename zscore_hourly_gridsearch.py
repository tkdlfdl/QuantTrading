"""
Z-SCORE ABNORMAL RETURN STRATEGY — HOURLY DATA GRID SEARCH
============================================================
Logic:  Compute Z-score on hourly returns per stock.
        If Z > threshold (abnormally large positive return this hour):
          LONG: buy next hour and hold (momentum continuation)
          SHORT: sell next hour and hold (mean reversion)
        Non-overlapping trades, equal-weight across top-N stocks.

Grid (hourly):
  Lookback  (h): [6, 13, 26, 52, 104, 156]   (~1d, 2d, 4d, 8d, 2wk, 3wk)
  Z Threshold:   [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
  Hold period(h):[1, 2, 4, 6, 8, 13, 26]     (~1h to 4d)
  Top-N:         [5, 10, 20]
  Direction:     [LONG, SHORT]

Total = 6 x 7 x 7 x 3 x 2 = 1,764 combinations
Data:  S&P500 + NASDAQ100 hourly, 2020-2026 (5.85 years)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, os, time
warnings.filterwarnings("ignore")
from itertools import product

t0 = time.time()

# ── performance (trade-level) ─────────────────────────────────
def perf(s, years, rf=0.02):
    if len(s) < 5:
        return dict(sharpe=np.nan, sortino=np.nan, max_dd=np.nan, ann_ret=np.nan, total_ret=np.nan)
    tpy  = len(s) / years          # actual trades per year
    rfp  = rf / tpy
    exc  = s - rfp
    std  = s.std()
    sh   = exc.mean() / std * np.sqrt(tpy) if std > 0 else 0.0
    dn   = s[s < 0].std(ddof=0)
    so   = exc.mean() / dn  * np.sqrt(tpy) if dn  > 0 else 0.0
    w    = (1 + s).cumprod()
    dd   = (w / w.cummax() - 1).min()
    tr   = w.iloc[-1] - 1
    ar   = (1 + tr) ** (1 / years) - 1 if tr > -1 else -1.0
    return dict(sharpe=sh, sortino=so, max_dd=dd, ann_ret=ar, total_ret=tr)

# ── data ─────────────────────────────────────────────────────
print("Loading data...")
qqq_raw = pd.read_parquet("data/cache/qqq_hourly_close.parquet")["QQQ"]
stk_raw = pd.read_parquet("data/cache/merged_hourly_close.parquet")

qqq_h = qqq_raw.copy(); qqq_h.index = qqq_h.index.floor("h")
stk_h = stk_raw.copy(); stk_h.index = stk_h.index.floor("h")
qqq_h = qqq_h[~qqq_h.index.duplicated(keep="last")]
stk_h = stk_h[~stk_h.index.duplicated(keep="last")]
idx   = qqq_h.index.intersection(stk_h.index)
stocks = stk_h.loc[idx]

daily_cols = set(pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet").columns)
sp_cols    = [c for c in stocks.columns if c in daily_cols]
valid      = stocks[sp_cols].columns[stocks[sp_cols].isna().mean() < 0.30]
stocks     = stocks[valid].ffill()

ret_np  = stocks.pct_change().clip(-0.10, 0.10).fillna(0).values.astype(np.float32)
log_ret = np.log1p(ret_np.clip(-0.10, 0.10))
cumlog  = np.cumsum(log_ret, axis=0)
T, N    = ret_np.shape
years   = (idx[-1] - idx[0]).days / 365.25

print(f"Data: {T} hourly bars x {N} stocks  |  {idx[0].date()} to {idx[-1].date()}  ({years:.2f} yrs)")

# ── grids ────────────────────────────────────────────────────
lookback_list  = [6, 13, 26, 52, 104, 156]
z_thresh_list  = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
hold_list      = [1, 2, 4, 6, 8, 13, 26]
topn_list      = [5, 10, 20]
direction_list = ["LONG", "SHORT"]

total = (len(lookback_list) * len(z_thresh_list) *
         len(hold_list)     * len(topn_list)     * len(direction_list))
print(f"\nGrid: {total} combinations")
print(f"  Lookback  (h): {lookback_list}")
print(f"  Z Threshold:   {z_thresh_list}")
print(f"  Hold     (h):  {hold_list}")
print(f"  Top-N:         {topn_list}")
print(f"  Direction:     {direction_list}")

# ── PRE-COMPUTE Z-score arrays ────────────────────────────────
print("\nPre-computing Z-score matrices...")
z_cache = {}
for lb in lookback_list:
    ret_df = pd.DataFrame(ret_np)
    mu     = ret_df.T.rolling(lb, min_periods=max(lb//2,2)).mean().T.values
    sig    = ret_df.T.rolling(lb, min_periods=max(lb//2,2)).std().T.values
    sig    = np.where(sig < 1e-8, 1e-8, sig)
    z      = np.nan_to_num((ret_np - mu) / sig, nan=0.0, posinf=0.0, neginf=0.0)
    z_cache[lb] = z.astype(np.float32)
    pcts   = [(z > thr).mean()*100 for thr in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]]
    print(f"  lb={lb:3d}h  " + "  ".join(f"Z>{t}:{v:.3f}%" for t,v in
          zip([1.5,2.0,2.5,3.0,4.0,5.0], pcts)))

# ── PRE-COMPUTE compound forward returns ─────────────────────
print("Pre-computing compound forward returns...")
fwd_cache = {}
for h in hold_list:
    fwd = np.zeros((T, N), dtype=np.float32)
    fwd[:T-h] = np.expm1(cumlog[h:] - cumlog[:T-h])
    fwd_cache[h] = fwd
    print(f"  hold={h:2d}h  avg_fwd={fwd[:T-h].mean()*100:.4f}%")

# ── GRID SEARCH ──────────────────────────────────────────────
print(f"\nRunning grid search ({total} combos)...")
t1 = time.time()
results = []

for lb, zthr, hold, topn, direction in product(
        lookback_list, z_thresh_list, hold_list, topn_list, direction_list):

    z   = z_cache[lb]
    fwd = fwd_cache[hold]
    warmup = lb + 1

    trade_rets, trade_dates = [], []
    i = warmup
    while i < T - hold:
        z_prev = z[i-1]
        mask   = z_prev > zthr
        n_sig  = mask.sum()
        if n_sig == 0:
            i += 1
            continue

        # Top-N by Z-score among those above threshold
        sig_idx  = np.where(mask)[0]
        n_pick   = min(topn, len(sig_idx))
        top_idx  = sig_idx[np.argpartition(z_prev[sig_idx], -n_pick)[-n_pick:]]

        # Compound return over hold hours
        r_raw = fwd[i, top_idx].mean()
        r = (float(r_raw) if direction == "LONG" else -float(r_raw)) - 0.001
        r = max(-0.50, min(2.00, r))

        trade_rets.append(r)
        trade_dates.append(idx[i])
        i += hold   # non-overlapping

    if len(trade_rets) < 5:
        continue

    s  = pd.Series(trade_rets, index=pd.DatetimeIndex(trade_dates))
    p  = perf(s, years)
    if not np.isfinite(p['sharpe']):
        continue

    results.append(dict(
        Lookback=lb, Z_Thresh=zthr, Hold=hold, TopN=topn, Direction=direction,
        Ann_Ret=p['ann_ret'], Sharpe=p['sharpe'], Sortino=p['sortino'],
        MaxDD=p['max_dd'], Trades=len(s), Total_Ret=p['total_ret']
    ))

df = pd.DataFrame(results).sort_values("Sharpe", ascending=False).reset_index(drop=True)
print(f"Done in {time.time()-t1:.1f}s  ({len(df)} valid combos)")

# ── RESULTS ──────────────────────────────────────────────────
print(f"\n{'='*130}")
print("TOP 30 COMBINATIONS  (sorted by Sharpe)")
print(f"{'='*130}")
fmt = {"Ann_Ret":"{:.2%}".format,"Sharpe":"{:.4f}".format,"Sortino":"{:.4f}".format,
       "MaxDD":"{:.2%}".format,"Total_Ret":"{:.2%}".format}
print(df.head(30).to_string(index=False, formatters=fmt))

print(f"\nPositive Sharpe:  {(df.Sharpe>0).sum()}/{len(df)}")
print(f"Sharpe > 0.5:     {(df.Sharpe>0.5).sum()}/{len(df)}")
print(f"Sharpe > 1.0:     {(df.Sharpe>1.0).sum()}/{len(df)}")
print(f"Sharpe > 2.0:     {(df.Sharpe>2.0).sum()}/{len(df)}")

# ── SENSITIVITY ──────────────────────────────────────────────
print(f"\n{'='*130}")
print("PARAMETER SENSITIVITY")
print(f"{'='*130}")
for col, label in [("Direction","Direction"), ("Z_Thresh","Z Threshold"),
                   ("Lookback","Lookback (h)"), ("Hold","Hold (h)"), ("TopN","Top-N")]:
    g = df.groupby(col)["Sharpe"].agg(["mean","max","count"]).reset_index()
    g.columns = [label,"Avg Sharpe","Best Sharpe","N"]
    print(f"\n{g.to_string(index=False)}")

# ── BEST COMBO ───────────────────────────────────────────────
best = df.iloc[0]
print(f"\n{'='*130}")
print("BEST PARAMETERS")
print(f"{'='*130}")
print(f"  Direction:    {best.Direction}")
print(f"  Lookback:     {int(best.Lookback)}h  (~{best.Lookback/6.5:.1f} sessions)")
print(f"  Z Threshold:  {best.Z_Thresh}")
print(f"  Hold Period:  {int(best.Hold)}h  (~{best.Hold/6.5:.1f} sessions)")
print(f"  Top-N:        {int(best.TopN)}")
print(f"\n  Annual Return: {best.Ann_Ret:.2%}")
print(f"  Sharpe Ratio:  {best.Sharpe:.4f}")
print(f"  Sortino Ratio: {best.Sortino:.4f}")
print(f"  Max Drawdown:  {best.MaxDD:.2%}")
print(f"  Total Trades:  {int(best.Trades)}")

# ── REBUILD BEST — YEARLY ─────────────────────────────────────
lb_b, zthr_b = int(best.Lookback), best.Z_Thresh
hold_b, topn_b, dir_b = int(best.Hold), int(best.TopN), best.Direction
z_b   = z_cache[lb_b]; fwd_b = fwd_cache[hold_b]
warmup = lb_b + 1

best_rets, best_dates = [], []
i = warmup
while i < T - hold_b:
    z_prev = z_b[i-1]
    mask   = z_prev > zthr_b
    if mask.sum() == 0: i += 1; continue
    sig_idx  = np.where(mask)[0]
    n_pick   = min(topn_b, len(sig_idx))
    top_idx  = sig_idx[np.argpartition(z_prev[sig_idx], -n_pick)[-n_pick:]]
    r_raw    = fwd_b[i, top_idx].mean()
    r = (float(r_raw) if dir_b=="LONG" else -float(r_raw)) - 0.001
    best_rets.append(max(-0.50, min(2.00, r)))
    best_dates.append(idx[i]); i += hold_b

s_best = pd.Series(best_rets, index=pd.DatetimeIndex(best_dates))
w_best = (1 + s_best).cumprod()

# Hourly wealth for correct MaxDD
h_rets = np.zeros(T)
i = warmup
while i < T - hold_b:
    z_prev = z_b[i-1]
    mask   = z_prev > zthr_b
    if mask.sum() == 0: i += 1; continue
    sig_idx  = np.where(mask)[0]
    n_pick   = min(topn_b, len(sig_idx))
    top_idx  = sig_idx[np.argpartition(z_prev[sig_idx], -n_pick)[-n_pick:]]
    for j in range(i, min(i+hold_b, T)):
        h_rets[j] = ret_np[j, top_idx].mean() * (1 if dir_b=="LONG" else -1)
    i += hold_b

h_wealth = pd.Series((1+h_rets).cumprod(), index=idx)
hourly_dd = (h_wealth / h_wealth.cummax() - 1).min()

print(f"\n  Max Drawdown (hourly wealth): {hourly_dd:.2%}")

print(f"\n{'='*130}")
print("YEARLY BREAKDOWN  (Best Parameters)")
print(f"{'='*130}")
print(f"  {'Year':<6}  {'Return':>9}  {'Sharpe':>8}  {'Sortino':>9}  {'MaxDD(hr)':>10}  {'Trades':>7}")
print(f"  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*10}  {'-'*7}")

for yr in sorted(s_best.index.year.unique()):
    ys = s_best[s_best.index.year == yr]
    yh = h_wealth[h_wealth.index.year == yr]
    if len(ys) == 0: continue
    ytr = (1+ys).prod()-1
    ytpy = len(ys)
    rfp  = 0.02 / max(ytpy,1)
    ysh  = (ys-rfp).mean()/ys.std()*np.sqrt(ytpy) if ys.std()>0 else 0
    ydn  = ys[ys<0].std(ddof=0)
    yso  = (ys-rfp).mean()/ydn*np.sqrt(ytpy) if ydn>0 else 0
    ydd  = (yh/yh.cummax()-1).min()
    print(f"  {yr:<6}  {ytr:>9.2%}  {ysh:>8.3f}  {yso:>9.3f}  {ydd:>10.2%}  {len(ys):>7}")

# ── PLOTS ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(f"Z-Score Strategy — Hourly Data (S&P500+NASDAQ100, 2020-2026)\n"
             f"Best: {dir_b}, LB={lb_b}h, Z>{zthr_b}, Hold={hold_b}h, Top{topn_b}",
             fontsize=12, fontweight="bold")

# 1. Wealth curve
axes[0,0].plot(w_best.index, w_best.values, lw=2, color="steelblue",
               label=f"Strategy ({best.Ann_Ret:.1%}/yr, Sharpe={best.Sharpe:.3f})")
axes[0,0].set_yscale("log")
axes[0,0].set_title("Cumulative Wealth (Best Parameters)", fontweight="bold")
axes[0,0].set_ylabel("Wealth (log)"); axes[0,0].legend(fontsize=9); axes[0,0].grid(True, alpha=0.3)

# 2. Sharpe by Direction
s_dir = df.groupby("Direction")["Sharpe"].agg(["mean","max"]).reset_index()
x = np.arange(len(s_dir))
axes[0,1].bar(x-0.2, s_dir["mean"], 0.4, label="Avg", alpha=0.7, color="steelblue")
axes[0,1].bar(x+0.2, s_dir["max"],  0.4, label="Best", alpha=0.7, color="green")
axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(s_dir["Direction"])
axes[0,1].axhline(0, color="black", lw=0.8)
axes[0,1].set_title("Sharpe by Direction", fontweight="bold")
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3, axis="y")

# 3. Sharpe by Z threshold
s_z = df.groupby("Z_Thresh")["Sharpe"].agg(["mean","max"]).reset_index()
axes[0,2].plot(s_z["Z_Thresh"], s_z["mean"], marker="o", lw=2, label="Avg", color="steelblue")
axes[0,2].plot(s_z["Z_Thresh"], s_z["max"],  marker="s", lw=2, ls="--", label="Best", color="green")
axes[0,2].axhline(0, color="black", lw=0.8)
axes[0,2].set_title("Sharpe vs Z Threshold", fontweight="bold")
axes[0,2].set_xlabel("Z Threshold"); axes[0,2].legend(); axes[0,2].grid(True, alpha=0.3)

# 4. Sharpe by Hold Period
s_h = df.groupby("Hold")["Sharpe"].agg(["mean","max"]).reset_index()
axes[1,0].plot(s_h["Hold"], s_h["mean"], marker="o", lw=2, label="Avg", color="darkorange")
axes[1,0].plot(s_h["Hold"], s_h["max"],  marker="s", lw=2, ls="--", label="Best", color="green")
axes[1,0].axhline(0, color="black", lw=0.8)
axes[1,0].set_title("Sharpe vs Hold Period (hours)", fontweight="bold")
axes[1,0].set_xlabel("Hold (hours)"); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

# 5. Sharpe by Lookback
s_lb = df.groupby("Lookback")["Sharpe"].agg(["mean","max"]).reset_index()
axes[1,1].plot(s_lb["Lookback"], s_lb["mean"], marker="o", lw=2, label="Avg", color="purple")
axes[1,1].plot(s_lb["Lookback"], s_lb["max"],  marker="s", lw=2, ls="--", label="Best", color="green")
axes[1,1].axhline(0, color="black", lw=0.8)
axes[1,1].set_title("Sharpe vs Lookback (hours)", fontweight="bold")
axes[1,1].set_xlabel("Lookback (hours)"); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

# 6. Yearly returns
yr_data = []
for yr in sorted(s_best.index.year.unique()):
    ys = s_best[s_best.index.year == yr]
    if len(ys) > 0:
        yr_data.append((yr, (1+ys).prod()-1))
if yr_data:
    yrs, yrets = zip(*yr_data)
    clrs = ["green" if r >= 0 else "red" for r in yrets]
    axes[1,2].bar([str(y) for y in yrs], [r*100 for r in yrets], color=clrs, alpha=0.7)
    axes[1,2].axhline(0, color="black", lw=0.8)
axes[1,2].set_title("Yearly Returns — Best Strategy (%)", fontweight="bold")
axes[1,2].set_ylabel("Return (%)"); axes[1,2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/zscore_hourly_gridsearch.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: results/zscore_hourly_gridsearch.png")

df.to_csv("results/zscore_hourly_grid_results.csv", index=False)
print(f"Saved: results/zscore_hourly_grid_results.csv")
print(f"\nTotal runtime: {time.time()-t0:.1f}s")
print("DONE")
