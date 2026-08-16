"""
Z-SCORE ABNORMAL RETURN STRATEGY — GRID SEARCH
================================================
Logic:
  1. For each stock, compute rolling mean and std of daily returns
  2. Z-score = (today's return - rolling_mean) / rolling_std
  3. If Z-score is abnormally HIGH (stock had unusually large positive return):
       - LONG:  buy next day (momentum/continuation)
       - SHORT: sell next day (mean reversion back to normal)
  4. Hold for hold_days, equal-weight across top-N stocks

Grid:
  Lookback:      [10, 20, 40, 60, 120] days
  Z threshold:   [1.0, 1.5, 2.0, 2.5, 3.0]
  Hold days:     [1, 2, 3, 5, 10]
  Top-N stocks:  [5, 10, 20]
  Direction:     [LONG, SHORT]

Total = 5 x 5 x 5 x 3 x 2 = 750 combinations
Data:  S&P500 + NASDAQ100, 1997-2026 (29.4 years)
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

# ── performance ──────────────────────────────────────────────
def perf(ret, tpy=252, rf=0.02):
    if len(ret) < 20: return dict(sharpe=np.nan, sortino=np.nan, max_dd=np.nan, ann_ret=np.nan)
    rf_d = rf / tpy
    exc  = ret - rf_d
    std  = ret.std()
    sh   = exc.mean() / std * np.sqrt(tpy) if std > 0 else 0
    dn   = ret[ret < 0].std(ddof=0)
    so   = exc.mean() / dn  * np.sqrt(tpy) if dn  > 0 else 0
    w    = (1 + ret).cumprod()
    dd   = (w / w.cummax() - 1).min()
    tr   = w.iloc[-1] - 1
    ar   = (1 + tr) ** (1 / (len(ret)/tpy)) - 1 if tr > -1 else -1
    return dict(sharpe=sh, sortino=so, max_dd=dd, ann_ret=ar, total_ret=tr)

# ── data ─────────────────────────────────────────────────────
print("Loading data...")
close_data = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
ret_df     = close_data.pct_change().ffill().fillna(0)

years  = (close_data.index[-1] - close_data.index[0]).days / 365.25
T, N   = ret_df.shape
ret_np = ret_df.values.astype(np.float32)

print(f"Data: {T} days x {N} stocks  |  {close_data.index[0].date()} to {close_data.index[-1].date()}")
print(f"Years: {years:.1f}")

# ── grids ────────────────────────────────────────────────────
lookback_list  = [10, 20, 40, 60, 120]
z_thresh_list  = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
hold_list      = [1, 2, 3, 5, 10]
topn_list      = [5, 10, 20]
direction_list = ["LONG", "SHORT"]

total = (len(lookback_list)*len(z_thresh_list)*
         len(hold_list)*len(topn_list)*len(direction_list))
print(f"\nGrid: {total} combinations")
print(f"  Lookback (days):  {lookback_list}")
print(f"  Z threshold:      {z_thresh_list}")
print(f"  Hold days:        {hold_list}")
print(f"  Top-N:            {topn_list}")
print(f"  Direction:        {direction_list}")

# ── PRE-COMPUTE: Z-score matrices for each lookback ──────────
print("\nPre-computing Z-score matrices...")
z_cache = {}
for lb in lookback_list:
    mu    = pd.DataFrame(ret_np).T.rolling(lb).mean().T.values
    sig   = pd.DataFrame(ret_np).T.rolling(lb).std().T.values
    sig   = np.where(sig < 1e-8, 1e-8, sig)
    z     = (ret_np - mu) / sig
    z_cache[lb] = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    pcts = {t: (z_cache[lb] > t).mean()*100 for t in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]}
    print(f"  lb={lb:3d}d  " + "  ".join(f"pct>{t}:{v:.3f}%" for t,v in pcts.items()))

# ── PRE-COMPUTE: position matrix for each (lb, z_thr, hold, topn, dir) ──
# For speed: vectorize position construction
# position[t, s] = 1 if stock s is held long (or -1 short) on day t

print("\nRunning grid search...")
results = []

for lb, zthr, hold, topn, direction in product(
        lookback_list, z_thresh_list, hold_list, topn_list, direction_list):

    warmup = lb + 1
    z      = z_cache[lb]

    # daily portfolio returns
    daily_rets = []

    for t in range(warmup, T):
        # Signal from previous day's z-score
        z_prev = z[t-1]

        # Stocks with |z| > threshold (only abnormally HIGH returns)
        if direction == "LONG":
            # Buy stocks with high positive z (continuation/momentum)
            eligible_mask = z_prev > zthr
        else:
            # Short stocks with high positive z (mean reversion)
            eligible_mask = z_prev > zthr

        n_eligible = eligible_mask.sum()
        if n_eligible == 0:
            daily_rets.append(0.0)
            continue

        # Pick top-N by z-score magnitude
        eligible_idx = np.where(eligible_mask)[0]
        z_eligible   = z_prev[eligible_idx]
        n_pick       = min(topn, len(eligible_idx))
        top_idx      = eligible_idx[np.argpartition(z_eligible, -n_pick)[-n_pick:]]

        # Calculate forward return over hold_days (compound)
        end_t = min(t + hold, T)
        fwd_compound = np.prod(1 + ret_np[t:end_t, :][:, top_idx], axis=0) - 1

        if direction == "LONG":
            port_ret = fwd_compound.mean()
        else:
            port_ret = -fwd_compound.mean()   # short = negative of stock return

        # Transaction cost: 0.1% per trade
        port_ret -= 0.001
        daily_rets.append(port_ret)

    # Only keep non-overlapping signal days (skip during hold)
    # Actually: this approach gives overlapping trades - fix with stride
    # Re-run with non-overlapping:
    daily_rets_nonoverlap = []
    dates = []
    t = warmup
    while t < T:
        z_prev = z[t-1]
        eligible_mask = z_prev > zthr
        n_eligible = eligible_mask.sum()
        if n_eligible == 0:
            t += 1
            continue
        eligible_idx = np.where(eligible_mask)[0]
        z_eligible   = z_prev[eligible_idx]
        n_pick       = min(topn, len(eligible_idx))
        top_idx      = eligible_idx[np.argpartition(z_eligible, -n_pick)[-n_pick:]]

        end_t = min(t + hold, T)
        fwd   = np.prod(1 + ret_np[t:end_t, :][:, top_idx], axis=0) - 1
        ret_t = (fwd.mean() if direction=="LONG" else -fwd.mean()) - 0.001

        daily_rets_nonoverlap.append(ret_t)
        dates.append(close_data.index[t])
        t += hold   # non-overlapping: jump forward

    if len(daily_rets_nonoverlap) < 10:
        continue

    s      = pd.Series(daily_rets_nonoverlap, index=pd.DatetimeIndex(dates))
    w      = (1 + s).cumprod()
    tr     = w.iloc[-1] - 1
    ar     = (1 + tr) ** (1 / years) - 1 if tr > -1 else -1
    actual_tpy = len(s) / years
    rfp    = 0.02 / actual_tpy
    std    = s.std()
    sh     = (s-rfp).mean() / std * np.sqrt(actual_tpy) if std > 0 else 0
    dn     = s[s<0].std(ddof=0)
    so     = (s-rfp).mean() / dn  * np.sqrt(actual_tpy) if dn  > 0 else 0
    dd     = (w / w.cummax() - 1).min()

    if not np.isfinite(sh): continue

    results.append(dict(
        Lookback=lb, Z_Thresh=zthr, Hold=hold, TopN=topn, Direction=direction,
        Ann_Ret=ar, Sharpe=sh, Sortino=so, MaxDD=dd, Trades=len(s), Total_Ret=tr
    ))

df = pd.DataFrame(results).sort_values("Sharpe", ascending=False).reset_index(drop=True)
print(f"\nGrid search done in {time.time()-t0:.1f}s  ({len(df)} valid combos)")

# ── RESULTS ──────────────────────────────────────────────────
print(f"\n{'='*130}")
print("TOP 30 COMBINATIONS  (sorted by Sharpe)")
print(f"{'='*130}")
fmt = {"Ann_Ret":"{:.2%}".format,"Sharpe":"{:.4f}".format,"Sortino":"{:.4f}".format,
       "MaxDD":"{:.2%}".format,"Total_Ret":"{:.2%}".format}
print(df.head(30).to_string(index=False, formatters=fmt))

print(f"\nPositive Sharpe:  {(df.Sharpe>0).sum()}/{len(df)}")
print(f"Sharpe > 1:       {(df.Sharpe>1).sum()}/{len(df)}")
print(f"Sharpe > 2:       {(df.Sharpe>2).sum()}/{len(df)}")

# ── SENSITIVITY ──────────────────────────────────────────────
print(f"\n{'='*130}")
print("PARAMETER SENSITIVITY")
print(f"{'='*130}")
for col, label in [("Direction","Direction"), ("Z_Thresh","Z Threshold"),
                   ("Lookback","Lookback (d)"), ("Hold","Hold Days"), ("TopN","Top-N")]:
    g = df.groupby(col)["Sharpe"].agg(["mean","max","count"]).reset_index()
    g.columns = [label,"Avg Sharpe","Best Sharpe","N"]
    print(f"\n{g.to_string(index=False)}")

# ── BEST COMBO ───────────────────────────────────────────────
best = df.iloc[0]
print(f"\n{'='*130}")
print("BEST PARAMETERS")
print(f"{'='*130}")
print(f"  Direction:    {best.Direction}")
print(f"  Lookback:     {int(best.Lookback)} days")
print(f"  Z Threshold:  {best.Z_Thresh}")
print(f"  Hold Period:  {int(best.Hold)} days")
print(f"  Top-N:        {int(best.TopN)}")
print(f"\n  Annual Return: {best.Ann_Ret:.2%}")
print(f"  Sharpe Ratio:  {best.Sharpe:.4f}")
print(f"  Sortino Ratio: {best.Sortino:.4f}")
print(f"  Max Drawdown:  {best.MaxDD:.2%}")
print(f"  Total Trades:  {int(best.Trades)}")

# ── REBUILD BEST FOR YEARLY ───────────────────────────────────
lb, zthr, hold, topn, direction = (int(best.Lookback), best.Z_Thresh,
                                    int(best.Hold), int(best.TopN), best.Direction)
z = z_cache[lb]
warmup = lb + 1

rets_best, dates_best = [], []
t = warmup
while t < T:
    z_prev = z[t-1]
    mask   = z_prev > zthr
    if mask.sum() == 0: t += 1; continue
    eidx   = np.where(mask)[0]
    n_pick = min(topn, len(eidx))
    tidx   = eidx[np.argpartition(z_prev[eidx], -n_pick)[-n_pick:]]
    end_t  = min(t + hold, T)
    fwd    = np.prod(1 + ret_np[t:end_t, :][:, tidx], axis=0) - 1
    r      = (fwd.mean() if direction=="LONG" else -fwd.mean()) - 0.001
    rets_best.append(r); dates_best.append(close_data.index[t])
    t += hold

s_best = pd.Series(rets_best, index=pd.DatetimeIndex(dates_best))
w_best = (1 + s_best).cumprod()
actual_tpy_best = len(s_best) / years

print(f"\n{'='*130}")
print("YEARLY BREAKDOWN  (Best Parameters)")
print(f"{'='*130}")
print(f"  {'Year':<6}  {'Return':>9}  {'Sharpe':>8}  {'Sortino':>9}  {'MaxDD':>9}  {'Trades':>7}")
print(f"  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*7}")

for yr in sorted(s_best.index.year.unique()):
    ys = s_best[s_best.index.year == yr]
    yw = w_best[w_best.index.year == yr]
    if len(ys) == 0: continue
    ytr = (1+ys).prod()-1
    ytpy = len(ys)
    rfp  = 0.02 / max(ytpy, 1)
    ysh  = (ys-rfp).mean()/ys.std()*np.sqrt(ytpy) if ys.std()>0 else 0
    ydn  = ys[ys<0].std(ddof=0)
    yso  = (ys-rfp).mean()/ydn*np.sqrt(ytpy) if ydn>0 else 0
    ydd  = (yw/yw.cummax()-1).min()
    print(f"  {yr:<6}  {ytr:>9.2%}  {ysh:>8.3f}  {yso:>9.3f}  {ydd:>9.2%}  {len(ys):>7}")

# ── PLOTS ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Z-Score Abnormal Return Strategy — Grid Search (S&P500+NASDAQ100, 1997-2026)",
             fontsize=13, fontweight="bold")

# 1. Wealth curve
axes[0,0].plot(w_best.index, w_best.values, lw=2, color="steelblue", label=f"Strategy ({best.Ann_Ret:.1%}/yr)")
axes[0,0].set_yscale("log")
axes[0,0].set_title(f"Best: {direction}, LB={lb}d, Z>{zthr}, Hold={hold}d, Top{topn}", fontweight="bold")
axes[0,0].set_ylabel("Wealth (log)"); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

# 2. Sharpe by Direction
s_dir = df.groupby("Direction")["Sharpe"].agg(["mean","max"]).reset_index()
axes[0,1].bar(s_dir["Direction"], s_dir["mean"], alpha=0.7, label="Avg", color="steelblue")
axes[0,1].bar(s_dir["Direction"], s_dir["max"],  alpha=0.4, label="Best", color="green")
axes[0,1].set_title("Sharpe by Direction", fontweight="bold")
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3, axis="y")

# 3. Sharpe by Z threshold
s_z = df.groupby("Z_Thresh")["Sharpe"].agg(["mean","max"]).reset_index()
axes[0,2].plot(s_z["Z_Thresh"], s_z["mean"], marker="o", lw=2, label="Avg", color="steelblue")
axes[0,2].plot(s_z["Z_Thresh"], s_z["max"],  marker="s", lw=2, ls="--", label="Best", color="green")
axes[0,2].axhline(0, color="black", lw=0.8)
axes[0,2].set_title("Sharpe by Z-Score Threshold", fontweight="bold")
axes[0,2].set_xlabel("Z Threshold"); axes[0,2].legend(); axes[0,2].grid(True, alpha=0.3)

# 4. Sharpe by Hold Days
s_h = df.groupby("Hold")["Sharpe"].agg(["mean","max"]).reset_index()
axes[1,0].plot(s_h["Hold"], s_h["mean"], marker="o", lw=2, label="Avg", color="darkorange")
axes[1,0].plot(s_h["Hold"], s_h["max"],  marker="s", lw=2, ls="--", label="Best", color="green")
axes[1,0].axhline(0, color="black", lw=0.8)
axes[1,0].set_title("Sharpe by Hold Days", fontweight="bold")
axes[1,0].set_xlabel("Hold Days"); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

# 5. Sharpe by Lookback
s_lb = df.groupby("Lookback")["Sharpe"].agg(["mean","max"]).reset_index()
axes[1,1].plot(s_lb["Lookback"], s_lb["mean"], marker="o", lw=2, label="Avg", color="purple")
axes[1,1].plot(s_lb["Lookback"], s_lb["max"],  marker="s", lw=2, ls="--", label="Best", color="green")
axes[1,1].axhline(0, color="black", lw=0.8)
axes[1,1].set_title("Sharpe by Lookback Period", fontweight="bold")
axes[1,1].set_xlabel("Lookback (days)"); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

# 6. Sharpe by Top-N
s_n = df.groupby("TopN")["Sharpe"].agg(["mean","max"]).reset_index()
axes[1,2].bar(s_n["TopN"].astype(str), s_n["mean"], alpha=0.7, label="Avg", color="steelblue")
axes[1,2].bar(s_n["TopN"].astype(str), s_n["max"],  alpha=0.4, label="Best", color="green")
axes[1,2].set_title("Sharpe by Top-N Stocks", fontweight="bold")
axes[1,2].set_xlabel("Top-N"); axes[1,2].legend(); axes[1,2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/zscore_strategy_gridsearch.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: results/zscore_strategy_gridsearch.png")

df.to_csv("results/zscore_strategy_grid_results.csv", index=False)
print(f"Saved: results/zscore_strategy_grid_results.csv")
print(f"\nTotal runtime: {time.time()-t0:.1f}s")
print("DONE")
