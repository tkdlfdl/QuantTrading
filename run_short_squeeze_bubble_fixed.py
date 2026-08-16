"""
SHORT SQUEEZE + BUBBLE SCORE — FIXED POSITION SIZING
=====================================================
Bug fixed: previous version treated each trade signal as 100% of capital,
but ~24 trades overlap at any time → artificial 24x leverage → MaxDD -99%.

Fix: build a DAILY portfolio return series where each day's return is the
equal-weight average of ALL positions currently open that day.
This correctly reflects 1/N sizing with N = concurrent open positions.

Universe  : High short-interest stocks (results/short_interest_universe.csv)
Signal    : Per-stock bubble score < threshold (oversold + heavily shorted)
Entry     : LONG at open of bar t+1  (strictly no look-ahead)
Exit      : close of bar t + hold_hours

Grid:
  universe_size : 20, 40, 60, 80
  bubble_ma_h   : 52, 104, 156, 208 hours
  threshold     : -0.8, -0.6, -0.4, -0.2
  hold_h        : 4, 8, 13, 26, 52 hours
  top_n         : 3, 5, 10
Total: 4×4×4×5×3 = 960 combos
"""
import warnings, os, sys, time
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product

t0 = time.time()
os.makedirs("results", exist_ok=True)
RF = 0.02

# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────
print("Loading data...")
si_df    = pd.read_csv("results/short_interest_universe.csv") \
             .sort_values("shortPercentOfFloat", ascending=False)

hourly_c = pd.read_parquet("data/cache/merged_hourly_close.parquet")
hourly_o = pd.read_parquet("data/cache/merged_hourly_open.parquet")
hourly_c.index = hourly_c.index.floor("h")
hourly_o.index = hourly_o.index.floor("h")
hourly_c = hourly_c[~hourly_c.index.duplicated(keep="last")]
hourly_o = hourly_o[~hourly_o.index.duplicated(keep="last")]
idx      = hourly_c.index.intersection(hourly_o.index)
hourly_c = hourly_c.loc[idx]; hourly_o = hourly_o.loc[idx]

T     = len(idx)
years = (idx[-1] - idx[0]).days / 365.25
dates = idx.normalize().unique()          # unique trading DAYS

# Universe
max_univ = 80
si_avail = [t for t in si_df["ticker"] if t in hourly_c.columns]
si_df_av = si_df[si_df["ticker"].isin(si_avail)].reset_index(drop=True)
top80    = si_df_av["ticker"].head(max_univ).tolist()
prices80 = hourly_c[top80].ffill()
opens80  = hourly_o[top80].ffill()

print(f"Hourly: {T} bars  {idx[0].date()} to {idx[-1].date()}  ({years:.2f} yrs)")
print(f"Universe: top-{max_univ} short-interest stocks available in cache")
print(f"Top 20: {top80[:20]}")

# ─────────────────────────────────────────────────────────────
# PRE-COMPUTE BUBBLE SCORES (causal, no look-ahead)
# ─────────────────────────────────────────────────────────────
bubble_ma_list  = [52, 104, 156, 208]

print(f"\nPre-computing bubble scores ({len(bubble_ma_list)} MA windows)...")
bub_cache = {}
for ma_w in bubble_ma_list:
    lp   = np.log(prices80.replace(0, np.nan).ffill())
    fair = prices80.rolling(ma_w, min_periods=ma_w // 2).mean()
    res  = lp - np.log(fair)
    z    = ((res - res.rolling(ma_w, min_periods=ma_w // 2).mean())
            / res.rolling(ma_w, min_periods=ma_w // 2).std())
    # No look-ahead: score at bar t uses data only through bar t
    bub_cache[ma_w] = np.tanh(z / 2).fillna(0).values.astype(np.float32)
    p5 = (bub_cache[ma_w] < -0.5).mean()*100
    p8 = (bub_cache[ma_w] < -0.8).mean()*100
    print(f"  MA={ma_w}h  pct<-0.5:{p5:.2f}%  pct<-0.8:{p8:.2f}%")

# ─────────────────────────────────────────────────────────────
# PRE-COMPUTE PER-STOCK FORWARD RETURNS
# entry = open[t+1], exit = close[min(t+hold_h, T-1)]
# ─────────────────────────────────────────────────────────────
hold_list = [4, 8, 13, 26, 52]

print(f"Pre-computing forward returns ({len(hold_list)} hold periods)...")
prices_np = prices80.values.astype(np.float32)
opens_np  = opens80.values.astype(np.float32)

fwd_cache = {}
for h in hold_list:
    fwd = np.zeros((T, max_univ), dtype=np.float32)
    for t in range(T - h - 1):
        ep = opens_np[t + 1]           # entry open
        xp = prices_np[t + h]          # exit close
        valid = (ep > 0) & (xp > 0) & np.isfinite(ep) & np.isfinite(xp)
        fwd[t, valid] = xp[valid] / ep[valid] - 1
    fwd_cache[h] = np.clip(fwd, -0.50, 5.0)
    print(f"  hold={h}h  avg_fwd={fwd_cache[h].mean()*100:.3f}%")

# ─────────────────────────────────────────────────────────────
# PERFORMANCE (on DAILY portfolio returns)
# ─────────────────────────────────────────────────────────────
def perf_daily(daily_ret: pd.Series, yrs: float) -> dict:
    """All metrics on proper daily return series."""
    if len(daily_ret) < 20:
        return dict(sharpe=np.nan, sortino=np.nan, maxdd=np.nan,
                    ann_ret=np.nan, total_ret=np.nan, win_rate=np.nan)
    rf_d = RF / 252
    exc  = daily_ret - rf_d
    std  = daily_ret.std()
    sh   = exc.mean() / std * np.sqrt(252) if std > 0 else 0
    dn   = daily_ret[daily_ret < 0].std(ddof=0)
    so   = exc.mean() / dn  * np.sqrt(252) if dn  > 0 else 0
    w    = (1 + daily_ret).cumprod()
    dd   = (w / w.cummax() - 1).min()
    tr   = w.iloc[-1] - 1
    ar   = (1 + tr) ** (1 / yrs) - 1 if tr > -1 else -1
    # Win rate: days with positive return (only active days)
    active = daily_ret[daily_ret != 0]
    wr = (active > 0).mean() if len(active) > 0 else 0.5
    return dict(sharpe=sh, sortino=so, maxdd=dd, ann_ret=ar,
                total_ret=tr, win_rate=wr, n_active=len(active))

# ─────────────────────────────────────────────────────────────
# GRID SEARCH — FIXED: build hourly position matrix → daily P&L
# ─────────────────────────────────────────────────────────────
universe_sizes  = [20, 40, 60, 80]
bubble_thr_list = [-0.8, -0.6, -0.4, -0.2]
topn_list       = [3, 5, 10]
TC              = 0.001

total = len(universe_sizes)*len(bubble_ma_list)*len(bubble_thr_list)*len(hold_list)*len(topn_list)
print(f"\nGrid search: {total} combinations (with correct position sizing)...")
t1 = time.time()

results = []

for univ_size, ma_w, thr, hold_h, top_n in product(
        universe_sizes, bubble_ma_list, bubble_thr_list, hold_list, topn_list):

    bub    = bub_cache[ma_w][:, :univ_size]
    fwd    = fwd_cache[hold_h][:, :univ_size]
    warmup = ma_w + 1

    # ── Build hourly position matrix  [T × univ_size]
    # pos[t, s] = return of stock s if held during bar t, else 0
    # (reflects proportional equal-weight: each active stock = 1/n_active share)
    hourly_pos_ret = np.zeros((T, univ_size), dtype=np.float32)
    free_at        = np.zeros(univ_size, dtype=np.int32)

    for t in range(warmup, T - hold_h - 1):
        scores    = bub[t]
        signal    = scores < thr
        available = signal & (free_at <= t)
        if not available.any(): continue

        avail_idx    = np.where(available)[0]
        n_pick       = min(top_n, len(avail_idx))
        chosen_local = np.argpartition(scores[avail_idx], n_pick - 1)[:n_pick]
        chosen       = avail_idx[chosen_local]

        # Record the per-stock per-bar return during the hold window
        entry_open = opens_np[t + 1, :univ_size][chosen]   # open at t+1
        for h_off in range(hold_h):
            bar = t + 1 + h_off
            if bar >= T: break
            bar_close = prices_np[bar, :univ_size][chosen]
            valid = (entry_open > 0) & (bar_close > 0) & np.isfinite(entry_open) & np.isfinite(bar_close)
            if not valid.any(): continue
            hourly_pos_ret[bar, chosen[valid]] = (
                bar_close[valid] / entry_open[valid] - 1
            )

        free_at[chosen] = t + hold_h

    # ── Daily portfolio return = mean of all active positions each day
    # Convert hourly position returns to daily
    bar_dates  = idx.normalize()
    daily_rows = {}
    for d in bar_dates.unique():
        day_mask = (bar_dates == d)
        day_pos  = hourly_pos_ret[day_mask]   # [n_hours_in_day × univ_size]
        # Active = any non-zero position during the day
        active   = day_pos != 0
        n_active = active.any(axis=0).sum()
        if n_active == 0:
            daily_rows[d] = 0.0
            continue
        # Equal-weight: each stock gets 1/n_active of portfolio
        # Stock return for the day = compound its hourly returns
        stock_day_rets = np.zeros(univ_size)
        for s in range(univ_size):
            hrs = day_pos[:, s]
            if (hrs != 0).any():
                stock_day_rets[s] = float(np.prod(1 + hrs) - 1)
        active_rets         = stock_day_rets[active.any(axis=0)]
        daily_rows[d]       = float(active_rets.mean()) - TC / hold_h  # amortize TC

    daily_ret = pd.Series(daily_rows).sort_index()
    daily_ret.index = pd.to_datetime(daily_ret.index)

    p = perf_daily(daily_ret, years)
    if not np.isfinite(p["sharpe"]): continue

    results.append(dict(
        UnivSize=univ_size, MA_h=ma_w, Threshold=thr,
        Hold_h=hold_h, TopN=top_n,
        Sharpe=p["sharpe"], Sortino=p["sortino"],
        Ann_Ret=p["ann_ret"], MaxDD=p["maxdd"],
        Total_Ret=p["total_ret"], Win_Rate=p["win_rate"],
        Active_Days=p["n_active"]
    ))

df = pd.DataFrame(results).sort_values("Sharpe", ascending=False).reset_index(drop=True)
print(f"Done in {time.time()-t1:.1f}s  ({len(df)} valid combos)")

# ─────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*130}")
print("TOP 30 COMBINATIONS (sorted by Sharpe) — FIXED POSITION SIZING")
print(f"{'='*130}")
fmt = {"Sharpe":"{:.4f}".format,"Sortino":"{:.4f}".format,
       "Ann_Ret":"{:.2%}".format,"MaxDD":"{:.2%}".format,
       "Total_Ret":"{:.2%}".format,"Win_Rate":"{:.1%}".format}
print(df.head(30).to_string(index=False, formatters=fmt))

print(f"\nPositive Sharpe: {(df.Sharpe>0).sum()}/{len(df)}")
print(f"Sharpe > 0.5:    {(df.Sharpe>0.5).sum()}/{len(df)}")
print(f"Sharpe > 1.0:    {(df.Sharpe>1.0).sum()}/{len(df)}")

print(f"\n{'='*130}")
print("PARAMETER SENSITIVITY")
print(f"{'='*130}")
for col, label in [("UnivSize","Universe Size"),("MA_h","Bubble MA (h)"),
                   ("Threshold","Threshold"),("Hold_h","Hold (h)"),("TopN","Top-N")]:
    g = df.groupby(col)["Sharpe"].agg(["mean","max","count"]).reset_index()
    g.columns = [label,"Avg Sharpe","Best Sharpe","N"]
    print(f"\n{g.to_string(index=False)}")

# ─────────────────────────────────────────────────────────────
# BEST COMBO — YEARLY BREAKDOWN
# ─────────────────────────────────────────────────────────────
best = df.iloc[0]
print(f"\n{'='*130}")
print("BEST PARAMETERS (FIXED)")
print(f"{'='*130}")
print(f"  Universe:  top-{int(best.UnivSize)} short-interest stocks")
print(f"  Bubble MA: {int(best.MA_h)}h  |  Threshold: {best.Threshold}")
print(f"  Hold:      {int(best.Hold_h)}h  |  Top-N: {int(best.TopN)}")
print(f"\n  Annual Return:  {best.Ann_Ret:.2%}")
print(f"  Sharpe Ratio:   {best.Sharpe:.4f}")
print(f"  Sortino Ratio:  {best.Sortino:.4f}")
print(f"  Max Drawdown:   {best.MaxDD:.2%}")
print(f"  Win Rate:       {best.Win_Rate:.1%}")
print(f"  Active Days:    {int(best.Active_Days)}")

# Rebuild best daily series
univ_b = int(best.UnivSize); ma_b = int(best.MA_h)
thr_b  = best.Threshold; hold_b = int(best.Hold_h); topn_b = int(best.TopN)

bub_b    = bub_cache[ma_b][:, :univ_b]
warmup   = ma_b + 1
free_at  = np.zeros(univ_b, dtype=np.int32)
hourly_pr = np.zeros((T, univ_b), dtype=np.float32)

for t in range(warmup, T - hold_b - 1):
    scores = bub_b[t]; signal = scores < thr_b
    available = signal & (free_at <= t)
    if not available.any(): continue
    avail_idx = np.where(available)[0]
    n_pick    = min(topn_b, len(avail_idx))
    chosen    = avail_idx[np.argpartition(scores[avail_idx], n_pick-1)[:n_pick]]
    ep        = opens_np[t+1, :univ_b][chosen]
    for h_off in range(hold_b):
        bar = t + 1 + h_off
        if bar >= T: break
        xp    = prices_np[bar, :univ_b][chosen]
        valid = (ep > 0) & (xp > 0) & np.isfinite(ep) & np.isfinite(xp)
        if valid.any():
            hourly_pr[bar, chosen[valid]] = xp[valid] / ep[valid] - 1
    free_at[chosen] = t + hold_b

bar_dates  = idx.normalize()
daily_rows = {}
for d in bar_dates.unique():
    day_mask = (bar_dates == d)
    day_pos  = hourly_pr[day_mask]
    active   = day_pos != 0
    n_active = active.any(axis=0).sum()
    if n_active == 0: daily_rows[d] = 0.0; continue
    stock_rets = np.zeros(univ_b)
    for s in range(univ_b):
        hrs = day_pos[:, s]
        if (hrs != 0).any():
            stock_rets[s] = float(np.prod(1 + hrs) - 1)
    daily_rows[d] = float(stock_rets[active.any(axis=0)].mean()) - TC / hold_b

daily_best = pd.Series(daily_rows).sort_index()
daily_best.index = pd.to_datetime(daily_best.index)
w_best = (1 + daily_best).cumprod()

print(f"\n{'='*130}")
print("YEARLY BREAKDOWN")
print(f"{'='*130}")
print(f"  {'Year':<6}  {'Return':>9}  {'Ann Ret':>9}  {'Sharpe':>8}  "
      f"{'Sortino':>9}  {'MaxDD':>9}  {'WinRate':>8}  {'Active Days':>12}")
print(f"  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*12}")

yearly_rows = []
for yr in sorted(daily_best.index.year.unique()):
    ys = daily_best[daily_best.index.year == yr]
    yw = w_best[w_best.index.year == yr]
    if len(ys) == 0: continue
    ytr = (1 + ys).prod() - 1
    yar = (1 + ytr) ** (252 / max(len(ys), 1)) - 1 if ytr > -1 else -1
    ysh = ys.mean() / ys.std() * np.sqrt(252) if ys.std() > 0 else 0
    ydn = ys[ys < 0].std(ddof=0)
    yso = ys.mean() / ydn * np.sqrt(252) if ydn > 0 else 0
    ydd = (yw / yw.cummax() - 1).min()
    act = (ys[ys != 0] > 0)
    ywr = act.mean() if len(act) > 0 else 0.5
    n_act = int((ys != 0).sum())
    print(f"  {yr:<6}  {ytr:>9.2%}  {yar:>9.2%}  {ysh:>8.3f}  "
          f"{yso:>9.3f}  {ydd:>9.2%}  {ywr:>8.1%}  {n_act:>12}")
    yearly_rows.append(dict(Year=yr, Return=ytr, Ann_Ret=yar,
                            Sharpe=ysh, Sortino=yso, MaxDD=ydd,
                            Win_Rate=ywr, Active_Days=n_act))

# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(
    f"Short Squeeze + Bubble Score (FIXED — Equal-Weight Daily P&L)\n"
    f"Universe: top-{univ_b} short-int  |  "
    f"MA={ma_b}h, Thr={thr_b}, Hold={hold_b}h, Top-{topn_b}\n"
    f"Ann={best.Ann_Ret:.1%}  Sharpe={best.Sharpe:.3f}  MaxDD={best.MaxDD:.1%}",
    fontsize=11, fontweight="bold")

axes[0,0].plot(w_best.index, w_best.values, lw=2, color="steelblue")
axes[0,0].set_title("Cumulative Wealth (CORRECT)", fontweight="bold")
axes[0,0].set_ylabel("Wealth Multiple"); axes[0,0].grid(True, alpha=0.3)

dd_s = w_best / w_best.cummax() - 1
axes[0,1].fill_between(dd_s.index, dd_s.values, 0, alpha=0.6, color="red")
axes[0,1].set_title(f"Drawdown  (Max={best.MaxDD:.2%})", fontweight="bold")
axes[0,1].grid(True, alpha=0.3)

g = df.groupby("Threshold")["Sharpe"].agg(["mean","max"]).reset_index()
axes[0,2].plot(g["Threshold"], g["mean"], marker="o", lw=2, label="Avg", color="steelblue")
axes[0,2].plot(g["Threshold"], g["max"],  marker="s", lw=2, ls="--", label="Best", color="green")
axes[0,2].axhline(0, color="black", lw=0.8)
axes[0,2].set_title("Sharpe vs Bubble Threshold", fontweight="bold")
axes[0,2].set_xlabel("Threshold"); axes[0,2].legend(); axes[0,2].grid(True, alpha=0.3)

g = df.groupby("Hold_h")["Sharpe"].agg(["mean","max"]).reset_index()
axes[1,0].plot(g["Hold_h"], g["mean"], marker="o", lw=2, label="Avg", color="darkorange")
axes[1,0].plot(g["Hold_h"], g["max"],  marker="s", lw=2, ls="--", label="Best", color="green")
axes[1,0].axhline(0, color="black", lw=0.8)
axes[1,0].set_title("Sharpe vs Hold Period (hours)", fontweight="bold")
axes[1,0].set_xlabel("Hold (h)"); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

g = df.groupby("UnivSize")["Sharpe"].agg(["mean","max"]).reset_index()
axes[1,1].plot(g["UnivSize"], g["mean"], marker="o", lw=2, label="Avg", color="purple")
axes[1,1].plot(g["UnivSize"], g["max"],  marker="s", lw=2, ls="--", label="Best", color="green")
axes[1,1].axhline(0, color="black", lw=0.8)
axes[1,1].set_title("Sharpe vs Universe Size", fontweight="bold")
axes[1,1].set_xlabel("# Stocks"); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

if yearly_rows:
    ydf  = pd.DataFrame(yearly_rows)
    clrs = ["green" if r >= 0 else "red" for r in ydf["Return"]]
    axes[1,2].bar(ydf["Year"].astype(str), ydf["Return"]*100, color=clrs, alpha=0.75)
    axes[1,2].axhline(0, color="black", lw=0.8)
    axes[1,2].set_title("Yearly Returns (%)", fontweight="bold")
    axes[1,2].set_ylabel("Return (%)"); axes[1,2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("results/short_squeeze_bubble_fixed.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: results/short_squeeze_bubble_fixed.png")

df.to_csv("results/short_squeeze_bubble_fixed_grid.csv", index=False)
pd.DataFrame(yearly_rows).to_csv("results/short_squeeze_bubble_fixed_yearly.csv", index=False)
print("Saved: short_squeeze_bubble_fixed_grid.csv  |  short_squeeze_bubble_fixed_yearly.csv")
print(f"\nTotal runtime: {time.time()-t0:.1f}s  |  DONE")
