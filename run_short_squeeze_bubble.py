"""
Short Squeeze + Bubble Score Strategy  [REWRITTEN]
===================================================
Universe  : High short-interest stocks (from results/short_interest_universe.csv,
            fetched via yfinance shortPercentOfFloat — static ranking proxy).

Signal    : Per-stock bubble score < threshold  ->  stock is depressed (squeeze setup)
            No lookahead: bubble score at bar t uses data through bar t only.

Entry     : LONG at open of bar t+1
Exit      : close of bar t + hold_hours

Bubble formula (same as QQQ bubble strategy):
  residual = log(close) - log(rolling_mean(close, ma_window))
  z        = (residual - mean(residual)) / std(residual)  over same window
  bubble   = tanh(z / 2)    <- bounded (-1, +1)

Grid      : universe_size x ma_window x threshold x hold_hours x top_n
Total     : 4 x 4 x 4 x 5 x 3 = 960 combinations
Data      : 2019-01-02 to 2026-06-02 (7.41 years, merged hourly cache)
TC        : 0.1% round-trip per trade
"""
import sys, warnings, os, time
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

# Short interest universe (static ranking — yfinance cached)
si_df    = pd.read_csv("results/short_interest_universe.csv") \
             .sort_values("shortPercentOfFloat", ascending=False)
print(f"SI cache: {len(si_df)} tickers")
print(f"Top 20: {si_df['ticker'].head(20).tolist()}")

# Hourly data from merged cache (no download)
hourly_c = pd.read_parquet("data/cache/merged_hourly_close.parquet")
hourly_o = pd.read_parquet("data/cache/merged_hourly_open.parquet")
hourly_c.index = hourly_c.index.floor("h")
hourly_o.index = hourly_o.index.floor("h")
hourly_c = hourly_c[~hourly_c.index.duplicated(keep="last")]
hourly_o = hourly_o[~hourly_o.index.duplicated(keep="last")]
idx = hourly_c.index.intersection(hourly_o.index)
hourly_c = hourly_c.loc[idx]; hourly_o = hourly_o.loc[idx]
T     = len(idx)
years = (idx[-1] - idx[0]).days / 365.25
print(f"Hourly: {T} bars  {idx[0].date()} to {idx[-1].date()}  ({years:.2f} yrs)")

# ─────────────────────────────────────────────────────────────
# BUILD UNIVERSE SETS
# ─────────────────────────────────────────────────────────────
universe_sizes = [20, 40, 60, 80]
max_univ = 80
si_avail = [t for t in si_df["ticker"] if t in hourly_c.columns]
si_df_av = si_df[si_df["ticker"].isin(si_avail)].reset_index(drop=True)
top80    = si_df_av["ticker"].head(max_univ).tolist()

# Build per-size sets for fast lookup
univ_sets = {n: set(top80[:n]) for n in universe_sizes}

prices80 = hourly_c[top80].ffill()
opens80  = hourly_o[top80].ffill()
print(f"Available SI tickers in cache: {len(si_avail)}, using top {max_univ}")
print(f"Top 20 squeeze universe: {top80[:20]}")

# ─────────────────────────────────────────────────────────────
# BUBBLE SCORE (per-stock, causal)
# ─────────────────────────────────────────────────────────────
bubble_ma_list  = [52, 104, 156, 208]   # hours
bubble_thr_list = [-0.8, -0.6, -0.4, -0.2]
hold_list       = [4, 8, 13, 26, 52]
topn_list       = [3, 5, 10]

print(f"\nPre-computing bubble scores ({len(bubble_ma_list)} MA windows)...")
bub_cache = {}
for ma_w in bubble_ma_list:
    lp   = np.log(prices80.replace(0, np.nan).ffill())
    fair = prices80.rolling(ma_w, min_periods=ma_w // 2).mean()
    res  = lp - np.log(fair)
    z    = ((res - res.rolling(ma_w, min_periods=ma_w // 2).mean())
            / res.rolling(ma_w, min_periods=ma_w // 2).std())
    bub  = np.tanh(z / 2).fillna(0)
    # strict no look-ahead: score at bar t uses data through bar t
    bub_cache[ma_w] = bub.values.astype(np.float32)
    neg_frac = (bub_cache[ma_w] < -0.5).mean()
    print(f"  MA={ma_w}h  pct<-0.5:{neg_frac*100:.2f}%  pct<-0.8:{(bub_cache[ma_w]<-0.8).mean()*100:.2f}%")

# Pre-compute forward returns: entry=open[t+1], exit=close[t+hold]
print(f"Pre-computing forward returns ({len(hold_list)} hold periods)...")
prices_np = prices80.values.astype(np.float32)
opens_np  = opens80.values.astype(np.float32)

fwd_cache = {}
for h in hold_list:
    fwd = np.zeros((T, max_univ), dtype=np.float32)
    ep  = opens_np[1:T]              # open[t+1]
    xp  = prices_np[np.minimum(np.arange(1, T) + h - 1, T - 1)]  # close[t+hold]
    valid = (ep > 0) & (xp > 0) & np.isfinite(ep) & np.isfinite(xp)
    fwd[:T-1] = np.where(valid, xp / ep - 1, 0.0)
    fwd_cache[h] = np.clip(fwd, -0.5, 5.0)
    print(f"  hold={h}h  avg_fwd={fwd_cache[h][:T-1].mean()*100:.3f}%")

ticker_arr = np.array(top80)

# ─────────────────────────────────────────────────────────────
# PERFORMANCE HELPER
# ─────────────────────────────────────────────────────────────
def perf(s, yrs):
    if len(s) < 5: return dict(sharpe=np.nan, sortino=np.nan, maxdd=np.nan,
                                ann_ret=np.nan, total_ret=np.nan, win_rate=np.nan)
    tpy = len(s) / yrs; rfp = RF / tpy; exc = s - rfp
    std = s.std()
    sh  = exc.mean() / std * np.sqrt(tpy) if std > 0 else 0
    dn  = s[s < 0].std(ddof=0)
    so  = exc.mean() / dn  * np.sqrt(tpy) if dn  > 0 else 0
    w   = (1 + s).cumprod()
    dd  = (w / w.cummax() - 1).min()
    tr  = w.iloc[-1] - 1
    ar  = (1 + tr) ** (1 / yrs) - 1 if tr > -1 else -1
    return dict(sharpe=sh, sortino=so, maxdd=dd, ann_ret=ar,
                total_ret=tr, win_rate=(s > 0).mean())

# ─────────────────────────────────────────────────────────────
# GRID SEARCH
# ─────────────────────────────────────────────────────────────
TC    = 0.001
total = len(universe_sizes)*len(bubble_ma_list)*len(bubble_thr_list)*len(hold_list)*len(topn_list)
print(f"\nGrid search: {total} combinations...")
t1 = time.time()

results = []
for univ_size, ma_w, thr, hold_h, top_n in product(
        universe_sizes, bubble_ma_list, bubble_thr_list, hold_list, topn_list):

    bub     = bub_cache[ma_w][:, :univ_size]
    fwd     = fwd_cache[hold_h][:, :univ_size]
    warmup  = ma_w + 1
    free_at = np.zeros(univ_size, dtype=np.int32)

    trade_rets, trade_dates = [], []

    for t in range(warmup, T - hold_h - 1):
        scores   = bub[t]
        signal   = scores < thr
        available = signal & (free_at <= t)
        if not available.any(): continue

        avail_idx = np.where(available)[0]
        n_pick    = min(top_n, len(avail_idx))
        # pick lowest bubble scores (most depressed)
        chosen_local = np.argpartition(scores[avail_idx], n_pick - 1)[:n_pick]
        chosen    = avail_idx[chosen_local]

        # Return: entry open[t+1], exit close[t+hold_h]
        net_ret = float(fwd[t, chosen].mean()) - TC
        net_ret = max(-0.5, min(5.0, net_ret))

        trade_rets.append(net_ret)
        trade_dates.append(idx[t])
        free_at[chosen] = t + hold_h

    if len(trade_rets) < 5:
        continue

    s = pd.Series(trade_rets, index=pd.DatetimeIndex(trade_dates))
    p = perf(s, years)
    if not np.isfinite(p["sharpe"]): continue

    results.append(dict(
        UnivSize=univ_size, MA_h=ma_w, Threshold=thr,
        Hold_h=hold_h, TopN=top_n,
        Sharpe=p["sharpe"], Sortino=p["sortino"],
        Ann_Ret=p["ann_ret"], MaxDD=p["maxdd"],
        Total_Ret=p["total_ret"], Win_Rate=p["win_rate"],
        Trades=len(s)
    ))

df = pd.DataFrame(results).sort_values("Sharpe", ascending=False).reset_index(drop=True)
print(f"Done in {time.time()-t1:.1f}s  ({len(df)} valid combos)")

# ─────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*130}")
print("TOP 30 COMBINATIONS (sorted by Sharpe)")
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
# BEST COMBO YEARLY BREAKDOWN
# ─────────────────────────────────────────────────────────────
best = df.iloc[0]
print(f"\n{'='*130}")
print("BEST PARAMETERS")
print(f"{'='*130}")
print(f"  Universe:  top-{int(best.UnivSize)} short-interest stocks")
print(f"  Bubble MA: {int(best.MA_h)}h (~{best.MA_h/6.5:.0f} sessions)")
print(f"  Threshold: {best.Threshold}  (buy when per-stock bubble < this)")
print(f"  Hold:      {int(best.Hold_h)}h (~{best.Hold_h/6.5:.1f} sessions)")
print(f"  Top-N:     {int(best.TopN)}")
print(f"\n  Annual Return:  {best.Ann_Ret:.2%}")
print(f"  Sharpe Ratio:   {best.Sharpe:.4f}")
print(f"  Sortino Ratio:  {best.Sortino:.4f}")
print(f"  Max Drawdown:   {best.MaxDD:.2%}")
print(f"  Win Rate:       {best.Win_Rate:.1%}")
print(f"  Total Trades:   {int(best.Trades)}")

# Rebuild best series
univ_b  = int(best.UnivSize); ma_b  = int(best.MA_h)
thr_b   = best.Threshold;     hold_b = int(best.Hold_h); topn_b = int(best.TopN)

bub_b   = bub_cache[ma_b][:, :univ_b]
fwd_b   = fwd_cache[hold_b][:, :univ_b]
warmup  = ma_b + 1
free_at = np.zeros(univ_b, dtype=np.int32)
tb, td  = [], []

for t in range(warmup, T - hold_b - 1):
    scores    = bub_b[t]; signal = scores < thr_b
    available = signal & (free_at <= t)
    if not available.any(): continue
    avail_idx = np.where(available)[0]
    n_pick    = min(topn_b, len(avail_idx))
    chosen    = avail_idx[np.argpartition(scores[avail_idx], n_pick - 1)[:n_pick]]
    net_ret   = float(fwd_b[t, chosen].mean()) - TC
    tb.append(max(-0.5, min(5.0, net_ret))); td.append(idx[t])
    free_at[chosen] = t + hold_b

s_best = pd.Series(tb, index=pd.DatetimeIndex(td))
w_best = (1 + s_best).cumprod()

print(f"\n{'='*130}")
print("YEARLY BREAKDOWN (Best Parameters)")
print(f"{'='*130}")
print(f"  {'Year':<6}  {'Return':>9}  {'Ann Ret':>9}  {'Sharpe':>8}  "
      f"{'Sortino':>9}  {'MaxDD':>9}  {'WinRate':>8}  {'Trades':>7}")
print(f"  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*7}")

yearly_rows = []
for yr in sorted(s_best.index.year.unique()):
    ys = s_best[s_best.index.year == yr]
    yw = w_best[w_best.index.year == yr]
    if len(ys) == 0: continue
    ytr = (1 + ys).prod() - 1
    yar = (1 + ytr) ** (252 / max(len(ys), 1)) - 1 if ytr > -1 else -1
    ysh = ys.mean() / ys.std() * np.sqrt(252) if ys.std() > 0 else 0
    ydn = ys[ys < 0].std(ddof=0)
    yso = ys.mean() / ydn * np.sqrt(252) if ydn > 0 else 0
    ydd = (yw / yw.cummax() - 1).min()
    ywr = (ys > 0).mean()
    act = int((ys != 0).sum())
    print(f"  {yr:<6}  {ytr:>9.2%}  {yar:>9.2%}  {ysh:>8.3f}  "
          f"{yso:>9.3f}  {ydd:>9.2%}  {ywr:>8.1%}  {act:>7}")
    yearly_rows.append(dict(Year=yr, Return=ytr, Ann_Ret=yar,
                            Sharpe=ysh, Sortino=yso, MaxDD=ydd,
                            Win_Rate=ywr, Trades=act))

# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(
    f"Short Squeeze + Bubble Score (2019-2026)\n"
    f"Universe: top-{univ_b} short-int stocks  |  "
    f"Bubble MA={ma_b}h, Thr={thr_b}, Hold={hold_b}h, Top-{topn_b}\n"
    f"Ann={best.Ann_Ret:.1%}  Sharpe={best.Sharpe:.3f}  "
    f"MaxDD={best.MaxDD:.1%}  Win={best.Win_Rate:.0%}  Trades={int(best.Trades)}",
    fontsize=11, fontweight="bold")

axes[0,0].plot(w_best.index, w_best.values, lw=2, color="steelblue")
axes[0,0].set_title("Cumulative Wealth", fontweight="bold")
axes[0,0].set_ylabel("Wealth Multiple"); axes[0,0].grid(True, alpha=0.3)

dd_s = w_best / w_best.cummax() - 1
axes[0,1].fill_between(dd_s.index, dd_s.values, 0, alpha=0.6, color="red")
axes[0,1].set_title(f"Drawdown  (Max={best.MaxDD:.1%})", fontweight="bold")
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
plt.savefig("results/short_squeeze_bubble_backtest.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: results/short_squeeze_bubble_backtest.png")

df.to_csv("results/short_squeeze_bubble_grid.csv", index=False)
pd.DataFrame(yearly_rows).to_csv("results/short_squeeze_bubble_yearly.csv", index=False)
print("Saved: short_squeeze_bubble_grid.csv  |  short_squeeze_bubble_yearly.csv")
print(f"\nTotal runtime: {time.time()-t0:.1f}s  |  DONE")
