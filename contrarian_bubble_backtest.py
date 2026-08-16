"""
CONTRARIAN BUBBLE SCORE STRATEGY
==================================
Logic:
  When a stock's price drops significantly below its rolling trend
  (bubble score < threshold), BUY — expect mean reversion.

  bubble = tanh( z / 2 )   where z normalises log-price residual
  Signal: bubble[t] < threshold  →  LONG at open[t+1]
  Exit  : close of bar t + hold_hours

Universe : ALL S&P500 + NASDAQ100 stocks in hourly cache (~405 tickers)
           No short-interest filter — pure price-based contrarian signal

Sizing   : Equal weight across all positions open that day
TC       : 0.1% per trade

Grid:
  MA window  : 52, 104, 156, 208 hours
  Threshold  : -0.8, -0.6, -0.4, -0.2
  Hold       : 4, 8, 13, 26, 52 hours
  Top-N      : 3, 5, 10
  Total      : 4 x 4 x 5 x 3 = 240 combinations
"""
import warnings, os, sys, time
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product

t0 = time.time()
os.makedirs("results", exist_ok=True)
RF = 0.02; TC = 0.001

# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────
print("Loading data...")
hc = pd.read_parquet("data/cache/merged_hourly_close.parquet")
ho = pd.read_parquet("data/cache/merged_hourly_open.parquet")
hc.index = hc.index.floor("h");  ho.index = ho.index.floor("h")
hc = hc[~hc.index.duplicated("last")]; ho = ho[~ho.index.duplicated("last")]
idx = hc.index.intersection(ho.index)

# Full S&P500 + NASDAQ100 universe — include ALL 515 tickers
# Stocks with high NaN (newer additions like ARM, GEV, KVUE) are included:
# their bubble score will be NaN during their pre-IPO period → no signal fires
from data.universe import get_universe
sp_nasdaq = set(get_universe())
valid = [c for c in hc.columns if c in sp_nasdaq]  # all 515, no NaN filter

hc = hc.loc[idx, valid].ffill()
ho = ho.loc[idx, valid].ffill()
T     = len(idx)
U     = len(valid)
years = (idx[-1] - idx[0]).days / 365.25

print(f"Period : {idx[0].date()} to {idx[-1].date()}  ({years:.2f} yrs)")
print(f"Universe: {U} tickers (full S&P500 + NASDAQ100, all tickers including new listings)")

prices = hc.values.astype(np.float32)
opens  = ho.values.astype(np.float32)

# ─────────────────────────────────────────────────────────────
# DAILY RETURN INFRASTRUCTURE
# ─────────────────────────────────────────────────────────────
bar_day     = idx.normalize().values
trading_days = np.unique(bar_day)
day_to_int  = {d: i for i, d in enumerate(trading_days)}
bar_day_int = np.array([day_to_int[d] for d in bar_day], dtype=np.int32)
D = len(trading_days)

day_last  = np.zeros(D, dtype=np.int32)
day_first = np.zeros(D, dtype=np.int32)
for t in range(T):
    day_last[bar_day_int[t]] = t
for t in range(T-1, -1, -1):
    day_first[bar_day_int[t]] = t

daily_close  = prices[day_last]
daily_ret_cc = np.zeros((D, U), dtype=np.float32)
daily_ret_cc[1:] = daily_close[1:] / np.maximum(daily_close[:-1], 1e-8) - 1
daily_ret_cc = np.clip(daily_ret_cc, -0.20, 0.20)

# ─────────────────────────────────────────────────────────────
# BUBBLE SCORES (causal)
# ─────────────────────────────────────────────────────────────
MA_LIST = [13, 26, 52, 104, 156, 208]
print(f"\nPre-computing bubble scores...")
bub_cache = {}
for ma in MA_LIST:
    df_p = pd.DataFrame(prices, index=idx, columns=valid)
    lp   = np.log(df_p.replace(0, np.nan).ffill())
    fair = df_p.rolling(ma, min_periods=ma//2).mean()
    res  = lp - np.log(fair.replace(0, np.nan))
    z    = (res - res.rolling(ma, min_periods=ma//2).mean()) \
           / res.rolling(ma, min_periods=ma//2).std()
    bub_cache[ma] = np.tanh(z/2).fillna(0).values.astype(np.float32)
    print(f"  MA={ma}h  pct<-0.8: {(bub_cache[ma]<-0.8).mean()*100:.2f}%  "
          f"pct<-0.6: {(bub_cache[ma]<-0.6).mean()*100:.2f}%")

# ─────────────────────────────────────────────────────────────
# BACKTEST CORE
# ─────────────────────────────────────────────────────────────
def run_combo(ma, thr, hold_h, top_n):
    bub     = bub_cache[ma]
    warmup  = ma + 1
    free_at = np.zeros(U, dtype=np.int32)

    daily_num = np.zeros(D, dtype=np.float64)
    daily_den = np.zeros(D, dtype=np.float64)

    for t in range(warmup, T - hold_h - 1):
        scores    = bub[t]
        available = (scores < thr) & (free_at <= t)
        if not available.any(): continue

        avail_idx = np.where(available)[0]
        n_pick    = min(top_n, len(avail_idx))
        chosen    = avail_idx[np.argpartition(scores[avail_idx], n_pick-1)[:n_pick]]

        entry_bar = t + 1
        exit_bar  = min(t + hold_h, T - 1)
        entry_day = bar_day_int[entry_bar]
        exit_day  = bar_day_int[exit_bar]
        days      = np.arange(entry_day, exit_day + 1)

        for s in chosen:
            ep = opens[entry_bar, s]
            xp = prices[exit_bar, s]
            if ep <= 0 or xp <= 0 or not (np.isfinite(ep) and np.isfinite(xp)):
                continue

            day_rets = daily_ret_cc[days, s].copy()

            # entry day: open → close
            dc_entry = prices[day_last[entry_day], s]
            day_rets[0] = (dc_entry / ep - 1) if dc_entry > 0 else 0.0

            # exit day: prev close → exit close
            if len(days) > 1:
                prev_c = prices[day_last[exit_day - 1], s]
                day_rets[-1] = (xp / prev_c - 1) if prev_c > 0 else 0.0

            day_rets = np.clip(day_rets, -0.20, 0.20)
            daily_num[days] += day_rets
            daily_den[days] += 1.0

        free_at[chosen] = exit_bar

    active = daily_den > 0
    port   = np.zeros(D, dtype=np.float64)
    port[active] = daily_num[active] / daily_den[active] - TC / hold_h
    return pd.Series(port, index=pd.to_datetime(trading_days)), active

def metrics(s, yrs):
    if len(s) < 20: return {}
    rf  = RF / 252; exc = s - rf; std = s.std()
    sh  = exc.mean() / std * np.sqrt(252) if std > 0 else 0
    dn  = s[s < 0].std(ddof=0)
    so  = exc.mean() / dn  * np.sqrt(252) if dn  > 0 else 0
    w   = (1 + s).cumprod()
    dd  = (w / w.cummax() - 1).min()
    tr  = w.iloc[-1] - 1
    ar  = (1 + tr) ** (1/yrs) - 1 if tr > -1 else -1
    act = s[s != 0]; wr = (act > 0).mean() if len(act) > 0 else 0.5
    return dict(sharpe=sh, sortino=so, maxdd=dd, ann_ret=ar,
                total_ret=tr, win_rate=wr, active_days=int((s!=0).sum()))

# ─────────────────────────────────────────────────────────────
# GRID SEARCH
# ─────────────────────────────────────────────────────────────
THR_LIST  = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2]
HOLD_LIST = [2, 4, 6, 8, 13, 26, 52]
TOPN_LIST = [5, 10, 20]
total = len(MA_LIST)*len(THR_LIST)*len(HOLD_LIST)*len(TOPN_LIST)
print(f"\nGrid search: {total} combinations (6 MA x 8 thr x 7 hold x 3 topN)...")
t1 = time.time()

results = []
for ma, thr, hold_h, top_n in product(MA_LIST, THR_LIST, HOLD_LIST, TOPN_LIST):
    s, _ = run_combo(ma, thr, hold_h, top_n)
    m = metrics(s, years)
    if not m or not np.isfinite(m["sharpe"]): continue
    results.append(dict(MA_h=ma, Threshold=thr, Hold_h=hold_h, TopN=top_n, **m))

df = pd.DataFrame(results).sort_values("sharpe", ascending=False).reset_index(drop=True)
print(f"Done in {time.time()-t1:.1f}s  ({len(df)} valid combos)")

# ─────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*120}")
print("TOP 20 COMBINATIONS (Contrarian Bubble Strategy — full S&P500+NASDAQ100)")
print(f"{'='*120}")
fmt = {"sharpe":"{:.4f}".format,"sortino":"{:.4f}".format,
       "ann_ret":"{:.2%}".format,"maxdd":"{:.2%}".format,
       "total_ret":"{:.2%}".format,"win_rate":"{:.1%}".format}
print(df.head(20).to_string(index=False, formatters=fmt))

print(f"\nPositive Sharpe: {(df.sharpe>0).sum()}/{len(df)}")
print(f"Sharpe > 0.5:    {(df.sharpe>0.5).sum()}/{len(df)}")
print(f"Sharpe > 1.0:    {(df.sharpe>1.0).sum()}/{len(df)}")
print(f"Sharpe > 2.0:    {(df.sharpe>2.0).sum()}/{len(df)}")

print(f"\n{'='*120}")
print("PARAMETER SENSITIVITY")
print(f"{'='*120}")
for col, label in [("MA_h","Bubble MA (h)"),("Threshold","Threshold"),
                   ("Hold_h","Hold (h)"),("TopN","Top-N")]:
    g = df.groupby(col)["sharpe"].agg(["mean","max","count"]).reset_index()
    g.columns = [label,"Avg Sharpe","Best Sharpe","N"]
    print(f"\n{g.to_string(index=False)}")

# ─────────────────────────────────────────────────────────────
# BEST COMBO YEARLY
# ─────────────────────────────────────────────────────────────
best = df.iloc[0]
print(f"\n{'='*120}")
print("BEST PARAMETERS")
print(f"{'='*120}")
print(f"  Bubble MA:  {int(best.MA_h)}h (~{best.MA_h/6.5:.0f} sessions)")
print(f"  Threshold:  {best.Threshold}  (buy when bubble < this)")
print(f"  Hold:       {int(best.Hold_h)}h")
print(f"  Top-N:      {int(best.TopN)}")
print(f"\n  Annual Return: {best.ann_ret:.2%}")
print(f"  Sharpe:        {best.sharpe:.4f}")
print(f"  Sortino:       {best.sortino:.4f}")
print(f"  Max Drawdown:  {best.maxdd:.2%}")
print(f"  Win Rate:      {best.win_rate:.1%}")
print(f"  Active Days:   {int(best.active_days)}")

s_best, _ = run_combo(int(best.MA_h), best.Threshold,
                       int(best.Hold_h), int(best.TopN))
w_best = (1 + s_best).cumprod()

print(f"\n{'='*120}")
print("YEARLY BREAKDOWN")
print(f"{'='*120}")
print(f"  {'Year':<6}  {'Return':>9}  {'Ann Ret':>9}  {'Sharpe':>8}  "
      f"{'Sortino':>9}  {'MaxDD':>9}  {'WinRate':>8}  {'Active Days':>12}")
print(f"  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*12}")

yearly_rows = []
for yr in sorted(s_best.index.year.unique()):
    ys = s_best[s_best.index.year == yr]
    yw = w_best[w_best.index.year == yr]
    if len(ys) == 0: continue
    ytr = (1+ys).prod()-1
    yar = (1+ytr)**(252/max(len(ys),1))-1 if ytr>-1 else -1
    ysh = ys.mean()/ys.std()*np.sqrt(252) if ys.std()>0 else 0
    ydn = ys[ys<0].std(ddof=0)
    yso = ys.mean()/ydn*np.sqrt(252) if ydn>0 else 0
    ydd = (yw/yw.cummax()-1).min()
    act = ys[ys!=0]; ywr = (act>0).mean() if len(act)>0 else 0.5
    n   = int((ys!=0).sum())
    print(f"  {yr:<6}  {ytr:>9.2%}  {yar:>9.2%}  {ysh:>8.3f}  "
          f"{yso:>9.3f}  {ydd:>9.2%}  {ywr:>8.1%}  {n:>12}")
    yearly_rows.append(dict(Year=yr, Return=ytr, Ann_Ret=yar,
                            Sharpe=ysh, Sortino=yso, MaxDD=ydd,
                            Win_Rate=ywr, Active_Days=n))

# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(
    f"Contrarian Bubble Score Strategy (S&P500+NASDAQ100, 2019-2026)\n"
    f"Buy when per-stock bubble < {best.Threshold} | "
    f"MA={int(best.MA_h)}h | Hold={int(best.Hold_h)}h | Top-{int(best.TopN)}\n"
    f"Ann={best.ann_ret:.1%}  Sharpe={best.sharpe:.3f}  MaxDD={best.maxdd:.2%}  "
    f"Win={best.win_rate:.0%}",
    fontsize=11, fontweight="bold")

axes[0,0].plot(w_best.index, w_best.values, lw=2, color="steelblue")
axes[0,0].set_title("Cumulative Wealth", fontweight="bold")
axes[0,0].set_ylabel("Wealth Multiple"); axes[0,0].grid(True, alpha=0.3)

dd_s = w_best / w_best.cummax() - 1
axes[0,1].fill_between(dd_s.index, dd_s.values, 0, alpha=0.6, color="red")
axes[0,1].set_title(f"Drawdown (Max={best.maxdd:.2%})", fontweight="bold")
axes[0,1].grid(True, alpha=0.3)

# Sensitivity charts
for ax, (col, lbl, clr) in zip(
        [axes[0,2], axes[1,0], axes[1,1]],
        [("Threshold","Bubble Threshold","steelblue"),
         ("Hold_h","Hold Period (h)","darkorange"),
         ("MA_h","Bubble MA (h)","purple")]):
    g = df.groupby(col)["sharpe"].agg(["mean","max"]).reset_index()
    ax.plot(g[col], g["mean"], marker="o", lw=2, label="Avg", color=clr)
    ax.plot(g[col], g["max"],  marker="s", lw=2, ls="--", label="Best", color="green")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Sharpe vs {lbl}", fontweight="bold")
    ax.set_xlabel(lbl); ax.legend(); ax.grid(True, alpha=0.3)

if yearly_rows:
    ydf  = pd.DataFrame(yearly_rows)
    clrs = ["green" if r >= 0 else "red" for r in ydf["Return"]]
    axes[1,2].bar(ydf["Year"].astype(str), ydf["Return"]*100, color=clrs, alpha=0.75)
    axes[1,2].axhline(0, color="black", lw=0.8)
    axes[1,2].set_title("Yearly Returns (%)", fontweight="bold")
    axes[1,2].set_ylabel("Return (%)"); axes[1,2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("results/contrarian_bubble_backtest_v2.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: results/contrarian_bubble_backtest.png")

df.to_csv("results/contrarian_bubble_grid_v2.csv", index=False)
pd.DataFrame(yearly_rows).to_csv("results/contrarian_bubble_yearly_v2.csv", index=False)
print(f"Saved: contrarian_bubble_grid_v2.csv  |  contrarian_bubble_yearly_v2.csv")
print(f"\nTotal runtime: {time.time()-t0:.1f}s  |  DONE")
