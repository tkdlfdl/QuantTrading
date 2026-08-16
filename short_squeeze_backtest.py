"""
SHORT SQUEEZE STRATEGY — CORRECT IMPLEMENTATION
=================================================
Universe : Top-N stocks by short interest (results/short_interest_universe.csv)
Signal   : Per-stock bubble score drops below threshold  (oversold + heavily shorted)
Entry    : LONG at open of next bar  (no look-ahead)
Exit     : Close of bar t + hold_hours
Sizing   : Equal weight across all positions open that day → no leverage blow-up

Key fix vs prior version:
  - Daily portfolio return = mean return of all positions ACTIVE that day
  - Handles concurrent positions correctly (no compound blow-up)
  - Correct intraday entry: uses actual open price on entry day
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

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
print("Loading data...")

# S&P500 + NASDAQ100 universe
from data.universe import get_universe
sp_nasdaq = set(get_universe())
print(f"S&P500 + NASDAQ100 universe: {len(sp_nasdaq)} tickers")

# Short interest — filter to S&P500 + NASDAQ100 only
si_raw = pd.read_csv("results/short_interest_universe.csv") \
           .sort_values("shortPercentOfFloat", ascending=False).reset_index(drop=True)
si = si_raw[si_raw["ticker"].isin(sp_nasdaq)].reset_index(drop=True)
print(f"Short interest (in S&P500+NASDAQ100): {len(si)} tickers cached")
print(f"Top 20 most shorted: {si['ticker'].head(20).tolist()}")

# Hourly OHLC
hc = pd.read_parquet("data/cache/merged_hourly_close.parquet")
ho = pd.read_parquet("data/cache/merged_hourly_open.parquet")
hc.index = hc.index.floor("h");  ho.index = ho.index.floor("h")
hc = hc[~hc.index.duplicated("last")]; ho = ho[~ho.index.duplicated("last")]
idx = hc.index.intersection(ho.index)
hc = hc.loc[idx].ffill();  ho = ho.loc[idx].ffill()
T = len(idx);  years = (idx[-1]-idx[0]).days/365.25
print(f"Hourly: {T} bars  {idx[0].date()} → {idx[-1].date()}  ({years:.2f} yrs)")

# ─────────────────────────────────────────────────────────────
# 2. UNIVERSE — top-80 by short interest, present in hourly data
# ─────────────────────────────────────────────────────────────
MAX_U = 80
avail  = [t for t in si["ticker"] if t in hc.columns]
si_ok  = si[si["ticker"].isin(avail)].head(MAX_U).reset_index(drop=True)
tickers = si_ok["ticker"].tolist()
print(f"\nUniverse: {len(tickers)} tickers  (top-{MAX_U} by shortPercentOfFloat)")

print(f"{'Rank':>5}  {'Ticker':<8}  {'Short%Float':>12}  {'ShortRatio':>11}")
for i, row in si_ok.head(20).iterrows():
    print(f"  {i+1:>3}  {row.ticker:<8}  {row.shortPercentOfFloat*100:>11.1f}%  {row.shortRatio:>11.2f}")

prices = hc[tickers].values.astype(np.float32)   # [T × U]
opens  = ho[tickers].values.astype(np.float32)   # [T × U]

# ─────────────────────────────────────────────────────────────
# 3. PRE-COMPUTE DAILY STOCK RETURNS (for portfolio P&L)
# ─────────────────────────────────────────────────────────────
# Map each hourly bar to its trading day
bar_day = idx.normalize().values                        # [T] datetime64
trading_days = np.unique(bar_day)                       # sorted unique days
day_to_int   = {d: i for i, d in enumerate(trading_days)}
bar_day_int  = np.array([day_to_int[d] for d in bar_day], dtype=np.int32)
D = len(trading_days)

# For each day and stock: last bar index (for close) and first bar index
day_last  = np.zeros(D, dtype=np.int32)
day_first = np.zeros(D, dtype=np.int32)
for t in range(T):
    d = bar_day_int[t]
    day_last[d] = t
for t in range(T-1, -1, -1):
    d = bar_day_int[t]
    day_first[d] = t

# Daily close prices [D × U]
daily_close = prices[day_last]                         # close of last bar each day

# Daily stock returns (close-to-close) [D × U]
# day 0: NaN/0, day d: close[d] / close[d-1] - 1
daily_ret_cc = np.zeros((D, len(tickers)), dtype=np.float32)
daily_ret_cc[1:] = daily_close[1:] / np.maximum(daily_close[:-1], 1e-8) - 1
daily_ret_cc = np.clip(daily_ret_cc, -0.20, 0.20)

# Entry-day return: close[entry_day] / open[entry_bar] - 1
# We'll compute this per-trade during backtest

# ─────────────────────────────────────────────────────────────
# 4. PRE-COMPUTE BUBBLE SCORES (causal — no look-ahead)
# ─────────────────────────────────────────────────────────────
MA_LIST  = [52, 104, 156, 208]   # hours
print(f"\nPre-computing bubble scores ({len(MA_LIST)} MA windows)...")

bub_cache = {}
for ma in MA_LIST:
    df_p = pd.DataFrame(prices, index=idx, columns=tickers)
    lp   = np.log(df_p.replace(0, np.nan).ffill())
    fair = df_p.rolling(ma, min_periods=ma//2).mean()
    res  = lp - np.log(fair.replace(0, np.nan))
    z    = (res - res.rolling(ma, min_periods=ma//2).mean()) \
           / res.rolling(ma, min_periods=ma//2).std()
    bub  = np.tanh(z/2).fillna(0).values.astype(np.float32)
    bub_cache[ma] = bub
    print(f"  MA={ma}h  pct<-0.5: {(bub<-0.5).mean()*100:.2f}%  "
          f"pct<-0.8: {(bub<-0.8).mean()*100:.2f}%")

# ─────────────────────────────────────────────────────────────
# 5. GRID SEARCH
# ─────────────────────────────────────────────────────────────
# For each combo, build trade list then compute correct daily portfolio returns

UNIV_SIZES  = [20, 40, 60, 80]
THR_LIST    = [-0.8, -0.6, -0.4, -0.2]
HOLD_LIST   = [4, 8, 13, 26, 52]   # hours
TOPN_LIST   = [3, 5, 10]
TC          = 0.001    # round-trip cost per trade
RF          = 0.02

total = len(UNIV_SIZES)*len(MA_LIST)*len(THR_LIST)*len(HOLD_LIST)*len(TOPN_LIST)
print(f"\nGrid: {total} combos  (4×4×4×5×3)")
t1 = time.time()

def run_combo(univ_n, ma, thr, hold_h, top_n):
    """Returns daily portfolio return Series (correct equal-weight sizing)."""
    bub    = bub_cache[ma][:, :univ_n]
    pr     = prices[:, :univ_n]
    op     = opens[:, :univ_n]
    warmup = ma + 1
    free_at = np.zeros(univ_n, dtype=np.int32)  # bar when each ticker is free

    # ── collect trades ────────────────────────────────────────
    # Each trade: (signal_bar, entry_bar, exit_bar, stock_indices)
    trades = []
    for t in range(warmup, T - hold_h - 1):
        scores    = bub[t]
        available = (scores < thr) & (free_at <= t)
        if not available.any(): continue
        avail_idx = np.where(available)[0]
        n_pick    = min(top_n, len(avail_idx))
        # pick stocks with LOWEST bubble score (most depressed)
        chosen    = avail_idx[np.argpartition(scores[avail_idx], n_pick-1)[:n_pick]]
        entry_bar = t + 1
        exit_bar  = min(t + hold_h, T - 1)
        trades.append((entry_bar, exit_bar, chosen))
        free_at[chosen] = exit_bar

    if len(trades) < 5:
        return None

    # ── build daily portfolio return ──────────────────────────
    # For each day d: collect positions open that day, compute equal-weight P&L
    # Position active on day d if: day_of(entry_bar) <= d <= day_of(exit_bar)

    # daily_pnl[d] = sum of (weight × stock_daily_return) for all active positions
    daily_port_num = np.zeros(D, dtype=np.float64)  # sum of returns
    daily_port_den = np.zeros(D, dtype=np.float64)  # sum of weights (= n_active)

    for (entry_bar, exit_bar, chosen) in trades:
        entry_day = bar_day_int[entry_bar]
        exit_day  = bar_day_int[exit_bar]
        days      = np.arange(entry_day, exit_day + 1)

        for s in chosen:
            ep = op[entry_bar, s]
            xp = pr[exit_bar, s]
            if ep <= 0 or xp <= 0 or not (np.isfinite(ep) and np.isfinite(xp)):
                continue

            # Middle days: use pre-computed close-to-close (vectorized)
            day_rets = daily_ret_cc[days, s].copy()

            # Override entry day: open[entry_bar] → close[day_last[entry_day]]
            day_close_entry = pr[day_last[entry_day], s]
            day_rets[0] = (day_close_entry / ep - 1) if day_close_entry > 0 else 0.0

            # Override exit day: close[day_last[exit_day-1]] → close[exit_bar]
            if len(days) > 1:
                prev_close = pr[day_last[exit_day - 1], s]
                day_rets[-1] = (xp / prev_close - 1) if prev_close > 0 else 0.0

            day_rets = np.clip(day_rets, -0.20, 0.20)
            daily_port_num[days] += day_rets
            daily_port_den[days] += 1.0

    # Equal-weight portfolio return each day
    active_mask = daily_port_den > 0
    port_ret    = np.zeros(D, dtype=np.float64)
    port_ret[active_mask] = (daily_port_num[active_mask]
                             / daily_port_den[active_mask]) - TC / hold_h

    s_ret = pd.Series(port_ret, index=pd.to_datetime(trading_days))
    return s_ret

def calc_metrics(s, yrs):
    if s is None or len(s) < 20: return {}
    rf_d = RF / 252
    exc  = s - rf_d
    std  = s.std()
    sh   = exc.mean() / std * np.sqrt(252) if std > 0 else 0
    dn   = s[s < 0].std(ddof=0)
    so   = exc.mean() / dn  * np.sqrt(252) if dn  > 0 else 0
    w    = (1 + s).cumprod()
    dd   = (w / w.cummax() - 1).min()
    tr   = w.iloc[-1] - 1
    ar   = (1 + tr) ** (1 / yrs) - 1 if tr > -1 else -1
    act  = s[s != 0]
    wr   = (act > 0).mean() if len(act) > 0 else 0.5
    return dict(sharpe=sh, sortino=so, maxdd=dd, ann_ret=ar,
                total_ret=tr, win_rate=wr, active_days=int(active_mask_count(s)))

def active_mask_count(s): return int((s != 0).sum())

results = []
combo_n = 0
for univ_n, ma, thr, hold_h, top_n in product(
        UNIV_SIZES, MA_LIST, THR_LIST, HOLD_LIST, TOPN_LIST):
    combo_n += 1
    s_ret = run_combo(univ_n, ma, thr, hold_h, top_n)
    m = calc_metrics(s_ret, years)
    if not m or not np.isfinite(m.get("sharpe", np.nan)): continue
    results.append(dict(UnivSize=univ_n, MA_h=ma, Threshold=thr,
                        Hold_h=hold_h, TopN=top_n, **m))
    if combo_n % 100 == 0:
        print(f"  {combo_n}/{total}  elapsed={time.time()-t1:.0f}s  "
              f"valid={len(results)}  best_sh={max(r['sharpe'] for r in results):.3f}")

df = pd.DataFrame(results).sort_values("sharpe", ascending=False).reset_index(drop=True)
print(f"\nDone in {time.time()-t1:.1f}s  ({len(df)} valid combos)")

# ─────────────────────────────────────────────────────────────
# 6. RESULTS
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*130}")
print("TOP 30 COMBINATIONS (Sharpe — correct equal-weight daily P&L)")
print(f"{'='*130}")
fmt = {"sharpe":"{:.4f}".format,"sortino":"{:.4f}".format,
       "ann_ret":"{:.2%}".format,"maxdd":"{:.2%}".format,
       "total_ret":"{:.2%}".format,"win_rate":"{:.1%}".format}
print(df.head(30).to_string(index=False, formatters=fmt))

print(f"\nPositive Sharpe: {(df.sharpe>0).sum()}/{len(df)}")
print(f"Sharpe > 0.5:    {(df.sharpe>0.5).sum()}/{len(df)}")
print(f"Sharpe > 1.0:    {(df.sharpe>1.0).sum()}/{len(df)}")

print(f"\n{'='*130}")
print("PARAMETER SENSITIVITY")
print(f"{'='*130}")
for col, label in [("UnivSize","Universe Size"),("MA_h","Bubble MA (h)"),
                   ("Threshold","Threshold"),("Hold_h","Hold (h)"),("TopN","Top-N")]:
    g = df.groupby(col)["sharpe"].agg(["mean","max","count"]).reset_index()
    g.columns = [label, "Avg Sharpe", "Best Sharpe", "N"]
    print(f"\n{g.to_string(index=False)}")

# ─────────────────────────────────────────────────────────────
# 7. BEST COMBO DETAIL
# ─────────────────────────────────────────────────────────────
best = df.iloc[0]
print(f"\n{'='*130}")
print("BEST PARAMETERS")
print(f"{'='*130}")
print(f"  Universe:  top-{int(best.UnivSize)} short-interest stocks")
print(f"  Bubble MA: {int(best.MA_h)}h (~{best.MA_h/6.5:.0f} sessions)")
print(f"  Threshold: {best.Threshold}  (BUY when per-stock bubble < this)")
print(f"  Hold:      {int(best.Hold_h)}h (~{best.Hold_h/6.5:.1f} sessions)")
print(f"  Top-N:     {int(best.TopN)}")
print(f"\n  Annual Return:  {best.ann_ret:.2%}")
print(f"  Sharpe Ratio:   {best.sharpe:.4f}")
print(f"  Sortino Ratio:  {best.sortino:.4f}")
print(f"  Max Drawdown:   {best.maxdd:.2%}")
print(f"  Win Rate:       {best.win_rate:.1%}")
print(f"  Active Days:    {int(best.active_days)}")

# Rebuild best daily series
best_s = run_combo(int(best.UnivSize), int(best.MA_h),
                   best.Threshold, int(best.Hold_h), int(best.TopN))
w_best = (1 + best_s).cumprod()

print(f"\n{'='*130}")
print("YEARLY BREAKDOWN (Best Parameters)")
print(f"{'='*130}")
print(f"  {'Year':<6}  {'Return':>9}  {'Ann Ret':>9}  {'Sharpe':>8}  "
      f"{'Sortino':>9}  {'MaxDD':>9}  {'WinRate':>8}  {'Active Days':>12}")
print(f"  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*12}")

yearly_rows = []
for yr in sorted(best_s.index.year.unique()):
    ys = best_s[best_s.index.year == yr]
    yw = w_best[w_best.index.year == yr]
    if len(ys) == 0: continue
    ytr = (1+ys).prod()-1
    yar = (1+ytr)**(252/max(len(ys),1))-1 if ytr>-1 else -1
    ysh = ys.mean()/ys.std()*np.sqrt(252) if ys.std()>0 else 0
    ydn = ys[ys<0].std(ddof=0)
    yso = ys.mean()/ydn*np.sqrt(252) if ydn>0 else 0
    ydd = (yw/yw.cummax()-1).min()
    act = ys[ys!=0]; ywr = (act>0).mean() if len(act)>0 else 0.5
    n_act = int((ys!=0).sum())
    print(f"  {yr:<6}  {ytr:>9.2%}  {yar:>9.2%}  {ysh:>8.3f}  "
          f"{yso:>9.3f}  {ydd:>9.2%}  {ywr:>8.1%}  {n_act:>12}")
    yearly_rows.append(dict(Year=yr, Return=ytr, Ann_Ret=yar, Sharpe=ysh,
                            Sortino=yso, MaxDD=ydd, Win_Rate=ywr, Active_Days=n_act))

# ─────────────────────────────────────────────────────────────
# 8. PLOTS
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(
    f"Short Squeeze + Bubble Score  (2019-2026)\n"
    f"Universe: top-{int(best.UnivSize)} short-int  |  "
    f"Bubble MA={int(best.MA_h)}h, Thr={best.Threshold}, "
    f"Hold={int(best.Hold_h)}h, Top-{int(best.TopN)}\n"
    f"Ann={best.ann_ret:.1%}  Sharpe={best.sharpe:.3f}  MaxDD={best.maxdd:.2%}",
    fontsize=11, fontweight="bold")

axes[0,0].plot(w_best.index, w_best.values, lw=2, color="steelblue")
axes[0,0].set_title("Cumulative Wealth", fontweight="bold")
axes[0,0].set_ylabel("Wealth Multiple"); axes[0,0].grid(True, alpha=0.3)

dd_s = w_best / w_best.cummax() - 1
axes[0,1].fill_between(dd_s.index, dd_s.values, 0, alpha=0.6, color="red")
axes[0,1].set_title(f"Drawdown  (Max={best.maxdd:.2%})", fontweight="bold")
axes[0,1].grid(True, alpha=0.3)

for ax, (col, lbl, clr) in zip(
        [axes[0,2], axes[1,0], axes[1,1]],
        [("Threshold","Bubble Threshold","steelblue"),
         ("Hold_h","Hold Period (h)","darkorange"),
         ("UnivSize","Universe Size","purple")]):
    g = df.groupby(col)["sharpe"].agg(["mean","max"]).reset_index()
    ax.plot(g[col], g["mean"], marker="o", lw=2, label="Avg", color=clr)
    ax.plot(g[col], g["max"],  marker="s", lw=2, ls="--", label="Best", color="green")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Sharpe vs {lbl}", fontweight="bold")
    ax.set_xlabel(lbl); ax.legend(); ax.grid(True, alpha=0.3)

if yearly_rows:
    ydf  = pd.DataFrame(yearly_rows)
    clrs = ["green" if r>=0 else "red" for r in ydf["Return"]]
    axes[1,2].bar(ydf["Year"].astype(str), ydf["Return"]*100, color=clrs, alpha=0.75)
    axes[1,2].axhline(0, color="black", lw=0.8)
    axes[1,2].set_title("Yearly Returns (%)", fontweight="bold")
    axes[1,2].set_ylabel("Return (%)"); axes[1,2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("results/short_squeeze_correct.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: results/short_squeeze_correct.png")

df.to_csv("results/short_squeeze_correct_grid.csv", index=False)
pd.DataFrame(yearly_rows).to_csv("results/short_squeeze_correct_yearly.csv", index=False)
print(f"Saved: grid.csv  |  yearly.csv")
print(f"\nTotal runtime: {time.time()-t0:.1f}s  |  DONE")
