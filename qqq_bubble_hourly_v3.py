"""
QQQ BUBBLE + LONG MOMENTUM — EXTENDED GRID SEARCH v3
=====================================================
Strategy:  When QQQ bubble score LOW → buy top-N momentum stocks
Data:      2020-07-27 to 2026-06-02  (5.85 years, 9778 hourly bars)

EXTENDED GRID:
  Bubble MA windows:   5  [252, 500, 1000, 1500, 2000]
  Bubble thresholds:  10  [-0.1 to -0.9]
  Momentum lookbacks: 10  [1h .. 52h]
  Hold periods:       13  [2h .. 130h = ~4 weeks]
  Top-N stocks:        3  [5, 10, 20]

Total = 5 x 10 x 10 x 13 x 3 = 19,500 combinations

Speed: pre-compute forward returns + top-N indices so inner loop
       has zero numpy calls — runs in ~30-60 seconds.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, warnings, time
warnings.filterwarnings("ignore")
from itertools import product

t0 = time.time()

# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────
print("Loading + aligning data...")
qqq_raw    = pd.read_parquet("data/cache/qqq_hourly_close.parquet")["QQQ"]
stocks_raw = pd.read_parquet("data/cache/merged_hourly_close.parquet")

# Floor to hour so :00 and :30 timestamps align
qqq_h    = qqq_raw.copy();    qqq_h.index    = qqq_h.index.floor("h")
stocks_h = stocks_raw.copy(); stocks_h.index = stocks_h.index.floor("h")
qqq_h    = qqq_h[~qqq_h.index.duplicated(keep="last")]
stocks_h = stocks_h[~stocks_h.index.duplicated(keep="last")]

idx     = qqq_h.index.intersection(stocks_h.index)
qqq     = qqq_h.loc[idx]
stocks  = stocks_h.loc[idx]

# Filter to S&P500/NASDAQ100 universe (cross-reference with daily data)
try:
    daily_cols = set(pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet").columns)
    sp_nq_cols = [c for c in stocks.columns if c in daily_cols]
    stocks = stocks[sp_nq_cols]
    print(f"Filtered to S&P500/NASDAQ100 universe: {len(sp_nq_cols)} tickers")
except:
    pass

# Keep columns with < 30% NaN, then ffill only (no fillna(0) — avoids fake prices)
valid   = stocks.columns[stocks.isna().mean() < 0.30]
stocks  = stocks[valid].ffill()   # no fillna(0): leave leading NaN as NaN

years   = (idx[-1] - idx[0]).days / 365.25
n, ncols = stocks.shape
print(f"Bars: {n}  |  {idx[0].date()} to {idx[-1].date()}  ({years:.2f} yrs)")
print(f"Stocks: {ncols} tickers (S&P500/NASDAQ100, <30% NaN)")

# Returns: fill NaN with 0 after pct_change, clip extreme values (+/-10% per bar)
ret_raw = stocks.pct_change().clip(-0.10, 0.10)
ret_np  = ret_raw.fillna(0).values.astype(np.float32)  # (T, N)
thr_ann = 252 * 6.5   # hours/year for Sharpe annualisation

# ─────────────────────────────────────────────────────────────
# BUBBLE SCORE
# ─────────────────────────────────────────────────────────────
def bubble_score(price, ma_w, z_w):
    log_p   = np.log(price)
    fair    = price.rolling(ma_w).mean()
    log_f   = np.log(fair)
    res     = log_p - log_f
    z       = (res - res.rolling(z_w).mean()) / res.rolling(z_w).std()
    return np.tanh(z / 2)

# ─────────────────────────────────────────────────────────────
# GRIDS
# ─────────────────────────────────────────────────────────────
bubble_ma_list  = [252, 500, 1000, 1500, 2000]
bubble_thr_list = [-0.1, -0.15, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9]
mom_lb_list     = [2, 3, 6, 10, 13, 20, 26, 40, 52, 78]   # start from 2h; add 78h (~2 weeks)
hold_list       = [2, 4, 6, 8, 13, 20, 26, 40, 52, 65, 78, 104, 130]
topn_list       = [5, 10, 20]

total = (len(bubble_ma_list) * len(bubble_thr_list) *
         len(mom_lb_list) * len(hold_list) * len(topn_list))

print(f"\nGrid: {total} combinations")
print(f"  Bubble MA (h):     {bubble_ma_list}")
print(f"  Bubble threshold:  {bubble_thr_list}")
print(f"  Mom lookback (h):  {mom_lb_list}")
print(f"  Hold period  (h):  {hold_list}")
print(f"  Top-N:             {topn_list}")

# ─────────────────────────────────────────────────────────────
# PRE-COMPUTE: bubble arrays
# ─────────────────────────────────────────────────────────────
print("\nPre-computing bubble scores...")
bub_cache = {}
for maw in bubble_ma_list:
    b = bubble_score(qqq, maw, maw).fillna(0).values.astype(np.float32)
    bub_cache[maw] = b
    p3, p5, p7 = (b < -0.3).mean()*100, (b < -0.5).mean()*100, (b < -0.7).mean()*100
    print(f"  ma={maw:4d}h  min={b.min():.3f}  max={b.max():.3f}  "
          f"pct<-0.3:{p3:.1f}%  pct<-0.5:{p5:.1f}%  pct<-0.7:{p7:.1f}%")

# ─────────────────────────────────────────────────────────────
# PRE-COMPUTE: compound forward return per hold period
# FIX: use log-cumsum so we get (1+r1)*(1+r2)*...-1  per stock
# NOT mean(r1, r2, ...) which was 100x too small
# fwd[h][i,j] = total compound return of stock j from bar i+1 to i+h
# ─────────────────────────────────────────────────────────────
print("Pre-computing compound forward returns (FIX: product not mean)...")
log_ret    = np.log1p(ret_np.clip(-0.10, 0.10))   # log(1+r), shape (T, N)
cumlog     = np.cumsum(log_ret, axis=0)             # cumulative log-return
fwd_cache  = {}
for h in hold_list:
    fwd = np.zeros((n, ncols), dtype=np.float32)
    valid_end = n - h
    # compound return from bar i+1 to i+h:  exp(cumlog[i+h] - cumlog[i]) - 1
    fwd[:valid_end] = np.expm1(cumlog[h:] - cumlog[:valid_end])
    fwd_cache[h] = fwd
    print(f"  hold={h:3d}h  done  (sample mean compound ret: {fwd[:valid_end].mean()*100:.3f}%)")

# ─────────────────────────────────────────────────────────────
# PRE-COMPUTE: top-N indices per momentum lookback + topn
# top_idx_cache[lb][topn] = (T, topn) int16 array
# ─────────────────────────────────────────────────────────────
print("Pre-computing top-N momentum indices...")
mom_cache  = {}   # raw momentum (T, N)
topn_cache = {}   # top-N indices

for lb in mom_lb_list:
    m_raw = stocks.pct_change(lb).fillna(0).values.astype(np.float32)
    mom_cache[lb] = m_raw
    topn_cache[lb] = {}
    for topn in topn_list:
        # argpartition gives top-N (unsorted but correct)
        top_idx = np.argpartition(m_raw, -topn, axis=1)[:, -topn:]  # (T, topn)
        topn_cache[lb][topn] = top_idx
    print(f"  lookback={lb:3d}h  done")

# ─────────────────────────────────────────────────────────────
# PRE-COMPUTE: trade return at each bar for each (lb, topn, hold)
# trade_ret_cache[lb][topn][hold] = float32 array (T,)
# trade_ret_cache[lb][topn][hold][i] = avg fwd return of top-N at bar i
# ─────────────────────────────────────────────────────────────
print("Pre-computing per-bar trade returns...")
trade_ret_cache = {}
for lb in mom_lb_list:
    trade_ret_cache[lb] = {}
    for topn in topn_list:
        trade_ret_cache[lb][topn] = {}
        ti = topn_cache[lb][topn]   # (T, topn)
        for h in hold_list:
            fwd = fwd_cache[h]       # (T, N)
            # gather top-N returns: fwd[rows, ti[rows]]
            rows    = np.arange(n)
            sel_ret = fwd[rows[:, None], ti]   # (T, topn)
            trade_ret_cache[lb][topn][h] = sel_ret.mean(axis=1).astype(np.float32)  # (T,)
        print(f"  lb={lb:3d}h  topn={topn:2d}  done")

print(f"Pre-computation done in {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────
# GRID SEARCH  — pure Python loop, all numpy pre-computed
# ─────────────────────────────────────────────────────────────
print(f"\nRunning {total} combinations...")
t1 = time.time()

results = []
combo_n  = 0

for maw, thr, lb, hold, topn in product(
        bubble_ma_list, bubble_thr_list, mom_lb_list, hold_list, topn_list):

    combo_n += 1
    if combo_n % 2000 == 0:
        elapsed = time.time() - t1
        eta = elapsed / combo_n * (total - combo_n)
        print(f"  {combo_n}/{total}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    bub       = bub_cache[maw]
    tr_arr    = trade_ret_cache[lb][topn][hold]   # (T,) pre-computed returns
    warmup    = max(maw, lb) + 5

    # non-overlapping trade loop (no numpy inside)
    trade_rets = []
    i = warmup
    while i < n - hold:
        if bub[i] < thr:
            r = float(tr_arr[i]) - 0.001   # subtract 0.1% cost
            if r < -0.50: r = -0.50    # allow up to -50% drawdown on one trade
            if r >  2.00: r =  2.00    # allow up to 200% gain on one trade
            trade_rets.append(r)
            i += hold
        else:
            i += 1

    if len(trade_rets) < 10:
        continue

    s  = np.array(trade_rets, dtype=np.float64)
    w  = np.cumprod(1.0 + s)
    if not np.isfinite(w[-1]) or w[-1] <= 0:
        continue
    tr  = w[-1] - 1.0
    ar  = (1.0 + tr) ** (1.0/years) - 1.0 if tr > -1 else -1.0
    std = s.std()
    # CORRECT: annualise by ACTUAL trade frequency (n_trades / years)
    # NOT thr_ann/hold which assumes always-in-trade
    actual_tpy = len(s) / years
    sh  = s.mean() / std * np.sqrt(actual_tpy) if std > 0 else 0.0
    if not np.isfinite(sh): continue
    dn  = s[s < 0].std()
    so  = s.mean() / dn * np.sqrt(actual_tpy) if dn > 0 else 0.0
    dd  = (w / np.maximum.accumulate(w) - 1.0).min()
    wr  = (s > 0).mean() * 100.0

    results.append((maw, thr, lb, hold, topn, ar, sh, so, dd, wr, len(s), tr))

print(f"Grid search done in {time.time()-t1:.1f}s  ({len(results)} valid results)")

# ─────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────
cols = ["ma_window","bubble_thr","mom_lb","hold_h","top_n",
        "annual_ret","sharpe","sortino","max_dd","win_rate","n_trades","total_ret"]
df = pd.DataFrame(results, columns=cols).sort_values("sharpe", ascending=False).reset_index(drop=True)

print("\n" + "="*160)
print("TOP 40 COMBINATIONS  (sorted by Sharpe)")
print("="*160)
fmt = {"annual_ret":"{:.2%}".format, "sharpe":"{:.3f}".format, "sortino":"{:.3f}".format,
       "max_dd":"{:.2%}".format,     "win_rate":"{:.1f}".format,
       "total_ret":"{:.2%}".format}
print(df.head(40)[cols].to_string(index=False, formatters=fmt))

print(f"\nPositive-Sharpe combos: {(df.sharpe > 0).sum()}/{len(df)}")
print(f"Sharpe > 1 combos:      {(df.sharpe > 1).sum()}/{len(df)}")
print(f"Sharpe > 2 combos:      {(df.sharpe > 2).sum()}/{len(df)}")

# ─────────────────────────────────────────────────────────────
# PARAMETER SENSITIVITY
# ─────────────────────────────────────────────────────────────
print("\n" + "="*160)
print("PARAMETER SENSITIVITY  (avg / best Sharpe per parameter value)")
print("="*160)
for col, label in [("bubble_thr","Bubble Threshold"),
                   ("ma_window", "Bubble MA (h)"),
                   ("mom_lb",    "Mom Lookback (h)"),
                   ("hold_h",    "Hold Period (h)"),
                   ("top_n",     "Top-N")]:
    g = df.groupby(col)["sharpe"].agg(["mean","max","count"]).reset_index()
    g.columns = [label,"Avg Sharpe","Best Sharpe","N"]
    g["Avg Sharpe"]  = g["Avg Sharpe"].map("{:.3f}".format)
    g["Best Sharpe"] = g["Best Sharpe"].map("{:.3f}".format)
    print(f"\n{g.to_string(index=False)}")

# ─────────────────────────────────────────────────────────────
# BEST COMBO — rebuild full series
# ─────────────────────────────────────────────────────────────
best  = df.iloc[0]
maw_b = int(best.ma_window);  thr_b = best.bubble_thr
lb_b  = int(best.mom_lb);     hold_b= int(best.hold_h)
topn_b= int(best.top_n)

print("\n" + "="*160)
print("BEST PARAMETERS")
print("="*160)
print(f"  MA Window:         {maw_b}h  (~{maw_b/6.5:.0f} trading days)")
print(f"  Bubble Threshold:  {thr_b}")
print(f"  Momentum Lookback: {lb_b}h  (~{lb_b/6.5:.1f} sessions)")
print(f"  Hold Period:       {hold_b}h  (~{hold_b/6.5:.1f} sessions = ~{hold_b/32.5:.1f} weeks)")
print(f"  Top-N Stocks:      {topn_b}")
print(f"\n  Annual Return: {best.annual_ret:.2%}")
print(f"  Sharpe Ratio:  {best.sharpe:.4f}")
print(f"  Max Drawdown:  {best.max_dd:.2%}")
print(f"  Win Rate:      {best.win_rate:.1f}%")
print(f"  Total Trades:  {int(best.n_trades)}")

bub_b   = bub_cache[maw_b]
tr_b    = trade_ret_cache[lb_b][topn_b][hold_b]
warmup  = max(maw_b, lb_b) + 5

best_rets, best_ts = [], []
i = warmup
while i < n - hold_b:
    if bub_b[i] < thr_b:
        r = float(tr_b[i]) - 0.001
        r = max(-0.10, min(0.10, r))
        best_rets.append(r); best_ts.append(idx[i])
        i += hold_b
    else:
        i += 1

bs_s = pd.Series(best_rets, index=pd.DatetimeIndex(best_ts))
bs_w = (1 + bs_s).cumprod()

# QQQ benchmark
qqq_bench = (1 + qqq.pct_change().fillna(0).loc[bs_w.index[0]:]).cumprod()
qqq_bench /= qqq_bench.iloc[0]

# ─────────────────────────────────────────────────────────────
# YEARLY
# ─────────────────────────────────────────────────────────────
print("\n" + "="*160)
print("YEARLY PERFORMANCE  (Best Parameters)")
print("="*160)
rows = []
for yr in sorted(bs_s.index.year.unique()):
    ys = bs_s[bs_s.index.year == yr]
    yw = bs_w[bs_w.index.year == yr]
    if len(ys) == 0: continue
    tr = (1+ys).prod()-1
    sh = ys.mean()/ys.std()*np.sqrt(thr_ann) if ys.std()>0 else 0
    dd = (yw/yw.cummax()-1).min()
    rows.append(dict(Year=yr, Return=f"{tr:.2%}", Sharpe=f"{sh:.3f}",
                     MaxDD=f"{dd:.2%}", Trades=len(ys)))
print(pd.DataFrame(rows).to_string(index=False))

# ─────────────────────────────────────────────────────────────
# PLOTS  (3x3 grid)
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(22, 16))
fig.suptitle("QQQ Bubble + Long Momentum — Hourly Grid Search Results", fontsize=14, fontweight="bold")

# 1. Wealth curve
ax = axes[0, 0]
ax.plot(bs_w.index, bs_w.values, lw=2, label=f"Strategy (best)", color="steelblue")
ax.plot(qqq_bench.index, qqq_bench.values, lw=1.5, ls="--", label="QQQ Buy&Hold", color="gray")
ax.set_yscale("log"); ax.set_title("Wealth Curve (Best Parameters)", fontweight="bold")
ax.set_ylabel("Wealth (log)"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# 2. Bubble score timeline
ax = axes[0, 1]
bfull = pd.Series(bub_cache[maw_b], index=idx)
ax.plot(bfull.index, bfull.values, lw=0.7, color="navy", alpha=0.7)
ax.axhline(thr_b, color="red", ls="--", lw=1.5, label=f"Threshold {thr_b}")
ax.axhline(0, color="black", lw=0.5)
ax.set_ylim(-1, 1); ax.set_title(f"QQQ Bubble Score (MA={maw_b}h)", fontweight="bold")
ax.set_ylabel("Bubble Score"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# 3. Sharpe distribution
ax = axes[0, 2]
pos = df[df.sharpe > 0]["sharpe"]
ax.hist(df["sharpe"].clip(-5, 5), bins=60, color="steelblue", alpha=0.7, edgecolor="none")
ax.axvline(0, color="red", lw=1.5)
ax.set_title(f"Sharpe Distribution (all {len(df)} combos)", fontweight="bold")
ax.set_xlabel("Sharpe Ratio"); ax.set_ylabel("Count"); ax.grid(True, alpha=0.3)

# 4. Sharpe vs bubble threshold (each MA as separate line)
ax = axes[1, 0]
for maw_v in bubble_ma_list:
    sub = df[df.ma_window == maw_v].groupby("bubble_thr")["sharpe"].mean()
    ax.plot(sub.index, sub.values, marker="o", lw=2, label=f"MA={maw_v}h")
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Avg Sharpe vs Bubble Threshold", fontweight="bold")
ax.set_xlabel("Bubble Threshold"); ax.set_ylabel("Sharpe"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 5. Sharpe vs hold period
ax = axes[1, 1]
s_hold = df.groupby("hold_h")["sharpe"].agg(["mean", "max"]).reset_index()
ax.plot(s_hold["hold_h"], s_hold["mean"], marker="o", lw=2, label="Avg Sharpe", color="darkorange")
ax.plot(s_hold["hold_h"], s_hold["max"],  marker="s", lw=2, ls="--", label="Best Sharpe", color="green")
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Sharpe vs Hold Period (hours)", fontweight="bold")
ax.set_xlabel("Hold Period (h)"); ax.set_ylabel("Sharpe"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# 6. Sharpe vs momentum lookback
ax = axes[1, 2]
s_mom = df.groupby("mom_lb")["sharpe"].agg(["mean", "max"]).reset_index()
ax.plot(s_mom["mom_lb"], s_mom["mean"], marker="o", lw=2, label="Avg Sharpe", color="purple")
ax.plot(s_mom["mom_lb"], s_mom["max"],  marker="s", lw=2, ls="--", label="Best Sharpe", color="green")
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Sharpe vs Momentum Lookback (hours)", fontweight="bold")
ax.set_xlabel("Momentum Lookback (h)"); ax.set_ylabel("Sharpe"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# 7. Heatmap: bubble_thr vs hold_h (avg Sharpe)
ax = axes[2, 0]
piv = df.groupby(["bubble_thr","hold_h"])["sharpe"].mean().unstack("hold_h")
piv_clipped = piv.clip(-3, 3)
im = ax.imshow(piv_clipped.values, aspect="auto", cmap="RdYlGn", vmin=-2, vmax=2)
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=45, fontsize=7)
ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels(piv.index, fontsize=8)
ax.set_title("Heatmap: Bubble Thr vs Hold Period\n(avg Sharpe)", fontweight="bold")
ax.set_xlabel("Hold Period (h)"); ax.set_ylabel("Bubble Threshold")
plt.colorbar(im, ax=ax, shrink=0.8)

# 8. Heatmap: mom_lb vs hold_h (avg Sharpe)
ax = axes[2, 1]
piv2 = df.groupby(["mom_lb","hold_h"])["sharpe"].mean().unstack("hold_h")
piv2_clipped = piv2.clip(-3, 3)
im2 = ax.imshow(piv2_clipped.values, aspect="auto", cmap="RdYlGn", vmin=-2, vmax=2)
ax.set_xticks(range(len(piv2.columns))); ax.set_xticklabels(piv2.columns, rotation=45, fontsize=7)
ax.set_yticks(range(len(piv2.index)));   ax.set_yticklabels(piv2.index, fontsize=8)
ax.set_title("Heatmap: Mom Lookback vs Hold Period\n(avg Sharpe)", fontweight="bold")
ax.set_xlabel("Hold Period (h)"); ax.set_ylabel("Momentum Lookback (h)")
plt.colorbar(im2, ax=ax, shrink=0.8)

# 9. Yearly returns bar
ax = axes[2, 2]
if rows:
    ydf  = pd.DataFrame(rows)
    yr_r = [float(r.strip("%"))/100 for r in ydf["Return"]]
    clrs = ["green" if r >= 0 else "red" for r in yr_r]
    ax.bar(ydf["Year"].astype(str), [r*100 for r in yr_r], color=clrs, alpha=0.7)
    ax.axhline(0, color="black", lw=0.8)
ax.set_title("Yearly Returns — Best Strategy (%)", fontweight="bold")
ax.set_ylabel("Return (%)"); ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/qqq_bubble_hourly_v3.png", dpi=150, bbox_inches="tight")
print("\nSaved: results/qqq_bubble_hourly_v3.png")

df.to_csv("results/qqq_bubble_hourly_grid_v3_new.csv", index=False)
print("Saved: results/qqq_bubble_hourly_grid_v3.csv")
print(f"\nTotal runtime: {time.time()-t0:.1f}s")
print("DONE")
