"""
QQQ BUBBLE + LONG MOMENTUM STOCKS — HOURLY GRID SEARCH
=======================================================
Bubble Score: calculated on QQQ hourly price using exact function
Signal:       when bubble score is LOW → buy top momentum stocks
Hold Period:  couple of hours to 1 day (grid search)
Data:         QQQ + 516 stocks, 2020-07-27 to 2026-06-02 (5.85 years)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings("ignore")
from itertools import product

# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────
print("Loading data...")
qqq_raw   = pd.read_parquet("data/cache/qqq_hourly_close.parquet")["QQQ"]
stocks_raw = pd.read_parquet("data/cache/merged_hourly_close.parquet")

# Align on common timestamps
idx = qqq_raw.index.intersection(stocks_raw.index)
qqq    = qqq_raw.loc[idx]
stocks = stocks_raw.loc[idx].ffill().fillna(0)

years = (idx[-1] - idx[0]).days / 365.25
print(f"Aligned: {len(idx)} hours  |  {idx[0].date()} to {idx[-1].date()}  ({years:.2f} yrs)")

# Only keep columns with < 30% NaN across the original stocks data (before ffill)
valid_cols = stocks_raw.columns[stocks_raw.isna().mean() < 0.30]
stocks = stocks[valid_cols]
print(f"Stocks after quality filter: {len(valid_cols)} tickers")

ret_stocks = stocks.pct_change().fillna(0)
trading_hours_per_year = 252 * 6.5     # annualisation factor

# ─────────────────────────────────────────────────────────────
# BUBBLE SCORE — exact function as provided
# ─────────────────────────────────────────────────────────────
def calculate_bubble_score_proxy(price, ma_window=252, z_window=252):
    log_price      = np.log(price)
    fair_value     = price.rolling(ma_window).mean()
    log_fair_value = np.log(fair_value)
    residual       = log_price - log_fair_value
    z = (residual - residual.rolling(z_window).mean()) / residual.rolling(z_window).std()
    return np.tanh(z / 2)

# ─────────────────────────────────────────────────────────────
# GRID SEARCH PARAMETERS
# ─────────────────────────────────────────────────────────────
#
# ma_window:          how many hourly bars for "fair value" MA
#                     252h  = ~39 trading days
#                     500h  = ~77 trading days
#                     1000h = ~154 trading days
#
# bubble_threshold:   entry threshold (buy when score < this)
#                     -0.3, -0.5, -0.7
#
# mom_lookback:       hours to compute stock momentum ranking
#                     6h (1 session), 13h (~2 sessions), 26h (~4 sessions)
#
# hold_hours:         how long to hold the bought stocks
#                     4, 8, 13, 26  (half-day to ~4 sessions)
#
# top_n:              number of top-momentum stocks to buy
#                     5, 10

ma_windows         = [252, 500, 1000]
bubble_thresholds  = [-0.3, -0.5, -0.7]
mom_lookbacks      = [6, 13, 26]
hold_hours_list    = [4, 8, 13, 26]
top_n_list         = [5, 10]

total = (len(ma_windows) * len(bubble_thresholds) *
         len(mom_lookbacks) * len(hold_hours_list) * len(top_n_list))
print(f"\nGrid: {total} combinations\n")
print(f"  ma_windows:        {ma_windows}")
print(f"  bubble_thresholds: {bubble_thresholds}")
print(f"  mom_lookbacks:     {mom_lookbacks}")
print(f"  hold_hours:        {hold_hours_list}")
print(f"  top_n:             {top_n_list}")

# ─────────────────────────────────────────────────────────────
# PRE-COMPUTE BUBBLE SCORES (once per ma_window)
# ─────────────────────────────────────────────────────────────
bubble_cache = {}
for maw in ma_windows:
    b = calculate_bubble_score_proxy(qqq, ma_window=maw, z_window=maw).fillna(0)
    bubble_cache[maw] = b
    print(f"  Bubble (ma={maw:4d}h): min={b.min():.3f}  max={b.max():.3f}  "
          f"pct<-0.3: {(b<-0.3).mean()*100:.1f}%  "
          f"pct<-0.5: {(b<-0.5).mean()*100:.1f}%  "
          f"pct<-0.7: {(b<-0.7).mean()*100:.1f}%")

# ─────────────────────────────────────────────────────────────
# GRID SEARCH
# ─────────────────────────────────────────────────────────────
print("\nRunning grid search...")

results = []
n = len(idx)

for maw, thr, momlb, hold, topn in product(
        ma_windows, bubble_thresholds, mom_lookbacks, hold_hours_list, top_n_list):

    bubble   = bubble_cache[maw]
    # momentum lookback return for each stock
    mom_ret  = stocks.pct_change(momlb).fillna(0)

    # warmup: need enough bars for both bubble and momentum
    warmup   = max(maw * 2, momlb) + 1

    strat_ret = []

    i = warmup
    while i < n - hold:
        bs = bubble.iloc[i]
        if bs < thr:
            # rank stocks by mom_lookback return at bar i, pick top topn
            row_mom   = mom_ret.iloc[i]
            top_cols  = row_mom.nlargest(topn).index

            # average return of those stocks over next hold bars
            future    = ret_stocks.iloc[i+1 : i+1+hold][top_cols].values.mean()
            strat_ret.append(np.clip(future - 0.001, -0.10, 0.10))  # 0.1% round-trip cost
            i += hold    # non-overlapping: jump forward by hold
        else:
            strat_ret.append(0.0)
            i += 1

    if len(strat_ret) < 50:
        continue

    s  = pd.Series(strat_ret)
    w  = (1 + s).cumprod()
    tr = w.iloc[-1] - 1
    ar = (1 + tr) ** (1 / years) - 1 if tr > -1 else -1.0
    sh = s.mean() / s.std() * np.sqrt(trading_hours_per_year) if s.std() > 0 else 0
    dd = (w / w.cummax() - 1).min()
    wr = (s > 0).mean() * 100
    n_trades = (s != 0).sum()

    results.append(dict(
        ma_window         = maw,
        bubble_threshold  = thr,
        mom_lookback      = momlb,
        hold_hours        = hold,
        top_n             = topn,
        total_return      = tr,
        annual_return     = ar,
        sharpe            = sh,
        max_drawdown      = dd,
        win_rate          = wr,
        n_trades          = n_trades,
    ))

df = pd.DataFrame(results).sort_values("sharpe", ascending=False)

# ─────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*130)
print("TOP 20 COMBINATIONS  (sorted by Sharpe)")
print("="*130)
cols_display = ["ma_window","bubble_threshold","mom_lookback","hold_hours","top_n",
                "annual_return","sharpe","max_drawdown","win_rate","n_trades"]
print(df.head(20)[cols_display].to_string(index=False,
      formatters={
          "annual_return":  "{:.2%}".format,
          "sharpe":         "{:.4f}".format,
          "max_drawdown":   "{:.2%}".format,
          "win_rate":       "{:.1f}%".format,
      }))

best = df.iloc[0]
print("\n" + "="*130)
print("BEST PARAMETERS")
print("="*130)
print(f"  MA Window:         {int(best.ma_window)}h  (~{best.ma_window/6.5:.0f} trading days)")
print(f"  Bubble Threshold:  {best.bubble_threshold}")
print(f"  Momentum Lookback: {int(best.mom_lookback)}h  (~{best.mom_lookback/6.5:.1f} sessions)")
print(f"  Hold Period:       {int(best.hold_hours)}h  (~{best.hold_hours/6.5:.1f} sessions)")
print(f"  Top N Stocks:      {int(best.top_n)}")
print()
print(f"  Annual Return:     {best.annual_return:.2%}")
print(f"  Sharpe Ratio:      {best.sharpe:.4f}")
print(f"  Max Drawdown:      {best.max_drawdown:.2%}")
print(f"  Win Rate:          {best.win_rate:.1f}%")
print(f"  Total Trades:      {int(best.n_trades)}")

# ─────────────────────────────────────────────────────────────
# BEST COMBO — run full series for plotting
# ─────────────────────────────────────────────────────────────
maw   = int(best.ma_window)
thr   = best.bubble_threshold
momlb = int(best.mom_lookback)
hold  = int(best.hold_hours)
topn  = int(best.top_n)

bubble  = bubble_cache[maw]
mom_ret = stocks.pct_change(momlb).fillna(0)
warmup  = max(maw * 2, momlb) + 1

best_series  = []
best_idx     = []
bubble_ts    = []

i = warmup
while i < n - hold:
    bs = bubble.iloc[i]
    bubble_ts.append((idx[i], bs))
    if bs < thr:
        row_mom  = mom_ret.iloc[i]
        top_cols = row_mom.nlargest(topn).index
        future   = ret_stocks.iloc[i+1 : i+1+hold][top_cols].values.mean()
        r = np.clip(future - 0.001, -0.10, 0.10)
        best_series.append(r)
        best_idx.append(idx[i+1])
        i += hold
    else:
        best_series.append(0.0)
        best_idx.append(idx[i])
        i += 1

bs_series = pd.Series(best_series, index=best_idx)
bs_wealth  = (1 + bs_series).cumprod()
bub_df     = pd.DataFrame(bubble_ts, columns=["ts","bubble"]).set_index("ts")

# QQQ buy & hold for comparison
qqq_ret    = qqq.pct_change().fillna(0)
qqq_wealth = (1 + qqq_ret.loc[bs_wealth.index[0]:]).cumprod()
qqq_wealth = qqq_wealth / qqq_wealth.iloc[0]

# ─────────────────────────────────────────────────────────────
# YEARLY BREAKDOWN (best params)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*130)
print("YEARLY PERFORMANCE (Best Parameters)")
print("="*130)
yearly = []
for yr in sorted(bs_series.index.year.unique()):
    yr_ret = bs_series[bs_series.index.year == yr]
    yr_w   = bs_wealth[bs_wealth.index.year == yr]
    if len(yr_ret) == 0:
        continue
    tr  = (1 + yr_ret).prod() - 1
    sh  = yr_ret.mean() / yr_ret.std() * np.sqrt(trading_hours_per_year) if yr_ret.std() > 0 else 0
    mdd = (yr_w / yr_w.cummax() - 1).min()
    nt  = (yr_ret != 0).sum()
    yearly.append(dict(Year=yr, Return=tr, Sharpe=sh, MaxDD=mdd, Trades=nt))

yr_df = pd.DataFrame(yearly)
print(yr_df.to_string(index=False,
      formatters={"Return":"{:.2%}".format,"Sharpe":"{:.3f}".format,"MaxDD":"{:.2%}".format}))

# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(18, 15))

# 1. Wealth curve
axes[0,0].plot(bs_wealth.index, bs_wealth.values, lw=2, label="QQQ Bubble + Long Mom", color="steelblue")
axes[0,0].plot(qqq_wealth.index, qqq_wealth.values, lw=1.5, ls="--", label="QQQ Buy & Hold", color="gray")
axes[0,0].set_yscale("log")
axes[0,0].set_title(f"Wealth — Best Strategy (Bubble<{thr}, Hold={hold}h, TopN={topn})", fontweight="bold")
axes[0,0].set_ylabel("Wealth (log)")
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. Bubble score over time
axes[0,1].plot(bub_df.index, bub_df["bubble"].values, lw=1, color="navy", alpha=0.7)
axes[0,1].axhline(thr, color="red", ls="--", lw=1.5, label=f"Threshold {thr}")
axes[0,1].axhline(0, color="black", ls="-", lw=0.5)
axes[0,1].set_ylim(-1, 1)
axes[0,1].set_title("QQQ Bubble Score (hourly)", fontweight="bold")
axes[0,1].set_ylabel("Bubble Score")
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 3. Sharpe by bubble threshold
s_by_thr = df.groupby("bubble_threshold")["sharpe"].mean()
axes[1,0].bar(s_by_thr.index.astype(str), s_by_thr.values, color="steelblue", alpha=0.7)
axes[1,0].axhline(0, color="black", lw=0.8)
axes[1,0].set_title("Avg Sharpe by Bubble Threshold", fontweight="bold")
axes[1,0].set_xlabel("Bubble Threshold")
axes[1,0].set_ylabel("Sharpe Ratio")
axes[1,0].grid(True, alpha=0.3, axis="y")

# 4. Sharpe by hold period
s_by_hold = df.groupby("hold_hours")["sharpe"].mean()
axes[1,1].bar(s_by_hold.index.astype(str), s_by_hold.values, color="darkorange", alpha=0.7)
axes[1,1].axhline(0, color="black", lw=0.8)
axes[1,1].set_title("Avg Sharpe by Hold Period (hours)", fontweight="bold")
axes[1,1].set_xlabel("Hold Period (h)")
axes[1,1].set_ylabel("Sharpe Ratio")
axes[1,1].grid(True, alpha=0.3, axis="y")

# 5. Sharpe by momentum lookback
s_by_mom = df.groupby("mom_lookback")["sharpe"].mean()
axes[2,0].bar(s_by_mom.index.astype(str), s_by_mom.values, color="green", alpha=0.7)
axes[2,0].axhline(0, color="black", lw=0.8)
axes[2,0].set_title("Avg Sharpe by Momentum Lookback (hours)", fontweight="bold")
axes[2,0].set_xlabel("Momentum Lookback (h)")
axes[2,0].set_ylabel("Sharpe Ratio")
axes[2,0].grid(True, alpha=0.3, axis="y")

# 6. Yearly returns bar
colors = ["green" if r > 0 else "red" for r in yr_df["Return"]]
axes[2,1].bar(yr_df["Year"].astype(str), yr_df["Return"]*100, color=colors, alpha=0.7)
axes[2,1].axhline(0, color="black", lw=0.8)
axes[2,1].set_title("Yearly Returns — Best Strategy (%)", fontweight="bold")
axes[2,1].set_ylabel("Return (%)")
axes[2,1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/qqq_bubble_hourly_momentum.png", dpi=150, bbox_inches="tight")
print("\nSaved: results/qqq_bubble_hourly_momentum.png")

df.to_csv("results/qqq_bubble_hourly_grid_results.csv", index=False)
print("Saved: results/qqq_bubble_hourly_grid_results.csv")

print("\nDONE")
