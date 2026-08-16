"""
QQQ BUBBLE + LONG MOMENTUM STOCKS — HOURLY EXTENDED GRID SEARCH
================================================================
Strategy:  When QQQ bubble score is LOW → buy top-N momentum stocks
Bubble:    calculate_bubble_score_proxy (exact function as given)
Data:      2020-07-27 to 2026-06-02  (~5.8 years)
Fix:       Normalize timestamps to hour-floor so QQQ (:00) and
           stocks (:30) align on the same hourly bar.

Grid (expanded):
  Bubble MA windows:     3  values  [252h, 500h, 1000h]
  Bubble thresholds:     8  values  [-0.1 .. -0.8]
  Momentum lookbacks:    8  values  [3h, 6h, 10h, 13h, 20h, 26h, 40h, 52h]
  Hold periods:         10  values  [2h, 4h, 6h, 8h, 13h, 20h, 26h, 40h, 52h, 65h]
  Top-N stocks:          3  values  [5, 10, 20]

Total = 3 x 8 x 8 x 10 x 3 = 5,760 combinations
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
# LOAD + ALIGN DATA
# ─────────────────────────────────────────────────────────────
print("Loading data...")
qqq_raw    = pd.read_parquet("data/cache/qqq_hourly_close.parquet")["QQQ"]
stocks_raw = pd.read_parquet("data/cache/merged_hourly_close.parquet")

# Normalize timestamps to hour-floor so :00 and :30 bars merge
qqq_h    = qqq_raw.copy();    qqq_h.index    = qqq_h.index.floor("h")
stocks_h = stocks_raw.copy(); stocks_h.index = stocks_h.index.floor("h")

# Deduplicate (keep last) after flooring
qqq_h    = qqq_h[~qqq_h.index.duplicated(keep="last")]
stocks_h = stocks_h[~stocks_h.index.duplicated(keep="last")]

# Intersect on common hourly timestamps
idx = qqq_h.index.intersection(stocks_h.index)
qqq    = qqq_h.loc[idx]
stocks = stocks_h.loc[idx]

# Drop columns with >30% NaN, then forward-fill remaining gaps
valid_cols = stocks.columns[stocks.isna().mean() < 0.30]
stocks = stocks[valid_cols].ffill().fillna(0)

years = (idx[-1] - idx[0]).days / 365.25
print(f"Aligned: {len(idx)} hourly bars  |  {idx[0].date()} to {idx[-1].date()}  ({years:.2f} yrs)")
print(f"Stocks after quality filter: {len(valid_cols)} tickers")

ret_stocks = stocks.pct_change().fillna(0)
ret_stocks_np = ret_stocks.values           # numpy for speed
stocks_np     = stocks.values
trading_hours_per_year = 252 * 6.5

# ─────────────────────────────────────────────────────────────
# BUBBLE SCORE (exact function as provided)
# ─────────────────────────────────────────────────────────────
def calculate_bubble_score_proxy(price, ma_window=252, z_window=252):
    log_price      = np.log(price)
    fair_value     = price.rolling(ma_window).mean()
    log_fair_value = np.log(fair_value)
    residual       = log_price - log_fair_value
    z = (residual - residual.rolling(z_window).mean()) / residual.rolling(z_window).std()
    return np.tanh(z / 2)

# ─────────────────────────────────────────────────────────────
# GRIDS
# ─────────────────────────────────────────────────────────────
# Bubble grid  —  MA window and entry threshold
bubble_ma_windows  = [252, 500, 1000]
bubble_thresholds  = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]

# Momentum grid  —  lookback in hours (independent from bubble grid)
mom_lookbacks      = [3, 6, 10, 13, 20, 26, 40, 52]

# Hold period grid  —  hours to hold after entry
hold_hours_list    = [2, 4, 6, 8, 13, 20, 26, 40, 52, 65]

# Position size grid
top_n_list         = [5, 10, 20]

total = (len(bubble_ma_windows) * len(bubble_thresholds) *
         len(mom_lookbacks) * len(hold_hours_list) * len(top_n_list))

print(f"\nGrid:")
print(f"  Bubble MA windows  (h): {bubble_ma_windows}    ({len(bubble_ma_windows)} values)")
print(f"  Bubble thresholds:      {bubble_thresholds}  ({len(bubble_thresholds)} values)")
print(f"  Momentum lookbacks (h): {mom_lookbacks}  ({len(mom_lookbacks)} values)")
print(f"  Hold periods       (h): {hold_hours_list}  ({len(hold_hours_list)} values)")
print(f"  Top-N stocks:           {top_n_list}  ({len(top_n_list)} values)")
print(f"  Total combinations:     {total}")

# ─────────────────────────────────────────────────────────────
# PRE-COMPUTE BUBBLE SCORES AND MOMENTUM RETURNS (numpy)
# ─────────────────────────────────────────────────────────────
print("\nPre-computing bubble scores...")
bubble_cache = {}
for maw in bubble_ma_windows:
    b = calculate_bubble_score_proxy(qqq, ma_window=maw, z_window=maw).fillna(0)
    bubble_cache[maw] = b.values
    pcts = [(b < t).mean()*100 for t in [-0.3, -0.5, -0.7]]
    print(f"  ma={maw:4d}h  min={b.min():.3f}  max={b.max():.3f}  "
          f"pct<-0.3:{pcts[0]:.1f}%  pct<-0.5:{pcts[1]:.1f}%  pct<-0.7:{pcts[2]:.1f}%")

print("Pre-computing momentum return arrays...")
mom_cache = {}
for lb in mom_lookbacks:
    m = stocks.pct_change(lb).fillna(0).values   # shape (T, N_stocks)
    mom_cache[lb] = m
    print(f"  lookback={lb:3d}h  done")

n        = len(idx)
col_arr  = np.arange(stocks.shape[1])   # column indices

# ─────────────────────────────────────────────────────────────
# GRID SEARCH  (non-overlapping trades)
# ─────────────────────────────────────────────────────────────
print(f"\nRunning grid search ({total} combinations)...")
results = []
combo_n = 0

for maw, thr, momlb, hold, topn in product(
        bubble_ma_windows, bubble_thresholds, mom_lookbacks, hold_hours_list, top_n_list):

    combo_n += 1
    if combo_n % 500 == 0:
        print(f"  {combo_n}/{total}...")

    bubble_arr = bubble_cache[maw]
    mom_arr    = mom_cache[momlb]
    warmup     = max(maw, momlb) + 5

    strat_ret  = []
    i = warmup
    while i < n - hold:
        if bubble_arr[i] < thr:
            # Top-N stocks by momentum at bar i
            top_idx = np.argpartition(mom_arr[i], -topn)[-topn:]
            # Average return over next `hold` bars
            future_ret = ret_stocks_np[i+1:i+1+hold, :][:, top_idx].mean()
            strat_ret.append(np.clip(future_ret - 0.001, -0.10, 0.10))
            i += hold          # non-overlapping: skip forward
        else:
            i += 1

    if len(strat_ret) < 20:
        results.append(dict(
            ma_window=maw, bubble_threshold=thr, mom_lookback=momlb,
            hold_hours=hold, top_n=topn,
            annual_return=np.nan, sharpe=np.nan, max_drawdown=np.nan,
            win_rate=np.nan, n_trades=0, total_return=np.nan))
        continue

    s  = np.array(strat_ret)
    w  = np.cumprod(1 + s)
    tr = w[-1] - 1
    ar = (1 + tr) ** (1/years) - 1 if tr > -1 else -1.0
    sh = s.mean() / s.std() * np.sqrt(trading_hours_per_year) if s.std() > 0 else 0
    dd = (w / np.maximum.accumulate(w) - 1).min()
    wr = (s > 0).mean() * 100

    results.append(dict(
        ma_window=maw, bubble_threshold=thr, mom_lookback=momlb,
        hold_hours=hold, top_n=topn,
        total_return=tr, annual_return=ar, sharpe=sh,
        max_drawdown=dd, win_rate=wr, n_trades=len(s)))

# ─────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────
df = pd.DataFrame(results).dropna(subset=["sharpe"])
df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)

print("\n" + "="*150)
print("TOP 30 COMBINATIONS  (sorted by Sharpe Ratio)")
print("="*150)
fmt = {
    "annual_return": "{:.2%}".format,
    "sharpe":        "{:.4f}".format,
    "max_drawdown":  "{:.2%}".format,
    "win_rate":      "{:.1f}".format,
}
cols = ["ma_window","bubble_threshold","mom_lookback","hold_hours","top_n",
        "annual_return","sharpe","max_drawdown","win_rate","n_trades"]
print(df.head(30)[cols].to_string(index=False, formatters=fmt))

# ─────────────────────────────────────────────────────────────
# BEST COMBO — full wealth series
# ─────────────────────────────────────────────────────────────
best = df.iloc[0]
maw   = int(best.ma_window)
thr   = best.bubble_threshold
momlb = int(best.mom_lookback)
hold  = int(best.hold_hours)
topn  = int(best.top_n)

print("\n" + "="*150)
print("BEST PARAMETERS")
print("="*150)
print(f"  MA Window:         {maw}h  (~{maw/6.5:.0f} trading days)")
print(f"  Bubble Threshold:  {thr}")
print(f"  Momentum Lookback: {momlb}h  (~{momlb/6.5:.1f} sessions)")
print(f"  Hold Period:       {hold}h  (~{hold/6.5:.1f} sessions = ~{hold/6.5/5:.1f} weeks)")
print(f"  Top N Stocks:      {topn}")
print(f"\n  Annual Return:     {best.annual_return:.2%}")
print(f"  Sharpe Ratio:      {best.sharpe:.4f}")
print(f"  Max Drawdown:      {best.max_drawdown:.2%}")
print(f"  Win Rate:          {best.win_rate:.1f}%")
print(f"  Total Trades:      {int(best.n_trades)}")

# Rebuild best series for plots and yearly breakdown
bubble_arr = bubble_cache[maw]
mom_arr    = mom_cache[momlb]
warmup     = max(maw, momlb) + 5

best_rets, best_ts = [], []
i = warmup
while i < n - hold:
    if bubble_arr[i] < thr:
        top_idx    = np.argpartition(mom_arr[i], -topn)[-topn:]
        future_ret = ret_stocks_np[i+1:i+1+hold, :][:, top_idx].mean()
        r = np.clip(future_ret - 0.001, -0.10, 0.10)
        best_rets.append(r)
        best_ts.append(idx[i])
        i += hold
    else:
        i += 1

bs_series = pd.Series(best_rets, index=pd.DatetimeIndex(best_ts))
bs_wealth  = (1 + bs_series).cumprod()

# QQQ buy & hold benchmark
qqq_ret    = qqq.pct_change().fillna(0).loc[bs_wealth.index[0]:]
qqq_wealth = (1 + qqq_ret).cumprod()
qqq_wealth = qqq_wealth / qqq_wealth.iloc[0]

# ─────────────────────────────────────────────────────────────
# YEARLY BREAKDOWN
# ─────────────────────────────────────────────────────────────
print("\n" + "="*150)
print("YEARLY PERFORMANCE  (Best Parameters)")
print("="*150)
yearly = []
for yr in sorted(bs_series.index.year.unique()):
    yr_s = bs_series[bs_series.index.year == yr]
    yr_w = bs_wealth[bs_wealth.index.year == yr]
    if len(yr_s) == 0: continue
    tr  = (1 + yr_s).prod() - 1
    sh  = yr_s.mean() / yr_s.std() * np.sqrt(trading_hours_per_year) if yr_s.std() > 0 else 0
    mdd = (yr_w / yr_w.cummax() - 1).min()
    yearly.append(dict(Year=yr, Return=f"{tr:.2%}", Sharpe=f"{sh:.3f}",
                        MaxDD=f"{mdd:.2%}", Trades=len(yr_s)))
print(pd.DataFrame(yearly).to_string(index=False))

# Summary stats by dimension
print("\n" + "="*150)
print("PARAMETER SENSITIVITY")
print("="*150)
for dim, label in [("bubble_threshold","Bubble Threshold"),
                   ("ma_window","MA Window (h)"),
                   ("mom_lookback","Momentum Lookback (h)"),
                   ("hold_hours","Hold Period (h)"),
                   ("top_n","Top-N")]:
    grp = df.groupby(dim)["sharpe"].agg(["mean","max","min"]).reset_index()
    grp.columns = [label,"Avg Sharpe","Best Sharpe","Worst Sharpe"]
    print(f"\n{grp.to_string(index=False)}")

# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(18, 15))

# 1. Wealth
axes[0,0].plot(bs_wealth.index, bs_wealth.values, lw=2,
               label=f"Strategy (thr={thr},hold={hold}h,top{topn})", color="steelblue")
axes[0,0].plot(qqq_wealth.index, qqq_wealth.values, lw=1.5, ls="--",
               label="QQQ Buy&Hold", color="gray")
axes[0,0].set_yscale("log")
axes[0,0].set_title("Wealth Curve — Best Parameters", fontweight="bold")
axes[0,0].set_ylabel("Wealth (log)")
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

# 2. Bubble score
bub_full = pd.Series(bubble_cache[maw], index=idx)
axes[0,1].plot(bub_full.index, bub_full.values, lw=0.8, color="navy", alpha=0.7)
axes[0,1].axhline(thr, color="red", ls="--", lw=1.5, label=f"Threshold {thr}")
axes[0,1].axhline(0, color="black", ls="-", lw=0.5)
axes[0,1].set_ylim(-1, 1)
axes[0,1].set_title(f"QQQ Bubble Score (MA={maw}h)", fontweight="bold")
axes[0,1].set_ylabel("Bubble Score")
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

# 3. Sharpe vs bubble threshold  (heatmap-style: each MA window as line)
for maw_v in bubble_ma_windows:
    sub = df[df.ma_window == maw_v].groupby("bubble_threshold")["sharpe"].mean()
    axes[1,0].plot(sub.index, sub.values, marker="o", lw=2, label=f"MA={maw_v}h")
axes[1,0].axhline(0, color="black", lw=0.8)
axes[1,0].set_title("Avg Sharpe vs Bubble Threshold", fontweight="bold")
axes[1,0].set_xlabel("Bubble Threshold")
axes[1,0].set_ylabel("Sharpe Ratio")
axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

# 4. Sharpe vs hold period
s_by_hold = df.groupby("hold_hours")["sharpe"].mean()
axes[1,1].plot(s_by_hold.index, s_by_hold.values, marker="o", lw=2, color="darkorange")
axes[1,1].axhline(0, color="black", lw=0.8)
axes[1,1].set_title("Avg Sharpe vs Hold Period (hours)", fontweight="bold")
axes[1,1].set_xlabel("Hold Period (hours)")
axes[1,1].set_ylabel("Sharpe Ratio")
axes[1,1].grid(True, alpha=0.3)

# 5. Sharpe vs momentum lookback
s_by_mom = df.groupby("mom_lookback")["sharpe"].mean()
axes[2,0].plot(s_by_mom.index, s_by_mom.values, marker="o", lw=2, color="green")
axes[2,0].axhline(0, color="black", lw=0.8)
axes[2,0].set_title("Avg Sharpe vs Momentum Lookback (hours)", fontweight="bold")
axes[2,0].set_xlabel("Momentum Lookback (hours)")
axes[2,0].set_ylabel("Sharpe Ratio")
axes[2,0].grid(True, alpha=0.3)

# 6. Yearly returns bar
if yearly:
    yr_df  = pd.DataFrame(yearly)
    yr_ret = [float(r.strip("%"))/100 for r in yr_df["Return"]]
    clrs   = ["green" if r > 0 else "red" for r in yr_ret]
    axes[2,1].bar(yr_df["Year"].astype(str), [r*100 for r in yr_ret], color=clrs, alpha=0.7)
    axes[2,1].axhline(0, color="black", lw=0.8)
    axes[2,1].set_title("Yearly Returns — Best Strategy (%)", fontweight="bold")
    axes[2,1].set_ylabel("Return (%)")
    axes[2,1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/qqq_bubble_hourly_momentum_v2.png", dpi=150, bbox_inches="tight")
print("\nSaved: results/qqq_bubble_hourly_momentum_v2.png")

df.to_csv("results/qqq_bubble_hourly_grid_v2.csv", index=False)
print("Saved: results/qqq_bubble_hourly_grid_v2.csv")

print(f"\nPositive Sharpe combinations: {(df.sharpe > 0).sum()}/{len(df)}")
print("DONE")
