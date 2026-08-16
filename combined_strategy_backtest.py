"""
COMBINED STRATEGY BACKTEST
===========================
Strategy A: Daily Momentum + Leverage + UVXY (2020-2026)
Strategy B: QQQ Bubble Hourly + Long Momentum (2020-2026)

Tests:
  1. Fixed allocation grid: 0/100, 10/90, ..., 100/0 (A/B split)
  2. Momentum allocation: dynamically weight by trailing 60-day Sharpe

Common period: 2020-07-27 to 2026-06-02 (5.85 years, 1471 daily bars)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings("ignore")

# ── performance formula ──────────────────────────────────────
def perf(ret, rf=0.02, tpy=252):
    if len(ret) < 2: return dict(sharpe=0, sortino=0, max_dd=0, ann_ret=0, total_ret=0)
    rf_per = rf / tpy
    exc    = ret - rf_per
    std    = ret.std()
    sh     = exc.mean() / std * np.sqrt(tpy) if std > 0 else 0
    dn     = ret[ret < 0].std(ddof=0)
    so     = exc.mean() / dn  * np.sqrt(tpy) if dn  > 0 else 0
    w      = (1 + ret).cumprod()
    dd     = (w / w.cummax() - 1).min()
    tr     = w.iloc[-1] - 1
    years  = len(ret) / tpy
    ar     = (1 + tr) ** (1 / years) - 1 if tr > -1 else -1
    return dict(sharpe=sh, sortino=so, max_dd=dd, ann_ret=ar, total_ret=tr)

# ════════════════════════════════════════════════════════════════
# STRATEGY A: Daily Momentum + Leverage + UVXY
# ════════════════════════════════════════════════════════════════
print("Running Strategy A: Daily Momentum + Leverage + UVXY...")

close_data = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
OVERLAP_START = "2020-07-27"
OVERLAP_END   = "2026-06-02"

lookback = 140; holding = 40; top = 5; trading_days = 252
close     = close_data.copy()
ret_daily = close.pct_change().ffill().fillna(0)
ret_mom   = close.pct_change(lookback).ffill().fillna(0)

# ─ generate momentum daily returns (full history for bubble warmup)
strat_returns_A = []
for i in range(lookback + 1, len(ret_mom), holding):
    ranking  = ret_mom.iloc[i-1:i].rank(axis=1, ascending=False)
    ranked   = np.argsort(ranking.values[0])
    short_n  = int(ret_mom.iloc[:,ranked[:top]].iloc[i-1:i].lt(0).sum().sum())
    long_n   = top - short_n
    if long_n <= 0: continue
    idx_end  = min(i + holding, len(ret_mom))
    for j in range(i, idx_end):
        date = ret_daily.index[j]
        lret = short_ret = 0
        if long_n > 0:
            ls   = np.sign(ret_mom.iloc[:,ranked[:long_n]].iloc[i-1:i]).abs()
            lr   = ls.mul(np.array(ret_daily.iloc[:,ranked[:long_n]].iloc[j:j+1])[0])
            lret = lr.values.mean() * long_n
        if short_n > 0:
            ss   = np.sign(ret_mom.iloc[:,ranked[-short_n:]].iloc[i-1:i]).abs() * -1
            sr   = ss.mul(np.array(ret_daily.iloc[:,ranked[-short_n:]].iloc[j:j+1])[0])
            short_ret = sr.values.mean() * short_n
        mom_ret = (lret + short_ret) / top - 0.005 / holding
        # hedge return
        h_ret = 0.0
        if "UVXY" in close.columns and pd.notna(close.loc[date,"UVXY"]):
            h_ret = ret_daily.loc[date,"UVXY"]
        elif "^VIX" in close.columns and pd.notna(close.loc[date,"^VIX"]):
            vix_r = ret_daily.loc[date,"^VIX"]
            h_ret = (2.0*vix_r - 0.002 - 0.25*vix_r**2 if date < pd.Timestamp("2018-02-28")
                     else 1.5*vix_r - 0.0015 - 0.25*vix_r**2)
        strat_returns_A.append({"Date":date,"Momentum":mom_ret,"Hedge":h_ret})

dfA = pd.DataFrame(strat_returns_A).set_index("Date").dropna()

# ─ bubble score on momentum equity curve
def calc_bubble(price, ma=120, z=240):
    lp = np.log(np.maximum(price, 1e-6))
    f  = lp.rolling(ma).mean()
    r  = lp - f
    zs = (r - r.rolling(z).mean()) / r.rolling(z).std()
    return np.tanh(zs / 2)

base_w   = (1 + dfA).cumprod() / (1 + dfA).cumprod().iloc[0]
bubble_A = calc_bubble(base_w["Momentum"], 120, 240)

# ─ apply overlays
hedge_entry = 0.85; hedge_alloc = 0.5; hedge_hold = 40
lev_entry   = -0.88; lev_mult = 0.25;  lev_hold  = 50
lev_cost    = 0.1 / trading_days

raw_hedge_sig = bubble_A > hedge_entry
raw_lev_sig   = bubble_A < lev_entry
h_sig = raw_hedge_sig.shift(1).fillna(False)
l_sig = raw_lev_sig.shift(1).fillna(False)

daily_A = []
h_rem = l_rem = 0
for date in dfA.index:
    if h_rem == 0 and h_sig.loc[date]: h_rem = hedge_hold
    if l_rem == 0 and l_sig.loc[date]: l_rem = lev_hold
    base = dfA.loc[date,"Momentum"]
    if h_rem > 0:
        r = (1 - hedge_alloc)*base + hedge_alloc*dfA.loc[date,"Hedge"]
        h_rem -= 1
    elif l_rem > 0:
        r = base + lev_mult*base - lev_mult*lev_cost
        l_rem -= 1
    else:
        r = base
    daily_A.append(r)

series_A_full = pd.Series(daily_A, index=dfA.index)

# Trim to overlap period
series_A = series_A_full.loc[OVERLAP_START:OVERLAP_END]
years_overlap = (series_A.index[-1] - series_A.index[0]).days / 365.25
print(f"  Strategy A: {len(series_A)} daily bars  ({series_A.index[0].date()} to {series_A.index[-1].date()})")
pA = perf(series_A, tpy=252)
print(f"  Ann={pA['ann_ret']:.2%}  Sharpe={pA['sharpe']:.4f}  MaxDD={pA['max_dd']:.2%}")

# ════════════════════════════════════════════════════════════════
# STRATEGY B: QQQ Bubble Hourly + Momentum -> aggregate to daily
# ════════════════════════════════════════════════════════════════
print("\nRunning Strategy B: QQQ Bubble Hourly Momentum...")

qqq_raw    = pd.read_parquet("data/cache/qqq_hourly_close.parquet")["QQQ"]
stocks_raw = pd.read_parquet("data/cache/merged_hourly_close.parquet")

qqq_h  = qqq_raw.copy();    qqq_h.index  = qqq_h.index.floor("h")
stk_h  = stocks_raw.copy(); stk_h.index  = stk_h.index.floor("h")
qqq_h  = qqq_h[~qqq_h.index.duplicated(keep="last")]
stk_h  = stk_h[~stk_h.index.duplicated(keep="last")]
idx_h  = qqq_h.index.intersection(stk_h.index)
qqq    = qqq_h.loc[idx_h]
stocks = stk_h.loc[idx_h]

daily_cols = set(close_data.columns)
sp_cols    = [c for c in stocks.columns if c in daily_cols]
valid      = stocks[sp_cols].columns[stocks[sp_cols].isna().mean() < 0.30]
stocks     = stocks[valid].ffill()

ret_np  = stocks.pct_change().clip(-0.10, 0.10).fillna(0).values.astype(np.float32)
log_ret = np.log1p(ret_np.clip(-0.10, 0.10))
cumlog  = np.cumsum(log_ret, axis=0)
n       = len(idx_h)

def bubble_hourly(price, ma_w=500):
    lp = np.log(price); f = price.rolling(ma_w).mean()
    r  = lp - np.log(f)
    z  = (r - r.rolling(ma_w).mean()) / r.rolling(ma_w).std()
    return np.tanh(z / 2)

bub_B    = bubble_hourly(qqq, 500).fillna(0).values
mom_40   = stocks.pct_change(40).fillna(0).values.astype(np.float32)
HOLD_B   = 52; TOPN_B = 5; THR_B = -0.8
warmup_B = 500 + 40 + 5

HOLD_B_arr = np.zeros((n, stocks.shape[1]), dtype=np.float32)
HOLD_B_arr[:n-HOLD_B] = np.expm1(cumlog[HOLD_B:] - cumlog[:n-HOLD_B])

# hourly return series (0 in cash)
hourly_rets_B = np.zeros(n, dtype=np.float64)
i = warmup_B
while i < n - HOLD_B:
    if bub_B[i] < THR_B:
        top_idx = np.argpartition(mom_40[i], -TOPN_B)[-TOPN_B:]
        for j in range(i+1, i+1+HOLD_B):
            if j < n:
                hourly_rets_B[j] = float(ret_np[j, top_idx].mean())
        i += HOLD_B
    else:
        i += 1

hourly_series_B = pd.Series(hourly_rets_B, index=idx_h)

# Aggregate hourly → daily compound return
def to_daily(hourly_s):
    hourly_s.index = pd.DatetimeIndex(hourly_s.index)
    return hourly_s.groupby(hourly_s.index.date).apply(lambda x: (1+x).prod()-1)

daily_B_raw = to_daily(hourly_series_B)
daily_B_raw.index = pd.to_datetime(daily_B_raw.index)
series_B    = daily_B_raw.loc[OVERLAP_START:OVERLAP_END].reindex(series_A.index).fillna(0)

print(f"  Strategy B: {len(series_B)} daily bars  ({series_B.index[0].date()} to {series_B.index[-1].date()})")
pB = perf(series_B, tpy=252)
print(f"  Ann={pB['ann_ret']:.2%}  Sharpe={pB['sharpe']:.4f}  MaxDD={pB['max_dd']:.2%}")

# ════════════════════════════════════════════════════════════════
# COMBINED: FIXED ALLOCATION GRID
# ════════════════════════════════════════════════════════════════
print("\nRunning fixed allocation grid...")

alloc_grid = [i/10 for i in range(11)]   # 0.0, 0.1, ..., 1.0  (fraction = weight of A)
grid_results = []

for alpha in alloc_grid:
    combined = alpha * series_A + (1 - alpha) * series_B
    p = perf(combined, tpy=252)
    grid_results.append(dict(
        alpha_A=alpha, alpha_B=round(1-alpha,1),
        ann_ret=p['ann_ret'], sharpe=p['sharpe'],
        sortino=p['sortino'], max_dd=p['max_dd'],
        total_ret=p['total_ret']
    ))

grid_df = pd.DataFrame(grid_results)

print(f"\n{'='*110}")
print("FIXED ALLOCATION GRID RESULTS")
print(f"{'='*110}")
print(f"  {'A%':>5}  {'B%':>5}  {'AnnRet':>8}  {'TotalRet':>10}  {'Sharpe':>8}  {'Sortino':>9}  {'MaxDD':>8}")
print(f"  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*9}  {'-'*8}")
for _, r in grid_df.iterrows():
    marker = " <-- BEST SHARPE" if r['sharpe'] == grid_df['sharpe'].max() else ""
    marker2 = " <-- BEST RETURN" if r['ann_ret'] == grid_df['ann_ret'].max() else ""
    print(f"  {r['alpha_A']:>4.0%}  {r['alpha_B']:>4.0%}  {r['ann_ret']:>8.2%}  {r['total_ret']:>10.2%}  "
          f"{r['sharpe']:>8.4f}  {r['sortino']:>9.4f}  {r['max_dd']:>8.2%}{marker}{marker2}")

# ════════════════════════════════════════════════════════════════
# COMBINED: MOMENTUM ALLOCATION (dynamic)
# ════════════════════════════════════════════════════════════════
print("\nRunning momentum (dynamic) allocation...")

# Rolling 60-day Sharpe for each strategy → allocate proportionally
window = 60

def rolling_sharpe(s, w=60):
    rm = s.rolling(w).mean()
    rs = s.rolling(w).std()
    return (rm / rs * np.sqrt(252)).fillna(0)

sh_A = rolling_sharpe(series_A, window)
sh_B = rolling_sharpe(series_B, window)

# Weight A by positive Sharpe fraction: w_A = max(sh_A,0) / (max(sh_A,0)+max(sh_B,0))
sh_A_pos = sh_A.clip(lower=0)
sh_B_pos = sh_B.clip(lower=0)
total_sh  = sh_A_pos + sh_B_pos

# If both Sharpe <= 0, use 50/50; else proportional
w_A = pd.Series(np.where(total_sh > 0, sh_A_pos / total_sh, 0.5), index=series_A.index)
w_B = 1 - w_A

combined_mom = w_A * series_A + w_B * series_B
p_mom = perf(combined_mom, tpy=252)

print(f"\n  Avg weight A (daily):  {w_A.mean():.1%}")
print(f"  Avg weight B (hourly): {w_B.mean():.1%}")
print(f"  Ann={p_mom['ann_ret']:.2%}  Sharpe={p_mom['sharpe']:.4f}  "
      f"Sortino={p_mom['sortino']:.4f}  MaxDD={p_mom['max_dd']:.2%}")

# ════════════════════════════════════════════════════════════════
# YEARLY BREAKDOWN — best combo + momentum alloc
# ════════════════════════════════════════════════════════════════
best_alpha = grid_df.loc[grid_df['sharpe'].idxmax(), 'alpha_A']
best_combined = best_alpha * series_A + (1 - best_alpha) * series_B
best_w  = (1 + best_combined).cumprod()
w_A_full = (1 + series_A).cumprod()
w_B_full = (1 + series_B).cumprod()
w_mom    = (1 + combined_mom).cumprod()

print(f"\n{'='*110}")
print(f"YEARLY BREAKDOWN  (Best Fixed={best_alpha:.0%}A/{1-best_alpha:.0%}B  vs  Momentum Alloc)")
print(f"{'='*110}")
print(f"  {'Year':<6}  {'Strat A':>8}  {'Strat B':>8}  {'Best Fixed':>10}  {'Sharpe Fix':>11}  {'Mom Alloc':>10}  {'Sharpe Mom':>11}  {'QQQ':>8}")

qqq_daily_ret = qqq.pct_change().fillna(0)
qqq_daily = qqq_daily_ret.groupby(qqq_daily_ret.index.date).apply(lambda x: (1+x).prod()-1)
qqq_daily.index = pd.to_datetime(qqq_daily.index)
qqq_overlap = qqq_daily.loc[OVERLAP_START:OVERLAP_END].reindex(series_A.index).fillna(0)

for yr in sorted(series_A.index.year.unique()):
    yA  = series_A[series_A.index.year == yr]
    yB  = series_B[series_B.index.year == yr]
    yC  = best_combined[best_combined.index.year == yr]
    yM  = combined_mom[combined_mom.index.year == yr]
    yQ  = qqq_overlap[qqq_overlap.index.year == yr]

    rA = (1+yA).prod()-1; rB = (1+yB).prod()-1
    rC = (1+yC).prod()-1; rM = (1+yM).prod()-1; rQ = (1+yQ).prod()-1

    wC = (1+yC).cumprod(); wM = (1+yM).cumprod()
    shC = yC.mean()/yC.std()*np.sqrt(252) if yC.std()>0 else 0
    shM = yM.mean()/yM.std()*np.sqrt(252) if yM.std()>0 else 0

    print(f"  {yr:<6}  {rA:>8.2%}  {rB:>8.2%}  {rC:>10.2%}  {shC:>11.4f}  {rM:>10.2%}  {shM:>11.4f}  {rQ:>8.2%}")

print(f"\n  {'TOTAL':<6}  {pA['ann_ret']:>8.2%}  {pB['ann_ret']:>8.2%}  "
      f"{perf(best_combined)['ann_ret']:>10.2%}  {perf(best_combined)['sharpe']:>11.4f}  "
      f"{p_mom['ann_ret']:>10.2%}  {p_mom['sharpe']:>11.4f}  "
      f"{perf(qqq_overlap)['ann_ret']:>8.2%}")

# ════════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("Combined Strategy: Daily Momentum + Hourly QQQ Bubble", fontsize=13, fontweight="bold")

# 1. Wealth curves
qqq_w = (1 + qqq_overlap).cumprod()
axes[0,0].plot(w_A_full.index, w_A_full.values, lw=2, label=f"Strategy A - Daily ({pA['ann_ret']:.1%}/yr)", color="steelblue")
axes[0,0].plot(w_B_full.index, w_B_full.values, lw=2, label=f"Strategy B - Hourly ({pB['ann_ret']:.1%}/yr)", color="orange")
axes[0,0].plot(best_w.index, best_w.values, lw=2.5, label=f"Best Fixed {best_alpha:.0%}A/{1-best_alpha:.0%}B ({perf(best_combined)['ann_ret']:.1%}/yr)", color="green")
axes[0,0].plot(w_mom.index, w_mom.values, lw=2, ls="--", label=f"Momentum Alloc ({p_mom['ann_ret']:.1%}/yr)", color="purple")
axes[0,0].plot(qqq_w.index, qqq_w.values, lw=1.5, ls=":", label=f"QQQ ({perf(qqq_overlap)['ann_ret']:.1%}/yr)", color="gray")
axes[0,0].set_title("Wealth Curves (2020-2026)", fontweight="bold")
axes[0,0].set_ylabel("Wealth Multiple"); axes[0,0].legend(fontsize=8); axes[0,0].grid(True, alpha=0.3)

# 2. Sharpe vs allocation
axes[0,1].plot([f"{r['alpha_A']:.0%}" for _,r in grid_df.iterrows()],
               grid_df['sharpe'], marker="o", lw=2, color="steelblue", label="Sharpe")
axes[0,1].axhline(p_mom['sharpe'], color="purple", ls="--", lw=1.5, label=f"Momentum Alloc ({p_mom['sharpe']:.3f})")
axes[0,1].set_title("Sharpe Ratio vs Allocation (A%)", fontweight="bold")
axes[0,1].set_xlabel("% Allocated to Strategy A (Daily)")
axes[0,1].set_ylabel("Sharpe Ratio"); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

# 3. Annual return vs allocation
axes[1,0].bar([f"{r['alpha_A']:.0%}" for _,r in grid_df.iterrows()],
              grid_df['ann_ret']*100, color="steelblue", alpha=0.7)
axes[1,0].axhline(p_mom['ann_ret']*100, color="purple", ls="--", lw=1.5, label=f"Momentum Alloc ({p_mom['ann_ret']:.1%})")
axes[1,0].set_title("Annual Return vs Allocation (A%)", fontweight="bold")
axes[1,0].set_xlabel("% Allocated to Strategy A (Daily)")
axes[1,0].set_ylabel("Annual Return (%)"); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3, axis="y")

# 4. MaxDD vs allocation
axes[1,1].bar([f"{r['alpha_A']:.0%}" for _,r in grid_df.iterrows()],
              grid_df['max_dd']*100, color="darkred", alpha=0.7)
axes[1,1].axhline(p_mom['max_dd']*100, color="purple", ls="--", lw=1.5, label=f"Momentum Alloc ({p_mom['max_dd']:.1%})")
axes[1,1].set_title("Max Drawdown vs Allocation (A%)", fontweight="bold")
axes[1,1].set_xlabel("% Allocated to Strategy A (Daily)")
axes[1,1].set_ylabel("Max Drawdown (%)"); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/combined_strategy_backtest.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: results/combined_strategy_backtest.png")

grid_df.to_csv("results/combined_strategy_grid.csv", index=False)
print("Saved: results/combined_strategy_grid.csv")
print("\nDONE")
