"""
COMBINED PORTFOLIO BACKTEST 1997-2026
======================================
Strategies from STRATEGY_DOCUMENTATION_INDEX.md:
  A: Daily Momentum + Leverage + UVXY   (1997-2026, daily)
  B: QQQ Bubble Hourly Momentum         (2020-2026, hourly -> daily)
  C: Intraday MR + Momentum Flip        (2019-2026, hourly -> daily)

Availability timeline:
  1997-01-02 to 2018-12-31: A only
  2019-01-02 to 2020-07-26: A + C
  2020-07-27 to 2026-06-02: A + B + C

Allocation methods:
  1. Fixed Equal Weight: 1/N among available strategies
  2. Momentum Allocation: trailing 60-day Sharpe-weighted
  3. Individual strategies (A, B, C) for comparison
  4. S&P500 / QQQ buy & hold benchmarks
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, os, sys
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RF = 0.02

def perf(s, tpy=252, label=""):
    if len(s) < 20:
        return {}
    years = len(s) / tpy
    rf_d  = RF / tpy
    exc   = s - rf_d
    std   = s.std()
    sh    = exc.mean() / std * np.sqrt(tpy) if std > 0 else 0
    dn    = s[s < 0].std(ddof=0)
    so    = exc.mean() / dn  * np.sqrt(tpy) if dn  > 0 else 0
    w     = (1 + s).cumprod()
    dd    = (w / w.cummax() - 1).min()
    tr    = w.iloc[-1] - 1
    ar    = (1 + tr) ** (1 / years) - 1 if tr > -1 else -1
    wr    = (s > 0).mean()
    pos_y = sum(1 for yr in s.index.year.unique()
                if (1 + s[s.index.year == yr]).prod() - 1 > 0)
    tot_y = len(s.index.year.unique())
    return dict(label=label, ann_ret=ar, sharpe=sh, sortino=so,
                max_dd=dd, total_ret=tr, win_rate=wr,
                pos_years=f"{pos_y}/{tot_y}", years=years)

# ════════════════════════════════════════════════════════════════
# STRATEGY A: Daily Momentum + Leverage + UVXY (1997-2026)
# ════════════════════════════════════════════════════════════════
print("=" * 100)
print("Generating Strategy A: Daily Momentum + Leverage + UVXY (1997-2026)...")

close_data = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
lookback = 140; holding = 40; top = 5

close     = close_data.copy()
ret_daily = close.pct_change().ffill().fillna(0)
ret_mom   = close.pct_change(lookback).ffill().fillna(0)

strat_A_raw = []
for i in range(lookback + 1, len(ret_mom), holding):
    ranking = ret_mom.iloc[i-1:i].rank(axis=1, ascending=False)
    ranked  = np.argsort(ranking.values[0])
    short_n = int(ret_mom.iloc[:, ranked[:top]].iloc[i-1:i].lt(0).sum().sum())
    long_n  = top - short_n
    if long_n <= 0: continue
    for j in range(i, min(i + holding, len(ret_mom))):
        date = ret_daily.index[j]
        lret = sret = 0
        if long_n > 0:
            ls   = np.sign(ret_mom.iloc[:, ranked[:long_n]].iloc[i-1:i]).abs()
            lr   = ls.mul(np.array(ret_daily.iloc[:, ranked[:long_n]].iloc[j:j+1])[0])
            lret = lr.values.mean() * long_n
        if short_n > 0:
            ss   = np.sign(ret_mom.iloc[:, ranked[-short_n:]].iloc[i-1:i]).abs() * -1
            sr   = ss.mul(np.array(ret_daily.iloc[:, ranked[-short_n:]].iloc[j:j+1])[0])
            sret = sr.values.mean() * short_n
        mom_r = (lret + sret) / top - 0.005 / holding
        h_ret = 0.0
        if "UVXY" in close.columns and pd.notna(close.loc[date, "UVXY"]):
            h_ret = ret_daily.loc[date, "UVXY"]
        elif "^VIX" in close.columns and pd.notna(close.loc[date, "^VIX"]):
            vix_r = ret_daily.loc[date, "^VIX"]
            h_ret = (2.0*vix_r - 0.002 - 0.25*vix_r**2 if date < pd.Timestamp("2018-02-28")
                     else 1.5*vix_r - 0.0015 - 0.25*vix_r**2)
        strat_A_raw.append({"Date": date, "Momentum": mom_r, "Hedge": h_ret})

dfA    = pd.DataFrame(strat_A_raw).set_index("Date").dropna()
base_w = (1 + dfA).cumprod() / (1 + dfA).cumprod().iloc[0]

def calc_bubble(price, ma=120, z=240):
    lp = np.log(np.maximum(price, 1e-6))
    f  = lp.rolling(ma).mean()
    r  = lp - f
    zs = (r - r.rolling(z).mean()) / r.rolling(z).std()
    return np.tanh(zs / 2)

bub_A = calc_bubble(base_w["Momentum"], 120, 240)
h_sig = (bub_A > 0.85).shift(1).fillna(False)
l_sig = (bub_A < -0.88).shift(1).fillna(False)
lev_cost = 0.1 / 252

retA_list = []; h_rem = l_rem = 0
for date in dfA.index:
    if h_rem == 0 and h_sig.loc[date]: h_rem = 40
    if l_rem == 0 and l_sig.loc[date]: l_rem = 50
    base = dfA.loc[date, "Momentum"]
    if h_rem > 0:
        r = 0.5 * base + 0.5 * dfA.loc[date, "Hedge"]; h_rem -= 1
    elif l_rem > 0:
        r = base + 0.25 * base - 0.25 * lev_cost; l_rem -= 1
    else:
        r = base
    retA_list.append(r)

series_A = pd.Series(retA_list, index=dfA.index)
print(f"  Strategy A: {len(series_A)} daily bars  {series_A.index[0].date()} to {series_A.index[-1].date()}")
pA = perf(series_A, label="Daily Mom+Leverage+UVXY")
print(f"  Ann={pA['ann_ret']:.2%}  Sharpe={pA['sharpe']:.4f}  MaxDD={pA['max_dd']:.2%}")

# ════════════════════════════════════════════════════════════════
# STRATEGY C: Intraday MR + Momentum Flip (2019-2026)
# ════════════════════════════════════════════════════════════════
print("\nGenerating Strategy C: Intraday MR (2019-2026)...")

from strategies.intraday_mean_reversion import run_intraday_mean_reversion

daily_close_C  = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
hourly_close_C = pd.read_parquet("data/cache/merged_hourly_close.parquet")
hourly_open_C  = pd.read_parquet("data/cache/merged_hourly_open.parquet")

hourly_close_C.index = hourly_close_C.index.floor("h")
hourly_open_C.index  = hourly_open_C.index.floor("h")
hourly_close_C = hourly_close_C[~hourly_close_C.index.duplicated(keep="last")]
hourly_open_C  = hourly_open_C[~hourly_open_C.index.duplicated(keep="last")]

daily_cols = set(daily_close_C.columns)
h_cols     = set(hourly_close_C.columns) & set(hourly_open_C.columns)
common_C   = sorted(daily_cols & h_cols)
valid_C    = [c for c in common_C
              if hourly_close_C[c].isna().mean() < 0.30 and
                 hourly_open_C[c].isna().mean()  < 0.30]

start_C = hourly_close_C.index[0].date()
end_C   = hourly_close_C.index[-1].date()

best_ret_C, _, _ = run_intraday_mean_reversion(
    daily_close  = daily_close_C[valid_C].ffill().loc[str(start_C):str(end_C)],
    hourly_open  = hourly_open_C[valid_C].ffill(),
    hourly_close = hourly_close_C[valid_C].ffill(),
    sigma_grid=[4.0], flip_hold_days_grid=[3], lookback_grid=[20], top_n_grid=[5],
    transaction_cost=0.001, short_borrow_rate=0.08
)
series_C = best_ret_C.rename("IntradayMR")
print(f"  Strategy C: {len(series_C)} daily bars  {series_C.index[0].date()} to {series_C.index[-1].date()}")
pC = perf(series_C, label="Intraday MR")
print(f"  Ann={pC['ann_ret']:.2%}  Sharpe={pC['sharpe']:.4f}  MaxDD={pC['max_dd']:.2%}")

# ════════════════════════════════════════════════════════════════
# STRATEGY B: QQQ Bubble Hourly Momentum (2020-2026)
# ════════════════════════════════════════════════════════════════
print("\nGenerating Strategy B: QQQ Bubble Hourly Momentum (2020-2026)...")

qqq_raw = pd.read_parquet("data/cache/qqq_hourly_close.parquet")["QQQ"]
stk_raw = pd.read_parquet("data/cache/merged_hourly_close.parquet")

qqq_h = qqq_raw.copy(); qqq_h.index = qqq_h.index.floor("h")
stk_h = stk_raw.copy(); stk_h.index = stk_h.index.floor("h")
qqq_h = qqq_h[~qqq_h.index.duplicated(keep="last")]
stk_h = stk_h[~stk_h.index.duplicated(keep="last")]
idx_B = qqq_h.index.intersection(stk_h.index)
qqq_B = qqq_h.loc[idx_B]
stk_B = stk_h.loc[idx_B]

valid_B = [c for c in stk_B.columns if c in daily_cols and stk_B[c].isna().mean() < 0.30]
stk_B   = stk_B[valid_B].ffill()
ret_B   = stk_B.pct_change().clip(-0.10, 0.10).fillna(0).values.astype(np.float32)
logB    = np.log1p(ret_B.clip(-0.10, 0.10))
cumlogB = np.cumsum(logB, axis=0)
n_B     = len(idx_B)

def bub_score(price, ma_w=500):
    lp = np.log(price); f = price.rolling(ma_w).mean()
    r  = lp - np.log(f)
    z  = (r - r.rolling(ma_w).mean()) / r.rolling(ma_w).std()
    return np.tanh(z / 2)

bub_B  = bub_score(qqq_B, 500).fillna(0).values
mom_B  = stk_B.pct_change(40).fillna(0).values.astype(np.float32)
HOLD_B = 52; TOPN_B = 5; warmup_B = 545

fwd_B = np.zeros((n_B, len(valid_B)), dtype=np.float32)
fwd_B[:n_B-HOLD_B] = np.expm1(cumlogB[HOLD_B:] - cumlogB[:n_B-HOLD_B])

hrB = np.zeros(n_B)
i = warmup_B
while i < n_B - HOLD_B:
    if bub_B[i] < -0.8:
        top_idx = np.argpartition(mom_B[i], -TOPN_B)[-TOPN_B:]
        for j in range(i+1, i+1+HOLD_B):
            if j < n_B:
                hrB[j] = float(ret_B[j, top_idx].mean())
        i += HOLD_B
    else:
        i += 1

hrB_s   = pd.Series(hrB, index=idx_B)
daily_B = hrB_s.groupby(hrB_s.index.date).apply(lambda x: (1+x).prod()-1)
daily_B.index = pd.to_datetime(daily_B.index)
series_B = daily_B.rename("QQQ_Bubble")
print(f"  Strategy B: {len(series_B)} daily bars  {series_B.index[0].date()} to {series_B.index[-1].date()}")
pB = perf(series_B, label="QQQ Bubble Hourly")
print(f"  Ann={pB['ann_ret']:.2%}  Sharpe={pB['sharpe']:.4f}  MaxDD={pB['max_dd']:.2%}")

# ════════════════════════════════════════════════════════════════
# BUILD COMBINED DAILY RETURN FRAME (1997-2026)
# ════════════════════════════════════════════════════════════════
print("\nBuilding combined daily return frame...")

all_dates = series_A.index
df_all = pd.DataFrame(index=all_dates)
df_all["A"] = series_A
df_all["C"] = series_C.reindex(all_dates).fillna(0)
df_all["B"] = series_B.reindex(all_dates).fillna(0)

# Availability flags
df_all["avail_A"] = True
df_all["avail_C"] = df_all.index >= pd.Timestamp("2019-01-02")
df_all["avail_B"] = df_all.index >= pd.Timestamp("2020-07-27")

# ── 1. FIXED EQUAL WEIGHT ──────────────────────────────────────
n_avail = df_all[["avail_A","avail_C","avail_B"]].sum(axis=1)
w_fixed_A = 1.0 / n_avail
w_fixed_C = df_all["avail_C"].astype(float) / n_avail
w_fixed_B = df_all["avail_B"].astype(float) / n_avail

series_fixed = (w_fixed_A * df_all["A"] +
                w_fixed_C * df_all["C"] +
                w_fixed_B * df_all["B"])
series_fixed.name = "Fixed Equal Weight"

# ── 2. MOMENTUM ALLOCATION ────────────────────────────────────
WINDOW = 60

def rolling_sh(s, w=WINDOW):
    rm = s.rolling(w, min_periods=10).mean()
    rs = s.rolling(w, min_periods=10).std()
    return (rm / rs * np.sqrt(252)).fillna(0)

sh_A = rolling_sh(df_all["A"])
sh_C = rolling_sh(df_all["C"])
sh_B = rolling_sh(df_all["B"])

# Positive Sharpe only, proportional; unavailable = 0
wA = sh_A.clip(lower=0) * df_all["avail_A"]
wC = sh_C.clip(lower=0) * df_all["avail_C"]
wB = sh_B.clip(lower=0) * df_all["avail_B"]
tot_w = wA + wC + wB

# Fallback: if all zero, use equal weight of available
fallback_A = w_fixed_A.where(tot_w == 0, 0)
fallback_C = w_fixed_C.where(tot_w == 0, 0)
fallback_B = w_fixed_B.where(tot_w == 0, 0)

wA_norm = wA / tot_w.replace(0, np.nan).fillna(1) + fallback_A
wC_norm = wC / tot_w.replace(0, np.nan).fillna(1) + fallback_C
wB_norm = wB / tot_w.replace(0, np.nan).fillna(1) + fallback_B

series_mom = (wA_norm * df_all["A"] +
              wC_norm * df_all["C"] +
              wB_norm * df_all["B"])
series_mom.name = "Momentum Allocation"

# ── QQQ benchmark ──────────────────────────────────────────────
if "QQQ" in close_data.columns:
    qqq_bench = close_data["QQQ"].pct_change().fillna(0).reindex(all_dates).fillna(0)
else:
    qqq_bench = pd.Series(0, index=all_dates)

spy_bench = close_data["SPY"].pct_change().fillna(0).reindex(all_dates).fillna(0) \
            if "SPY" in close_data.columns else pd.Series(0, index=all_dates)

# ════════════════════════════════════════════════════════════════
# PERFORMANCE COMPARISON
# ════════════════════════════════════════════════════════════════
strategies = {
    "A: Daily Mom+Lev+UVXY": series_A,
    "C: Intraday MR":         series_C.reindex(all_dates).fillna(0),
    "B: QQQ Bubble Hourly":   series_B.reindex(all_dates).fillna(0),
    "Fixed Equal Weight":     series_fixed,
    "Momentum Allocation":    series_mom,
    "QQQ Buy & Hold":         qqq_bench,
    "SPY Buy & Hold":         spy_bench,
}

print(f"\n{'='*110}")
print("OVERALL PERFORMANCE COMPARISON (starting 1997)")
print(f"{'='*110}")
print(f"  {'Strategy':<30}  {'AnnRet':>8}  {'Sharpe':>7}  {'Sortino':>8}  {'MaxDD':>8}  {'TotalRet':>10}  {'PosYrs':>8}  {'WinRate':>8}")
print(f"  {'-'*30}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}")

results = {}
for name, s in strategies.items():
    p = perf(s, label=name)
    results[name] = p
    if p:
        print(f"  {name:<30}  {p['ann_ret']:>8.2%}  {p['sharpe']:>7.4f}  {p['sortino']:>8.4f}  "
              f"{p['max_dd']:>8.2%}  {p['total_ret']:>10.2%}  {p['pos_years']:>8}  {p['win_rate']:>8.1%}")

# ════════════════════════════════════════════════════════════════
# YEARLY BREAKDOWN
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*110}")
print("YEARLY BREAKDOWN")
print(f"{'='*110}")
print(f"  {'Year':<6}  {'A:DailyMom':>10}  {'C:IntradMR':>10}  {'B:QQQBub':>10}  {'FixedEW':>10}  {'MomAlloc':>10}  {'QQQ':>8}  {'SPY':>8}  {'Avail':>8}")
print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")

for yr in range(1997, 2027):
    yA  = series_A[series_A.index.year == yr]
    yC  = series_C.reindex(all_dates)[all_dates.year == yr].fillna(0)
    yB  = series_B.reindex(all_dates)[all_dates.year == yr].fillna(0)
    yF  = series_fixed[series_fixed.index.year == yr]
    yM  = series_mom[series_mom.index.year == yr]
    yQ  = qqq_bench[qqq_bench.index.year == yr]
    yS  = spy_bench[spy_bench.index.year == yr]
    if len(yA) == 0: continue

    rA = (1+yA).prod()-1; rC = (1+yC).prod()-1; rB = (1+yB).prod()-1
    rF = (1+yF).prod()-1; rM = (1+yM).prod()-1
    rQ = (1+yQ).prod()-1; rS = (1+yS).prod()-1

    avail = "A"
    if yr >= 2019: avail = "A+C"
    if yr >= 2021: avail = "A+B+C"

    print(f"  {yr:<6}  {rA:>10.2%}  {rC:>10.2%}  {rB:>10.2%}  {rF:>10.2%}  {rM:>10.2%}  {rQ:>8.2%}  {rS:>8.2%}  {avail:>8}")

# ════════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle("Portfolio Backtest 1997-2026: Strategy Index Comparison",
             fontsize=14, fontweight="bold")

colors = {"A: Daily Mom+Lev+UVXY": "steelblue",
          "C: Intraday MR":         "orange",
          "B: QQQ Bubble Hourly":   "green",
          "Fixed Equal Weight":     "darkred",
          "Momentum Allocation":    "purple",
          "QQQ Buy & Hold":         "gray",
          "SPY Buy & Hold":         "lightgray"}

# 1. Full period wealth (log scale)
ax = axes[0, 0]
for name, s in strategies.items():
    w = (1 + s).cumprod()
    lw = 2.5 if "Alloc" in name or "Fixed" in name else 1.5
    ls = "--" if "Hold" in name else "-"
    ax.plot(w.index, w.values, label=name, lw=lw, ls=ls,
            color=colors.get(name, "black"), alpha=0.85)
ax.set_yscale("log")
ax.set_title("Cumulative Wealth 1997-2026 (log)", fontweight="bold")
ax.set_ylabel("Wealth Multiple"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# 2. Wealth zoomed: 2019-2026 (when all available)
ax = axes[0, 1]
start_zoom = pd.Timestamp("2019-01-01")
for name, s in strategies.items():
    sz = s[s.index >= start_zoom]
    if len(sz) == 0: continue
    wz = (1 + sz).cumprod(); wz /= wz.iloc[0]
    lw = 2.5 if "Alloc" in name or "Fixed" in name else 1.5
    ls = "--" if "Hold" in name else "-"
    ax.plot(wz.index, wz.values, label=name, lw=lw, ls=ls,
            color=colors.get(name, "black"), alpha=0.85)
ax.set_title("Normalized Wealth 2019-2026 (all strategies)", fontweight="bold")
ax.set_ylabel("Normalized Wealth"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# 3. Annual return comparison (last 6 years)
ax = axes[0, 2]
plot_strats = ["A: Daily Mom+Lev+UVXY", "Fixed Equal Weight", "Momentum Allocation", "QQQ Buy & Hold"]
years_plot  = list(range(2019, 2027))
x = np.arange(len(years_plot)); width = 0.2
for k, name in enumerate(plot_strats):
    s = strategies[name]
    yrets = [(1 + s[s.index.year == yr]).prod() - 1 for yr in years_plot]
    ax.bar(x + k*width, [r*100 for r in yrets], width, label=name,
           color=colors.get(name), alpha=0.75)
ax.set_xticks(x + width*1.5); ax.set_xticklabels([str(y) for y in years_plot], fontsize=8)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Yearly Returns 2019-2026 (%)", fontweight="bold")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis="y")

# 4. Drawdown comparison (Fixed vs Momentum vs A)
ax = axes[1, 0]
for name in ["A: Daily Mom+Lev+UVXY", "Fixed Equal Weight", "Momentum Allocation"]:
    s = strategies[name]; w = (1 + s).cumprod()
    dd = w / w.cummax() - 1
    ax.fill_between(dd.index, dd.values, 0, alpha=0.35, color=colors.get(name))
    ax.plot(dd.index, dd.values, lw=1.2, color=colors.get(name), label=name)
ax.set_title("Drawdown Comparison (Full Period)", fontweight="bold")
ax.set_ylabel("Drawdown"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 5. Sharpe ratio bar
ax = axes[1, 1]
plot_names = list(strategies.keys())
sharpes = [results[n].get("sharpe", 0) for n in plot_names]
bar_clrs = [colors.get(n, "steelblue") for n in plot_names]
bars = ax.bar(range(len(plot_names)), sharpes, color=bar_clrs, alpha=0.75)
ax.set_xticks(range(len(plot_names)))
ax.set_xticklabels([n.replace(" ", "\n") for n in plot_names], fontsize=7)
ax.axhline(0, color="black", lw=0.8); ax.axhline(1, color="red", lw=1, ls="--", alpha=0.5)
ax.set_title("Sharpe Ratio Comparison", fontweight="bold")
ax.set_ylabel("Sharpe Ratio"); ax.grid(True, alpha=0.3, axis="y")
for bar, sh in zip(bars, sharpes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{sh:.2f}", ha="center", va="bottom", fontsize=8)

# 6. Dynamic weights (Momentum Allocation)
ax = axes[1, 2]
ax.stackplot(wA_norm.index, wA_norm.values, wC_norm.values, wB_norm.values,
             labels=["A: Daily Mom", "C: Intraday MR", "B: QQQ Bubble"],
             colors=["steelblue", "orange", "green"], alpha=0.7)
ax.set_title("Momentum Allocation Weights Over Time", fontweight="bold")
ax.set_ylabel("Portfolio Weight")
ax.set_ylim(0, 1); ax.legend(loc="lower right", fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/portfolio_combined_1997_2026.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: results/portfolio_combined_1997_2026.png")

# Summary CSV
rows = []
for name, p in results.items():
    if p:
        rows.append({**{"Strategy": name}, **p})
pd.DataFrame(rows).to_csv("results/portfolio_combined_summary.csv", index=False)
print("Saved: results/portfolio_combined_summary.csv")
print("\nDONE")
