"""
FULL PORTFOLIO ANALYSIS — Sharpe, MaxDD, Contribution (1997-2026)
Save all results to CSV + generate comprehensive .md report
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, os, sys
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────
# REBUILD STRATEGIES (fast)
# ─────────────────────────────────────────────────────────────
print("Building strategy return series...")

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

dfA = pd.DataFrame(strat_A_raw).set_index("Date").dropna()

def calc_bubble(price, ma=120, z=240):
    lp = np.log(np.maximum(price, 1e-6)); f = lp.rolling(ma).mean(); r = lp - f
    return np.tanh(((r - r.rolling(z).mean()) / r.rolling(z).std()) / 2)

base_w = (1 + dfA).cumprod() / (1 + dfA).cumprod().iloc[0]
bub_A  = calc_bubble(base_w["Momentum"], 120, 240)
h_sig  = (bub_A > 0.85).shift(1).fillna(False)
l_sig  = (bub_A < -0.88).shift(1).fillna(False)
lev_c  = 0.1 / 252

retA_list = []; h_rem = l_rem = 0
for date in dfA.index:
    if h_rem == 0 and h_sig.loc[date]: h_rem = 40
    if l_rem == 0 and l_sig.loc[date]: l_rem = 50
    base = dfA.loc[date, "Momentum"]
    if h_rem > 0:   r = 0.5*base + 0.5*dfA.loc[date, "Hedge"]; h_rem -= 1
    elif l_rem > 0: r = base + 0.25*base - 0.25*lev_c;          l_rem -= 1
    else:           r = base
    retA_list.append(r)
series_A = pd.Series(retA_list, index=dfA.index)
print(f"  A done: {series_A.index[0].date()} to {series_A.index[-1].date()}")

from strategies.intraday_mean_reversion import run_intraday_mean_reversion
daily_close_C  = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
hourly_close_C = pd.read_parquet("data/cache/merged_hourly_close.parquet")
hourly_open_C  = pd.read_parquet("data/cache/merged_hourly_open.parquet")
hourly_close_C.index = hourly_close_C.index.floor("h"); hourly_open_C.index = hourly_open_C.index.floor("h")
hourly_close_C = hourly_close_C[~hourly_close_C.index.duplicated(keep="last")]
hourly_open_C  = hourly_open_C[~hourly_open_C.index.duplicated(keep="last")]
daily_cols = set(daily_close_C.columns)
h_cols     = set(hourly_close_C.columns) & set(hourly_open_C.columns)
valid_C    = [c for c in sorted(daily_cols & h_cols)
              if hourly_close_C[c].isna().mean() < 0.30 and hourly_open_C[c].isna().mean() < 0.30]
s_C, e_C   = hourly_close_C.index[0].date(), hourly_close_C.index[-1].date()
best_ret_C, _, _ = run_intraday_mean_reversion(
    daily_close  = daily_close_C[valid_C].ffill().loc[str(s_C):str(e_C)],
    hourly_open  = hourly_open_C[valid_C].ffill(),
    hourly_close = hourly_close_C[valid_C].ffill(),
    sigma_grid=[4.0], flip_hold_days_grid=[3], lookback_grid=[20], top_n_grid=[5],
    transaction_cost=0.001, short_borrow_rate=0.08)
series_C = best_ret_C.rename("IntradayMR")
print(f"  C done: {series_C.index[0].date()} to {series_C.index[-1].date()}")

qqq_raw = pd.read_parquet("data/cache/qqq_hourly_close.parquet")["QQQ"]
stk_raw = pd.read_parquet("data/cache/merged_hourly_close.parquet")
qqq_h = qqq_raw.copy(); qqq_h.index = qqq_h.index.floor("h")
stk_h = stk_raw.copy(); stk_h.index = stk_h.index.floor("h")
qqq_h = qqq_h[~qqq_h.index.duplicated(keep="last")]
stk_h = stk_h[~stk_h.index.duplicated(keep="last")]
idx_B = qqq_h.index.intersection(stk_h.index)
qqq_B = qqq_h.loc[idx_B]; stk_B = stk_h.loc[idx_B]
valid_B = [c for c in stk_B.columns if c in daily_cols and stk_B[c].isna().mean() < 0.30]
stk_B = stk_B[valid_B].ffill()
ret_B = stk_B.pct_change().clip(-0.10, 0.10).fillna(0).values.astype(np.float32)
logB  = np.log1p(ret_B.clip(-0.10, 0.10)); cumlogB = np.cumsum(logB, axis=0); n_B = len(idx_B)

def bub_score(price, ma_w=500):
    lp = np.log(price); f = price.rolling(ma_w).mean(); r = lp - np.log(f)
    return np.tanh(((r - r.rolling(ma_w).mean()) / r.rolling(ma_w).std()) / 2)

bub_B = bub_score(qqq_B, 500).fillna(0).values
mom_B = stk_B.pct_change(40).fillna(0).values.astype(np.float32)
fwd_B = np.zeros((n_B, len(valid_B)), dtype=np.float32)
fwd_B[:n_B-52] = np.expm1(cumlogB[52:] - cumlogB[:n_B-52])
hrB = np.zeros(n_B); i = 545
while i < n_B - 52:
    if bub_B[i] < -0.8:
        top_idx = np.argpartition(mom_B[i], -5)[-5:]
        for j in range(i+1, i+53):
            if j < n_B: hrB[j] = float(ret_B[j, top_idx].mean())
        i += 52
    else: i += 1
hrB_s   = pd.Series(hrB, index=idx_B)
daily_B = hrB_s.groupby(hrB_s.index.date).apply(lambda x: (1+x).prod()-1)
daily_B.index = pd.to_datetime(daily_B.index)
series_B = daily_B.rename("QQQ_Bubble")
print(f"  B done: {series_B.index[0].date()} to {series_B.index[-1].date()}")

# ── Portfolios ────────────────────────────────────────────────
all_dates = series_A.index
sA = series_A
sC = series_C.reindex(all_dates).fillna(0)
sB = series_B.reindex(all_dates).fillna(0)

avail_C = all_dates >= pd.Timestamp("2019-01-02")
avail_B = all_dates >= pd.Timestamp("2020-07-27")
n_avail = 1 + avail_C.astype(int) + avail_B.astype(int)
wFA = pd.Series(1.0 / n_avail, index=all_dates)
wFC = pd.Series(avail_C.astype(float) / n_avail, index=all_dates)
wFB = pd.Series(avail_B.astype(float) / n_avail, index=all_dates)
series_fixed = wFA*sA + wFC*sC + wFB*sB

def rolling_sh(s, w=60):
    rm = s.rolling(w, min_periods=10).mean()
    rs = s.rolling(w, min_periods=10).std()
    return (rm / rs * np.sqrt(252)).fillna(0)

shA = rolling_sh(sA).clip(lower=0)
shC = rolling_sh(sC).clip(lower=0) * pd.Series(avail_C, index=all_dates)
shB = rolling_sh(sB).clip(lower=0) * pd.Series(avail_B, index=all_dates)
tot = shA + shC + shB
wMA = (shA / tot.replace(0,np.nan)).fillna(wFA)
wMC = (shC / tot.replace(0,np.nan)).fillna(wFC)
wMB = (shB / tot.replace(0,np.nan)).fillna(wFB)
series_mom = wMA*sA + wMC*sC + wMB*sB

qqq_ret = close_data["QQQ"].pct_change().fillna(0).reindex(all_dates).fillna(0) \
          if "QQQ" in close_data.columns else pd.Series(0, index=all_dates)
spy_ret = close_data["SPY"].pct_change().fillna(0).reindex(all_dates).fillna(0) \
          if "SPY" in close_data.columns else pd.Series(0, index=all_dates)

# ─────────────────────────────────────────────────────────────
# COMPUTE YEARLY STATS: Return, Sharpe, MaxDD
# ─────────────────────────────────────────────────────────────
def yr_stats(s, yr):
    ys = s[s.index.year == yr]
    if len(ys) < 2:
        return dict(ret=np.nan, sharpe=np.nan, maxdd=np.nan)
    tr  = (1 + ys).prod() - 1
    sh  = ys.mean() / ys.std() * np.sqrt(252) if ys.std() > 0 else 0.0
    w   = (1 + ys).cumprod()
    mdd = (w / w.cummax() - 1).min()
    return dict(ret=tr, sharpe=sh, maxdd=mdd)

print("\nComputing yearly stats...")
series_map = {
    "A": sA, "C": sC, "B": sB,
    "Fixed_EW": series_fixed, "Mom_Alloc": series_mom,
    "QQQ": qqq_ret, "SPY": spy_ret
}

yearly_rows = []
for yr in range(1997, 2027):
    if not any(all_dates.year == yr): continue
    avail = "A" if yr < 2019 else ("A+C" if yr < 2021 else "A+B+C")
    row = {"Year": yr, "Available": avail}
    for name, s in series_map.items():
        st = yr_stats(s, yr)
        row[f"Ret_{name}"]    = st["ret"]
        row[f"Sharpe_{name}"] = st["sharpe"]
        row[f"MaxDD_{name}"]  = st["maxdd"]
    yearly_rows.append(row)

yr_df = pd.DataFrame(yearly_rows)
yr_df.to_csv("results/portfolio_full_yearly_stats.csv", index=False)
print("Saved: results/portfolio_full_yearly_stats.csv")

# ─────────────────────────────────────────────────────────────
# OVERALL STATS
# ─────────────────────────────────────────────────────────────
def overall_stats(s, name):
    years = len(s) / 252
    w     = (1 + s).cumprod()
    tr    = w.iloc[-1] - 1
    ar    = (1 + tr) ** (1 / years) - 1 if tr > -1 else -1
    sh    = s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0
    dn    = s[s < 0].std(ddof=0)
    so    = s.mean() / dn * np.sqrt(252) if dn > 0 else 0
    mdd   = (w / w.cummax() - 1).min()
    pos_y = sum((1+s[s.index.year==yr]).prod()-1 > 0 for yr in s.index.year.unique())
    tot_y = len(s.index.year.unique())
    wr    = (s > 0).mean()
    return dict(Strategy=name, Ann_Ret=ar, Total_Ret=tr, Sharpe=sh,
                Sortino=so, MaxDD=mdd, Pos_Years=f"{pos_y}/{tot_y}",
                Win_Rate=wr, Years=f"{years:.1f}")

overall_rows = [overall_stats(s, n) for n, s in series_map.items()]
overall_df   = pd.DataFrame(overall_rows)
overall_df.to_csv("results/portfolio_overall_stats.csv", index=False)
print("Saved: results/portfolio_overall_stats.csv")

# ─────────────────────────────────────────────────────────────
# PRINT FULL TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*150)
print("FULL YEARLY STATS: Return | Sharpe | MaxDD")
print("="*150)
print(f"  {'Year':<5}  {'Avail':<8}  "
      f"{'Ret_A':>8} {'Sh_A':>7} {'DD_A':>8}  "
      f"{'Ret_C':>8} {'Sh_C':>7} {'DD_C':>8}  "
      f"{'Ret_B':>8} {'Sh_B':>7} {'DD_B':>8}  "
      f"{'Ret_F':>8} {'Sh_F':>7} {'DD_F':>8}  "
      f"{'Ret_M':>8} {'Sh_M':>7} {'DD_M':>8}  "
      f"{'QQQ':>8}")
print("  " + "-"*148)

for _, r in yr_df.iterrows():
    yr    = int(r["Year"])
    avail = r["Available"]

    def f(col, pct=True):
        v = r.get(col, np.nan)
        if pd.isna(v): return "     n/a"
        return f"{v*100:>7.1f}%" if pct else f"{v:>7.3f}"

    print(f"  {yr:<5}  {avail:<8}  "
          f"{f('Ret_A')} {f('Sharpe_A',False)} {f('MaxDD_A')}  "
          f"{f('Ret_C')} {f('Sharpe_C',False)} {f('MaxDD_C')}  "
          f"{f('Ret_B')} {f('Sharpe_B',False)} {f('MaxDD_B')}  "
          f"{f('Ret_Fixed_EW')} {f('Sharpe_Fixed_EW',False)} {f('MaxDD_Fixed_EW')}  "
          f"{f('Ret_Mom_Alloc')} {f('Sharpe_Mom_Alloc',False)} {f('MaxDD_Mom_Alloc')}  "
          f"{f('Ret_QQQ')}")

# Overall summary
print("\n" + "="*100)
print("OVERALL PERFORMANCE SUMMARY (Full Period)")
print("="*100)
print(f"  {'Strategy':<22}  {'AnnRet':>8}  {'Sharpe':>7}  {'Sortino':>8}  {'MaxDD':>8}  {'TotalRet':>12}  {'PosYrs':>8}  {'WinRate':>8}")
print("  " + "-"*98)
for _, r in overall_df.iterrows():
    print(f"  {r['Strategy']:<22}  {r['Ann_Ret']:>8.2%}  {r['Sharpe']:>7.4f}  "
          f"{r['Sortino']:>8.4f}  {r['MaxDD']:>8.2%}  {r['Total_Ret']:>12.2%}  "
          f"{r['Pos_Years']:>8}  {r['Win_Rate']:>8.1%}")

# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 2, figsize=(22, 24))
fig.suptitle("Portfolio Full Analysis: Return + Sharpe + MaxDD (1997-2026)",
             fontsize=14, fontweight="bold")

clr = {"A":"steelblue","C":"orange","B":"green",
       "Fixed_EW":"darkred","Mom_Alloc":"purple","QQQ":"gray"}

def yr_bar(ax, col, title, clr_key, yr_min=1997):
    sub = yr_df[yr_df["Year"] >= yr_min].copy()
    vals = sub[col].fillna(0).values * 100
    colors = [clr.get(clr_key,"steelblue") if v >= 0 else "tomato" for v in vals]
    ax.bar(sub["Year"].astype(str), vals, color=colors, alpha=0.75)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(title, fontweight="bold"); ax.set_ylabel("%")
    ax.tick_params(axis="x", rotation=45, labelsize=7); ax.grid(True, alpha=0.3, axis="y")

def yr_bar_sh(ax, col, title, clr_key, yr_min=1997):
    sub = yr_df[yr_df["Year"] >= yr_min].copy()
    vals = sub[col].fillna(0).values
    colors = [clr.get(clr_key,"steelblue") if v >= 0 else "tomato" for v in vals]
    ax.bar(sub["Year"].astype(str), vals, color=colors, alpha=0.75)
    ax.axhline(0, color="black", lw=0.8); ax.axhline(1, color="green", lw=1, ls="--", alpha=0.5)
    ax.set_title(title, fontweight="bold"); ax.set_ylabel("Sharpe")
    ax.tick_params(axis="x", rotation=45, labelsize=7); ax.grid(True, alpha=0.3, axis="y")

# Row 0: Strategy A
yr_bar(axes[0,0], "Ret_A", "A: Daily Mom+Lev+UVXY — Yearly Return (%)", "A")
yr_bar_sh(axes[0,1], "Sharpe_A", "A: Daily Mom+Lev+UVXY — Yearly Sharpe", "A")

# Row 1: C + B side by side
x = np.arange(len(yr_df[yr_df["Year"]>=2019])); w = 0.35
sub19 = yr_df[yr_df["Year"]>=2019]
ax = axes[1,0]
ax.bar(x-w/2, sub19["Ret_C"].fillna(0)*100, w, label="C: Intraday MR", color="orange", alpha=0.75)
ax.bar(x+w/2, sub19["Ret_B"].fillna(0)*100, w, label="B: QQQ Bubble", color="green", alpha=0.75)
ax.set_xticks(x); ax.set_xticklabels(sub19["Year"].astype(str), fontsize=9)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("B & C: Yearly Return (%) — 2019+", fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")

ax = axes[1,1]
ax.bar(x-w/2, sub19["Sharpe_C"].fillna(0), w, label="C: Intraday MR",
       color=["orange" if v>=0 else "tomato" for v in sub19["Sharpe_C"].fillna(0)], alpha=0.75)
ax.bar(x+w/2, sub19["Sharpe_B"].fillna(0), w, label="B: QQQ Bubble",
       color=["green" if v>=0 else "tomato" for v in sub19["Sharpe_B"].fillna(0)], alpha=0.75)
ax.set_xticks(x); ax.set_xticklabels(sub19["Year"].astype(str), fontsize=9)
ax.axhline(0, color="black", lw=0.8); ax.axhline(1, color="gray", lw=1, ls="--", alpha=0.5)
ax.set_title("B & C: Yearly Sharpe — 2019+", fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")

# Row 2: Portfolio comparison — Return and Sharpe
x_all = np.arange(len(yr_df)); w2 = 0.35
ax = axes[2,0]
ax.bar(x_all-w2/2, yr_df["Ret_Fixed_EW"].fillna(0)*100, w2,
       label="Fixed EW", color="darkred", alpha=0.7)
ax.bar(x_all+w2/2, yr_df["Ret_Mom_Alloc"].fillna(0)*100, w2,
       label="Mom Alloc", color="purple", alpha=0.7)
ax.set_xticks(x_all[::3]); ax.set_xticklabels(yr_df["Year"].values[::3].astype(str), fontsize=8)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Portfolio Return: Fixed EW vs Momentum Alloc (%)", fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")

ax = axes[2,1]
ax.bar(x_all-w2/2, yr_df["Sharpe_Fixed_EW"].fillna(0), w2,
       label="Fixed EW", color="darkred", alpha=0.7)
ax.bar(x_all+w2/2, yr_df["Sharpe_Mom_Alloc"].fillna(0), w2,
       label="Mom Alloc", color="purple", alpha=0.7)
ax.set_xticks(x_all[::3]); ax.set_xticklabels(yr_df["Year"].values[::3].astype(str), fontsize=8)
ax.axhline(0, color="black", lw=0.8); ax.axhline(1, color="green", lw=1, ls="--", alpha=0.5)
ax.set_title("Portfolio Sharpe: Fixed EW vs Momentum Alloc", fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")

# Row 3: MaxDD comparison
ax = axes[3,0]
ax.bar(x_all-w2/2, yr_df["MaxDD_Fixed_EW"].fillna(0)*100, w2,
       label="Fixed EW", color="darkred", alpha=0.7)
ax.bar(x_all+w2/2, yr_df["MaxDD_Mom_Alloc"].fillna(0)*100, w2,
       label="Mom Alloc", color="purple", alpha=0.7)
ax.set_xticks(x_all[::3]); ax.set_xticklabels(yr_df["Year"].values[::3].astype(str), fontsize=8)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Portfolio Max Drawdown: Fixed EW vs Momentum Alloc (%)", fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")

ax = axes[3,1]
for name, col, lw in [("A", "MaxDD_A", 2), ("C", "MaxDD_C", 1.5), ("B", "MaxDD_B", 1.5),
                       ("Fixed EW", "MaxDD_Fixed_EW", 2.5), ("Mom Alloc", "MaxDD_Mom_Alloc", 2.5),
                       ("QQQ", "MaxDD_QQQ", 1)]:
    sub = yr_df.dropna(subset=[col])
    ax.plot(sub["Year"], sub[col]*100, marker="o", lw=lw, markersize=4,
            label=name, color=clr.get(name.replace(" ","_").split(":")[0].strip()[:1]
                                       if ":" in name else name.replace(" ","_"), None),
            alpha=0.8)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Yearly MaxDD: All Strategies", fontweight="bold")
ax.set_ylabel("MaxDD (%)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/portfolio_full_analysis.png", dpi=150, bbox_inches="tight")
print("\nSaved: results/portfolio_full_analysis.png")

# ─────────────────────────────────────────────────────────────
# WRITE MARKDOWN REPORT
# ─────────────────────────────────────────────────────────────
print("Writing markdown report...")

md_lines = []
md_lines.append("# Portfolio Backtest Results: All Strategies (1997-2026)")
md_lines.append(f"**Generated:** June 2026  ")
md_lines.append(f"**Data:** Daily (1997-2026) + Hourly (2019-2026)  ")
md_lines.append(f"**Universe:** S&P500 + NASDAQ100 (405-524 tickers)  ")
md_lines.append("")
md_lines.append("---")
md_lines.append("")
md_lines.append("## Strategies Included")
md_lines.append("")
md_lines.append("| ID | Strategy | Data | Available From |")
md_lines.append("|----|----------|------|----------------|")
md_lines.append("| A | Daily Momentum + 1.25x Leverage + UVXY Hedge | Daily | 1997-01-02 |")
md_lines.append("| C | Intraday Mean Reversion + Momentum Flip | Hourly | 2019-01-02 |")
md_lines.append("| B | QQQ Bubble Score + Long Momentum Stocks | Hourly | 2020-07-27 |")
md_lines.append("")
md_lines.append("**Allocation Methods:**")
md_lines.append("- **Fixed Equal Weight:** 1/N among available strategies each day")
md_lines.append("- **Momentum Allocation:** 60-day trailing Sharpe-weighted, rebalanced daily")
md_lines.append("")
md_lines.append("---")
md_lines.append("")
md_lines.append("## Overall Performance (Full Period)")
md_lines.append("")
md_lines.append("| Strategy | Annual Return | Sharpe | Sortino | Max Drawdown | Total Return | Pos Years | Win Rate |")
md_lines.append("|----------|---|---|---|---|---|---|---|")
for _, r in overall_df.iterrows():
    md_lines.append(
        f"| **{r['Strategy']}** | {r['Ann_Ret']:.2%} | {r['Sharpe']:.4f} | "
        f"{r['Sortino']:.4f} | {r['MaxDD']:.2%} | {r['Total_Ret']:.2%} | "
        f"{r['Pos_Years']} | {r['Win_Rate']:.1%} |")
md_lines.append("")
md_lines.append("---")
md_lines.append("")
md_lines.append("## Yearly Performance: Return | Sharpe | Max Drawdown")
md_lines.append("")
md_lines.append("| Year | Avail | Ret A | Sh A | DD A | Ret C | Sh C | DD C | Ret B | Sh B | DD B | Ret Fixed | Sh Fixed | DD Fixed | Ret Mom | Sh Mom | DD Mom | QQQ |")
md_lines.append("|------|-------|-------|------|------|-------|------|------|-------|------|------|-----------|----------|----------|---------|--------|--------|-----|")

for _, r in yr_df.iterrows():
    yr = int(r["Year"])
    def fv(col, pct=True, na="n/a"):
        v = r.get(col, np.nan)
        if pd.isna(v): return na
        return f"{v*100:.1f}%" if pct else f"{v:.2f}"
    line = (f"| {yr} | {r['Available']} | "
            f"{fv('Ret_A')} | {fv('Sharpe_A',False)} | {fv('MaxDD_A')} | "
            f"{fv('Ret_C')} | {fv('Sharpe_C',False)} | {fv('MaxDD_C')} | "
            f"{fv('Ret_B')} | {fv('Sharpe_B',False)} | {fv('MaxDD_B')} | "
            f"{fv('Ret_Fixed_EW')} | {fv('Sharpe_Fixed_EW',False)} | {fv('MaxDD_Fixed_EW')} | "
            f"{fv('Ret_Mom_Alloc')} | {fv('Sharpe_Mom_Alloc',False)} | {fv('MaxDD_Mom_Alloc')} | "
            f"{fv('Ret_QQQ')} |")
    md_lines.append(line)

md_lines.append("")
md_lines.append("---")
md_lines.append("")
md_lines.append("## Key Observations")
md_lines.append("")
md_lines.append("### MaxDD by Period")
md_lines.append("")
md_lines.append("| Period | A MaxDD | Fixed EW MaxDD | Mom Alloc MaxDD | Notes |")
md_lines.append("|--------|---------|----------------|-----------------|-------|")

period_notes = {
    "2000": "Tech crash",
    "2002": "Post-dot-com",
    "2008": "Financial crisis",
    "2011": "Debt ceiling",
    "2020": "COVID crash",
    "2022": "Rate hike bear"
}
for yr_str, note in period_notes.items():
    row = yr_df[yr_df["Year"] == int(yr_str)]
    if row.empty: continue
    r = row.iloc[0]
    md_lines.append(
        f"| {yr_str} ({note}) | {r['MaxDD_A']*100:.1f}% | "
        f"{r['MaxDD_Fixed_EW']*100:.1f}% | {r['MaxDD_Mom_Alloc']*100:.1f}% | "
        f"QQQ: {r['MaxDD_QQQ']*100:.1f}% |")

md_lines.append("")
md_lines.append("### Diversification Effect (2021-2026, all three available)")
md_lines.append("")
md_lines.append("| Year | A Sharpe | C Sharpe | B Sharpe | Fixed EW Sharpe | Mom Alloc Sharpe |")
md_lines.append("|------|----------|----------|----------|-----------------|-----------------|")
for _, r in yr_df[yr_df["Year"] >= 2021].iterrows():
    md_lines.append(
        f"| {int(r['Year'])} | {r['Sharpe_A']:.3f} | {r['Sharpe_C']:.3f} | "
        f"{r['Sharpe_B']:.3f} | {r['Sharpe_Fixed_EW']:.3f} | {r['Sharpe_Mom_Alloc']:.3f} |")

md_lines.append("")
md_lines.append("---")
md_lines.append("")
md_lines.append("## Files")
md_lines.append("")
md_lines.append("| File | Description |")
md_lines.append("|------|-------------|")
md_lines.append("| `portfolio_full_yearly_stats.csv` | All yearly stats (Return, Sharpe, MaxDD) |")
md_lines.append("| `portfolio_overall_stats.csv` | Overall performance per strategy |")
md_lines.append("| `portfolio_full_analysis.png` | 8-panel chart: returns, Sharpe, MaxDD |")
md_lines.append("| `portfolio_combined_1997_2026.png` | Wealth curves + allocation chart |")
md_lines.append("")
md_lines.append("---")
md_lines.append("")
md_lines.append("**Repository:** https://github.com/tkdlfdl/QuantTrading  ")
md_lines.append("**Last Updated:** June 2026")

md_text = "\n".join(md_lines)
with open("PORTFOLIO_BACKTEST_RESULTS.md", "w", encoding="utf-8") as f:
    f.write(md_text)
print("Saved: PORTFOLIO_BACKTEST_RESULTS.md")
print("\nDONE")
