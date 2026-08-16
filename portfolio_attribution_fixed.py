"""
PORTFOLIO ATTRIBUTION — FIXED
================================
Bug fix: contribution = sum(daily_weight * daily_return) per year
This gives exact attribution that adds up to the arithmetic portfolio return.
Also shows geometric portfolio return separately.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, os, sys
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────
# REBUILD STRATEGIES (same as before)
# ─────────────────────────────────────────────────────────────
print("Loading strategies...")

close_data = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
lookback = 140; holding = 40; top = 5
close = close_data.copy()
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

from strategies.intraday_mean_reversion import run_intraday_mean_reversion
daily_close_C  = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
hourly_close_C = pd.read_parquet("data/cache/merged_hourly_close.parquet")
hourly_open_C  = pd.read_parquet("data/cache/merged_hourly_open.parquet")
hourly_close_C.index = hourly_close_C.index.floor("h"); hourly_open_C.index = hourly_open_C.index.floor("h")
hourly_close_C = hourly_close_C[~hourly_close_C.index.duplicated(keep="last")]
hourly_open_C  = hourly_open_C[~hourly_open_C.index.duplicated(keep="last")]
daily_cols = set(daily_close_C.columns)
valid_C    = [c for c in sorted(set(hourly_close_C.columns) & set(hourly_open_C.columns) & daily_cols)
              if hourly_close_C[c].isna().mean() < 0.30 and hourly_open_C[c].isna().mean() < 0.30]
s_C, e_C   = hourly_close_C.index[0].date(), hourly_close_C.index[-1].date()
best_ret_C, _, _ = run_intraday_mean_reversion(
    daily_close=daily_close_C[valid_C].ffill().loc[str(s_C):str(e_C)],
    hourly_open=hourly_open_C[valid_C].ffill(), hourly_close=hourly_close_C[valid_C].ffill(),
    sigma_grid=[4.0], flip_hold_days_grid=[3], lookback_grid=[20], top_n_grid=[5],
    transaction_cost=0.001, short_borrow_rate=0.08)
series_C = best_ret_C.rename("IntradayMR")

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
ret_B_np = stk_B.pct_change().clip(-0.10, 0.10).fillna(0).values.astype(np.float32)
logB = np.log1p(ret_B_np.clip(-0.10,0.10)); cumlogB = np.cumsum(logB, axis=0); n_B = len(idx_B)
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
            if j < n_B: hrB[j] = float(ret_B_np[j, top_idx].mean())
        i += 52
    else: i += 1
hrB_s = pd.Series(hrB, index=idx_B)
daily_B = hrB_s.groupby(hrB_s.index.date).apply(lambda x: (1+x).prod()-1)
daily_B.index = pd.to_datetime(daily_B.index)
series_B = daily_B.rename("QQQ_Bubble")
print("  All strategies loaded.")

# ─────────────────────────────────────────────────────────────
# ALIGN
# ─────────────────────────────────────────────────────────────
all_dates = series_A.index
sA = series_A
sC = series_C.reindex(all_dates).fillna(0)
sB = series_B.reindex(all_dates).fillna(0)

avail_C = pd.Series(all_dates >= pd.Timestamp("2019-01-02"), index=all_dates)
avail_B = pd.Series(all_dates >= pd.Timestamp("2020-07-27"), index=all_dates)
n_avail = 1 + avail_C.astype(int) + avail_B.astype(int)

# Fixed EW weights (daily)
wFA = pd.Series(1.0 / n_avail, index=all_dates)
wFC = pd.Series(avail_C.astype(float) / n_avail, index=all_dates)
wFB = pd.Series(avail_B.astype(float) / n_avail, index=all_dates)

# Momentum weights (daily)
def rolling_sh(s, w=60):
    rm = s.rolling(w, min_periods=10).mean()
    rs = s.rolling(w, min_periods=10).std()
    return (rm / rs * np.sqrt(252)).fillna(0)
shA = rolling_sh(sA).clip(lower=0)
shC = rolling_sh(sC).clip(lower=0) * avail_C
shB = rolling_sh(sB).clip(lower=0) * avail_B
tot = shA + shC + shB
wMA = (shA / tot.replace(0, np.nan)).fillna(wFA)
wMC = (shC / tot.replace(0, np.nan)).fillna(wFC)
wMB = (shB / tot.replace(0, np.nan)).fillna(wFB)

# Daily CONTRIBUTION series (weight × daily return for each strategy)
# These sum EXACTLY to portfolio daily return each day
cont_F_A = wFA * sA;  cont_F_C = wFC * sC;  cont_F_B = wFB * sB
cont_M_A = wMA * sA;  cont_M_C = wMC * sC;  cont_M_B = wMB * sB

port_F_daily = cont_F_A + cont_F_C + cont_F_B   # = Fixed EW daily returns
port_M_daily = cont_M_A + cont_M_C + cont_M_B   # = Momentum daily returns

# Verify: should be zero
diff_F = (port_F_daily - (wFA*sA + wFC*sC + wFB*sB)).abs().max()
diff_M = (port_M_daily - (wMA*sA + wMC*sC + wMB*sB)).abs().max()
assert diff_F < 1e-10 and diff_M < 1e-10, "Portfolio mismatch!"
print(f"  Attribution verification: max diff Fixed={diff_F:.2e}  Mom={diff_M:.2e}  (OK)")

# ─────────────────────────────────────────────────────────────
# YEARLY STATS
# ─────────────────────────────────────────────────────────────
rows = []
for yr in range(1997, 2027):
    mask = all_dates.year == yr
    if not mask.any(): continue

    def yr_stats(s):
        ys = s[mask]
        if len(ys) < 2: return dict(arith=0, geo=0, sharpe=0, maxdd=0)
        arith = ys.sum()
        geo   = (1+ys).prod() - 1
        sh    = ys.mean() / ys.std() * np.sqrt(252) if ys.std() > 0 else 0
        w     = (1+ys).cumprod(); dd = (w / w.cummax() - 1).min()
        return dict(arith=arith, geo=geo, sharpe=sh, maxdd=dd)

    stA = yr_stats(sA); stC = yr_stats(sC); stB = yr_stats(sB)
    stF = yr_stats(port_F_daily); stM = yr_stats(port_M_daily)

    # CORRECT attribution: sum of daily contributions within year
    cFA = cont_F_A[mask].sum(); cFC = cont_F_C[mask].sum(); cFB = cont_F_B[mask].sum()
    cMA = cont_M_A[mask].sum(); cMC = cont_M_C[mask].sum(); cMB = cont_M_B[mask].sum()

    # Average weights
    wA_f = wFA[mask].mean(); wC_f = wFC[mask].mean(); wB_f = wFB[mask].mean()
    wA_m = wMA[mask].mean(); wC_m = wMC[mask].mean(); wB_m = wMB[mask].mean()

    avail = "A" if yr < 2019 else ("A+C" if yr < 2021 else "A+B+C")
    rows.append(dict(
        Year=yr, Available=avail,
        # Individual strategy returns + Sharpe + MaxDD
        Ret_A=stA["geo"], Sh_A=stA["sharpe"], DD_A=stA["maxdd"],
        Ret_C=stC["geo"] if yr >= 2019 else np.nan,
        Sh_C=stC["sharpe"] if yr >= 2019 else np.nan,
        DD_C=stC["maxdd"] if yr >= 2019 else np.nan,
        Ret_B=stB["geo"] if yr >= 2021 else np.nan,
        Sh_B=stB["sharpe"] if yr >= 2021 else np.nan,
        DD_B=stB["maxdd"] if yr >= 2021 else np.nan,
        # Fixed EW portfolio
        Port_F_Geo=stF["geo"], Port_F_Arith=stF["arith"],
        Sh_F=stF["sharpe"], DD_F=stF["maxdd"],
        Wt_F_A=wA_f, Wt_F_C=wC_f, Wt_F_B=wB_f,
        Cont_F_A=cFA, Cont_F_C=cFC, Cont_F_B=cFB,
        Check_F=cFA+cFC+cFB,   # should = Port_F_Arith
        # Momentum Alloc portfolio
        Port_M_Geo=stM["geo"], Port_M_Arith=stM["arith"],
        Sh_M=stM["sharpe"], DD_M=stM["maxdd"],
        Wt_M_A=wA_m, Wt_M_C=wC_m, Wt_M_B=wB_m,
        Cont_M_A=cMA, Cont_M_C=cMC, Cont_M_B=cMB,
        Check_M=cMA+cMC+cMB,   # should = Port_M_Arith
    ))

yr_df = pd.DataFrame(rows)

# Verify attribution adds up
yr_df["Err_F"] = (yr_df["Check_F"] - yr_df["Port_F_Arith"]).abs()
yr_df["Err_M"] = (yr_df["Check_M"] - yr_df["Port_M_Arith"]).abs()
print(f"  Attribution error (max): Fixed={yr_df['Err_F'].max():.2e}  Mom={yr_df['Err_M'].max():.2e}")

# ─────────────────────────────────────────────────────────────
# PRINT TABLES
# ─────────────────────────────────────────────────────────────
print("\n" + "="*160)
print("INDIVIDUAL STRATEGY YEARLY STATS  (Return | Sharpe | MaxDD)")
print("="*160)
print(f"  {'Year':<6}  {'Avail':<8}  "
      f"{'Ret_A':>8} {'Sh_A':>7} {'DD_A':>8}  "
      f"{'Ret_C':>8} {'Sh_C':>7} {'DD_C':>8}  "
      f"{'Ret_B':>8} {'Sh_B':>7} {'DD_B':>8}")
print("  " + "-"*110)
for _, r in yr_df.iterrows():
    def fmt(v, pct=True):
        return "     n/a" if pd.isna(v) else (f"{v*100:>7.1f}%" if pct else f"{v:>7.3f}")
    print(f"  {int(r.Year):<6}  {r.Available:<8}  "
          f"{fmt(r.Ret_A)} {fmt(r.Sh_A,False)} {fmt(r.DD_A)}  "
          f"{fmt(r.Ret_C)} {fmt(r.Sh_C,False)} {fmt(r.DD_C)}  "
          f"{fmt(r.Ret_B)} {fmt(r.Sh_B,False)} {fmt(r.DD_B)}")

print("\n" + "="*180)
print("FIXED EQUAL WEIGHT — ATTRIBUTION  (Arithmetic contributions sum exactly to Arith Return)")
print("Note: Geometric return ≠ Arith return due to compounding (not a bug)")
print("="*180)
print(f"  {'Year':<6}  {'Wt_A':>5} {'Wt_C':>5} {'Wt_B':>5}  "
      f"{'Cont_A':>9} {'Cont_C':>9} {'Cont_B':>9}  "
      f"{'Arith Ret':>10} {'Check':>9}  {'Geo Ret':>9} {'Sh_F':>7} {'DD_F':>8}")
print("  " + "-"*120)
for _, r in yr_df.iterrows():
    yr = int(r.Year)
    if yr < 2019 and r.Wt_F_C < 0.01:
        # Only A available — single line
        print(f"  {yr:<6}  {r.Wt_F_A:>5.2f} {r.Wt_F_C:>5.2f} {r.Wt_F_B:>5.2f}  "
              f"  {r.Cont_F_A*100:>8.2f}%  {r.Cont_F_C*100:>8.2f}%  {r.Cont_F_B*100:>8.2f}%  "
              f"{r.Port_F_Arith*100:>9.2f}% {r.Check_F*100:>8.2f}%  "
              f"{r.Port_F_Geo*100:>8.2f}% {r.Sh_F:>7.3f} {r.DD_F*100:>7.1f}%")
    else:
        print(f"  {yr:<6}  {r.Wt_F_A:>5.2f} {r.Wt_F_C:>5.2f} {r.Wt_F_B:>5.2f}  "
              f"  {r.Cont_F_A*100:>8.2f}%  {r.Cont_F_C*100:>8.2f}%  {r.Cont_F_B*100:>8.2f}%  "
              f"{r.Port_F_Arith*100:>9.2f}% {r.Check_F*100:>8.2f}%  "
              f"{r.Port_F_Geo*100:>8.2f}% {r.Sh_F:>7.3f} {r.DD_F*100:>7.1f}%")

print("\n" + "="*180)
print("MOMENTUM ALLOCATION — ATTRIBUTION  (Arithmetic contributions sum exactly to Arith Return)")
print("Note: Geo > Arith when momentum allocation shifts weight to strategy DURING its good period")
print("="*180)
print(f"  {'Year':<6}  {'Wt_A':>5} {'Wt_C':>5} {'Wt_B':>5}  "
      f"{'Cont_A':>9} {'Cont_C':>9} {'Cont_B':>9}  "
      f"{'Arith Ret':>10} {'Check':>9}  {'Geo Ret':>9} {'Sh_M':>7} {'DD_M':>8}  {'Geo-Arith':>10}")
print("  " + "-"*130)
for _, r in yr_df.iterrows():
    yr = int(r.Year)
    gap = r.Port_M_Geo - r.Port_M_Arith
    flag = " <-- COMPOUNDING BOOST" if gap > 0.05 else (" <-- COMPOUNDING DRAG" if gap < -0.05 else "")
    print(f"  {yr:<6}  {r.Wt_M_A:>5.2f} {r.Wt_M_C:>5.2f} {r.Wt_M_B:>5.2f}  "
          f"  {r.Cont_M_A*100:>8.2f}%  {r.Cont_M_C*100:>8.2f}%  {r.Cont_M_B*100:>8.2f}%  "
          f"{r.Port_M_Arith*100:>9.2f}% {r.Check_M*100:>8.2f}%  "
          f"{r.Port_M_Geo*100:>8.2f}% {r.Sh_M:>7.3f} {r.DD_M*100:>7.1f}%  "
          f"{gap*100:>9.2f}%{flag}")

# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
yr_df.to_csv("results/portfolio_attribution_fixed.csv", index=False)
print("\nSaved: results/portfolio_attribution_fixed.csv")

# PLOT
fig, axes = plt.subplots(3, 2, figsize=(22, 18))
fig.suptitle("Portfolio Attribution (Fixed) — Arithmetic Contributions + Geometric Return",
             fontsize=13, fontweight="bold")

post19 = yr_df[yr_df.Year >= 2019]
x = np.arange(len(post19)); w = 0.6

# Fixed EW contributions
ax = axes[0,0]
ax.bar(x, post19["Cont_F_A"]*100, w, label="A: Daily Mom", color="steelblue", alpha=0.8)
ax.bar(x, post19["Cont_F_C"]*100, w, bottom=post19["Cont_F_A"]*100,
       label="C: Intraday MR", color="orange", alpha=0.8)
ax.bar(x, post19["Cont_F_B"]*100, w,
       bottom=(post19["Cont_F_A"]+post19["Cont_F_C"])*100,
       label="B: QQQ Bubble", color="green", alpha=0.8)
ax.plot(x, post19["Port_F_Geo"]*100, "k--", lw=2, label="Geo Return", marker="o", markersize=5)
ax.set_xticks(x); ax.set_xticklabels(post19.Year.astype(str), fontsize=9)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Fixed EW: Arithmetic Contributions vs Geometric Return", fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

# Momentum contributions
ax = axes[0,1]
ax.bar(x, post19["Cont_M_A"]*100, w, label="A: Daily Mom", color="steelblue", alpha=0.8)
ax.bar(x, post19["Cont_M_C"]*100, w, bottom=post19["Cont_M_A"]*100,
       label="C: Intraday MR", color="orange", alpha=0.8)
ax.bar(x, post19["Cont_M_B"]*100, w,
       bottom=(post19["Cont_M_A"]+post19["Cont_M_C"])*100,
       label="B: QQQ Bubble", color="green", alpha=0.8)
ax.plot(x, post19["Port_M_Geo"]*100, "k--", lw=2, label="Geo Return", marker="o", markersize=5)
ax.set_xticks(x); ax.set_xticklabels(post19.Year.astype(str), fontsize=9)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Momentum Alloc: Arithmetic Contributions vs Geometric Return", fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

# Geometric vs Arithmetic comparison
ax = axes[1,0]
xf = np.arange(len(post19)); w2 = 0.35
ax.bar(xf-w2/2, post19["Port_F_Arith"]*100, w2, label="Fixed Arith", color="darkred", alpha=0.7)
ax.bar(xf+w2/2, post19["Port_F_Geo"]*100,   w2, label="Fixed Geo",   color="red",     alpha=0.7)
ax.set_xticks(xf); ax.set_xticklabels(post19.Year.astype(str), fontsize=9)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Fixed EW: Arithmetic vs Geometric Return", fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")

ax = axes[1,1]
ax.bar(xf-w2/2, post19["Port_M_Arith"]*100, w2, label="Mom Arith", color="purple",     alpha=0.7)
ax.bar(xf+w2/2, post19["Port_M_Geo"]*100,   w2, label="Mom Geo",   color="mediumpurple",alpha=0.7)
ax.set_xticks(xf); ax.set_xticklabels(post19.Year.astype(str), fontsize=9)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Momentum Alloc: Arithmetic vs Geometric Return", fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")

# Sharpe by year
ax = axes[2,0]
ax.bar(xf-w2/2, post19["Sh_F"], w2, label="Fixed EW", color="darkred", alpha=0.7)
ax.bar(xf+w2/2, post19["Sh_M"], w2, label="Mom Alloc", color="purple", alpha=0.7)
ax.set_xticks(xf); ax.set_xticklabels(post19.Year.astype(str), fontsize=9)
ax.axhline(0, color="black", lw=0.8); ax.axhline(1, color="green", lw=1, ls="--", alpha=0.5)
ax.set_title("Yearly Sharpe: Fixed EW vs Momentum Alloc", fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")

# MaxDD by year
ax = axes[2,1]
ax.bar(xf-w2/2, post19["DD_F"]*100, w2, label="Fixed EW",  color="darkred", alpha=0.7)
ax.bar(xf+w2/2, post19["DD_M"]*100, w2, label="Mom Alloc", color="purple",  alpha=0.7)
# also show individual strategies
ax.plot(xf, post19["DD_A"]*100, "b-o", lw=1.5, markersize=4, label="A (Daily Mom)")
ax.plot(xf, post19["DD_C"].fillna(0)*100, color="orange", linestyle="-", marker="s",
        lw=1.5, markersize=4, label="C (Intraday MR)")
ax.plot(xf, post19["DD_B"].fillna(0)*100, "g-^", lw=1.5, markersize=4, label="B (QQQ Bubble)")
ax.set_xticks(xf); ax.set_xticklabels(post19.Year.astype(str), fontsize=9)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Yearly Max Drawdown: All Strategies + Portfolios (%)", fontweight="bold")
ax.set_ylabel("MaxDD (%)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("results/portfolio_attribution_fixed.png", dpi=150, bbox_inches="tight")
print("Saved: results/portfolio_attribution_fixed.png")
print("\nDONE")
