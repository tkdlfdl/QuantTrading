"""
YEARLY SHARPE + CONTRIBUTION ANALYSIS
=======================================
For each year: individual strategy Sharpe + contribution to portfolio
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, os, sys
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Load pre-generated strategy returns ──────────────────────────
# Re-run strategy generation (fast — reuse same code blocks)

print("Loading strategy return series...")

# ── Strategy A ────────────────────────────────────────────────────
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
print(f"  A: {series_A.index[0].date()} to {series_A.index[-1].date()}")

# ── Strategy C ────────────────────────────────────────────────────
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
print(f"  C: {series_C.index[0].date()} to {series_C.index[-1].date()}")

# ── Strategy B ────────────────────────────────────────────────────
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
stk_B = stk_B[valid_B].ffill()
ret_B = stk_B.pct_change().clip(-0.10, 0.10).fillna(0).values.astype(np.float32)
logB  = np.log1p(ret_B.clip(-0.10, 0.10)); cumlogB = np.cumsum(logB, axis=0)
n_B   = len(idx_B)

def bub_score(price, ma_w=500):
    lp = np.log(price); f = price.rolling(ma_w).mean(); r = lp - np.log(f)
    return np.tanh(((r - r.rolling(ma_w).mean()) / r.rolling(ma_w).std()) / 2)

bub_B  = bub_score(qqq_B, 500).fillna(0).values
mom_B  = stk_B.pct_change(40).fillna(0).values.astype(np.float32)
fwd_B  = np.zeros((n_B, len(valid_B)), dtype=np.float32)
fwd_B[:n_B-52] = np.expm1(cumlogB[52:] - cumlogB[:n_B-52])
hrB = np.zeros(n_B)
i = 545
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
print(f"  B: {series_B.index[0].date()} to {series_B.index[-1].date()}")

# ── Align to common date index ────────────────────────────────────
all_dates = series_A.index
sA = series_A
sC = series_C.reindex(all_dates).fillna(0)
sB = series_B.reindex(all_dates).fillna(0)

avail_A = pd.Series(True,  index=all_dates)
avail_C = pd.Series(all_dates >= pd.Timestamp("2019-01-02"), index=all_dates)
avail_B = pd.Series(all_dates >= pd.Timestamp("2020-07-27"), index=all_dates)

# ── Fixed Equal Weight ────────────────────────────────────────────
n_avail    = avail_A.astype(int) + avail_C.astype(int) + avail_B.astype(int)
wFA = 1.0 / n_avail
wFC = avail_C.astype(float) / n_avail
wFB = avail_B.astype(float) / n_avail
series_fixed = wFA*sA + wFC*sC + wFB*sB

# ── Momentum Allocation ───────────────────────────────────────────
def rolling_sh(s, w=60):
    rm = s.rolling(w, min_periods=10).mean()
    rs = s.rolling(w, min_periods=10).std()
    return (rm / rs * np.sqrt(252)).fillna(0)

shA = rolling_sh(sA).clip(lower=0) * avail_A
shC = rolling_sh(sC).clip(lower=0) * avail_C
shB = rolling_sh(sB).clip(lower=0) * avail_B
tot = shA + shC + shB

wMA = (shA / tot.replace(0, np.nan)).fillna(wFA)
wMC = (shC / tot.replace(0, np.nan)).fillna(wFC)
wMB = (shB / tot.replace(0, np.nan)).fillna(wFB)
series_mom = wMA*sA + wMC*sC + wMB*sB

# ════════════════════════════════════════════════════════════════
# YEARLY SHARPE + CONTRIBUTION TABLE
# ════════════════════════════════════════════════════════════════
print("\n" + "="*140)
print("YEARLY SHARPE  (each strategy, using daily returns, annualised at sqrt(252))")
print("="*140)
print(f"  {'Year':<6}  {'A Sharpe':>10}  {'C Sharpe':>10}  {'B Sharpe':>10}  "
      f"{'Fixed Sh':>10}  {'Mom Sh':>10}  {'QQQ Sh':>9}  {'Avail':>8}")
print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*8}")

qqq_ret = (close_data["QQQ"].pct_change().fillna(0).reindex(all_dates).fillna(0)
           if "QQQ" in close_data.columns else pd.Series(0, index=all_dates))

def yr_sharpe(s, yr):
    ys = s[s.index.year == yr]
    if len(ys) < 2 or ys.std() == 0: return 0.0
    return float(ys.mean() / ys.std() * np.sqrt(252))

rows = []
for yr in range(1997, 2027):
    if not any(all_dates.year == yr): continue
    shA_y = yr_sharpe(sA, yr)
    shC_y = yr_sharpe(sC, yr) if yr >= 2019 else np.nan
    shB_y = yr_sharpe(sB, yr) if yr >= 2021 else np.nan
    shF_y = yr_sharpe(series_fixed, yr)
    shM_y = yr_sharpe(series_mom,   yr)
    shQ_y = yr_sharpe(qqq_ret, yr)

    avail = "A" if yr < 2019 else ("A+C" if yr < 2021 else "A+B+C")

    def fmt(v):
        return f"{v:>10.4f}" if np.isfinite(v) else f"{'n/a':>10}"

    print(f"  {yr:<6}  {fmt(shA_y)}  {fmt(shC_y)}  {fmt(shB_y)}  "
          f"{fmt(shF_y)}  {fmt(shM_y)}  {shQ_y:>9.4f}  {avail:>8}")
    rows.append(dict(Year=yr, Sh_A=shA_y, Sh_C=shC_y, Sh_B=shB_y,
                     Sh_Fixed=shF_y, Sh_Mom=shM_y, Sh_QQQ=shQ_y, Avail=avail))

yr_df = pd.DataFrame(rows)

# ════════════════════════════════════════════════════════════════
# CONTRIBUTION TABLE
# ════════════════════════════════════════════════════════════════
print("\n" + "="*160)
print("YEARLY CONTRIBUTION TO FIXED EQUAL WEIGHT PORTFOLIO  (weight × return per year)")
print("="*160)
print(f"  {'Year':<6}  {'Wt_A':>6}  {'Wt_C':>6}  {'Wt_B':>6}  "
      f"{'Cont_A':>9}  {'Cont_C':>9}  {'Cont_B':>9}  {'Port Ret':>10}  {'Chk':>8}")
print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*10}  {'-'*8}")

contrib_rows = []
for yr in range(1997, 2027):
    mask = all_dates.year == yr
    if not mask.any(): continue

    yA = sA[mask]; yC = sC[mask]; yB = sB[mask]
    yF = series_fixed[mask]
    wA = wFA[mask].mean(); wC = wFC[mask].mean(); wB = wFB[mask].mean()

    rA = (1+yA).prod()-1; rC = (1+yC).prod()-1; rB = (1+yB).prod()-1
    rF = (1+yF).prod()-1

    cA = wA * rA; cC = wC * rC; cB = wB * rB
    chk = cA + cC + cB

    print(f"  {yr:<6}  {wA:>6.2f}  {wC:>6.2f}  {wB:>6.2f}  "
          f"  {cA:>8.2%}  {cC:>8.2%}  {cB:>8.2%}  {rF:>10.2%}  {chk:>8.2%}")
    contrib_rows.append(dict(Year=yr, Wt_A=wA, Wt_C=wC, Wt_B=wB,
                             Cont_A=cA, Cont_C=cC, Cont_B=cB, Port_Ret=rF))

print("\n" + "="*160)
print("YEARLY CONTRIBUTION TO MOMENTUM ALLOCATION PORTFOLIO")
print("="*160)
print(f"  {'Year':<6}  {'Wt_A':>6}  {'Wt_C':>6}  {'Wt_B':>6}  "
      f"{'Cont_A':>9}  {'Cont_C':>9}  {'Cont_B':>9}  {'Port Ret':>10}")
print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*10}")

for yr in range(1997, 2027):
    mask = all_dates.year == yr
    if not mask.any(): continue

    yA = sA[mask]; yC = sC[mask]; yB = sB[mask]
    yM = series_mom[mask]
    wA = wMA[mask].mean(); wC = wMC[mask].mean(); wB = wMB[mask].mean()

    rA = (1+yA).prod()-1; rC = (1+yC).prod()-1; rB = (1+yB).prod()-1
    rM = (1+yM).prod()-1

    cA = wA * rA; cC = wC * rC; cB = wB * rB

    print(f"  {yr:<6}  {wA:>6.2f}  {wC:>6.2f}  {wB:>6.2f}  "
          f"  {cA:>8.2%}  {cC:>8.2%}  {cB:>8.2%}  {rM:>10.2%}")

# ════════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 2, figsize=(20, 18))
fig.suptitle("Yearly Sharpe & Portfolio Contribution Analysis (1997-2026)",
             fontsize=14, fontweight="bold")

yrs = yr_df["Year"].values

# 1. Yearly Sharpe - Strategy A
ax = axes[0,0]
clrs = ["steelblue" if v >= 0 else "tomato" for v in yr_df["Sh_A"].fillna(0)]
ax.bar(yr_df["Year"].astype(str), yr_df["Sh_A"].fillna(0), color=clrs, alpha=0.75)
ax.axhline(0, color="black", lw=0.8); ax.axhline(1, color="green", lw=1, ls="--", alpha=0.6)
ax.set_title("A: Daily Mom+Lev+UVXY — Yearly Sharpe", fontweight="bold")
ax.set_ylabel("Sharpe Ratio"); ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.grid(True, alpha=0.3, axis="y")

# 2. Yearly Sharpe - B and C (where available)
ax = axes[0,1]
valid_C_yrs = yr_df[yr_df["Year"] >= 2019]
valid_B_yrs = yr_df[yr_df["Year"] >= 2021]
x_c = np.arange(len(valid_C_yrs)); w = 0.4
ax.bar(x_c - w/2, valid_C_yrs["Sh_C"].fillna(0), w,
       color=["orange" if v >= 0 else "tomato" for v in valid_C_yrs["Sh_C"].fillna(0)],
       alpha=0.75, label="C: Intraday MR")
# Overlay B
valid_B_in_C = valid_C_yrs[valid_C_yrs["Year"] >= 2021]
offset_map   = {yr: i for i, yr in enumerate(valid_C_yrs["Year"].values)}
x_b = [offset_map[y] for y in valid_B_in_C["Year"].values]
ax.bar([xi + w/2 for xi in x_b], valid_B_in_C["Sh_B"].fillna(0), w,
       color=["green" if v >= 0 else "tomato" for v in valid_B_in_C["Sh_B"].fillna(0)],
       alpha=0.75, label="B: QQQ Bubble")
ax.set_xticks(x_c); ax.set_xticklabels(valid_C_yrs["Year"].astype(str), fontsize=9)
ax.axhline(0, color="black", lw=0.8); ax.axhline(1, color="green", lw=1, ls="--", alpha=0.6)
ax.set_title("B & C — Yearly Sharpe (available years only)", fontweight="bold")
ax.set_ylabel("Sharpe Ratio"); ax.legend(); ax.grid(True, alpha=0.3, axis="y")

# 3. Fixed EW vs Momentum Allocation Sharpe
ax = axes[1,0]
x = np.arange(len(yr_df)); w = 0.35
ax.bar(x - w/2, yr_df["Sh_Fixed"].fillna(0), w,
       color=["darkred" if v >= 0 else "tomato" for v in yr_df["Sh_Fixed"].fillna(0)],
       alpha=0.7, label="Fixed EW")
ax.bar(x + w/2, yr_df["Sh_Mom"].fillna(0), w,
       color=["purple" if v >= 0 else "tomato" for v in yr_df["Sh_Mom"].fillna(0)],
       alpha=0.7, label="Momentum Alloc")
ax.set_xticks(x[::3]); ax.set_xticklabels(yr_df["Year"].values[::3].astype(str), fontsize=8)
ax.axhline(0, color="black", lw=0.8); ax.axhline(1, color="green", lw=1, ls="--", alpha=0.6)
ax.set_title("Portfolio Sharpe: Fixed EW vs Momentum Allocation", fontweight="bold")
ax.set_ylabel("Sharpe Ratio"); ax.legend(); ax.grid(True, alpha=0.3, axis="y")

# 4. Contribution — Fixed EW (stacked bar post-2018)
ax = axes[1,1]
post = [r for r in contrib_rows if r["Year"] >= 2019]
post_df = pd.DataFrame(post)
x = np.arange(len(post_df)); w = 0.6
ax.bar(x, post_df["Cont_A"]*100, w, label="A Contribution", color="steelblue", alpha=0.8)
ax.bar(x, post_df["Cont_C"]*100, w, bottom=post_df["Cont_A"]*100,
       label="C Contribution", color="orange", alpha=0.8)
ax.bar(x, post_df["Cont_B"]*100, w,
       bottom=(post_df["Cont_A"]+post_df["Cont_C"])*100,
       label="B Contribution", color="green", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(post_df["Year"].astype(str), fontsize=9)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Fixed EW: Yearly Contribution per Strategy (2019+)", fontweight="bold")
ax.set_ylabel("Contribution (%)"); ax.legend(); ax.grid(True, alpha=0.3, axis="y")

# 5. Contribution — Momentum Allocation
ax = axes[2,0]
mom_rows = []
for yr in range(2019, 2027):
    mask = all_dates.year == yr
    if not mask.any(): continue
    yA = sA[mask]; yC = sC[mask]; yB = sB[mask]
    rA = (1+yA).prod()-1; rC = (1+yC).prod()-1; rB = (1+yB).prod()-1
    wA = wMA[mask].mean(); wC = wMC[mask].mean(); wB = wMB[mask].mean()
    mom_rows.append(dict(Year=yr, Cont_A=wA*rA, Cont_C=wC*rC, Cont_B=wB*rB,
                         Wt_A=wA, Wt_C=wC, Wt_B=wB))
mom_post = pd.DataFrame(mom_rows)
x = np.arange(len(mom_post))
ax.bar(x, mom_post["Cont_A"]*100, w, label="A Contribution", color="steelblue", alpha=0.8)
ax.bar(x, mom_post["Cont_C"]*100, w, bottom=mom_post["Cont_A"]*100,
       label="C Contribution", color="orange", alpha=0.8)
ax.bar(x, mom_post["Cont_B"]*100, w,
       bottom=(mom_post["Cont_A"]+mom_post["Cont_C"])*100,
       label="B Contribution", color="green", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(mom_post["Year"].astype(str), fontsize=9)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Momentum Alloc: Yearly Contribution per Strategy (2019+)", fontweight="bold")
ax.set_ylabel("Contribution (%)"); ax.legend(); ax.grid(True, alpha=0.3, axis="y")

# 6. Average weight per year — Momentum Allocation
ax = axes[2,1]
ax.stackplot(mom_post["Year"], mom_post["Wt_A"], mom_post["Wt_C"], mom_post["Wt_B"],
             labels=["A: Daily Mom", "C: Intraday MR", "B: QQQ Bubble"],
             colors=["steelblue","orange","green"], alpha=0.8)
ax.set_title("Avg Yearly Weight — Momentum Allocation", fontweight="bold")
ax.set_ylabel("Portfolio Weight"); ax.set_ylim(0, 1)
ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/portfolio_yearly_sharpe_contribution.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: results/portfolio_yearly_sharpe_contribution.png")

# Save combined table
yr_df.to_csv("results/portfolio_yearly_sharpe.csv", index=False)
pd.DataFrame(contrib_rows).to_csv("results/portfolio_yearly_contribution_fixed.csv", index=False)
mom_post.to_csv("results/portfolio_yearly_contribution_mom.csv", index=False)
print("Saved: yearly_sharpe.csv, contribution_fixed.csv, contribution_mom.csv")
print("\nDONE")
