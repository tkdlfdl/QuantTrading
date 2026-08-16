"""
4-STRATEGY PORTFOLIO — FIXED WEIGHT + MOMENTUM ALLOCATION
==========================================================
Strategies:
  A: Daily Momentum + 1.25x Leverage + UVXY  (1997-2026)
  B: QQQ Bubble Hourly Momentum              (2020-2026)
  C: Intraday MR + Momentum Flip             (2019-2026)
  D: Contrarian Bubble Score                 (2019-2026)  ← NEW

Allocation methods:
  Fixed EW:  1/N among strategies available on each day
  Momentum:  60-day trailing Sharpe weighted, rebalanced daily

Availability:
  1997-2018:        A only
  2019-2020-07-26:  A + C + D
  2020-07-27+:      A + B + C + D
"""
import warnings, os, sys, time
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
os.makedirs("results", exist_ok=True)
RF = 0.02

# ════════════════════════════════════════════════════════════════
# STRATEGY A — Daily Momentum + Leverage + UVXY (1997-2026)
# ════════════════════════════════════════════════════════════════
print("Building Strategy A: Daily Momentum + Leverage + UVXY...")

close_data = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
lookback = 140; holding = 40; top = 5
close = close_data.copy()
ret_daily = close.pct_change().ffill().fillna(0)
ret_mom   = close.pct_change(lookback).ffill().fillna(0)

raw_A = []
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
            ls = np.sign(ret_mom.iloc[:, ranked[:long_n]].iloc[i-1:i]).abs()
            lr = ls.mul(np.array(ret_daily.iloc[:, ranked[:long_n]].iloc[j:j+1])[0])
            lret = lr.values.mean() * long_n
        if short_n > 0:
            ss = np.sign(ret_mom.iloc[:, ranked[-short_n:]].iloc[i-1:i]).abs() * -1
            sr = ss.mul(np.array(ret_daily.iloc[:, ranked[-short_n:]].iloc[j:j+1])[0])
            sret = sr.values.mean() * short_n
        mom_r = (lret + sret) / top - 0.010 / holding   # 0.5% round-trip per cycle
        h_ret = 0.0
        if "UVXY" in close.columns and pd.notna(close.loc[date, "UVXY"]):
            h_ret = ret_daily.loc[date, "UVXY"]
        elif "^VIX" in close.columns and pd.notna(close.loc[date, "^VIX"]):
            vix_r = ret_daily.loc[date, "^VIX"]
            h_ret = (2.0*vix_r - 0.002 - 0.25*vix_r**2 if date < pd.Timestamp("2018-02-28")
                     else 1.5*vix_r - 0.0015 - 0.25*vix_r**2)
        raw_A.append({"Date": date, "Momentum": mom_r, "Hedge": h_ret})

dfA = pd.DataFrame(raw_A).set_index("Date").dropna()

def calc_bubble_daily(price, ma=120, z=240):
    lp = np.log(np.maximum(price, 1e-6)); f = lp.rolling(ma).mean(); r = lp - f
    return np.tanh(((r - r.rolling(z).mean()) / r.rolling(z).std()) / 2)

base_w = (1 + dfA).cumprod() / (1 + dfA).cumprod().iloc[0]
bub_A  = calc_bubble_daily(base_w["Momentum"], 120, 240)
h_sig  = (bub_A > 0.85).shift(1).fillna(False)
l_sig  = (bub_A < -0.88).shift(1).fillna(False)

retA = []; h_rem = l_rem = 0
for date in dfA.index:
    if h_rem == 0 and h_sig.loc[date]: h_rem = 40
    if l_rem == 0 and l_sig.loc[date]: l_rem = 50
    base = dfA.loc[date, "Momentum"]
    if h_rem > 0:   r = 0.5*base + 0.5*dfA.loc[date, "Hedge"]; h_rem -= 1
    elif l_rem > 0: r = base + 0.25*base - 0.25*(0.1/252);      l_rem -= 1
    else:           r = base
    retA.append(r)
series_A = pd.Series(retA, index=dfA.index)
print(f"  A: {series_A.index[0].date()} to {series_A.index[-1].date()}")

# ════════════════════════════════════════════════════════════════
# STRATEGY C — Intraday MR + Momentum Flip (2019-2026)
# ════════════════════════════════════════════════════════════════
print("Building Strategy C: Intraday MR...")
from strategies.intraday_mean_reversion import run_intraday_mean_reversion

daily_C  = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
hourly_c = pd.read_parquet("data/cache/merged_hourly_close.parquet")
hourly_o = pd.read_parquet("data/cache/merged_hourly_open.parquet")
hourly_c.index = hourly_c.index.floor("h"); hourly_o.index = hourly_o.index.floor("h")
hourly_c = hourly_c[~hourly_c.index.duplicated("last")]
hourly_o = hourly_o[~hourly_o.index.duplicated("last")]
d_cols = set(daily_C.columns); h_cols = set(hourly_c.columns) & set(hourly_o.columns)
valid_C = [c for c in sorted(d_cols & h_cols)
           if hourly_c[c].isna().mean() < 0.30 and hourly_o[c].isna().mean() < 0.30]
s_C, e_C = hourly_c.index[0].date(), hourly_c.index[-1].date()
best_C, _, _ = run_intraday_mean_reversion(
    daily_close  = daily_C[valid_C].ffill().loc[str(s_C):str(e_C)],
    hourly_open  = hourly_o[valid_C].ffill(),
    hourly_close = hourly_c[valid_C].ffill(),
    sigma_grid=[4.0], flip_hold_days_grid=[3], lookback_grid=[20], top_n_grid=[5],
    transaction_cost=0.0025, short_borrow_rate=0.08)   # 0.25% per phase
series_C = best_C.rename("IntradayMR")
print(f"  C: {series_C.index[0].date()} to {series_C.index[-1].date()}")

# ════════════════════════════════════════════════════════════════
# STRATEGY B — QQQ Bubble Hourly Momentum (2020-2026)
# ════════════════════════════════════════════════════════════════
print("Building Strategy B: QQQ Bubble Hourly...")
qqq_raw = pd.read_parquet("data/cache/qqq_hourly_close.parquet")["QQQ"]
stk_raw = pd.read_parquet("data/cache/merged_hourly_close.parquet")
qqq_h = qqq_raw.copy(); qqq_h.index = qqq_h.index.floor("h")
stk_h = stk_raw.copy(); stk_h.index = stk_h.index.floor("h")
qqq_h = qqq_h[~qqq_h.index.duplicated("last")]
stk_h = stk_h[~stk_h.index.duplicated("last")]
idx_B = qqq_h.index.intersection(stk_h.index)
qqq_B = qqq_h.loc[idx_B]
stk_B = stk_h.loc[idx_B]
valid_B = [c for c in stk_B.columns if c in d_cols and stk_B[c].isna().mean() < 0.30]
stk_B = stk_B[valid_B].ffill()
ret_B = stk_B.pct_change().clip(-0.10, 0.10).fillna(0).values.astype(np.float32)
logB  = np.log1p(ret_B.clip(-0.10, 0.10)); cumlogB = np.cumsum(logB, axis=0); n_B = len(idx_B)

def bub_score_h(price, ma_w=500):
    lp = np.log(price); f = price.rolling(ma_w).mean(); r = lp - np.log(f)
    return np.tanh(((r - r.rolling(ma_w).mean()) / r.rolling(ma_w).std()) / 2)

bub_B = bub_score_h(qqq_B, 500).fillna(0).values
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
# Deduct 0.25% one-way TC (0.5% round-trip) amortized over 52h hold
daily_B = daily_B - (0.0025 / 52) * (daily_B != 0).astype(float)
series_B = daily_B.rename("QQQ_Bubble")
print(f"  B: {series_B.index[0].date()} to {series_B.index[-1].date()}")

# ════════════════════════════════════════════════════════════════
# STRATEGY D — Contrarian Bubble Score (2019-2026)
# ════════════════════════════════════════════════════════════════
print("Building Strategy D: Contrarian Bubble Score (MA=104h, Thr=-0.8, Hold=13h, Top=20)...")

from data.universe import get_universe
sp_nasdaq = set(get_universe())
hc = hourly_c.copy(); ho = hourly_o.copy()
idx_D = hc.index.intersection(ho.index)
hc = hc.loc[idx_D]; ho = ho.loc[idx_D]
valid_D = [c for c in hc.columns if c in sp_nasdaq]
hc_D = hc[valid_D].ffill(); ho_D = ho[valid_D].ffill()
T_D = len(idx_D); U_D = len(valid_D)

prices_D = hc_D.values.astype(np.float32)
opens_D  = ho_D.values.astype(np.float32)

# Daily infrastructure
bar_day_D    = idx_D.normalize().values
tdays_D      = np.unique(bar_day_D)
day2int_D    = {d: i for i, d in enumerate(tdays_D)}
bar_day_int_D = np.array([day2int_D[d] for d in bar_day_D], dtype=np.int32)
D_D = len(tdays_D)
day_last_D  = np.zeros(D_D, dtype=np.int32)
day_first_D = np.zeros(D_D, dtype=np.int32)
for t in range(T_D):      day_last_D[bar_day_int_D[t]]  = t
for t in range(T_D-1,-1,-1): day_first_D[bar_day_int_D[t]] = t
daily_close_D = prices_D[day_last_D]
daily_ret_cc_D = np.zeros((D_D, U_D), dtype=np.float32)
daily_ret_cc_D[1:] = daily_close_D[1:] / np.maximum(daily_close_D[:-1], 1e-8) - 1
daily_ret_cc_D = np.clip(daily_ret_cc_D, -0.20, 0.20)

# Bubble score (MA=104h)
MA_D = 104
df_p = pd.DataFrame(prices_D, index=idx_D, columns=valid_D)
lp   = np.log(df_p.replace(0, np.nan).ffill())
fair = df_p.rolling(MA_D, min_periods=MA_D//2).mean()
res  = lp - np.log(fair.replace(0, np.nan))
z    = (res - res.rolling(MA_D, min_periods=MA_D//2).mean()) \
       / res.rolling(MA_D, min_periods=MA_D//2).std()
bub_D = np.tanh(z/2).fillna(0).values.astype(np.float32)

# Run best combo (Thr=-0.8, Hold=13h, Top=20)
THR_D = -0.8; HOLD_D = 13; TOPN_D = 20; TC = 0.0025   # 0.25% one-way (0.5% round-trip)
warmup_D = MA_D + 1
free_at_D = np.zeros(U_D, dtype=np.int32)
daily_num_D = np.zeros(D_D, dtype=np.float64)
daily_den_D = np.zeros(D_D, dtype=np.float64)

for t in range(warmup_D, T_D - HOLD_D - 1):
    scores = bub_D[t]
    avail  = (scores < THR_D) & (free_at_D <= t)
    if not avail.any(): continue
    avail_idx = np.where(avail)[0]
    n_pick    = min(TOPN_D, len(avail_idx))
    chosen    = avail_idx[np.argpartition(scores[avail_idx], n_pick-1)[:n_pick]]
    eb  = t + 1; xb = min(t + HOLD_D, T_D - 1)
    ed  = bar_day_int_D[eb]; xd = bar_day_int_D[xb]
    days = np.arange(ed, xd + 1)
    for s in chosen:
        ep = opens_D[eb, s]; xp = prices_D[xb, s]
        if ep <= 0 or xp <= 0 or not (np.isfinite(ep) and np.isfinite(xp)): continue
        dr = daily_ret_cc_D[days, s].copy()
        dc = prices_D[day_last_D[ed], s]
        dr[0] = (dc / ep - 1) if dc > 0 else 0.0
        if len(days) > 1:
            pc = prices_D[day_last_D[xd-1], s]
            dr[-1] = (xp / pc - 1) if pc > 0 else 0.0
        daily_num_D[days] += np.clip(dr, -0.20, 0.20)
        daily_den_D[days] += 1.0
    free_at_D[chosen] = xb

active_D = daily_den_D > 0
port_D   = np.zeros(D_D, dtype=np.float64)
port_D[active_D] = daily_num_D[active_D] / daily_den_D[active_D] - TC / HOLD_D
series_D = pd.Series(port_D, index=pd.to_datetime(tdays_D)).rename("Contrarian")
print(f"  D: {series_D.index[0].date()} to {series_D.index[-1].date()}")

# ════════════════════════════════════════════════════════════════
# ALIGN ALL STRATEGIES
# ════════════════════════════════════════════════════════════════
print("\nAligning strategies on common daily index...")
all_dates = series_A.index
sA = series_A
sB = series_B.reindex(all_dates).fillna(0)
sC = series_C.reindex(all_dates).fillna(0)
sD = series_D.reindex(all_dates).fillna(0)

avail_B = pd.Series(all_dates >= pd.Timestamp("2020-07-27"), index=all_dates)
avail_C = pd.Series(all_dates >= pd.Timestamp("2019-01-02"), index=all_dates)
avail_D = pd.Series(all_dates >= pd.Timestamp("2019-01-02"), index=all_dates)
n_avail = (1 + avail_B.astype(int) + avail_C.astype(int) + avail_D.astype(int))

# ── Fixed Equal Weight ────────────────────────────────────────
wFA = pd.Series(1.0 / n_avail, index=all_dates)
wFB = avail_B.astype(float) / n_avail
wFC = avail_C.astype(float) / n_avail
wFD = avail_D.astype(float) / n_avail
series_fixed = wFA*sA + wFB*sB + wFC*sC + wFD*sD

# ── Momentum Allocation (60-day rolling Sharpe) ───────────────
def rolling_sh(s, w=60):
    rm = s.rolling(w, min_periods=10).mean()
    rs = s.rolling(w, min_periods=10).std()
    return (rm / rs * np.sqrt(252)).fillna(0)

shA = rolling_sh(sA).clip(lower=0)
shB = rolling_sh(sB).clip(lower=0) * avail_B
shC = rolling_sh(sC).clip(lower=0) * avail_C
shD = rolling_sh(sD).clip(lower=0) * avail_D
tot = shA + shB + shC + shD
wMA = (shA / tot.replace(0, np.nan)).fillna(wFA)
wMB = (shB / tot.replace(0, np.nan)).fillna(wFB)
wMC = (shC / tot.replace(0, np.nan)).fillna(wFC)
wMD = (shD / tot.replace(0, np.nan)).fillna(wFD)
series_mom = wMA*sA + wMB*sB + wMC*sC + wMD*sD

qqq_ret = close_data["QQQ"].pct_change().fillna(0).reindex(all_dates).fillna(0) \
          if "QQQ" in close_data.columns else pd.Series(0, index=all_dates)

# ════════════════════════════════════════════════════════════════
# PERFORMANCE METRICS
# ════════════════════════════════════════════════════════════════
def perf(s, label=""):
    if len(s) < 20: return {}
    years = len(s) / 252; rf_d = RF / 252
    exc = s - rf_d; std = s.std()
    sh  = exc.mean() / std * np.sqrt(252) if std > 0 else 0
    dn  = s[s < 0].std(ddof=0)
    so  = exc.mean() / dn  * np.sqrt(252) if dn  > 0 else 0
    w   = (1 + s).cumprod()
    dd  = (w / w.cummax() - 1).min()
    tr  = w.iloc[-1] - 1
    ar  = (1 + tr) ** (1/years) - 1 if tr > -1 else -1
    pos = sum((1+s[s.index.year==y]).prod()-1>0 for y in s.index.year.unique())
    tot = len(s.index.year.unique())
    return dict(label=label, ann_ret=ar, sharpe=sh, sortino=so,
                maxdd=dd, total_ret=tr, pos_years=f"{pos}/{tot}")

strategies = {
    "A: Daily Mom+Lev+UVXY": sA,
    "B: QQQ Bubble":         sB,
    "C: Intraday MR":        sC,
    "D: Contrarian Bubble":  sD,
    "Fixed EW (A+B+C+D)":   series_fixed,
    "Mom Alloc (A+B+C+D)":  series_mom,
    "QQQ B&H":              qqq_ret,
}

print(f"\n{'='*120}")
print("OVERALL PERFORMANCE — ALL STRATEGIES + PORTFOLIOS")
print(f"{'='*120}")
print(f"  {'Strategy':<28}  {'Ann Ret':>9}  {'Sharpe':>8}  {'Sortino':>9}  "
      f"{'MaxDD':>9}  {'Total Ret':>12}  {'Pos Yrs':>9}")
print("  " + "-"*110)
for name, s in strategies.items():
    p = perf(s, name)
    if not p: continue
    print(f"  {name:<28}  {p['ann_ret']:>9.2%}  {p['sharpe']:>8.4f}  "
          f"{p['sortino']:>9.4f}  {p['maxdd']:>9.2%}  "
          f"{p['total_ret']:>12.2%}  {p['pos_years']:>9}")

# ── Yearly breakdown ──────────────────────────────────────────
print(f"\n{'='*140}")
print("YEARLY RETURN")
print(f"{'='*140}")
print(f"  {'Year':<6}  {'A':>9}  {'B':>9}  {'C':>9}  {'D':>9}  "
      f"{'Fixed EW':>10}  {'Mom Alloc':>10}  {'QQQ':>9}  {'Avail'}")
print("  " + "-"*130)
for yr in range(1997, 2027):
    if not any(all_dates.year == yr): continue
    def yr_ret(s): return (1+s[s.index.year==yr]).prod()-1
    rA=yr_ret(sA); rB=yr_ret(sB); rC=yr_ret(sC); rD=yr_ret(sD)
    rF=yr_ret(series_fixed); rM=yr_ret(series_mom); rQ=yr_ret(qqq_ret)
    avail = "A"
    if yr >= 2019: avail = "A+C+D"
    if yr >= 2021: avail = "A+B+C+D"
    def p(v): return f"{v*100:>8.1f}%"
    print(f"  {yr:<6}  {p(rA)}  {p(rB)}  {p(rC)}  {p(rD)}  "
          f"  {p(rF)}  {p(rM):>10}  {p(rQ)}  {avail}")

print(f"\n{'='*140}")
print("YEARLY SHARPE")
print(f"{'='*140}")
print(f"  {'Year':<6}  {'A':>8}  {'B':>8}  {'C':>8}  {'D':>8}  "
      f"{'Fixed EW':>10}  {'Mom Alloc':>10}  {'QQQ':>9}")
print("  " + "-"*110)
def yr_sharpe(s, yr):
    ys = s[s.index.year == yr]
    return ys.mean()/ys.std()*np.sqrt(252) if len(ys)>1 and ys.std()>0 else 0.0
for yr in range(1997, 2027):
    if not any(all_dates.year == yr): continue
    shA=yr_sharpe(sA,yr); shB=yr_sharpe(sB,yr); shC=yr_sharpe(sC,yr)
    shD=yr_sharpe(sD,yr); shF=yr_sharpe(series_fixed,yr)
    shM=yr_sharpe(series_mom,yr); shQ=yr_sharpe(qqq_ret,yr)
    def s(v): return f"{v:>8.3f}"
    print(f"  {yr:<6}  {s(shA)}  {s(shB)}  {s(shC)}  {s(shD)}  "
          f"{s(shF):>10}  {s(shM):>10}  {s(shQ):>9}")

print(f"\n{'='*140}")
print("YEARLY MAX DRAWDOWN")
print(f"{'='*140}")
print(f"  {'Year':<6}  {'A':>9}  {'B':>9}  {'C':>9}  {'D':>9}  "
      f"{'Fixed EW':>10}  {'Mom Alloc':>10}  {'QQQ':>9}")
print("  " + "-"*120)
def yr_dd(s, yr):
    ys = s[s.index.year == yr]
    if len(ys) < 2: return 0.0
    w = (1+ys).cumprod(); return float((w/w.cummax()-1).min())
for yr in range(1997, 2027):
    if not any(all_dates.year == yr): continue
    ddA=yr_dd(sA,yr); ddB=yr_dd(sB,yr); ddC=yr_dd(sC,yr)
    ddD=yr_dd(sD,yr); ddF=yr_dd(series_fixed,yr)
    ddM=yr_dd(series_mom,yr); ddQ=yr_dd(qqq_ret,yr)
    def d(v): return f"{v*100:>8.1f}%"
    print(f"  {yr:<6}  {d(ddA)}  {d(ddB)}  {d(ddC)}  {d(ddD)}  "
          f"  {d(ddF)}  {d(ddM):>10}  {d(ddQ):>9}")

# ── Contribution (2019+) ──────────────────────────────────────
print(f"\n{'='*140}")
print("FIXED EW — YEARLY CONTRIBUTION per STRATEGY (2019+)")
print(f"{'='*140}")
print(f"  {'Year':<6}  {'Wt_A':>6} {'Wt_B':>6} {'Wt_C':>6} {'Wt_D':>6}  "
      f"{'Cont_A':>9} {'Cont_B':>9} {'Cont_C':>9} {'Cont_D':>9}  "
      f"{'Arith':>9}  {'Geo':>9}")
print("  " + "-"*120)
for yr in range(2019, 2027):
    if not any(all_dates.year == yr): continue
    mask = all_dates.year == yr
    wA_y=wFA[mask].mean(); wB_y=wFB[mask].mean()
    wC_y=wFC[mask].mean(); wD_y=wFD[mask].mean()
    cA=(wFA*sA)[mask].sum(); cB=(wFB*sB)[mask].sum()
    cC=(wFC*sC)[mask].sum(); cD=(wFD*sD)[mask].sum()
    arith = cA+cB+cC+cD
    geo   = (1+series_fixed[mask]).prod()-1
    print(f"  {yr:<6}  {wA_y:>6.2f} {wB_y:>6.2f} {wC_y:>6.2f} {wD_y:>6.2f}  "
          f"  {cA*100:>8.1f}%  {cB*100:>8.1f}%  {cC*100:>8.1f}%  {cD*100:>8.1f}%  "
          f"  {arith*100:>8.1f}%  {geo*100:>8.1f}%")

print(f"\n{'='*140}")
print("MOMENTUM ALLOC — YEARLY CONTRIBUTION per STRATEGY (2019+)")
print(f"{'='*140}")
print(f"  {'Year':<6}  {'Wt_A':>6} {'Wt_B':>6} {'Wt_C':>6} {'Wt_D':>6}  "
      f"{'Cont_A':>9} {'Cont_B':>9} {'Cont_C':>9} {'Cont_D':>9}  "
      f"{'Arith':>9}  {'Geo':>9}")
print("  " + "-"*120)
for yr in range(2019, 2027):
    if not any(all_dates.year == yr): continue
    mask = all_dates.year == yr
    wA_y=wMA[mask].mean(); wB_y=wMB[mask].mean()
    wC_y=wMC[mask].mean(); wD_y=wMD[mask].mean()
    cA=(wMA*sA)[mask].sum(); cB=(wMB*sB)[mask].sum()
    cC=(wMC*sC)[mask].sum(); cD=(wMD*sD)[mask].sum()
    arith = cA+cB+cC+cD
    geo   = (1+series_mom[mask]).prod()-1
    print(f"  {yr:<6}  {wA_y:>6.2f} {wB_y:>6.2f} {wC_y:>6.2f} {wD_y:>6.2f}  "
          f"  {cA*100:>8.1f}%  {cB*100:>8.1f}%  {cC*100:>8.1f}%  {cD*100:>8.1f}%  "
          f"  {arith*100:>8.1f}%  {geo*100:>8.1f}%")

# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("4-Strategy Portfolio: Fixed EW vs Momentum Allocation (1997-2026)",
             fontsize=13, fontweight="bold")

clr = {"A":"steelblue","B":"green","C":"orange","D":"crimson",
       "Fixed EW":"darkred","Mom Alloc":"purple","QQQ":"gray"}

# 1. Wealth curves (full period, log)
ax = axes[0,0]
for name, s, c in [("A",sA,"steelblue"),("B",sB,"green"),("C",sC,"orange"),
                   ("D",sD,"crimson"),("Fixed EW",series_fixed,"darkred"),
                   ("Mom Alloc",series_mom,"purple"),("QQQ",qqq_ret,"gray")]:
    w = (1+s).cumprod()
    lw = 2.5 if "Alloc" in name or "Fixed" in name else 1.4
    ls = "--" if name == "QQQ" else "-"
    ax.plot(w.index, w.values, lw=lw, ls=ls, color=c, label=name, alpha=0.85)
ax.set_yscale("log"); ax.set_title("Cumulative Wealth (log)", fontweight="bold")
ax.set_ylabel("Wealth Multiple"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 2. Wealth 2019+ (normalized)
ax = axes[0,1]
start = pd.Timestamp("2019-01-01")
for name, s, c in [("A",sA,"steelblue"),("B",sB,"green"),("C",sC,"orange"),
                   ("D",sD,"crimson"),("Fixed EW",series_fixed,"darkred"),
                   ("Mom Alloc",series_mom,"purple"),("QQQ",qqq_ret,"gray")]:
    sz = s[s.index >= start]
    if len(sz) == 0: continue
    w = (1+sz).cumprod(); w = w / w.iloc[0]
    lw = 2.5 if "Alloc" in name or "Fixed" in name else 1.4
    ax.plot(w.index, w.values, lw=lw, color=c, label=name, alpha=0.85)
ax.set_title("Normalized Wealth 2019+ (all strategies)", fontweight="bold")
ax.set_ylabel("Normalized Wealth"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 3. Yearly returns 2019+
ax = axes[1,0]
years_plot = list(range(2019, 2027))
x = np.arange(len(years_plot)); w_b = 0.12
for k, (name, s, c) in enumerate([("A",sA,"steelblue"),("B",sB,"green"),
                                    ("C",sC,"orange"),("D",sD,"crimson"),
                                    ("Fixed EW",series_fixed,"darkred"),
                                    ("Mom Alloc",series_mom,"purple"),
                                    ("QQQ",qqq_ret,"gray")]):
    yrets = [(1+s[s.index.year==y]).prod()-1 for y in years_plot]
    ax.bar(x + k*w_b - 3*w_b, [r*100 for r in yrets], w_b,
           label=name, color=c, alpha=0.75)
ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years_plot], fontsize=8)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Yearly Returns 2019-2026 (%)", fontweight="bold")
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3, axis="y")

# 4. Drawdown comparison
ax = axes[1,1]
for name, s, c in [("A",sA,"steelblue"),("D",sD,"crimson"),
                   ("Fixed EW",series_fixed,"darkred"),
                   ("Mom Alloc",series_mom,"purple")]:
    w = (1+s).cumprod(); dd = w/w.cummax()-1
    ax.plot(dd.index, dd.values*100, lw=1.5, color=c, label=name, alpha=0.8)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Max Drawdown Comparison", fontweight="bold")
ax.set_ylabel("Drawdown (%)"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/portfolio_4strategy_tc025.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: results/portfolio_4strategy_tc025.png")

# Save yearly summary
rows = []
for yr in range(1997, 2027):
    if not any(all_dates.year == yr): continue
    avail = "A" if yr < 2019 else ("A+C+D" if yr < 2021 else "A+B+C+D")
    rows.append(dict(Year=yr, Available=avail,
        Ret_A=yr_ret(sA), Sh_A=yr_sharpe(sA,yr), DD_A=yr_dd(sA,yr),
        Ret_B=yr_ret(sB), Sh_B=yr_sharpe(sB,yr), DD_B=yr_dd(sB,yr),
        Ret_C=yr_ret(sC), Sh_C=yr_sharpe(sC,yr), DD_C=yr_dd(sC,yr),
        Ret_D=yr_ret(sD), Sh_D=yr_sharpe(sD,yr), DD_D=yr_dd(sD,yr),
        Ret_Fixed=yr_ret(series_fixed), Sh_Fixed=yr_sharpe(series_fixed,yr),
        DD_Fixed=yr_dd(series_fixed,yr),
        Ret_Mom=yr_ret(series_mom), Sh_Mom=yr_sharpe(series_mom,yr),
        DD_Mom=yr_dd(series_mom,yr),
        Ret_QQQ=yr_ret(qqq_ret)))
pd.DataFrame(rows).to_csv("results/portfolio_4strategy_tc025_yearly.csv", index=False)
print("Saved: results/portfolio_4strategy_tc025_yearly.csv")
print(f"\nTotal runtime: {time.time()-t0:.1f}s  |  DONE")
