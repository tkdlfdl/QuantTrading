"""
Combined Portfolio: Book A + Book F + Book D + Book C + Book B
==============================================================
Strategies:
  Book A: Daily Momentum + 1.25x Lev + UVXY  lookback=140d, top_n=5, rb=40d | 1997-2026
  Book F: Universe Hourly Momentum            lb=750h, hold=200h, top_n=5    | 2020-2026
  Book D: Contrarian Bubble   MA=104h, thresh=0.8, hold=8h,  top_n=20        | 2019-2026
  Book C: Intraday MR         sigma=4.0, lb=20d, flip=3d, top_n=5            | 2019-2026
  Book B: QQQ Bubble          MA=200h, Z=100h, buy=0.8, hold=24h             | 2020-2026

Portfolios:
  1. Fixed Equal Weight  : equal weight among available strategies
  2. Momentum Allocation : proportional to rolling return, max 50% per strategy
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

from strategies.contrarian_bubble_hourly import run_contrarian_bubble_hourly
from strategies.intraday_mean_reversion  import run_intraday_mean_reversion
from strategies.qqq_bubble_hourly        import run_qqq_bubble_hourly

TRADING_DAYS = 252
OUT_DIR = Path("results"); OUT_DIR.mkdir(exist_ok=True)

# Book A locked params (matches live/config.py PARAMS["A"])
A_PARAMS = dict(
    lookback_days=140, rebalance_days=40, top_n=5,
    bubble_ma_days=120, bubble_z_days=240,
    lev_threshold=-0.88, lev_mult=0.25, lev_hold_days=50,
    hedge_threshold=0.85, hedge_alloc=0.50, hedge_hold_days=40,
    tc_per_cycle=0.010,
    lev_cost_ann=0.10,
)


def _sharpe(s):
    std = s.std()
    return float(np.sqrt(TRADING_DAYS) * s.mean() / std) if std > 0 else np.nan

def _mdd(s):
    w = (1 + s).cumprod()
    return float((w / w.cummax() - 1).min())

def _cagr(s):
    w = (1 + s).cumprod()
    years = len(s) / TRADING_DAYS
    return float(w.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan

def _stats(s):
    ds = s[s < 0].std()
    so = float(np.sqrt(TRADING_DAYS) * s.mean() / ds) if ds > 0 else np.nan
    return dict(Sharpe=_sharpe(s), Sortino=so, CAGR=_cagr(s),
                Total_Return=float((1+s).cumprod().iloc[-1]-1),
                Max_DD=_mdd(s))


def daily_bubble_on_curve(equity, ma_days, z_days):
    """Bubble score on Book A's own equity curve (from live/signals.py)."""
    lp   = np.log(np.maximum(equity, 1e-9))
    fair = lp.rolling(ma_days).mean()
    r    = lp - fair
    z    = (r - r.rolling(z_days).mean()) / r.rolling(z_days).std()
    return np.tanh(z / 2)


def run_book_a(daily_close):
    """
    Replicate replay_A from live/settle.py using A_PARAMS.
    Returns daily return series.
    """
    p = A_PARAMS
    close = daily_close.copy()
    lookback = p["lookback_days"]
    holding  = p["rebalance_days"]
    top      = p["top_n"]

    ret_daily = close.pct_change().ffill().fillna(0)
    ret_mom   = close.pct_change(lookback).ffill().fillna(0)

    rows = []
    for i in range(lookback + 1, len(ret_mom), holding):
        ranking = ret_mom.iloc[i-1:i].rank(axis=1, ascending=False)
        ranked  = np.argsort(ranking.values[0])
        for j in range(i, min(i + holding, len(ret_mom))):
            date = ret_daily.index[j]
            ls = np.sign(ret_mom.iloc[:, ranked[:top]].iloc[i-1:i]).abs()
            lr = ls.mul(np.array(ret_daily.iloc[:, ranked[:top]].iloc[j:j+1])[0])
            lret = lr.values.mean() * top
            mom_r = lret / top - p["tc_per_cycle"] / holding
            h_ret = 0.0
            if "UVXY" in close.columns and pd.notna(close.loc[date, "UVXY"]):
                h_ret = ret_daily.loc[date, "UVXY"]
            elif "^VIX" in close.columns and pd.notna(close.loc[date, "^VIX"]):
                v = ret_daily.loc[date, "^VIX"]
                h_ret = (2.0*v - 0.002 - 0.25*v**2
                         if date < pd.Timestamp("2018-02-28")
                         else 1.5*v - 0.0015 - 0.25*v**2)
            rows.append({"Date": date, "Momentum": mom_r, "Hedge": h_ret})

    dfA = pd.DataFrame(rows).set_index("Date").dropna()
    if dfA.empty:
        return pd.Series(dtype=float)

    base_w = (1 + dfA).cumprod() / (1 + dfA).cumprod().iloc[0]
    bub    = daily_bubble_on_curve(base_w["Momentum"], p["bubble_ma_days"], p["bubble_z_days"])
    h_sig  = (bub > p["hedge_threshold"]).shift(1).fillna(False)
    l_sig  = (bub < p["lev_threshold"]).shift(1).fillna(False)
    lev_cost = p["lev_cost_ann"] / TRADING_DAYS

    out = []; h_rem = l_rem = 0
    for date in dfA.index:
        if h_rem == 0 and h_sig.loc[date]: h_rem = p["hedge_hold_days"]
        if l_rem == 0 and l_sig.loc[date]: l_rem = p["lev_hold_days"]
        base = dfA.loc[date, "Momentum"]
        if h_rem > 0:
            r = (1 - p["hedge_alloc"])*base + p["hedge_alloc"]*dfA.loc[date, "Hedge"]
            h_rem -= 1
        elif l_rem > 0:
            r = base + p["lev_mult"]*base - p["lev_mult"]*lev_cost
            l_rem -= 1
        else:
            r = base
        out.append(r)

    return pd.Series(out, index=dfA.index)


# ---------------------------------------------------------------------------
# Book E -- Reddit Sentiment Long-Only
# ---------------------------------------------------------------------------
E_PARAMS = dict(
    ma_window=15, z_window=40, mild=0.5, extreme=0.6,
    hold_days=8, top_n=5, min_mentions=5,
    sentiment_scale=0.05,
)

def run_book_e(daily_close):
    """Replicate replay_E logic. Returns daily return Series (NaN-free, 0 when inactive)."""
    try:
        import duckdb
    except ImportError:
        print("   [E] duckdb not installed, skipping")
        return pd.Series(dtype=float)

    db_path = Path("data/market_data.duckdb")
    if not db_path.exists():
        print("   [E] sentiment DB not found, skipping")
        return pd.Series(dtype=float)

    con = duckdb.connect(str(db_path), read_only=True)
    sd = con.execute("select date,symbol,weighted_compound,mention_count from sentiment_daily").df()
    con.close()
    if sd.empty:
        return pd.Series(dtype=float)

    p = E_PARAMS
    sd["date"] = pd.to_datetime(sd["date"])
    sent = sd.pivot_table(index="date", columns="symbol", values="weighted_compound", aggfunc="mean")
    ment = sd.pivot_table(index="date", columns="symbol", values="mention_count", aggfunc="sum").fillna(0)

    cal  = pd.bdate_range(sent.index.min(), sent.index.max())
    syms = [s for s in sent.columns if s in daily_close.columns]
    sent = sent.reindex(cal)[syms]
    cum  = (ment.reindex(cal)[syms].fillna(0) > 0).cumsum()
    px   = daily_close.reindex(cal)[syms].ffill()
    ret  = px.pct_change().fillna(0)
    n    = len(cal)

    # Bubble score on sentiment price index
    idx_df = pd.DataFrame(index=cal, columns=syms, dtype=float)
    for s in syms:
        si   = 100.0 * (1 + sent[s].fillna(0).clip(-1, 1) * p["sentiment_scale"]).cumprod()
        lp   = np.log(si.replace(0, np.nan).ffill())
        fair = si.rolling(p["ma_window"]).mean()
        res  = lp - np.log(fair)
        z    = (res - res.rolling(p["z_window"]).mean()) / res.rolling(p["z_window"]).std()
        idx_df[s] = np.tanh(z / 2)

    sig  = idx_df.shift(1)
    warm = p["ma_window"] + p["z_window"] + 1
    tc   = TC_ONE_WAY

    daily = pd.Series(0.0, index=cal)
    i = warm
    while i < n - 1:
        elig  = cum.iloc[i - 1]
        elig  = elig[elig >= p["min_mentions"]].index
        sc    = sig.iloc[i][elig].dropna()
        cap   = sc[sc < -p["extreme"]].nsmallest(p["top_n"])
        momo  = sc[(sc > p["mild"]) & (sc <= p["extreme"])].nlargest(p["top_n"])
        longs = pd.concat([cap, momo]).drop_duplicates()
        end   = min(i + p["hold_days"], n)
        if len(longs) == 0:
            i = end
            continue
        for j in range(i, end):
            r = ret.iloc[j][longs.index].mean()
            if j == i:
                r -= tc
            daily.iloc[j] = r
        i = end

    daily.index = pd.to_datetime(daily.index)
    return daily


# ---------------------------------------------------------------------------
# Book F -- Universe Hourly Momentum (non-overlapping, lb=750h, hold=200h, top5)
# ---------------------------------------------------------------------------
TC_ONE_WAY = 0.001
F_LOOKBACK = 750
F_HOLD     = 200
F_TOP_N    = 5

def run_book_f(hourly_close, hourly_open):
    """Non-overlapping cross-sectional momentum (Book F). Returns daily return Series."""
    hc_np = hourly_close.values.astype(np.float32)
    ho_np = hourly_open.values.astype(np.float32)
    bar_ts = hourly_close.index
    n, n_tick = hc_np.shape

    # Momentum matrix
    mom = np.full((n, n_tick), np.nan, dtype=np.float32)
    for i in range(F_LOOKBACK, n):
        with np.errstate(divide="ignore", invalid="ignore"):
            prev = hc_np[i - F_LOOKBACK]
            mom[i] = np.where(prev > 0, hc_np[i] / prev - 1, np.nan)

    # Position matrix H[bar, ticker] = weight if held
    H          = np.zeros((n, n_tick), dtype=np.float32)
    entry_mask = np.zeros(n, dtype=bool)
    exit_mask  = np.zeros(n, dtype=bool)

    i = F_LOOKBACK
    while i + F_HOLD < n:
        row   = mom[i]
        valid = np.where(np.isfinite(row))[0]
        if len(valid) >= F_TOP_N:
            top_idx = valid[np.argsort(row[valid])[-F_TOP_N:]]
            eb = i + 1; xb = i + F_HOLD
            H[eb:xb+1, top_idx] = 1.0 / F_TOP_N
            entry_mask[eb] = True
            exit_mask[xb]  = True
        i += F_HOLD

    with np.errstate(divide="ignore", invalid="ignore"):
        o2c   = np.where((ho_np > 0) & np.isfinite(ho_np) & np.isfinite(hc_np),
                         hc_np / ho_np - 1, 0.0)
        c_prev = np.vstack([hc_np[:1], hc_np[:-1]])
        c2c    = np.where((c_prev > 0) & np.isfinite(c_prev) & np.isfinite(hc_np),
                          hc_np / c_prev - 1, 0.0)

    bar_ret = np.where(entry_mask[:, None], o2c, c2c)
    port_h  = (H * bar_ret).sum(axis=1)
    tc_adj  = np.zeros(n)
    tc_adj[entry_mask] -= TC_ONE_WAY
    tc_adj[exit_mask]  -= TC_ONE_WAY
    port_h += tc_adj
    port_h[H.sum(axis=1) == 0] = 0.0

    s     = pd.Series(port_h.astype(float), index=bar_ts)
    daily = s.groupby(s.index.normalize()).apply(lambda g: float((1+g).prod() - 1))
    daily.index = pd.to_datetime(daily.index)
    return daily


# ===========================================================================
# STEP 1: Load data
# ===========================================================================
SEP = "=" * 72
print(SEP)
print("STEP 1: Loading data")
print(SEP)

hc_all = pd.read_parquet("data/cache/merged_hourly_close.parquet")
ho_all = pd.read_parquet("data/cache/merged_hourly_open.parquet")
hc_all.index = hc_all.index.floor("h")
ho_all.index = ho_all.index.floor("h")
hc_all = hc_all[~hc_all.index.duplicated(keep="last")]
ho_all = ho_all[~ho_all.index.duplicated(keep="last")]

daily_all = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")

qqq_cache = pd.read_parquet("data/cache/qqq_hourly.parquet")
qqq_hc = qqq_cache["close"].ffill()
qqq_ho = qqq_cache["open"].ffill()

common = sorted(set(hc_all.columns) & set(ho_all.columns) & set(daily_all.columns))
valid  = [c for c in common
          if hc_all[c].isna().mean() < 0.30 and ho_all[c].isna().mean() < 0.30]
hc = hc_all[valid].ffill()
ho = ho_all[valid].ffill()
dc = daily_all[valid].ffill()
dc = dc.loc[str(hc.index[0].date()):str(hc.index[-1].date())]

print(f"Hourly universe : {len(valid)} tickers  {hc.index[0].date()} -> {hc.index[-1].date()}")
print(f"QQQ hourly      : {len(qqq_hc)} bars  {qqq_hc.index[0].date()} -> {qqq_hc.index[-1].date()}")
print(f"Daily extended  : {daily_all.shape[1]} tickers  {daily_all.index[0].date()} -> {daily_all.index[-1].date()}")


# ===========================================================================
# STEP 2: Run each strategy
# ===========================================================================
print("\n" + SEP)
print("STEP 2: Running strategies (best params, single combo each)")
print(SEP)

# Book A - Daily Momentum + Leverage + UVXY
print("\n[A] Daily Momentum lb=140d, top5, rb=40d, lev=1.25x, hedge=UVXY...")
ret_A_full = run_book_a(daily_all)
print(f"   Book A full period: {ret_A_full.index[0].date()} -> {ret_A_full.index[-1].date()}")
print(f"   Book A: Sharpe={_sharpe(ret_A_full):.3f}  CAGR={_cagr(ret_A_full):.1%}  MaxDD={_mdd(ret_A_full):.1%}")

# Book D - Contrarian Bubble
print("\n[D] Contrarian Bubble MA=104h, thresh=0.8, hold=8h, top_n=20...")
ret_D_raw, _, _ = run_contrarian_bubble_hourly(
    hourly_open=ho, hourly_close=hc,
    ma_window_grid=[104], buy_threshold_grid=[0.8],
    hold_hours_grid=[8], top_n_grid=[20],
)
ret_D = ret_D_raw.rename("BookD")
print(f"   Book D: Sharpe={_sharpe(ret_D):.3f}  CAGR={_cagr(ret_D):.1%}  MaxDD={_mdd(ret_D):.1%}")

# Book C - Intraday MR
print("\n[C] Intraday MR sigma=4.0, lookback=20d, flip=3d, top_n=5...")
ret_C_raw, _, _ = run_intraday_mean_reversion(
    daily_close=dc, hourly_open=ho, hourly_close=hc,
    sigma_grid=[4.0], flip_hold_days_grid=[3],
    lookback_grid=[20], top_n_grid=[5],
    transaction_cost=0.001, short_borrow_rate=0.08,
)
ret_C = ret_C_raw.rename("BookC")
print(f"   Book C: Sharpe={_sharpe(ret_C):.3f}  CAGR={_cagr(ret_C):.1%}  MaxDD={_mdd(ret_C):.1%}")

# Book E - Reddit Sentiment
print("\n[E] Reddit Sentiment Long-Only (2024-2026)...")
ret_E_daily = run_book_e(daily_all)
ret_E = ret_E_daily.rename("BookE")
if len(ret_E) > 0:
    print(f"   Book E: Sharpe={_sharpe(ret_E[ret_E != 0]):.3f}  CAGR={_cagr(ret_E[ret_E != 0]):.1%}  MaxDD={_mdd(ret_E[ret_E != 0]):.1%}  Period: {ret_E.index[0].date()} -> {ret_E.index[-1].date()}")

# Book F - Universe Hourly Momentum
print("\n[F] Universe Hourly Momentum lb=750h, hold=200h, top_n=5...")
ret_F_daily = run_book_f(hc, ho)
ret_F = ret_F_daily.rename("BookF")
print(f"   Book F: Sharpe={_sharpe(ret_F):.3f}  CAGR={_cagr(ret_F):.1%}  MaxDD={_mdd(ret_F):.1%}")

# Book B - QQQ Bubble
print("\n[B] QQQ Bubble MA=200h, Z=100h, buy=0.8, hold=24h...")
ret_B_raw, _, _ = run_qqq_bubble_hourly(
    hourly_open=qqq_ho, hourly_close=qqq_hc,
    ma_window_grid=[200], z_window_grid=[100],
    buy_threshold_grid=[0.8], short_threshold_grid=[0.95],
    hold_hours_grid=[24], transaction_cost=0.001,
    short_borrow_rate=0.08, enable_short=False,
)
ret_B = ret_B_raw.rename("BookB")
print(f"   Book B: Sharpe={_sharpe(ret_B):.3f}  CAGR={_cagr(ret_B):.1%}  MaxDD={_mdd(ret_B):.1%}")


# ===========================================================================
# STEP 3: Align to common daily index (2019 onward)
# ===========================================================================
print("\n" + SEP)
print("STEP 3: Aligning date ranges (portfolio starts 2019)")
print(SEP)

dc_idx = ret_D.index.union(ret_C.index)
full_idx = pd.date_range(dc_idx.min(), dc_idx.max(), freq="B")

# Slice Book A to the portfolio window
ret_A = ret_A_full.reindex(full_idx, fill_value=0.0).rename("BookA")
ret_E = ret_E_daily.reindex(full_idx, fill_value=0.0).rename("BookE") if len(ret_E_daily) > 0 else pd.Series(0.0, index=full_idx, name="BookE")
ret_F = ret_F.reindex(full_idx, fill_value=0.0)
ret_D = ret_D.reindex(full_idx, fill_value=0.0)
ret_C = ret_C.reindex(full_idx, fill_value=0.0)
ret_B = ret_B.reindex(full_idx, fill_value=0.0)

b_start  = qqq_hc.index[0].normalize()
b_active = full_idx >= b_start

a_first_nonzero = ret_A_full.index[0].normalize()
a_active = full_idx >= a_first_nonzero

f_first  = ret_F_daily.index[0].normalize()
f_active = full_idx >= f_first

# Book E: active only when sentiment data exists (warmup ~56 days after data start)
if len(ret_E_daily) > 0:
    e_warm   = E_PARAMS["ma_window"] + E_PARAMS["z_window"] + 1
    e_start  = ret_E_daily.index[0]
    e_first  = pd.bdate_range(e_start, periods=e_warm + 1)[-1].normalize()
    e_active = full_idx >= e_first
else:
    e_first  = full_idx[-1]   # never active
    e_active = full_idx >= full_idx[-1] + pd.Timedelta(days=1)

years = len(full_idx) / TRADING_DAYS
print(f"Portfolio period: {full_idx[0].date()} -> {full_idx[-1].date()}  ({years:.2f} yr)")
print(f"Book A active from: {a_first_nonzero.date()}")
print(f"Book E active from: {e_first.date()}")
print(f"Book F active from: {f_first.date()}")
print(f"Book B active from: {b_start.date()}")


# ===========================================================================
# STEP 4: Fixed weight portfolios
# ===========================================================================
print("\n" + SEP)
print("STEP 4: Fixed weight portfolios")
print(SEP)

# Fixed D+C only (50/50) — baseline
port_DC = (0.5 * ret_D + 0.5 * ret_C).rename("Fixed_DC")

# Fixed EW: equal weight among active books (D+C always; A, F, B phased in)
port_EW = pd.Series(0.0, index=full_idx, name="Fixed_EW")
for i, dt in enumerate(full_idx):
    has_a = a_active[i]; has_e = e_active[i]; has_f = f_active[i]; has_b = b_active[i]
    active_n = 2 + int(has_a) + int(has_e) + int(has_f) + int(has_b)
    w = 1.0 / active_n
    port_EW.iloc[i] = (w * ret_D.iloc[i] + w * ret_C.iloc[i]
                       + (w * ret_A.iloc[i] if has_a else 0.0)
                       + (w * ret_E.iloc[i] if has_e else 0.0)
                       + (w * ret_F.iloc[i] if has_f else 0.0)
                       + (w * ret_B.iloc[i] if has_b else 0.0))

for name, port in [("Fixed 50%D+50%C", port_DC),
                   ("Fixed EqualWeight (A+F+D+C+B)", port_EW)]:
    st = _stats(port)
    print(f"\n  {name}")
    print(f"    Sharpe={st['Sharpe']:.3f}  CAGR={st['CAGR']:.1%}  "
          f"Total={st['Total_Return']:+.1%}  MaxDD={st['Max_DD']:.1%}")


# ===========================================================================
# STEP 5: Momentum allocation (max 50% per strategy)
# ===========================================================================
print("\n" + SEP)
print("STEP 5: Momentum allocation grid search (max 50% cap per strategy)")
print(SEP)

MAX_ALLOC     = 0.50
LOOKBACK_GRID  = [20, 30, 60, 90, 120]
REBALANCE_GRID = [1, 5, 10, 20, 60]

n = len(full_idx)
A_vals = ret_A.values; E_vals = ret_E.values; F_vals = ret_F.values
D_vals = ret_D.values; C_vals = ret_C.values; B_vals = ret_B.values
a_act = a_active; e_act = e_active; f_act = f_active; b_act = b_active


def run_mom_alloc(lookback, rebalance):
    port = np.zeros(n)
    wA = wE = wF = wD = wC = wB = 0.0

    for i in range(n):
        use_A = bool(a_act[i]); use_E = bool(e_act[i])
        use_F = bool(f_act[i]); use_B = bool(b_act[i])

        if i >= lookback and i % rebalance == 0:
            roll_D = float(np.prod(1 + D_vals[i-lookback:i]) - 1)
            roll_C = float(np.prod(1 + C_vals[i-lookback:i]) - 1)
            roll_A = (float(np.prod(1 + A_vals[i-lookback:i]) - 1)
                      if use_A and a_act[i - lookback] else 0.0)
            roll_E = (float(np.prod(1 + E_vals[i-lookback:i]) - 1)
                      if use_E and e_act[i - lookback] else 0.0)
            roll_F = (float(np.prod(1 + F_vals[i-lookback:i]) - 1)
                      if use_F and f_act[i - lookback] else 0.0)
            roll_B = (float(np.prod(1 + B_vals[i-lookback:i]) - 1)
                      if use_B and b_act[i - lookback] else 0.0)

            pos_A = max(0.0, roll_A) if use_A else 0.0
            pos_E = max(0.0, roll_E) if use_E else 0.0
            pos_F = max(0.0, roll_F) if use_F else 0.0
            pos_D = max(0.0, roll_D)
            pos_C = max(0.0, roll_C)
            pos_B = max(0.0, roll_B) if use_B else 0.0
            total = pos_A + pos_E + pos_F + pos_D + pos_C + pos_B

            if total > 1e-9:
                rA=pos_A/total; rE=pos_E/total; rF=pos_F/total
                rD=pos_D/total; rC=pos_C/total; rB=pos_B/total

                for _ in range(8):
                    exc = sum(max(0.0, r - MAX_ALLOC) for r in [rA,rE,rF,rD,rC,rB])
                    if exc < 1e-9: break
                    rA=min(rA,MAX_ALLOC); rE=min(rE,MAX_ALLOC); rF=min(rF,MAX_ALLOC)
                    rD=min(rD,MAX_ALLOC); rC=min(rC,MAX_ALLOC); rB=min(rB,MAX_ALLOC)
                    uncapped = sum(r < MAX_ALLOC for r in [rA,rE,rF,rD,rC,rB])
                    if uncapped > 0:
                        each = exc / uncapped
                        if rA < MAX_ALLOC: rA = min(rA + each, MAX_ALLOC)
                        if rE < MAX_ALLOC: rE = min(rE + each, MAX_ALLOC)
                        if rF < MAX_ALLOC: rF = min(rF + each, MAX_ALLOC)
                        if rD < MAX_ALLOC: rD = min(rD + each, MAX_ALLOC)
                        if rC < MAX_ALLOC: rC = min(rC + each, MAX_ALLOC)
                        if rB < MAX_ALLOC: rB = min(rB + each, MAX_ALLOC)

                wA,wE,wF,wD,wC,wB = rA,rE,rF,rD,rC,rB
            else:
                n_active = 2 + int(use_A) + int(use_E) + int(use_F) + int(use_B)
                eq = 1.0 / n_active
                wD = wC = eq
                wA = eq if use_A else 0.0; wE = eq if use_E else 0.0
                wF = eq if use_F else 0.0; wB = eq if use_B else 0.0
        elif i < lookback:
            n_active = 2 + int(use_A) + int(use_E) + int(use_F) + int(use_B)
            eq = 1.0 / n_active
            wD = wC = eq
            wA = eq if use_A else 0.0; wE = eq if use_E else 0.0
            wF = eq if use_F else 0.0; wB = eq if use_B else 0.0

        port[i] = (wA*A_vals[i] + wE*E_vals[i] + wF*F_vals[i]
                   + wD*D_vals[i] + wC*C_vals[i] + wB*B_vals[i])

    return pd.Series(port, index=full_idx)


print(f"Grid: {len(LOOKBACK_GRID)} lookback x {len(REBALANCE_GRID)} rebalance = "
      f"{len(LOOKBACK_GRID)*len(REBALANCE_GRID)} combos")

mom_results = []
best_sh   = -np.inf
best_port = None
best_params = None

for lb, rb in product(LOOKBACK_GRID, REBALANCE_GRID):
    port = run_mom_alloc(lb, rb)
    st   = _stats(port)
    mom_results.append(dict(lookback=lb, rebalance=rb, **st))
    if pd.notna(st["Sharpe"]) and st["Sharpe"] > best_sh:
        best_sh     = st["Sharpe"]
        best_port   = port.rename("MomAlloc_Best")
        best_params = dict(lookback=lb, rebalance=rb)

mom_df = pd.DataFrame(mom_results).sort_values("Sharpe", ascending=False)

print(f"\n{'Lookback':>9} {'Rebal':>6} {'Sharpe':>8} {'CAGR':>7} {'Total':>8} {'MaxDD':>8}")
print("-" * 56)
for _, r in mom_df.iterrows():
    print(f"{int(r['lookback']):>9} {int(r['rebalance']):>6} "
          f"{r['Sharpe']:>8.3f} {r['CAGR']:>7.1%} "
          f"{r['Total_Return']:>+8.1%} {r['Max_DD']:>8.1%}")

print(f"\nSensitivity - Lookback:")
for lb, grp in mom_df.groupby("lookback"):
    print(f"  {lb:>4}d  avg={grp['Sharpe'].mean():.3f}  best={grp['Sharpe'].max():.3f}")

print(f"\nSensitivity - Rebalance:")
for rb, grp in mom_df.groupby("rebalance"):
    print(f"  {rb:>4}d  avg={grp['Sharpe'].mean():.3f}  best={grp['Sharpe'].max():.3f}")


# ===========================================================================
# STEP 6: Summary + yearly breakdown
# ===========================================================================
print("\n" + SEP)
print("STEP 6: Summary")
print(SEP)

portfolios = {
    "Book A only":           ret_A,
    "Book E only":           ret_E.where(e_active, 0.0),
    "Book F only":           ret_F.where(f_active, 0.0),
    "Book D only":           ret_D,
    "Book C only":           ret_C,
    "Book B only":           ret_B.where(b_active, 0.0),
    "Fixed 50/50 D+C":      port_DC,
    "Fixed EW (A+E+F+D+C+B)": port_EW,
    f"MomAlloc lb={best_params['lookback']}d rb={best_params['rebalance']}d": best_port,
}

print(f"\n  {'Strategy':<36} {'Sharpe':>7} {'CAGR':>7} {'Total':>8} {'MaxDD':>8}")
print(f"  {'-'*36} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")
for name, port in portfolios.items():
    st = _stats(port)
    print(f"  {name:<36} {st['Sharpe']:>7.3f} {st['CAGR']:>7.1%} "
          f"{st['Total_Return']:>+8.1%} {st['Max_DD']:>8.1%}")

print(f"\n{SEP}")
print(f"YEARLY BREAKDOWN - MomAlloc (lb={best_params['lookback']}d, rb={best_params['rebalance']}d)  vs  Fixed DC  vs  Books")
print(SEP)
print(f"  {'Year':>6} {'MomAlloc':>10} {'Fixed DC':>10} {'Book A':>8} {'Book F':>8} {'Book D':>8} {'Book C':>8} {'Book B':>8}")
print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

def yr_ret(s, yr): return float((1 + s[s.index.year == yr]).prod() - 1)
def yr_mdd(s, yr):
    sl = s[s.index.year == yr]
    w  = (1 + sl).cumprod()
    return float((w / w.cummax() - 1).min()) if len(w) else 0.0

years_list = sorted(best_port.index.year.unique())

print(f"  {'Year':>6} {'MomAlloc':>10} {'FixedEW':>9} {'Book A':>8} {'Book E':>8} {'Book F':>8} {'Book D':>8} {'Book C':>8} {'Book B':>8}")
print(f"  {'-'*6} {'-'*10} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for yr in years_list:
    flag = " <- B starts" if yr == b_start.year else ""
    print(f"  {yr:>6} {yr_ret(best_port,yr):>+10.2%} {yr_ret(port_EW,yr):>+9.2%}"
          f" {yr_ret(ret_A,yr):>+8.2%} {yr_ret(ret_E,yr):>+8.2%} {yr_ret(ret_F,yr):>+8.2%}"
          f" {yr_ret(ret_D,yr):>+8.2%} {yr_ret(ret_C,yr):>+8.2%} {yr_ret(ret_B,yr):>+8.2%}{flag}")

print(f"\n{SEP}")
print(f"YEARLY MAX DRAWDOWN")
print(SEP)
print(f"  {'Year':>6} {'MomAlloc':>10} {'FixedEW':>9} {'Book A':>8} {'Book E':>8} {'Book F':>8} {'Book D':>8} {'Book C':>8} {'Book B':>8}")
print(f"  {'-'*6} {'-'*10} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for yr in years_list:
    flag = " <- B starts" if yr == b_start.year else ""
    print(f"  {yr:>6} {yr_mdd(best_port,yr):>10.2%} {yr_mdd(port_EW,yr):>9.2%}"
          f" {yr_mdd(ret_A,yr):>8.2%} {yr_mdd(ret_E,yr):>8.2%} {yr_mdd(ret_F,yr):>8.2%}"
          f" {yr_mdd(ret_D,yr):>8.2%} {yr_mdd(ret_C,yr):>8.2%} {yr_mdd(ret_B,yr):>8.2%}{flag}")


# ===========================================================================
# STEP 6b: Correlation matrix
# ===========================================================================
print(f"\n{SEP}")
print("CORRELATION MATRIX (daily returns, 2019-2026)")
print(SEP)

corr_df = pd.DataFrame({
    "A": ret_A, "E": ret_E, "F": ret_F, "D": ret_D, "C": ret_C, "B": ret_B,
    "FixedEW": port_EW, "MomAlloc": best_port,
})
corr = corr_df.corr()

header = f"  {'':>9}" + "".join(f"{c:>10}" for c in corr.columns)
print(header)
print("  " + "-" * (9 + 10 * len(corr.columns)))
for row in corr.index:
    vals = "".join(f"{corr.loc[row, c]:>10.3f}" for c in corr.columns)
    print(f"  {row:>9}{vals}")

# Also print pairwise sorted by absolute correlation (books only)
books_only = corr_df[["A", "E", "F", "D", "C", "B"]].corr()
pairs = []
cols = list(books_only.columns)
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        pairs.append((cols[i], cols[j], books_only.iloc[i, j]))
pairs.sort(key=lambda x: abs(x[2]), reverse=True)
print(f"\n  Pairwise correlations (books only, sorted):")
for a, b, r in pairs:
    bar = "#" * int(abs(r) * 20)
    print(f"    {a} vs {b}: {r:>+.3f}  {bar}")


# ===========================================================================
# STEP 7: Charts
# ===========================================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 13),
                          gridspec_kw={"height_ratios": [3, 1.5, 1.5]})

colors = {
    "Book A only":             "gold",
    "Book E only":             "hotpink",
    "Book F only":             "limegreen",
    "Book D only":             "steelblue",
    "Book C only":             "darkorange",
    "Book B only":             "purple",
    "Fixed 50/50 D+C":        "gray",
    "Fixed EW (A+E+F+D+C+B)": "green",
}
best_label = f"MomAlloc lb={best_params['lookback']}d rb={best_params['rebalance']}d"

for name, port in portfolios.items():
    w  = (1 + port).cumprod(); w /= w.iloc[0]
    lw = 2.5 if "MomAlloc" in name else (1.8 if "Fixed EW" in name else 1.0)
    ls = "-" if ("MomAlloc" in name or "Fixed" in name) else "--"
    col = colors.get(name, "crimson")
    axes[0].plot(w.index, w.values, label=name, lw=lw, ls=ls, color=col)

st_best = _stats(best_port)
axes[0].set_title(
    f"Portfolio: A+F+D+C+B | MomAlloc lb={best_params['lookback']}d "
    f"rb={best_params['rebalance']}d | Sharpe={st_best['Sharpe']:.3f}  "
    f"CAGR={st_best['CAGR']:.1%}  MaxDD={st_best['Max_DD']:.1%}",
    fontsize=10
)
axes[0].set_ylabel("Cumulative Wealth")
axes[0].legend(fontsize=8, ncol=2); axes[0].grid(True, alpha=0.3)

# Drawdown
dd = (1 + best_port).cumprod()
dd = dd / dd.cummax() - 1
axes[1].fill_between(dd.index, dd.values, 0, alpha=0.5, color="crimson",
                     label=f"MomAlloc MaxDD={st_best['Max_DD']:.1%}")
dd_dc = (1 + port_DC).cumprod(); dd_dc = dd_dc / dd_dc.cummax() - 1
axes[1].plot(dd_dc.index, dd_dc.values, color="gray", lw=1.0, label="Fixed DC")
axes[1].set_ylabel("Drawdown"); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

# Yearly bar chart
years_list = sorted(best_port.index.year.unique())
ma_yr = [float((1 + best_port[best_port.index.year==y]).prod()-1) for y in years_list]
dc_yr = [float((1 + port_DC[port_DC.index.year==y]).prod()-1) for y in years_list]
x = np.arange(len(years_list)); bw = 0.35
axes[2].bar(x - bw/2, [r*100 for r in ma_yr], bw, label="MomAlloc", color="crimson", alpha=0.7)
axes[2].bar(x + bw/2, [r*100 for r in dc_yr], bw, label="Fixed DC",  color="gray",   alpha=0.6)
axes[2].axhline(0, color="black", lw=0.8)
axes[2].set_xticks(x); axes[2].set_xticklabels([str(y) for y in years_list])
axes[2].set_ylabel("Return (%)"); axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out = OUT_DIR / "portfolio_abfdc.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
plt.close()
print(f"\n[Chart saved: {out}]")

mom_df.to_excel(OUT_DIR / "portfolio_abfdc_grid.xlsx", index=False)
print(f"[Grid saved: {OUT_DIR / 'portfolio_abfdc_grid.xlsx'}]")

print("\nDONE")
