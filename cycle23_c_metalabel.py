"""
Queue #27 — Book C meta-labeling filter (Joubert 2022, #98).

PRE-REGISTERED (no tuning, single pass):
  Model     : ridge-logistic (sklearn LogisticRegression, C=1.0, l2), fit on
              per-trade rows; label = total trade P&L > 0.
  Features  : |z| depth, signal-day return, ticker 20d vol, candidate count
              that day, direction, SPY signal-day return. All known at signal.
  CV        : chronological expanding window — refit each Jan 1 on all prior
              trades; first 2 years are burn-in (unsized).
  Sizing    : p<0.40 -> 0, 0.40..0.55 -> 0.5, >0.55 -> 1.0 (fixed a priori).
  Gate      : LW p<0.10 on the sized-vs-baseline daily delta, else rejected.

ANCHOR: reconstructed per-trade loop at C's locked params (sigma=4, flip=3,
lb=20, top5, overlap cap) must reproduce the official module run (+/-0.05,
corr>=0.995) or VOID.
"""
from __future__ import annotations
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from tools.metrics import metrics_from_returns
from tools.significance import sharpe_delta_test

TD, END = 252, "2026-07-08"
t0 = time.time()

hc = pd.read_parquet("data/cache/merged_hourly_close.parquet")
hc.index = pd.to_datetime(hc.index)
ho = pd.read_parquet("data/cache/merged_hourly_open.parquet")
ho.index = pd.to_datetime(ho.index)
daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)

common = sorted(set(hc.columns) & set(ho.columns) & set(daily.columns) - {"SPY"})
s_d, e_d = hc.index[0].date(), hc.index[-1].date()
daily_c = daily[common].ffill().loc[str(s_d):str(e_d)]
hoo, hcc = ho[common].ffill(), hc[common].ffill()
spy_r = daily["SPY"].pct_change()

SIGMA, FLIP, LB, TOPN = 4.0, 3, 20, 5
TCP, BORROW = 0.001, 0.08

daily_ret = daily_c.pct_change()
roll_mean = daily_ret.rolling(LB).mean()
roll_std = daily_ret.rolling(LB).std()
vol20 = daily_ret.rolling(20).std()

day_idx = {}
for ts in hoo.index:
    day_idx.setdefault(ts.date(), []).append(ts)
trading_days_list = sorted(day_idx.keys())
day_to_idx = {d: i for i, d in enumerate(trading_days_list)}
trade_pairs = list(zip(trading_days_list[:-1], trading_days_list[1:]))

TRADING_HOURS = 6.5
daily_borrow = BORROW / TD
hourly_borrow = daily_borrow / TRADING_HOURS

rows = []           # per-trade: exec_date, features..., trade_ret
busy = {}
for sig_date, exec_date in trade_pairs:
    sig_ts = pd.Timestamp(sig_date)
    if sig_ts not in daily_ret.index:
        continue
    r = daily_ret.loc[sig_ts]
    mu = roll_mean.loc[sig_ts]
    sd = roll_std.loc[sig_ts]
    z = ((r - mu) / sd).where(sd > 0).dropna()
    long_c = z[z < -SIGMA].nsmallest(TOPN)
    short_c = z[z > SIGMA].nlargest(TOPN)
    positions = {t: 1 for t in long_c.index}
    positions.update({t: -1 for t in short_c.index})
    _exec_i = day_to_idx.get(exec_date)
    if _exec_i is not None:
        for t in [t for t in positions if busy.get(t, -1) >= _exec_i]:
            positions.pop(t)
    n_cand = len(positions)
    if not positions:
        continue
    bars_exec = day_idx.get(exec_date, [])
    if len(bars_exec) < 2:
        continue
    p1e, p1x, p2e = bars_exec[0], bars_exec[0], bars_exec[1]
    exec_i = day_to_idx[exec_date]
    p2_exit_i = exec_i + FLIP
    if p2_exit_i >= len(trading_days_list):
        continue
    p2_exit_day = trading_days_list[p2_exit_i]
    bars_p2 = day_idx.get(p2_exit_day, [])
    if not bars_p2:
        continue
    p2x = bars_p2[-1]
    spy_day = float(spy_r.get(sig_ts, 0.0))
    for ticker, direction in positions.items():
        try:
            ep1, xp1 = hoo.at[p1e, ticker], hcc.at[p1x, ticker]
            if pd.isna(ep1) or pd.isna(xp1) or ep1 <= 0:
                continue
            p1_borrow = hourly_borrow if direction == -1 else 0.0
            p1_ret = (xp1 / ep1 - 1) * direction - TCP - p1_borrow
            ep2, xp2 = hoo.at[p2e, ticker], hcc.at[p2x, ticker]
            if pd.isna(ep2) or pd.isna(xp2) or ep2 <= 0:
                continue
            p2_borrow = daily_borrow * FLIP if direction == 1 else 0.0
            p2_ret = (xp2 / ep2 - 1) * (-direction) - TCP - p2_borrow
            tr = p1_ret + p2_ret
            busy[ticker] = p2_exit_i
            rows.append(dict(
                exec_date=pd.Timestamp(exec_date),
                zdepth=abs(float(z.get(ticker, 0.0))),
                sig_ret=float(r.get(ticker, 0.0)),
                tvol=float(vol20.loc[sig_ts].get(ticker, np.nan)),
                n_cand=n_cand, direction=direction, spy=spy_day,
                ret=tr))
        except (KeyError, TypeError):
            continue

df = pd.DataFrame(rows).dropna()
print(f"trades: {len(df)}  ({time.time()-t0:.0f}s)")

base_daily = df.groupby("exec_date")["ret"].mean()
# ANCHOR vs official module
from strategies.intraday_mean_reversion import run_intraday_mean_reversion
off, _, _ = run_intraday_mean_reversion(
    daily_close=daily_c, hourly_open=hoo, hourly_close=hcc,
    sigma_grid=[SIGMA], flip_hold_days_grid=[FLIP], lookback_grid=[LB],
    top_n_grid=[TOPN], transaction_cost=TCP, short_borrow_rate=BORROW)
off = off.dropna()
off.index = pd.to_datetime(off.index)
ju = base_daily.index.intersection(off.index)
cc = float(np.corrcoef(base_daily.reindex(ju).fillna(0), off.reindex(ju).fillna(0))[0, 1])
m_my = metrics_from_returns(base_daily.reindex(ju).values, TD)
m_off = metrics_from_returns(off.reindex(ju).values, TD)
print(f"ANCHOR: reconstructed {m_my['sharpe']:.3f} vs official {m_off['sharpe']:.3f} "
      f"(diff {m_my['sharpe']-m_off['sharpe']:+.3f}, corr {cc:.4f})")
if abs(m_my["sharpe"] - m_off["sharpe"]) > 0.05 or cc < 0.995:
    print("ANCHOR FAILED — VOID")
    sys.exit(1)

# ---- meta-label: expanding chronological refit each January ----
def _fit_logistic_l2(X, y, lam=1.0, iters=50):
    """Ridge-logistic via Newton-IRLS; X standardized with intercept col."""
    n, k = X.shape
    w = np.zeros(k)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        W = p * (1 - p)
        reg = lam * np.eye(k)
        reg[0, 0] = 0.0                      # don't penalize intercept
        H = X.T @ (X * W[:, None]) + reg
        g = X.T @ (y - p) - reg @ w
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


FEATS = ["zdepth", "sig_ret", "tvol", "n_cand", "direction", "spy"]
df = df.sort_values("exec_date").reset_index(drop=True)
df["year"] = df["exec_date"].dt.year
years = sorted(df["year"].unique())
df["size"] = 1.0                    # burn-in default: full size
MIN_TRAIN = 50                      # C trades are rare (sigma=4): ~25/yr
for y in years[2:]:
    tr_mask = df["year"] < y
    te_mask = df["year"] == y
    if tr_mask.sum() < MIN_TRAIN:
        continue
    Xtr = df.loc[tr_mask, FEATS].values
    mu_, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = np.column_stack([np.ones(tr_mask.sum()), (Xtr - mu_) / sd_])
    w = _fit_logistic_l2(Xtr, (df.loc[tr_mask, "ret"] > 0).values.astype(float))
    Xte = df.loc[te_mask, FEATS].values
    Xte = np.column_stack([np.ones(te_mask.sum()), (Xte - mu_) / sd_])
    p = 1.0 / (1.0 + np.exp(-Xte @ w))
    df.loc[te_mask, "size"] = np.where(p < 0.40, 0.0, np.where(p < 0.55, 0.5, 1.0))

df["sret"] = df["ret"] * df["size"]
sized_daily = df.groupby("exec_date")["sret"].mean()
jd = base_daily.index
b = base_daily[(jd >= "2019-01-02") & (jd <= END)]
s = sized_daily.reindex(b.index).fillna(0)
m_b = metrics_from_returns(b.values, TD)
m_s = metrics_from_returns(s.values, TD)
r = sharpe_delta_test(s, b)
sig = "  <-- SIGNIFICANT" if r["significant_p10"] else ""
print(f"baseline C : Sharpe {m_b['sharpe']:.3f} MaxDD {m_b['max_dd']:.1%}")
print(f"meta-sized : Sharpe {m_s['sharpe']:.3f} MaxDD {m_s['max_dd']:.1%} "
      f"(delta {r['delta']:+.3f}, p {r['p_one_sided']:.3f}){sig}")
print(f"sizing distribution: {df[df['year']>=years[2]]['size'].value_counts(normalize=True).round(2).to_dict()}")
print(f"total {time.time()-t0:.0f}s")
