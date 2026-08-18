"""
Cycle 27 M5 — pooled cross-sectional HistGB classification (brief:
research/briefs/ml_classification_2026-08.md).

Per stock-day features (price-only + market context), per-stock label vs its
OWN trailing 63d mu +/- k*sigma. Pooled training across all names; refit
every 2 years on trailing 8y (row cap 500k, random subsample seed 0);
OOS 2012+. Portfolio: equal-weight long all +1 predictions, short all -1,
daily rebalance; 0.1%/side turnover cost + 8%/yr borrow on short gross.
"""
from __future__ import annotations
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from tools.metrics import metrics_from_returns

TD, TC, BORROW = 252, 0.001, 0.08
t0 = time.time()
rng = np.random.RandomState(0)

daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)
stocks = [c for c in daily.columns if not c.startswith("^") and c not in {"UVXY"}]
px = daily[stocks].ffill()
ret = px.pct_change()
jump = ret.abs().rolling(252).max()          # splice guard

spy = daily["SPY"].dropna()
spy_r = spy.pct_change()
vix = daily["^VIX"].dropna() if "^VIX" in daily.columns else None

# ---- per-stock feature blocks (v2: full technical-indicator set) ----
from ml_features import tech_features, market_features
F = tech_features(px)
F["rvrank"] = ret.rolling(5).sum().rank(axis=1, pct=True)   # 5d cross-sec rank
F["momrank"] = px.pct_change(126).rank(axis=1, pct=True)    # cross-sec mom rank
mu63 = ret.rolling(63).mean()
sd63 = ret.rolling(63).std()

# market context (broadcast)
mk = market_features(px.index, daily, breadth_px=px).ffill()

r_next = ret.shift(-1)
FEATS = list(F.keys()) + list(mk.columns)

from sklearn.ensemble import HistGradientBoostingClassifier

def build_rows(date_mask):
    """Stack (stock, day) rows for the masked dates. Returns X, y-dict, meta."""
    idx = px.index[date_mask]
    blocks = []
    for d in idx:
        ok = (sd63.loc[d] > 0) & ret.loc[d].notna() & (jump.loc[d] <= 1.0)
        names = ok[ok].index
        if len(names) < 50:
            continue
        row = np.column_stack(
            [F[f].loc[d, names].values for f in F] +
            [np.full(len(names), mk.loc[d, c]) for c in mk.columns])
        rn = r_next.loc[d, names].values
        m_, s_ = mu63.loc[d, names].values, sd63.loc[d, names].values
        blocks.append((d, names, row, rn, m_, s_))
    return blocks

print("building row blocks...", flush=True)
all_dates = (px.index >= "2001-01-01") & (px.index <= "2026-07-07")
blocks = build_rows(all_dates)
print(f"{len(blocks)} usable days  ({time.time()-t0:.0f}s)", flush=True)

for k in (1.0, 1.5, 2.0):
    daily_ret = {}
    n_long = n_short = n_hit = n_pred = 0
    for oos_start in range(2012, 2027, 2):
        tr_lo, tr_hi = oos_start - 8, oos_start
        Xtr, ytr = [], []
        for d, names, row, rn, m_, s_ in blocks:
            if tr_lo <= d.year < tr_hi:
                lab = np.where(rn > m_ + k * s_, 1, np.where(rn < m_ - k * s_, -1, 0))
                keep = np.isfinite(rn) & np.isfinite(row).all(axis=1)
                Xtr.append(row[keep]); ytr.append(lab[keep])
        if not Xtr:
            continue
        Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
        if len(Xtr) > 500_000:
            sel = rng.choice(len(Xtr), 500_000, replace=False)
            Xtr, ytr = Xtr[sel], ytr[sel]
        mdl = HistGradientBoostingClassifier(max_depth=6, random_state=0)
        mdl.fit(Xtr, ytr)
        for d, names, row, rn, m_, s_ in blocks:
            if not (oos_start <= d.year < oos_start + 2):
                continue
            keep = np.isfinite(row).all(axis=1)
            if keep.sum() < 20:
                continue
            pred = np.zeros(len(names))
            pred[keep] = mdl.predict(row[keep])
            lab = np.where(rn > m_ + k * s_, 1, np.where(rn < m_ - k * s_, -1, 0))
            n_pred += int(keep.sum()); n_hit += int((pred[keep] == lab[keep]).sum())
            longs = pred == 1; shorts = pred == -1
            n_long += int(longs.sum()); n_short += int(shorts.sum())
            lr = np.nanmean(rn[longs]) if longs.any() else 0.0
            sr = -np.nanmean(rn[shorts]) if shorts.any() else 0.0
            gross_l = 1.0 if longs.any() else 0.0
            gross_s = 1.0 if shorts.any() else 0.0
            # daily full-turnover approximation: cost = 2 legs/day per active book
            cost = (gross_l + gross_s) * 2 * TC
            borrow = gross_s * BORROW / TD
            daily_ret[d] = 0.5 * lr * gross_l + 0.5 * sr * gross_s - cost * 0.5 - borrow * 0.5
        print(f"  k={k} refit {oos_start}: train {len(Xtr)} rows  "
              f"({time.time()-t0:.0f}s)", flush=True)
    s = pd.Series(daily_ret).sort_index().dropna()
    m = metrics_from_returns(s.values, TD)
    rec = s[s.index >= "2020-01-01"]
    mr = metrics_from_returns(rec.values, TD)
    print(f"M5 k={k}: acc {n_hit/max(n_pred,1):.1%} | avg longs/day "
          f"{n_long/max(len(s),1):.0f} shorts/day {n_short/max(len(s),1):.0f} | "
          f"Sharpe {m['sharpe']:+.2f} CAGR {m['cagr']:+.1%} MaxDD {m['max_dd']:.0%} | "
          f"2020+ {mr['sharpe']:+.2f}", flush=True)
print(f"\ntotal {time.time()-t0:.0f}s")
