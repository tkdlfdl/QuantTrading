"""
Cycle 27 — ML next-day classification ladder (see brief
research/briefs/ml_classification_2026-08.md). M1-M4 on SPY here; M5 (panel)
runs separately if anything here shows life OR regardless as pre-registered.

Label: +1 / -1 / 0 vs trailing 63d mu +/- k*sigma. Walk-forward yearly refit,
no hyperparameter search, costs 0.1%/side + 8%/yr short borrow.
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

daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)
spy = daily["SPY"].dropna()
r = spy.pct_change()

# ---- features (v2: technical-indicator block + market context) ----
from ml_features import tech_features, market_features
tf = tech_features(spy)
feat = pd.DataFrame(tf, index=spy.index)
stocks = [c for c in daily.columns if not c.startswith("^") and c not in {"UVXY"}]
px = daily[stocks].ffill()
mk = market_features(spy.index, daily, breadth_px=px)
feat = pd.concat([feat, mk.drop(columns=[c for c in ("spy_mom21", "spy_mom63",
                                                     "spy_vol21") if c in mk])],
                 axis=1)

SIMPLE = ["lag1", "lag3", "lag5", "vol21"]
RICH = list(feat.columns)

mu63 = r.rolling(63).mean()
sd63 = r.rolling(63).std()
r_next = r.shift(-1)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

MODELS = [
    ("M1 logistic simple", SIMPLE,
     lambda: LogisticRegression(C=1.0, max_iter=2000)),
    ("M2 logistic rich", RICH,
     lambda: LogisticRegression(C=1.0, max_iter=2000)),
    ("M3 random forest", RICH,
     lambda: RandomForestClassifier(n_estimators=400, max_depth=6,
                                    random_state=0, n_jobs=-1)),
    ("M4 hist-GB", RICH,
     lambda: HistGradientBoostingClassifier(max_depth=6, random_state=0)),
]

print("=== SPY next-day classification, walk-forward (first OOS year 2005) ===")
for k in (1.0, 1.5, 2.0):
    y = pd.Series(0, index=spy.index)
    y[r_next > mu63 + k * sd63] = 1
    y[r_next < mu63 - k * sd63] = -1
    data = feat.copy()
    data["y"] = y
    data = data.dropna()
    data = data[data.index <= "2026-07-07"]
    base = data["y"].value_counts(normalize=True)
    print(f"\n--- k={k}: base rates +1 {base.get(1,0):.1%} / 0 {base.get(0,0):.1%} "
          f"/ -1 {base.get(-1,0):.1%} ---")
    years = sorted(set(data.index.year))
    for name, cols, mk in MODELS:
        preds = pd.Series(np.nan, index=data.index)
        for yr in [yy for yy in years if yy >= 2005]:
            tr = data[data.index.year < yr]
            te = data[data.index.year == yr]
            if len(tr) < 500 or len(te) == 0:
                continue
            sc = StandardScaler().fit(tr[cols])
            mdl = mk()
            mdl.fit(sc.transform(tr[cols]), tr["y"])
            preds.loc[te.index] = mdl.predict(sc.transform(te[cols]))
        pr = preds.dropna()
        yy = data.loc[pr.index, "y"]
        acc = float((pr == yy).mean())
        maj = float((yy == 0).mean())
        pos = pr.shift(1).fillna(0)                     # trade next day
        rr = r.reindex(pos.index).fillna(0)
        strat = (pos * rr - pos.diff().abs().fillna(0) * TC
                 - (pos < 0).astype(float) * BORROW / TD)
        m = metrics_from_returns(strat.values, TD)
        rec = strat[strat.index >= "2015-01-01"]
        mr = metrics_from_returns(rec.values, TD) if len(rec) > 100 else {"sharpe": np.nan}
        n_tr = int(pos.diff().abs().gt(0).sum())
        print(f"  {name}: acc {acc:.1%} (majority {maj:.1%}) | "
              f"Sharpe {m['sharpe']:+.2f} CAGR {m['cagr']:+.1%} MaxDD {m['max_dd']:.0%} | "
              f"2015+ {mr['sharpe']:+.2f} | pos-changes {n_tr} | "
              f"long {float((pr==1).mean()):.1%} short {float((pr==-1).mean()):.1%}  "
              f"[{time.time()-t0:.0f}s]")
print(f"\ntotal {time.time()-t0:.0f}s")
