"""
Improvement Cycle 2 battery — queue ideas #6, #7, #10 (mlcs_sweep_2026-08).

Baseline to beat (capped-C champion): Sharpe 2.580 | CAGR 43.5% | MaxDD -8.9%.

#6  JM panic gate (Shu-Yu-Mulvey 2024, simplified statistical jump model):
    2-state model on SPY daily features (10d return, 20d vol), centroids fit
    on PRE-2019 data only (k-means, 20 restarts deterministic-seeded by grid),
    online greedy decoding with transition penalty lambda; NO leakage. Bear
    state -> halve A/F -> D, and tighten vol target 15% -> 12%.
    A/B against the current Daniel-Moskowitz gate.

#7  F turning-point state machine (Garg-Goulding-Harvey-Mazzoleni 2023):
    SPY slow momentum (252d sign) x fast (21d sign): Bull(+,+), Correction(+,-),
    Rebound(-,+), Bear(-,-). Momentum suffers at turning points -> Book F
    exposure 0.5x in Correction & Rebound. Lagged 1 day. Test standalone on F,
    then inside champion.

#10 EG online allocator (Mhammedi-Rakhlin lineage): exponentiated-gradient
    weights over books, eta = sqrt(8 ln N / T) (theory, not tuned), weekly
    update, 50% cap; same 15% vol-target overlay for a fair A/B vs ivol champ.

All signals lagged >= 1 day. Records improvements; verifier run separately.
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from tools.metrics import metrics_from_returns, sharpe_ratio, max_drawdown
from tools.record import record_performance, record_improvement

TD = 252
CHAMP = dict(sharpe=2.580, cagr=0.435, max_dd=-0.089)
START = "2019-01-02"

def load(n):
    df = pd.read_csv(f"strategies/performance/{n}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)

BOOKS = {"A": load("book_a_retest"), "C": load("book_c_overlap_cap"),
         "D": load("book_d_retest"), "F": load("book_f_retest")}
idx = sorted(set().union(*[s.index for s in BOOKS.values()]))
idx = pd.DatetimeIndex([d for d in idx if d >= pd.Timestamp(START)])
R = pd.DataFrame({k: BOOKS[k].reindex(idx).fillna(0.0) for k in BOOKS})

daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)
spy = daily["SPY"].dropna()
spy_ret = spy.pct_change()

def ivol_weights_series(R, win=60, rebal=21):
    """Reproduce champion ivol weighting: weight dict per rebalance block."""
    n = len(R); W = np.zeros((n, R.shape[1])); w = None
    for t in range(n):
        if w is None or t % rebal == 0:
            hist = R.iloc[max(0, t-win):t]
            vol = hist.std() * np.sqrt(TD)
            iv = np.array([1.0/v if np.isfinite(v) and v > 1e-9 else 0.0 for v in vol])
            w = iv/iv.sum() if iv.sum() else np.ones(R.shape[1])/R.shape[1]
        W[t] = w
    return W

def vol_target(ser, tgt=0.15, win=20):
    rv = ser.rolling(win).std().shift(1) * np.sqrt(TD)
    return ser * (tgt/rv).clip(upper=1.0).fillna(1.0)

def stats(tag, ser, base=CHAMP):
    m = metrics_from_returns(ser.dropna().values, TD)
    imp = ((m["sharpe"] > base["sharpe"] or m["cagr"] > base["cagr"])
           and m["max_dd"] >= base["max_dd"]*1.2)
    print(f"  {tag:<34} Sharpe {m['sharpe']:.3f} | CAGR {m['cagr']:.1%} | "
          f"MaxDD {m['max_dd']:.1%}{'  <-- beats champ frontier' if imp else ''}")
    return m, imp

cols = list(R.columns)
iA, iC, iD, iF = [cols.index(k) for k in ["A", "C", "D", "F"]]
W_base = ivol_weights_series(R)

# ═════════════ #6 JM panic gate ═════════════
print("=== #6 Statistical jump-model gate (vs DM panic gate) ===")
feat = pd.DataFrame({
    "r10": spy_ret.rolling(10).mean(),
    "v20": spy_ret.rolling(20).std(),
}).dropna()
train = feat[feat.index < "2019-01-01"]
mu_f, sd_f = train.mean(), train.std()          # scaler fit PRE-2019 only
Z_all = (feat - mu_f) / sd_f
Ztr = Z_all[Z_all.index < "2019-01-01"].values

# deterministic 2-means on training data (20 seeded restarts)
best_inertia, best_c = np.inf, None
for seed in range(20):
    rng = np.random.default_rng(seed)
    c = Ztr[rng.choice(len(Ztr), 2, replace=False)]
    for _ in range(50):
        d = ((Ztr[:, None, :] - c[None])**2).sum(-1)
        lab = d.argmin(1)
        newc = np.array([Ztr[lab == k].mean(0) if (lab == k).any() else c[k] for k in range(2)])
        if np.allclose(newc, c): break
        c = newc
    inertia = ((Ztr - c[lab])**2).sum()
    if inertia < best_inertia: best_inertia, best_c = inertia, c
bear_k = int(np.argmax(best_c[:, 1]))            # higher-vol centroid = bear
LAM = 4.0                                        # transition penalty (frozen a priori)

Z = Z_all.values
state = np.zeros(len(Z), dtype=int)
for t in range(1, len(Z)):
    dists = ((Z[t] - best_c)**2).sum(-1)
    dists += LAM * (np.arange(2) != state[t-1])
    state[t] = int(dists.argmin())
jm_bear = pd.Series(state == bear_k, index=Z_all.index).shift(1).reindex(idx).fillna(False)
print(f"  bear days in window: {int(jm_bear.sum())}/{len(idx)} ({jm_bear.mean():.1%})")

port = np.zeros(len(R))
for t in range(len(R)):
    w = W_base[t].copy()
    if jm_bear.iloc[t]:
        freed = 0.5*(w[iA] + w[iF]); w[iA] *= 0.5; w[iF] *= 0.5; w[iD] += freed
    port[t] = float(w @ R.iloc[t].values)
ser = pd.Series(port, index=idx)
rv = ser.rolling(20).std().shift(1) * np.sqrt(TD)
tgt = pd.Series(np.where(jm_bear.values, 0.12, 0.15), index=idx)
ser_jm = ser * (tgt/rv).clip(upper=1.0).fillna(1.0)
m_jm, imp_jm = stats("champ + JM gate", ser_jm)
if imp_jm:
    record_performance(name="portfolio_champ_jmgate", dates=ser_jm.index, returns=ser_jm.values,
        params={"gate": "jump-model 2-state", "lambda": LAM, "bear_tgt": 0.12},
        data_period=f"{idx.min().date()}..{idx.max().date()}", periods_per_year=TD,
        extra={"cycle": "Cycle 2 #6", "paper": "Shu-Yu-Mulvey 2024"})
    record_improvement("Statistical jump-model panic gate on champion",
        "Shu, Yu & Mulvey 2024 J. Asset Mgmt (registry #42)",
        CHAMP, m_jm, ["portfolio_champ_jmgate"],
        f"Replaces DM gate; bear days {int(jm_bear.sum())}; bear vol-target 12%.")
    print("  recorded portfolio_champ_jmgate + improvements_log entry")

# ═════════════ #7 F turning-point state machine ═════════════
print("\n=== #7 Book F turning-point state machine ===")
slow = np.sign(spy.pct_change(252)); fast = np.sign(spy.pct_change(21))
turning = ((slow * fast) < 0).shift(1).reindex(idx).fillna(False)   # Correction|Rebound
print(f"  turning-point days: {int(turning.sum())}/{len(idx)} ({turning.mean():.1%})")
f_sm = R["F"] * np.where(turning.values, 0.5, 1.0)
m_f0, _ = stats("Book F baseline", R["F"], dict(sharpe=1.548, cagr=0.757, max_dd=-0.398))
m_f1, _ = stats("Book F + state machine", f_sm, dict(sharpe=1.548, cagr=0.757, max_dd=-0.398))
# champion with F_sm
R2 = R.copy(); R2["F"] = f_sm
W2 = ivol_weights_series(R2)
port = np.array([float(W2[t] @ R2.iloc[t].values) for t in range(len(R2))])
ser_fsm = vol_target(pd.Series(port, index=idx))
m_fsm, imp_fsm = stats("champ with F-state-machine", ser_fsm)
if imp_fsm:
    record_performance(name="portfolio_champ_fsm", dates=ser_fsm.index, returns=ser_fsm.values,
        params={"overlay": "F 0.5x in Correction/Rebound (SPY 252d x 21d sign)"},
        data_period=f"{idx.min().date()}..{idx.max().date()}", periods_per_year=TD,
        extra={"cycle": "Cycle 2 #7", "paper": "Garg-Goulding-Harvey-Mazzoleni 2023 JFE"})
    record_improvement("Book F turning-point state machine (0.5x in Correction/Rebound)",
        "Garg-Goulding-Harvey-Mazzoleni 2023 JFE + Wood-Zohren CPD (registry #43, #44)",
        CHAMP, m_fsm, ["portfolio_champ_fsm"],
        f"F standalone: Sharpe {m_f0['sharpe']:.2f}->{m_f1['sharpe']:.2f}, MaxDD {m_f0['max_dd']:.1%}->{m_f1['max_dd']:.1%}.")
    print("  recorded portfolio_champ_fsm + improvements_log entry")

# ═════════════ #10 EG online allocator ═════════════
print("\n=== #10 EG online allocator (theory eta, no tuning) ===")
N = R.shape[1]; T = len(R)
eta = float(np.sqrt(8*np.log(N)/T))
w = np.ones(N)/N
port = np.zeros(T)
for t in range(T):
    port[t] = float(w @ R.iloc[t].values)
    if t % 5 == 4:                                  # weekly update
        r5 = R.iloc[max(0,t-4):t+1].sum().values    # week's book returns
        w = w * np.exp(eta * r5)
        w = np.minimum(w / w.sum(), 0.50)
        w = w / w.sum()
ser_eg = vol_target(pd.Series(port, index=idx))
m_eg, imp_eg = stats(f"EG allocator + VT (eta={eta:.3f})", ser_eg)
if imp_eg:
    record_performance(name="portfolio_eg_voltgt", dates=ser_eg.index, returns=ser_eg.values,
        params={"eta": eta, "cap": 0.5, "update": "weekly"},
        data_period=f"{idx.min().date()}..{idx.max().date()}", periods_per_year=TD,
        extra={"cycle": "Cycle 2 #10", "paper": "Mhammedi-Rakhlin 2022 COLT lineage"})
    record_improvement("EG online allocator + vol target (zero tuned hyperparameters)",
        "Mhammedi & Rakhlin 2022 COLT lineage (registry #51)",
        CHAMP, m_eg, ["portfolio_eg_voltgt"], f"eta={eta:.4f} from theory.")
    print("  recorded portfolio_eg_voltgt + improvements_log entry")

print(f"\nchampion baseline: Sharpe {CHAMP['sharpe']} | CAGR {CHAMP['cagr']:.1%} | MaxDD {CHAMP['max_dd']:.1%}")
