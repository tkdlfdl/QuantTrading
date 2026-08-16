"""
Improvement Cycle 5 battery — queue ideas #4, #5, #2.

All Book-F experiments run on the SAME validated bar-machinery that reproduced
F = 1.548 in the all-books retest; ANCHOR run must match or the battery voids
(quant-developer lesson, Cycle 4).

#4  Faber/TSMOM trend gate (Faber 2007; Moskowitz-Ooi-Pedersen 2012):
    per-book gate on A and F inside the champion: book weight -> 0 when the
    book's own trailing 210-trading-day (~10-month) return < 0, lagged 1 day.
    Freed weight goes to cash (de-risk only).

#5  George-Hwang 52-week-high ranking (2004 JF): swap F's ranking from
    750h total return to nearness-to-52wk-high = close / rolling_max(close,
    1750h). Everything else locked (hold 200h, top5, non-overlap).

#2  Lou-Polk-Skouras overnight re-timing (2019 JFE): momentum profits accrue
    overnight -> enter at the LAST bar of the signal day (instead of next bar),
    exit at the exit-day FIRST bar priced at its OPEN. Same signal, same costs.

Baselines: F 1.548/-39.8% (anchor); champion (capped C) 2.580/43.5%/-8.9%.
"""
from __future__ import annotations
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from tools.metrics import metrics_from_returns
from tools.record import record_performance, record_improvement

TD = 252
F_BASE = dict(sharpe=1.548, cagr=0.757, max_dd=-0.398)
CHAMP = dict(sharpe=2.580, cagr=0.435, max_dd=-0.089)
LB, HOLD, TOPN, TC = 750, 200, 5, 0.001

t0 = time.time()
ho = pd.read_parquet("data/cache/merged_hourly_open.parquet"); ho.index = pd.to_datetime(ho.index)
hc = pd.read_parquet("data/cache/merged_hourly_close.parquet"); hc.index = pd.to_datetime(hc.index)
common = sorted(set(ho.columns) & set(hc.columns)); ho, hc = ho[common], hc[common]
hc_np = hc.values.astype(np.float32); ho_np = ho.values.astype(np.float32)
bar_ts = hc.index
n, U = hc_np.shape
day_of_bar = bar_ts.normalize()
# last bar index of each day / first bar index of each day
df_days = pd.Series(np.arange(n), index=day_of_bar)
last_bar_of_day = df_days.groupby(level=0).max()
first_bar_of_day = df_days.groupby(level=0).min()

with np.errstate(divide="ignore", invalid="ignore"):
    mom_np = hc_np / np.vstack([np.full((LB, U), np.nan, dtype=np.float32), hc_np[:-LB]]) - 1
    o2c = np.where((ho_np > 0) & np.isfinite(ho_np) & np.isfinite(hc_np), hc_np/ho_np - 1, 0.0)
    c_prev = np.vstack([hc_np[:1], hc_np[:-1]])
    c2c = np.where((c_prev > 0) & np.isfinite(c_prev) & np.isfinite(hc_np), hc_np/c_prev - 1, 0.0)
    o2prev_c = np.where((c_prev > 0) & np.isfinite(c_prev) & np.isfinite(ho_np), ho_np/c_prev - 1, 0.0)

def run_f(score_np, entry_mode="next_bar", exit_mode="close"):
    """Validated F machinery with pluggable ranking + entry/exit timing."""
    H = np.zeros((n, U), dtype=np.float32)
    em = np.zeros(n, bool); xm = np.zeros(n, bool); xo = np.zeros(n, bool)
    i = LB
    while i + HOLD < n:
        row = score_np[i]; valid = np.where(np.isfinite(row))[0]
        if len(valid) >= TOPN:
            top_idx = valid[np.argsort(row[valid])[-TOPN:]]
            if entry_mode == "next_bar":
                eb = i + 1
            else:  # "last_bar_of_day": last bar of the day containing bar i
                eb = int(last_bar_of_day.loc[day_of_bar[i]])
                if eb <= i: eb = i + 1          # signal at day's last bar -> next bar
            xb = i + HOLD
            if exit_mode == "day_open":         # exit at first bar of exit day, at OPEN
                xb = int(first_bar_of_day.loc[day_of_bar[min(i + HOLD, n-1)]])
                if xb <= eb: xb = i + HOLD
            H[eb:xb+1, top_idx] = 1.0/TOPN
            em[eb] = True; xm[xb] = True
            if exit_mode == "day_open": xo[xb] = True
        i += HOLD
    bar_ret = np.where(em[:, None], o2c, np.where(xo[:, None], o2prev_c, c2c))
    port = (H * bar_ret).sum(axis=1)
    port[em] -= TC; port[xm] -= TC
    port[H.sum(axis=1) == 0] = 0.0
    s = pd.Series(port.astype(float), index=bar_ts)
    d = s.groupby(s.index.normalize()).apply(lambda g: float((1+g).prod()-1))
    d.index = pd.to_datetime(d.index)
    return d

def show(tag, d, base):
    m = metrics_from_returns(d.dropna().values, TD)
    print(f"  {tag:<40} Sharpe {m['sharpe']:.3f} | CAGR {m['cagr']:.1%} | MaxDD {m['max_dd']:.1%}")
    return m

# ── ANCHOR ──
print("=== anchor: validated F machinery must reproduce 1.548 ===")
d_anchor = run_f(mom_np)
m_anchor = show("F anchor (750h ranking, standard timing)", d_anchor, F_BASE)
if abs(m_anchor["sharpe"] - F_BASE["sharpe"]) > 0.05:
    print("ANCHOR FAILED — battery VOID"); sys.exit(1)
print(f"  anchor OK ({time.time()-t0:.0f}s)")

# ── #5 George-Hwang 52wk-high ranking ──
print("\n=== #5 52-week-high ranking swap ===")
W52 = 1750
roll_max = pd.DataFrame(hc_np, index=bar_ts).rolling(W52, min_periods=W52).max().values
with np.errstate(divide="ignore", invalid="ignore"):
    gh_np = np.where(roll_max > 0, hc_np / roll_max, np.nan).astype(np.float32)
d_gh = run_f(gh_np)
m_gh = show("F with 52wk-high ranking", d_gh, F_BASE)
if m_gh["sharpe"] > F_BASE["sharpe"] + 0.05 and m_gh["max_dd"] >= F_BASE["max_dd"]*1.2:
    record_performance(name="book_f_52wkhigh", dates=d_gh.dropna().index, returns=d_gh.dropna().values,
        params={"ranking": "close/rollmax1750h", "hold": HOLD, "top_n": TOPN},
        data_period=f"{d_gh.index.min().date()}..{d_gh.index.max().date()}", periods_per_year=TD,
        extra={"cycle": "Cycle 5 #5", "paper": "George & Hwang 2004 JF"})
    record_improvement("Book F ranking swapped to 52-week-high nearness",
        "George & Hwang 2004 JF (registry #31)", F_BASE, m_gh, ["book_f_52wkhigh"])
    print("  IMPROVED -> recorded book_f_52wkhigh")
else:
    print("  no gain")

# ── #2 LPS overnight re-timing ──
print("\n=== #2 LPS overnight re-timing (enter last bar, exit day-open) ===")
d_lps = run_f(mom_np, entry_mode="last_bar_of_day", exit_mode="day_open")
m_lps = show("F re-timed (LPS)", d_lps, F_BASE)
if m_lps["sharpe"] > F_BASE["sharpe"] + 0.05 and m_lps["max_dd"] >= F_BASE["max_dd"]*1.2:
    record_performance(name="book_f_lps_timing", dates=d_lps.dropna().index, returns=d_lps.dropna().values,
        params={"entry": "last bar of signal day", "exit": "exit-day open", "lb": LB, "hold": HOLD},
        data_period=f"{d_lps.index.min().date()}..{d_lps.index.max().date()}", periods_per_year=TD,
        extra={"cycle": "Cycle 5 #2", "paper": "Lou, Polk & Skouras 2019 JFE"})
    record_improvement("Book F overnight-aligned entry/exit timing",
        "Lou, Polk & Skouras 2019 JFE (registry #33)", F_BASE, m_lps, ["book_f_lps_timing"])
    print("  IMPROVED -> recorded book_f_lps_timing")
else:
    print("  no gain")

# ── #4 Faber trend gate on A and F inside champion ──
print("\n=== #4 Faber 10-month trend gate on A & F in champion ===")
def load(nm):
    df = pd.read_csv(f"strategies/performance/{nm}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)
BOOKS = {"A": load("book_a_retest"), "C": load("book_c_overlap_cap"),
         "D": load("book_d_retest"), "F": load("book_f_retest")}
idx = sorted(set().union(*[s.index for s in BOOKS.values()]))
idx = pd.DatetimeIndex([d for d in idx if d >= pd.Timestamp("2019-01-02")])
R = pd.DataFrame({k: BOOKS[k].reindex(idx).fillna(0.0) for k in BOOKS})
wealth = (1 + R).cumprod()
gate = {}
for b in ["A", "F"]:
    tr = wealth[b].pct_change(210)             # ~10 months of trading days
    gate[b] = (tr > 0).shift(1).fillna(True).values   # True = keep
n2 = len(R)
W = np.zeros((n2, 4)); w = None
cols = list(R.columns); iA, iF = cols.index("A"), cols.index("F")
for t in range(n2):
    if w is None or t % 21 == 0:
        hist = R.iloc[max(0, t-60):t]
        vol = hist.std() * np.sqrt(TD)
        iv = np.array([1.0/v if np.isfinite(v) and v > 1e-9 else 0.0 for v in vol])
        w = iv/iv.sum() if iv.sum() else np.ones(4)/4
    wt = w.copy()
    if not gate["A"][t]: wt[iA] = 0.0          # gated weight -> cash (de-risk)
    if not gate["F"][t]: wt[iF] = 0.0
    W[t] = wt
port = np.array([float(W[t] @ R.iloc[t].values) for t in range(n2)])
ser = pd.Series(port, index=idx)
rv = ser.rolling(20).std().shift(1) * np.sqrt(TD)
ser = ser * (0.15/rv).clip(upper=1.0).fillna(1.0)
m_fg = show("champion + Faber gate on A/F", ser, CHAMP)
if (m_fg["sharpe"] > CHAMP["sharpe"] or m_fg["cagr"] > CHAMP["cagr"]) and m_fg["max_dd"] >= CHAMP["max_dd"]*1.2:
    record_performance(name="portfolio_champ_fabergate", dates=ser.index, returns=ser.values,
        params={"gate": "own 210d return < 0 -> weight 0 (A,F)"},
        data_period=f"{idx.min().date()}..{idx.max().date()}", periods_per_year=TD,
        extra={"cycle": "Cycle 5 #4", "paper": "Faber 2007; Moskowitz-Ooi-Pedersen 2012"})
    record_improvement("Faber/TSMOM trend gate on A & F inside champion",
        "Faber 2007 JWM; Moskowitz-Ooi-Pedersen 2012 JFE (registry #24, #29)",
        CHAMP, m_fg, ["portfolio_champ_fabergate"])
    print("  IMPROVED -> recorded portfolio_champ_fabergate")
else:
    print("  no gain")
print(f"\ntotal {time.time()-t0:.0f}s")
