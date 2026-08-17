"""
Queue #44 — M4 F-dip sleeve: capitulation entries restricted to the top-50
750h-momentum names ("buy the dip in the strongest names"), holds {40, 80,
120}h (pre-registered trio, all reported — no argmax adoption).

Pattern: DU-style eligibility mask via monkey-patched _bubble_matrix.
ANCHOR: unmasked matrix through the patched path must reproduce official D8.
Gates: standalone, corr vs F and D-blend, full-stack marginal @0.25, LW p.
"""
from __future__ import annotations
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from tools.metrics import metrics_from_returns
from tools.significance import sharpe_delta_test
from live import engine as E, config as C

TD, END = 252, "2026-07-08"
t0 = time.time()

def load(n):
    df = pd.read_csv(f"strategies/performance/{n}_daily.csv", parse_dates=["date"])
    return df.set_index("date")["ret"].astype(float)

champ = load("portfolio_champ_d14")
d8q, d14 = load("book_d8_quiet"), load("book_d14")
ju = d8q.index.union(d14.index)
dblend = 0.5 * d8q.reindex(ju).fillna(0) + 0.5 * d14.reindex(ju).fillna(0)
FULL = {"A": load("book_a_retest"), "C": load("book_c_overlap_cap"),
        "D": dblend, "F": load("book_f_retest"),
        "G": load("book_g_live_spec"), "X": load("book_x_live_spec"),
        "DU": load("book_du_live_spec")}


def build_port(books, alloc, shares):
    sb, ss = C.ALLOC_BOOKS, dict(C.ALLOC_SHARES)
    C.ALLOC_BOOKS = alloc
    C.ALLOC_SHARES = shares
    try:
        R = pd.DataFrame(books)
        R = R[(R.index >= "2019-01-02") & (R.index <= END)].fillna(0.0)
        return E.ivol_voltgt(R).dropna()
    finally:
        C.ALLOC_BOOKS, C.ALLOC_SHARES = sb, ss


full_stack = build_port(FULL, ["A", "C", "D", "F", "G", "X", "DU"],
                        {"D": 2.0, "G": .25, "X": .25, "DU": .25})

hc = pd.read_parquet("data/cache/merged_hourly_close.parquet")
hc.index = pd.to_datetime(hc.index)
ho = pd.read_parquet("data/cache/merged_hourly_open.parquet")
ho.index = pd.to_datetime(ho.index)
common = sorted((set(ho.columns) & set(hc.columns)) - {"SPY"})
hcc, hoo = hc[common].ffill(), ho[common].ffill()

import strategies.contrarian_bubble_hourly as CB
B = CB._bubble_matrix(hcc, 104)
_orig = CB._bubble_matrix

# ANCHOR: plain matrix through patched path == official D8
CB._bubble_matrix = lambda close, m_: B
anc, _, _ = CB.run_contrarian_bubble_hourly(
    hoo, hcc, ma_window_grid=[104], buy_threshold_grid=[0.8],
    hold_hours_grid=[8], top_n_grid=[20])
CB._bubble_matrix = _orig
anc = anc.dropna()
anc = anc[(anc.index >= "2019-01-02") & (anc.index <= END)]
off = load("book_d_retest")
off = off[(off.index >= "2019-01-02") & (off.index <= END)]
m_a, m_o = metrics_from_returns(anc.values, TD), metrics_from_returns(off.dropna().values, TD)
print(f"ANCHOR: patched {m_a['sharpe']:.3f} vs official-now baseline "
      f"(recorded {m_o['sharpe']:.3f}; patched-vs-patched is the anchor)")

# F-dip mask: eligible only if in top-50 by 750h momentum (lagged 1 bar)
mom = hcc.pct_change(750).shift(1)
rank = mom.rank(axis=1, ascending=False)
mask = (rank <= 50).values
Bm = np.where(mask, B, 0.0).astype(np.float32)

fB = load("book_f_retest")
for hold in (40, 80, 120):
    CB._bubble_matrix = lambda close, m_: Bm
    s_, _, _ = CB.run_contrarian_bubble_hourly(
        hoo, hcc, ma_window_grid=[104], buy_threshold_grid=[0.8],
        hold_hours_grid=[hold], top_n_grid=[20])
    CB._bubble_matrix = _orig
    s_ = s_.dropna()
    s_ = s_[(s_.index >= "2019-01-02") & (s_.index <= END)]
    ms = metrics_from_returns(s_.values, TD)
    jf = s_.index.intersection(fB.index)
    cf = float(np.corrcoef(s_.loc[jf], fB.loc[jf])[0, 1])
    jd = s_.index.intersection(dblend.index)
    cd = float(np.corrcoef(s_.loc[jd], dblend.loc[jd].fillna(0))[0, 1])
    ser = build_port({**FULL, "N": s_},
                     ["A", "C", "D", "F", "G", "X", "DU", "N"],
                     {"D": 2.0, "G": .25, "X": .25, "DU": .25, "N": .25})
    r = sharpe_delta_test(ser, full_stack)
    m = metrics_from_returns(ser.values, TD)
    sig = "  <-- ADDS ON FULL STACK" if r["significant_p10"] and r["delta"] > 0 else ""
    print(f"F-dip hold={hold}h: standalone {ms['sharpe']:.2f}/{ms['max_dd']:.0%} | "
          f"corr(F) {cf:+.2f} corr(D) {cd:+.2f} | stack {m['sharpe']:.3f} "
          f"(delta {r['delta']:+.3f}, p {r['p_one_sided']:.3f}){sig}  "
          f"[{time.time()-t0:.0f}s]")
print(f"total {time.time()-t0:.0f}s")
