"""#44 follow-up: perturbation grid around the significant F-dip cell
(hold=40h, mask top-50, top_n=20). Cells: hold {30,40,50} x mask {40,50,60}.
Robustness bar: majority of cells positive-delta on full stack, spec cell not
an outlier. Anchor inherited from cycle26_fdip (same patched path)."""
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
mom = hcc.pct_change(750).shift(1)
rank = mom.rank(axis=1, ascending=False)

pos_cells = 0
cells = 0
for maskN in (40, 50, 60):
    Bm = np.where((rank <= maskN).values, B, 0.0).astype(np.float32)
    for hold in (30, 40, 50):
        CB._bubble_matrix = lambda close, m_: Bm
        s_, _, _ = CB.run_contrarian_bubble_hourly(
            hoo, hcc, ma_window_grid=[104], buy_threshold_grid=[0.8],
            hold_hours_grid=[hold], top_n_grid=[20])
        CB._bubble_matrix = _orig
        s_ = s_.dropna()
        s_ = s_[(s_.index >= "2019-01-02") & (s_.index <= END)]
        ser = build_port({**FULL, "N": s_},
                         ["A", "C", "D", "F", "G", "X", "DU", "N"],
                         {"D": 2.0, "G": .25, "X": .25, "DU": .25, "N": .25})
        r = sharpe_delta_test(ser, full_stack)
        cells += 1
        if r["delta"] > 0:
            pos_cells += 1
        mark = " <SPEC>" if (maskN, hold) == (50, 40) else ""
        print(f"mask top-{maskN} hold={hold}h: delta {r['delta']:+.3f} "
              f"p {r['p_one_sided']:.3f}{mark}  [{time.time()-t0:.0f}s]", flush=True)
print(f"\nplateau: {pos_cells}/{cells} cells positive")
