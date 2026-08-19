"""
Cycle 30 (manual improvement cycle) — three ABSENT premium classes vs the
full v4 stack (registry #156-158). Gates: standalone, corr(champ), stress,
FULL-STACK marginal @0.25 shares, LW p<0.10. Costs 0.1%/side; 8%/yr borrow
where short (none here — all long-only implementations).
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
from data.db.client import get_conn

TD, TC, END = 252, 0.001, "2026-07-08"
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
        "DU": load("book_du_live_spec"), "FD": load("book_fdip_40h")}
worst = champ.nsmallest(int(len(champ) * 0.05)).index
SHARES = {"D": 2.0, "G": .25, "X": .25, "DU": .25, "FD": .25}
ORDER = ["A", "C", "D", "F", "G", "X", "DU", "FD"]


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


full_stack = build_port(FULL, ORDER, SHARES)
print(f"full stack (v4): {metrics_from_returns(full_stack.values, TD)['sharpe']:.3f}")


def judge(name, stream):
    stream = stream[stream.index <= END].dropna()
    if len(stream) < 300:
        print(f"  {name}: series too short ({len(stream)}) — no verdict")
        return
    ms = metrics_from_returns(stream.values, TD)
    s19 = stream.reindex(champ.index).fillna(0)
    cc = float(np.corrcoef(s19, champ)[0, 1])
    sd = float(stream.reindex(worst).fillna(0).mean())
    rec = stream[stream.index >= "2022-01-01"]
    mr = metrics_from_returns(rec.values, TD) if len(rec) > 100 else {"sharpe": np.nan}
    print(f"  {name}: {stream.index.min().date()}..: Sharpe {ms['sharpe']:.2f} "
          f"CAGR {ms['cagr']:.1%} MaxDD {ms['max_dd']:.0%} | 2022+ {mr['sharpe']:.2f} "
          f"| corr {cc:+.2f} | stress {sd:+.2%}/d")
    ser = build_port({**FULL, "N": stream}, ORDER + ["N"], {**SHARES, "N": .25})
    r = sharpe_delta_test(ser, full_stack)
    m = metrics_from_returns(ser.values, TD)
    sig = "  <-- ADDS ON FULL STACK" if r["significant_p10"] and r["delta"] > 0 else ""
    print(f"    FULL-STACK +N @0.25: {m['sharpe']:.3f} (delta {r['delta']:+.3f}, "
          f"p {r['p_one_sided']:.3f}){sig}")


daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)
stocks = [c for c in daily.columns if not c.startswith("^")
          and c not in {"UVXY", "SPY", "QQQ"}]
px = daily[stocks].ffill()
ret = px.pct_change()
jump = ret.abs().rolling(252).max()

# ══════════ (A) high-volume return premium ══════════
print(f"\n=== (A) high-volume return premium (GKM 2001) ===")
con = get_conn()
vol = con.execute("SELECT ts, symbol, volume FROM ohlcv WHERE interval='1d' "
                  "AND ts >= '2017-06-01'").df()
vol["ts"] = pd.to_datetime(vol["ts"])
V = vol.pivot_table(index="ts", columns="symbol", values="volume").sort_index()
V = V.reindex(columns=stocks)
vz = np.log((V / V.rolling(60).median()).clip(0.05, 20)).shift(1)
flag = pd.DataFrame(0.0, index=px.index, columns=stocks)
vz_al = vz.reindex(px.index)
for i, d in enumerate(px.index):
    if d < pd.Timestamp("2018-06-01"):
        continue
    row = vz_al.loc[d].dropna()
    row = row[jump.loc[d][row.index] <= 1.0]
    if len(row) < 100:
        continue
    cols = [flag.columns.get_loc(c) for c in row.nlargest(20).index]
    flag.iloc[i, cols] = 1.0
member = flag.rolling(21).max()               # in book for 21d after a shock
w = member.div(member.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).shift(1)
sA = ((ret.fillna(0) * w).sum(axis=1)
      - w.diff().abs().sum(axis=1).fillna(0) * TC)
sA = sA[sA.index >= "2018-07-01"].dropna()
judge("high-volume premium", sA)

# ══════════ (B) short-vol carry (SVXY, panic-gated) ══════════
print(f"\n=== (B) short-vol carry (SVXY 0.5x, panic-gated) ({time.time()-t0:.0f}s) ===")
import yfinance as yf
sv = yf.download("SVXY", start="2011-10-01", interval="1d", auto_adjust=True,
                 progress=False)["Close"]
if isinstance(sv, pd.DataFrame):
    sv = sv.iloc[:, 0]
sv.index = pd.to_datetime(sv.index).tz_localize(None)
sv_r = sv.pct_change()
spy = daily["SPY"].dropna()
spy_r = spy.pct_change()
vol63 = spy_r.rolling(63).std() * np.sqrt(TD)
panic = ((spy.pct_change(504) < 0)
         & (vol63 > vol63.rolling(756).quantile(0.8))).shift(1)
pos = (~panic.fillna(False)).astype(float).reindex(sv_r.index).ffill().fillna(1.0)
# post-deleverage era only (0.5x since 2018-02-27) — the tradable instrument today
sB = (sv_r * pos - pos.diff().abs().fillna(0) * TC).dropna()
sB = sB[sB.index >= "2018-03-01"]
judge("short-vol carry", sB)
# ungated control for attribution
sB0 = sv_r.dropna()
sB0 = sB0[sB0.index >= "2018-03-01"]
m0 = metrics_from_returns(sB0[sB0.index <= END].values, TD)
print(f"    ungated SVXY control: Sharpe {m0['sharpe']:.2f} MaxDD {m0['max_dd']:.0%}")

# ══════════ (C) deep-drawdown stabilization recovery ══════════
print(f"\n=== (C) drawdown-stabilization recovery ({time.time()-t0:.0f}s) ===")
dd52 = px / px.rolling(252).max() - 1
vol21 = ret.rolling(21).std()
vol_peak = vol21.rolling(63).max()
stab = (vol21 < 0.5 * vol_peak)
month_end = px.index.to_series().dt.month.diff().fillna(1) != 0
me_dates = px.index[month_end]
posC = pd.DataFrame(0.0, index=px.index, columns=stocks)
for d in me_dates:
    if d < pd.Timestamp("1999-06-01"):
        continue
    elig = (dd52.loc[d] < -0.40) & stab.loc[d] & (jump.loc[d] <= 1.0)
    names = dd52.loc[d][elig[elig.fillna(False)].index].dropna()
    if len(names) < 5:
        continue
    top = names.nsmallest(20).index          # deepest drawdowns
    loc = px.index.get_loc(d)
    until = min(loc + 63, len(px.index) - 1)
    cols = [posC.columns.get_loc(c) for c in top]
    posC.iloc[loc:until + 1, cols] = 1.0
w = posC.div(posC.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).shift(1)
sC = ((ret.fillna(0) * w).sum(axis=1)
      - w.diff().abs().sum(axis=1).fillna(0) * TC)
sC = sC[sC.index >= "1999-06-01"].dropna()
judge("drawdown-stabilization recovery", sC)
print(f"\ntotal {time.time()-t0:.0f}s")
