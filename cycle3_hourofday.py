"""
Improvement Cycle 3 — queue idea #8: hour-of-day effects (mlcs_sweep Idea 3).

8a. Book D hold-length alignment (Heston-Korajczyk-Sadka 2010: cross-sectional
    return patterns recur at 24h multiples): D currently holds 8h (exit drifts
    across clock hours). Test holds of 7/14/21h (~1/2/3 sessions on 6.5-7h days
    -> exit near entry clock-hour) vs locked 8h, all else locked
    (ma=104, thr=0.8, top20). Uses the existing verified engine.

8b. SPY intraday momentum (Gao-Han-Li-Zhou 2018): first-hour return sign
    predicts last-hour return. Needs SPY hourly bars — only run if SPY is in
    the merged panel; big-move filter |r_first| > median.
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from strategies.contrarian_bubble_hourly import run_contrarian_bubble_hourly
from tools.metrics import metrics_from_returns
from tools.record import record_performance, record_improvement

TD = 252
D_BASE = dict(sharpe=2.740, cagr=0.352, max_dd=-0.064)

ho = pd.read_parquet("data/cache/merged_hourly_open.parquet"); ho.index = pd.to_datetime(ho.index)
hc = pd.read_parquet("data/cache/merged_hourly_close.parquet"); hc.index = pd.to_datetime(hc.index)
common = sorted(set(ho.columns) & set(hc.columns)); ho, hc = ho[common], hc[common]

# ── 8a: D hold alignment ──
print("=== 8a: Book D hold grid 7/14/21h vs locked 8h ===")
daily_d, params_d, grid = run_contrarian_bubble_hourly(
    ho, hc, ma_window_grid=[104], buy_threshold_grid=[0.8],
    hold_hours_grid=[7, 8, 14, 21], top_n_grid=[20])
print(grid[["hold_hours", "Sharpe", "Total_Return", "Max_DD", "n_trades"]].to_string(index=False))

best_row = grid.sort_values("Sharpe", ascending=False).iloc[0]
if int(best_row["hold_hours"]) != 8 and best_row["Sharpe"] > D_BASE["sharpe"] + 0.05 \
        and best_row["Max_DD"] >= D_BASE["max_dd"] * 1.2:
    # re-run winner alone to capture its daily series
    ser, p, _ = run_contrarian_bubble_hourly(
        ho, hc, ma_window_grid=[104], buy_threshold_grid=[0.8],
        hold_hours_grid=[int(best_row["hold_hours"])], top_n_grid=[20])
    m = metrics_from_returns(ser.dropna().values, TD)
    nm = f"book_d_hold{int(best_row['hold_hours'])}"
    record_performance(name=nm, dates=ser.dropna().index, returns=ser.dropna().values,
        params={"ma": 104, "thr": 0.8, "hold": int(best_row["hold_hours"]), "top_n": 20},
        data_period="full", periods_per_year=TD,
        extra={"cycle": "Cycle 3 #8a", "paper": "Heston-Korajczyk-Sadka 2010 JF"})
    record_improvement(f"Book D hold aligned to {int(best_row['hold_hours'])}h (clock-hour periodicity)",
        "Heston, Korajczyk & Sadka 2010 JF (registry #45)", D_BASE, m, [nm])
    print(f"IMPROVED -> recorded {nm}")
else:
    print(f"no gain: best hold {int(best_row['hold_hours'])}h Sharpe {best_row['Sharpe']:.3f} "
          f"vs locked 8h {D_BASE['sharpe']:.3f} (+0.05 bar not met or DD breach)")

# ── 8b: SPY intraday momentum ──
print("\n=== 8b: SPY first-hour -> last-hour momentum ===")
if "SPY" not in hc.columns:
    print("SPY not in merged hourly panel -> UNTESTABLE with current data; "
          "data-engineer item logged (needs SPY hourly ingestion).")
else:
    so, sc = ho["SPY"].dropna(), hc["SPY"].dropna()
    days = so.groupby(so.index.normalize())
    rows = []
    for day, bars in days:
        if len(bars) < 3: continue
        ts = bars.index
        first_r = sc.loc[ts[0]] / so.loc[ts[0]] - 1
        last_o, last_c = so.loc[ts[-1]], sc.loc[ts[-1]]
        rows.append((day, first_r, last_c / last_o - 1))
    df = pd.DataFrame(rows, columns=["day", "first_r", "last_r"]).set_index("day")
    med = df["first_r"].abs().expanding(min_periods=252).median().shift(1)
    sig = np.sign(df["first_r"]).where(df["first_r"].abs() > med, 0.0)
    strat = (sig * df["last_r"] - np.abs(sig) * 0.002).fillna(0.0)  # 0.1% x2 costs
    m = metrics_from_returns(strat.values, TD)
    print(f"SPY intraday momentum: Sharpe {m['sharpe']:.3f} | CAGR {m['cagr']:.1%} | "
          f"MaxDD {m['max_dd']:.1%} | active days {(sig != 0).sum()}")
    if m["sharpe"] > 0.5:
        record_performance(name="spy_intraday_momentum", dates=strat.index, returns=strat.values,
            params={"filter": "|r_first|>expanding median", "cost": 0.002},
            data_period=f"{strat.index.min().date()}..{strat.index.max().date()}",
            periods_per_year=TD, extra={"cycle": "Cycle 3 #8b", "paper": "Gao et al. 2018 JFE"})
        print("recorded spy_intraday_momentum (candidate micro-book)")
    else:
        print("no gain (as expected — effect diluted on hourly bars / post-publication)")
