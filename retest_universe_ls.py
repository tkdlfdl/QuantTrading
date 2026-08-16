"""Retest: Universe Hourly Bubble L/S (documented Strategy 2, non-book).
Documented best params: ma=100, z=100, buy=0.8, short=0.95, hold=8h, top_n=10.
Documented metrics (2024-06-20..2026-06-18): Sharpe 0.454, +19.85%, MaxDD -14.0%.
Retested here on FULL merged hourly history (earliest-data rule) + documented window.
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import pandas as pd
from strategies.universe_bubble_hourly import run_universe_bubble_hourly
from tools.metrics import sharpe_ratio, max_drawdown, metrics_from_returns
from tools.record import record_performance

TD = 252
ho = pd.read_parquet("data/cache/merged_hourly_open.parquet")
hc = pd.read_parquet("data/cache/merged_hourly_close.parquet")
ho.index = pd.to_datetime(ho.index); hc.index = pd.to_datetime(hc.index)
common = sorted(set(ho.columns) & set(hc.columns)); ho, hc = ho[common], hc[common]

for tag, lo in [("full_2019", None), ("doc_window", "2024-06-20")]:
    h_o, h_c = (ho, hc) if lo is None else (ho.loc[lo:], hc.loc[lo:])
    ser, params, _ = run_universe_bubble_hourly(
        h_o, h_c,
        ma_window_grid=[100], z_window_grid=[100],
        buy_threshold_grid=[0.8], short_threshold_grid=[0.95],
        hold_hours_grid=[8], top_n_grid=[10],
    )
    ser = ser.dropna()
    m = metrics_from_returns(ser.values, TD)
    print(f"\n[universe L/S | {tag}] Sharpe {m['sharpe']:.3f} | CAGR {m['cagr']:.1%} | "
          f"total {m['total_return']:.1%} | MaxDD {m['max_dd']:.1%} | days {m['n']}")
    for y, x in ser.groupby(ser.index.year):
        print(f"   {y}: ret {(1+x).prod()-1:+8.2%}  sharpe {sharpe_ratio(x.values,TD):6.2f}  "
              f"mdd {max_drawdown(x.values):8.2%}")
    if tag == "full_2019":
        record_performance(
            name="universe_bubble_ls_retest",
            dates=ser.index, returns=ser.values,
            params=dict(ma=100, z=100, buy=0.8, short=0.95, hold=8, top_n=10),
            data_period=f"{ser.index.min().date()}..{ser.index.max().date()}",
            periods_per_year=TD,
            extra={"retest": "2026-08-15 all-books retest",
                   "documented": "Sharpe 0.454 on 2024-06..2026-06 window only"})
        print("  recorded -> strategies/performance/universe_bubble_ls_retest_*")
