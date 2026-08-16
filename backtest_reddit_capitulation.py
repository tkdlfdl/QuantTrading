"""
Reddit Sentiment CAPITULATION strategy (contrarian) — backtest.

Thesis (literature-aligned: high sentiment -> reversal, washed-out sentiment ->
rebound; Baker-Wurgler, Da-Engelberg-Gao FEARS, Tetlock): BUY capitulation, FADE
euphoria. Sentiment is smoothed (moving average) then converted to a per-stock
z-score (how extreme is today's smoothed sentiment vs its own recent range).

4-ZONE SEGMENTATION of the smoothed-sentiment z-score:
    STRONG negative : z <= -z_strong      -> LONG, weight 2  (deep capitulation)
    negative        : -z_strong < z < 0   -> LONG, weight 1  (mild capitulation)
    positive        : 0 <= z < +z_strong  -> FLAT            (no edge)
    STRONG positive : z >= +z_strong      -> FADE            (flat long-only / SHORT in L/S)

Long-only is the "proper" Book-E-style version; a long/short variant additionally
shorts the STRONG-positive (euphoria) names as a hedge.

No-lookahead: z shifted 1 day. Costs (CLAUDE.md): 0.1% entry + 0.1% exit = 0.2%
round-trip per rebalance (full turnover); 8%/yr borrow on short weight. Records
via tools/record.py, metrics via tools/metrics.py.
"""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

from data.db.client import get_conn
from tools.metrics import sharpe_ratio, max_drawdown, metrics_from_returns
from tools.record import record_performance

START = "2019-01-01"
TRADING_DAYS = 252
TXN_COST = 0.001          # per side
BORROW_DAILY = 0.08 / TRADING_DAYS


def load_panels():
    conn = get_conn()
    sent = conn.execute("""
        SELECT date, symbol, weighted_compound, mention_count
        FROM sentiment_daily WHERE date >= ?
    """, [START]).df()
    sent["date"] = pd.to_datetime(sent["date"])
    score = sent.pivot_table(index="date", columns="symbol", values="weighted_compound")
    ment  = sent.pivot_table(index="date", columns="symbol", values="mention_count")

    px = conn.execute("""
        SELECT ts, symbol, close FROM ohlcv
        WHERE interval='1d' AND ts >= ? AND symbol IN (SELECT DISTINCT symbol FROM sentiment_daily)
    """, [START]).df()
    px["ts"] = pd.to_datetime(px["ts"])
    close = px.pivot_table(index="ts", columns="symbol", values="close").sort_index()
    close = close.loc[close.index >= pd.Timestamp(START)]

    syms = [s for s in score.columns if s in close.columns]
    close = close[syms]
    score = score.reindex(close.index)[syms]
    ment  = ment.reindex(close.index)[syms].fillna(0.0)
    return close, score, ment


def run_one(close, price_ret, signal_z, elig, z_strong, hold, top_n, mode, warmup):
    dates = close.index
    n = len(dates)
    rows = []
    for i in range(warmup, n - hold, hold):
        d = dates[i]
        z = signal_z.loc[d].where(elig.loc[d]).dropna()
        if z.empty:
            for j in range(i, min(i + hold, n)):
                rows.append((dates[j], 0.0))
            continue

        # 4-zone segmentation
        strong_neg = z[z <= -z_strong].nsmallest(top_n)          # deepest capitulation
        neg        = z[(z > -z_strong) & (z < 0)].nsmallest(top_n)
        strong_pos = z[z >= z_strong].nlargest(top_n)            # euphoria

        # Long book: capitulation names, strong_neg weighted 2x negative
        long_w = {}
        for s in strong_neg.index:
            long_w[s] = 2.0
        for s in neg.index:
            long_w[s] = long_w.get(s, 0.0) + 1.0

        short_w = {}
        if mode == "long_short":
            for s in strong_pos.index:
                short_w[s] = 1.0

        has_long, has_short = bool(long_w), bool(short_w)
        if not has_long and not has_short:
            for j in range(i, min(i + hold, n)):
                rows.append((dates[j], 0.0))
            continue

        # normalise each side to sum-of-abs-weights = 1
        if has_long:
            tot = sum(long_w.values())
            long_w = {k: v / tot for k, v in long_w.items()}
        if has_short:
            tot = sum(short_w.values())
            short_w = {k: v / tot for k, v in short_w.items()}
        side_scale = 0.5 if (has_long and has_short) else 1.0

        long_names = list(long_w); long_vec = np.array([long_w[k] for k in long_names])
        short_names = list(short_w); short_vec = np.array([short_w[k] for k in short_names])

        hold_end = min(i + hold, n)
        for j in range(i, hold_end):
            r = 0.0
            if has_long:
                pr = price_ret.loc[dates[j], long_names].values
                r += side_scale * float(np.nansum(long_vec * pr))
            if has_short:
                pr = price_ret.loc[dates[j], short_names].values
                r += side_scale * float(np.nansum(short_vec * -pr))
                r -= side_scale * BORROW_DAILY
            if j == i:
                r -= 2 * TXN_COST * (side_scale if has_long else 0) \
                     + (2 * TXN_COST * side_scale if has_short else 0)
            rows.append((dates[j], float(r)))

    if not rows:
        return None
    ser = pd.Series(dict(rows)).sort_index()
    return ser[~ser.index.duplicated(keep="last")]


def main():
    close, score, ment = load_panels()
    price_ret = close.pct_change().fillna(0.0)
    print(f"Loaded: {close.shape[1]} symbols, {len(close)} days "
          f"({close.index.min().date()}..{close.index.max().date()})")

    ma_grid, zwin_grid = [5, 10, 20], [30, 60]
    zstrong_grid, hold_grid, topn_grid = [1.0, 1.5], [10, 20], [15, 30]
    modes = ["long_only", "long_short"]

    results, best = [], None
    for ma, zwin in itertools.product(ma_grid, zwin_grid):
        sent_ma = score.fillna(0.0).rolling(ma, min_periods=ma).mean()
        roll_m = sent_ma.rolling(zwin, min_periods=zwin).mean()
        roll_s = sent_ma.rolling(zwin, min_periods=zwin).std()
        z = (sent_ma - roll_m) / roll_s.replace(0, np.nan)
        signal_z = z.shift(1)
        elig = (ment.rolling(ma, min_periods=1).sum().shift(1) >= 3)
        warmup = ma + zwin + 1
        for z_strong, hold, top_n, mode in itertools.product(
                zstrong_grid, hold_grid, topn_grid, modes):
            ser = run_one(close, price_ret, signal_z, elig, z_strong, hold, top_n, mode, warmup)
            if ser is None or len(ser) < 100:
                continue
            m = metrics_from_returns(ser.values, TRADING_DAYS)
            row = {"mode": mode, "ma": ma, "zwin": zwin, "z_strong": z_strong,
                   "hold": hold, "top_n": top_n, "sharpe": m["sharpe"], "cagr": m["cagr"],
                   "total_return": m["total_return"], "max_dd": m["max_dd"], "n_days": m["n"]}
            results.append(row)
            if best is None or m["sharpe"] > best[1]["sharpe"]:
                best = (row, m, ser)

    grid = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    pd.set_option("display.width", 220)
    print(f"\n=== GRID (top 15 by Sharpe, {len(grid)} combos) ===")
    print(grid.head(15).to_string(index=False,
          formatters={"sharpe": "{:.3f}".format, "cagr": "{:.1%}".format,
                      "total_return": "{:.1%}".format, "max_dd": "{:.1%}".format}))
    print(f"\nRobustness: positive Sharpe {(grid.sharpe>0).sum()}/{len(grid)} | "
          f">0.5 {(grid.sharpe>0.5).sum()} | >1.0 {(grid.sharpe>1.0).sum()}")
    print("Long-only vs L/S best Sharpe:",
          {mo: round(grid[grid['mode']==mo].sharpe.max(), 3) for mo in modes})

    row, m, ser = best
    print(f"\n=== BEST === {row}")
    yr_rows = []
    for y, x in ser.groupby(ser.index.year):
        yr_rows.append({"year": int(y), "ret": f"{(1+x).prod()-1:.2%}",
                        "sharpe": f"{sharpe_ratio(x.values, TRADING_DAYS):.2f}",
                        "max_dd": f"{max_drawdown(x.values):.2%}", "days": len(x)})
    print(pd.DataFrame(yr_rows).to_string(index=False))

    name = "book_e_reddit_capitulation"
    record_performance(name=name, dates=ser.index, returns=ser.values,
        params={k: row[k] for k in ("mode","ma","zwin","z_strong","hold","top_n")},
        data_period=f"{ser.index.min().date()}..{ser.index.max().date()}",
        periods_per_year=TRADING_DAYS,
        extra={"thesis": "contrarian capitulation; 4-zone sentiment z-score; "
                          "buy strong-negative, fade strong-positive",
               "note": "sentiment coverage degrades post-2024"})
    print(f"recorded -> strategies/performance/{name}_daily.csv + _history.json")


if __name__ == "__main__":
    main()
