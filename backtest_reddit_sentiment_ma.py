"""
Reddit Sentiment Moving-Average strategy — backtest.

Idea (as specified): read Reddit posts -> per-post VADER sentiment -> daily
per-stock sentiment -> MOVING AVERAGE of that sentiment -> trade on the smoothed
signal. Cross-sectional, long-only, equal-weight, non-overlapping holds.

Two directions are tested (we do not assume the sign a priori):
  - momentum : long the top_n stocks by HIGHEST smoothed sentiment
  - contrarian: long the top_n stocks by LOWEST (most negative) smoothed sentiment

No-lookahead: sentiment MA is shifted 1 day; signal at T uses data through T-1.
Costs (per CLAUDE.md): 0.1% per trade on entry; long-only so no borrow. Idle
windows (no eligible names) earn 0 (flat).

Data: DuckDB sentiment_daily (weighted_compound, mention_count) + ohlcv 1d close.
Records results via tools/record.py and prints metrics via tools/metrics.py.
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
TXN_COST = 0.001   # 0.1% per position on entry

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_panels():
    conn = get_conn()
    sent = conn.execute("""
        SELECT date, symbol, weighted_compound, mention_count
        FROM sentiment_daily WHERE date >= ?
    """, [START]).df()
    sent["date"] = pd.to_datetime(sent["date"])
    sent_score = sent.pivot_table(index="date", columns="symbol", values="weighted_compound")
    sent_ment  = sent.pivot_table(index="date", columns="symbol", values="mention_count")

    syms = list(sent_score.columns)
    px = conn.execute("""
        SELECT ts, symbol, close FROM ohlcv
        WHERE interval = '1d' AND ts >= ? AND symbol IN (SELECT DISTINCT symbol FROM sentiment_daily)
    """, [START]).df()
    px["ts"] = pd.to_datetime(px["ts"])
    close = px.pivot_table(index="ts", columns="symbol", values="close").sort_index()

    # Align to trading calendar from prices; sentiment reindexed onto it.
    close = close.loc[close.index >= pd.Timestamp(START)]
    syms = [s for s in syms if s in close.columns]
    close = close[syms]
    sent_score = sent_score.reindex(close.index)[syms]      # NaN where no posts
    sent_ment  = sent_ment.reindex(close.index)[syms].fillna(0.0)
    return close, sent_score, sent_ment


# ---------------------------------------------------------------------------
# Backtest one parameter set
# ---------------------------------------------------------------------------
def run_one(close, sent_score, sent_ment, ma_window, hold, top_n, direction,
            min_mentions=3):
    dates = close.index
    price_ret = close.pct_change().fillna(0.0)

    # Sentiment MA: missing sentiment treated as 0 (neutral, no buzz).
    sent_ma = sent_score.fillna(0.0).rolling(ma_window, min_periods=ma_window).mean()
    # Eligibility: enough recent discussion within the MA window.
    recent_ment = sent_ment.rolling(ma_window, min_periods=1).sum()

    # No lookahead: signal known before the trade day.
    signal = sent_ma.shift(1)
    elig   = recent_ment.shift(1) >= min_mentions

    warmup = ma_window + 1
    rows = []
    n = len(dates)
    for i in range(warmup, n - hold, hold):
        d = dates[i]
        s = signal.loc[d].where(elig.loc[d]).dropna()
        if s.empty:
            # flat window
            for j in range(i, min(i + hold, n)):
                rows.append((dates[j], 0.0))
            continue
        if direction == "momentum":
            picks = s.nlargest(top_n).index
        else:  # contrarian
            picks = s.nsmallest(top_n).index
        hold_end = min(i + hold, n)
        for j in range(i, hold_end):
            r = price_ret.loc[dates[j], picks].mean()
            if j == i:
                # full turnover each rebalance: 0.1% entry + 0.1% exit = 0.2% round-trip
                r -= 2 * TXN_COST
            rows.append((dates[j], float(r)))

    if not rows:
        return None
    ser = pd.Series(dict(rows)).sort_index()
    ser = ser[~ser.index.duplicated(keep="last")]
    return ser


def summarize(ser):
    m = metrics_from_returns(ser.values, TRADING_DAYS)
    return m


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
def main():
    close, sent_score, sent_ment = load_panels()
    print(f"Loaded: {close.shape[1]} symbols, {len(close)} trading days "
          f"({close.index.min().date()}..{close.index.max().date()})")

    ma_grid   = [5, 10, 20]
    hold_grid = [5, 10, 20]
    topn_grid = [10, 20]
    dirs      = ["momentum", "contrarian"]

    results = []
    best = {"momentum": None, "contrarian": None}
    for direction, ma, hold, top_n in itertools.product(dirs, ma_grid, hold_grid, topn_grid):
        ser = run_one(close, sent_score, sent_ment, ma, hold, top_n, direction)
        if ser is None or len(ser) < 100:
            continue
        m = summarize(ser)
        row = {"direction": direction, "ma": ma, "hold": hold, "top_n": top_n,
               "sharpe": m["sharpe"], "cagr": m["cagr"], "total_return": m["total_return"],
               "max_dd": m["max_dd"], "n_days": m["n"]}
        results.append(row)
        if best[direction] is None or m["sharpe"] > best[direction][1]["sharpe"]:
            best[direction] = (row, m, ser)

    grid = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    pd.set_option("display.width", 200)
    print("\n=== GRID (top 15 by Sharpe) ===")
    print(grid.head(15).to_string(index=False,
          formatters={"sharpe": "{:.3f}".format, "cagr": "{:.1%}".format,
                      "total_return": "{:.1%}".format, "max_dd": "{:.1%}".format}))

    print("\n=== Robustness ===")
    print(f"combos: {len(grid)} | positive Sharpe: {(grid.sharpe>0).sum()}/{len(grid)} "
          f"| Sharpe>0.5: {(grid.sharpe>0.5).sum()} | Sharpe>1.0: {(grid.sharpe>1.0).sum()}")

    for direction in dirs:
        if best[direction] is None:
            continue
        row, m, ser = best[direction]
        print(f"\n=== BEST {direction.upper()} === {row}")
        # yearly
        yr_rows = []
        for y, x in ser.groupby(ser.index.year):
            yr_rows.append({
                "year": int(y),
                "ret": f"{(1+x).prod()-1:.2%}",
                "sharpe": f"{sharpe_ratio(x.values, TRADING_DAYS):.2f}",
                "max_dd": f"{max_drawdown(x.values):.2%}",
                "days": len(x),
            })
        print(pd.DataFrame(yr_rows).to_string(index=False))

        name = f"book_e_reddit_sentiment_ma_{direction}"
        record_performance(
            name=name,
            dates=ser.index, returns=ser.values,
            params={k: row[k] for k in ("ma","hold","top_n","direction")},
            data_period=f"{ser.index.min().date()}..{ser.index.max().date()}",
            periods_per_year=TRADING_DAYS,
            extra={"note": "MA-of-Reddit-sentiment; weighted_compound; min_mentions=3; "
                           "coverage degrades post-2024 (see backtest notes)"},
        )
        print(f"recorded -> strategies/performance/{name}_daily.csv + _history.json")


if __name__ == "__main__":
    main()
