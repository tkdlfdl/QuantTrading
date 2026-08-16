# Strategy performance records

Machine-readable performance recorded by the `backtest` agent (via
`tools/record.py`). Two files per strategy:

- `<name>_daily.csv` — daily performance: `date, ret, wealth`
- `<name>_history.json` — historical performance: params, data period, summary
  metrics (Sharpe / Max DD / CAGR / total return), and a yearly breakdown.

`tools/correlation.py` reads the `*_daily.csv` files here to compute the
cross-strategy correlation matrix. Human-readable write-ups live in the
top-level `BACKTEST_PERFORMANCE.md`.
