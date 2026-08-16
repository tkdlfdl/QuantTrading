# QuantTrading Project Context & Guidelines

Quant trading portfolio with multiple algorithmic strategies. This file holds
the project's non-negotiable rules; deeper reference lives in the files below.

## Critical Rules

1. **No approximations** — all metrics come from actual backtests; never estimate or project. If data is missing, say "data not available".
2. **Earliest available data** — backtest from the earliest data point (min 2 years); no cherry-picking; document data periods.
3. **Multiple strategies** — combine only strategies available at that time; don't backfill later-created ones; track when each was active.
4. **Real costs** — 0.1% per trade, 8%/yr short borrow, slippage in prices, no leverage unless stated.

## Subagents & pipeline

New strategies flow through four specialists, each defined in `.claude/agents/`:

```
researcher → data-engineer → quant-developer → backtest → verifier
  (idea)       (get data)      (write code)     (evaluate)   (sanity gate)
```

- **`researcher`** — sources & vets trading ideas from top finance journals; writes a Strategy Brief to `research/briefs/<slug>.md` with signal definition, data needs, and an evaluation plan.
- **`data-engineer`** — reads the brief's data requirements, fetches data, and stores it in DuckDB (`data/market_data.duckdb`) / parquet caches via the `data/ingestion.py` pipeline.
- **`quant-developer`** — codifies the brief's signal into a `Strategy` subclass (`strategies/`) on the stored data, plus a runner.
- **`backtest`** — evaluates: runs backtests, computes metrics, grid search, L/S decomposition, portfolio combination. Owns the methodology and cost model.
- **`verifier`** — sanity gate after every backtest: impossible-metric checks (negative return + positive Sharpe, MaxDD = 0), record consistency, integrity smells via `tools/verify.py`; investigates flags, approves or rejects recording; log at `strategies/performance/verification_log.csv`.

Delegate all backtest/metrics work to `backtest`; delegate idea→data→code work down the chain above.

## Where things live

- **`BACKTEST_PERFORMANCE.md`** — human-readable source of truth for results (Books A–F + portfolios); update on any parameter/metric change.
- **`strategies/performance/`** — machine-readable records the `backtest` agent writes for every run: `<name>_daily.csv` (daily perf) + `<name>_history.json` (historical perf).
- **`tools/`** — `metrics.py` (Sharpe + Max DD), `correlation.py` (cross-strategy correlation), `record.py` (write daily + historical performance).
- **`research/briefs/`** — researcher's strategy briefs (created on demand).
- **`SELF_IMPROVEMENT_PLAN.md`** — the improvement loop: weekly adaptation cycles consume `research/improvement_queue.md`, judged by hard promotion/rollback rules (§4), verifier-gated, lessons written back into the agent files. Live-capital changes always stay human-gated.
- **`SELF_IMPROVEMENT_STRATEGY_PLAN.md`** — the idea engine feeding that loop: monthly journal sweeps across finance + ML/CS + hard-science venues generate strategies targeting the Sharpe/MaxDD/CAGR frontier; survivors incubate as zero-weight live books ≥60 trading days before any ALLOC_BOOKS graduation (human-gated).
