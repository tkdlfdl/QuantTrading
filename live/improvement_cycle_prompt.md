Run ONE Improvement Cycle for this trading project, per SELF_IMPROVEMENT_PLAN.md §2 and SELF_IMPROVEMENT_STRATEGY_PLAN.md. Work autonomously; this is a scheduled unattended run.

Steps, in order:

1. MONITOR — compare live performance vs backtest expectation: read live/state/eod_book_daily.csv and equity.csv; check the champion (IvolVT) drawdown vs the rollback bound (revert-to-FixedEW rule fires at live DD worse than -11.1% or trailing 60d Sharpe < 0 AND below backtest's worst rolling 60d). Check data health: age of data/cache/merged_hourly_*.parquet, DuckDB sentiment staleness. If a rollback condition fires, STOP after writing the incident report — do not proceed to new ideas.

2. SELECT — open research/improvement_queue.md, take the TOP queue item. If the queue has fewer than 3 items, ALSO launch a literature sweep for the next bucket in the monthly rotation (finance → ML/CS → hard-science; check research/briefs/ for which ran last) and merge its output into the registry and queue.

3-5. BUILD, TEST, VERIFY the selected item:
- Honor the integrity rules (SELF_IMPROVEMENT_STRATEGY_PLAN §1.3): no look-ahead (signals shifted ≥1 bar, rolling-only statistics, next-bar execution), no leakage (chronological splits, no full-sample scalers).
- Follow the quant-developer lesson: PATCH validated engines for candidate tests, never re-implement; every experiment needs a sanity anchor that must reproduce the official baseline (±0.05 Sharpe) or the experiment is VOID.
- Run `python -m tools.verify <name>` on any recorded result; a FAIL must be investigated per the verifier agent's procedure before anything is reported.

6. JUDGE — promotion rule (SELF_IMPROVEMENT_PLAN §4): Sharpe > champion +0.10 or CAGR > champion +5pp, MaxDD ≤ 1.2× champion's, robustness ≥60% of a ±20% parameter perturbation grid. The current champion frontier is stated in BACKTEST_PERFORMANCE.md (2026-08-15: 2.580 / 43.5% / -8.9% — verify against the latest recorded portfolio_champion_* history.json in strategies/performance/ in case it moved).

7. RECORD — record_performance for any new series; if anything improved, record_improvement (attribution: which idea/paper, exact deltas — no anonymous improvements); update research/papers_read.md verdicts, move the queue item to Done in research/improvement_queue.md, and update BACKTEST_PERFORMANCE.md if headline metrics changed.

8. REFLECT — if a new failure mode or methodological lesson emerged, append it (dated, append-only) to the relevant .claude/agents/*.md file. Log cycle wall-clock per stage in a one-line entry appended to live/state/improvement_cycle_ledger.csv (date, item, verdict, minutes_per_stage).

HARD SAFETY RAILS — never violate:
- NEVER change DRY_RUN, --live, ALLOC_BOOKS, LIVE_BOOK, leverage settings, or broker code. If the cycle's result argues for such a change (e.g., promotion to live), write it as a PENDING HUMAN DECISION at the top of your final report and in the queue's Done entry.
- Never edit numbers to pass checks; a voided experiment is reported as voided.
- Keep all bookkeeping files consistent (registry, queue, ledger, improvements log).

End with a concise report: item tested, verdict with numbers, records written, any pending human decisions.
