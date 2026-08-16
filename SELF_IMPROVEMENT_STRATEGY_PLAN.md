# Strategy Self-Improvement Plan (Idea Engine → Live)

Companion to `SELF_IMPROVEMENT_PLAN.md` (which defines the loop mechanics,
promotion/rollback rules, and safety rails). **This document defines where new
strategy ideas come from and how they graduate to live trading.**

**Standing objective — improve the portfolio frontier on all three axes:**

| Metric | Current champion (IvolVT, verified) | Direction |
|--------|-------------------------------------|-----------|
| Sharpe | 2.514 | maximize |
| MaxDD | -8.9% | keep ≤ ~-11% (1.2× bar) |
| CAGR | 45.3% | maximize subject to DD bar |

An idea "improves the portfolio" if, added as a book or overlay, the combined
portfolio beats this frontier per the promotion rule (SELF_IMPROVEMENT_PLAN §4).

---

## Phase 1 — Idea generation (multi-discipline journal engine)

### 1.1 Source universe (researchers read ALL of these, rotating monthly)

**Finance (existing tier table in `.claude/agents/researcher.md`):**
JF, JFE, RFS (S-tier); JFQA, Review of Finance, RAPS (A+); Journal of Financial
Markets, J. Financial Econometrics, JPM, J. Futures Markets, Mathematical
Finance, Finance & Stochastics (A); Quantitative Finance; FAJ (practitioner).

**Computer science / Machine learning:**
| Venue | What to mine for |
|-------|-----------------|
| NeurIPS, ICML, ICLR | sequence models for returns, representation learning, OOD robustness, conformal prediction |
| KDD, AAAI, WWW | time-series mining, anomaly detection, graph learning on markets |
| JMLR, TMLR | validated methods with theory (regularization, online learning) |
| arXiv q-fin.* (PM, TR, ST), cs.LG, stat.ML | fastest-moving; treat as leads needing corroboration |
| ACM/IEEE trading-systems venues | execution, microstructure, limit-order-book models |

**Other hard sciences:**
| Venue | What to mine for |
|-------|-----------------|
| Nature, Science, PNAS | econophysics, collective behavior, early-warning signals for regime shifts |
| Physical Review E / Physica A | correlation-matrix cleaning (RMT), network/contagion structure, volatility cascades |
| Econometrica, J. Econometrics | forecast evaluation, regime-switching, high-dim covariance estimation |
| Management Science, Operations Research | portfolio optimization under constraints, robust optimization, inventory→execution analogies |
| Complexity/network science journals | lead-lag networks, centrality-based stock selection |

### 1.2 Idea generation — NO LIMITATIONS on origin

Anything that could improve return, MaxDD, or Sharpe is in scope. All four
idea classes are equally legitimate:

1. **Direct from a paper** — implement a published strategy on our data
   (e.g., 52-week-high momentum).
2. **Improve an existing book** — modify signals, timing, sizing, or risk of
   Books A-F (e.g., the C overlap cap, LPS overnight re-timing of F).
3. **Synthesis of multiple papers** — combine mechanisms from several papers
   (possibly across disciplines) into a strategy no single paper describes
   (e.g., Nagel liquidity-provision + early-warning vol signal + conformal
   position sizing = a new reversal book).
4. **Original ideas** — hypotheses from our own data, live observations,
   verifier findings, or first principles, with no paper behind them. A
   pattern noticed in our own recorded series is a valid starting point.

Novelty is not restricted — but every idea, regardless of origin, must state
an **economic rationale** (why should this edge exist and persist?) and passes
the same gates as everything else. Ideas synthesized or original get NO
McLean-Pontiff prior to lean on, so their robustness bar is effectively
higher: no published out-of-sample evidence means our own perturbation tests
and incubation are the only evidence there is.

### 1.3 HARD RULES — backtest integrity (non-negotiable, verifier-enforced)

**Rule 1: NO LOOK-AHEAD.** A signal used at bar T may only use information
available strictly before T:
- Signals/scores shifted ≥1 bar before use (the house convention);
- Rolling statistics only — never full-sample means, stds, quantiles, or
  normalizations computed over data that includes the future;
- Parameters chosen on a period may not be "best" selected using that same
  period's outcome and then reported as out-of-sample;
- Execution realism: trade at the NEXT bar's price after the signal bar, never
  the signal bar's close.

**Rule 2: NO DATA LEAKAGE.** No information may cross from the future or from
outside the tradable information set:
- Point-in-time universe — no survivorship (today's index membership must not
  select the historical universe where avoidable; document the bias where the
  cached data can't avoid it);
- For ML-based ideas: strict chronological train/validation/test splits, no
  shuffling across time, no feature computed with future data (including
  scalers/PCA fitted on the full sample), no target leakage in features,
  embargo gaps between train and test at least one holding period wide;
- No peeking at the test period during development — grid searches report the
  FULL grid (the Book B/C lesson), never only the winning cell;
- Data revisions: use the data as it would have been known at the time where
  the source allows.

Violations of either rule invalidate a backtest regardless of its metrics.
The verifier checks for the smells (signal-bar execution, full-sample
normalization, suspiciously smooth equity curves, zero-loss streaks) and a
FAIL on integrity cannot be explained away.

### 1.4 Process & hygiene (every sweep)

1. **Registry first:** check `research/papers_read.md`. Re-reading listed
   papers IS allowed, but **unread new papers always take priority** — re-reads
   are lower priority and need a reason (new synthesis angle, changed data,
   contradicting evidence since the last verdict). Don't re-read out of habit;
   the registry verdict usually suffices. Original/synthesized ideas get
   registry entries too (family: "original").
2. Parallel researcher agents per discipline bucket, monthly rotation
   (fin → ML → hard-science → ...), each returning a structured brief
   (`research/briefs/`) with: origin class (1-4 above), citations if any,
   mechanism + economic rationale, OUR-DATA implementability tag, and expected
   effect on Sharpe / MaxDD / CAGR.
3. **Published-idea expectations get the McLean-Pontiff haircut** (~50%);
   synthesized/original ideas get no expectation credit at all until tested.
4. Output: new items appended to `research/improvement_queue.md`, ranked by
   (expected frontier improvement × implementability ÷ effort).
5. Quota discipline: a sweep that yields zero implementable ideas still writes
   its negative results to the registry — coverage is progress.

### 1.5 Backtest gate (unchanged, non-negotiable)

Full-history, daily-convention, real costs, robustness perturbation, verifier
PASS, and the §1.3 integrity rules — per SELF_IMPROVEMENT_PLAN §4. Only
verified beats-the-frontier candidates proceed to Phase 2.

---

## Phase 2 — Live incubation (paper money first, always)

New strategies do NOT go straight into the champion. They earn it forward:

### 2.1 Incubation book
- Backtest-passed strategy gets a new book letter (G, H, ...) in `live/`:
  wired into `run_daily` settle with locked params, **weight 0 in the champion
  allocator** (like B/E today — tracked, not funded).
- Runs forward-only for a minimum of **60 trading days** (~3 months).

### 2.2 Graduation criteria (incubation → ALLOC_BOOKS)
All four required:
1. **Tracking:** live daily returns consistent with backtest expectation —
   rolling 60d Sharpe above the backtest's 20th-percentile rolling-60d Sharpe
   (i.e., live looks like a draw from the backtest distribution, not a new
   regime);
2. **No structural breaks:** no verifier FAIL on the live series; no data-feed
   staleness incidents;
3. **Portfolio math still improves:** champion re-estimated WITH the incubated
   book (its live+backtest series) still beats the current frontier per the
   promotion rule;
4. **Human sign-off** — adding a book to ALLOC_BOOKS is a capital decision
   (SELF_IMPROVEMENT_PLAN §6: the loop proposes, the human disposes).

### 2.3 Failure paths
- Tracking breach or verifier FAIL during incubation → demote to monitoring
  (like B/E), post-mortem written to the registry (verdict updated), lesson
  appended to agent files.
- Passed incubation but champion math doesn't improve → keep as bench
  strategy; re-evaluate when book correlations shift.

### 2.4 Champion evolution
When a graduated book enters ALLOC_BOOKS, the allocator itself re-runs the
construction shootout (ivol/vol-target remains default; challengers from the
construction literature re-tested at the new N) — the allocator is subject to
the same promotion rule as strategies.

---

## Cadence summary

| When | What |
|------|------|
| Daily | Execution + monitor (drift/rollback, data health) |
| Weekly | Improvement Cycle: consume queue, test, verify, record |
| Monthly | Discipline-rotating journal sweep (Phase 1) refills queue |
| Quarterly | Frontier review: is the champion improving? Loop post-mortem if not; incubation graduations reviewed |

**First actions under this plan:**
1. Next monthly sweep = **ML/CS bucket** (finance was swept 2026-08-15; 60yr).
2. Queue items #1-5 already pending — Phase 1 output feeds behind them.
3. First incubation candidate will be whichever queue item produces a NEW book
   (Medhat-Schmeling short-term momentum is the likeliest).
