# Portfolio Self-Improvement Plan

**Sources:**
- *Continual Harness: Online Adaptation for Self-Improving Foundation Agents*
  (Karten et al., arXiv 2605.09998) — agents alternate **execution ↔ adaptation**
  phases inside one continuous run; prompts/memory/skills refine automatically;
  a stronger judge relabels rollouts (process-reward co-learning).
- *Anthropic Institute: Recursive Self-Improvement* — automate the improvement
  loop itself; humans move to judgment/direction; automated review gates,
  experiment loops (run → measure → rewrite → iterate), parallel investigation,
  and **Amdahl's-law bottleneck shifting** (each automation moves the constraint).

**Thesis:** treat the portfolio as the *artifact* and the agent pipeline as the
*harness*. The trading system never resets (live paper trading is the single
continuous episode); improvement happens in scheduled adaptation phases that
consume a research queue, are judged by hard metrics + the verifier, and write
their lessons back into the agents' own instruction files.

---

## 1. Mapping: paper mechanism → our implementation

| Mechanism (source) | Our implementation | Status |
|---|---|---|
| Continuous episode, no resets | live/ paper engine, forward-only track record, Windows task daily | ✅ exists |
| Execution phase | Champion IvolVT trades daily; books A–F tracked | ✅ exists |
| Adaptation phase | **Improvement Cycle** (§3) — scheduled, consumes queue | 🔨 to build |
| Automated review gate (Anthropic: catches ~33% of bugs) | `verifier` agent + `tools/verify.py` + verification_log.csv | ✅ exists — already caught B/C/E inflation, HOOD 2x leverage |
| Process reward / judge relabeling | Hard promotion rule (§4): Sharpe/MaxDD bar + verifier PASS decides, not narrative | 🔨 formalize |
| Persistent memory, no re-work | `research/papers_read.md` (never reread), recorded series in `strategies/performance/` | ✅ exists |
| Skill/prompt self-refinement | Post-cycle: lessons appended to `.claude/agents/*.md` failure-mode lists (§5) | 🔨 to build |
| Parallel investigation | Parallel research agents (used for 60yr surveys) | ✅ exists |
| Amdahl bottleneck tracking | Cycle ledger logs where wall-clock went; slowest stage = next automation target | 🔨 to build |
| Human = judgment & direction | Human gates ONLY: live capital changes, new data sources, rule changes (§6) | policy |

## 2. The loop (one Improvement Cycle)

```
┌─ EXECUTION (continuous) ──────────────────────────────────────────┐
│ daily: run_daily.py settles books, IvolVT trades, equity.csv grows │
└──────────────────────────────┬────────────────────────────────────┘
                               │ weekly (or on-demand)
┌─ ADAPTATION ─────────────────▼────────────────────────────────────┐
│ 1 MONITOR  live vs expectation: rolling 60d Sharpe, DD vs backtest │
│            bounds; data-health checks (sentiment staleness, cache  │
│            age). Breach → rollback rule (§4) fires first.          │
│ 2 SELECT   next item from research/improvement_queue.md            │
│            (or new literature sweep if queue < 3 items)            │
│ 3 BUILD    researcher→data-engineer→quant-developer as needed      │
│ 4 TEST     backtest agent: locked-param, daily-convention, full    │
│            history, costs — same standards as the 2026-08 retest   │
│ 5 VERIFY   verifier agent: PASS/FLAG/FAIL → verification_log.csv   │
│ 6 JUDGE    promotion rule (§4) vs current champion — mechanical    │
│ 7 RECORD   record_performance + papers_read.md verdict + docs;     │
│            every improvement ALSO gets an attributed entry in      │
│            research/improvements_log.md via record_improvement():  │
│            which idea/paper improved performance and by how much   │
│ 8 REFLECT  new failure mode? → append to agent .md files;          │
│            log cycle wall-clock per stage → bottleneck ledger      │
└───────────────────────────────────────────────────────────────────┘
```

## 3. Cadence & trigger

- **Weekly cycle** (recommended: Saturday, after Friday settle): steps 1–8.
- **Daily micro-monitor** (piggybacks run_daily): step 1 only — drift &
  rollback check, one line appended to `live/state/monitor_log.csv`.
- Queue: `research/improvement_queue.md` — prioritized, each item has a spec
  and its source paper. Refilled by literature sweeps when < 3 items.

## 4. Hard rules (the "process reward" — no narrative judgment)

**Promotion (candidate replaces/augments champion):**
1. Full-history backtest (earliest data, daily convention, project costs);
2. Verifier verdict PASS (or FLAG-EXPLAINED with benign cause);
3. Sharpe > champion + 0.10 **or** CAGR > champion + 5pp, with MaxDD ≤ 1.2×
   champion's;
4. Robustness: improvement survives in ≥ 60% of a ±20% parameter perturbation
   grid (no single-point wins — the Book C lesson);
5. Recorded + registered before adoption; champion history preserved (never
   overwrite the old champion's record).

**Rollback (live degradation):**
- Trailing 60d live Sharpe < 0 **and** below backtest's worst rolling-60d
  Sharpe, **or** live drawdown > 1.25 × backtest MaxDD (i.e. -11.1% for
  IvolVT's -8.9%) → revert LIVE_BOOK to FixedEW, alert, open a post-mortem
  item at the top of the queue.
- Data staleness (the Book E lesson): any feed a book depends on older than
  its threshold → that book's weight → 0 automatically (already implemented
  for E; generalize).

## 5. Self-refinement of the agents (the recursive part)

After every cycle, lessons are written back into the harness itself:
- New failure mode found by verifier → appended to `verifier.md` known-modes
  list and `backtest.md` mistakes section (e.g. this session added: block
  attribution, overlapping-trade leverage, favorable-window selection).
- New data pitfall → `data-engineer.md`.
- Rejected idea + why → `papers_read.md` so no future cycle re-tests it.
- Memory index updated so the next session starts from current truth.
This is the paper's prompt/skill co-learning, applied to agent instruction
files instead of model weights. **Agent-file edits are diffs a human can review
— keep them append-only and dated.**

## 6. Safety rails (Anthropic's emphasis: verification > velocity)

- **Never self-promote to real money.** The loop owns paper trading; moving
  capital, changing DRY_RUN/--live, adding leverage, or new brokers = human.
- Verifier is not bypassable; a FAIL result cannot be recorded as final.
- No-leverage invariant: vol-target scale ≤ 1.0 enforced in config.
- Every cycle's actions are logged (verification_log, monitor_log, git
  commits) — the audit trail *is* the verification infrastructure.
- Overfitting police: promotion needs robustness (rule 4.4) + the McLean-
  Pontiff decay haircut on any literature-sourced expectation.
- The loop may propose rule changes to §4 but never apply them itself.

## 7. Bottleneck ledger (Amdahl)

Current cycle-time estimate by stage: literature (agent, ~10 min) → data
(mostly cached) → build (30–60 min) → backtest (5–30 min) → verify (<1 min)
→ record/docs (~5 min). **Current bottleneck: build (quant-developer time) —
first automation target: reusable overlay/allocator harness so candidates are
config, not code.** Re-measure each cycle; the constraint will move.

## 8. Rollout phases

- **Phase 0 (done):** pipeline agents, registry, verifier, recorded baselines,
  champion live.
- **Phase 1 (next):** `tools/monitor.py` (drift + rollback check, callable from
  run_daily) + `research/improvement_queue.md` seeded with the 5 pending items.
- **Phase 2:** weekly scheduled Improvement Cycle (Claude Code scheduled agent
  or Windows task invoking the cycle prompt); first cycles run the queue:
  LPS overnight timing → Book C overlap cap → Medhat-Schmeling → Faber gate.
- **Phase 3 (recursive):** cycle updates agent .md files automatically
  (append-only), literature sweeps self-refill the queue, bottleneck ledger
  drives which tool gets built next.

**Success metric for the loop itself:** champion's verified full-history Sharpe
and live-vs-backtest tracking error, per quarter — if cycles stop producing
verified improvements, the loop's own process becomes the queue item.
