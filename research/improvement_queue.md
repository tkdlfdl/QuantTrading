# Improvement Queue

Consumed top-down by Improvement Cycles (see SELF_IMPROVEMENT_PLAN.md §2-3).
Each item: spec, source, expected effect, and what "done" means. When an item
completes, move it to the Done section with its verdict; when the queue drops
below 3 items, run a literature sweep to refill (check papers_read.md first —
re-testing rejected ideas is low priority and needs new evidence or a new
angle; unread papers and untested ideas come first).

## Queue

### 11. HAR-RV vol forecast for the vol-target overlay  [TOP PRIORITY]
- **Spec:** replace 20d trailing vol in the champion's overlay with expanding-
  window HAR on log-RV (daily/weekly/monthly components); variants HARQ,
  semivariance, leverage. Chronological fit only, no full-sample estimation.
- **Source:** Corsi 2009 + ABDL 2003 + BPQ 2016 + Patton-Sheppard (#60-64).
- **Expect (haircut):** Sharpe +0.05-0.15 — arithmetic transmission to frontier.
- **Done:** champion A/B vs 2.580/-8.9%; robustness grid; verified.

### 12. Cross-sectional dispersion throttle on A/F  [pre-test first]
- **Spec:** high CS dispersion -> weak momentum (Stivers-Sun): throttle A/F.
  STAGE 1: 1997-2018 pre-test on Book A only — kill cheaply if absent in-sample.
- **Source:** Stivers & Sun 2010 JFQA (#78).

### 13. Correlation-spike fragility gate  [MaxDD]
- **Spec:** mean pairwise correlation z-shift (60d window) tightens vol cap and
  throttles A/F ONLY (never D — registry #47 lesson: D's edge needs beta).
- **Source:** Preis et al. 2012 Sci.Rep. + Kritzman et al. 2011 (#68, #71).

### 14. CDaR-constrained book allocation  [allocator challenger]
- **Spec:** LP: maximize mean return s.t. Conditional Drawdown-at-Risk bound,
  monthly, expanding window; A/B vs ivol+voltgt. Modest prior (optimizer record).
- **Source:** Chekhlov-Uryasev-Zabarankin 2005 + Goldberg-Mahmoud 2017 (#74-75).

### 15. Vol-forecast committee (rides on #11)  [robustness]
- **Spec:** inverse-QLIKE-weighted combo {20d, HAR, EWMA, Yang-Zhang} for the
  overlay; free once #11 exists.
- **Source:** Bates & Granger 1969 (#77); Yang-Zhang 2000 (#65).

## Done

- **2026-08-15 | Cycle 6 | Idea #3 Medhat-Schmeling — tested-no-gain**
  0.753/-56.9% (28yr) vs unconditioned control 0.695/-72.2%; condition real but
  +0.06 only; corr 0.55 to F. Fails new-book bar. book_g candidate NOT recorded.
- **2026-08-15 | #8b-data + Gao test — done / tested-no-gain**
  SPY hourly ingested (Alpaca 2020-07+ + yfinance, 92.3% coverage, caches backed
  up). Gao intraday momentum: Sharpe -9.5 — killed by 0.2% round-trip costs.

- **2026-08-15 | Cycle 5 | Ideas #2/#4/#5 — tested-no-gain (anchor exact 1.548)**
  52wk-high ranking: Sharpe -0.01 (strong negative — wrong stock type for F's edge);
  LPS re-timing 1.424 < 1.548; Faber gate on A/F 2.566 ≈ 2.580 (rarely fires).

- **2026-08-15 | Cycle 4 | Idea #9 residual bubble score — tested-no-gain (informative)**
  Official engine, sanity anchor exact (alpha=0 = 2.740): alpha=0.5 -> 2.178,
  alpha=1.0 -> 1.362. Market-mode component IS part of D's edge — beta-driven
  oversold names rebound harder. v1 custom engine failed its anchor and was
  VOIDED (lesson: patch validated engines, never re-implement for candidate tests).

- **2026-08-15 | Cycle 3 | Idea #8a hour-of-day alignment — tested-no-gain**
  Locked 8h hold already optimal (2.740); 7h 2.566, 14h 2.637, 21h 2.507.
  14h/21h raise total return at ~2x DD (noted for a future high-return variant).
  8b (SPY intraday momentum) untestable — SPY hourly missing; data item queued.

- **2026-08-15 | Cycle 2 | Ideas #6/#7/#10 — tested-no-gain (all below frontier)**
  JM gate 2.498 (trigger-happy: 300 bear days vs DM's 11); F state machine: helps F
  standalone (-33.5% vs -39.8% DD) but wash in champion (2.576/-8.5 vs 2.580/-8.9);
  EG allocator 2.151/-13.1 (good zero-tuning insurance, loses to ivol). Registry #42-44, #51 updated.

- **2026-08-15 | Cycle 1 | Book C per-ticker overlap cap — IMPROVED (verified)**
  Standalone C: Sharpe 0.592→0.675, MaxDD -53.7%→-33.0% (HOOD 2x stacking removed;
  verifier PASS, extreme day +93%→+37%). Champion with capped C: Sharpe 2.514→2.580,
  MaxDD -8.9% unchanged, CAGR 45.3%→43.5%; robustness 9/9 perturbation grid.
  Records: book_c_overlap_cap_*, portfolio_champion_cappedC_*.
  Adoption into live C engine = human-gated (pending).
