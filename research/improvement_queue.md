# Improvement Queue

Consumed top-down by Improvement Cycles (see SELF_IMPROVEMENT_PLAN.md §2-3).
Each item: spec, source, expected effect, and what "done" means. When an item
completes, move it to the Done section with its verdict; when the queue drops
below 3 items, run a literature sweep to refill (check papers_read.md first —
re-testing rejected ideas is low priority and needs new evidence or a new
angle; unread papers and untested ideas come first).

## Queue

## Done

- **2026-08-16 | Cycle 11 | #14 CDaR allocation — tested-no-gain**
  nu=5%: 2.476/-10.4%; nu=8%: 2.279/-13.2% vs champion 2.580/-8.9%. Fifth
  optimization-based allocator to lose to inverse-vol + vol-target (DeMiguel
  small-N lesson reconfirmed). QUEUE NOW EMPTY -> next sweep refills (rotation:
  finance bucket is next; Saturday cycle triggers it automatically).

- **2026-08-16 | Cycle 11 | #14 CDaR allocation — tested-no-gain**
  nu=5%: 2.476/-10.4%; nu=8%: 2.279/-13.2% vs champion 2.580/-8.9%. Fifth
  optimization-based allocator to lose to inverse-vol + vol-target (DeMiguel
  small-N lesson reconfirmed). QUEUE NOW EMPTY -> next sweep refills (rotation:
  finance bucket is next; Saturday cycle triggers it automatically).

- **2026-08-16 | Cycle 10 | #17-20 four new-book candidates — ALL BENCHED**
  GKM volume 0.875 standalone but corr 0.54 to D (portfolio 2.398); TOM 0.255
  (dead after costs); overnight-share 0.988/corr<=0.28 — best bench, portfolio
  2.511; VRP 0.347 < matched-beta 0.505. Portfolio-contribution rule enforced
  on all four; none improved 2.580/-8.9%.

- **2026-08-16 | Cycle 9 | #16 breadth capitulation timing — CANDIDATE PASSES**
  Original synthesis (D's score -> market breadth -> QQQ timing). H=20d:
  Sharpe 0.896, CAGR 12.5%, MaxDD -23.4%; corr D 0.38 / A 0.26 / F 0.30 / C 0.02;
  verifier PASS. Caveats: 2020-07+ window only (QQQ hourly), 52 signals, 2022 -19%.
  Recorded book_h_breadth_timing. UPDATE 2026-08-16: portfolio-contribution
  test (new standing rule) BENCHED it — champ+H 2.504 < 2.580; ivol over-weights
  its cash-heavy profile and 12.5% CAGR dilutes. No incubation. Bench note:
  conditional funding on signal days = possible future variant.

- **2026-08-16 | Cycle 8 | Ideas #12/#13 — tested-no-gain**
  Dispersion: killed by 1997-2018 pre-test (t=-0.73) without touching OOS.
  Corr-spike gate: 2.532 < 2.580 (panic gate already covers fragility).

- **2026-08-15 | Cycle 7 | Ideas #11/#15 vol-forecast upgrade — tested-no-gain**
  Anchor exact (trail20=2.580). EWMA94 2.597 (wash), HAR-on-daily-r^2 2.274
  (proxy too noisy — needs true intraday RV; book-level hourly RV = future item),
  committee 2.559 (dragged by HAR). Registry #60-65, #77 closed.

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
  Adoption into live C engine = human-gated (pending).## Done

- **2026-08-16 | Cycle 11 | #14 CDaR allocation — tested-no-gain**
  nu=5%: 2.476/-10.4%; nu=8%: 2.279/-13.2% vs champion 2.580/-8.9%. Fifth
  optimization-based allocator to lose to inverse-vol + vol-target (DeMiguel
  small-N lesson reconfirmed). QUEUE NOW EMPTY -> next sweep refills (rotation:
  finance bucket is next; Saturday cycle triggers it automatically).

- **2026-08-16 | Cycle 10 | #17-20 four new-book candidates — ALL BENCHED**
  GKM volume 0.875 standalone but corr 0.54 to D (portfolio 2.398); TOM 0.255
  (dead after costs); overnight-share 0.988/corr<=0.28 — best bench, portfolio
  2.511; VRP 0.347 < matched-beta 0.505. Portfolio-contribution rule enforced
  on all four; none improved 2.580/-8.9%.

- **2026-08-16 | Cycle 9 | #16 breadth capitulation timing — CANDIDATE PASSES**
  Original synthesis (D's score -> market breadth -> QQQ timing). H=20d:
  Sharpe 0.896, CAGR 12.5%, MaxDD -23.4%; corr D 0.38 / A 0.26 / F 0.30 / C 0.02;
  verifier PASS. Caveats: 2020-07+ window only (QQQ hourly), 52 signals, 2022 -19%.
  Recorded book_h_breadth_timing. UPDATE 2026-08-16: portfolio-contribution
  test (new standing rule) BENCHED it — champ+H 2.504 < 2.580; ivol over-weights
  its cash-heavy profile and 12.5% CAGR dilutes. No incubation. Bench note:
  conditional funding on signal days = possible future variant.

- **2026-08-16 | Cycle 8 | Ideas #12/#13 — tested-no-gain**
  Dispersion: killed by 1997-2018 pre-test (t=-0.73) without touching OOS.
  Corr-spike gate: 2.532 < 2.580 (panic gate already covers fragility).

- **2026-08-15 | Cycle 7 | Ideas #11/#15 vol-forecast upgrade — tested-no-gain**
  Anchor exact (trail20=2.580). EWMA94 2.597 (wash), HAR-on-daily-r^2 2.274
  (proxy too noisy — needs true intraday RV; book-level hourly RV = future item),
  committee 2.559 (dragged by HAR). Registry #60-65, #77 closed.

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
