# Improvement Queue

Consumed top-down by Improvement Cycles (see SELF_IMPROVEMENT_PLAN.md §2-3).
Each item: spec, source, expected effect, and what "done" means. When an item
completes, move it to the Done section with its verdict; when the queue drops
below 3 items, run a literature sweep to refill (check papers_read.md first —
re-testing rejected ideas is low priority and needs new evidence or a new
angle; unread papers and untested ideas come first).

## Queue

### 21. D sector-relative ranking tie-break  [Hameed-Mian 2015, #91]
- **Spec:** keep raw score < -0.8 gate (beta bounce is part of the edge, #47);
  rank eligible names by WITHIN-SECTOR score z instead of raw score. GICS-lite
  sectors from static large-cap map.
### 22. D end-of-day pressure entry tilt  [Baltussen et al. 2024, #92]
- **Spec:** entries triggered in the final 2 bars deferred to next open vs
  taken immediately — A/B on D8 sleeve.
### 23. D path-composition diagnostic  [Barardehi 2026, #93 — analysis only]
- **Spec:** decompose each D entry's trailing move into overnight/intraday/jump
  components; measure rebound by cohort. Zero mining risk; informs #24.
### 24. Sleeve assignment by reversal speed  [Dai et al. 2024 FAJ, #94]
- **Spec:** route fast/rich oversold names to D8, slow/strong to D14 (replaces
  identical selection in both sleeves). Test only if #23 shows cohort spread.

### 25. D phase-tranche + threshold barbell  [TOP — extends the proven ensemble win]
- **Spec:** (i) tranche each D sleeve across entry phases (start offsets 0..hold-1,
  1/hold weight each — the JT construction); (ii) equal-weight a threshold
  barbell (0.7/0.9) beside 0.8. Turnover nets across variants.
- **Source:** Hoffstein 2019; JT 1993; DeMiguel; cs_hardsci_sweep2 idea 1.
- **Gate:** significance p<0.10 vs 2.781 champion.

### 26. D asymmetric entry/exit bands  [Blitz FAJ 2023, #97]
- **Spec:** enter < -0.8, but exit only when score > -0.4 (lazy exit), DD-capped
  at 21h max hold. Cuts re-entry churn.

### 27. Book C meta-labeling filter  [Joubert 2022, #98]
- **Spec:** PRE-REGISTERED single ridge-logistic on (sigma-day features) sizing
  C's trades 0/0.5/1 — the only sanctioned adaptivity; chronological CV.

## Done

- **2026-08-17 | Book X deep grid (lb x hold) — spec confirmed**
  20/20 positive delta, 15/20 p<0.10; lookback axis reproduces the canonical
  momentum term structure (3m noise -> 12m peak -> 18m rolloff); hold cadence
  flat 21-63d. Incubating spec (252d/monthly) = plateau center; no change.
  Cumulative robustness 32/32 cells across both grids.

- **2026-08-17 | Cross-asset ETF momentum (Book X) — INCUBATING**
  New data (15 ETFs cached 2002+) unlocked first non-single-stock book.
  12/12 perturbation cells positive; +0.115 p=0.034 @0.25 shares; positive in
  2008/2020/2022. Ungated spec (gate duplicates champion vol-target). v1 run
  VOIDED (ffill position bug — caught by impossible -93% MaxDD). Zero-weight
  incubation with Book G; both review after 60 trading days.

- **2026-08-16 | Appraisal-sizing round | Overnight-share (Book G candidate) — PASSES**
  User-driven relaxation (less-correlated, not orthogonal) -> Treynor-Black bar:
  Sharpe > corr x 2.79. G: standalone 0.988, corr 0.13, bar 0.37. At 0.25 ivol
  shares: champion 2.788 -> 2.852 (+0.070); robustness **27/27 cells positive**
  (23/27 p<0.10 individually). Earlier bench = full-slot sizing artifact.
  Breadth-H re-tested under both methods: genuinely benched. Contribution test
  amended (backtest agent 4b). **PENDING HUMAN DECISION: Phase-2 incubation of
  Book G (zero-weight settle tracking, 60 trading days) before capital.**

- **2026-08-16 | Cycle 14 | Orthogonal program #28-32 — ALL tested-no-gain (structural)**
  Carry study: UVXY bleed -1.08%/d in calm, +0.92%/d when VIX rising. Convexity
  book paid +0.88%/d on stress days but carry killed the portfolio delta
  (-0.043, p=0.12). Resid-F 0.42 (beta = part of the edge, again). Crisis-short
  +0.02 p=0.29. XLU beta-in-disguise (-0.235). CONCLUSION: with price-only data
  the vol-target overlay IS the crisis hedge at zero premium — champion design
  locally complete. Registry #108.

- **2026-08-16 | Cycle 13 | D21 third sleeve — tested-no-gain (theory-confirming)**
  3-sleeve champion 2.842 (+0.061) but p=0.113 misses the significance gate.
  Diminishing cohort returns exactly as Hoffstein RTL predicts; HM half-life
  says pressure mostly harvested by 21h. Two sleeves stand. (v1 run VOIDED —
  ALLOC_BOOKS filter silently dropped the candidate; harness lesson logged.)
- **2026-08-16 | Semivariance targeting (Wang-Yan) — tested-no-gain**
  CAGR 52.5% but Sharpe -0.14, DD breach. Aggressive-profile variant documented.
- **2026-08-16 | Significance gate ADOPTED (Ledoit-Wolf 2008) — process ⭐**
  D14 vindicated p=0.005; HAR-X retracted p=0.258. Rule 4b added to plan.

- **2026-08-16 | Cycle 12 | Introspection battery — TWO IMPROVEMENTS, ONE PROMOTION-GRADE**
  (a) HAR-X w/ SPY intraday RV: 2.632 (+0.05, marginal ⭐ — fixed Cycle 7's proxy
  problem); (b) H cash-sleeve: 2.628 (+0.048, near-miss); (c) **D14 fifth book:
  2.781 / 43.9% / -8.2% — +0.20 Sharpe, better DD, robustness 11/11, verifier
  PASS. QUALIFIES FOR CHAMPION PROMOTION — pending human decision to add the
  D14 sleeve to the live engine (settle replay + ALLOC_BOOKS + IvolVT).**
  Stack (D14+HAR-X) adds nothing (2.770, -6.5pp CAGR) — simple config wins.
  Records: portfolio_champ_d14, portfolio_champ_harx, book_d14_daily.csv.

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
