# Improvement Queue

Consumed top-down by Improvement Cycles (see SELF_IMPROVEMENT_PLAN.md §2-3).
Each item: spec, source, expected effect, and what "done" means. When an item
completes, move it to the Done section with its verdict; when the queue drops
below 3 items, run a literature sweep to refill (check papers_read.md first —
re-testing rejected ideas is low priority and needs new evidence or a new
angle; unread papers and untested ideas come first).

## Queue

### 44. M4 F-dip sleeve  [after DU decision, ~mid-Oct incubation review]
- **Spec:** capitulation entries within top-50 750h-momentum names, 40-120h holds.

### 45. NDX membership history acquisition  [DATA — unblocks index-rebal reverse side]
- **Spec:** scrape Nasdaq annual reconstitution press releases (2008-2025) to
  build NDX membership through time. Unblocks: (a) "S&P member added to NDX"
  event study; (b) December reconstitution prediction (mkt-cap rank vs
  smallest member — shares_outstanding.parquet x price gives cap history);
  (c) exact historical dual-index flags for the #139 migration cohort.
  Note: live Wikipedia NDX constituents table is GONE (universe scraper
  falls back to cache); pinned-revision workaround documented in cycle25.

### 46. Earnings-calendar acquisition retry  [DATA — throttled 2026-08-17]
- **Spec:** fetch_earnings_dates.py is checkpoint/resume-ready; per-ticker
  yfinance earnings endpoint was hard rate-limited (batch chart endpoint
  unaffected). Retry off-hours / lower frequency; then run
  cycle24_earnings_premium.py (Frazzini-Lamont gates pre-registered).

## Done

- **2026-08-17 | Cycle 23: QUEUE CLEARED — #41/#34 shipped, #23/#24/#26/#40/#27 verdicts, 0 adoptions**
  ENGINEERING: #41 quiet ordering wired into live replay_D (D8 only; anchor
  raw 1.662 unchanged, quiet 1.842 on live basis; fail-soft to raw). #34
  daily-panel splices ROOT-FIXED: 24 phantom >100% moves repaired (rebased
  adjusted refetch; 104 tickers had basis drift), sync_daily_close now
  adjusted-source + per-ticker rebase + jump guard + completed-days-only.
  RESEARCH (all anchored EXACTLY after finding the engine's one-way-TC
  convention; prior #26 VOID resolved): #23 diagnostic — cohorts real
  (deepest rebound hardest; gap-heavy slower), unlocked #24; #24 routing
  fails both ways (averaging-beats-adaptation, 6th confirmation); #26 lazy
  exits lose everywhere; #40 defer-to-open SIGNIFICANTLY HARMFUL (-0.213
  p=0.001 — the first hours ARE the rebound); #27 C meta-labeling harmful
  (-0.244 p=0.095, 175 trades ever = nothing learnable). Registry #131-134.
  Queue is now EMPTY except #44 (gated on DU decision, mid-Oct). Stale
  merged items (#21/#22/#39 dups) removed.

- **2026-08-17 | Cycle 22: BLOCKER RE-TACKLE — all three data blockers RESOLVED, zero adoptions**
  (A) #37 FOMC: scrape v2 (meeting-anchored regex, last day of span) PASSED
  validation 29/31 plausible years -> cached data/cache/fomc_dates.csv (232
  dates 1997-2026, permanent). Announcement-day SPY book: Sharpe 0.19 /
  CAGR 0.7% after costs (avg event day +0.19%, 8 days/yr exposure) — drift
  exists but is too thin to be a book; post-2011 decay consistent with
  Lucca-Moench literature. #37 CLOSED (tested, fails gates).
  (B) VIX-TS last open angle: ^VIX9D>^VIX inversion tightening the vol target
  HURTS at every level (10%: -0.069 p=0.071; 8%: -0.126 p=0.035 significantly
  harmful; 12%: -0.025). Inversion days are exactly when D's capitulation
  signals are richest — cutting exposure there fights the champion's engine.
  VIX-TS thread now FULLY closed (UVXY unbuyable #124 + overlay angle dead).
  (C) Shares-data harvest extension: quiet ordering on D14 sleeve improves
  standalone (2.706 vs 2.665, anchor -0.000) but champion delta +0.036
  p=0.175 — fails rule 4b. Quiet ordering stays D8-ONLY (simplifies #41
  wiring: only the D8 sleeve changes).

- **2026-08-17 | Queue #42/#43 — one ⭐⭐ adoption-grade find, one decisive validation**
  #43 D-1997: mechanism requires intraday frequency (moat, not failure). #42:
  QUIET-CAPITULATION ordering +0.087 p=0.022, 5/5 robust, supersedes sector-rel;
  recorded book_d8_quiet; live wiring folded into #41.

- **2026-08-17 | Mix-ideation M2/M5 — tested-no-gain (anchored)**
  top_n ensemble wash; F lookback blend drags. D-ensemble axes map complete.
  M1-A/M3/M4 queued (#42-44).

- **2026-08-17 | #26/#23 attempt — VOIDED (harness, not verdicts)**
  #26 asym-bands custom loop produced -0.333 vs 2.740 WITHOUT an anchor run —
  violates the patch-with-anchor rule; result void, idea UNTESTED. #23
  diagnostic crashed pre-output. Both return to queue for properly anchored
  implementations (Saturday cycle). Lesson re-confirmed: custom engine loops
  without anchors produce unusable numbers.

- **2026-08-17 | Data-unlock round (Cycle 21) — 2 permanent assets, 3 questions closed**
  FOMC scrape rejected by own validation (honest block; #37 needs better source).
  VIX term structure cached -> convexity FINALLY CLOSED (#124: UVXY bleeds in
  all regimes incl. backwardation). Shares outstanding cached (523 names, 2015+)
  -> MS retested with TRUE turnover: better (0.999) but corr 0.67 to F —
  redundant, closed on the merits with the right variable.

- **2026-08-17 | #39b/c D filters — battery CONCLUDED**
  MAX filter wash; gap filter significantly harmful (p=0.002). D-defense
  battery final score: sector-relative ranking adopted (⭐), 2 filters rejected.
  Queue now: #37 (FOMC data), #40/#41 (engine work), #23/#24/#26/#27 (careful
  D/C work) — all Saturday-cycle material. No untested quick candidates remain.

- **2026-08-17 | #39a industry-adjusted D8 — IMPROVED ⭐ + ALL-IN computed**
  Sector map downloaded (permanent: sector_map.json). Adjusted D8: 2.808
  standalone, champion 2.800 (+0.019 p=0.038), perturbation 4/6. ALL-IN
  (adjusted blend + G/X/DU): **Sharpe 2.930 / CAGR 43.4% / MaxDD -7.7%**
  (delta +0.149 p=0.016, recorded portfolio_allin_v2). Live wiring = #41.

- **2026-08-17 | Cycle 20 | #36 + #38 — tested-no-gain**
  Rebalancing-flow (long-only adaptation): -0.22 standalone, harmful on stack
  (p=0.017); spec caveat: short leg dropped. Path convexity: 0.96 standalone,
  clears bare champion, absorbed by full stack (+0.023) — the recurring
  pattern. Remaining queue: #37 (FOMC calendar), #39 (D defense battery),
  #40 (closing-auction tilt) — Saturday cycle material.

- **2026-08-17 | #33 rollback recalibration + RE-BASELINE — DONE**
  Settle-basis champion computed (portfolio_ivolvt_settle_basis): Sharpe 1.43 /
  CAGR 24.3% / MaxDD -30.9% -> mechanical bound -38.6% (config
  ROLLBACK_MAXDD_BOUND; flagged for possible tighter absolute override).
  Track record re-baselined: inception 2026-08-18; incident record archived
  (live/state/archive_incident_2026-08/). Dual-basis rule documented.

- **2026-08-17 | #35 POST-MORTEM (rollback breach) — CLOSED, strategy exonerated**
  -28.4% live DD fully attributed to 4 engine defects: (1) replay_F tail-block
  guard kept F structurally flat since inception; (2) flat F -> zero vol -> ivol
  max-weighted it 62.8% (the Book-H cash-drag artifact, in production); (3) A
  held phantom-splice picks (DD/DELL); (4) data starvation since 7/8. ALL FIXED
  AND VERIFIED: F invested 48/48 days at sane 5.8% weight, guards in A/F/G,
  pipeline self-healing. Fixed-engine replay of the same window: -5.8% vs the
  broken engine's -28.4%. RECOMMENDATION: re-baseline IvolVT (option b) —
  reset track-record inception; rollback to FixedEW NOT indicated.

- **2026-08-17 | Cycle 17 | new data + 3 strategies — all no-gain vs full stack**
  Permanent data adds: yields_daily.parquet (1995+), etf_extended_close.parquet
  (29 assets incl. countries + BTC/ETH). Expanded-X redundant (corr 0.68);
  crypto trend good standalone (1.10) but nothing at portfolio; curve tilt dead.
  The full-stack bar (~2.92 projected) is holding against everything in-hand
  data can produce.

- **2026-08-17 | Cycle 16 | four structural candidates — all no-gain, all informative**
  F-tranching wash (RTL needs idiosyncratic cohort differences; F's cohorts are
  regime-identical); D barbell significantly NEGATIVE (0.8 threshold is optimal,
  p=0.041); asset-class capitulation dead; sector reversal 18/18 vs champion but
  ZERO marginal on full stack (redundant with D-complex). Additive capacity of
  in-hand data is saturating — three incubators + D-edge surgical queue remain
  the frontier.

- **2026-08-17 | Cycle 15 | four new-book candidates — one find**
  (ii) weekly x-asset reversal DEAD; (iv) credit regime no-gain; (i) sector
  rotation 9/9 positive BUT redundant vs incubating Book X (corr 0.58, marginal
  +0.003) — benched to prevent double-funding the ETF-momentum factor;
  (iii) **D-uptrend sleeve: champ +0.046 @ p=0.010, robustness 3/3 — PENDING
  DECISION: incubate or adopt as third D-family sleeve** (would lift D-complex
  concentration; sector ETF panel cached as side benefit, unlocks queue #21).

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

### 47. S&P announcement-date acquisition  [DATA — the only live index-rebal angle]
- **Spec:** #139 killed all post-effective-date variants; literature edge is
  announcement->effective (5-10 day window). Source: S&P DJI press-release
  archive (spglobal.com news; dates only, no paywall content needed).
  If acquired: front-run test on adds AND the dual-index migration cohort
  (which showed the STRONGEST post-effective decay = most pre-priced).
