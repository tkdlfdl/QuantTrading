# Mix-Ideation Sweep — 2026-08 (CS/ML x proven components, and component x component)

**Date:** 2026-08-17
**Mission:** Pure synthesis — recombine our proven LEGO bricks (D-core bubble machinery, A/F momentum, C event-reversal, portfolio overlays) with each other and with verified CS/ML methods. Precedent: our two biggest wins were exactly this class (exit-horizon sleeve ensembling +0.20; sector-relative ranking +0.019 p=0.038).
**Frontier:** Champion Sharpe 2.800 / CAGR 44.5% / MaxDD -8.2% (D8 sector-rel + D14 + A/F/C, ivol + 15% vol-target + panic gate). ALL-IN v2 projection 2.930 (with G/X/DU at incubation shares).
**Integrity baseline:** signals shifted >= 1 bar; rolling statistics only; next-bar execution; chronological CV with embargo >= 1 holding period (Lopez de Prado purged/embargoed CV); patch-with-anchor pattern mandatory — every candidate run must first reproduce its baseline exactly on the validated engine (Cycle-4 lesson: custom re-implementations get VOIDED); Ledoit-Wolf significance gate (rule 4b); synthesis ideas get NO expectation credit until tested.
**Citation status:** every named method verified to exist (web search 2026-08-17 or previously verified in `mlcs_sweep_2026-08.md`). One item flagged as listing-only (see R4). None fabricated.

---

## 0. Constraints inherited from our own evidence (binding on all candidates)

These are PROVEN facts from the improvements log / papers_read — candidates that fight them were rejected at the door:

1. **Beta is part of D's edge** — residual/market-mode stripping alpha=0.5 -> 2.178, alpha=1.0 -> 1.362 (Cycle 4). No residualization candidates.
2. **Threshold 0.8 is optimal** — barbell/asymmetric threshold significantly negative (p=0.041, Cycle 16). No threshold tinkering.
3. **Gap filter significantly harmful (p=0.002); MAX filter wash** (#39b/c). Gap/news drops rebound fine — no entry-quality *filters* that cut breadth.
4. **Averaging beats adaptation** (Zakamulin; jump models lost; adaptive params dead). Ensembling = averaging across fixed specs — allowed and proven. Learned *state-switching* — banned.
5. **Ranking refinements work** (sector-relative +0.019 p=0.038) while *filters* fail. The fertile axis on D is HOW TO ORDER the eligible set, never whether to trade it.
6. **Take-every-signal breadth is load-bearing** — D is robust 300/300 grid combos precisely because it harvests the whole cross-section of capitulation. Meta-labeling as a *trade-skip gate* on D threatens this; as a *ranking/sizing tilt within the eligible set* it does not.

---

## 1. Candidate catalog — everything considered

### Advanced to Top 5
- **M1. D selection meta-ranker** (CS/ML x D) — staged feature battery then learned combiner on the eligible-set ordering problem. → TOP 5 #1
- **M2. D top_n sleeve ensemble** (D x ensemble theory) — top-10/20/30 cohort blend. → TOP 5 #2
- **M3. D-score weekly book on the 1997+ daily panel ("D-1997")** (D x frequency transplant) — 28yr out-of-sample-in-time validation + possible new book. → TOP 5 #3
- **M4. F-dip sleeve: momentum universe x capitulation timing** (F x D cross-breed) — buy the dip in top-750h-momentum names. → TOP 5 #4
- **M5. F lookback cohort blend 500h/750h/1000h** (F x ensemble theory). → TOP 5 #5

### Considered and REJECTED (with reasons)
- **R1. GKM high-volume premium as a standalone D-conditioned book.** Already benched Cycle 10 (standalone 0.875 but corr 0.54 to D — redundant). Survives only as a *feature inside M1* (Gervais-Kaniel-Mingelgrin 2001 JF), where within-D correlation is the point, not a bug.
- **R2. Overnight-share (G's signal) as its own D-variant book.** Book G is incubating on exactly this signal (Lou-Polk-Skouras 2019 JFE); a second vehicle double-funds it. Survives only as an M1 feature, flagged: if M1 adopts it with material weight, G's incubation review must test overlap.
- **R3. Deep RL / FinRL-class portfolio agents (recent CS venues).** Sweep of KDD/NeurIPS-adjacent 2024-2026 output found no published cost-aware OOS result that survives our 10bp/side bar (the recurring pattern: Lim-Zohren-Roberts tolerate 2-3bp; DRL papers rarely model costs at all, and the ones that do test on China A-shares or report in-sample-adjacent splits). REJECT for this cycle; standing conclusion from `mlcs_sweep_2026-08.md` unchanged.
- **R4. Neural rankers / LambdaRank-IC-style listwise objectives for M1.** A listing "LambdaRankIC: Directly Optimizing Rank IC" (arXiv 2605.00501) surfaced in search but was not verified beyond the listing — do not cite as evidence. Regardless: our selection sample (~1,500 active trade days, ~20-50 eligible names each) is far too small for neural LTR. M1 caps model class at ridge-logistic / shallow GBT (LightGBM, Ke et al. 2017 NeurIPS) / pairwise LambdaMART (Burges 2010) with heavy regularization.
- **R5. Meta-labeling as a D trade-skip gate.** Violates constraint 6 (breadth is load-bearing). Meta-labeling's home here is Book C (queue #27, Singh-Joubert 2022 "Does Meta-Labeling Add to Signal Efficacy?", verified) where trades are few and fat-tailed. For D it is re-scoped into M1 (ordering, not skipping).
- **R6. Residual bubble score / beta-stripped D variants.** Dead (constraint 1).
- **R7. Learned adaptive thresholds or regime-switched parameters.** Dead (constraints 2, 4).
- **R8. A/F winners as a D universe *filter* (restrict D to trend-up names).** This already exists as the D-uptrend sleeve (Cycle 15: champ +0.046, p=0.010, robustness 3/3, PENDING human decision). Do not duplicate; M4 is deliberately the *mirror* construct on F's side and must be tested against DU for redundancy.
- **R9. D-score capitulation on the 29-asset ETF panel.** Asset-class capitulation dead (Cycle 16); weekly cross-asset reversal DEAD (Cycle 15).
- **R10. C x D cross: rank C's 4-sigma events by bubble-score depth.** C has ~17 events/yr — adding a ranker to a 127-trade/7.5yr sample is pure overfit surface; C's fragility (34/540 combos) says feed it *less* structure, not more. Queue #27 (sizing) is the only sanctioned C adaptivity. REJECT.
- **R11. D score-definition ensembling (MA=52/104/156 blend).** Legitimate averaging axis but the grid shows 104h clearly dominant and neighbors materially weaker (unlike the flat-ish hold and top_n axes) — blending in weaker score definitions is dilution, not diversification. Parked; revisit only if M2 shows the ensemble mechanism generalizes across axes.
- **R12. GKX-style universe-level return forecaster (Gu-Kelly-Xiu 2020 RFS).** Quarter-scale effort, replaces rather than augments proven books; deferred exactly as in the ML sweep. M1 is the GKX idea shrunk to the one place we have evidence ML-sized refinements pay: D's ordering.

---

## 2. TOP 5 (ranked by expected champion delta x implementability)

---

### #1 — M1: D selection meta-ranker (staged: pre-registered feature battery -> learned combiner)

- **Origin components:** D-core eligible-set construction (proven) + sector-relative ranking win (proven, p=0.038) + meta-labeling *sizing* concept (Lopez de Prado 2018; Singh-Joubert 2022) + learning-to-rank at the cross-section (Poh-Lim-Zohren-Roberts, "Building Cross-Sectional Systematic Strategies By Learning to Rank", arXiv:2012.07149 / JFDS 2021 — LTR beats predict-then-rank on CRSP 1980-2019) + newly cached TRUE shares outstanding (turnover) and sector map.
- **Mechanism:** On each entry bar D has ~20-50 eligible names (score < -0.8) and takes the 20 most negative. Score depth is a one-feature ranker. The sector-relative win proves the ordering is improvable; the hypothesis is that a small set of capitulation-quality features re-orders the eligible set better than depth alone. Candidate features (ALL computable at t-1, pre-registered, frozen before any test): (a) score depth; (b) within-sector score z (current champion ranking); (c) turnover spike: day-volume/shares vs trailing 50d percentile (GKM mechanism — attention/visibility rebound; failed standalone at book level, untested as within-D ordering); (d) trailing 20d realized vol; (e) distance from 52wk high (George-Hwang 2004 — note 52wk-high ranking FAILED for F's momentum edge; unknown sign for capitulation rebound — that is fine, the model learns sign); (f) hour-of-day of signal (hold-alignment was no-gain, ranking role untested); (g) trailing 60d overnight-share (LPS clientele tilt — flag per R2); (h) 750h momentum rank (ties to DU sleeve finding).
- **Leak-free spec:**
  - **Stage A (univariate, this cycle):** each feature as a tie-break re-ranker exactly like the sector-rel test — rank eligible names by feature (within the score<-0.8 gate, gate untouched), run the validated D8 engine, Ledoit-Wolf gate vs anchor. 8 cheap runs, zero fitting. Any feature that individually clears p<0.10 OR shows |delta| > 0.02 with the right robustness profile graduates to Stage B.
  - **Stage B (learned combiner, next cycle):** ridge-logistic (default) or depth<=3 GBT, label = trade beats eligible-set median return over the sleeve's hold (relative label — removes market component of the label without touching the signal, so constraint 1 is respected). Walk-forward: train on expanding window ending T-embargo (embargo = 1 hold), refit annually, weights FROZEN between refits. Output used to order the eligible set only — always still take 20 names (constraint 6). Optional sizing tilt (1.25x top decile / 0.75x bottom decile of model score, weights renormalized) as a separate A/B.
  - Turnover/overnight features exist 2015+ (shares outstanding cache): pre-2015 those features are set to cross-sectional median (missing-at-known-date, no leakage), and a 2015+ subperiod readout is mandatory.
- **Expected corr profile:** not a new book — modifies D8's ordering. Champion correlation structure unchanged; risk is concentration drift (monitor sector/name overlap vs current D8 picks; report average Jaccard of selected sets vs baseline).
- **Expected delta:** sector-rel bought +0.019 from one feature. Stage A realistic outcome: 1-2 more features of similar size. Stage B upside if interactions exist (LTR literature says ordering objectives beat scoring objectives); honest expectation champion +0.02 to +0.06. No credit until tested.
- **Implementation sketch (patch-with-anchor):** extend the `rank` mode plumbing just added for #41 (`PARAMS["D"]["rank"]="sector_rel"`) with `rank="feature:<name>"` and `rank="model:<path>"` in `strategies/contrarian_bubble_hourly.py`'s selection step (the `np.argpartition(scores[avail_idx], ...)` line is the single patch point). Anchor: `rank="raw"` must reproduce D8 = 2.740-blend / 2.808 sector-rel exactly. Feature matrix precomputed to parquet in `data/`; model artifact versioned. Stage A ~1 day; Stage B ~2-3 days.

---

### #2 — M2: D top_n sleeve ensemble (top-10 / top-20 / top-30 cohorts)

- **Origin components:** D-core + our proven exit-horizon ensembling (D8+D14, +0.20) + forecast-combination logic (Rapach-Strauss-Zhou 2010 RFS: equal-weight combinations dominate single specs OOS) + Hoffstein rebalance-timing-luck framing (cohort diversification).
- **Mechanism:** top_n is the last un-ensembled D parameter axis. Top-10 is a concentration bet on the deepest capitulations; top-30 dilutes toward the breadth of the signal. The grid shows the axis is well-behaved (all combos positive — 300/300), so blending is averaging across *good* specs, exactly the regime where averaging beats selection (constraint 4 working FOR us). Also mechanically reduces single-name weight (max position 1/30 in one sleeve vs 1/20), trimming idiosyncratic gap risk.
- **Leak-free spec:** three sleeves = the existing validated engine at top_n in {10, 20, 30}, everything else locked (ma=104h, thr=0.8, sector-rel rank, 8h/14h hold structure per current blend); equal-weight sleeve returns daily. No new parameters, no fitting — this is a pure config composition. Caution from D21: the third *hold* sleeve missed the significance gate (p=0.113, diminishing cohort returns) — pre-register the same standard here: adopt only if blended delta clears p<0.10 AND perturbation grid >= 4/6.
- **Expected corr profile:** sleeves will be highly correlated with each other (same signal) — that is expected; the delta comes from variance reduction and position-size smoothing, not decorrelation. Champion structure unchanged.
- **Expected delta:** small, +0.01 to +0.05 champion Sharpe with slightly better MaxDD (smaller max position). Near-free to test.
- **Implementation sketch (patch-with-anchor):** zero engine changes — three runs of the existing D runner + a 10-line blend in the champion composer, same pattern as the D14 adoption. Anchor: top-20-only blend must equal current champion 2.800. Half a day.

---

### #3 — M3: "D-1997" — the D score formula at weekly horizon on the daily panel

- **Origin components:** D-core score formula (tanh(z/2) of log-price vs MA residual) + the 1997+ daily OHLCV panel (already used by A) + classic short-term reversal literature as the external prior (Jegadeesh 1990 JF; Lehmann 1990 QJE; Nagel 2012 RFS "Evaporating Liquidity": reversal = compensation for liquidity provision, spikes in stress) + our sector-relative ranking upgrade.
- **Mechanism:** D monetizes liquidity-provision rebound at hourly frequency but its evidence window is 2019+ only. Transplant the exact formula to daily bars: MA=15d (~104h/7), z-window=15d, buy score < -0.8, hold 2-3d, top-20, sector-relative ordering. Two payoffs: (i) **evidence**: 28 years spanning 2000-02, 2008, 2011, 2015, 2018 — if the score works at weekly horizon across those regimes, D's core mechanism gets an out-of-sample-in-time confirmation no amount of 2019+ testing can buy; if it does NOT work pre-2010, that is a material warning about D's live half-life; (ii) **book**: daily-frequency reversal sleeve with different holding rhythm than D (2-3d vs 8-14h), potentially fundable.
- **Leak-free spec:** daily engine = the contrarian engine's logic on daily bars (signal from close t-1, enter open t, exit close t+hold); costs 10bp/side; sector map static (note: survivorship — the 523-name panel is today's large caps; mandatory caveat on pre-2010 results, mitigate by also running on the point-in-time-ish subset with full 1997 history). Grid pre-registered and SMALL: ma in {10, 15, 20}d, hold in {2, 3, 5}d, top_n in {10, 20} — 18 combos, report ALL (no cherry-pick), headline = center cell (15d/2d/20). Known adjacent negatives respected: this is single-name daily reversal, NOT sector-ETF reversal (killed as redundant at portfolio level) and NOT cross-asset weekly reversal (dead).
- **Expected corr profile:** to D: 0.30-0.45 expected (same mechanism, different frequency — same relationship as D8/D14 but weaker). HARD GATE: if daily-return corr to D-complex > 0.5, it is evidence-only, not a book. To A/F: near zero-to-negative (reversal vs momentum). To C: low (C is 4-sigma events; this is -0.8-score breadth).
- **Expected delta:** as a book, honest expectation Sharpe 0.5-0.9 standalone after costs (large-cap daily reversal decayed post-2000 per the literature; 2-3d holds amortize costs better than 1d classic STR). As evidence, binary and priceless either way. No credit until tested.
- **Implementation sketch (patch-with-anchor):** new thin runner `run_contrarian_bubble_daily.py` reusing `_bubble_matrix` verbatim on the daily close panel (import, don't re-implement); anchor: run it on the 2019+ daily-resampled window and confirm the return profile is sane vs D's daily attribution (not exact — different bars — but directionally consistent; any wild divergence = engine bug). 1-1.5 days.

---

### #4 — M4: F-dip sleeve — capitulation entries restricted to the top-momentum universe

- **Origin components:** F's momentum ranking (750h lookback, proven) + D's entry timing (score < -0.8, proven) + the DU-sleeve finding (Cycle 15: D-side trend-conditioning, +0.046 p=0.010, robustness 3/3 — proof the D x momentum interaction is real) + "buy-the-dip in uptrends" as the joint signal.
- **Mechanism:** F enters its top-5 names at rebalance time regardless of price position — entries after a local run-up are a known drag (F's -39.6% MaxDD partly = buying extended names). Mirror of DU: instead of tilting D toward trending names, tilt F's *entry timing* toward capitulation. Universe each bar = top-50 by trailing 750h return; entry = raw bubble score < -0.8 (unchanged threshold, constraint 2); hold LONGER than D — 40h/120h grid — so the position rides the momentum continuation the universe filter selects for, not just the 8h bounce. The two proven edges are orthogonal in trigger (slow ranking vs fast dislocation): the cross uses each where it is strong — F for WHAT to buy, D for WHEN.
- **Leak-free spec:** momentum rank and bubble score both computed through t-1; entry at t+1 open; per-ticker non-overlap as in D; top_n=5-10 concurrent; costs 10bp/side. Grid pre-registered: universe cutoff {30, 50}, hold {40h, 120h}, top_n {5, 10} — 8 combos, report all. Warmup 750h (2019 partially absorbed, as F). REDUNDANCY GATES (both hard): daily-return corr to F < 0.6 AND corr to DU sleeve < 0.6; if DU is adopted first, M4 must additionally beat champion+DU, not bare champion.
- **Expected corr profile:** to F: 0.3-0.5 (same names, different entry clock); to D: 0.2-0.4 (same trigger, different universe/hold); to A: 0.2-0.4; to C/B: ~0. The interesting outcome is a return stream *between* D and F that diversifies both; the likely failure mode is "F with extra steps."
- **Expected delta:** no credit (synthesis). Directional prior from DU's p=0.010: the interaction term is real on the D side; unknown whether it survives the F-side framing at longer holds. If it works: a fifth uncorrelated-ish sleeve for the D/F complex, champion +0.03-0.08.
- **Implementation sketch (patch-with-anchor):** patch the validated contrarian engine with an `eligible_mask` argument (universe filter computed from the same close panel via F's cumulative-return code, imported from the F module — no re-implementation); anchor: `eligible_mask=None` must reproduce D8 exactly; then mask + hold grid. ~1.5-2 days.

---

### #5 — M5: F lookback cohort blend (500h / 750h / 1000h sleeves)

- **Origin components:** F-core + the F grid's own sensitivity table (500h avg 0.470, 750h 0.533, 1000h 0.682 — all positive, axis monotone and still improving at the boundary) + combination logic (Rapach-Strauss-Zhou 2010) + D8/D14 precedent.
- **Mechanism:** F is our thinnest-evidence book at optimal params (~7 rebalances/yr, 56 trades total) — a single-point spec on a monotone axis whose optimum sits at the grid edge. Three lookback sleeves hold genuinely different portfolios (5-month vs 3.5-month vs 7-month winners differ at turning points), so blending reduces spec risk AND rebalance-timing luck simultaneously, and effectively triples trade count for inference. Must answer the Cycle-16 F-tranching wash head-on: that test phase-shifted the SAME 750h signal (cohorts regime-identical -> no RTL benefit); lookback cohorts differ in *composition*, which is the ingredient the wash showed was missing. Also note 1000h sleeve partially de-risks the "axis still improving at boundary" overfit concern by construction (we hold the neighborhood, not the point).
- **Leak-free spec:** three runs of the validated F engine at lb in {500, 750, 1000}h, hold=200h, top_n=5, everything else locked; equal-weight daily blend; no fitting anywhere. Warmup = 1000h for the longest sleeve (~5 months of 2019 absorbed). Pre-registered adoption bar: blended-F replaces single-F in champion only if champion delta >= 0 with MaxDD no worse AND blend's standalone Sharpe within 0.1 of single-F (we accept a small standalone give-up for spec robustness only if the champion does not pay for it).
- **Expected corr profile:** sleeves mutually correlated 0.7-0.9 (expected — same factor, staggered horizon); blend's corr to A likely nudges up slightly (broader momentum exposure). Champion structure otherwise unchanged.
- **Expected delta:** champion +0.00 to +0.04; the real payoff is F's fragility profile (spec risk at grid boundary, 56-trade evidence) — insurance priced near zero. Cheapest item on the list.
- **Implementation sketch (patch-with-anchor):** zero engine changes; three F runs + blend in the composer (same pattern as M2/D14). Anchor: 750h-only must reproduce current F standalone 1.548-basis series exactly. Half a day.

---

## 3. Suggested sequencing

1. **M2 + M5 first** (same afternoon): pure config composition, both reuse the D14-adoption pattern, immediate champion A/Bs.
2. **M1 Stage A** next (1 day, 8 univariate runs) — highest expected-value axis; Stage B only on Stage-A graduates.
3. **M3** (evidence run has standalone value regardless of book outcome).
4. **M4 last** — wait for the DU sleeve human decision (R8) so the redundancy gate is well-defined.

## 4. Registry entries suggested

| Item | Family | Disposition |
|---|---|---|
| Poh, Lim, Zohren & Roberts (arXiv:2012.07149, JFDS 2021) — Learning to Rank cross-sectional strategies | ML/ranking | TEST at selection level (M1 Stage B) |
| Singh & Joubert 2022 — Does Meta-Labeling Add to Signal Efficacy? | ML/sizing | Already queued for C (#27); re-scoped for D as ordering-only (M1) |
| Ke et al. 2017 (NeurIPS) — LightGBM | tool | Tool-shelf for M1 Stage B |
| Rapach, Strauss & Zhou 2010 (RFS) — Out-of-sample forecast combination | ensembling | Prior support for M2/M5; registry keeper |
| Jegadeesh 1990 (JF); Lehmann 1990 (QJE); Nagel 2012 (RFS) — short-term reversal / liquidity provision | reversal | External prior for M3 (D-1997) |
| George & Hwang 2004 (JF) — 52-week high | ranking feature | M1 Stage-A feature only (failed as F ranker — different role here) |
| Recent DRL/FinRL corpus (2024-2026) | ML/RL | REJECT — no cost-aware OOS at our 10bp bar (R3) |
| LambdaRankIC (arXiv 2605.00501) | ML/ranking | UNVERIFIED listing — do not cite as evidence (R4) |
