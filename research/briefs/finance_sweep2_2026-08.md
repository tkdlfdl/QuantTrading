# Finance-Bucket Deep Sweep, Round 2 — August 2026

**Scope:** Short-horizon reversal (Book D's dominant edge) + recent (2015–2026) finance-journal
innovations not in the registry. Threads: (1) reversal refinements, (2) multi-horizon/tranching,
(3) liquidity-provision economics, (4) crowding/decay risk, (5) recent JF/JFE/RFS price/volume-only
signals with OOS evidence.
**Constraint set:** US-equity OHLCV only — daily 1997+ (~523 names incl. SPY/UVXY/^VIX, volume),
hourly 2019+ (515 names). No fundamentals/news feeds/earnings calendar/order-book/auction-imbalance
data. Static GICS sector map is buildable (one-time; survivorship caveat applies).
**Frontier to beat:** Champion portfolio (ivol + 15% vol-target + panic gate + D8/D14 sleeves)
Sharpe **2.781, CAGR 43.9%, MaxDD -8.2%** (2019–2026). Book D alone: 2.74, 300/300 combos positive,
8/8 years.
**Integrity baseline for all specs:** signals from bars ≤ t−1; rolling/expanding statistics only;
next-bar execution; chronological splits; McLean–Pontiff ~50% haircut on published effects;
synthesis/original ideas get NO expectation credit until tested.
**Tested-and-failed (not re-proposed):** VIX-scaling of D (#28 — D self-loads in high vol),
market-mode/residual stripping of D's score (#47–48 — the beta component IS part of the edge),
volume-shock book (#36 — corr 0.54 to D), hold-grid replacement 7/14/21h (#45 — but the *blend*
was untested and became D14, registry #84), overnight re-timing at multi-week holds (#33),
Medhat–Schmeling turnover conditioning as a book (#34).

All citations verified via web search 2026-08-16 (four parallel verification threads). None
fabricated. Unverifiable items are explicitly flagged.

---

## 1. Papers reviewed (one-line verdicts, incl. dead ends)

### A. Reversal refinements (cross-sectional conditioning of D's signal)

1. **Hameed & Mian (2015), "Industries and Stock Return Reversals," *JFQA* 50(1-2):89–117.** —
   VERIFIED. Intra-industry (industry-demeaned) reversals are larger, more consistent, and
   **prevalent in large, liquid stocks**; driven by non-informational order imbalances; stronger
   after market declines and in volatile periods. **VERDICT: strongest refinement candidate —
   feeds Idea 2. Tension with #47 resolved: sector-demeaning removes industry momentum (an
   anti-signal per Da–Liu–Schaumburg), not the market-beta bounce that #47 showed D needs.**
2. **Da, Liu & Schaumburg (2014), "A Closer Look at the Short-Term Return Reversal,"
   *Management Science* 60(3):658–674.** — VERIFIED. Reversal decomposes into across-industry
   momentum (bad) + within-industry components; only the residual non-fundamental part is
   reliably positive; residual-targeted reversal earns ~3x risk-adjusted (1982–2009). **VERDICT:
   full residual isolation needs analyst cash-flow proxies we lack; the implementable slice is
   sector-relative ranking (Idea 2).**
3. **Tetlock (2011), "All the News That's Fit to Reprint," *RFS* 24(5):1481–1512.** — VERIFIED.
   Stale-news-day returns reverse over the following week. **VERDICT: no news feed — mechanism
   context only for the gap-proxy filter (Idea 3 variant).** Distinct from Tetlock 2007 (#2).
4. **Chan, W.S. (2003), "Stock Price Reaction to News and No-News," *JFE* 70(2):223–260.** —
   VERIFIED. News-driven moves drift (esp. bad news); no-news extreme moves reverse — but effect
   mainly in smaller stocks. **VERDICT: rationale for news-proxy exclusion; small-cap caveat
   blunts urgency for our universe.**
5. **So & Wang (2014), "News-driven return reversals: Liquidity provision ahead of earnings
   announcements," *JFE* 114(1):20–35.** — VERIFIED. Short-term reversal returns are **~6x
   larger around earnings announcements** — MM inventory-risk compensation spikes
   pre-announcement. **VERDICT: kills the naive "exclude earnings names" idea — the sign is
   OPPOSITE; danger is only fading the announcement move itself. Reframed as gap-proxy variant
   in Idea 3.**
6. **Da, Gurun & Warachka (2014), "Frog in the Pan," *RFS* 27(7):2171–2218.** — VERIFIED.
   Momentum lives in continuous information arrival, dies/reverses under discrete (jumpy)
   arrival. Inverse for us: jumpy declines revert, smooth declines continue. **VERDICT:
   path-jumpiness feature for Idea 3's diagnostic (12-month formation in paper → our 104h
   window is synthesis, no expectation credit).**

### B. Overnight/intraday decomposition corpus

7. **Bogousslavsky (2021), "The Cross-Section of Intraday and Overnight Returns," *JFE*
   141(1):172–194.** — VERIFIED. Anomalies accrue at specific times of day (size/illiquidity in
   last 30 min). **VERDICT: motivates hour-of-day attribution of D's own fills — cheap
   diagnostic inside Idea 3.**
8. **Barardehi, Bogousslavsky & Muravyev, "What Drives Momentum and Reversal? Evidence from Day
   and Night Signals," accepted *RFS* (2026), SSRN 4069509.** — VERIFIED. 1926–2019:
   intraday-formed signals drive momentum/underreaction (trading-revealed private info);
   overnight-formed signals (news) don't. **VERDICT: gives a testable sign prediction for
   splitting D's 104h deviation into overnight vs intraday legs — core of Idea 3. Note: sign is
   contested between this (overnight-oversold reverts better) and the pressure logic
   (intraday-pressure reverts) — exactly why the spec is diagnostic-first.**
9. **Akbas, Boehmer, Jiang & Koch (2022), "Overnight returns, daytime reversals, and future
   stock returns," *JFE* 145(3):850–875.** — VERIFIED. Persistent overnight-up/daytime-down
   "tug of war" predicts returns. **VERDICT: corroborating; monthly horizon — context.**
10. **Berkman, Koch, Tuttle & Zhang (2012), "Paying Attention: Overnight Returns and the Hidden
    Cost of Buying at the Open," *JFQA*.** — VERIFIED. Attention stocks open high, reverse
    intraday. **VERDICT: context for open-hour execution caution.**
11. **Baltussen, Da & Soebhag, "End-of-Day Reversal" (SSRN 5039009, 2024; 2nd place Quantpedia
    Awards 2025).** — VERIFIED (working paper). Intraday losers (prior close→3:30pm) outperform
    winners ~0.24%/day in the final half-hour; driven by positive pressure on losers (retail
    attention + short-cover); survives in largest caps. **VERDICT: standalone sleeve needs
    sub-hourly execution (fails our bar granularity/cost tier), but the *entry-tilt* form is a
    free overlay on D — Idea 4.**

### C. Multi-horizon / tranching / rebalance timing luck (who studied our D14 discovery)

12. **Jegadeesh & Titman (1993), *JF* 48(1) — overlapping-cohort construction** (registry #26,
    re-read for a NEW angle). — VERIFIED. K overlapping 1/K-weight cohorts, one entered per
    period; pure variance-reduction over formation timing with no expected-return cost.
    **VERDICT: the canonical precedent — our D8+D14 blend is a two-cohort version along the
    EXIT axis; the ENTRY axis is untested on D (Idea 1). Update #26's note, no new row.**
13. **Hoffstein, Sibears & Faber (2019), "Rebalance Timing Luck: The Difference between Hired
    and Fired," *Journal of Index Investing* 10(1):27–36.** — VERIFIED. RTL = dispersion between
    identical portfolios differing only in rebalance date; N staggered sub-portfolios cut
    timing-luck **variance by 1/N**. **VERDICT: the formal frame for the +0.20 D14 result;
    practitioner venue — haircut effect sizes, but the variance math is mechanical.**
14. **Hoffstein, Faber & Braun (2020), "The (Dumb) Timing Luck of Smart Beta," SSRN 3673910.** —
    VERIFIED. RTL often >100bp/yr for factor indices; calendar-year gaps >40% from schedule
    choice alone; RTL rises with turnover and concentration. **VERDICT: D is high-turnover and
    concentrated (top-20) — exactly the RTL-exposed profile; supports Idea 1.**
15. **Gârleanu & Pedersen (2013), "Dynamic Trading with Predictable Returns and Transaction
    Costs," *JF* 68(6):2309–2340.** — VERIFIED. Optimal policy trades partially toward an aim
    portfolio weighted to slow-decaying signals — mathematically an exponentially-weighted
    ladder of past cohorts, i.e., the continuous-time limit of horizon blending. **VERDICT:
    theory says blend exits in proportion to residual alpha net of costs — measure D's
    alpha-by-hour-held curve rather than guessing (Idea 1 diagnostic).**
16. **Qian, Sorensen & Hua (2007), "Information Horizon, Portfolio Turnover, and Optimal Alpha
    Models," *JPM* 34(1):27–40.** — VERIFIED. Weight signals by strength vs decay vs turnover;
    one-month reversal is their type-case fast-decay signal. **VERDICT: supporting framework.**
17. **Novy-Marx (2012), "Is Momentum Really Momentum?" *JFE* 103(3):429–453.** — VERIFIED. The
    "echo" is a formation-window result for momentum, not a reversal exit-horizon result.
    **VERDICT: rejected — not applicable. Notable: the exit-horizon term structure of HOURLY
    reversal appears unpublished — our alpha-decay curve is potentially original territory.**

### D. Liquidity-provision economics (why D is paid, and for how long)

18. **Hendershott & Menkveld (2014), "Price Pressures," *JFE* 114(3):405–423.** — VERIFIED.
    Transitory pressure from intermediary inventory averages **0.49% with half-life 0.92 days**
    (~6.5 trading hours) for NYSE stocks. **VERDICT: the quantitative anchor for D's horizon —
    8h ≈ one pressure half-life, 14h harvests the second; exponential decay implies positive
    but diminishing edge beyond 14h → third-sleeve test (Idea 1).**
19. **Duffie (2010), Presidential Address: "Asset Price Dynamics with Slow-Moving Capital,"
    *JF* 65(4):1237–1267.** — VERIFIED. Reversal after demand shocks is front-loaded with a
    long tail as outside capital arrives. **VERDICT: theoretical justification for
    multi-horizon exits — alpha is distributed along a decay curve, not at a point.**
20. **Bogousslavsky & Muravyev (2023), "Who trades at the close?" *J. Financial Markets* 66.** —
    VERIFIED. Closing auctions 7.5% of volume (2018, vs 3.1% 2010); closing-price deviations
    reverse largely overnight. **VERDICT: no auction feed, but last-hour return/volume-share
    proxies are computable — feeds Idea 4's conditioning.**
21. **Grossman & Miller (1988), "Liquidity and Market Structure," *JF* 43(3):617–633.** —
    VERIFIED. Foundational immediacy-provision frame. **VERDICT: context only.**
22. **Dai, Medhat, Novy-Marx & Rizova (2024), "Reversals and the Returns to Liquidity
    Provision," *FAJ* 80(2):122–151 (NBER w30917).** — VERIFIED. **Higher volatility → stronger
    but faster-decaying reversals; lower turnover → slower but eventually stronger reversals.**
    **VERDICT: the sleeper hit of this sweep — implies assigning names to D's 8h vs 14h sleeves
    by stock-level vol/turnover instead of duplicating the universe — Idea 5.**
23. **Bid-ask bounce contamination (classic critiques + EOD-paper controls).** — Checked.
    Large-cap spreads 1–3bp vs our 10bp/side cost assumption. **VERDICT: our cost model already
    over-covers bounce; no action.**

### E. Crowding/decay risk (is reversal dying? — answer: not at our horizon)

24. **Chordia, Subrahmanyam & Tong (2014), *J. Accounting & Economics* 58(1):41–58.** —
    VERIFIED. Anomaly returns roughly halved post-decimalization; attenuation tracks arbitrage
    capital. Reversal-specific point estimates not extractable from open sources — flag for
    full-text read. **VERDICT: canonical decay citation for daily/weekly reversal; silent on
    hourly.**
25. **Jacobs & Müller (2020), "Anomalies across the globe: Once public, no longer existent?"
    *JFE* 135(1):213–230.** — VERIFIED. US is the only market with reliable post-publication
    decay. **VERDICT: keep the ~50% haircut; nothing new to implement.**
26. **Ignashkina, Rinne & Suominen (2022), "Short-term reversals, returns to liquidity provision
    and the costs of immediacy," *J. Banking & Finance* 138.** — VERIFIED. ~29% of excess
    returns revert within a month; reversion is exponential; ~20% of daily volatility is
    transitory; reversal present even in the 100–500 largest stocks, decayed-but-surviving in
    recent decades. **VERDICT: best single citation that large-cap reversal = compensated
    liquidity provision that persists; transitory-vol share is a computable conditioning
    feature (feeds Idea 5).**
27. **Kurth, Eisler, Rej & Bouchaud (2026), "Is Trend Still Your Friend? A Microstructural
    Account of the Demise of Short-Term Trend-Following," arXiv:2607.01550.** — VERIFIED
    (preprint; CFM/Bouchaud group). Short-term (intraday-to-daily) trend profitability has died
    through the 2010s–2020s with **mean reversion increasingly dominant at short horizons**;
    mechanism: flat-inventory algorithmic market-making absorbing directional flow. **VERDICT:
    strongest recent evidence the short-horizon landscape shifted TOWARD reversal — consistent
    with D's 8/8 years; preprint, so weight accordingly.**
28. **"Mesoscale niche" hypothesis** (multi-hour reversal too slow for HFT inventory cycles,
    too fast for daily-rebalance arbitrage capital). — **NO direct citation exists.** Kurth et
    al. + Ignashkina et al. are consistent with it, but the horizon-gap argument is our own
    synthesis — origin class 4, labeled as such wherever used.

### F. Recent price/volume-only signals (2018–2026) — mostly logged for later

29. **Jiang, Kelly & Xiu (2023), "(Re-)Imag(in)ing Price Trends," *JF* 78(6):3193–3249.** —
    VERIFIED. CNNs on price/volume chart images beat standard momentum/reversal signals; pure
    price/volume input. Large-cap/VW attenuation not verifiable from open abstract — check
    before committing. **VERDICT: pairs with pending #53 (Gu–Kelly–Xiu) as a quarter-scale
    Book G project; not a D enhancement. Registry: pending.**
30. **Kelly, Malamud & Zhou (2024), "The Virtue of Complexity in Return Prediction," *JF*
    79(1).** — VERIFIED. Over-parameterized market timing works OOS; but uses Goyal–Welch
    predictors (some fundamental) and market-timing overlays keep washing in our champion
    (#24/#42/#77). **VERDICT: deprioritized.**
31. **Dong, Li, Rapach & Zhou (2022), "Anomalies and the Expected Market Return," *JF*
    77(1):639–681.** — VERIFIED. Anomaly long-short returns predict the market OOS. **VERDICT:
    another market-timing gate — our least productive category; deprioritized.**
32. **Della Corte & Kosowski, "Market Closure and Short-Term Reversal" (working paper, CICF).**
    — VERIFIED (WP). Reversal concentrated around market closure. **VERDICT: corroborates
    Idea 4; not separately actionable.**

### Dead ends (searched, nothing usable)
- VIX-term-structure / vol-timing 2020–2026 that is price-only + top-journal + OOS: nothing
  beyond registry #80 (tested-no-gain); found only VIX-futures-dependent practitioner work.
- ML-enhanced short-term reversal with true OOS at journal quality: nothing beyond JKX (#29
  above); remaining hits are low-tier venues.
- Published exit-horizon term structure for intraday/hourly reversal: does not appear to exist
  (opportunity, not a source).
- Blanket earnings-exclusion for reversal: refuted by So–Wang (sign is opposite).

---

## 2. TOP 5 IDEAS (ranked by frontier improvement × implementability ÷ effort)

---

### IDEA 1 — Entry-staggered overlapping cohorts + third exit sleeve (the JT/RTL completion of D14) ⭐ HIGH PRIORITY
- **Origin class:** 1–2 (direct implementation of Jegadeesh–Titman 1993 overlapping construction
  + Hoffstein RTL; we already validated the exit-axis half of it as D14, registry #84).
- **Sources:** [#12 JT 1993], [#13, #14 RTL], [#15 Gârleanu–Pedersen], [#18 Hendershott–Menkveld],
  [#19 Duffie]. Registry #45, #84.
- **Mechanism:** D14 proved exit-horizon noise averaging adds +0.20 Sharpe. RTL theory says
  timing-luck variance falls as 1/N in the number of staggered cohorts — and D currently runs
  a SINGLE entry anchor per sleeve. Two untested axes remain: (a) a third exit sleeve (~21h ≈
  three Hendershott–Menkveld pressure half-lives; #45 showed 21h-as-replacement raises return
  +910% vs +640% at ~2x DD — the *blend* dilutes that DD); (b) entry staggering: each hour h,
  enter a 1/8-weight cohort of that bar's qualifying oversold names (scores from h−1), exit
  after the sleeve hold — the exact JT overlapping estimator, running 8 concurrent cohorts.
  This also removes dependence on the single rebalance anchor (a live-engine fragility) and
  smooths turnover.
- **Economic rationale:** Duffie/H-M show reversal alpha is distributed along an exponential
  decay curve, not concentrated at one horizon; JT/RTL show cohort averaging is variance
  reduction with no expected-return cost. 1/√k math: 1→2 tranches removed ~29% of timing-noise
  std (we captured this as +0.20); 2→3 removes ~13pp more; expect sharply diminishing returns
  after 3–4 — test exactly one addition at each axis, then stop.
- **Signal spec (integrity-hardened):** No change to score, threshold, or top-20 rule. (a) D21
  sleeve: identical engine, hold=21h, equal-weight D8/D14/D21, evaluated at champion level vs
  2.781/-8.2%. (b) Entry-stagger: 8 overlapping 1/8-weight cohorts, entry at bar h open on
  scores through h−1, measure RTL directly by also running the 8 possible single-anchor
  variants (their dispersion = the harvestable noise). Success: higher champion Sharpe, or same
  Sharpe with materially lower anchor dispersion (a robustness win worth banking even at +0.00).
- **Expected effect:** +0.05–0.15 champion Sharpe (haircut applied to the mechanical 1/√k
  extrapolation of our own +0.20 measurement; not a published-effect haircut since the anchor
  evidence is in-house). Failure mode is a wash, not a blow-up.
- **Implementation sketch:** D21 = config change (~1h). Entry-stagger = cohort loop in the D
  engine (~150 LOC, half day). Both reuse the existing champion harness.

---

### IDEA 2 — Sector-relative ranking tie-breaker (Hameed–Mian within-industry reversal)
- **Origin class:** 2 (improve: published cross-sectional refinement adapted to preserve D's
  beta component per registry #47).
- **Sources:** [#1 Hameed–Mian 2015], [#2 Da–Liu–Schaumburg 2014]; registry #47–48.
- **Mechanism:** D ranks by raw bubble score, so its top-20 loads on whichever sector fell
  hardest — partly fading *industry momentum*, which DLS show is the negative-alpha component
  of standard reversal. Hameed–Mian: industry-demeaned reversal is stronger and lives in large,
  liquid stocks (our universe), amplified after market declines (when D fires most). The fix
  that respects #47: keep the −0.8 gate on the RAW score (preserves the market-beta bounce that
  residualization destroyed), but rank qualifying names by sector-demeaned deviation to pick
  the 20 — buying the most oversold names *relative to their own sector*.
- **Signal spec:** One-time static GICS map (11 sectors, ~515 tickers; survivorship caveat
  documented — sector labels are near-time-invariant vs prices, low leakage risk). z_sect =
  stock's 104h log-deviation minus equal-weight sector-mean deviation, all through t−1. Gate:
  raw score < −0.8 (unchanged). Rank: z_sect ascending, top 20. A/B against locked D8/D14 on
  the identical harness, 2019–2026, champion level. Secondary read: per-sector concentration of
  D's current fills (diagnostic for whether the tie-break binds at all).
- **Expected effect:** After ~50% haircut on HM's within-industry uplift, +0.0–0.1 D-sleeve
  Sharpe; also a possible DD reduction via sector-concentration dilution. If the gate list
  rarely exceeds 20 names, the tie-break never binds — check signal-count first (cheap kill).
- **Implementation sketch:** `data/sector_map.py` (static dict) + ranking swap behind a flag in
  the D engine. ~half day including the diagnostic.

---

### IDEA 3 — Path-composition diagnostic: overnight vs intraday vs jumpiness of the oversold move
- **Origin class:** 3 (synthesis: Barardehi–Bogousslavsky–Muravyev day/night signals + Chan
  news-drift + So–Wang + Frog-in-the-Pan, adapted to D's 104h formation window).
- **Sources:** [#8 BBM RFS 2026], [#4 Chan 2003], [#5 So–Wang 2014], [#6 FIP], [#7 Bogousslavsky
  2021], [#3 Tetlock 2011].
- **Mechanism:** D's deviation is path-blind: a −0.8 score built by one overnight earnings gap
  is treated the same as one accumulated through 104 hours of selling pressure. The literature
  says these have different futures — but disagrees on the sign: BBM (overnight = news, no
  continuation signal; intraday = trading-revealed info that *continues*) vs the
  pressure/Grossman–Miller logic (intraday selling pressure is what reverts) vs Chan (news
  moves drift — and gaps are news). FIP adds a second feature: jumpy paths revert, smooth
  paths continue. Because the sign is genuinely contested, the spec is DIAGNOSTIC-FIRST — zero
  mining risk before any strategy change.
- **Signal spec:** For every historical D fill (2019–2026), decompose the trailing 104h
  deviation at entry into: overnight share (Σ open−prev close legs), intraday share
  (Σ close−open legs), and jumpiness (share contributed by the largest 5 hourly bars). All from
  bars ≤ t−1. Sort realized trade returns by feature terciles. Promotion rule (pre-registered):
  only if the top-vs-bottom tercile spread exceeds 2x the average trade return does the feature
  graduate to a ranking tie-break (never a gate — protects signal breadth); then A/B at
  champion level. Variant (So–Wang/Chan proxy): flag entries whose deviation is >50% one
  overnight gap with volume >2x 20d median — the "post-news drop that may not rebound" cohort —
  and read its subsample stats in the same pass.
- **Expected effect:** No credit until the diagnostic reads out (synthesis). Value is
  informational either way: if path composition doesn't matter, D's edge is pure pressure
  harvesting and the earnings worry is empirically closed for our universe.
- **Implementation sketch:** ~100 LOC attribution script over stored D fills; ~2 hours. Only on
  a positive read-out does any engine change happen.

---

### IDEA 4 — End-of-day pressure entry tilt (Baltussen–Da–Soebhag EOD reversal)
- **Origin class:** 2 (improve: published intraday pattern used as a free entry-quality overlay,
  not a standalone sleeve — the standalone form needs sub-hourly execution we don't have).
- **Sources:** [#11 EOD reversal], [#20 Bogousslavsky–Muravyev close], [#32 Della Corte–
  Kosowski]; registry #45 (hour-of-day holds failed — this conditions on *path into the close*,
  a different object).
- **Mechanism:** The last-hour of the session carries structural positive pressure on intraday
  losers (retail attention buying, short-covering into the close) and growing auction flow
  whose price deviations revert overnight. D entries near the close on names that fell hardest
  intraday should therefore carry extra tailwind; D entries on names whose oversold score was
  *manufactured in the last bar* may capture the overnight snap-back for free. Which
  conditioning form wins is empirical; both are observable at decision time.
- **Signal spec:** Within D's qualifying list at the final entry bar of each day, over-weight
  (or tie-break toward) names in the bottom quintile of prior close→15:00 same-day return —
  fully observable before the 15:00–16:00 execution bar. No new trades, no threshold change, no
  extra cost. A/B at champion level; also read the diagnostic slice "entries whose last-bar
  return was < −1σ" from Idea 3's attribution pass before building anything.
- **Expected effect:** Small (+0.0x) after haircut — the published 0.24%/day is L/S gross in a
  30-minute window we can only approximate hourly. Justified purely by its near-zero cost and
  shared infrastructure with Idea 3.
- **Implementation sketch:** Piggybacks on Idea 3's attribution script; engine change is a
  ranking tweak on the last daily entry bar only. ~2 hours incremental.

---

### IDEA 5 — Vol/turnover-matched sleeve assignment (Dai–Medhat–Novy-Marx–Rizova horizon matching)
- **Origin class:** 3 (synthesis: DMNR's cross-sectional reversal-speed result mapped onto our
  multi-sleeve architecture).
- **Sources:** [#22 DMNR FAJ 2024], [#26 Ignashkina et al. 2022], [#18 Hendershott–Menkveld].
- **Mechanism:** DMNR: high-vol names revert stronger but *faster*; low-turnover names revert
  slower but eventually stronger. Our D8 and D14 sleeves currently trade the SAME names at two
  horizons — harvesting timing noise but ignoring that the optimal horizon differs by name.
  Assign each qualifying name to the sleeve matching its predicted reversal speed: high
  trailing-vol names → D8 (fast decay: exit before the edge is gone), low-vol/low-turnover
  names → D14/D21 (slow decay: hold for the larger eventual bounce). Same total book, same
  entry rule — the sleeves stop being clones and become a horizon-matched book.
- **Signal spec:** Speed proxy = 20d realized vol z-score (and/or 20d volume-intensity z, our
  #34 proxy), computed through t−1. Qualifying names sorted by proxy; top half → 8h sleeve,
  bottom half → 14h sleeve (median split — no free threshold). A/B at champion level vs the
  clone-sleeve D8/D14 champion. Caution: this *reduces* the timing-noise diversification that
  produced +0.20 (each name now held at one horizon, not two) — the test is whether horizon
  matching beats noise averaging, or whether a hybrid (match + partial overlap) dominates both.
  Run after Idea 1 settles so the sleeve architecture is final.
- **Expected effect:** No credit until tested (synthesis); DMNR's published cross-sectional
  spread is monthly-horizon — transfer to hourly is unproven. Informative either way about
  WHERE D's edge decays fastest.
- **Implementation sketch:** Sleeve-router function in the D engine (~80 LOC); reuses existing
  vol stats. ~half day. Sequenced after Idea 1.

---

## 3. For the registry (proposed new rows)

Baselines unchanged (2026-08-15 retest): champion 2.781/-8.2% is the bar. `pending` = specced
above, not yet run. Suggested numbering continues from #84.

| # | Paper (Author, Year, Journal) | Family | Verdict (proposed) | Notes |
|---|-------------------------------|--------|--------------------|-------|
| 85 | Hameed & Mian (2015), JFQA | Reversal/industry | pending | Idea 2: sector-demeaned ranking tie-break; large-cap prevalence verified |
| 86 | Da, Liu & Schaumburg (2014), Mgmt Sci | Reversal/decomposition | pending | Residual reversal 3x; implementable slice = sector-relative (merges into #85) |
| 87 | Tetlock (2011), RFS | Reversal/news | rejected | Stale-news reversal needs news feed; mechanism context for #89 proxy |
| 88 | Chan (2003), JFE | Reversal/news | rejected | News drifts / no-news reverts; mainly small caps — proxy context only |
| 89 | So & Wang (2014), JFE | Reversal/earnings | pending | Reversal ~6x LARGER around earnings — kills blanket exclusion; gap-proxy subsample in Idea 3 |
| 90 | Bogousslavsky (2021), JFE | Intraday anomalies | pending | Hour-of-day accrual; feeds Idea 3/4 diagnostics |
| 91 | Akbas, Boehmer, Jiang & Koch (2022), JFE | Overnight/intraday | rejected | Tug-of-war predictor at monthly horizon; context |
| 92 | Berkman, Koch, Tuttle & Zhang (2012), JFQA | Overnight/attention | rejected | Open-price attention premium; execution caution only |
| 93 | Barardehi, Bogousslavsky & Muravyev (2026), RFS (acc.) | Day/night signals | pending | Core of Idea 3 path decomposition; sign contested vs pressure logic |
| 94 | Da, Gurun & Warachka (2014), RFS | Info discreteness (FIP) | pending | Jumpiness feature in Idea 3; 12m→104h transfer is synthesis |
| 95 | Baltussen, Da & Soebhag (2024), SSRN 5039009 | EOD reversal | pending | Idea 4 entry tilt; standalone sleeve infeasible at hourly bars (WP, not yet journal) |
| 96 | Hoffstein, Sibears & Faber (2019), JII | Tranching/RTL | pending | 1/N timing-luck variance; formalizes D14 (+0.20) — Idea 1 |
| 97 | Hoffstein, Faber & Braun (2020), SSRN | Tranching/RTL | pending | RTL >100bp/yr, worst for concentrated/high-turnover — D's profile |
| 98 | Gârleanu & Pedersen (2013), JF | Optimal trading/decay | pending | Partial-adjustment = horizon-blend limit; motivates alpha-by-hour-held curve |
| 99 | Qian, Sorensen & Hua (2007), JPM | Signal decay/turnover | rejected | Framework only; no distinct test beyond #98 |
| 100 | Novy-Marx (2012), JFE | Momentum echo | rejected | Formation-window result, not exit-horizon; NB hourly reversal term structure unpublished |
| 101 | Hendershott & Menkveld (2014), JFE | Liquidity/price pressure | pending | 0.49% pressure, half-life 0.92d ≈ 6.5h — quantitative anchor for 8h/14h/21h holds |
| 102 | Duffie (2010), JF | Slow-moving capital | rejected | Theory support for multi-horizon exits; nothing to implement directly |
| 103 | Bogousslavsky & Muravyev (2023), JFM | Closing auction | pending | Close-deviation overnight reversal; last-hour proxies feed Idea 4 |
| 104 | Grossman & Miller (1988), JF | Liquidity/immediacy | rejected | Foundational frame; context |
| 105 | Dai, Medhat, Novy-Marx & Rizova (2024), FAJ | Reversal speed x-section | pending | Vol→fast/strong, low-turnover→slow/strong — Idea 5 sleeve matching |
| 106 | Ignashkina, Rinne & Suominen (2022), JBF | Reversal/liquidity | pending | Exponential reversion; ~20% of daily vol transitory; large-cap reversal persists |
| 107 | Chordia, Subrahmanyam & Tong (2014), JAE | Meta/decay | rejected | Anomalies ~halved post-2001; reversal point estimates need full text; keeps haircut honest |
| 108 | Jacobs & Müller (2020), JFE | Meta/decay | rejected | US-only post-publication decay; reaffirms McLean–Pontiff prior |
| 109 | Kurth, Eisler, Rej & Bouchaud (2026), arXiv:2607.01550 | Microstructure/regime | rejected | Short-term trend dead, MR dominant at short horizons — strategic tailwind for D (preprint) |
| 110 | Jiang, Kelly & Xiu (2023), JF | ML/price images | pending | Book G candidate with #53; check large-cap attenuation before committing |
| 111 | Kelly, Malamud & Zhou (2024), JF | ML/market timing | rejected | Needs non-price predictors; timing gates keep washing here |
| 112 | Dong, Li, Rapach & Zhou (2022), JF | Anomaly/market timing | rejected | Same category as #111; deprioritized |
| 113 | Della Corte & Kosowski (WP), CICF | Closure reversal | rejected | Corroborates #95/#103; not separately actionable |

Registry #26 (Jegadeesh–Titman 1993): update Notes to add the overlapping-cohort re-read angle
(Idea 1) rather than adding a duplicate row. Registry #45: D21-as-blend now formally motivated
(was flagged as "possible future high-return variant").

**Queue order proposed:** Idea 1a (D21 blend, ~1h) → Idea 3 diagnostic (~2h, informs 3/4) →
Idea 1b (entry-stagger, half day) → Idea 2 (sector tie-break, half day) → Idea 4 (if Idea 3
read-out positive) → Idea 5 (after sleeve architecture settles).

*Sweep executed 2026-08-16 via four parallel verification threads; all effect sizes carry the
McLean–Pontiff ~50% haircut prior; synthesis/original items (#28 mesoscale niche, Ideas 3/5)
carry zero expectation credit until tested.*
