# CS/ML + Hard-Science Bucket Sweep, Round 2 — August 2026

**Scope:** Five angles: (1) cost-aware ML for residual/short-horizon reversal, (2) parameter-ensemble/model-combination theory for trading rules, (3) physics/complexity (price impact, vol asymmetries, OU mean-reversion), (4) optimal execution/entry timing for known-alpha trades, (5) online/adaptive parameter selection with regret guarantees.
**Constraint set:** US large-cap OHLCV only (daily 1997+, hourly 2019+, +SPY/UVXY/^VIX, volume). Costs 0.1–0.25%/side. No tick/order-book/fundamentals.
**Frontier to beat:** Champion Sharpe **2.781 / MaxDD -8.2%** (registry #84: D8+D14 sleeves = 75% of capital, + A, F, C). Promotion gate: Ledoit–Wolf bootstrap Sharpe-delta p<0.10 (registry #85).
**Integrity baseline:** signals shifted ≥1 bar; rolling stats only; chronological splits with embargo ≥ 1 hold; McLean–Pontiff ~50% haircut on published effects; synthesis ideas get NO expectation credit until tested.
**Verification:** all citations verified via web search 2026-08-15/16 (5 search agents + 1 adversarial checker; 14 load-bearing claims: 11 CONFIRMED, 3 CORRECTED — corrections noted inline, 0 refuted). Nothing fabricated.

**Registry cross-checks that reshaped this sweep (internal evidence trumps published claims):**
- Registry #28 (Nagel 2012): **VIX-scaling Book D already tested-no-gain** (2.479 < 2.501) — D self-loads in high-VIX via signal frequency. Portfolio-level liquidity-provision timing is DEAD; only *cross-sectional* (name-level) conditioning remains untested.
- Registry #47/#48 (Bun–Bouchaud–Potters, Guijarro-Ordóñez et al.): **residualizing the bubble score degrades D** (2.178/1.362 vs 2.740) — oversold-with-the-market bounces harder than idiosyncratically-oversold. This overrides the Blitz residual-reversal thread (below) for our signal.
- Registry #45 (Heston–Korajczyk–Sadka): D hold grid 7/14/21h already tested (< 8h). Clock-aligned *exits* are done; entry-side execution is not.
- Registry #46 (Gao et al. 2018): SPY intraday momentum dead at our cost tier.

---

## 1. Papers reviewed (one-line verdicts)

### Angle 1 — Cost-aware ML for short-horizon/residual reversal alpha

| Paper (verified venue) | One-line verdict |
|---|---|
| Detzel, Novy-Marx & Velikov (2023), *J. Finance* 78(3) | Net-of-cost model comparison framework — adopt the scoring logic, not a signal source. |
| Jensen, Kelly, Malamud & Pedersen (2026), *RFS* adv. art. hhag022 [verified published] | Put costs INSIDE the objective so the model learns slow implementable signals — principle transfers, full pipeline doesn't. |
| Avramov, Cheng & Metzker (2023), *Mgmt Sci* 69(5) | Deep return-prediction dies under reasonable costs/liquid universes — top-journal confirmation of our GKX kill. |
| Azevedo, Hoegner & Velikov (2023/24), SSRN 4702406 [WP] | Best-case for cost-aware ML: net alpha survives, but at monthly rebalance — argues for slow overlays only. |
| DeMiguel, Martín-Utrera, Nogales & Uppal (2020), *RFS* 33(5) | "Trading diversification": netting trades across signals rescues cost-marginal alphas — cheap idea: net D-entries vs F-exits internally. |
| Tobek & Hronec (2021), *J. Fin. Markets* 56 | ML anomaly composite survives on liquid universe but needs fundamentals — outside our data. |
| Blitz, Huij, Lansdorp & Verbeek (2013), *JFM* 16(3) [confirmed: 2x Sharpe net-of-cost, large caps] | Canonical residual reversal — **rejected here: contradicted by internal #47/#48** (beta is part of our edge). |
| Blitz, van der Grient & Honarvar (2024), *JPM* 50(6) | Plain 1-mo reversal decayed; only residual survives — same internal conflict; monthly horizon, not ours. |
| Guijarro-Ordóñez, Pelger & Zanotti (2025), *Mgmt Sci* | Deep stat-arb on factor residuals — already registry #48, tested-no-gain for our signal. |
| Epstein, Wang, Choi & Pelger (2025), arXiv 2510.11616 [WP] | Net Sharpe 2.3 at **5bp** costs — 2–5x below our tier; lesson (joint cost-aware fitting) only. |
| Blitz, Hanauer, Honarvar, Huisman & van Vliet (2023), *FAJ* 79(4) [confirmed] | Short-term alpha survives costs ONLY with asymmetric buy/sell bands (aggressive entry, lazy exit) — **directly portable → Idea 4**. |
| Dai, Medhat, Novy-Marx & Rizova (2024), *FAJ* 80(2) / NBER w30917 [confirmed, authors + directionality] | Reversal conditions on state: high vol → faster/stronger, low turnover → more persistent — **cross-sectional conditioning untested here → Idea 3**. |
| Krauss, Do & Huck (2017), *EJOR* 259(2) | Trees ≈ deep nets on daily OHLCV features; gross-only, daily full turnover — dead at our costs, keep the GBT>DNN lesson. |
| Shen & Xiu (2024), NBER w33421 [WP] | Weak-dense signal regime → ridge/linear beats sparse/deep — model-selection guidance for any future ML ranker. |
| Blitz, Hanauer, Hoogteijling & Howard (2023), *JFDS* 5(4) | 1-mo-target ML alpha ≈ 0 net post-2004; longer targets survive — target-horizon is first-order if we ever build Book G. |

### Angle 2 — Parameter ensembles / rebalance timing luck / combination

| Paper | One-line verdict |
|---|---|
| Hoffstein, Sibears & Faber (2019), *J. Index Investing* 10(1) [corrected: 1/N reduction is on the STD-DEV measure of timing luck] | Offset-portfolio dispersion is real and harvestable by tranching — **core of Idea 1**. |
| Hoffstein, Faber & Braun (2020), SSRN 3673910 [WP] | Hundreds of bps/yr dispersion from rebalance date alone in factor indices — the phase axis is not a detail. |
| Jegadeesh & Titman (1993), *JF* 48(1) (registry #26) | Overlapping-cohort construction = the standard academic ensemble over start dates — re-read for construction, not signal. |
| DeMiguel, Garlappi & Uppal (2009), *RFS* 22(5) (registry #16) | Estimation error swamps optimization — equal-weight the parameter variants, never Sharpe-weight them. |
| Rapach, Strauss & Zhou (2010), *RFS* 23(2) | Simple average of model specs beats every individual model OOS — maps to averaging bubble-score specs. |
| Timmermann (2006), *Handbook of Econ. Forecasting* | Equal weights near-optimal when forecasts are correlated with similar variances — exactly our grid. |
| Smith & Wallis (2009), *Oxf. Bull. Econ. Stat.* 71(3) | Formal reason: weight-estimation variance > equal-weight bias — do not fit ensemble weights. |
| Sullivan, Timmermann & White (1999), *JF* 54(5) | In-sample-best rule fails OOS (Reality Check) — the grid-average, not the corner-best, is the honest live estimate. |
| Goulding, Harvey & Mazzoleni (2023), *JFE* 149(3) (registry #43) [corrected: Garg NOT on published JFE or FAJ versions] | Confirmed: static intermediate blend captures nearly all of dynamic speed-selection (DYN 0.52 vs MED 0.51 Sharpe over 50yr) — averaging ≈ adaptation, at far lower model risk. |
| Levine & Pedersen (2016), *FAJ* 72(3) | All MA/momentum rules are linear price filters; ensembling = filter smoothing; variant overlap nets turnover. |
| Neely, Rapach, Tu & Zhou (2014), *Mgmt Sci* 60(7) | Pooled MA-rule family carries common signal + parameter noise — average scores BEFORE thresholding. |
| Etienne, Ohana et al. (2025), arXiv 2510.23150 [WP] | Parameter ensembles saturate fast; a spread "barbell" ≈ full grid — run 3 spread variants, not 30. |

### Angle 3 — Physics/complexity: impact, vol asymmetries, OU theory

| Paper | One-line verdict |
|---|---|
| Tóth et al. (2011), *Phys. Rev. X* 1, 021006 [confirmed] | Square-root impact law I ≈ Y·σ·√(Q/V) — the correct nonlinear capacity model for the D books (category: sizing/capacity). |
| Bucci et al. (2019), *PRL* 122, 108302 | Impact linear below ~0.1% ADV, √ above — gives the exact participation ceiling below which our flat cost model is honest. |
| Sato & Kanazawa (2025), *PRL* 135, 257401 | δ=1/2 strictly universal (all TSE stocks/accounts) — fix the exponent, one fewer free parameter. |
| Bucci et al. (2019), *Mkt Microstruct. & Liq.* (slow decay) | Impact decays to ~2/3 peak by day-end, power-law after — transient-impact decay ≈ our 8–14h alpha window; extensions beyond ~2–3d buy little. |
| Donier & Bonart (2015), *Mkt Microstruct. & Liq.* | Uninformed-flow impact reverts almost fully — the physics of WHY fading large uninformed moves pays. |
| Hendershott & Menkveld (2014), *JFE* 114(3) [confirmed: 0.49% pressure, 0.92d half-life] | Independent validation that 8–14h holds sit at the transitory-pressure half-life; per-trade gross alpha ceiling ~0.3–0.5% — our cost band is survivable but tight. |
| Nagel (2012), *RFS* 25(7) (registry #28) | Confirmed VIX predicts reversal returns — but portfolio-level scaling already tested-no-gain here; closed. |
| Frazzini, Israel & Moskowitz (2018), SSRN 3229719 | $1.7T live-execution √-cost model, OOS-validated — use to check whether 0.1–0.25%/side is conservative at our AUM (likely yes for large caps at small size). |
| Bouchaud, Matacz & Potters (2001), *PRL* 87 | Leverage effect (index ≫ single-stock) — mechanism already harvested by our Moreira–Muir vol-target overlay; no new lever. |
| Zumbach (2009), *Quant. Fin.* 9(5) | Time-reversal asymmetry: long-window vol informs short-window, not vice versa — validates our EWMA-long-window choice; no new lever. |
| Gatheral, Jaisson & Rosenbaum (2018), *QF* 18(6) (registry #66) | Rough vol: modest RV-forecast gains — already rejected (#60 outcome moot). |
| Cont & Das (2023), arXiv 2203.13820 | Measured roughness may be estimation artefact — reinforces #66 rejection. |
| Wang, Xiao & Yu (2023), *J. Econometrics* 232(2) | fOU RV forecaster — moot per registry #60 (daily proxy defeats RV machinery here). |
| Leung & Li (2015), *IJTAF* 18(3) [confirmed: theory-only, costs in objective] | OU-optimal entry band + exit threshold under costs/stop-loss — principled replacement for fixed clock exit → **feeds Idea 4**. |

### Angle 4 — Execution / entry timing with hourly OHLC

| Paper | One-line verdict |
|---|---|
| Handa & Schwartz (1996), *JF* 51(5) | Limit-buy earns spread only on transitory declines — conceptually aligned with our signal but unfillable-simulatable from bars. |
| Linnainmaa (2010), *JF* 65(4) | Executed limit buys into declines are adversely selected — **bans "bar-Low-touched = filled" backtests; process rule adopted**. |
| Lorenz & Almgren (2011), *Appl. Math. Fin.* 18(5) | Adaptive schedules: aggressiveness-in-the-money — bar-level analogue: split entry across 2 bars when price still falling. |
| Cartea & Jaimungal (2016), *Math. & Fin. Econ.* 10(3) | With expected short-term drift, back-load (buyer, falling) or front-load (bounce) — the clean theory for WHEN to enter → Idea 2 A/B design. |
| Heston, Korajczyk & Sadka (2010), *JF* 65(4) (registry #45) [confirmed] | Same-half-hour periodicity ≥40 days — exit side already tested; entry-hour conditioning remains open (folded into Idea 2 diagnostics). |
| Bogousslavsky (2016), *JF* 71(6) | Infrequent-rebalancing mechanism behind intraday periodicity — mechanism support only. |
| Bogousslavsky (2021), *JFE* 141(1) | Mispricing returns accrue intraday and partly reverse at the close — whether D holds span the close is a first-order design fact → Idea 2 decomposition. |
| Bogousslavsky & Muravyev (2023), *JFM* 66 [confirmed: 8.1bp avg close-auction deviation, ~85% reverts by next morning] | Documented, sized overnight reversion of close dislocations — the 5–10bp entry-improvement candidate → **core of Idea 2**. |
| Gao, Han, Li & Zhou (2018), *JFE* 129(2) (registry #46) | Already tested: dead at our cost tier; only residual use = "down-into-close continues" caution for late-day entries. |
| Zawadowski, Andor & Kertész (2006), *QF* 6(4) | Spreads widen sharply right after extreme moves, eating contrarian paper profit — adopt 2x-cost stress on entries in the bar after >2σ moves. |
| Berkowitz, Logue & Noser (1988), *JF* 43(1) | VWAP is a benchmark, not an achievable fill; (O+H+L+C)/4≈VWAP is UNVERIFIED practice — sensitivity band only, never base-case. |
| Guéant & Royer (2014), *SIAM J. Fin. Math.* 5 | VWAP optimal control needs intraday volume curves finer than hourly — not implementable. |

### Angle 5 — Online/adaptive parameter selection

| Paper | One-line verdict |
|---|---|
| Herbster & Warmuth (1998), *Machine Learning* 32(2) | Fixed-Share tracking regret = the right theory frame; its own bound degrades to uniform averaging when experts are near-indistinguishable — i.e., converges to our ensemble. |
| Garivier & Moulines (2011), ALT/LNCS 6925 | Discounted/SW-UCB for switching bandits — bandit feedback wastes data in our full-information setting; bandit branch CLOSED. |
| Cesa-Bianchi & Lugosi (2006), CUP | Θ(√(T ln N)) regret floor — with ~10³ decision rounds/yr and ~1–3bp/trade gaps between grid neighbors, the learnable gap is ~20–50x below the 1-yr regret floor; decades needed to distinguish hold=8h from 14h online. |
| Singer (1997), *Int. J. Neural Systems* 8(4) | Switching portfolios beat universal portfolios on 22yr NYSE incl. costs — but experts were assets (huge spread), not correlated rule-variants; weak transfer. |
| Li & Hoi (2014), *ACM Comput. Surv.* 46(3) | Online PS survey: frictionless gains routinely vanish after costs — confirms our EG/ONS kill by survey. |
| Liu & Cartlidge (2023), EMSS/arXiv 2208.02901 | Closest match (online continuous-parameter adaptation) exists ONLY in simulation — the absence of real-data post-cost evidence is the finding. |
| ACE Vol. 83 (2024) bandit strategy-selection | Low-quality venue; documents that peer-reviewed bandit rule-selection literature is essentially empty — zero weight. |
| Zakamulin (2014), *J. Asset Mgmt* 15(4) [corrected: OOS adaptive-lookback selection performs at best marginally better than BUY-AND-HOLD, with frictions] | Strongest direct empirical evidence against adaptive lookback selection — don't adapt the 104h MA. |
| Garg/Goulding–Harvey–Mazzoleni FAJ (2024) "Breaking Bad Trends" [corrected: published author list is Goulding, Harvey, Mazzoleni] | Speed adaptation helps only slowly/structurally, at monthly frequency with ~century samples — does not transfer to hourly with 7yr. |
| Joubert (2022), *JFDS* 4(3) + companions | Meta-labeling: secondary model filters/sizes the fixed primary rule — the best-evidenced form of adaptivity for us; adapts EXPOSURE, not parameters → Idea 5. Vendor-affiliated; haircut effect sizes. |
| Bailey, Borwein, López de Prado & Zhu (2017), *J. Comput. Fin.* 20(4) | PBO: sequential selection among correlated low-SNR variants is unreliable — online selection = repeated in-sample selection; favors the ensemble. |

**Angle-5 conclusion: the literature decisively favors AVERAGING over ADAPTATION at our frequency and sample size.** For adaptation: only monthly-frequency, century-sample results (and even there the static blend ≈ dynamic). Against: Zakamulin (direct OOS failure with costs), PBO, the regret arithmetic, and an empty peer-reviewed record for bandit rule-selection. This is independent theoretical backing for the D8+D14 ensemble result (+0.201, p=0.005) and for Idea 1.

---

## 2. TOP 5 RANKED IDEAS

### IDEA 1 — Phase-tranche + parameter-barbell ensemble for the D sleeves (widen the proven 2-horizon ensemble)
- **Origin class:** 1 (direct implementation of published mechanism) + 4 (extends our own D8+D14 result, +0.201 Sharpe, p=0.005).
- **Citations:** Hoffstein–Sibears–Faber (JII 2019); Hoffstein–Faber–Braun (SSRN 2020); Jegadeesh–Titman (JF 1993, overlapping cohorts); DeMiguel et al. (RFS 2009); Smith–Wallis (OBES 2009); Rapach et al. (RFS 2010); Goulding–Harvey–Mazzoleni (JFE 2023, blend≈dynamic); Etienne et al. (arXiv 2025, barbell saturation).
- **Mechanism:** Each D sleeve is one arbitrary draw from a phase distribution: an 8h-hold cycle can start at 8 different bars, a 14h cycle at 14. Rebalance-timing-luck theory says the dispersion across these phase-shifted twins is pure noise, removable ~1/N (std-dev of the luck term) by running K staggered 1/K tranches — the Jegadeesh–Titman overlapping-cohort construction. Same logic on the signal axis: equal-weight a small barbell of thresholds (e.g., 0.7/0.9 around the locked 0.8) and MA lookbacks, because estimation error on "which cell is best" exceeds the cells' true differences (DeMiguel/Smith–Wallis), and the grid-average — not the corner — is the honest live estimate (Sullivan–Timmermann–White).
- **Leak-free spec:** No new fitted parameters. Step 1 (diagnostic): re-run D8 as 8 phase-shifted single-tranche variants and D14 as 14; measure Sharpe dispersion — that dispersion is the harvestable luck. Step 2: convert each sleeve to K overlapping tranches (K=4 for D8, K=7 for D14 if capital granularity binds), 1/K capital each, entries staggered 1–2 bars. Step 3: threshold barbell {0.7, 0.8, 0.9} equal-weighted, positions netted before costs (variant overlap nets most turnover — Levine–Pedersen). Chronological eval 2019–2026 on the existing D harness; promotion via Ledoit–Wolf p<0.10.
- **Expected effect on 2.781/-8.2%:** Directional prior only: variance reduction at ~equal mean → modest Sharpe gain, flat-to-better MaxDD, smoother capital deployment (no more all-in/all-out cycle lumps). The D14 addition already demonstrated the mechanism pays here; tranching is the same mechanism at finer grain. NO numeric credit until tested.
- **Implementation sketch:** `strategies/contrarian_bubble_hourly.py` already parameterizes hold; add `phase_offset` and run the K variants through the existing runner; aggregate daily returns 1/K; net share deltas before applying the 0.1% cost. ~1 day of work, zero new data.

### IDEA 2 — Entry-execution A/B for D entries: signal-bar close vs next-bar open, + overnight/close decomposition
- **Origin class:** 1 (direct) + 3 (synthesis of execution theory with our books).
- **Citations:** Bogousslavsky–Muravyev (JFM 2023: 8.1bp close-auction deviation, ~85% overnight reversion); Bogousslavsky (JFE 2021: day-end anomaly reversal); Cartea–Jaimungal (MAFE 2016: schedule vs expected drift); Lorenz–Almgren (AMF 2011: adaptive splitting); Zawadowski et al. (QF 2006: post-extreme spread widening); Linnainmaa (JF 2010: limit-fill simulation ban).
- **Mechanism:** Our per-trade gross alpha is tens of bps; published, verified price effects of 5–10bp exist at the close/overnight boundary and in the bar-timing of entry after extreme moves. Execution timing is the one lever that adds return without touching the signal. Cartea–Jaimungal: if post-signal drift is still negative (continuation), a buyer optimally back-loads — testable as enter-at-signal-bar-close vs next-bar-open vs split-across-2-bars. Bogousslavsky: whether the 8h/14h hold spans the close determines exposure to day-end reversal of mispricing returns.
- **Leak-free spec:** Same signals, same holds; vary ONLY the entry price rule: (i) next-bar open [current], (ii) signal-bar close (requires signal computable before close — verify no look-ahead: bubble score uses bar t-1 close, so entry at t close is legal), (iii) 50/50 split across bars t and t+1. Decompose every historical D trade's P&L into intraday vs overnight legs (hourly bars + daily O/C suffice). Stress: 2x entry cost on entries in the bar immediately after a >2σ hourly move (Zawadowski). PROHIBITED: any limit-order fill simulation from bar Lows (Linnainmaa adverse selection).
- **Expected effect:** If entry improvement averages even 3–5bp on ~1,478 active trade-days, effect is material at book level; equally possible the current rule is already optimal — either way the overnight/intraday decomposition tells us WHERE D's alpha accrues, which gates future variants. No credit until tested.
- **Implementation sketch:** Entry-price rule is one line in the D backtester; decomposition is a reporting pass over existing trade logs. ~1–2 days.

### IDEA 3 — Cross-sectional liquidity-provision conditioning inside the oversold basket (vol up-weight, turnover down-weight)
- **Origin class:** 1 (direct implementation).
- **Citations:** Dai–Medhat–Novy-Marx–Rizova (FAJ 80(2) 2024 / NBER w30917 — verified authors and directionality); mechanism lineage: Nagel (RFS 2012), Hendershott–Menkveld (JFE 2014).
- **Mechanism:** Reversal is payment for liquidity provision; the payment is larger and faster where volatility is high, and more persistent where turnover is low. Registry #28 killed *portfolio-level* VIX timing (D self-loads in high-VIX). Untested: the *cross-sectional* version — among the 20 selected oversold names, tilt weight toward high-recent-vol / low-turnover names instead of equal weight. Distinct mechanism from #28: it changes WHICH dislocations we get paid most for, not WHEN we trade.
- **Leak-free spec:** At entry, compute per-name 20d realized vol and 20d volume-based turnover proxy (dollar volume / rolling avg — note registry #34 caveat: volume-intensity proxy, not true share turnover) through bar t-1. Weight ∝ rank(vol) − rank(turnover), bounded [0.5x, 2x] of equal weight, renormalized. Hold rules unchanged. Chronological eval; promotion gate as usual.
- **Expected effect:** Published effect is at monthly horizon (haircut hard); our internal analogue evidence is mixed (registry #36: volume shocks select the same names D buys). Prior: small Sharpe delta either way; cheap to test; also yields a diagnostic of what D's edge loads on. No credit until tested.
- **Implementation sketch:** Weighting function swap in D's portfolio-construction step; ~half a day.

### IDEA 4 — Cost-mitigating asymmetric entry/exit bands ("aggressive entry, lazy exit") with OU-derived exit threshold
- **Origin class:** 1 (direct) + 3 (synthesis with Leung–Li theory).
- **Citations:** Blitz, Hanauer, Honarvar, Huisman & van Vliet (FAJ 2023 — verified: net short-term alpha exists ONLY with cost-mitigating band rules); Leung–Li (IJTAF 2015 — OU optimal entry band + exit threshold under costs, theory-only); Hendershott–Menkveld (JFE 2014 — 0.92d pressure half-life bounds sensible max hold).
- **Mechanism:** D currently exits on a clock (8h/14h) regardless of state, then frequently re-buys names still oversold — paying round-trip costs to hold the same exposure. The FAJ paper shows precisely this fix rescues net alpha for short-horizon signals: enter on a strict threshold (z < −0.8) but exit lazily — only when the z-score has reverted above a band (e.g., z > −0.2/0.0, grid) or a hard cap (14–21h) is hit. Leung–Li supplies the principled band: for an OU process the optimal exit is a level, not a time.
- **Leak-free spec:** Exit rule: at each bar, exit iff z_{t-1} > exit_band OR bars_held ≥ cap. Grid: exit_band ∈ {−0.4, −0.2, 0.0}, cap ∈ {14h, 21h}. All z-scores from bar t−1. Caution flags: registry #45 showed longer clock holds raise total return at ~2x DD — the cap must bind; also measure turnover reduction directly (this idea can win on cost saving even at flat gross).
- **Expected effect:** Turnover down materially on re-entered names (D churns names that stay oversold); at 0.2% round trip, saved churn is direct Sharpe. Risk: DD creep via longer effective holds — the promotion gate plus the ~1.2x MaxDD bar (registry rules) decides. No credit until tested.
- **Implementation sketch:** Exit condition change in the D loop + turnover accounting; ~1 day. Test AFTER Idea 1 (bands interact with tranching; run factorial {tranche} × {band} at the end).

### IDEA 5 — Meta-labeling exposure overlay on Book C (secondary win-probability filter on a fixed primary rule)
- **Origin class:** 1 (direct implementation).
- **Citations:** Joubert (JFDS 4(3) 2022) + JFDS companions (calibration/sizing, ensemble meta-labeling); theoretical guardrails: Bailey et al. PBO (JCF 2017), Shen–Xiu (ridge for weak signals).
- **Mechanism:** The angle-5 conclusion is that parameters must NOT adapt — but *exposure* can, safely, via a secondary classifier that predicts whether the primary signal's next trade wins, trained on trade outcomes. It filters false positives and sizes true ones; because it only reduces/reweights trades, costs help rather than hurt. Book C is the right first target (Sharpe 0.93 standalone, -20.8% MaxDD, fragile 4σ signal, ~127 trades — the book with the most to gain from a false-positive filter), NOT the D books (high Sharpe already; a filter can only remove trades from a signal where breadth is the edge).
- **Leak-free spec:** Features at signal time (all through t−1): signal z magnitude, market 1d/5d return, VIX level & 5d change, name's 20d vol, breadth of simultaneous signals. Model: L2-regularized logistic (per Shen–Xiu, linear for weak signals; NOT GBT at n≈127). Expanding-window training with ≥1-hold embargo, first 3 years burn-in, output = position scale ∈ {0, 0.5, 1}. Guard: with ~127 trades, PBO risk is severe — pre-register the single spec above, no feature search; if p≥0.10, kill and log.
- **Expected effect:** Vendor-affiliated literature reports Sharpe/MaxDD gains — apply heavy haircut. Realistic goal: trim C's -20.8% MaxDD tail (its worst trades cluster in identifiable states) with flat Sharpe → portfolio DD headroom. No credit until tested.
- **Implementation sketch:** Standalone `research/` script reading C's trade log; sklearn logistic; ~1–2 days.

### Sidebar (not ranked — implementation hygiene, category (c) risk control)
**Square-root-law capacity/cost audit** (Tóth 2011 PRX; Bucci 2019 PRL; Sato–Kanazawa 2025 PRL; Frazzini–Israel–Moskowitz 2018): one-off notebook computing per-name participation of the D books at current and 10x/100x AUM vs the ~0.1% ADV linear-impact ceiling; adopt cost(Q)=Y·σ·√(Q/V) with δ fixed at 0.5 above it. Also note Bucci (slow decay) + Hendershott–Menkveld: transient-pressure decay bounds per-trade gross alpha at ~0.3–0.5% and validates the 8–14h hold as sitting at the pressure half-life — an independent physics-side confirmation that D's economics are real and that hold extensions beyond ~2–3 days buy little.

---

## 3. For the registry

Suggested new rows (numbering continues from #85; already-registered papers updated in place, not duplicated):

| # | Paper (Author, Year, Journal) | Family | Verdict (suggested) | Notes |
|---|---|---|---|---|
| 86 | Hoffstein, Sibears & Faber (2019), JII | Ensemble/timing luck | pending | Idea 1 core — phase-tranche the D sleeves; 1/N reduction of timing-luck std-dev |
| 87 | Hoffstein, Faber & Braun (2020), SSRN WP | Ensemble/timing luck | pending | Support for #86 — offset dispersion is 100s of bps/yr even monthly |
| 88 | Rapach, Strauss & Zhou (2010), RFS | Forecast combination | rejected | Support-only: averaging specs beats picking; no standalone test |
| 89 | Smith & Wallis (2009), OBES | Forecast combination | rejected | Theory: never fit ensemble weights; support for equal-weight variants |
| 90 | Sullivan, Timmermann & White (1999), JF | Methodology/snooping | rejected | Grid-average, not corner-best, is the honest live estimate; process prior |
| 91 | Levine & Pedersen (2016), FAJ | Signal theory | rejected | Rules=linear filters; ensembling=smoothing; turnover nets across variants |
| 92 | Neely, Rapach, Tu & Zhou (2014), Mgmt Sci | Technical rules pooling | rejected | Average scores before thresholding — folded into Idea 1 design |
| 93 | Etienne et al. (2025), arXiv 2510.23150 | Ensemble saturation | rejected | Barbell ≈ full grid; caps Idea 1 variant count at ~3/axis; WP |
| 94 | Blitz, Huij, Lansdorp & Verbeek (2013), JFM | Residual reversal | rejected | Contradicted by internal #47/#48: beta IS part of D's edge at hourly horizon |
| 95 | Blitz, van der Grient & Honarvar (2024), JPM | Residual reversal | rejected | Same conflict; monthly horizon; logs the decay of plain 1-mo reversal |
| 96 | Blitz, Hanauer, Honarvar, Huisman & van Vliet (2023), FAJ | ST signals/cost bands | pending | Idea 4 — asymmetric entry/exit bands rescue net short-horizon alpha |
| 97 | Dai, Medhat, Novy-Marx & Rizova (2024), FAJ | Reversal conditioning | pending | Idea 3 — cross-sectional vol/turnover tilt (distinct from dead #28 timing) |
| 98 | Detzel, Novy-Marx & Velikov (2023), JF | Methodology/net-of-cost | rejected | Scoring framework; no signal; informs promotion-gate philosophy |
| 99 | Jensen, Kelly, Malamud & Pedersen (2026), RFS | ML/cost-in-objective | rejected | Principle (train with cost penalty) noted for any future Book G; pipeline too heavy |
| 100 | Avramov, Cheng & Metzker (2023), Mgmt Sci | ML critique | rejected | Top-journal confirmation of our GKX kill (#53 stays deferred) |
| 101 | DeMiguel, Martín-Utrera, Nogales & Uppal (2020), RFS | Trade netting | pending | Cheap test: net D-entry vs F-exit orders before costs across books |
| 102 | Shen & Xiu (2024), NBER WP | ML theory/weak signals | rejected | Process prior: ridge/linear for weak-dense; constrains Idea 5 + any Book G |
| 103 | Krauss, Do & Huck (2017), EJOR | ML stat-arb | rejected | Gross-only daily turnover; dead at 10–25bp; GBT≈DNN lesson kept |
| 104 | Epstein, Wang, Choi & Pelger (2025), arXiv WP | ML stat-arb | rejected | Needs 5bp costs; joint cost-aware fitting lesson only |
| 105 | Blitz, Hanauer, Hoogteijling & Howard (2023), JFDS | ML horizon | rejected | Target-horizon first-order; 1-mo ML alpha ≈0 net post-2004 |
| 106 | Tobek & Hronec (2021), JFM | ML anomalies | untestable | Needs fundamentals beyond OHLCV |
| 107 | Tóth et al. (2011), Phys. Rev. X | Impact/capacity | pending | Sidebar audit — √-law capacity model for D books |
| 108 | Bucci et al. (2019), PRL | Impact/capacity | pending | Linear-below-0.1%-ADV ceiling; part of sidebar audit |
| 109 | Sato & Kanazawa (2025), PRL | Impact/capacity | rejected | Universality of δ=0.5 — fixes exponent in #107; no separate test |
| 110 | Bucci et al. (2019), MML (slow decay) | Impact decay | rejected | Alpha-ceiling insight (~1/3 of peak impact); no separate test |
| 111 | Donier & Bonart (2015), MML | Impact mechanism | rejected | Why fading uninformed flow pays; context only |
| 112 | Hendershott & Menkveld (2014), JFE | Price pressure | rejected | 0.92d half-life independently validates 8–14h holds; context |
| 113 | Frazzini, Israel & Moskowitz (2018), SSRN | Trading costs | pending | Calibrate whether 0.1–0.25%/side is conservative at our size (sidebar) |
| 114 | Bouchaud, Matacz & Potters (2001), PRL | Leverage effect | rejected | Mechanism already harvested via vol-target overlay (#10) |
| 115 | Zumbach (2009), Quant. Fin. | Vol asymmetry | rejected | Long→short vol information direction; validates current overlay; no lever |
| 116 | Cont & Das (2023), arXiv | Rough vol critique | rejected | Reinforces #66 rejection |
| 117 | Wang, Xiao & Yu (2023), J. Econometrics | Vol forecasting | rejected | Moot per #60 (proxy defeats RV machinery) |
| 118 | Leung & Li (2015), IJTAF | OU optimal trading | pending | Idea 4 — level-based (not clock) exit; theory-only, we supply the OOS |
| 119 | Handa & Schwartz (1996), JF | Limit orders | rejected | Conceptual anchor; not simulatable from bars |
| 120 | Linnainmaa (2010), JF | Limit orders/adverse sel. | rejected | PROCESS RULE: no bar-Low limit-fill backtests, ever |
| 121 | Lorenz & Almgren (2011), AMF | Adaptive execution | rejected | AIM heuristic → split-entry variant inside Idea 2 |
| 122 | Cartea & Jaimungal (2016), MAFE | Execution w/ alpha | pending | Idea 2 design — schedule entry vs expected post-signal drift |
| 123 | Bogousslavsky (2016), JF | Intraday autocorr | rejected | Mechanism support for #45/#124; no separate test |
| 124 | Bogousslavsky (2021), JFE | Intraday/overnight | pending | Idea 2 — decompose D P&L into intraday vs overnight legs |
| 125 | Bogousslavsky & Muravyev (2023), JFM | Closing auction | pending | Idea 2 core — 8.1bp close deviation, ~85% overnight reversion |
| 126 | Zawadowski, Andor & Kertész (2006), QF | Post-extreme spreads | pending | Adopt 2x-cost stress on entries right after >2σ bars (Idea 2 + C) |
| 127 | Berkowitz, Logue & Noser (1988), JF | VWAP benchmark | rejected | (O+H+L+C)/4≈VWAP is UNVERIFIED; sensitivity band only |
| 128 | Guéant & Royer (2014), SIAM JFM | VWAP execution | rejected | Needs sub-hourly volume curves |
| 129 | Herbster & Warmuth (1998), Mach. Learn. | Online learning | rejected | Fixed-Share converges to uniform avg when experts near-equal — supports Idea 1 |
| 130 | Garivier & Moulines (2011), ALT | Bandits | rejected | Bandit feedback wasteful in full-information setting; branch closed |
| 131 | Singer (1997), IJNS | Switching portfolios | rejected | Experts=assets not rule-variants; weak transfer |
| 132 | Li & Hoi (2014), ACM CSUR | Online PS survey | rejected | Post-cost degradation across surveyed methods; confirms #51 kill |
| 133 | Liu & Cartlidge (2023), EMSS | Bandit param adaptation | rejected | Simulation-only; no real-data post-cost evidence exists in this line |
| 134 | Zakamulin (2014), J. Asset Mgmt | Adaptive lookbacks | rejected | Direct OOS failure of adaptive lookback selection w/ frictions — do not adapt 104h MA |
| 135 | Goulding, Harvey & Mazzoleni (2024), FAJ "Breaking Bad Trends" | Adaptive speed | rejected | Monthly/century-scale only; static blend ≈ dynamic (see corrected #43 note: Garg not on published versions) |
| 136 | Joubert (2022), JFDS + companions | Meta-labeling | pending | Idea 5 — exposure filter on Book C; pre-registered single spec, n≈127 caution |
| 137 | Bailey, Borwein, López de Prado & Zhu (2017), JCF | PBO/methodology | rejected | Online selection = sequential snooping; process prior favoring ensembles |

**Registry corrections to existing rows:** #43 — published JFE author list is Goulding, Harvey & Mazzoleni (Garg only on early WP drafts); add note that static blend captures ~all of dynamic gain (DYN 0.52 vs MED 0.51). #45 — entry-side timing remains open (Idea 2) even though hold-grid was tested.

---

*Prepared 2026-08-15/16. Search: 5 parallel verified sweeps (~65 papers screened, 53 retained above). Adversarial verification: 14 load-bearing claims checked — 11 confirmed, 3 corrected (Hoffstein 1/N applies to std-dev measure; Zakamulin benchmark is buy-and-hold; Garg authorship), 0 refuted.*
