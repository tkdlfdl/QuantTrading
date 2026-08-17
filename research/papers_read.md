# Paper Registry — read tracking & verdicts

**Rules:**
1. Every paper reviewed for this project gets exactly one row. **Check this
   file before reading any paper.** Re-reading a listed paper is allowed but
   LOW priority — unread new papers come first; re-read only with a reason
   (new synthesis angle, changed data, evidence contradicting the verdict).
   On re-read, update the row (new date + revised verdict) instead of adding
   a duplicate.
2. `Verdict` values:
   - **IMPROVED** ⭐ — implemented here and beat the relevant baseline (higher
     Sharpe or return with comparable MaxDD). The implementation and recorded
     series are named in Notes.
   - `tested-no-gain` — implemented, did not improve.
   - `rejected` — read, not implementable or contradicted by evidence.
   - `untestable` — needs data we don't have (fundamentals, options, futures...).
   - `pending` — read and specced, implementation not yet run.
3. Baselines for "improvement" (2026-08-15 honest retest): Book D 2.740/-6.4%,
   Book F 1.548/-39.8%, Book A 1.301/-65.4%, FixedEW portfolio 2.060/-19.0%,
   MomAlloc 1.970/-24.5%. Comparable MaxDD = not more than ~1.2x worse.

| # | Paper (Author, Year, Journal) | Family | Read | Verdict | Notes |
|---|-------------------------------|--------|------|---------|-------|
| 1 | Antweiler & Frank (2004), JF | Sentiment | 2026-08-15 | rejected | Message-board sentiment: effect real but economically small vs costs (see reddit-sentiment-ma brief) |
| 2 | Tetlock (2007), JF | Sentiment | 2026-08-15 | rejected | Media pessimism → short-horizon reversal; contrarian sign contradicts naive long-hype |
| 3 | Da, Engelberg & Gao (2015), RFS | Sentiment | 2026-08-15 | rejected | FEARS index: 1-2 day reversal only; sub-cost magnitude at our frequency |
| 4 | Heston & Sinha (2017), FAJ | Sentiment | 2026-08-15 | tested-no-gain | Weekly-smoothed sentiment horizon extension — motivated the MA variants (best 1.13, still beta not alpha) |
| 5 | Bradley et al. (2024), RFS | Sentiment/Reddit | 2026-08-15 | rejected | WSB Due-Diligence predictability eliminated post-GameStop; kills naive Reddit strategies |
| 6 | Reichenbach & Walther (2023) | Sentiment/Reddit | 2026-08-15 | rejected | Reddit portfolios: negative 1yr alpha |
| 7 | Kim & Kim (2014) | Sentiment | 2026-08-15 | rejected | Null on 32M messages; sentiment follows prices — confirmed by our 0.87 variant correlations |
| 8 | Lachanski & Pav (2017) | Sentiment | 2026-08-15 | rejected | Bollen Twitter-mood is data snooping; fund closed <1yr |
| 9 | Bollen, Mao & Zeng (2011) | Sentiment | 2026-08-15 | rejected | Famous Twitter-mood result; failed replication (see #8) |
| 10 | Moreira & Muir (2017), JF | Construction/vol-managed | 2026-08-15 | **IMPROVED** ⭐ | Vol-target overlay on portfolio: `portfolio_ivol_voltgt_nob` Sharpe **2.501/-8.9%** vs EW 2.06/-19.0%; also `portfolio_ew_voltgt_*` |
| 11 | Harvey, Hoyle, Korgaonkar et al. (2018), JPM | Construction/vol-targeting | 2026-08-15 | **IMPROVED** ⭐ | Same overlay evidence — vol targeting works on equity-like risk assets; confirmed on our books |
| 12 | Barroso & Santa-Clara (2015), JFE | Momentum crash fix | 2026-08-15 | **IMPROVED** ⭐ | Risk-managed momentum on Book F: `book_f_riskmanaged` Sharpe 1.609/-10.3% vs 1.548/-39.8% (CAGR falls 76%→18%; best used at portfolio level) |
| 13 | Qian (2005), PanAgora wp | Construction/risk parity | 2026-08-15 | tested-no-gain | Plain inverse-vol: CAGR ↑ (59-75%) but MaxDD -28% breaches bar; needs the vol overlay (see #10) to shine |
| 14 | Maillard, Roncalli & Teiletche (2010) | Construction/ERC | 2026-08-15 | tested-no-gain | ERC: MaxDD -13.7% best-in-class but Sharpe 1.57 ≪ EW 2.06 — over-allocates to low-vol low-return books |
| 15 | López de Prado (2016), JPM | Construction/HRP | 2026-08-15 | tested-no-gain | HRP: 1.82-1.94 Sharpe, good DD, but loses to plain EW on 5 assets — clustering adds nothing at N=5 |
| 16 | DeMiguel, Garlappi & Uppal (2009), RFS | Construction/1-N | 2026-08-15 | tested-no-gain | The 1/N null result — validates our Fixed-EW baseline; optimized weights indeed failed to beat it here (ERC, HRP, sharpe_w all lost) |
| 17 | Kelly (1956); MacLean, Thorp & Ziemba (2011) | Construction/Kelly | 2026-08-15 | **IMPROVED** ⭐ | Quarter-Kelly + shrinkage, no-B: `portfolio_qkelly_nob` Sharpe 2.098/-19.8% vs 2.013/-21.6% — marginal; dominated by #10 |
| 18 | Ledoit & Wolf (2004), JPM/JMA | Construction/shrinkage | 2026-08-15 | tested-no-gain | Covariance shrinkage used inside quarter-Kelly; enabler, not standalone improver |
| 19 | Markowitz (1952), JF | Construction/mean-variance | 2026-08-15 | rejected | Foundational; raw MV needs ~3,000 months of data for N=25 (per #16) — unusable at our N and history |
| 20 | Clarke, de Silva & Thorley (2006), JPM | Construction/min-var | 2026-08-15 | rejected | Min-var is return-blind; ERC test (#14) shows the failure mode on our books |
| 21 | Choueifaty & Coignard (2008), JPM | Construction/max-div | 2026-08-15 | rejected | Max diversification return-blind; survey-rejected for our setup |
| 22 | Black & Litterman (1992), FAJ | Construction/Bayesian | 2026-08-15 | untestable | Needs subjective views + market-cap prior; no natural prior for strategy books |
| 23 | Asness, Frazzini & Pedersen (2012), FAJ | Construction/risk parity | 2026-08-15 | tested-no-gain | Leverage-aversion case for risk parity; we run unlevered so the mechanism (levering low-vol) is unavailable |
| 24 | Faber (2007), JWM | Construction/trend TAA | 2026-08-15 | tested-no-gain | Gate on A/F in champion: 2.566 ≈ 2.580 — 210d returns nearly always positive 2019-2026, gate rarely fires |
| 25 | Donohue & Yip (2003), JPM | Construction/rebalancing | 2026-08-15 | rejected | Rebalancing-frequency effects; our 21d choice is within their no-harm range |
| 26 | Jegadeesh & Titman (1993), JF | Momentum | 2026-08-15 | tested-no-gain | Foundational cross-sectional momentum — already embodied in Books A/F; no new edge beyond current params |
| 27 | Daniel & Moskowitz (2016), JFE | Momentum crashes | 2026-08-15 | **IMPROVED** ⭐ (marginal) | Panic gate on champion: `portfolio_champ_panic_nob` 2.514/-8.9% vs 2.501 — only 11 panic days in window, mechanism barely exercised; keep as insurance |
| 28 | Nagel (2012), RFS | Reversal/liquidity | 2026-08-15 | tested-no-gain | VIX-scaling Book D: 2.479 < champion 2.501 — D already self-loads in high-VIX periods via signal frequency |
| 29 | Moskowitz, Ooi & Pedersen (2012), JFE | TS momentum | 2026-08-15 | tested-no-gain | Same test as #24 (own-trend gate) — wash in champion |
| 30 | De Bondt & Thaler (1985), JF | LT reversal | 2026-08-15 | rejected | 3-5yr reversal horizon too slow for our stack; decayed post-publication (per #35 haircut) |
| 31 | George & Hwang (2004), JF | 52-wk-high momentum | 2026-08-15 | tested-no-gain | STRONG negative: ranking swap gives Sharpe -0.01 vs 1.548 — near-high names are the OPPOSITE of F's explosive-mover edge on this universe/period |
| 32 | Gatev, Goetzmann & Rouwenhorst (2006), RFS | Pairs trading | 2026-08-15 | rejected | Profits decayed to ~0 post-2002 in follow-ups; high infra cost |
| 33 | Lou, Polk & Skouras (2019), JFE | Overnight/intraday | 2026-08-16 | **IMPROVED** ⭐ (candidate G) | Cross-sectional overnight-share book RESURRECTED by appraisal-sized allocation (0.25 ivol shares ~7%): champion 2.788->2.852, corr 0.13, **27/27 perturbation cells positive** (23 with p<0.10; base p=0.098 among the weakest). Earlier full-slot bench was a SIZING artifact. Pending Phase-2 incubation decision |
| 34 | Medhat & Schmeling (2022), RFS | ST momentum/turnover | 2026-08-15 | tested-no-gain | 28yr test: 0.753/-56.9% vs control 0.695/-72.2% — condition effect +0.06 real but small; corr 0.55 to F; fails new-book bar. Caveat: volume-intensity proxy, not true share turnover |
| 35 | McLean & Pontiff (2016), JF | Meta/decay | 2026-08-15 | rejected | 26% in-sample→58% post-publication decay haircut — applied as prior to all candidates, not a strategy itself |
| 36 | Gervais, Kaniel & Mingelgrin (2001), JF | Volume premium | 2026-08-16 | tested-no-gain | Standalone 0.875 but corr 0.54 to D; portfolio-contribution 2.398 < 2.580 — volume shocks select the same oversold names D already buys |
| 37 | Amihud (2002), JFM | Illiquidity | 2026-08-15 | rejected | Illiquidity premium lives in small caps outside our S&P500/NDX universe |
| 38 | Heston & Sadka (2008), JFE | Seasonality | 2026-08-15 | rejected | Same-calendar-month momentum: weak post-publication, monthly horizon poor fit |
| 39 | Ariel (1987); Lakonishok & Smidt (1988), JF | Turn-of-month | 2026-08-15 | rejected | TOM effect too small after costs at our position sizes; decayed |
| 40 | Cederburg, O'Doherty, Wang & Yan (2020), JFE | Vol-managed critique | 2026-08-15 | tested-no-gain | Caveat on #10: vol-managed fails OOS for many factors — our direct comparison passed (2.501 vs 2.06), which is the test they demand |
| 41 | (original — verifier finding, Cycle 1) | Original/risk fix | 2026-08-15 | **IMPROVED** ⭐ | Book C per-ticker overlap cap: C Sharpe 0.592→0.675, MaxDD -53.7%→-33.0%; champion 2.514→2.580 at same DD. Origin class 4 (own-system observation, no paper) |
| 42 | Shu, Yu & Mulvey (2024), J. Asset Mgmt | Regime/jump models | 2026-08-15 | tested-no-gain | JM gate: champ 2.498 < 2.580 (DM gate). 300 bear days = too trigger-happy vs DM's 11; de-risks rallies |
| 43 | Garg, Goulding, Harvey & Mazzoleni (2023), JFE | Momentum turning points | 2026-08-15 | tested-no-gain | F standalone MaxDD -39.8→-33.5% at same Sharpe, but champion 2.576/-8.5 ≈ 2.580/-8.9 — ivol+VT already absorbs F's DD |
| 44 | Wood, Roberts & Zohren (2022), JFDS | Momentum/ML CPD | 2026-08-15 | tested-no-gain | Fed Idea 2 (see #43) — wash at champion level |
| 45 | Heston, Korajczyk & Sadka (2010), JF | Intraday seasonality | 2026-08-15 | tested-no-gain | D hold grid 7/14/21h all < locked 8h (2.74). Note: 14h/21h raise total return (+872%/+910% vs +640%) at ~2x DD — possible future high-return variant |
| 46 | Gao, Han, Li & Zhou (2018), JFE | Intraday momentum | 2026-08-15 | tested-no-gain | SPY hourly ingested (92.3% coverage): Sharpe -9.5 — last-hour edge < our 0.2% round-trip cost; effect dead at retail cost tier |
| 47 | Bun, Bouchaud & Potters (2017), Phys. Reports | Covariance/RMT | 2026-08-15 | tested-no-gain | Residual bubble for D (official engine, sane anchor): alpha=0.5 Sharpe 2.178, alpha=1.0 1.362 vs raw 2.740 — market-mode component IS part of D's edge |
| 48 | Guijarro-Ordonez, Pelger & Zanotti (2021+), arXiv/SSRN | ML/statarb | 2026-08-15 | tested-no-gain | See #47 — residualization degrades D; oversold-with-the-market bounces harder than idiosyncratically-oversold |
| 49 | Ledoit & Wolf (2020), Ann. Statist. | Covariance/NLS | 2026-08-15 | untestable | Nonlinear shrinkage — shelf tool for future large-N book; see #50 caution |
| 50 | Bongiorno & Challet (2021/2023), arXiv | Covariance | 2026-08-15 | rejected | NLS suboptimal under non-stationarity; Average Oracle; author order on 2309.17219 to double-check |
| 51 | Mhammedi & Rakhlin (2022), COLT | Online learning | 2026-08-15 | tested-no-gain | EG+VT 2.151/-13.1% — respectable with zero tuning but loses to ivol champion 2.580/-8.9% |
| 52 | Lim, Zohren & Roberts (2019), JFDS | Momentum/ML | 2026-08-15 | rejected | Deep momentum networks need 2-3bp costs, ours 10-25bp; keep Sharpe-loss framing |
| 53 | Gu, Kelly & Xiu (2020), RFS | Cross-section/ML | 2026-08-15 | pending | Price-feature-only GBT as possible Book G — quarter-scale effort, deferred |
| 54 | Ryan (2026), arXiv:2608.01494 | Sizing/conformal | 2026-08-15 | rejected | Conformal Kelly failed its own pre-registered OOS; conformal-width throttle = open question |
| 55 | Wisniewski, Lindsay & Lindsay (2020), COPA | Conformal | 2026-08-15 | rejected | FX market-maker inventory domain — not transferable |
| 56 | Scheffer et al. (2009), Nature | Econophysics/EWS | 2026-08-15 | rejected | Context only for early-warning thread — killed by #57 |
| 57 | Guttal et al. (2016), PLOS ONE | Econophysics/EWS | 2026-08-15 | rejected | Valuable negative: no critical-slowing-down before financial crashes — kills CSD drawdown pre-alarm idea |
| 58 | Sornette LPPLS corpus (Bree-Joseph 2013; Shu 2024) | Econophysics/bubbles | 2026-08-15 | rejected | Fit fragility, mixed OOS — no LPPLS timing overlay |
| 59 | HMM regime literature (incl. arXiv:2406.09578) | Regime | 2026-08-15 | rejected | Superseded by jump models (#42) for our use |
| 60 | Corsi (2009), J. Fin. Econometrics | Vol forecasting/HAR | 2026-08-16 | tested-no-gain (revised) | HAR-X +0.052 REFUTED by significance test (p=0.258, LW bootstrap) — was noise, ⭐ retracted. The proxy diagnosis stands but no promotable gain |
| 61 | Andersen, Bollerslev, Diebold & Labys (2003), Econometrica | Vol forecasting/RV | 2026-08-15 | tested-no-gain | See #60 — daily proxy defeats the RV framework |
| 62 | Bollerslev, Patton & Quaedvlieg (2016), J. Econometrics | Vol forecasting/HARQ | 2026-08-15 | rejected | Moot given #60 outcome with our proxy |
| 63 | Patton & Sheppard (2015), REStat | Vol forecasting/semivariance | 2026-08-15 | rejected | Moot given #60 outcome with our proxy |
| 64 | Corsi & Renò (2012), JBES | Vol forecasting/leverage | 2026-08-15 | rejected | Moot given #60 outcome with our proxy |
| 65 | Yang & Zhang (2000), J. Business | Vol estimation/range | 2026-08-15 | untestable | Portfolio return series has no OHLC; would need book-level reconstruction |
| 66 | Gatheral, Jaisson & Rosenbaum (2018), Quant. Fin. | Vol/rough | 2026-08-15 | rejected | Rough vol validates HAR at our horizons; RFSV effort not justified |
| 67 | Ghashghaie et al. (1996), Nature | Econophysics/cascade | 2026-08-15 | rejected | Conceptual ancestor of HAR only |
| 68 | Preis, Kenett, Stanley, Helbing & Ben-Jacob (2012), Sci. Rep. | Correlation dynamics | 2026-08-16 | tested-no-gain | Corr-spike gate 2.532 < 2.580 — DM panic gate already covers fragile periods |
| 69 | Onnela et al. (2003), Phys. Rev. E | Correlation/MST | 2026-08-15 | rejected | Corroborates #68; MST machinery unnecessary |
| 70 | Munnix et al. (2012), Sci. Rep. | Correlation states | 2026-08-15 | rejected | Heavier cousin of #68 scalars; deferred |
| 71 | Kritzman, Li, Page & Rigobon (2011), JPM | Absorption ratio | 2026-08-16 | tested-no-gain | See #68 |
| 72 | Curme et al. (2015), Quant. Fin. | Lead-lag networks | 2026-08-15 | rejected | Needs <=15-min sampling; decayed post-2011 |
| 73 | Huth & Abergel (2014), J. Emp. Fin. | Lead-lag/HF | 2026-08-15 | rejected | Seconds-scale, tick data required |
| 74 | Chekhlov, Uryasev & Zabarankin (2005), IJTAF | OR/CDaR | 2026-08-16 | tested-no-gain | CDaR LP nu=5%: 2.476/-10.4%; nu=8%: 2.279 — loses to ivol+VT champion; 5th optimizer to fail vs simple weights (DeMiguel confirmed again) |
| 75 | Goldberg & Mahmoud (2017), Math. Fin. Econ. | OR/drawdown theory | 2026-08-16 | tested-no-gain | See #74 |
| 76 | Hamilton (1989), Econometrica | Regime switching | 2026-08-15 | rejected | Regime-gate slot resolved (#42, #27); do not rebuild |
| 77 | Bates & Granger (1969), ORQ | Forecast combination | 2026-08-15 | tested-no-gain | Committee 2.559 < anchor 2.580 — dragged by weak HAR member; EWMA-only wash (+0.017) |
| 78 | Stivers & Sun (2010), JFQA | Dispersion/momentum | 2026-08-16 | tested-no-gain | Pre-test 1997-2018 on Book A: t=-0.73, no effect — killed in-sample, OOS window preserved |
| 79 | Etula, Rinne, Suominen & Vaittinen (2020), RFS | Calendar/dash-for-cash | 2026-08-16 | tested-no-gain | TOM 1997-2026: Sharpe 0.255 — effect too weak after costs; portfolio 2.248 |
| 80 | Bollerslev, Tauchen & Zhou (2009), RFS | Variance risk premium | 2026-08-16 | tested-no-gain | VRP timing 0.347 < matched-beta 0.505 — fails the beta test outright |
| 81 | Greenwood & Sammon (2025), JF | Index reconstitution | 2026-08-16 | rejected | Inclusion effect gone in modern data; we also lack membership history |
| 82 | Baker-Haugen low-vol corpus (CFA 2024 review) | Low-vol anomaly | 2026-08-16 | rejected | Beta channel + 2015-2024 underperformance; parked as bench idea |
| 83 | (original — Cycle 9 synthesis) | Original/index timing | 2026-08-16 | tested-no-gain (BENCHED) | Breadth timing: standalone 0.896/corr≤0.38 PASSED, but portfolio-contribution test FAILED: champ+H 2.504 < 2.580 (ivol over-allocates to its 67%-cash profile; 12.5% CAGR dilutes). Bench note: conditional/satellite funding on signal days = possible future variant |
| 84 | (original — Cycle 12, registry #45 note) | Original/D sleeve | 2026-08-16 | **IMPROVED** ⭐⭐ | D14 sleeve as 5th book: champion 2.580 -> **2.781 / 43.9% / -8.2%**; robustness 11/11; verifier PASS. QUALIFIES FOR FULL PROMOTION (pending human sign-off for live adoption). H cash-sleeve near-miss (2.628, +0.048) also from this cycle |
| 85 | Ledoit & Wolf (2008), J. Empirical Finance | Methodology/testing | 2026-08-16 | **IMPROVED** ⭐ (process) | Studentized bootstrap Sharpe-delta test ADOPTED AS PROMOTION GATE (tools/significance.py, p<0.10). First use: D14 vindicated (+0.201, p=0.005); HAR-X refuted (+0.052, p=0.258). Giacomini-White 2006, Hansen-Lunde-Nason MCS logged for multi-variant sweeps |
| 86 | Hoffstein, Sibears & Faber-adjacent RTL (2019), JII | Timing luck/cohorts | 2026-08-16 | tested-no-gain (D21) | Rebalance-timing-luck 1/N math EXPLAINS our D14 win post-hoc. D21 third sleeve: +0.061, p=0.113 — marginal contribution decayed below gate. 2 sleeves suffice |
| 87 | Hendershott & Menkveld (2014), JFE | Price pressure | 2026-08-16 | rejected (context ⭐) | Transitory pressure 0.49%, half-life 0.92d (~6.5h) — quantitatively rationalizes D's 8h/14h holds and predicts D21 decay (confirmed). Best mechanism paper for D |
| 88 | So & Wang (2014), JFE | Reversal/earnings | 2026-08-16 | rejected (reassurance) | Reversal ~6x LARGER around announcements — kills the earnings-exclusion worry for D |
| 89 | Chordia et al. (2014); Jacobs & Müller (2020) | Reversal decay | 2026-08-16 | rejected (context) | Decay applies to daily/weekly reversal, not our hourly horizon |
| 90 | Kurth, Eisler, Rej & Bouchaud (2026), arXiv/CFM | Short-horizon landscape | 2026-08-16 | rejected (context) | Landscape shifting TOWARD mean reversion — strategic tailwind for D |
| 91 | Hameed & Mian (2015), JFQA | Sector-relative reversal | 2026-08-16 | pending | Within-industry reversal strongest in large caps — queue: D ranking tie-break (keep raw -0.8 gate per #47 lesson) |
| 92 | Baltussen, Da & Soebhag (2024) | EOD pressure | 2026-08-16 | pending | End-of-day pressure entry tilt for D — queue |
| 93 | Barardehi et al. (2026), RFS | Path composition | 2026-08-16 | pending | Overnight/intraday/jump split of oversold moves — DIAGNOSTIC first (sign contested) |
| 94 | Dai, Medhat, Novy-Marx & Rizova (2024), FAJ | Reversal speed | 2026-08-16 | pending | Vol/turnover-matched sleeve assignment (fast reversals->D8, slow->D14) — queue |
| 95 | Wang & Yan (2021), JBF | Semivariance targeting | 2026-08-16 | tested-no-gain | Downside targeting: CAGR 52.5% but Sharpe -0.14 and DD -10.2% breaches bar; documented as aggressive-profile variant |
| — | (finance_sweep2 brief) | — | 2026-08-16 | — | Full ~30-paper table incl. remaining verdicts: research/briefs/finance_sweep2_2026-08.md — merge into registry as items get tested |
| 96 | Bogousslavsky & Muravyev (2023), J. Fin. Markets | Execution/close auction | 2026-08-16 | pending | Verified 8.1bp close-auction deviation, ~85% overnight reversion — D entry-execution A/B (queue #22, upgraded spec) |
| 97 | Blitz et al. (2023), FAJ | Reversal bands | 2026-08-16 | pending | Net short-horizon alpha survives ONLY with asymmetric entry/exit bands — queue #25 (aggressive entry, lazy exit) |
| 98 | Joubert (2022), JFDS | Meta-labeling | 2026-08-16 | pending | Ridge-logistic exposure filter on Book C — pre-registered single spec only (queue #26) |
| 99 | Toth et al. (2011), PRX; Bucci et al. (2019), PRL | Impact/capacity | 2026-08-16 | rejected (context ⭐) | Square-root impact independently validates D's 8-14h holds and bounds per-trade alpha — capacity audit sidebar |
| 100 | Zakamulin OOS corpus; PBO literature | Meta/adaptation | 2026-08-16 | rejected (decisive) | AVERAGING BEATS ADAPTATION at our frequency — decades of data needed to distinguish hold=8 vs 14; kills adaptive-parameter threads permanently |
| — | (cs_hardsci_sweep2 brief) | — | 2026-08-16 | — | Full ~53-paper table: research/briefs/cs_hardsci_sweep2_2026-08.md; merge rows as items get tested |
| 101 | (structural finding — D blend consolidation) | Original/allocation | 2026-08-16 | **IMPROVED** ⭐ (structure) | Half the D14 gain was the ALLOCATION channel (2 ivol slots -> 74% D-complex weight). Naive 1-slot blend reverts it (-0.199, p=0.004); blend + ALLOC_SHARES["D"]=2 is identical to separate books (corr 0.9996, Sharpe 2.788). Live Book D = single 2-sleeve blend since 2026-08-16 |
| 102 | Moskowitz-Ooi-Pedersen TSMOM / Faber trend (standalone SPY probe) | Trend/crisis alpha | 2026-08-16 | tested-no-gain | Standalone monthly SPY trend (L/F 0.72, TSMOM L/S 0.45): stress-day P&L -0.86%/d on champ-worst days — MONTHLY TREND TOO SLOW for our daily-horizon stress; portfolio delta -0.18 (p=0.09). Sharpens orthogonal spec: need FAST convexity |
| 103 | Israelov & Nielsen (2015), JPM | Convexity timing | 2026-08-16 | pending | "Calm != cheap" — convexity buying needs an ignition condition, not just low VIX; gates V2 spec |
| 104 | Cheng (2019); Dew-Becker et al. | VRP dynamics | 2026-08-16 | pending | Variance premium FALLS as risk rises (convexity gets cheap exactly when needed); short-end structurally priciest — carry timeable not fixable. Feeds V2/V3 |
| 105 | Hanauer & Windmüller (2023), JBF | Residual momentum | 2026-08-16 | pending | Residual momentum ~half the crash risk of raw (corrected attribution from Blitz 2011) — queue: F re-rank (V4) |
| 106 | Frazzini-Pedersen BAB spread (dollar-neutral) | Beta-neutral | 2026-08-16 | rejected | 8%/yr borrow ~= entire large-cap BAB premium (Novy-Marx-Velikov); BAB crashes on funding stress = correlated with champion. Dead |
| 107 | Gayed & Bilello (Dow Award) | Defensive rotation | 2026-08-16 | pending | Utilities/SPY rotation (V5) — practitioner tier, hard corr gates required |
| — | (orthogonal_ideation brief) | — | 2026-08-16 | — | Full 11-candidate table: research/briefs/orthogonal_ideation_2026-08.md. NOTE: brief's V1 (standalone SPY trend) largely KILLED in parallel by probe #102 (stress-day P&L -0.86%/d); only the crisis-only short/flat sleeve variant remains untested |
| 108 | (Cycle 14 program — orthogonal candidates #28-32) | Orthogonality | 2026-08-16 | tested-no-gain (structural) | All 5 fail: convexity book pays +0.88%/d on stress days but carry exceeds benefit (complacent/low-VRP entries sit in the -1.08%/d VIX<15 bleed bucket per carry study); resid-F 0.42 (beta is part of F's edge too — #47 generalizes); crisis-short +0.019 p=0.29; XLU -0.235 (beta in disguise). STRUCTURAL CONCLUSION: with price-only data, paid convexity cannot beat carry at daily stress horizon — the de-risk-only vol-target overlay IS the crisis hedge, at zero premium. Champion design locally complete |
| 109 | (methodology — appraisal-ratio sizing) | Allocation/testing | 2026-08-16 | **IMPROVED** ⭐ (process) | Less-correlated books need Sharpe > corr x champ_Sharpe (Treynor-Black) and must be tested at APPRAISAL-SIZED fractional shares, never full ivol slots. Full-slot testing wrongly benched G (overnight-share). Contribution test amended |
| 110 | Faber (2007) GTAA / Antonacci dual momentum | Cross-asset momentum | 2026-08-17 | **IMPROVED** ⭐ (candidate X) | First non-single-stock book: 15 ETFs, 12-1 top-3 ungated monthly. Positive 2008/2020/2022; corr 0.22; champ+X @0.25 shares +0.115 (p=0.034), **12/12 cells positive** (ungated uniformly beats gated — champion vol-target already de-risks). v1 VOIDED (position-ffill bug). Deep grid 2026-08-17: lb x hold 20/20 positive, 15/20 p<0.10, canonical momentum term structure (63d noise -> 252d peak +0.13 -> 378d rolloff); spec 252/monthly = plateau center, unchanged. Cumulative 32/32 cells. INCUBATING alongside G |
| 111 | (Cycle 15 battery — 4 new-book candidates) | Mixed | 2026-08-17 | mixed | Weekly cross-asset reversal DEAD (0.01 — no asset-class reversal at 5d); credit-regime rotation no-gain (+0.021 p=0.35); sector rotation REDUNDANT vs Book X (corr 0.58, marginal +0.003 p=0.47 — same factor, narrower universe; bench prevents double-funding); D-uptrend sleeve = the find (see #112) |
| 112 | (original — Cycle 15 synthesis) | Original/D sleeve | 2026-08-17 | **IMPROVED** ⭐ (candidate) | D x momentum double-sort ("dips in uptrends", 126d filter): standalone 2.57/-6.8%, corr 0.48; as extra 0.25-share book champ +0.046 at **p=0.010** (strongest p in program); robustness 3/3, monotone in filter length (189d p=0.004). Third D-family sleeve candidate — pending incubation/adoption decision (D-complex concentration consideration) |
| 113 | (structural finding — dual cost basis) | Methodology | 2026-08-17 | **flagged** | Live settle engine (0.25%/side, user setting) vs research engines (0.1%, CLAUDE.md standard): settle-D 1.61 vs research-D 2.74 — cost drag dominates for high-turnover sleeves. All session A/Bs were research-basis both sides (relative verdicts VALID); live absolute expectations + rollback bounds must use settle-basis baselines. PENDING: recalibrate IvolVT rollback bound from settle-basis champion; consider whether 0.25% live assumption is right for D-class turnover |
| 114 | (Cycle 16 — four structural candidates) | Mixed | 2026-08-17 | tested-no-gain (all four, informative) | (a) F-tranching WASH: RTL cures idiosyncratic timing luck (D14's per-stock exits) not regime risk (F cohorts pick identical names); (b) D threshold barbell SIGNIFICANTLY negative (-0.060 p=0.041) — the 0.8 threshold is load-bearing, shallower dips have less edge; (c) asset-class capitulation dead (0.10) — no single-stock-style capitulation at ETF level; (d) sector mean reversion 18/18 vs champion BUT 0.000 marginal on full incubator stack — seat taken by D-complex+incubators |
| 115 | (Cycle 17 — new data + 3 strategies) | Mixed | 2026-08-17 | tested-no-gain (all 3) | NEW DATA CACHED: yields 1995+, 29-asset ETF panel incl. 8 countries, crypto 2014+. Expanded-X redundant vs X (corr 0.68, full-stack +0.016, swap negative); crypto trend 1.10 standalone/corr 0.08 but adds nothing at prudent sizing (stress-day -0.91%/d — crypto crashes with risk-off); curve tilt dead (-0.39, full-stack -0.111 p=0.046). Full-stack saturation holds at ~2.92 projected |
| 116 | Harvey, Mazzoleni & Melone (NBER w33554) + Kayacetin (2026) | Rebalancing flows/TOM | 2026-08-17 | tested-no-gain | Long-only adaptation FAILED (-0.22 standalone; full-stack -0.190 p=0.017). CAVEAT: our spec dropped the short leg + simplified the flow proxy — paper's exact construction unproven here, not disproven. Retest only with faithful L/S spec (shorts = human-gated) |
| 117 | Pre-FOMC drift literature (2024 OOS refresh) | Event/macro | 2026-08-17 | pending | Alive through 12/2024; 8-16 days/yr, ~0.5-0.6 standalone, uncorrelated — queue #37 (needs FOMC calendar) |
| 118 | Gulen & Woeppel (2026), JFQA | Price-path convexity | 2026-08-17 | tested-no-gain | Our spec: standalone 0.96, passes bare-champ (p=0.099) but full stack absorbs it (+0.023 p=0.295, corr 0.31 to D-core) — 5th candidate to die exactly this way. Spec caveat: simplified convexity measure |
| 119 | Stosik & Zaremba (2026) + Hameed-Mian construction | Reversal defense | 2026-08-17 | **IMPROVED** ⭐ | Industry-adjusted D8 (sector-rel ranking, raw gate): standalone 2.808 vs 2.740; champion 2.800 (+0.019, p=0.038); perturbation 4/6 (fails only ma=84). ALL-IN with incubators: **2.930/43.4%/-7.7%** (portfolio_allin_v2, p=0.016). Live replay_D wiring = open item; remaining #39 upgrades (gap/MAX filters) pending |
| 120 | Goyal, Jegadeesh & Wu + EOD-reversal corpus | Entry timing | 2026-08-17 | pending | EOD reversal dead standalone (3.8-6.9bps gross, corrected from circulating ~24bps) but FREE as D closing-auction entry tilt — queue #40 (merges old #22) |
| 121 | (recent-lit sweep traps) | Dead ends 2024-2026 | 2026-08-17 | rejected | FOMC even-week cycle, index-inclusion drift, LETF EOD momentum, retail-flow prediction, sector-overnight, country-ETF ML — all confirmed dead in recent data |
| 122 | (D-defense battery conclusion: MAX + gap filters) | Reversal filters | 2026-08-17 | tested-no-gain | MAX/lottery filter wash (+0.003 — gate+top20 already handles); gap filter SIGNIFICANTLY HARMFUL (-0.138 p=0.002 — overnight-gap drops rebound fine). 4th confirmation: D's raw signal is nearly complete; only sector-relative ranking (#119) improves it |
