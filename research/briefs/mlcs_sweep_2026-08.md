# ML/CS + Hard-Science Idea-Generation Sweep — August 2026

**Scope:** Monthly sweep over ML/CS venues (NeurIPS/ICML/PMLR, arXiv q-fin/cs.LG) and hard-science/econometrics buckets (Nature, PLOS ONE, Physics Reports, Annals of Statistics, JF/JFE/RFS, J. Asset Management).
**Constraint set:** US-equity OHLCV only (daily 1997+, hourly 2019+, ~515 names + SPY/UVXY/^VIX), Reddit sentiment 2019–2024ish. No fundamentals/options/order-book.
**Frontier to beat:** Champion portfolio Sharpe 2.514, CAGR 45.3%, MaxDD -8.9% (2019–2026). Improvement = higher Sharpe/CAGR with MaxDD ≥ -11%.
**Integrity baseline for all specs below:** signals shifted ≥1 bar; rolling statistics only; next-bar execution; chronological splits with embargo ≥ 1 holding period; point-in-time universe; McLean–Pontiff ~50% haircut on published effects; synthesis/original ideas get NO expectation credit until tested.

All citations below were verified via web search on 2026-08-15. None fabricated; verification notes flagged inline where relevant.

---

## 1. Papers reviewed (with one-line verdicts)

### Regime detection / changepoints
1. **Shu, Y., Yu, C., Mulvey, J.M. (2024). "Downside Risk Reduction Using Regime-Switching Signals: A Statistical Jump Model Approach." *Journal of Asset Management* (DOI 10.1057/s41260-024-00376-x; arXiv:2402.05272).** — Verified (arXiv + Springer). Jump model with transition penalty beats HMM and buy-hold OOS on US/DE/JP indices 1990–2023 for MaxDD and Sharpe; return-series features only. **VERDICT: strongest actionable paper of the sweep — direct fit for our panic gate.**
2. **Wood, K., Roberts, S., Zohren, S. (2022). "Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection." *Journal of Financial Data Science* 4(1):111–129 (arXiv:2105.13727).** — Verified (arXiv, ORA, JFDS, public GitHub). Online Gaussian-Process CPD severity score fed to a deep momentum network; +33% Sharpe 1995–2020, +66% in 2015–2020. **VERDICT: keep the CPD-severity mechanism, drop the LSTM — usable to de-risk Book F at turning points.**
3. **Garg, A., Goulding, C.L., Harvey, C.R., Mazzoleni, M.G. (2023). "Momentum Turning Points." *Journal of Financial Economics*.** — Verified (JFE/ScienceDirect, SSRN 3489539). Slow×fast momentum signal agreement defines 4 states; disagreement states (turning points) have predictably poor momentum returns; dynamic speed selection beats any static speed. **VERDICT: cheap, no-ML state machine that maps directly onto Book F's known weakness.**
4. **HMM regime-detection literature (various, e.g., arXiv:2406.09578 "Dynamic Asset Allocation with Asset-Specific Regime Forecasts").** — Verified. Plain HMMs suffer unstable state sequences and false alarms at daily frequency; jump models dominate in head-to-heads. **VERDICT: skip raw HMM; use jump model.**

### Position sizing / allocation (online learning, conformal)
5. **Ryan, R.J. (2026). "Conformal Kelly: Conformal Prediction Intervals as the Scale in Fractional Kelly Position Sizing." arXiv:2608.01494.** — Verified via arXiv abstract fetch. Conformal interval width scales fractional-Kelly S&P exposure; dev-window (2016–21) Sharpe 1.34, but **pre-registered OOS (2022+) underperformed passive** (8.5%/7.0% ann.). Calibration held (0.745 vs 0.750 target). **VERDICT: honest negative result — Kelly-scaling via conformal width fails OOS, but calibrated interval-width as a *risk throttle* (not return forecaster) remains untested and cheap.**
6. **Wisniewski, W., Lindsay, D., Lindsay, S. (2020). "Application of conformal prediction interval estimations to market makers' net positions." *Proceedings of COPA/PMLR v128*.** — Verified. FX market-maker inventory forecasting; wrong domain. **VERDICT: dead end for us; only useful as conformal-in-finance precedent.**
7. **Mhammedi, Z., Rakhlin, A. (2022). "Damped Online Newton Step for Portfolio Selection." *PMLR v178 (COLT)*.** — Verified. Logarithmic-regret OPS at near-linear cost; lineage: Cover (1991) universal portfolios, Zinkevich OGD O(√T), Hazan et al. ONS. **VERDICT: theory solid; regret is measured in log-wealth vs best *constant-rebalanced* portfolio — modest practical edge over inverse-vol for 4–6 books, worth a cheap A/B as allocator variant.**

### Covariance / RMT (econophysics)
8. **Bun, J., Bouchaud, J.-P., Potters, M. (2017). "Cleaning large correlation matrices: tools from Random Matrix Theory." *Physics Reports* 666:1–109.** — Verified. Rotationally-invariant estimator (RIE) cleans noise-dominated eigenvalues; improves out-of-sample Markowitz risk. **VERDICT: not needed for 4-book allocation (N too small), but the top-eigenmode (market mode) machinery enables residualizing our 515-name hourly panel.**
9. **Ledoit, O., Wolf, M. (2020). "Analytical nonlinear shrinkage of large-dimensional covariance matrices." *Annals of Statistics* 48(5).** — Verified (Project Euclid). ~1000× faster than QuEST, handles p>n. **VERDICT: the practical tool if we ever run Markowitz over the 515-name panel; registry-worthy, no immediate book.**
10. **Bongiorno, C., Challet, D. (2021ff). "Non-linear shrinkage of the price return covariance matrix is far from optimal for portfolio optimisation" (arXiv:2112.07521) + "Covariance matrix filtering: the Average Oracle vs NLS/DCC-NLS" (arXiv:2309.17219).** — Verified. Financial returns are non-stationary; simple "average oracle" eigenvalue targets beat NLS in realistic settings. **VERDICT: caution flag on 9 — favor simple exponential-weighted + RMT cleaning over fancy shrinkage.**

### Early-warning signals (hard science)
11. **Scheffer, M., et al. (2009). "Early-warning signals for critical transitions." *Nature* 461:53–59.** — Verified. Canonical CSD paper: rising lag-1 autocorrelation + variance before fold bifurcations. **VERDICT: context only — see 12 for the financial-market test.**
12. **Guttal, V., Raghavendra, S., Goel, N., Hoarau, Q. (2016). "Lack of Critical Slowing Down Suggests that Financial Meltdowns Are Not Critical Transitions, yet Rising Variability Could Signal Systemic Risk." *PLOS ONE* 11(1):e0144198.** — Verified. No CSD (autocorrelation) signal before historical crashes; rising *variability* does precede stress. **VERDICT: dead end for autocorrelation-based crash timing; our vol-target overlay already harvests the surviving signal (variance). Do not build a CSD indicator.**
13. **Sornette LPPLS thread: Shu, M. (2024) WIREs Comp. Stat. review (10.1002/wics.1649); Brée, D.S., Joseph, N.L. (2013). "Testing for financial crashes using the Log Periodic Power Law model." *Int. Rev. Financial Analysis*; "LPPLS bubble indicators over two centuries of the S&P 500" *Physica A* (2016).** — Verified. Fit fragility, parameter-range failures on 4/11 Hang Seng crashes, mixed OOS. **VERDICT: dead end as a tradable signal at our scale; monitoring-only at best.**

### Deep learning strategies (with published OOS)
14. **Lim, B., Zohren, S., Roberts, S. (2019). "Enhancing Time Series Momentum Strategies Using Deep Neural Networks." *Journal of Financial Data Science* (arXiv:1904.04912).** — Verified. Sharpe-loss LSTM on 88 futures; >2× improvement pre-cost, survives 2–3bp costs. **VERDICT: futures universe + tiny cost tolerance ≠ our 10bp equity costs; adopt the *Sharpe-as-loss + vol-scaling* framing only, not the model.**
15. **Guijarro-Ordonez, J., Pelger, M., Zanotti, G. (2021+). "Deep Learning Statistical Arbitrage." arXiv:2106.04028 / SSRN 3862004.** — Verified. Trade *residuals* of latent-factor model, not raw returns; consistently high OOS Sharpe. **VERDICT: full pipeline too heavy/leaky for us, but the residualization principle is the single best transferable mechanism for Book D (see Idea 4).**
16. **Gu, S., Kelly, B., Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning." *RFS* 33(5):2223–2273.** — Verified. Trees/NNs double regression-strategy performance; dominant predictors are price-based (momentum, liquidity, volatility) — i.e., largely computable from OHLCV. **VERDICT: registry keeper; a price-feature-only GBT cross-sectional model is a plausible future Book G but a full-quarter effort — not this month's top 5.**

### Intraday periodicity (usable with hourly bars)
17. **Heston, S.L., Korajczyk, R.A., Sadka, R. (2010). "Intraday Patterns in the Cross-Section of Stock Returns." *Journal of Finance* 65(4).** — Verified. Return continuation at half-hour intervals that are *exact multiples of one trading day*, persisting ≥40 days; not explained by volume/spread patterns. **VERDICT: directly testable on our hourly panel — hour-of-day equity flows are periodic; touches Books D/F exit timing and a standalone seasonality signal.**
18. **Gao, L., Han, Y., Li, S.Z., Zhou, G. (2018). "Market intraday momentum." *Journal of Financial Economics* 129(2):394–414.** — Verified. First half-hour SPY return (vs prior close) predicts last half-hour; stronger on volatile/high-volume days. **VERDICT: our hourly bars capture a coarse (first-hour → last-hour) version; cheap overlay test on SPY/QQQ.**

**Registry cross-reference (already read, cited for synthesis only):** Daniel–Moskowitz momentum crashes, Nagel short-term reversal-as-liquidity-provision, Moreira–Muir vol-managed portfolios, Lou–Polk–Skouras overnight/intraday tug-of-war, Heston–Sadka (daily seasonality), Kelly/MacLean–Thorp–Ziemba, Ledoit–Wolf (linear).

---

## 2. TOP 5 IDEAS (ranked by expected frontier improvement × implementability ÷ effort)

---

### IDEA 1 — Statistical Jump-Model regime gate (replace/augment the panic gate)
- **Origin class:** 1 (direct implementation) + 2 (improvement to champion overlay)
- **Sources:** Shu–Yu–Mulvey (2024) [#1]; contrast refs [#4].
- **Mechanism:** Fit a 2-state statistical jump model (k-means-style clustering of return/vol features with a fixed penalty λ per state transition) to daily SPY returns, rolling. The jump penalty forces regime persistence, killing the false-alarm whipsaws that plague HMMs and simple threshold panic gates. Bearish-state signal scales down gross exposure of the high-DD books (A, F) and/or tightens the vol-target.
- **Economic rationale:** Equity risk regimes are persistent (volatility clustering + slow institutional deleveraging + margin spirals). The edge survives because acting on regime signals costs conviction and career risk for discretionary managers, and the signal trades rarely (low capacity impact). Our current panic gate is a hand-tuned threshold; the JM is the same idea with a principled, tested persistence structure.
- **Signal spec (integrity-hardened):** Features from daily SPY OHLCV only, all rolling: {r_t, downside dev (halflife 10d), sorted local vol (halflife 21d)} standardized by *expanding or rolling* scalers (no full-sample). Refit JM weekly on trailing 1000d window; today's state inferred from features through t-1; allocation change executes at next open (t+1). λ chosen once on 1997–2018 daily data (pre-live-window), then frozen; 2019–2026 is the OOS test with ≥1-week embargo between fit window and evaluation. Gate: bear state ⇒ Book A and F weights ×0.5 (grid {0.0, 0.25, 0.5}) and vol-target cap 10% (vs 15%).
- **Data needed:** SPY daily 1997+ (have). Optionally ^VIX as a feature (have).
- **Expected effect:** Paper shows OOS MaxDD and Sharpe improvement on indices 1990–2023; with 50% haircut, expect champion MaxDD -8.9% → ~-7%-ish and Sharpe +0.05–0.15. CAGR roughly flat. Primary win is DD headroom that lets us lever CAGR back up within the -11% budget.
- **Implementation sketch:** `strategies/regime_jump_model.py` (~150 LOC: k-state fit = alternating assignment with transition penalty, no external deps beyond numpy/sklearn); wire into `run_portfolio_abdc.py` as a weight-multiplier hook next to the existing panic gate; A/B: {no gate, current panic gate, JM gate, both}.

---

### IDEA 2 — Turning-point state machine for Book F (slow×fast momentum agreement)
- **Origin class:** 3 (synthesis: Garg et al. states + Wood et al. changepoint-severity throttle, minus the deep learning) 
- **Sources:** Garg–Goulding–Harvey–Mazzoleni (2023) [#3]; Wood–Roberts–Zohren (2022) [#2]; registry: Daniel–Moskowitz.
- **Mechanism:** For each Book F candidate (and/or the aggregate book), compute SLOW momentum (existing 750h lookback) and FAST momentum (~100–150h). Four states: Bull (both +), Bear (both −), Correction (slow +, fast −), Rebound (slow −, fast +). In Correction/Rebound (turning-point states), scale Book F position to 0.5×; optionally require fast-signal confirmation before new entries. This is Wood et al.'s "respond to disequilibrium severity" idea implemented with Garg et al.'s transparent state machine instead of an LSTM.
- **Economic rationale:** Momentum crashes concentrate at turning points where the stale slow signal bets the wrong way (Daniel–Moskowitz); slow/fast disagreement is a real-time, price-only proxy for "the trend just broke." Persists because momentum capital is benchmarked to standard 6–12m formation windows and adjusts slowly.
- **Signal spec:** Fast/slow cumulative returns computed through bar t-1; state evaluated at each rebalance decision and at a daily overlay check (state change ⇒ resize at next hourly bar's open, respecting 10bp cost per resize — resize at most 1×/day to cap turnover). Grid: fast lb ∈ {100h, 150h, 200h}, turning-point scale ∈ {0, 0.25, 0.5}. Chronological eval 2019–2026 with the existing Book F harness; embargo = 200h (one hold).
- **Data needed:** merged hourly panel (have).
- **Expected effect:** No credit — must test (synthesis). Target: Book F MaxDD -39.6% → ≤ -30% with ≤10% Sharpe give-up; portfolio-level DD contribution from F shrinks, allowing higher F weight. Garg et al. report dynamic-speed > best static-speed Sharpe uniformly across markets (haircut: treat as directional prior only).
- **Implementation sketch:** Extend `strategies/universe_hourly_momentum` (Book F) with a `turning_point_state()` function reusing the cumulative-return code at a second lookback; add scale parameter to the live config; backtest via existing Book F grid runner.

---

### IDEA 3 — Hour-of-day periodicity: exit-alignment for Books D/F + SPY intraday-momentum micro-book
- **Origin class:** 1 (direct) + 2 (improvement)
- **Sources:** Heston–Korajczyk–Sadka (2010) [#17]; Gao–Han–Li–Zhou (2018) [#18]; registry: Lou–Polk–Skouras, Heston–Sadka.
- **Mechanism (a — periodicity alignment):** HKS show returns continue at lags that are exact multiples of one trading day (same clock-time), for ≥40 days. Our day has ~7 hourly bars; Book D's hold=8h and B's 24h deliberately drift across clock time. Test: (i) hold ∈ {7h, 14h, 21h} for D (exit at the same hour of day as entry, harvesting the same-hour continuation instead of fighting it); (ii) condition D's entries on hour-of-day — measure bubble-signal P&L by entry hour and restrict to profitable hours if stable across 2019–2022 vs 2023–2026 halves. **Mechanism (b — SPY intraday momentum):** sign of (first hourly bar close vs prior daily close) on SPY/QQQ predicts the final hour's return on high-|move| days; trade final hour only when first-hour |return| > rolling 80th percentile.
- **Economic rationale:** Same-clock-time institutional flows (401k sweeps, mutual-fund/ETF rebalancing, TWAP schedules) recur daily; late-informed traders and infrequent rebalancers (Bogousslavsky 2016) push the close in the direction of the open. Persists because flow timing is operationally sticky.
- **Signal spec:** (a) pure re-parameterization of existing backtests — no new leakage surface; hour-of-day entry stats computed on first half, validated on second half (chronological). (b) signal known at bar 1 close; enter at bar 2 open of the final hour... concretely: enter at the open of the last hourly bar, exit at close (MOC assumption), 10bp round trip; only when trigger fired at first-bar close (≥5.5h earlier — no look-ahead). Expect coarse hourly bars to dilute the published half-hour effect: apply haircut >50%.
- **Data needed:** hourly panel + SPY/QQQ hourly (have).
- **Expected effect:** (a) no credit — must test; even +0.1 Sharpe on Book D (2.74) at zero new infrastructure is high value. (b) published JFE effect is strong pre-cost on half-hours; after hourly dilution + 10bp, expect marginal standalone (Sharpe ~0.3–0.5 book) — value is its near-zero correlation and tiny capital footprint.
- **Implementation sketch:** (a) add `hold` values + `entry_hour_filter` to Book D grid runner (afternoon of work); (b) 100-LOC standalone `run_spy_intraday_momentum.py` on the hourly loader.

---

### IDEA 4 — Residual-return bubble score for Book D (market-mode removal via RMT)
- **Origin class:** 3 (synthesis: Bun–Bouchaud–Potters eigenmode machinery + Guijarro-Ordonez–Pelger–Zanotti "trade the residual" principle, applied to our bubble score)
- **Sources:** [#8], [#15]; registry: Nagel (reversal = liquidity provision on idiosyncratic shocks).
- **Mechanism:** Book D scores stocks on their own-price bubble z-score, so in broad sell-offs the "top-20 oversold" is dominated by high-beta market exposure, not stock-specific overreaction. Fix: each hour, compute rolling correlation matrix of hourly returns (trailing ~500 bars) over the 515-name panel; extract top eigenvector (market mode, per RMT clearly separated from the Marchenko–Pastur bulk); define residual return r̃ᵢ = rᵢ − βᵢ·r_mkt-mode; compute the bubble score on the residual price path. Rank/select on residual score; keep everything else in Book D unchanged. Variant: blend rank = α·residual-score + (1−α)·raw-score.
- **Economic rationale:** What mean-reverts in hours is *idiosyncratic* liquidity-demand overshoot (Nagel); market-wide moves contain momentum/continuation and drawdown risk. Residualizing buys the reversion and sheds the beta — the same reason GPZ trade factor residuals. Persists because liquidity provision at the single-name level is capital- and risk-limit-constrained precisely when signals are largest.
- **Signal spec:** Eigenvector estimated on returns through bar t-1 (rolling 500h window, recomputed daily to bound compute); βᵢ from the same trailing window; scores shifted 1 bar as today; selection at t, execution at t+1 open (existing convention). No full-sample estimation anywhere. Backtest 2019–2026 on the existing Book D harness; compare raw vs residual vs blend on Sharpe/MaxDD *and* 2022-type stress windows specifically.
- **Data needed:** merged hourly panel (have). Compute: one 515×515 eigendecomposition per day — trivial.
- **Expected effect:** No credit — must test. Risk-noted: Book D already excels in sell-offs (2022 +46%) partly *because* of the beta it harvests on rebounds — residualization could reduce CAGR while improving DD. That is exactly why the blend parameter α exists. Success criterion: any α with Sharpe ≥ 2.74 and MaxDD better than -6.4%, or a DD-matched CAGR gain.
- **Implementation sketch:** `data/market_mode.py` (rolling top-eigenvector + betas, cached daily to parquet); flag in `strategies/contrarian_bubble_hourly.py` to score on residual series; reuse Book D grid runner with α ∈ {0, 0.25, 0.5, 0.75, 1.0}.

---

### IDEA 5 — Online-learning book allocator with regret guarantee (EG/ONS vs inverse-vol)
- **Origin class:** 1 (direct implementation of a published family)
- **Sources:** Mhammedi–Rakhlin (2022) [#7]; lineage Cover (1991), Helmbold et al. EG, Hazan et al. ONS (cited for context, not registry credit).
- **Mechanism:** Treat the live books (A, C, D, F [, B, E]) as "experts." Run Exponentiated Gradient over book daily returns: wᵢ ← wᵢ·exp(η·rᵢ,t/⟨w,r_t⟩), normalized, with learning rate η set by theory (η = √(8 ln N / T) scale) — not tuned. Guarantees O(√T) regret vs the best fixed-weight portfolio in hindsight (log-wealth). Keep the existing 15% vol-target + gate on top; cap any book at 50% (projection back to the capped simplex).
- **Economic rationale:** Not a market edge — a *robustness* edge: regret bounds mathematically cap how badly adaptive weights can lose to the best static mix, unlike our MomAlloc heuristic (lb=20d, rb=60d) whose parameters are fit to the same history it's judged on. Diversification alpha with worst-case insurance.
- **Signal spec:** Weights updated from returns through t-1, applied at t+1 open; η from the closed-form theory schedule only (no grid — that is the point); weekly rebalance execution to control turnover with daily virtual updates. Evaluate 2019–2026 chronologically against Fixed-EW/inverse-vol and MomAlloc champions; also report 2019–2022-fit-free property (there are no fitted params — the whole period is OOS by construction).
- **Data needed:** book daily-return series from existing backtests (have).
- **Expected effect:** Theory promises no *out*performance of the best static mix — expect Sharpe within ±0.1 of champion; the payoff is removing MomAlloc's two tuned hyperparameters and their overfitting risk, plus graceful handling when a book decays live (e.g., Book E sentiment data degradation post-2024 gets auto-de-weighted). No credit beyond robustness — must test.
- **Implementation sketch:** ~80 LOC `portfolio/eg_allocator.py`; drop into `run_portfolio_abdc.py` beside MomAlloc; one afternoon including the capped-simplex projection.

---

### Explicit dead ends this sweep (do not re-investigate without new evidence)
- **Critical-slowing-down autocorrelation crash indicators** — Guttal et al. (2016) find no CSD before financial meltdowns; the surviving signal (rising variance) is already harvested by our vol-target overlay.
- **LPPLS bubble timing** — fit fragility and mixed OOS across studies; not tradable at our frequency/universe.
- **Conformal-Kelly return-forecast sizing** — pre-registered OOS underperformed passive in the source paper itself. (Conformal *width as vol-proxy for the overlay* remains a cheap open question, logged for a future sweep.)
- **Raw daily-frequency HMM regime filters** — dominated by jump models in published head-to-heads.

---

## 3. For the registry

| Paper | Family | Verdict suggestion |
|---|---|---|
| Shu, Yu & Mulvey 2024 (J. Asset Mgmt) — Statistical Jump Model downside protection | regime | **ADOPT — Idea 1 (JM panic gate)** |
| Garg, Goulding, Harvey & Mazzoleni 2023 (JFE) — Momentum Turning Points | momentum/regime | **ADOPT — Idea 2 (Book F state machine)** |
| Wood, Roberts & Zohren 2022 (JFDS) — Slow Momentum with Fast Reversion (CPD) | momentum/ML | ADAPT — mechanism only (CPD severity), no LSTM |
| Heston, Korajczyk & Sadka 2010 (JF) — Intraday cross-section periodicity | intraday seasonality | **ADOPT — Idea 3a (hold/entry-hour alignment)** |
| Gao, Han, Li & Zhou 2018 (JFE) — Market intraday momentum | intraday momentum | TEST — Idea 3b (SPY last-hour micro-book), expect dilution on hourly bars |
| Bun, Bouchaud & Potters 2017 (Phys. Reports) — RMT correlation cleaning | covariance/econophysics | ADAPT — market-mode removal for Idea 4; full RIE not needed at N=4 books |
| Guijarro-Ordonez, Pelger & Zanotti 2021+ (arXiv/SSRN) — Deep Learning StatArb | ML/reversal | ADAPT — residualization principle only (Idea 4); full pipeline too heavy |
| Ledoit & Wolf 2020 (Ann. Statist.) — Analytical nonlinear shrinkage | covariance | HOLD — tool on shelf for any future large-N Markowitz book |
| Bongiorno & Challet 2021/2023 (arXiv) — NLS suboptimal under non-stationarity; Average Oracle | covariance | HOLD — caution note attached to Ledoit-Wolf 2020 |
| Mhammedi & Rakhlin 2022 (COLT) — Damped ONS portfolio selection | online learning | TEST — Idea 5 (EG/ONS allocator A/B vs MomAlloc) |
| Lim, Zohren & Roberts 2019 (JFDS) — Deep Momentum Networks | momentum/ML | REJECT for direct use (2–3bp cost tolerance ≪ our 10bp); keep Sharpe-loss framing |
| Gu, Kelly & Xiu 2020 (RFS) — Empirical Asset Pricing via ML | cross-section/ML | DEFER — price-feature-only GBT is a plausible Book G; quarter-scale effort |
| Ryan 2026 (arXiv:2608.01494) — Conformal Kelly | sizing/conformal | REJECT (failed own pre-registered OOS); log conformal-width-as-risk-throttle as open question |
| Wisniewski, Lindsay & Lindsay 2020 (COPA) — Conformal MM net positions | conformal | DEAD END — wrong domain (FX inventory) |
| Scheffer et al. 2009 (Nature) — Early-warning signals for critical transitions | econophysics/EWS | CONTEXT ONLY |
| Guttal et al. 2016 (PLOS ONE) — No CSD before financial meltdowns | econophysics/EWS | **DEAD END (valuable negative)** — kills CSD thread |
| Sornette LPPLS corpus (Brée & Joseph 2013 IRFA; Shu 2024 WIREs; Physica A 2016) | econophysics/bubbles | DEAD END — fit fragility, mixed OOS |
| HMM regime lit. incl. arXiv:2406.09578 | regime | SUPERSEDED by jump models |

**Citation verification status:** all 18 entries verified to exist via web search / direct arXiv fetch (2026-08-15). No unverifiable citations. Note: "Average Oracle" author attribution (Bongiorno & Challet) taken from arXiv listings; exact author order on 2309.17219 should be double-checked at implementation time.
