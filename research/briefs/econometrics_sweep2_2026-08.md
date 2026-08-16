# Econometrics/Statistics Sweep — Round 2 (2026-08)

**Focus:** forecasting & risk-modeling methods for the PORTFOLIO OVERLAY (de-risk-only 15% vol target + DM panic gate on inverse-vol book weights). Champion: Sharpe 2.781 / MaxDD -8.2%. `sigma_hat` today = trailing 20d realized vol of portfolio daily returns.

**Registry context (round 1):** HAR on daily r² proxy HURT (2.27); HAR-X w/ SPY intraday RV +0.05 (significance unknown — addressed below); EWMA wash; QLIKE committee dragged; jump-regime gate lost to DM gate; Ledoit-Wolf/NLS shelved; CDaR LP lost; ERC/HRP/Kelly lost to inverse-vol (N=5).

**Integrity rules honored:** every spec below is expanding- or rolling-fit, signal at t uses data ≤ t-1, applied at t close for t+1 exposure. No full-sample estimation. All performance claims below are from the cited papers' samples, NOT projections for our system — every idea must be A/B'd on our books before any promotion.

---

## 1. Papers reviewed, with verdicts

### Vol targeting & its refinements

| Paper | Venue | Verdict for us |
|---|---|---|
| Moreira & Muir (2017), "Volatility-Managed Portfolios," *Journal of Finance* 72(4), 1611-1644 | JF | Canonical basis for the overlay. Mechanism: vol changes are not offset by proportional expected-return changes. Nothing new to mine; keep as reference. |
| Cederburg, O'Doherty, Wang & Yan (2020), "On the performance of volatility-managed portfolios," *JFE* 138(1), 95-117 | JFE | **Cautionary.** Across 103 strategies, real-time (expanding, implementable) versions of vol management generally earn *lower* Sharpe than unmanaged — spanning-regression alphas are structurally unstable OOS. Supports our de-risk-only (never lever up) design; warns against promoting overlay tweaks on in-sample deltas. Feeds Idea #1. |
| Harvey, Hoyle, Korgaonkar, Rattray, Sargaison & van Hemert (2018), "The Impact of Volatility Targeting," *JPM* 45(1), 14-33 | JPM | Vol targeting raises Sharpe for risk assets (equity/credit) only; main channel is tail cutting. Their evidence favors relatively fast estimators for equities (vol clustering is short-lived). Feeds Idea #3 (response speed). |
| Bongaerts, Kang & van Dijk (2020), "Conditional Volatility Targeting," *Financial Analysts Journal* 76(4) (SSRN 3636727) | FAJ | **Directly relevant.** Unconditional vol targeting is inconsistent; adjusting exposure *only in the extremes* (de-risk in high-vol states, no lever-up in calm) enhances Sharpe and cuts drawdowns/turnover. Our overlay is already conditional-ish (de-risk-only); their state-dependent trigger (act only above a high-vol percentile) is testable as a refinement. Feeds Idea #3. |
| Bernardi, Bianchi & Bianco (2022), "Smoothing volatility targeting," arXiv:2212.07288 | arXiv (unrefereed) | Smoothed predictive vol (stochastic-vol model) beats raw realized-variance scaling mainly via turnover reduction. Sanity check for us: our 20d trailing window is already smooth-ish; main takeaway is don't chase daily sigma_hat noise. Low priority standalone. |
| Wang & Yan (2021), "Downside risk and the performance of volatility-managed portfolios," *Journal of Banking & Finance* 131 | JBF | **Top candidate.** Scaling by *downside* volatility significantly beats scaling by total vol — in spanning regressions, direct Sharpe comparisons, and *real-time* strategies (i.e., survives the Cederburg critique). Mechanism: downside vol negatively predicts returns; upside vol does not. Feeds Idea #2. |

### Vol forecasting (beyond HAR)

| Paper | Venue | Verdict for us |
|---|---|---|
| Glosten, Jagannathan & Runkle (1993), *JF* 48(5); Nelson (1991), *Econometrica* 59(2) — GJR / EGARCH | JF / ECMA | Leverage-effect GARCH. For *targeting* (not point-forecast contests), the leverage asymmetry is the one feature plain trailing vol lacks. BUT: expanding MLE on ~1,900 obs of a 5-book portfolio return is fragile, and the same asymmetry is captured nonparametrically by downside semivariance (Wang-Yan) at zero estimation risk. Verdict: test only if Idea #2 helps, as its parametric cousin. |
| Hansen, Huang & Shek (2012), "Realized GARCH," *Journal of Applied Econometrics* 27(6), 877-906 | JAE | Joint return/RV model, beats standard GARCH when an RV measure exists. Our portfolio-level RV only exists from SPY-hourly era (2020+) via proxy; sample too short to fit expanding GARCH-family reliably before ~2022. Shelve. |
| Bollerslev, Hood, Huss & Pedersen (2018), "Risk Everywhere: Modeling and Managing Volatility," *RFS* 31(7), 2729-2773 | RFS | HF-data risk models + panel estimation give superior OOS vol forecasts and *utility gains net of t-costs*; gains depend on trading speed vs cost trade-off. Endorses our HAR-X (+0.05) direction and motivates panel-based sigma (we have the hourly stock panel). Feeds Ideas #4/#5. |
| Ghysels, Santa-Clara & Valkanov (2005), "There is a risk-return trade-off after all," *JFE* 76(3), 509-548; and "Predicting volatility: getting the most out of return data sampled at different frequencies," *JoE* (2006) | JFE/JoE | MIDAS: flexible-weight aggregation of daily squared returns. Functionally overlaps HAR (already tested: hurt on r² proxy, marginal with RV). MIDAS weights fitted expanding on ~1,900 obs = more estimation risk for the same information set. Deprioritize. |
| Parkinson (1980), *J. Business* 53(1); Garman & Klass (1980), *J. Business* 53(1); Molnár (2012), "Properties of range-based volatility estimators," *Int. Rev. Fin. Analysis* 21 | JB / IRFA | Range (OHLC) estimators are 5-14x more efficient than close-to-close per observation; Molnár finds Garman-Klass best all-round. Key for us: SPY OHLC exists **back to 1997** (unlike SPY hourly, 2020+), so a GK/Parkinson regressor gives HAR-X-style information over the FULL sample. Cheap to compute. Feeds Idea #4. |

### Downside / tail forecasting

| Paper | Venue | Verdict for us |
|---|---|---|
| Patton, Ziegel & Chen (2019), "Dynamic semiparametric models for expected shortfall (and Value-at-Risk)," *Journal of Econometrics* 211(2), 388-413 | JoE | Joint ES/VaR GAS models via the Fissler-Ziegel joint loss; semiparametric, no distributional assumption. Legit path to ES-targeting (`scale = min(1, ES*/ES_hat)`). Cost: numerical optimization, expanding refits, ~4 params on 1,900 obs. Verdict: promising but SECOND in the downside queue behind nonparametric semivariance targeting (same economics, zero fitting). |
| Engle & Manganelli (2004), "CAViaR: Conditional Autoregressive Value at Risk by Regression Quantiles," *JBES* 22(4), 367-381 | JBES | Dynamic quantile framework; with exogenous regressors it becomes a quantile regression for drawdown-onset using our breadth/dispersion/correlation internals. Feeds Idea #5. |
| Harvey, Leybourne & Newbold (1997), "Testing the equality of prediction mean squared errors," *IJF* 13(2), 281-291 | IJF | Small-sample correction to DM test. Use whenever we DM-test forecast losses. Feeds Idea #1. |

### Forecast / overlay evaluation

| Paper | Venue | Verdict for us |
|---|---|---|
| Diebold & Mariano (1995), "Comparing Predictive Accuracy," *JBES* 13, 253-263 | JBES | Foundation for loss-differential testing (use QLIKE loss per Patton 2011, *JoE* 160(1), on the forecast layer). |
| Giacomini & White (2006), "Tests of Conditional Predictive Ability," *Econometrica* 74(6) | ECMA | The correct DM variant for *rolling/expanding-window methods* (tests the method incl. estimation scheme, no parameter-uncertainty correction needed). Our exact setting. |
| Hansen, Lunde & Nason (2011), "The Model Confidence Set," *Econometrica* 79(2), 453-497 | ECMA | Multi-model version: from a sweep of K overlay variants, returns the subset containing the best at confidence 1-α. Kills cherry-picking the best of K backtests. |
| Ledoit & Wolf (2008), "Robust performance hypothesis testing with the Sharpe ratio," *Journal of Empirical Finance* 15(5), 850-859 | JEF | **The tool for our promotion decisions.** Studentized stationary-bootstrap (Politis-Romano 1994) CI for the *difference of two Sharpe ratios* on paired return series; valid under heavy tails + serial dependence. Feeds Idea #1. |

### Cross-sectional risk (we have the hourly stock panel)

| Paper | Venue | Verdict for us |
|---|---|---|
| Pollet & Wilson (2010), "Average correlation and stock market returns," *JFE* 96(3), 364-380 | JFE | Average pairwise correlation of stocks predicts market excess returns/risk beyond aggregate variance (variance ≈ avg-vol² × avg-corr decomposition; corr is the systematic part). Correlation spikes = diversification failure = exactly when a 5-book inverse-vol portfolio's *realized* vol jumps late. Feeds Idea #5. |
| Herskovic, Kelly, Lustig & Van Nieuwerburgh (2016), "The common factor in idiosyncratic volatility," *JFE* 119(2), 249-283 | JFE | Firm idio vol has a strong common factor (CIV); CIV innovations are priced and track household/aggregate risk. Panel-average idio vol is a computable daily state variable from our hourly panel. Feeds Idea #5. |
| Baltussen, van Bekkum & van der Grient (2018), "Unknown Unknowns: Uncertainty about Risk and Stock Returns," *JFQA* 53(4), 1615-1651 | JFQA | Vol-of-vol (their measure is option-implied; ours would be realized) relates to uncertainty about risk; high vol-of-vol states are bad states. Cross-sectional return result, not a targeting result — treat realized vol-of-vol as one candidate feature in the quantile gate, not a standalone overlay. |

---

## 2. TOP 5 ranked ideas

### #1 — Overlay promotion-testing protocol: Ledoit-Wolf Sharpe-delta bootstrap + GW/HLN on forecast losses + MCS for sweeps  *(process, ranked first because it protects every future promote)*

- **Citations:** Ledoit & Wolf (2008, JEF); Politis & Romano (1994, JASA — stationary bootstrap); Giacomini & White (2006, ECMA); Harvey, Leybourne & Newbold (1997, IJF); Hansen, Lunde & Nason (2011, ECMA); Patton (2011, JoE — QLIKE with imperfect proxies).
- **Problem:** we promote on +0.05–0.10 Sharpe deltas. Analytic approximation (illustrative, not a backtest number): for two overlay variants with daily-return correlation ρ on T obs, `SE_ann(ΔSR) ≈ sqrt(2(1-ρ)/T)·sqrt(252)` (ignoring higher moments). With T≈1,900 (2019-2026) and ρ≈0.98 (typical for two vol-overlay variants on the same book returns), SE ≈ 0.07 — so a +0.05 delta is ~0.7 SE. **Our HAR-X +0.05 promote is, a priori, indistinguishable from noise until tested.** The exact answer must come from the bootstrap below, which also handles tails/autocorrelation.
- **Procedure (adopt as standing rule):**
  1. **Paired design.** Same book returns, same costs; only sigma_hat differs → daily return series r_A, r_B, aligned.
  2. **Sharpe layer:** Ledoit-Wolf studentized stationary-bootstrap CI for ΔSR. B = 4,999; expected block length ~5–10d (their algorithm has a data-driven choice); promote only if the 90% CI excludes 0, i.e. one-sided p < 0.05–0.10.
  3. **Forecast layer (diagnostic):** daily QLIKE loss of sigma_hat_A vs sigma_hat_B against next-day realized proxy (squared return, or SPY-RV-scaled proxy); Giacomini-White conditional test (valid for rolling/expanding schemes) with HLN small-sample correction. A variant that doesn't even forecast better should not win the Sharpe test except by luck — use as a coherence check.
  4. **Sweeps (K > 2 variants):** run MCS at α = 0.10 on daily utility loss (e.g., negative of overlay return net of costs, or QLIKE). Report the surviving set, not the argmax. Never promote an argmax that sits in a wide MCS.
  5. **Secondary gates:** sign-consistency across calendar years (≥ 6/8), and MaxDD not worse by > 1pt. Log the p-value in the registry line.
- **Expected frontier effect:** none directly; prevents frontier erosion from noise-promotes (Cederburg et al. 2020 shows exactly this failure mode in the vol-management literature).
- **Implementation:** ~150 lines NumPy; stationary bootstrap is 20 lines; QLIKE+GW/HLN trivial. Backfill: re-test the HAR-X +0.05 promote first.

### #2 — Downside (semivariance) targeting: replace sigma_hat with trailing downside semideviation

- **Citations:** Wang & Yan (2021, JBF); Markowitz (1959) heritage; consistent with Harvey et al. (2018) tail-cutting channel.
- **Mechanism:** downside vol negatively predicts returns while upside vol does not; total vol conflates the two, so a 20d window polluted by *upside* spikes de-risks at exactly the wrong time (e.g., sharp rebounds — our books' best days). Semivariance targeting keeps exposure through upside vol, cuts it on downside vol, and captures the leverage effect nonparametrically (no GARCH fit).
- **Leak-free spec:** `semivar_t = (252/20)·Σ_{i=t-19..t} min(r_i - 0, 0)²` (zero threshold, daily portfolio returns up to close t); `sigma_down_t = sqrt(2·semivar_t)` (×2 rescales one-sided to two-sided so the 15% target keeps its meaning; also test recalibrated target on expanding pre-2021 data only); `scale_{t+1} = min(1, 0.15/sigma_down_t)`. No fitting at all → zero estimation leakage. Variants for the MCS sweep: window ∈ {10, 20, 40}d, threshold ∈ {0, expanding mean}.
- **Expected frontier effect:** paper shows significant Sharpe improvement over total-vol scaling incl. real-time; for us plausibly +0.03–0.10 Sharpe and equal-or-better MaxDD vs the current overlay (must verify via Idea #1 protocol; do not trust the range).
- **Implementation:** ~10 lines change in the overlay; 1 day incl. A/B + bootstrap test. **Highest value-per-effort in this sweep.**

### #3 — Asymmetric response speed: de-risk fast / re-risk slow composite estimator + conditional trigger

- **Citations:** Bongaerts, Kang & van Dijk (2020, FAJ) — act only in extreme states; Harvey et al. (2018, JPM) — short windows suit equities; Bernardi, Bianchi & Bianco (2022, arXiv:2212.07288) — smoothing cuts whipsaw/turnover.
- **Mechanism:** a single 20d window is simultaneously too slow into a crash (damage done before sigma_hat reacts) and too fast out of it (re-risks into the unstable aftermath, the "volatility trap"/whipsaw: sell low, re-buy high). Asymmetry fixes both without levering up anywhere.
- **Leak-free spec (no fitting):** `sigma_fast_t` = 5d trailing (or EWMA λ=0.8), `sigma_slow_t` = 60d trailing; `sigma_hat_t = max(sigma_fast_t, sigma_slow_t)`; `scale_{t+1} = min(1, 0.15/sigma_hat_t)`. Max() means: de-risk as soon as the fast estimator fires; re-risk only after BOTH have normalized (slow window enforces the waiting period). BKvD conditional variant: apply scaling only when sigma_hat_t is above its expanding-window 80th percentile (percentile computed on data ≤ t), else scale = 1 — reduces turnover and avoids scaling in benign regimes where it historically adds nothing.
- **Expected frontier effect:** primarily MaxDD improvement (faster crash response) with Sharpe roughly preserved; BKvD report Sharpe up + drawdowns/turnover down for the conditional version on equity factors. Interaction with the DM panic gate must be checked (overlap: both fire in panics; the composite may make the DM gate partially redundant — test with gate on AND off).
- **Implementation:** ~20 lines; grid {fast: 5/10d} × {slow: 40/60d} × {trigger: none/80th pct} through the MCS procedure from Idea #1. 1-2 days.

### #4 — Range-based (Garman-Klass / Parkinson / Yang-Zhang) SPY OHLC as full-history exogenous vol input

- **Citations:** Parkinson (1980, JB); Garman & Klass (1980, JB); Molnár (2012, IRFA — GK best in class); Bollerslev, Hood, Huss & Pedersen (2018, RFS — intraday info → utility gains at portfolio level, net of costs).
- **Mechanism:** registry says HAR-X with SPY intraday RV helped (+0.05, unverified) — but SPY hourly only exists 2020+. GK/Parkinson on SPY *daily OHLC* delivers most of the intraday-information gain (5-14x efficiency per obs vs close-to-close) **back to 1997**, covering the full overlay history and the panic-gate calibration era. Range-based vol is also same-day-reactive: a huge intraday range on a flat close raises GK vol immediately while close-to-close vol sees nothing.
- **Leak-free spec:** `GK_t = 0.5·(ln(H_t/L_t))² - (2ln2-1)·(ln(C_t/O_t))²` daily; 10d mean, annualized → `sigma_GK_t`. Map to portfolio scale via expanding-window beta: regress portfolio 20d vol on sigma_GK over data ≤ t-1 (refit monthly, expanding; min 2y warmup), OR skip regression entirely and use the conservative composite `sigma_hat_t = max(sigma_own20_t, b_t·sigma_GK_t)`. The no-regression variant `max(sigma_own, sigma_GK · median_ratio_expanding)` has near-zero estimation risk — prefer it first.
- **Expected frontier effect:** same channel as HAR-X (+0.05 claimed) but usable over the whole sample and cheaper; realistically 0 to +0.05 Sharpe with small MaxDD gain from same-day reactivity. Must pass Idea #1 protocol (the HAR-X result it would replace hasn't).
- **Implementation:** SPY OHLC via yfinance (have it); ~40 lines; 1 day. Combines naturally with Idea #3 (GK as the "fast" leg).

### #5 — Cross-sectional risk gauges from the hourly panel: average correlation (+ CIV, vol-of-vol) as drawdown-onset quantile gate

- **Citations:** Pollet & Wilson (2010, JFE) — avg pairwise correlation is the forward-looking systematic component of market variance; Herskovic, Kelly, Lustig & Van Nieuwerburgh (2016, JFE) — common idio vol factor tracks aggregate risk; Baltussen, van Bekkum & van der Grient (2018, JFQA) — vol-of-vol as uncertainty-about-risk; Engle & Manganelli (2004, JBES) — dynamic quantile framework for the gate.
- **Mechanism:** own-portfolio 20d vol is a *lagging* diagnosis. Correlation spikes and common-idio-vol spikes are *earlier* symptoms: when pairwise correlations jump, diversification across our 5 books' underlying stock positions fails *before* portfolio realized vol prints it. Answers the sweep question directly: does panel risk forecast our drawdowns better than own-vol?
- **Leak-free spec:** daily, from hourly panel (2019+): (a) `avgcorr_t` = mean pairwise correlation of hourly returns over trailing 10d across a fixed liquid subset (e.g., top 100 by history completeness — fixed list chosen once on pre-2020 data to avoid selection leakage); (b) `CIV_t` = cross-sectional mean of 10d idio vol (residual vs SPY-beta, beta from trailing 60d); (c) `volofvol_t` = std of daily panel-avg vol over trailing 20d. **Stage 1 (validation):** predictive quantile regression, expanding fit refit monthly with 2y warmup: 5th percentile of next-5d portfolio return on {avgcorr, CIV, volofvol, own-vol, DM-gate state}; check pinball-loss improvement vs own-vol-only via GW test. **Stage 2 (only if stage 1 wins):** gate `scale_{t+1} = min(1, 0.15/sigma_hat, 1{q̂5_t > -k}·1 + 1{q̂5_t ≤ -k}·0.5)` — i.e., halve exposure when the predicted 5% quantile breaches -k (k on expanding calibration).
- **Expected frontier effect:** uncertain — this is the exploratory slot. If avg-corr adds nothing beyond own-vol + DM gate at N=5 books, kill at stage 1 cheaply (no backtest needed, just forecast-loss tests). Upside case: earlier de-risking in correlation-driven events (e.g., 2022-style regime turns) → MaxDD improvement.
- **Implementation:** panel plumbing is the bulk (~1-2 days, panel is already cached in `data/cache/merged_hourly_close.parquet`); quantile reg via statsmodels. 2-3 days total.

**Explicitly deprioritized this round:** MIDAS (same information set as HAR, more estimation risk, HAR family already 1-for-2 against us); expanding GJR/EGARCH MLE (leverage effect obtained free via Idea #2; revisit only if #2 wins and we want its parametric refinement); Realized GARCH (needs RV history we lack pre-2020); Patton-Ziegel-Chen ES-GAS targeting (queue behind #2 — same downside economics, 10x the estimation machinery; promote to next sweep if #2 helps).

---

## 3. For the registry

| Idea | Origin | Verdict/status | Effort | Kill criterion |
|---|---|---|---|---|
| LW Sharpe-delta bootstrap + GW/HLN + MCS promotion protocol | Ledoit-Wolf 2008; Giacomini-White 2006; Hansen-Lunde-Nason 2011 | **ADOPT as process**; retro-test HAR-X +0.05 first | 1d | n/a (process) |
| Downside semivariance targeting (20d, de-risk-only) | Wang-Yan 2021 JBF | **TEST NEXT — top candidate** | 0.5-1d | ΔSharpe CI contains 0 AND MaxDD not improved |
| Asymmetric speed: max(fast 5d, slow 60d) + BKvD extreme-state trigger | Bongaerts-Kang-van Dijk 2020 FAJ; Harvey et al. 2018 | TEST (after #2; combine best) | 1-2d | No MaxDD gain vs champion; turnover up >2x |
| Garman-Klass SPY OHLC composite sigma (1997+ history) | Garman-Klass 1980; Molnár 2012; Bollerslev et al. 2018 | TEST (replaces/validates HAR-X promote) | 1d | Loses GW QLIKE test vs own-vol AND ΔSharpe CI contains 0 |
| Panel avg-correlation / CIV quantile drawdown gate | Pollet-Wilson 2010; Herskovic et al. 2016; Engle-Manganelli 2004 | STAGE-1 VALIDATE (forecast loss only, cheap kill) | 2-3d | Pinball loss not better than own-vol baseline (GW p>0.10) |
| MIDAS weights on daily r² | Ghysels et al. 2005/2006 | REJECT this round (HAR-family redundancy) | — | — |
| Expanding GJR/EGARCH for sigma_hat | GJR 1993; Nelson 1991 | DEFER behind semivariance | — | — |
| Realized GARCH | Hansen-Huang-Shek 2012 | SHELVE (RV history too short) | — | — |
| ES-GAS targeting (joint ES/VaR) | Patton-Ziegel-Chen 2019 JoE | QUEUE for sweep 3 iff semivariance wins | — | — |

**Standing rule proposed:** no overlay promotion on ΔSharpe alone; require Ledoit-Wolf one-sided p < 0.10 on the paired daily series + yearly sign-consistency + MaxDD non-degradation, and MCS-surviving (not argmax) when sweeping >2 variants.

*All citations verified via web search 2026-08-16. Paper performance figures are from the papers' own samples; nothing here is a projected result for our system.*
