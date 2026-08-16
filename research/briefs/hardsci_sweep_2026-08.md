# Hard-Science + Econometrics/OR Idea-Generation Sweep — August 2026

**Scope:** Monthly sweep, bucket 3 of 3 (finance and ML/CS already swept). Venues: Physical Review E / Physica A / Quantitative Finance (econophysics), Nature / Scientific Reports (collective behavior, systemic risk), Econometrica / J. Econometrics / JBES / REStat (regime switching, realized volatility, forecast combination), Management Science / Operations Research lineage (drawdown-constrained optimization, OR-origin forecasting).
**Constraint set:** US-equity OHLCV only — daily 1997+ (~523 names incl. SPY/UVXY/^VIX, with volume), hourly 2019+ (515 names). No fundamentals/options/order-book/tick data.
**Frontier to beat:** Champion portfolio (inverse-vol + 15% vol-target + panic gate) Sharpe 2.580, CAGR 43.5%, MaxDD -8.9% (2019–2026). Improvement = higher Sharpe/CAGR with MaxDD ≤ ~-11%.
**Integrity baseline for all specs:** signals shifted ≥1 bar; rolling/expanding statistics only (no full-sample scalers); next-bar execution; chronological splits; McLean–Pontiff ~50% haircut on published effects; synthesis/original ideas get NO expectation credit until tested.
**Tested-and-failed (not re-proposed):** residual/market-mode-stripped contrarian scores (registry #47–48), 52wk-high ranking (#31), overnight re-timing at multi-week holds (#33), per-book trend gates (#24/#29/#43), hour-of-day hold alignment (#45), jump-model regime gate (#42), CSD/LPPLS early warning (#56–57 + ML sweep).

All citations verified via web search on 2026-08-15. None fabricated.

---

## 1. Papers reviewed (with one-line verdicts)

### A. Realized-volatility econometrics (the vol-target-overlay thread — our biggest lever)
1. **Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized Volatility." *Journal of Financial Econometrics* 7(2):174–196.** — Verified (OUP, DOI 10.1093/jjfinec/nbp001). The HAR-RV model: tomorrow's RV regressed on daily/weekly(5d)/monthly(22d) average RV; parsimonious cascade structure reproduces long memory and beats short-memory AR and rolling-window historical vol. **VERDICT: the single most actionable paper of this sweep — drop-in upgrade for our overlay's 20d trailing-vol estimator (Idea 1).**
2. **Andersen, T.G., Bollerslev, T., Diebold, F.X., Labys, P. (2003). "Modeling and Forecasting Realized Volatility." *Econometrica* 71(2):579–625.** — Verified (Wiley, DOI 10.1111/1468-0262.00418). Foundations: RV from intraday returns is a (nearly) model-free vol measure; **log-RV is approximately Gaussian** and simple Gaussian AR systems on log-RV forecast well. **VERDICT: methodological backbone for Idea 1 — forecast log-RV, not RV.**
3. **Bollerslev, T., Patton, A.J., Quaedvlieg, R. (2016). "Exploiting the errors: A simple approach for improved volatility forecasting." *Journal of Econometrics* 192(1):1–18.** — Verified (ScienceDirect, DOI 10.1016/j.jeconom.2015.10.007). HARQ: scale the daily-lag coefficient by realized quarticity (RV measurement error) — more persistence when RV is precisely measured, faster mean-reversion when noisy; beats HAR on S&P 500 + DJIA constituents. **VERDICT: adopt as variant B of Idea 1; caution — RQ from only 7 hourly bars/day is noisy, may need weekly aggregation.**
4. **Patton, A.J., Sheppard, K. (2015). "Good Volatility, Bad Volatility: Signed Jumps and the Persistence of Volatility." *REStat* 97(3):683–697.** — Verified (MIT Press/JSTOR). Future vol driven mainly by **negative-return semivariance**; negative jumps raise, positive jumps lower future vol; semivariance-HAR improves OOS forecasts. **VERDICT: variant C of Idea 1 — we can compute daily downside semivariance from our hourly bars 2019+.**
5. **Corsi, F., Renò, R. (2012). "Discrete-Time Volatility Forecasting With Persistent Leverage Effect..." *JBES* 30(3):368–380.** — Verified (T&F, DOI 10.1080/07350015.2012.663261). LHAR: adding persistent negative-return (leverage) terms at daily/weekly/monthly horizons significantly improves S&P 500 vol forecasts. **VERDICT: cheap extra regressors for Idea 1 (works even on daily-only data pre-2019).**
6. **Gatheral, J., Jaisson, T., Rosenbaum, M. (2018). "Volatility is rough." *Quantitative Finance* 18(6):933–949.** — Verified (T&F; arXiv:1410.3394). Log-vol behaves like fBM with H≈0.1; RFSV yields improved RV forecasts. **VERDICT: context — validates log-vol modeling and HAR-like kernels (HAR approximates rough forecasts at our horizons); full RFSV estimation not worth the effort delta. Registry as rejected-for-implementation/context.**
7. **Yang, D., Zhang, Q. (2000). "Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices." *Journal of Business* 73(3):477–492.** — Verified (JSTOR, DOI 10.1086/209650). Minimum-variance unbiased range-based vol estimator handling overnight jumps and drift; dramatic efficiency gain over close-to-close. **VERDICT: we have daily OHLC 1997+ — use as the RV proxy where hourly bars don't exist, and as a component in the Idea 5 combination.**
8. **Ghashghaie, S., Breymann, W., Peinke, J., Talkner, P., Dodge, Y. (1996). "Turbulent cascades in foreign exchange markets." *Nature* 381:767–770.** — Verified (Nature). Information cascade from long to short timescales, analogy to Kolmogorov turbulence — the conceptual ancestor of HAR's heterogeneous-horizon structure. **VERDICT: background only; no direct implementation.**

### B. Correlation dynamics / market states (econophysics)
9. **Preis, T., Kenett, D.Y., Stanley, H.E., Helbing, D., Ben-Jacob, E. (2012). "Quantifying the Behavior of Stock Correlations Under Market Stress." *Scientific Reports* 2:752.** — Verified (Nature SR, DOI 10.1038/srep00752). Mean pairwise correlation of DJIA stocks **scales linearly with market stress** (normalized index drawdown/returns) across timescales; diversification melts exactly when needed. **VERDICT: core empirical basis for Idea 2 — correlation level is a real-time fragility gauge our vol estimator only sees with a lag.**
10. **Onnela, J.-P., Chakraborti, A., Kaski, K., Kertész, J., Kanto, A. (2003). "Dynamics of market correlations: Taxonomy and portfolio analysis." *Physical Review E* 68:056110.** — Verified (APS, DOI 10.1103/PhysRevE.68.056110). Minimum-spanning "asset tree" shrinks topologically in crashes (low mean occupation layer); tree structure differs between normal and crash regimes. **VERDICT: corroborates Idea 2; the MST machinery itself is unnecessary — scalar mean-correlation/eigenvalue statistics capture the same compression.**
11. **Münnix, M.C., Shimada, T., Schäfer, R., Leyvraz, F., Seligman, T.H., Guhr, T., Stanley, H.E. (2012). "Identifying States of a Financial Market." *Scientific Reports* 2:644.** — Verified (DOI 10.1038/srep00644). Clustering rolling correlation matrices yields a small number of discrete market states with jump dynamics; crisis states are distinct. **VERDICT: heavier cousin of Idea 2 — start with scalar correlation/absorption signals; revisit clustering only if scalars show signal.**
12. **Kritzman, M., Li, Y., Page, S., Rigobon, R. (2011). "Principal Components as a Measure of Systemic Risk." *Journal of Portfolio Management* 37(4):112–126.** — Verified (PM Research/SSRN). Absorption ratio (variance share of top eigenvectors, 500d window) measures market compactness/fragility; **standardized AR shift (15d vs 1yr) preceded large return differentials** around stress events. **VERDICT: the concrete trading-rule form of Idea 2; practitioner journal — treat effect sizes with extra skepticism.**

### C. Lead-lag networks (promising thread — honestly weak for our data)
13. **Curme, C., Tumminello, M., Mantegna, R.N., Stanley, H.E., Kenett, D.Y. (2015). "Emergence of statistically validated financial intraday lead-lag relationships." *Quantitative Finance* 15(8):1375–1386.** — Verified (T&F, DOI 10.1080/14697688.2015.1032545; arXiv:1401.0462). Validated lagged-correlation networks on US equities: networks are dense at **15-minute** sampling but thin dramatically at coarser sampling, and **shrank markedly from 2002–03 to 2011–12** as markets became more efficient. **VERDICT: at hourly resolution in 2019+, expect near-empty networks — deprioritize; cheap falsification test only (see dead ends).**
14. **Huth, N., Abergel, F. (2014). "High frequency lead/lag relationships — Empirical facts." *Journal of Empirical Finance* 26:41–58.** — Verified (ScienceDirect; arXiv:1111.7103). Lead-lag lives at **seconds-to-minutes** (tick data); liquid assets lead illiquid ones; effect concentrated around announcements. **VERDICT: confirms the horizon mismatch — kills hourly lead-lag as a priority. Registry: rejected (horizon).**

### D. Drawdown-constrained optimization (OR)
15. **Chekhlov, A., Uryasev, S., Zabarankin, M. (2005). "Drawdown Measure in Portfolio Optimization." *Int. J. Theoretical & Applied Finance* 8(1):13–58.** — Verified (WorldSci/SSRN 544742). Conditional Drawdown-at-Risk (CDaR): mean of worst (1−α) share of drawdowns; convex, **LP-formulable** on sample paths; contains MaxDD (α→1) and average drawdown (α=0) as limits. **VERDICT: our frontier constraint (MaxDD ≤ -11%) is literally this object — Idea 3 tests CDaR-constrained book allocation against inverse-vol.**
16. **Goldberg, L.R., Mahmoud, O. (2017). "Drawdown: from practice to theory and back again." *Mathematics and Financial Economics* 11(3):275–297.** — Verified (Springer, DOI 10.1007/s11579-016-0181-9; arXiv:1404.7493). Conditional Expected Drawdown (CED = tail mean of maxDD distribution): convex, linearly attributable to factors, and **far more sensitive to serial correlation of returns than vol or ES**. **VERDICT: the theoretical reason CDaR/CED allocation can beat inverse-vol — drawdown risk loads on loss autocorrelation, which variance ignores. Feeds Idea 3.**

### E. Regime switching, forecast combination, dispersion (econometrics)
17. **Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica* 57(2):357–384.** — Verified (JSTOR 1912559). Canonical Markov-switching model. **VERDICT: registry/context only — our regime-gate slot was already contested (jump model #42 lost to the DM panic gate); raw MS filters were dominated by jump models in published head-to-heads (ML sweep #4). Do not rebuild.**
18. **Bates, J.M., Granger, C.W.J. (1969). "The Combination of Forecasts." *Operational Research Quarterly* 20(4):451–468.** — Verified (Springer/JORS, DOI 10.1057/jors.1969.103). Inverse-past-error-weighted combinations of forecasts have lower MSE than either component — the founding OR result on forecast combination. **VERDICT: Idea 5 — combine vol forecasters instead of betting the overlay on one model.**
19. **Stivers, C., Sun, L. (2010). "Cross-Sectional Return Dispersion and Time Variation in Value and Momentum Premiums." *JFQA* 45(4):987–1014.** — Verified (Cambridge/SSRN 1064101). High recent cross-sectional return dispersion → **lower subsequent momentum premium** (and higher value premium); RD acts as a countercyclical leading state variable, robust to macro controls. **VERDICT: a *different* conditioning variable from our failed trend gates — cheap throttle test for Books A/F (Idea 4).**

**Registry cross-reference (already read, cited for synthesis only):** Moreira–Muir (#10 — the overlay this sweep upgrades), Daniel–Moskowitz panic gate (#27), Bun–Bouchaud–Potters RMT (#47 — market-mode is part of Book D's edge; informs Idea 2's "gate A/F, never D" rule), Barroso–Santa-Clara (#12), Cederburg et al. vol-managed critique (#40 — any new overlay variant must pass the same direct OOS comparison).

---

## 2. TOP 5 IDEAS (ranked by frontier improvement × implementability ÷ effort)

---

### IDEA 1 — HAR-family volatility forecast for the vol-target overlay ⭐ HIGH PRIORITY
- **Origin class:** 1 (direct implementation of a published, heavily replicated model family).
- **Sources:** Corsi (2009) [#1]; ABDL (2003) [#2]; HARQ [#3]; semivariance [#4]; leverage-HAR [#5]; Yang–Zhang [#7]. Registry: Moreira–Muir (#10), Cederburg (#40).
- **Mechanism:** The champion's single biggest win is the 15% vol-target overlay, currently driven by trailing 20d realized vol — an equal-weight backward window that (i) reacts a full window late after spikes, (ii) keeps exposure suppressed long after vol has decayed. HAR replaces it with a *forecast*: regress next-period log-RV on daily/weekly(5d)/monthly(22d) average log-RV. The cascade structure (long-horizon traders set the backdrop for short-horizon vol — Ghashghaie [#8], Corsi's heterogeneous-market hypothesis) captures both fast reaction and long memory. Variants: (B) HARQ — attenuate the daily lag when RV is noisily measured [#3]; (C) add downside-semivariance and leverage terms [#4, #5].
- **Economic rationale:** Vol timing works because vol is far more forecastable than returns and risk-taking capacity is slow-moving (Moreira–Muir). Any improvement in the vol forecast translates *mechanically* into better exposure timing — no new alpha source needed, no crowding channel. This is the rare upgrade whose transmission to the frontier is arithmetic, not hypothesis.
- **Signal spec (integrity-hardened):** Daily RV for the portfolio return series: 2019+ optionally from portfolio hourly returns; the universal path is daily proxies available 1997+ — squared daily return and Yang–Zhang range estimate on the portfolio/SPY. Regressions on log-vol; HAR coefficients estimated by OLS on an **expanding window refit monthly, using data through t−1 only**; forecast vol_hat(t) = sqrt(mean of h=1..21d forecasts) (horizon matched to the current 20d window); overlay weight w_t = min(1, 15%/vol_hat) applied at the **next day's open**. No full-sample estimation anywhere; 1997–2018 used only for coefficient-stability inspection, 2019–2026 is the comparison window against the incumbent 20d estimator (identical harness, per Cederburg's direct-comparison demand). Sanity guards: winsorize log-RV inputs at rolling 1st/99th percentiles; if HARQ's RQ term (from 7 hourly bars) is too noisy, drop variant B rather than aggregate-and-pray.
- **Data needed:** have everything (daily OHLC 1997+, hourly 2019+, existing portfolio return series).
- **Expected effect:** BPQ and the HAR literature report 10–30% forecast-error (QLIKE/MSE) reductions vs rolling historical vol; with the ~50% haircut, expect a modest but *direct* frontier gain — fewer days over-exposed into spikes (DD headroom) and fewer days under-exposed in recoveries (CAGR). Target: Sharpe +0.05–0.15 or CAGR +2–4pts at MaxDD ≤ -9%. Failure mode is a wash, not a blow-up — the overlay structure is unchanged.
- **Implementation sketch:** `portfolio/vol_forecast.py` (~120 LOC: RV builders incl. Yang–Zhang, HAR/LHAR/HARQ fits, expanding refit loop); a `vol_estimator` flag in the champion runner where 20d realized vol is currently computed; A/B/C/D: {20d (incumbent), HAR, HAR+leverage+semivar, HARQ}. ~1 day.

---

### IDEA 2 — Correlation-spike fragility gate (mean correlation / absorption ratio)
- **Origin class:** 3 (synthesis: Preis et al. stress-scaling + Kritzman et al. AR trading rule, adapted to our panel and applied only where it can't hurt Book D).
- **Sources:** [#9], [#12]; corroboration [#10], [#11]; registry: BBP (#47), Daniel–Moskowitz (#27).
- **Mechanism:** Compute daily, over our 523-name daily panel: (a) mean pairwise correlation over a trailing 60d window, and (b) absorption ratio = share of total variance in the top 10 eigenvectors of the trailing 250d correlation matrix. Signal = standardized shift (15d mean − 250d mean)/rolling std [Kritzman's form]. A spike says the market is compressing into one factor — portfolio diversification (our Sharpe 2.58 rests on near-zero cross-book correlation) is temporarily fictitious, and *portfolio* vol is about to exceed what the 20d estimator shows. Action on spike: tighten the vol-target cap (15%→10%) and/or halve Books A/F. **Never gates Book D** — registry #47 showed correlated sell-offs are precisely where D's edge lives.
- **Economic rationale:** Correlation rises with stress mechanically (single-factor panic flows, margin spirals, index-level hedging) and *precedes* realized portfolio-vol updates because it reprices co-movement before univariate vol windows catch up. It is a different measurement axis from everything in the champion today (which is univariate-vol- and drawdown-triggered).
- **Signal spec:** All windows rolling; eigendecomposition on returns through t−1 (one 523×523 decomposition per day — trivial); action at t open. Grid deliberately tiny to limit mining: signal ∈ {mean-corr z, AR z}, threshold ∈ {1σ, 2σ}, action ∈ {vol cap 10%, A/F ×0.5, both} = 12 cells; evaluate on 2019–2026 champion harness with 1997–2018 daily-book sanity run. Success = MaxDD improves ≥1pt with Sharpe give-up ≤0.05, creating DD budget to re-lever.
- **Data needed:** have (daily panel 1997+).
- **Expected effect:** No credit until tested (synthesis). Honest prior is mixed: our gates keep washing at champion level (#24, #42, #43) because ivol+VT already absorbs much of the stress response — but none of the tested gates measured *cross-sectional* compression. This is the cheapest untested axis with published stress-scaling evidence behind it.
- **Implementation sketch:** `portfolio/fragility.py` (~80 LOC) producing a daily signal series cached to parquet; wire as a second gate beside the panic gate in the champion runner. ~half day.

---

### IDEA 3 — CDaR-constrained book allocation (drawdown risk where drawdown is the constraint)
- **Origin class:** 1 (direct implementation: Chekhlov–Uryasev–Zabarankin LP), with Goldberg–Mahmoud as the mechanism explanation.
- **Sources:** [#15], [#16]; registry: inverse-vol/ERC/HRP results (#13–15), Kelly (#17).
- **Mechanism:** Replace (or A/B against) inverse-vol book weights with the solution of: maximize expected return subject to CDaR_α ≤ budget, weights in [0, 0.5], sum 1 — an LP on the trailing sample path (CUZ give the exact formulation). Rolling: solve on trailing 504d of daily book returns, α = 0.9 (mean of worst 10% of drawdowns), budget grid {4%, 6%, 8%}, re-solve monthly, weights applied next day, smoothed 50/50 with previous weights to cap turnover and LP instability. Keep the vol-target overlay and panic gate on top (also test overlay-off, since CDaR partially substitutes for it).
- **Economic rationale:** Our frontier is explicitly a drawdown constraint, yet our allocator optimizes a variance proxy. CED/CDaR load on the **serial correlation of losses** [#16] — a book whose losses cluster (F: -39.6% MaxDD from clustered momentum crashes) is far more drawdown-expensive per unit of vol than one whose losses are scattered (D: -6.4%). Inverse-vol cannot see this distinction; CDaR prices it directly.
- **Signal spec:** LP inputs are trailing returns through t−1 only; expected-return term = trailing 504d mean (or, more conservatively, maximize *minimum* book weight-stability — pure risk version) — flag: the mean-return term is the overfitting surface, so also run the return-agnostic variant (min CDaR s.t. fully invested), analogous to our ivol baseline. Chronological evaluation 2019–2026; embargo 21d between fit window end and each evaluation month (weights are monthly anyway).
- **Data needed:** book daily-return series (have).
- **Expected effect:** Published CDaR results show equal-return/lower-drawdown frontiers vs variance optimization; haircut + our history (every optimizer so far lost to simple weights, registry #13–16) → honest prior: most likely outcome is a wash with a genuinely interesting failure report; upside case is MaxDD -8.9% → -7% at similar CAGR, which converts to CAGR via re-levering the vol target. This is the strongest untested *allocator* idea remaining in the OR literature for our exact objective.
- **Implementation sketch:** `portfolio/cdar_alloc.py` (~150 LOC, scipy.optimize.linprog with the CUZ auxiliary-variable formulation); drop into the champion runner beside ivol. 1–2 days.

---

### IDEA 4 — Cross-sectional dispersion throttle on momentum books (A, F)
- **Origin class:** 1 (direct: Stivers–Sun), applied at book level.
- **Sources:** [#19]; registry: Daniel–Moskowitz (#27), failed trend gates (#24, #29, #43) — explicitly a *different* conditioning variable.
- **Mechanism:** RD_t = cross-sectional standard deviation of daily (and 20d) returns across the ~515-name universe; smooth with a 60d MA; z-score against a trailing 3yr window. Stivers–Sun: high RD regimes → momentum premium weak or negative (RD is a countercyclical state variable that spikes around market turning points — exactly where momentum crashes). Rule: RD z > 1 ⇒ Books A and F at 0.5× (grid {0.25, 0.5, 1.0}); D, C untouched.
- **Economic rationale:** Momentum profits come from continuation of a stable cross-sectional ordering; high dispersion marks re-sorting periods (leadership rotation, macro shocks) when the stale ranking is most likely wrong. Unlike a trend gate (level of past returns — tested, washed), RD measures the *stability of the cross-section itself*, which is the actual input to A/F's ranking step.
- **Signal spec:** RD computed from returns through t−1; z-scored with rolling stats only; act next open; monthly persistence requirement (signal must hold 5 consecutive days) to avoid whipsaw turnover. Long history available: validate the RD→momentum-premium relation on 1997–2018 daily data *first* (does high-RD predict low Book A returns in-sample?); only if the relation holds there, run the 2019–2026 champion A/B. This two-stage design spends the OOS window only once.
- **Data needed:** have (daily panel 1997+).
- **Expected effect:** Published relation is strong in 1962–2005 data; ~50% haircut + our repeated gate washes → prior: small. Ranked 4th because effort is ~half a day on existing gate plumbing and the 1997–2018 pre-test can kill it cheaply before touching the OOS window.
- **Implementation sketch:** add `dispersion_z` to the same fragility module as Idea 2 (shared plumbing); gate hook already exists in the champion runner.

---

### IDEA 5 — Bates–Granger forecast combination for the overlay vol estimator
- **Origin class:** 1 (direct: Bates–Granger 1969) + 3 (wrapper around Idea 1).
- **Sources:** [#18]; components from [#1], [#3], [#4], [#7].
- **Mechanism:** Instead of switching the overlay from the battle-tested 20d estimator to HAR outright, run the committee: {20d trailing RV, HAR (Idea 1), EWMA (λ=0.94), Yang–Zhang range vol}. Combined forecast = inverse-past-error weighted average, weights ∝ 1/(trailing 250d QLIKE loss of each forecaster), updated monthly, all losses computed on data through t−1.
- **Economic rationale:** The founding OR result on combination: differently mis-specified forecasters have partially independent errors, so the inverse-error combination has lower MSE than (almost always) any single member. For us it is model-risk insurance — vol-forecaster rankings are regime-dependent (HAR shines in persistent-vol regimes, simple windows in quiet ones), and the overlay is too important to bet on one specification chosen on 7 years of data.
- **Signal spec:** identical overlay plumbing to Idea 1; the only new object is the rolling loss ledger per forecaster. No tuned hyperparameters beyond the (fixed, conventional) 250d loss window — that is the point.
- **Data needed:** same as Idea 1.
- **Expected effect:** Combination ≈ best component with lower variance across sub-periods; adopt if it matches HAR's gains with better worst-year behavior. No separate credit — it inherits Idea 1's expectation.
- **Implementation sketch:** +40 LOC in `portfolio/vol_forecast.py`; free once Idea 1 exists. Test order: Idea 1 first, then this.

---

### Explicit dead ends this sweep (do not re-investigate without new evidence)
- **Hourly lead-lag networks** — Curme et al. [#13]: validated networks require ≤15-min sampling and shrank sharply 2002→2012; Huth–Abergel [#14]: the effect lives at seconds-to-minutes on tick data. Our hourly 2019+ panel is ~2 orders of magnitude too coarse, 8+ years further into the efficiency decay. If ever tested, it is a half-day falsification exercise, not a strategy candidate.
- **Markov-switching (Hamilton) regime gates** — regime-gate slot already lost twice in our stack (jump model #42 beat MS filters in the literature, then itself lost to the 11-day DM panic gate). The champion's remaining regime problem is not detection, it is that gating de-risks rallies.
- **Rough-volatility estimation (RFSV)** — supports the log-vol/HAR modeling choice [#6] but full fractional estimation adds nothing at 1–21d forecast horizons over HAR per the paper's own forecasting section.
- **Correlation-matrix state clustering (Münnix)** — strictly heavier version of Idea 2's scalars; revisit only if Idea 2's mean-corr/AR signals show life.
- **Turbulence-cascade formalism** — metaphor that motivated HAR; no additional testable content for our data [#8].

---

## 3. For the registry

| Paper (Author, Year, Journal) | Family | Verdict suggestion |
|---|---|---|
| Corsi (2009), J. Financial Econometrics | Vol forecasting/HAR | pending — Idea 1 (overlay estimator swap) |
| Andersen, Bollerslev, Diebold & Labys (2003), Econometrica | Vol forecasting/RV foundations | pending — enabler of Idea 1 (log-RV) |
| Bollerslev, Patton & Quaedvlieg (2016), J. Econometrics | Vol forecasting/HARQ | pending — Idea 1 variant B (RQ noise caveat) |
| Patton & Sheppard (2015), REStat | Vol forecasting/semivariance | pending — Idea 1 variant C |
| Corsi & Renò (2012), JBES | Vol forecasting/leverage | pending — Idea 1 variant C |
| Yang & Zhang (2000), J. Business | Vol estimation/range | pending — RV proxy for 1997+ daily OHLC; Idea 5 component |
| Gatheral, Jaisson & Rosenbaum (2018), Quant. Finance | Vol modeling/rough | rejected (context) — validates log-vol/HAR; RFSV not worth effort delta |
| Ghashghaie et al. (1996), Nature | Econophysics/cascade | rejected (context) — conceptual ancestor of HAR only |
| Preis, Kenett, Stanley, Helbing & Ben-Jacob (2012), Sci. Rep. | Correlation dynamics | pending — Idea 2 basis |
| Onnela, Chakraborti, Kaski, Kertész & Kanto (2003), Phys. Rev. E | Correlation dynamics/MST | rejected (context) — corroborates Idea 2; MST machinery unnecessary |
| Münnix et al. (2012), Sci. Rep. | Correlation dynamics/states | rejected (deferred) — heavier cousin of Idea 2 scalars |
| Kritzman, Li, Page & Rigobon (2011), JPM | Systemic risk/absorption ratio | pending — Idea 2 trading-rule form (practitioner-journal skepticism) |
| Curme, Tumminello, Mantegna, Stanley & Kenett (2015), Quant. Finance | Lead-lag networks | rejected — needs ≤15-min sampling; effect decayed post-2011 |
| Huth & Abergel (2014), J. Empirical Finance | Lead-lag/HF | rejected — seconds-to-minutes horizon, tick data required |
| Chekhlov, Uryasev & Zabarankin (2005), IJTAF | OR/drawdown optimization | pending — Idea 3 (CDaR LP allocator) |
| Goldberg & Mahmoud (2017), Math. Fin. Econ. | OR/drawdown theory | pending — Idea 3 mechanism (CED loads on loss autocorrelation) |
| Hamilton (1989), Econometrica | Regime switching | rejected — regime-gate slot resolved (see #42/#27); do not rebuild |
| Bates & Granger (1969), Operational Research Quarterly | Forecast combination | pending — Idea 5 (vol-forecast committee) |
| Stivers & Sun (2010), JFQA | Dispersion/momentum timing | pending — Idea 4 (two-stage: 1997–2018 pre-test first) |

**Suggested test order (effort-weighted):** Idea 1 → Idea 5 (free rider) → Idea 2 → Idea 4 (shares plumbing with 2; pre-test can kill it in-sample) → Idea 3.
