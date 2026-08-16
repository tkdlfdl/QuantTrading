# Portfolio Construction: 60-Year Literature Survey (1965–2025)

**Purpose:** Survey allocation methods for combining 4–6 systematic strategy books (daily return series available), and rank candidates to beat our current allocators:

| Allocator | Sharpe | CAGR | MaxDD |
|---|---|---|---|
| Fixed Equal Weight (benchmark to beat) | 2.060 | 48.0% | -19.0% |
| Momentum Allocation (20d lookback, 60d rebal, 50% cap) | 1.970 | — | -24.5% |

**Our setting:** N = 4–6 assets, ~7.5 years of daily history (~1,890 obs), pairwise correlations mostly 0.2–0.6 with one book near-zero to all, very heterogeneous book volatilities (standalone MaxDD ranges -6% to -65%). Target: Sharpe > 2.060, or CAGR > 48% with MaxDD ≤ ~-23%. Transaction cost standard: 0.1% per trade.

All citations below were verified via web search on 2026-08-15. None are fabricated; publication details confirmed against publisher/SSRN records.

---

## 1. Mean-Variance Optimization and Its Out-of-Sample Failure

**Citations (verified):**
- Markowitz, H. (1952). "Portfolio Selection." *The Journal of Finance*, 7(1), 77–91.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?" *The Review of Financial Studies*, 22(5), 1915–1953.

**Recipe (classical MV):** Estimate mean vector μ̂ and covariance Σ̂ from a trailing window (typically 60–120 months). Tangency weights `w ∝ Σ̂⁻¹ μ̂`, normalized to sum to 1; optionally long-only constrained. Rebalance monthly.

**Out-of-sample evidence:** DeMiguel et al. tested 14 optimization-based models (sample MV, Bayes-Stein, Black-Litterman-style, minimum variance, constrained variants, mixtures) across 7 empirical datasets. **None consistently beat 1/N** on Sharpe ratio, certainty-equivalent return, or turnover. They compute the estimation window needed for sample-based MV to beat 1/N in expectation: roughly **3,000 months of data for 25 assets** (and ~6,000 for 50). The gain from optimal diversification is more than offset by estimation error.

**Failure modes:** (a) Sample means are catastrophically noisy — MV weights are approximately `Σ⁻¹μ`, so mean errors are amplified by the inverse covariance; (b) extreme, unstable corner solutions; (c) huge turnover. Constrained (long-only) MV and minimum variance fare best among the 14 because they implicitly shrink.

**Relevance to us:** This is the null result our Fixed EW already exploits. With N=5 and 7.5 years, plain MV using sample means is nearly guaranteed to underperform Fixed EW after costs. **Do not implement raw MV.** However, our books (unlike the equity datasets in DGU) have persistently different Sharpe/vol profiles, so *covariance-only* and *shrunk* methods retain a real edge opportunity.

---

## 2. Minimum Variance

**Citation (verified):** Clarke, R., de Silva, H., & Thorley, S. (2006). "Minimum-Variance Portfolios in the U.S. Equity Market." *The Journal of Portfolio Management*, 33(1), 10–24.

**Recipe:** Drop expected returns entirely. `w = argmin w'Σ̂w` s.t. Σw=1 (and typically w≥0). Only the covariance matrix is needed. Clarke et al. used the 1,000 largest US stocks 1968–2005, monthly rebalance, with covariance estimated via Bayesian shrinkage / factor models to keep the problem well-conditioned. Closed form (unconstrained): `w ∝ Σ̂⁻¹ 1`.

**Out-of-sample evidence:** Over 1968–2005 the long-only minimum-variance portfolio delivered roughly **three-quarters of the market's volatility with comparable-or-better average returns**, i.e., a materially higher Sharpe than cap-weight — one of the founding results of the low-volatility literature.

**Failure modes:** Concentrates in the lowest-volatility assets (with N=5 it can put nearly everything in the lowest-vol book); ignores expected returns, so it sacrifices CAGR when high-vol assets have high returns — exactly our situation (the -65% MaxDD book is also the highest-CAGR book).

**Relevance to us:** Would likely crush our CAGR (over-allocating to the ~6%-CAGR low-vol book). Useful only as a *component* (e.g., the split rule inside HRP), not as the standalone allocator.

---

## 3. Risk Parity / Inverse Volatility / Equal Risk Contribution (ERC)

**Citations (verified):**
- Qian, E. (2005). "Risk Parity Portfolios: Efficient Portfolios Through True Diversification." PanAgora Asset Management white paper (coined the term "risk parity").
- Maillard, S., Roncalli, T., & Teiletche, J. (2010). "The Properties of Equally Weighted Risk Contribution Portfolios." *The Journal of Portfolio Management*, 36(4), 60–70.
- Asness, C., Frazzini, A., & Pedersen, L.H. (2012). "Leverage Aversion and Risk Parity." *Financial Analysts Journal*, 68(1), 47–59.

**Recipes:**
- **Naive risk parity / inverse volatility:** `w_i = σ̂_i⁻¹ / Σ_j σ̂_j⁻¹`, with σ̂_i the trailing volatility (e.g., 60-day rolling or EWMA λ=0.94). Ignores correlations. One parameter (the vol window).
- **ERC (full):** choose w such that every asset's risk contribution is equal: `w_i · (Σ̂w)_i = w_j · (Σ̂w)_j ∀ i,j`. No closed form for general Σ; solve numerically (e.g., minimize `Σ_{i,j} [w_i(Σ̂w)_i − w_j(Σ̂w)_j]²` s.t. Σw=1, w≥0, or cyclic coordinate descent). Maillard et al. prove ERC volatility lies **between** minimum variance and equal weight, and that under constant pairwise correlation **ERC reduces exactly to inverse-vol weights**.
- Rebalance monthly (weights drift slowly; daily rebalancing adds cost without benefit).

**Out-of-sample evidence:** Asness–Frazzini–Pedersen show a stocks/bonds risk-parity portfolio beat both the value-weighted market and 60/40 on Sharpe over **1926–2010 US data** (and in global samples), attributing the edge to leverage aversion making low-risk assets cheap per unit of risk. Maillard et al. show ERC beats EW on Sharpe in equity and multi-asset backtests with far lower turnover than MV.

**Failure modes:** (a) Equalizing risk ignores expected returns — if high-vol assets genuinely carry higher Sharpe, RP gives up return unless levered; (b) unlevered RP on assets with very different vols concentrates capital in low-vol sleeves; (c) vol estimates lag regime shifts.

**Relevance to us:** Directly attacks our biggest structural flaw: Fixed EW gives the -65%-MaxDD book the same *capital* but ~5–10× the *risk contribution* of the -6% book. Because our books' Sharpes are not proportional to their vols (the low-vol book D has the *highest* Sharpe, 2.7), shifting risk toward D should raise portfolio Sharpe. Restore CAGR with a vol-target overlay (§6). With N=5 and correlations 0–0.6, inverse-vol ≈ ERC to first order; ERC adds a small correlation-aware refinement (it will further trim the correlated momentum cluster A–F).

---

## 4. Maximum Diversification

**Citation (verified):** Choueifaty, Y., & Coignard, Y. (2008). "Toward Maximum Diversification." *The Journal of Portfolio Management*, 35(1), 40–51.

**Recipe:** Define the diversification ratio `DR(w) = (w'σ) / √(w'Σ̂w)` (weighted-average vol over portfolio vol). Most-Diversified Portfolio: `w = argmax DR(w)` s.t. Σw=1, w≥0. Equivalent to maximizing distance from the "all correlations = 1" case. Estimation: trailing covariance (they used 250-day windows), rebalance semi-annually/quarterly in the original.

**Out-of-sample evidence:** On US and Eurozone equities (1992–2008 era backtests) the MDP beat cap-weight, EW, and minimum variance on Sharpe, with lower drawdowns than cap-weight.

**Failure modes:** Like min-var it is expected-return-blind and can concentrate; with few assets the argmax is sensitive to correlation estimation error; solutions can flip discontinuously when correlations move.

**Relevance to us:** Would overweight the two near-zero-correlation books (C, B) regardless of their Sharpe — book C is our *most fragile* strategy and B our lowest-CAGR. Moderate promise at best; dominated by ERC for our use case. Not in top 5.

---

## 5. Hierarchical Risk Parity (HRP) and Clustering-Based Allocation

**Citation (verified):** López de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out of Sample." *The Journal of Portfolio Management*, 42(4), 59–69.

**Recipe (exact, three stages):**
1. **Tree clustering:** compute correlation matrix ρ̂ from trailing returns; distance `d_ij = √(½(1−ρ̂_ij))`; single-linkage hierarchical clustering on d.
2. **Quasi-diagonalization:** reorder assets so correlated ones sit adjacent (follow the dendrogram order).
3. **Recursive bisection:** start with w_i=1 for all; recursively split the ordered list into halves; for each half compute cluster variance `V_c = w̃'Σ̂_c w̃` with intra-cluster inverse-variance weights `w̃ ∝ diag(Σ̂_c)⁻¹`; allocate between the two halves by `α = 1 − V_1/(V_1+V_2)`; multiply weights down the tree.
- No matrix inversion, works even when Σ̂ is singular/ill-conditioned. Rebalance monthly with the trailing 6–12-month covariance.

**Out-of-sample evidence:** Monte Carlo experiments in the paper: HRP delivers **lower out-of-sample variance than CLA (mean-variance/min-var)** — even though minimum variance is CLA's in-sample objective — and lower realized risk than naive inverse-variance risk parity (reported OOS variance reduction vs CLA on the order of ~30%; treat exact figure as paper-specific to his simulation design).

**Failure modes:** Single-linkage chaining can produce unbalanced trees; the bisection ignores expected returns; with N as small as 5 the "hierarchy" is shallow (one or two splits), so HRP ≈ cluster-then-inverse-variance.

**Relevance to us:** Our correlation structure is *exactly* the block pattern HRP is built for: cluster {A, F, D} (momentum/hourly, ρ 0.2–0.5) vs. singletons {C}, {B}. HRP would first split risk between the cluster and the independents, then inverse-variance within — automatically preventing the momentum cluster from dominating, which plain inverse-vol does not fully fix. With N=5 you can even hard-code the tree (cluster known ex ante), removing the clustering estimation noise entirely.

---

## 6. Portfolio-Level Volatility Targeting

**Citations (verified):**
- Moreira, A., & Muir, T. (2017). "Volatility-Managed Portfolios." *The Journal of Finance*, 72(4), 1611–1644.
- Harvey, C.R., Hoyle, E., Korgaonkar, R., Rattray, S., Sargaison, M., & van Hemert, O. (2018). "The Impact of Volatility Targeting." *The Journal of Portfolio Management*, 45(1), 14–33.

**Recipe:**
- Moreira–Muir: scale the risky portfolio each month by `f_t = c / σ̂²_t` where σ̂²_t is *previous-month realized variance* of the portfolio (c set so the managed series has the same unconditional vol as the original). Variance scaling, monthly.
- Harvey et al.: scale exposure by `L_t = σ_target / σ̂_t` (vol scaling, not variance), σ̂_t from short-window realized/EWMA vol, applied daily with a leverage cap; tested on 60+ assets, daily data 1926–2017.

**Out-of-sample evidence:** Moreira–Muir: variance-managed versions of the market, value, momentum, profitability, and carry factors earn **large positive alphas relative to their unmanaged versions and higher Sharpe ratios**; the strategy takes *less* risk in recessions (contradicting the risk-return tradeoff intuition). Harvey et al.: vol targeting **raises Sharpe for "risk assets" (equities, credit) but not for bonds, FX, commodities**; crucially, for *all* assets it **cuts the left tail and reduces drawdowns**, because vol is persistent (vol clustering) and large losses arrive in high-vol states. The 2020 COVID crash out-of-sample validated the mechanism.

**Failure modes:** Whipsaw in V-shaped recoveries (de-levered at the bottom, misses the rebound); requires leverage in calm regimes to hit the target (cap it); adds turnover (mitigate with a rebalancing band, e.g., trade only when |L_t/L_current − 1| > 10%).

**Relevance to us:** This is an **overlay, composable with any weighting scheme**, and the only method on this list that can *raise CAGR and cut MaxDD simultaneously* if Sharpe is preserved. Our combined book behaves like a "risk asset" (equity-strategy P&L, vol clustering around events like 2020, 2022), i.e., the asset class where Harvey et al. find Sharpe *improves*. A vol-targeted EW or vol-targeted ERC at σ_target = current EW realized vol (~20–23% ann.) with L_max = 1.5 is the single highest-probability improvement available.

---

## 7. Kelly Criterion / Growth-Optimal and Fractional Kelly

**Citations (verified):**
- Kelly, J.L. Jr. (1956). "A New Interpretation of Information Rate." *Bell System Technical Journal*, 35, 917–926.
- Thorp, E.O. (2006). "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market." In *Handbook of Asset and Liability Management*, Vol. 1, North-Holland, 385–428.
- MacLean, L.C., Thorp, E.O., & Ziemba, W.T., eds. (2011). *The Kelly Capital Growth Investment Criterion: Theory and Practice.* World Scientific.

**Recipe:** Maximize expected log wealth. Continuous-time/Gaussian approximation for multiple assets: unconstrained growth-optimal weights `f* = Σ̂⁻¹ μ̂` (fractions of wealth; can exceed 1 = leverage). **Fractional Kelly:** `f = λ·f*` with λ ∈ [0.25, 0.5], holding the remainder in cash — the standard practitioner risk control because full Kelly is brutally volatile (half-Kelly gives ~75% of the growth rate with ~half the variance, and drawdown probabilities shrink dramatically; full Kelly has P(halving wealth) ≈ 1/2 in the idealized setting).

**Out-of-sample evidence:** MacLean–Thorp–Ziemba compile theory + case studies (Thorp's Princeton-Newport, sports betting syndicates): full Kelly dominates long-run growth but exhibits violent interim drawdowns; fractional Kelly traces a clean growth-security tradeoff. This is the only framework here that *explicitly optimizes CAGR* rather than Sharpe.

**Failure modes:** `Σ̂⁻¹μ̂` inherits ALL the estimation fragility of mean-variance (§1) — Kelly is MV with a specific risk aversion. Overbetting (λ too high, or μ̂ too optimistic) is catastrophic and asymmetric: betting 2× Kelly gives zero growth with high variance. Backtested book μ̂'s are inflated (selection bias), making raw Kelly systematically overbet.

**Relevance to us:** High-upside, high-risk candidate for the "CAGR > 48%" branch of our objective. Must be paired with heavy shrinkage of μ̂ (§8), λ ≤ 0.5, long-only, weight caps, and total leverage ≤ 1.5. Ranked, but below the covariance-only methods on probability of beating EW's Sharpe.

---

## 8. Bayesian / Shrinkage Approaches

**Citations (verified):**
- Black, F., & Litterman, R. (1992). "Global Portfolio Optimization." *Financial Analysts Journal*, 48(5), 28–43.
- Ledoit, O., & Wolf, M. (2004). "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices." *Journal of Multivariate Analysis*, 88(2), 365–411.
- Ledoit, O., & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix." *The Journal of Portfolio Management*, 30(4), 110–119.

**Recipes:**
- **Black–Litterman:** start from equilibrium (reverse-optimized) returns `Π = δΣw_mkt`, blend with investor views via Bayesian updating: `μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹Π + P'Ω⁻¹Q]`; feed μ_BL into MV. Designed to fix MV's corner solutions by anchoring to a sensible prior.
- **Ledoit–Wolf shrinkage:** `Σ̂_LW = δ*F + (1−δ*)S` — convex combination of the sample covariance S with a structured target F (scaled identity in the JMA paper; constant-correlation in the JPM paper), with the optimal shrinkage intensity δ* estimated analytically from the data. Distribution-free, closed-form, guarantees a well-conditioned invertible Σ̂.

**Out-of-sample evidence:** Ledoit–Wolf show shrinkage reduces out-of-sample tracking error and improves realized information ratios of optimized equity portfolios vs. sample-covariance versions. In DeMiguel et al.'s horse race, shrinkage-based and constrained models were the *closest* to 1/N (still not consistently better on their datasets).

**Failure modes:** BL requires a meaningful "market equilibrium" prior — undefined for proprietary strategy books (there is no market-cap weight for Book D); its value collapses to "shrink toward your prior," which for us just means shrink toward EW or ERC. LW shrinkage matters most when N is large relative to T; with N=5, T≈1,890 the sample covariance is already decent — LW is cheap insurance, not an edge by itself.

**Relevance to us:** Not a standalone allocator — an **ingredient**. Use LW (constant-correlation target) covariance inside ERC/HRP/Kelly, and shrink Kelly means toward the cross-book grand mean (a James-Stein/BL-flavored prior: "all books have the same true Sharpe until proven otherwise").

---

## 9. Regime / Trend Conditioning of Allocations

**Citation (verified):** Faber, M.T. (2007). "A Quantitative Approach to Tactical Asset Allocation." *The Journal of Wealth Management*, 9(4), 69–79. (SSRN #962461; the most-downloaded timing paper on SSRN.)

**Recipe (the simple classic):** For each asset: hold if price > 10-month simple moving average, else move that sleeve to cash. Evaluated monthly (month-end only). Applied to a 5-asset-class portfolio (US/foreign equities, bonds, commodities, REITs), 1973 onward.

**Out-of-sample evidence:** In-sample on US equities (back to 1901 in updates), out-of-sample across 20+ markets: the timing model historically delivered **roughly equity-like returns with bond-like volatility** — similar CAGR to buy-and-hold with volatility and maximum drawdown cut approximately in half, and it sidestepped 2008 in real time (published pre-crisis). The mechanism (return autocorrelation at monthly horizon + vol clustering) overlaps with vol-targeting.

**Adaptation to strategy books:** apply the trend filter to each book's *equity curve*: `w_i,t = 0 (or ½·w_i,base) if NAV_i < SMA_k(NAV_i)`, k ≈ 100–200 trading days, freed weight to cash or pro-rata to surviving books. This is "allocate away from books in drawdown."

**Failure modes:** Whipsaw cost in choppy sideways curves (each false signal costs 2×10bp per book); a book with mean-reverting P&L (our Book C: flat then spikes) gets systematically cut right before its payoff — trend filters on strategy equity curves work best on books whose P&L itself trends (A, F). Our own MomAlloc result (Sharpe 1.97 < 2.06 EW) is direct in-house evidence that naive return-chasing across books *hurts* — a caution against aggressive regime tilts.

**Relevance to us:** Use defensively and mildly (halve, don't zero; long SMA), if at all. Vol targeting (§6) captures most of the same drawdown protection with fewer parameters and no per-book whipsaw.

---

## 10. Rebalancing Frequency and Transaction-Cost-Aware Allocation

**Citation (verified):** Donohue, C., & Yip, K. (2003). "Optimal Portfolio Rebalancing with Transaction Costs." *The Journal of Portfolio Management*, 29(4), 49–63.

**Recipe / findings:** With proportional transaction costs the optimal policy is a **no-trade region** around target weights: trade only when weights drift outside a band, and trade **back to the band edge, not to the target**. Calendar rebalancing (monthly/quarterly) is a decent heuristic; tolerance-band rebalancing (e.g., trade when |w_i − w_i*| > 20% relative) dominates fixed-calendar at equal cost. DeMiguel et al. (2009, §1) also show turnover is where optimization-based rules lose most of their paper edge — 1/N's low turnover is a genuine component of its outperformance.

**Failure modes of ignoring this:** daily-rebalanced ERC/vol-targeting on 5 books at 10bp/trade can easily burn 1–3% CAGR.

**Relevance to us (concrete policy):** monthly weight refresh for any allocator below; for the vol-target overlay, adjust exposure only when the desired leverage differs from current by >10% (band), which typically means trading a few times a month in calm regimes and daily in crises — exactly when it pays.

---

---

# TOP 5 RANKED FOR OUR SETUP

Ranking criterion: probability of beating Fixed EW (Sharpe 2.060, CAGR 48.0%, MaxDD -19.0%) given N=5, T=7.5yr, ρ ∈ [0, 0.6] with one block {A,F,D} and two independents {C, B}, and extreme vol dispersion across books. Estimation-light methods first; mean-dependent methods last.

Common notation: `r_i,t` = daily return of book i; `σ̂_i,t` = EWMA vol, `σ̂²_i,t = λσ̂²_i,t−1 + (1−λ)r²_i,t`, λ = 0.94 (RiskMetrics), annualized ×√252. `Σ̂_t` = Ledoit–Wolf constant-correlation shrunk covariance on trailing 252 days. All weights long-only, refreshed **monthly** (§10) unless stated; 10bp cost per trade assumed in any backtest.

### #1 — Volatility-Targeted Equal Weight (overlay on the incumbent)
*Basis: Moreira–Muir 2017 JF; Harvey et al. 2018 JPM (§6).*
Keep Fixed EW book weights. Scale total exposure:
```
σ̂_p,t  = EWMA(λ=0.94) vol of the EW portfolio's daily returns, annualized
L_t     = clip( σ_target / σ̂_p,t ,  L_min=0.5 ,  L_max=1.5 )
w_i,t   = L_t / N_active,t          (remainder in cash; if L_t>1, leverage)
```
Set `σ_target` = full-sample realized vol of Fixed EW (so unconditional risk matches and comparisons are honest). Trade the overlay only when `|L_t/L_current − 1| > 0.10`.
**Why #1:** zero cross-sectional estimation (nothing to get wrong about which book is best); attacks MaxDD directly through the vol-clustering channel Harvey et al. validate for risk assets; the only candidate that plausibly raises CAGR *and* cuts MaxDD at once. Preserves everything that makes EW win.
**Risk:** V-shaped recovery whipsaw; leverage cap limits upside capture in calm bull years.

### #2 — Inverse-Volatility Weights (naive risk parity), optionally vol-targeted
*Basis: Qian 2005; Maillard–Roncalli–Teiletche 2010 (constant-ρ ⇒ ERC = inverse vol) (§3).*
```
w_i,t = (1/σ̂_i,t) / Σ_j (1/σ̂_j,t),   then cap w_i ≤ 0.40 and renormalize
```
σ̂ from trailing 60-day vol or EWMA. Monthly refresh. Then apply the #1 overlay on top (`L_t = σ_target/σ̂_p,t`, L_max = 1.5–2.0 since the unlevered IV portfolio is low-vol).
**Why #2:** one parameter; directly fixes EW's structural flaw — book A currently contributes ~5–10× the risk of book D per dollar despite a *lower* Sharpe. Shifting risk budget toward the high-Sharpe low-vol book (D, Sharpe 2.7) should raise portfolio Sharpe almost mechanically; the vol-target overlay restores the CAGR that de-risking A gives up.
**Risk:** unlevered version alone will land below 48% CAGR — the overlay is not optional if CAGR matters.

### #3 — ERC with Ledoit–Wolf covariance (correlation-aware refinement of #2)
*Basis: Maillard–Roncalli–Teiletche 2010; Ledoit–Wolf 2004 (§3, §8).*
```
Σ̂_t   = LW constant-correlation shrinkage on trailing 252d daily returns
w_t    = argmin_{w≥0, Σw=1}  Σ_{i<j} [ w_i(Σ̂_t w)_i − w_j(Σ̂_t w)_j ]²
```
(Solve with scipy SLSQP or cyclic coordinate descent; warm-start from inverse-vol.) Cap 40%, monthly, plus #1 overlay.
**Why #3:** with the {A,F,D} correlation block, ERC additionally trims the momentum cluster relative to inverse-vol, pushing risk toward the independent books — a genuine (small) improvement over #2 that costs only covariance estimation, which is reliable at N=5, T=1,890.
**Risk:** marginal gain over #2 may be within noise; slightly higher turnover.

### #4 — Hierarchical Risk Parity with a fixed (known) tree
*Basis: López de Prado 2016 JPM (§5).*
With N=5, skip the noisy clustering step and hard-code the dendrogram from our known correlation structure: `((A,F),D)` vs `C` vs `B`. Then recursive bisection:
```
At each split: α = 1 − V_left/(V_left + V_right),  V_c = w̃'Σ̂_c w̃,  w̃ ∝ 1/diag(Σ̂_c)
Allocate α to left branch, 1−α to right; within final clusters use inverse variance.
```
Monthly, LW covariance, plus #1 overlay.
**Why #4:** structurally the right answer for block-correlated books — it budgets risk *between clusters first*, so the three correlated momentum-flavored books can't jointly crowd out the diversifiers the way they do under EW and even inverse-vol. Fixing the tree removes HRP's main small-N weakness.
**Risk:** with only ~3 effective clusters, output will resemble #3; ranked below ERC only because the recursive-bisection allocation is ad hoc rather than optimal for a known structure.

### #5 — Fractional Kelly with double shrinkage (the CAGR play)
*Basis: Kelly 1956; Thorp 2006; MacLean–Thorp–Ziemba 2011; Ledoit–Wolf 2004; James–Stein/BL logic (§7, §8).*
```
μ̂_i,t  = δ·μ̄_t + (1−δ)·mean(r_i, trailing 252d)·252,   δ = 0.5,  μ̄ = cross-book average
f_t     = λ · Σ̂_LW,t⁻¹ μ̂_t,        λ = 0.25 (quarter-Kelly)
w_t     = clip(f_t, 0, 0.40);  if Σw > 1.5 rescale to Σw = 1.5;  if Σw < 1 hold cash or scale up to 1
```
Monthly refresh. No vol-overlay needed (Kelly already sizes with variance).
**Why #5:** the only candidate built to maximize *growth*, so it is the best shot at the "CAGR > 48%" branch; quarter-Kelly + 50% mean shrinkage + caps neutralize most of the overbetting catastrophe mode; at δ=1 it degenerates gracefully into #3-like covariance-only weights (nice fail-safe).
**Risk:** highest estimation burden of the five — trailing 252d means are noisy and our MomAlloc experience (return-chasing → Sharpe 1.97 < 2.06) is a live warning; MaxDD will likely exceed EW's -19% (budget for the ≤ -23% ceiling). Test δ ∈ {0.5, 0.75, 1.0}, λ ∈ {0.2, 0.25, 0.33} and demand robustness across the grid before trusting any cell.

### Explicitly NOT recommended for our setup
- **Raw mean-variance / tangency** (§1): DeMiguel et al.'s ~3,000-month data requirement vs. our 90 months.
- **Standalone minimum variance** (§2) and **maximum diversification** (§4 section): return-blind; would over-allocate to our lowest-CAGR and most-fragile books respectively.
- **Aggressive trend/regime switching across books** (§9): our own MomAlloc underperformance is in-sample evidence; if used at all, only as a mild defensive halving with a 150–200d SMA on book NAV.

### Suggested test protocol
Backtest all five on the books' daily return series 2019–2026 (books enter only when live per CLAUDE.md rule 3), 10bp per trade, monthly rebalance with the §10 no-trade bands, and report Sharpe / CAGR / MaxDD / turnover vs Fixed EW — plus a 2-year holdout or expanding-window walk-forward for #5, which is the only estimation-heavy candidate.

---

## Citation Verification Log (2026-08-15)

All verified via web search against publisher pages / SSRN / archives:

| # | Citation | Verified against |
|---|---|---|
| 1 | Markowitz 1952, JF 7(1):77–91 | Wiley Online Library |
| 2 | DeMiguel, Garlappi & Uppal 2009, RFS 22(5):1915–1953 | Oxford Academic |
| 3 | Clarke, de Silva & Thorley 2006, JPM 33(1):10–24 | jpm.pm-research.com |
| 4 | Qian 2005, PanAgora white paper | panagora.com (PDF live) |
| 5 | Maillard, Roncalli & Teiletche 2010, JPM 36(4):60–70 | SSRN #1271972, DOI 10.3905/jpm.2010.36.4.060 |
| 6 | Asness, Frazzini & Pedersen 2012, FAJ 68(1):47–59 | CFA Institute / AQR |
| 7 | Choueifaty & Coignard 2008, JPM 35(1):40–51 | jpm.pm-research.com |
| 8 | López de Prado 2016, JPM 42(4):59–69 | jpm.pm-research.com, SSRN #2708678 |
| 9 | Moreira & Muir 2017, JF 72(4):1611–1644 | Wiley Online Library |
| 10 | Harvey et al. 2018, JPM 45(1):14–33 | jpm.pm-research.com, SSRN #3175538 |
| 11 | Kelly 1956, BSTJ 35:917–926 | Wiley / Internet Archive |
| 12 | Thorp 2006, Handbook of ALM 1:385–428 | Publisher listing / gwern archive |
| 13 | MacLean, Thorp & Ziemba 2011, World Scientific | worldscientific.com |
| 14 | Black & Litterman 1992, FAJ 48(5):28–43 | CFA Institute / Taylor & Francis |
| 15 | Ledoit & Wolf 2004, JMA 88(2):365–411 | JMA PDF (ENS Lyon mirror) |
| 16 | Ledoit & Wolf 2004, JPM 30(4):110–119 | SSRN #433840 / ledoit.net |
| 17 | Faber 2007, JWM 9(4):69–79 | jwm.pm-research.com, SSRN #962461 |
| 18 | Donohue & Yip 2003, JPM 29(4):49–63 | jpm.pm-research.com |

No unverifiable citations. One quantitative claim flagged as approximate in-text: the ~30% OOS variance reduction of HRP vs CLA is specific to López de Prado's Monte Carlo design.
