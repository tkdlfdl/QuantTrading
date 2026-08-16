# 60-Year Academic Survey of Trading Strategies (1965–2025)

**Purpose:** Registry of seminal + best-modern papers per strategy family, filtered for implementability on our stack.
**Our data constraint:** US equities (S&P 500 + NASDAQ 100, ~515 names), price/volume ONLY — daily bars 1997+, hourly bars 2019+, plus ^VIX and UVXY. No fundamentals, options, futures, short-interest, or analyst data.
**Our honest baselines (targets to beat):**

| Book | Strategy | Sharpe | MaxDD |
|------|----------|--------|-------|
| D | Contrarian mean-reversion (hourly) | 2.74 | -6.4% |
| F | Hourly momentum | 1.55 | -39.8% |
| A | Daily momentum + UVXY hedge | 1.30 | -65% |
| Combined | A+F+D+C+B | 2.06 | -19% |

**Implementability tags:**
- `IMPLEMENTABLE-DAILY` — buildable from daily price/volume bars (1997+)
- `IMPLEMENTABLE-HOURLY` — needs intraday bars (2019+)
- `OVERLAY` — applies to any of our existing return series (position sizing / gating rule)
- `UNTESTABLE` — requires data we do not have (listed for the registry only)

**Citation status:** every paper below was verified to exist via web search (journal, volume, pages) unless explicitly marked otherwise. Specific point estimates that could not be independently re-verified in this pass are flagged "(paper figure, not re-verified)".

---

## Meta-result first: post-publication decay (read before trusting ANY number below)

**McLean, R. David & Pontiff, Jeffrey (2016), "Does Academic Research Destroy Stock Return Predictability?", *Journal of Finance* 71(1), 5–32.** *(Note: JF, not JFE.)*
- Sample: 97 published cross-sectional return predictors, replicated through 2013.
- Findings: predictor portfolio returns are **26% lower out-of-sample** (upper bound on data-mining) and **58% lower post-publication**. The 32-point difference is attributed to publication-informed arbitrage. Decay is larger for predictors with higher in-sample returns and lower arbitrage costs (i.e., large liquid stocks — exactly our universe).
- **Practical rule for this project:** haircut every published Sharpe/alpha below by ~40–60% before forming expectations, and more for large-cap-only implementations. Anything that survives that haircut AND a real backtest on our data is a candidate.

---

## Family 1 — Cross-sectional momentum + crash fixes

### Seminal
**Jegadeesh, Narasimhan & Titman, Sheridan (1993), "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency", *Journal of Finance* 48(1), 65–91.**
- Sample: NYSE/AMEX, 1965–1989.
- Signal: rank stocks on past J-month return (J = 3,6,9,12), long top decile / short bottom decile, hold K months (K = 3,6,9,12), overlapping portfolios. Standard modern variant is "12-1" (skip most recent month).
- Performance: ~0.9–1.3%/month for winner-minus-loser across the 16 (J,K) combos; best cell (J=12, K=3) ≈ 1.31%/month (≈12%/yr). Not explained by systematic risk.

**Carhart, Mark M. (1997), "On Persistence in Mutual Fund Performance", *Journal of Finance* 52(1), 57–82.**
- Canonized momentum as the fourth factor (PR1YR/UMD) alongside Fama-French 3. Mutual fund persistence 1962–1993 is explained by momentum exposure + expenses, not skill. Relevance here: momentum is a priced, persistent factor, not a fund-manager artifact.

### Crash fixes (most relevant to us — Book F MaxDD -39.8%, Book A -65%)
**Barroso, Pedro & Santa-Clara, Pedro (2015), "Momentum Has Its Moments", *Journal of Financial Economics* 116(1), 111–120.**
- Sample: US 1927–2011 (+ international).
- Signal: momentum risk is highly persistent and forecastable from the strategy's OWN realized volatility. Scale the WML portfolio to a **constant volatility target**: weight w_t = σ_target / σ̂_t, where σ̂_t = realized vol of daily WML returns over the previous **6 months (126 days)**, σ_target = 12% annualized.
- Performance: risk management "virtually eliminates crashes and nearly doubles the Sharpe ratio" — Sharpe 0.53 → 0.97; skewness −2.47 → −0.42 and excess kurtosis 18.24 → 2.68 (paper figures, not re-verified). Crucially, the scaling uses only the strategy's own trailing returns — zero new data required.
- **Tag: OVERLAY** (directly applicable to Book F and Book A daily return series).

**Daniel, Kent & Moskowitz, Tobias J. (2016), "Momentum Crashes", *Journal of Financial Economics* 122(2), 221–247.**
- Sample: US 1927–2013 + international + other asset classes.
- Findings: momentum crashes are partly forecastable. They occur in **"panic" states — following market declines and when market volatility is high — and coincide with market rebounds** (the short-loser leg behaves like a written call option on the rebounding market; long-only momentum suffers a milder version via loser-heavy rallies). Fix: a dynamic strategy scaling exposure by forecast conditional mean/variance of momentum, w_t ∝ μ̂_t / σ̂_t²; this roughly doubles the Sharpe of static momentum (≈0.62 → ≈1.19, paper figures, not re-verified).
- Bear-state indicator is computable from price data alone: trailing 24-month market return < 0, plus elevated recent realized vol.
- **Tag: OVERLAY** (gate/scale Books F and A).

### Related modern refinement
**Novy-Marx (2012, JFE, "Is momentum really momentum?")** — intermediate-horizon past returns (12–7 months back) drive momentum more than recent 6 months (not independently verified in this pass; secondary). Cheap robustness test for Book F lookback choice.

### Decay
Momentum spread returns weakened post-2000 in US large caps (Carhart-factor UMD had crashes 2002, 2009, 2016, 2020) but volatility-managed versions held up in BS15's own out-of-sample and international samples. Apply the McLean-Pontiff ~58% haircut to raw WML; the *risk-management overlays* are structural (variance forecastability), which decays far less than return forecastability.

---

## Family 2 — Time-series momentum (TSMOM)

**Moskowitz, Tobias J., Ooi, Yao Hua & Pedersen, Lasse Heje (2012), "Time Series Momentum", *Journal of Financial Economics* 104(2), 228–250.**
- Sample: 58 liquid futures (equity indices, FX, commodities, bonds), 1965–2009.
- Signal: for each instrument, go long if its **own past 12-month excess return > 0**, else short; hold 1 month; size each position to 40% annualized ex-ante vol (EWMA). Persistence for 1–12 months, partial reversal beyond.
- Performance: diversified TSMOM portfolio earns substantial abnormal returns (Sharpe > 1 diversified across all 58; per-instrument much lower), performs best in extreme markets ("crisis alpha").

**Critique — Huang, Dashan, Li, Jiangyuan, Wang, Liyao & Zhou, Guofu (2020), "Time Series Momentum: Is It There?", *Journal of Financial Economics* 135(3), 774–794.**
- Asset-by-asset time-series regressions show little evidence of TSM in- or out-of-sample; the pooled t-stat is unreliable; TSMOM strategy profits are statistically indistinguishable from a strategy using the historical mean (i.e., much of TSMOM = long-run risk premium + vol scaling, not trend). Implication: use TSMOM as a *risk gate*, not as a return source.
- Applied to our stack: single-instrument TSMOM on SPY/QQQ (12-month sign) is the classic "trend filter" that would have exited equities mid-2008 and early 2022. As a gate on momentum books it overlaps heavily with Daniel-Moskowitz's bear indicator.
- **Tag: IMPLEMENTABLE-DAILY (SPY/QQQ trend gate) / OVERLAY.**

---

## Family 3 — Reversal (long-term and short-term)

### Long-term reversal
**De Bondt, Werner F.M. & Thaler, Richard (1985), "Does the Stock Market Overreact?", *Journal of Finance* 40(3), 793–805.**
- Sample: CRSP monthly, 1926–1982.
- Signal: form portfolios on prior **36-month** cumulative returns; long extreme losers, short extreme winners; hold 36 months.
- Performance: loser portfolios outperform winner portfolios by ≈24.6% cumulative over the following 36 months, concentrated in Januarys (paper figure, not re-verified — search confirmed direction and significance but not the exact number).
- Post-publication: largely subsumed by size/value loadings in later work (Fama-French 1996); weak in large caps. Low priority for us.
- **Tag: IMPLEMENTABLE-DAILY** (but expected weak in our large-cap universe).

### Short-term reversal (our Book D's academic ancestry)
**Jegadeesh, Narasimhan (1990), "Evidence of Predictable Behavior of Security Returns", *Journal of Finance* 45(3), 881–898.**
- Sample: 1934–1987, monthly. Signal: 1-month return reversal (buy last month's losers, sell winners). Extreme-decile spread ≈ **2.49%/month**; also documents strong 12-month positive serial correlation (momentum seed).

**Lehmann, Bruce N. (1990), "Fads, Martingales, and Market Efficiency", *Quarterly Journal of Economics* 105(1), 1–28.**
- Sample: 1962–1986, weekly. Signal: prior-week winners/losers reverse the following week; profits claimed to survive bid-ask and plausible costs at the time.
- Decay: raw daily/weekly reversal in large caps was arbitraged hard by stat-arb desks from the 1990s; post-2000 it survives mainly (a) after cost-aware construction, (b) conditionally in stressed markets — see Nagel.

**Nagel, Stefan (2012), "Evaporating Liquidity", *Review of Financial Studies* 25(7), 2005–2039.**
- Reinterprets short-term reversal returns as **compensation for liquidity provision**. Expected reversal returns and conditional Sharpe ratios are **strongly increasing in VIX** — liquidity provision pays enormously in turmoil (2007–09), and even industry-portfolio reversal (unprofitable unconditionally) becomes highly profitable when VIX is high.
- **This is the academic explanation of why our Book D printed +46% in 2022.** It implies Book D's edge is state-dependent and can be *scaled with VIX* — and that Book D is a natural hedge for momentum's panic-state crashes (DM16: momentum dies when vol is high; Nagel: reversal thrives then).
- **Tag: OVERLAY (VIX-conditioned sizing of Book D). Directly implementable — we have ^VIX.**

**Medhat, Mamdouh & Schmeling, Maik (2022), "Short-term Momentum", *Review of Financial Studies* 35(3), 1480–1526.**
- Sample: NYSE/AMEX/Nasdaq nonfinancials, July 1963–December 2018 (+22 developed markets).
- Signal: double-sort on prior-month return × prior-month **share turnover** (volume/shares outstanding; relative volume works as a proxy). **Low-turnover stocks reverse; high-turnover stocks show short-term MOMENTUM** ("short-term momentum"), which is as profitable and persistent as conventional momentum, survives transaction costs, and is **strongest among the largest, most liquid stocks** — i.e., our exact universe.
- **Tag: IMPLEMENTABLE-DAILY** (we have price + volume; shares outstanding not needed if we use turnover proxies like volume/avg-volume). High priority: it refines both our reversal (Book D: exclude high-turnover names from fade candidates) and suggests a new signal.

---

## Family 4 — Low-volatility / Betting-against-beta

**Black, Fischer, Jensen, Michael C. & Scholes, Myron (1972), "The Capital Asset Pricing Model: Some Empirical Tests", in Jensen (ed.), *Studies in the Theory of Capital Markets*, Praeger.**
- 1926–1966: the empirical security market line is far flatter than CAPM predicts — low-beta stocks earn positive alpha. (Haugen & Heins mid-1970s work reached similar conclusions for volatility.)

**Baker, Malcolm, Bradley, Brendan & Wurgler, Jeffrey (2011), "Benchmarks as Limits to Arbitrage: Understanding the Low-Volatility Anomaly", *Financial Analysts Journal* 67(1), 40–54.**
- 1968–2008: low-vol/low-beta portfolios beat high-vol/high-beta cumulatively over 41 years. Explanation: leverage-constrained, benchmark-tracking institutions cannot arbitrage it.

**Frazzini, Andrea & Pedersen, Lasse Heje (2014), "Betting Against Beta", *Journal of Financial Economics* 111(1), 1–25.**
- Sample: US 1926–March 2012 + 20 international markets + bonds/futures.
- Signal: estimate ex-ante betas from rolling daily data (correlations 5y, vols 1y, shrunk toward 1); BAB = long leveraged low-beta portfolio, short delevered high-beta portfolio, both rescaled to beta 1 at formation.
- Performance: US equity BAB Sharpe **0.78** (1926–2012), roughly 2× value and 40% above momentum over the same period.

**Critique — Novy-Marx, Robert & Velikov, Mihail (2022), "Betting Against Betting Against Beta", *Journal of Financial Economics* 143(1), 80–106.**
- BAB's headline performance is driven by non-standard construction that implicitly equal-weights micro-caps (avg $1.05 per dollar committed to the bottom 1% of market cap) and by tilts toward profitability/investment factors, not beta arbitrage per se.
- **Implication for us: in a 515-name mega/large-cap universe, expect BAB to be weak.** A long-only low-vol *defensive sleeve* (lowest-vol quintile of our universe, vol from 252d daily returns) is the honest large-cap version — likely lower return, lower DD, useful only as ballast.
- **Tag: IMPLEMENTABLE-DAILY** (betas/vols from price data), but LOW expected edge in our universe.

---

## Family 5 — Volatility-managed portfolios (overlay)

**Moreira, Alan & Muir, Tyler (2017), "Volatility-Managed Portfolios", *Journal of Finance* 72(4), 1611–1644.**
- Signal: scale factor exposure by the **inverse of the previous month's realized daily variance**: w_t = c / RV²_{t−1}, c set so the managed series has the same unconditional vol as the unmanaged.
- Performance: positive alphas for market (~4.9%/yr, paper figure, not re-verified), value, momentum (largest gains), profitability, BAB, carry. Works because volatility spikes are not matched by proportional expected-return increases — so de-risking in high vol is nearly free.
- **Caveat — Cederburg, Scott, O'Doherty, Michael S., Wang, Feifei & Yan, Xuemin (2020), "On the Performance of Volatility-Managed Portfolios", *Journal of Financial Economics* 138(1), 95–117:** real-time (out-of-sample) implementations generally earn **lower** certainty-equivalents and Sharpes than the unmanaged portfolios; the positive spanning-regression alphas do not translate into direct outperformance for most factors. The big exception in the literature: **momentum**, where vol-management robustly helps (consistent with BS15/DM16 because momentum's vol is unusually persistent and its crashes are vol-timed).
- **Practical takeaway: apply variance-managed sizing to our MOMENTUM books and to the combined portfolio, and always A/B against the unmanaged version out-of-sample, per Cederburg.**
- **Tag: OVERLAY.**

---

## Family 6 — Seasonality

### Turn-of-month
**Ariel, Robert A. (1987), "A Monthly Effect in Stock Returns", *Journal of Financial Economics* 18(1), 161–174.**
- 1963–1981: cumulative equity returns accrue almost entirely in the first half of the trading month (defined to include the last trading day of the prior month); second half ≈ zero.

**Lakonishok, Josef & Smidt, Seymour (1988), "Are Seasonal Anomalies Real? A Ninety-Year Perspective", *Review of Financial Studies* 1(4), 403–425.**
- DJIA daily, 1897–1986. Turn-of-month days −1 to +3: cumulative **+0.473%** over the 4-day window vs **+0.349% for the entire average month** — i.e., TOM days account for ALL of the DJIA's average monthly gain; the rest of the month nets negative. Also documents turn-of-week/year/holiday effects.
- Modern confirmation: McConnell & Xu (2008, *Financial Analysts Journal*, "Equity Returns at the Turn of the Month") — effect persists post-1987 in US equities (surfaced in verification search; details not deeply verified).
- **Tag: IMPLEMENTABLE-DAILY / OVERLAY** (e.g., concentrate long exposure or new-position entries into TOM window; near-zero cost since it only re-times trades we already make).

### Cross-sectional (same-month) seasonality
**Heston, Steven L. & Sadka, Ronnie (2008), "Seasonality in the Cross-Section of Stock Returns", *Journal of Financial Economics* 87(2), 418–445.**
- Stocks that historically do well in calendar month m keep doing relatively well in month m — return response at annual lags 12, 24, 36, … out to 20 years, independent of size, industry, earnings dates.

**Keloharju, Matti, Linnainmaa, Juhani T. & Nyberg, Peter (2016), "Return Seasonalities", *Journal of Finance* 71(4), 1557–1590.**
- A strategy selecting stocks on historical same-calendar-month returns earns **≈13%/yr**; seasonalities also exist at daily frequency and in factors/commodities/indices.
- Caveat for us: daily data from 1997 gives max ~29 annual lags but per-name monthly seasonality estimates on 515 names will be noisy; long-short construction needed and turnover is monthly-full — costs matter.
- **Tag: IMPLEMENTABLE-DAILY.**

### Overnight vs intraday (we have hourly bars — rare edge)
**Lou, Dong, Polk, Christopher & Skouras, Spyros (2019), "A Tug of War: Overnight Versus Intraday Expected Returns", *Journal of Financial Economics* 134(1), 192–213.**
- Decompose close→open (overnight) vs open→close (intraday) returns. Across 14 trading strategies, profits are earned **either entirely overnight (momentum and related strategies) or entirely intraday, typically with opposite signs in the other session**. Firm-level overnight and intraday return components each persist for years, with cross-period reversal.
- Implication: WHEN you hold a position matters as much as WHAT you hold. Momentum longs earn their keep overnight; fading/value-type positions earn intraday.
- **Cost reality check for us:** a pure overnight-capture strategy trades 2×/day; at our 0.1%/side that is ≈0.2%/day ≈ 50%/yr drag — dead on arrival as a standalone. The implementable version is **execution timing of trades we already make** (enter momentum at the last bar of the day, exit at the open; schedule Book D fades to run intraday), which costs nothing extra.
- **Tag: IMPLEMENTABLE-HOURLY / OVERLAY (execution layer).**

---

## Family 7 — 52-week-high momentum

**George, Thomas J. & Hwang, Chuan-Yang (2004), "The 52-Week High and Momentum Investing", *Journal of Finance* 59(5), 2145–2176.**
- Signal: rank stocks by **nearness to 52-week high**, P_t / max(P, 252d); long the top 30%, short the bottom 30%, 6-month hold.
- Findings: nearness to the 52-week high explains a large share of momentum profits and **dominates past-return momentum (JT) and industry momentum** in head-to-head forecasts; unlike JT momentum, its profits do **not reverse long-run** (consistent with anchoring/underreaction rather than overreaction).
- For us: a one-line signal on our daily panel (rolling 252d max). Lower turnover than return momentum (nearness is sticky). Natural blend candidate with Book F's return-rank momentum; also a candidate crash-diagnostic (crash losses concentrate in "far-from-high" losers we don't short anyway — our books are long-only).
- Decay: still positive post-publication but reduced; subject to the same panic-state crash dynamics as other momentum (it is a momentum variant, not a hedge).
- **Tag: IMPLEMENTABLE-DAILY.**

---

## Family 8 — Pairs trading / statistical arbitrage

**Gatev, Evan, Goetzmann, William N. & Rouwenhorst, K. Geert (2006), "Pairs Trading: Performance of a Relative-Value Arbitrage Rule", *Review of Financial Studies* 19(3), 797–827.**
- Sample: US daily, 1962–2002.
- Signal: 12-month formation — match pairs by minimum sum of squared deviations between normalized (cumulative total return) price paths; 6-month trading — open long/short when spread diverges **> 2 historical standard deviations**, close on convergence (or period end).
- Performance: annualized excess returns **up to ~11–12%** on top-pairs portfolios, low market exposure, profits exceed conservative cost estimates through most of the sample.

**Do, Binh & Faff, Robert (2010), "Does Simple Pairs Trading Still Work?", *Financial Analysts Journal* 66(4), 83–95.**
- Confirms a continuing **downward trend in profitability** (mean excess returns roughly halved post-1990; near zero post-2002 after costs in their follow-up work), BUT the strategy spikes back to profitability in prolonged turbulence (2000–02, 2007–09).
- For us: distance-method pairs within 515 liquid names, 0.1%/side cost, is likely a graveyard in calm markets — same conclusion as our own Universe L/S experience (Sharpe 0.45). Could serve as a crisis-conditional book (echoes Nagel: convergence trading = liquidity provision, pays in high VIX). Low priority versus scaling Book D, which already monetizes the same effect more simply.
- **Tag: IMPLEMENTABLE-DAILY / IMPLEMENTABLE-HOURLY, but post-decay economics are marginal at our cost assumptions.**

---

## Family 9 — Volume / liquidity signals

**Amihud, Yakov (2002), "Illiquidity and Stock Returns: Cross-Section and Time-Series Effects", *Journal of Financial Markets* 5(1), 31–56.**
- ILLIQ_i = average over the period of daily |return| / dollar volume. Cross-section: illiquid stocks earn higher expected returns. Time series: expected market illiquidity raises required returns.
- For us: our 515 mega/large caps have tiny cross-sectional ILLIQ spread — the *premium* is mostly a small-cap phenomenon. Best use: ILLIQ (or |ret|/volume) as a **conditioning variable** — e.g., require minimum liquidity for Book D fade candidates, or use spikes in aggregate ILLIQ as a stress indicator.
- **Tag: IMPLEMENTABLE-DAILY (as conditioning variable; weak as standalone premium in our universe).**

**Gervais, Simon, Kaniel, Ron & Mingelgrin, Dan H. (2001), "The High-Volume Return Premium", *Journal of Finance* 56(3), 877–919.**
- Sample: NYSE 1963–1996 (later replicated broadly).
- Signal: classify a stock as "high-volume" if the formation-day (or week) trading volume is among the highest of the trailing ~50-day reference window; high-volume stocks **appreciate over the following month** (visibility/investor-recognition shock), low-volume stocks depreciate. Roughly +0.5% over the following 20 trading days for the zero-cost high-minus-low portfolio (paper figure, not re-verified); effect present in large stocks though stronger in small.
- Not explained by return autocorrelation, announcements, risk, or liquidity.
- For us: we have full volume history. Natural complements: (a) volume filter on Book D entries (a crash *with* volume spike = attention/capitulation → stronger snap-back prior), (b) standalone monthly high-volume tilt.
- **Tag: IMPLEMENTABLE-DAILY.**

---

## Family 10 — VIX / volatility-term-structure signals (^VIX + UVXY only)

**Whaley, Robert E. (2000), "The Investor Fear Gauge", *Journal of Portfolio Management* 26(3), 12–17.**
- Describes VIX construction; documents its strong negative contemporaneous relation with stock returns and its spike/mean-revert behavior. Foundation cite for VIX-conditioned rules (e.g., high VIX → subsequent equity returns above average).

**Bollerslev, Tim, Tauchen, George & Zhou, Hao (2009), "Expected Stock Returns and Variance Risk Premia", *Review of Financial Studies* 22(11), 4463–4492.**
- Signal: variance risk premium **VRP_t = implied variance (VIX²) − realized variance** (from recent SPX high-frequency/daily returns). High VRP predicts high future market returns; explains >15% of quarterly excess-return variation 1990–2005, dominating P/E, dividend yield, default spread.
- **Fully computable with our data:** VIX² minus realized variance of SPY from daily (or our hourly) bars. Usable as a portfolio-level risk-on/risk-off tilt.
- **Tag: IMPLEMENTABLE-DAILY / OVERLAY.**

**Simon, David P. & Campasano, Jim (2014), "The VIX Futures Basis: Evidence and Trading Strategies", *Journal of Derivatives* 21(3), 54–69.**
- The VIX futures basis (contango/backwardation) does not predict spot VIX changes but strongly predicts VIX **futures** returns; short futures in contango, long in backwardation.
- **Tag: UNTESTABLE** — requires VIX futures term-structure data. (UVXY's NAV decay embodies this premium, but *timing* the short requires the basis, which we don't observe. A crude proxy — VIX spot vs its own 21d mean as a contango stand-in — is testable but is NOT the published strategy; shorting UVXY also carries unbounded-loss/borrow-cost issues that violate our cost model.)

### Untestable registry (seminal papers requiring unavailable data — for completeness)
- **Simon & Campasano (2014)** — VIX futures basis (above). UNTESTABLE.
- **Post-earnings announcement drift** (Ball & Brown 1968; Bernard & Thomas 1989) — needs earnings dates/surprises. UNTESTABLE.
- **Accruals** (Sloan 1996), **profitability/quality** (Novy-Marx 2013; Asness et al. QMJ), **value** (Fama-French 1992) — need fundamentals. UNTESTABLE.
- **Short interest** (Rapach et al. 2016) and **analyst revisions** — need short-interest/analyst feeds. UNTESTABLE.
- (These are listed from general knowledge as registry placeholders; citations not individually re-verified in this pass.)

---

# TOP 5 RANKED CANDIDATES

Ranked by expected improvement to our stack given: Book F/A momentum drawdowns are our biggest weakness; Book D is our crown jewel; overlays are cheap to test and don't add new books; McLean-Pontiff decay penalizes fresh cross-sectional signals in large caps. **All expectations below are hypotheses to be settled by actual backtests on our data (per CLAUDE.md: no projected numbers).**

### #1 — Risk-managed momentum overlay on Books F and A (Barroso & Santa-Clara 2015)
**Why first:** attacks our single worst number (Book F MaxDD -39.8%, Book A -65%) with the best-replicated crash fix in the literature; needs zero new data; the mechanism (momentum vol persistence) is structural, not a return anomaly, so publication decay is minimal.
**Recipe:**
1. Take Book F's daily strategy return series (2019–2026 backtest; also Book A 1997–2026).
2. Each day compute σ̂_t = annualized std of the book's own daily returns over the trailing 126 trading days.
3. Exposure weight w_t = min(1.0, σ_target / σ̂_t) with σ_target tested over {10%, 12%, 15%, 20%} — capped at 1.0 since we don't lever; unallocated capital sits in cash (0%).
4. Update weekly (not daily) to limit turnover; charge 0.1% on the traded fraction |Δw|.
5. Report managed-vs-unmanaged Sharpe, CAGR, MaxDD, and 2020/2022/2025 sub-periods. Success = MaxDD materially reduced with Sharpe not lower.

### #2 — Panic-state gate on momentum (Daniel & Moskowitz 2016 + TSMOM trend filter)
**Why second:** complements #1 — BS15 scales continuously on own-vol; DM16 says the specific kill-zone is bear-market + high-vol + rebound. A discrete gate is simpler, more robust to estimation noise, and testable independently.
**Recipe:**
1. Bear indicator B_t = 1 if SPY total return over trailing 252 days < 0 (test 504d per DM16's 24-month spec).
2. Vol indicator V_t = 1 if SPY 63-day realized vol is in its top quintile (expanding window).
3. Gate: when B_t = 1 and V_t = 1, cut Book F/A momentum exposure to 50% (test 0%); redeploy the freed capital to Book D (see #3 — reversal pays best exactly then, per Nagel).
4. Non-overlapping with warmup; evaluate on 2019–2026 hourly-era and 1997–2026 daily-era (captures 2000–02, 2008, 2020, 2022).
5. Also test DM16's continuous version w_t ∝ μ̂_t/σ̂_t² as a refinement of #1.

### #3 — VIX-conditioned sizing of Book D (Nagel 2012)
**Why third:** highest-conviction *positive* overlay: our best book (Sharpe 2.74) is, academically, a liquidity-provision strategy whose conditional Sharpe rises with VIX. Instead of a fixed capital share, let Book D absorb more capital when VIX is elevated.
**Recipe:**
1. m_t = clip(VIX_{t-1} / median_252(VIX), 0.75, 1.75) as Book D's capital multiplier within the combined portfolio (borrowed pro-rata from momentum books — total exposure stays ≤ 1, no leverage).
2. Alternative discrete version: VIX_{t-1} > 25 → Book D weight ×1.5; VIX_{t-1} > 35 → ×2 (capped by available momentum capital). Grid over thresholds; beware overfitting — report full sensitivity table.
3. Combined with #2 this forms one coherent regime engine: calm → momentum-tilted; panic → reversal-tilted.
4. Metric: combined-portfolio Sharpe vs current 2.06 and MaxDD vs -19%, plus 2020 and 2022 sub-windows.

### #4 — Volatility-managed combined portfolio (Moreira & Muir 2017, with the Cederburg 2020 protocol)
**Why fourth:** one-parameter overlay on the whole stack; even if per-book gains are debatable (Cederburg), the combined book's variance is persistent, and the momentum share of the portfolio is exactly where MM17 works best.
**Recipe:**
1. w_t = min(1.0, c / RV²_{t−1}) on the combined daily return series, RV² = realized variance over the prior 21 trading days, c chosen on an expanding window so trailing managed vol matches unmanaged (no look-ahead — Cederburg's real-time requirement).
2. Weekly rebalance, 0.1% cost on |Δw|.
3. Decision rule per Cederburg: adopt ONLY if the managed series beats unmanaged in direct out-of-sample comparison (Sharpe AND MaxDD), not merely spanning alpha.

### #5 — Overnight/intraday execution layer from hourly bars (Lou, Polk & Skouras 2019)
**Why fifth:** we are unusually equipped (hourly bars, 515 names) to exploit the session split, and it re-times existing trades at ~zero incremental cost; ranked last only because expected magnitude is bps-per-trade, not a new return stream.
**Recipe:**
1. Diagnostic first: decompose Book F and Book D backtest P&L into overnight (prev close→open) vs intraday (open→close) components per holding. (Daily OHLC gives open; hourly bars refine intraday path.)
2. If Book F P&L is overnight-dominated (LPS prediction): move Book F entries from next-open to prior-day final hour, and exits from close to open. Re-run the 2019–2026 backtest with identical signals, changed execution timestamps only.
3. If Book D fade P&L is intraday-dominated: prefer same-day intraday exits where the 8h hold allows; avoid initiating fades in the final hour.
4. Explicitly do NOT build a standalone overnight-capture book: 2 trades/day × 0.1% ≈ 50%/yr cost drag kills it (documented here so we don't re-derive this).

### Honorable mentions (test after the top 5)
- **Short-term momentum via turnover double-sort (Medhat & Schmeling 2022)** — strongest-in-large-caps, price+volume only; both a new signal and a Book D entry filter (skip high-turnover crashers, which momentum-continue rather than revert).
- **52-week-high momentum (George & Hwang 2004)** — one-line signal; blend with Book F ranks for turnover reduction.
- **High-volume return premium (Gervais et al. 2001)** — volume-spike confirmation filter for Book D entries.
- **Turn-of-month tilt (Lakonishok & Smidt 1988)** — schedule new long entries into the TOM −1..+3 window; free to test.
- **Variance risk premium (Bollerslev et al. 2009)** — VIX² − realized variance as a portfolio risk-on tilt; computable today.

---

## Verification log
All citations above were confirmed to exist via web search on 2026-08-15 (journal, volume, pages checked against publisher/RePEc/SSRN records): Jegadeesh & Titman 1993; Carhart 1997; Barroso & Santa-Clara 2015; Daniel & Moskowitz 2016; Moskowitz, Ooi & Pedersen 2012; Huang, Li, Wang & Zhou 2020; De Bondt & Thaler 1985; Jegadeesh 1990; Lehmann 1990; Nagel 2012; Medhat & Schmeling 2022; Black, Jensen & Scholes 1972 (via secondary citations); Baker, Bradley & Wurgler 2011; Frazzini & Pedersen 2014; Novy-Marx & Velikov 2022; Moreira & Muir 2017; Cederburg, O'Doherty, Wang & Yan 2020; Ariel 1987 (via Lakonishok-Smidt's citation); Lakonishok & Smidt 1988; Heston & Sadka 2008; Keloharju, Linnainmaa & Nyberg 2016; Lou, Polk & Skouras 2019; George & Hwang 2004; Gatev, Goetzmann & Rouwenhorst 2006; Do & Faff 2010; Amihud 2002; Gervais, Kaniel & Mingelgrin 2001; McLean & Pontiff 2016 (JF, not JFE as commonly miscited); Whaley 2000; Bollerslev, Tauchen & Zhou 2009; Simon & Campasano 2014.
Not individually re-verified: Novy-Marx 2012 (intermediate momentum), McConnell & Xu 2008, and the untestable-registry placeholders (Ball & Brown 1968, Bernard & Thomas 1989, Sloan 1996, Fama & French 1992) — flagged inline.
Point estimates flagged "(paper figure, not re-verified)" were recalled from the papers but not independently confirmed in this search pass.
