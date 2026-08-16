# Orthogonal-Payoff Ideation Sweep — 2026-08

**Date:** 2026-08-16
**Mission:** NEW-BOOK candidates that are PAYOFF-SHAPE orthogonal to the champion — convex/defensive on stress days, not more momentum or reversal.
**Champion (anchor):** 2.78–2.79 Sharpe / -8.2% MaxDD (ivol+VT+panic-gate+D14, registry #84). ~75% of risk in hourly buy-the-panic liquidity provision → short-vol-like concave payoff, +0.30 SPY beta.
**Quantified target (loss anatomy, 2019–2026):** champion's worst-5% days average **SPY -1.14%, VIX +10.3%, UVXY +7.54%** (74% SPY-down, 69% VIX-up). The orthogonal book must be **positive or flat on exactly those days** without bleeding the unconditional UVXY carry (**-0.24%/day**) the rest of the time.
**Data (hard):** US large-cap OHLCV daily 1997+ / hourly 2019+; SPY, UVXY, ^VIX daily closes. NO VIX futures term structure, NO options, NO fundamentals. UVXY exists only from **2011-10** (inception) — all convexity backtests are ~15yr max, with only a handful of true stress events (Aug-2015, Feb-2018, Q4-2018, Mar-2020, 2022, plus 2024–2026 episodes). Costs 0.1–0.25%/side; house short-borrow convention **8%/yr**.
**Dead list respected (registry + prompt):** naive long UVXY, VRP-proxy SPY timing (#80), universe L/S bubble, shorting euphoria/anything, low-vol long-only (#82), pairs (#32), breadth capitulation timing (benched #83), TOM (#79), GKM (#36), overnight-share (benched #33), corr-spike gates (#68), jump/HMM gates (#42/#59), CSD (#57), LPPLS (#58).

**Portfolio-first evaluation rule (lesson of #83):** a candidate PASSES only if champion+candidate under the ivol+VT allocator beats 2.78 with p<0.10 (Ledoit-Wolf gate, registry #85) — standalone Sharpe is explicitly secondary. #83 failed not on signal but on allocation dilution (cash-heavy profile); every candidate below states its planned mitigation for that failure mode.

---

## Candidate Catalog (ALL considered, including rejected)

### V1. Standalone SPY Trend Book — long/flat/short monthly TSMOM ("crisis alpha as a book") — **ADVANCE, ranked #1**

- **Origin class:** Direct (literature). Note: registry #24/#29 tested trend only as a GATE on Books A/F (wash — their 210d returns were nearly always positive 2019–2026). A **standalone SPY trend book with a short/flat state was never tested.** Different payoff object: the gate could only remove exposure from momentum books; this book can be flat or net short the index when the champion is bleeding.
- **Citations (verified):**
  - Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum", *JFE* 104(2) — registry #29.
  - Hurst, Ooi & Pedersen (2017), "A Century of Evidence on Trend-Following Investing", *JPM* — 1880–2016, 67 markets; TSMOM positive in every decade and **performed well in 8 of the 10 largest 60/40 drawdowns**. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2993026
  - Greyserman & Kaminski (2014), *Trend Following with Managed Futures: The Search for Crisis Alpha*, Wiley — 800-year evidence; formalizes "crisis alpha" = trend's excess return during equity down-moves. https://onlinelibrary.wiley.com/doi/book/10.1002/9781118891018
- **Mechanism / why it persists:** Slow-moving investor flows and risk-management-driven deleveraging make large drawdowns serially correlated; a trend follower gets short (or flat) mid-crisis and monetizes continuation. Persists because it requires accepting years of whipsaw cost in calm bulls — exactly the anti-carry profile capital dislikes holding.
- **Leak-free spec:**
  - Signal at month-end close t: sign of SPY total return over trailing 12 months, skip last 5 days (also test 3m/6m/12m sign-vote blend, Faber 10m-MA variant). Rolling own-history only.
  - Position taken at **next day's open** (shift ≥1 bar): +1 SPY if up-trend; if down-trend, variant (a) flat, variant (b) -0.5 SPY, variant (c) -1.0 SPY. Rebalance checks monthly only (~2–6 trades/yr → costs negligible).
  - Also build the **"crisis-only" sleeve**: position = 0 in up-trend, short SPY in down-trend — isolates pure crisis alpha with zero bull-market beta overlap with the champion.
  - Backtestable 1997+ (29yr, 4 full bear markets: 2000–02, 2008, 2020, 2022) — the only candidate with real statistical power on stress behavior.
- **Stress-day corr to champion:** In sustained-stress regimes (2008/2020/2022-style), strongly negative — book is flat/short while champion's worst days cluster. Caveat: **no protection on isolated 1-day panics inside an up-trend** (monthly trend won't have flipped, e.g. Feb-2018, Aug-2024 flash spikes); the DM panic-gate overlay (#27) already covers part of that hole.
- **Expected standalone Sharpe (50% haircut + costs):** single-asset TSMOM gross Sharpe ~0.4–0.6 (MOP per-asset range) → **0.2–0.4 honest**. Long/flat variant carries SPY beta in bulls (overlaps champion's +0.30 beta); short/flat crisis-only variant has ~0 standalone Sharpe with fat right tail in bears — evaluate strictly at portfolio level.
- **Portfolio-contribution logic:** This is the canonical negative-stress-corr asset: modest standalone, negative corr concentrated exactly in champion's loss states. The 8% borrow convention costs a short-SPY sleeve ~0.7%/mo only while short (down-trend months ≈ 20–25% of history) — survivable, unlike a permanent short book.
- **#83-dilution mitigation:** test three integrations: (i) 6th book under ivol+VT; (ii) fixed 10–20% satellite outside the optimizer; (iii) champion-level position overlay (scale champion by trend state). (iii) has no cash-drag dilution channel at all.
- **Effort:** 1 day. **Kill criteria:** portfolio delta ≤0 or p≥0.10 on all three integrations.

---

### V2. Complacency-Timed Convexity — D-breadth INVERSION + VRP-proxy gate for small long-UVXY bursts — **ADVANCE, ranked #2 (original)**

- **Origin class:** Original synthesis (our own D-signal + verified convexity-timing literature). No Sharpe credit claimed.
- **Adjacent literature (verified, framing only):**
  - Cheng (2019), "The VIX Premium", *RFS* 32(1), 180–227 — the ex ante VIX premium **falls or stays flat when ex ante risk rises**; falling premium predicts rising ex post risk → conditional long-vol has windows of positive expectancy. https://academic.oup.com/rfs/article-abstract/32/1/180/5017289
  - Israelov & Nielsen (2015), "Still Not Cheap: Portfolio Protection in Calm Markets", *JPM* 41(4), 108–120 — **counter-evidence**: when option prices are low their expected value tends to be even lower; "VIX low" alone is NOT a buy signal. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2579232
  - Whaley (2013), "Trading Volatility: At What Cost?", *JPM* — short-term VIX ETPs in contango ~80% of days; structural decay quantified. https://www.ssrn.com/abstract=2261387
- **Mechanism:** Book D proves single-name oversold breadth is a priced panic signal (300/300 combos). Its INVERSION — a long spell with almost **no** oversold names + compressed VIX — marks liquidity-provision saturation/complacency: realized dip-buying leaves no one left to absorb the next shock. That is precisely when the champion's short-vol-like exposure is most crowded and convexity is closest to its cheap point (Israelov-Nielsen's warning is answered by requiring a **rising-risk confirmation**, per Cheng: don't buy calm, buy the turn out of calm).
- **Leak-free spec:**
  - Breadth_t = daily count of universe names with D bubble score < -0.8 (D's engine, already 1-bar-shifted). Complacency: 21d mean of Breadth_t in bottom decile of rolling 504d window.
  - VRP proxy: VIX_t − 21d realized SPY vol (annualized). Gate: proxy in bottom quintile of rolling 504d window **or falling over 10d** (Cheng's falling-premium analog).
  - Ignition (mandatory — this is what separates it from dead naive-long-UVXY): 5d realized vol > 21d realized vol AND VIX 5d change > 0.
  - All three true at close t → **buy UVXY at t+1 open, 5–10% of NAV**, hold max H ∈ {5,10,15}d; exit early if VIX 2yr percentile > 80 (sell the spike — never hold convexity after it pays); hard budget: max 3 concurrent/overlapping triggers, max ~30 in-market days/yr → worst-case carry bleed ≈ 30 × 0.24% × 7.5% NAV ≈ **-5bp/yr on portfolio** if it never pays.
- **Stress-day corr to champion:** negative by construction when in-market (long UVXY on days champion loses to UVXY +7.5%); zero (flat) otherwise. This is the only candidate that directly buys the exact instrument of the champion's loss profile.
- **Expected standalone Sharpe:** honest expectation **≈ 0, possibly negative** — this is a pure portfolio-contribution/skew book. UVXY history 2011+ only; success will hinge on 3–6 events → report event-level P&L table, not just Sharpe, and demand the bootstrap gate on the PORTFOLIO delta.
- **Registry-conflict check:** #80 killed VRP-proxy timing of **SPY beta**; this times **UVXY convexity** — different instrument, different payoff, and the kill reason (failed matched-beta test) doesn't apply to a 0-beta-budget hedge sleeve. Naive long UVXY is dead on carry; this caps carry exposure at ~30 days/yr by construction.
- **Effort:** 1–2 days (reuses D breadth infra from #83 work). **Kill criteria:** conditional UVXY expectancy over triggers ≤ unconditional -0.24%/day (i.e., conditioning adds nothing), or portfolio delta ≤ 0.

---

### V3. UVXY Carry-Regime Model — when is the drag weakest? (enabler study + optional trigger upgrade for V2) — **ADVANCE as STUDY, ranked #4**

- **Origin class:** Direct (literature) + adaptation to close-only data.
- **Citations (verified):**
  - Dew-Becker, Giglio, Le & Rodriguez (2017), "The Price of Variance Risk", *JFE* 123(2), 225–250 — 1996–2014: hedging news about **future** variance was ~costless; only **transitory realized** variance carries a (large, negative for longs) premium. Implication for us: UVXY (~30d constant maturity, the short end) sits at the **most expensive** point of the curve, and we cannot roll out the curve without futures data. The carry problem is structural, not fixable — only timeable. https://www.sciencedirect.com/science/article/abs/pii/S0304405X16302161
  - Whaley (2013) — as in V2.
- **Question:** regress forward 5d UVXY drift on observables available at close t: VIX level bucket, VIX minus 21d RV (VRP proxy), VIX 5d slope, VIX 2yr percentile. Hypothesis from futures literature: contango-drag is weakest (sometimes positive) when VIX is high/backwardated — proxied by VIX above its own 21d mean and rising. Output: a carry-regime map `E[UVXY drift | state]` on 2011–2026 daily data.
- **Why a study, not a book:** on its own this cannot clear any Sharpe hurdle (it's a conditional-mean map of a -0.24%/day asset). Value = upgrading V2's ignition gate and giving the verifier a falsifiable table (in which states was conditional drift ≥ 0, with CIs).
- **Effort:** 0.5 day. Pure analysis, no allocation risk.

---

### V4. Residual-Momentum Re-Rank of Book F — stress-beta reduction at the source (improvement, not new book) — **ADVANCE, ranked #3**

- **Origin class:** Direct (literature), routed to improvement queue.
- **Citations (verified):**
  - Blitz, Huij & Martens (2011), "Residual Momentum", *J. Empirical Finance* 18(3), 506–521 — ranking on Fama-French-residual returns instead of total returns roughly **doubles risk-adjusted momentum profits** and removes time-varying factor exposure. https://www.sciencedirect.com/science/article/abs/pii/S0927539811000041
  - Hanauer & Windmüller (2023), "Enhanced Momentum Strategies", *J. Banking & Finance* — residual/idiosyncratic momentum has materially **smaller crash drawdowns** than raw momentum (time-varying beta drives momentum crashes; residualization strips it). https://www.sciencedirect.com/science/article/abs/pii/S0378426622002928
  - Daniel & Moskowitz (2016) — registry #27 — the crash mechanism being neutralized.
- **Mechanism:** Momentum crashes are largely a **beta artifact**: after down-markets the winner portfolio is loaded with low-beta names and the loser side with high-beta; the rebound whipsaws it. Ranking on market-residual returns (no fundamentals needed — regression of daily stock returns on SPY over a rolling 252d window, rank on 12-1m cumulative residual scaled by residual vol) keeps the momentum premium while stripping the systematic component — reducing exactly the stress-day co-movement F contributes to the champion (F is the champion's second-largest stress-day loser after D).
- **Leak-free spec:** rolling 252d OLS beta per name (data through t only); residual r_i − beta_i × r_SPY cumulated over months t-12..t-1 (skip last month), divided by residual vol; same top-5 / hold-200h scaffolding as F. Execution unchanged (next bar). No shorting → borrow convention irrelevant.
- **Warning (prior evidence):** #31 (52wk-high re-rank of F) produced Sharpe -0.01 — this universe/period punishes rank-definition swaps toward "stability". Residual momentum is a different animal (keeps explosive movers, only de-betas the ranking), but treat #31 as a live base rate.
- **Portfolio-contribution logic:** contribution by **subtraction** — champion keeps F's return engine while its worst-5%-day SPY/VIX loading shrinks. No new allocation slot, no dilution channel, no new instrument. Cheapest test in this brief per unit of expected stress-profile improvement.
- **Expected effect:** F standalone Sharpe within ±0.1 of 1.55 baseline; champion stress-day tail (worst-5% mean) improves or test fails. **Kill criteria:** F Sharpe drops >0.15 or champion delta ≤0.
- **Effort:** 1 day.

---

### V5. Utilities/SPY Beta-Rotation Defensive Stream — **ADVANCE WITH RESERVATIONS, ranked #5**

- **Origin class:** Direct (practitioner literature — evidence tier below refereed).
- **Citations (verified):**
  - Gayed & Bilello (2014), "An Intermarket Approach to Beta Rotation: The Strategy, Signal, and Power of Utilities", **Charles H. Dow Award 2014**, SSRN — 4-week utilities-vs-market relative strength as risk-off rotation signal; utilities outperform in top-1% VIX regimes ~83% of the time in their sample. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2417974 (practitioner paper; no top-journal replication found — haircut accordingly)
- **Mechanism:** Utilities are bond-proxy, regulated-cash-flow names; institutional de-risking rotates into them **before** broad-index stress resolves — relative strength is a flow-based leading indicator. Unlike dead low-vol long-only (#82: pure beta channel), this is a **timing spread between two beta buckets**, in-market always, no cash drag.
- **Leak-free spec:** utilities basket = S&P 500 utilities names in our universe (NEE, DUK, SO, AEP, D, EXC, XEL, ED, WEC, PEG…, static list, survivorship-checked), equal-weight daily closes 1997+. Signal at close t: 21d total return of utils basket minus SPY. If positive → hold utils basket; else → hold SPY. Execute next open; min-hold 5d to damp churn (~10–20 switches/yr → ~2–4%/yr costs at 0.1–0.25%/side — material, must survive it).
- **Stress-day behavior:** on champion's worst-5% days this book holds utilities more often than not → losses damped (utils beta ~0.3–0.5) and occasionally positive — **flat-ish, not convex**. Meets the "flat" half of the target only.
- **Expected standalone Sharpe (haircut + costs):** 0.3–0.6. Dow-Award backtests are gross and pre-2014; McLean-Pontiff-style decay assumed.
- **Reservations:** (i) adjacency to the killed low-vol beta channel — hard gate: realized corr to champion must be < 0.35 and stress-day corr < 0; (ii) always-long-equity design overlaps champion's +0.30 beta in bulls; (iii) practitioner-only evidence tier.
- **Effort:** 1 day. **Kill criteria:** corr gates or portfolio delta ≤ 0.

---

## Rejected candidates (with reasons — do not silently re-propose)

| Candidate | Origin | Reject reason |
|---|---|---|
| **R1. Dollar-neutral BAB spread** (Frazzini-Pedersen 2014, *JFE* 111(1); Novy-Marx & Velikov 2022, *JFE* 143(1)) | Direct | Triple kill: (1) house 8%/yr borrow convention costs the ~$0.7 short leg ~4–5%/yr — comparable to the entire large-cap BAB premium; (2) Novy-Marx-Velikov show headline BAB rides equal-weighted micro-caps and non-standard beta shrinkage — our large-cap-only universe keeps the costs and loses the juice; (3) BAB is itself funding-liquidity-short: it CRASHES when margin constraints tighten (the F-P mechanism) — i.e., positively correlated with champion stress, failing the mission spec. Registry-worthy as a documented rejection. |
| **R2. Panic-gate state as standalone book** (Daniel-Moskowitz 2016) | Original | Only 11 panic days in 2019–2026 window (#27); no statistical basis for a standalone allocation; the state is already monetized as champion overlay. Nothing left to extract. |
| **R3. Post-spike short-vol / VIX-spike mean reversion** | Direct | Adds MORE concave short-vol payoff — the exact opposite of the mission. Also "shorting anything" is on the dead list. Immediate reject. |
| **R4. Flight-to-quality TLT/GLD sleeve** | Direct | Best-in-class stress hedge in the literature, but **outside the stated data constraint** (no bond/gold series). Logged as a data-acquisition recommendation only (daily TLT/GLD from 2002+ is one yfinance call); NOT counted in Top 5. |
| **R5. Dispersion trading proper** | Direct | Requires index/single-name options. Untestable with our data. The closest price-only cousin (cross-sectional dispersion gates) already failed as Stivers-Sun pre-test (#78). |
| **R6. Hourly SPY trend (fast crisis alpha)** | Synthesis | Gao et al. intraday momentum already dead at our cost tier (#46: Sharpe -9.5 after costs); fast trend on one instrument churns 50–100×/yr — cost wall insurmountable at 10–25bp/side. |

---

## TOP 5 — ranked by expected PORTFOLIO contribution (not standalone Sharpe)

| Rank | Candidate | Origin | Standalone Sharpe (honest) | Stress-day corr to champion | Why it should add vs 2.78 | Effort |
|---|---|---|---|---|---|---|
| 1 | **V1** SPY trend book, long/flat/short + crisis-only sleeve | Direct (Hurst-Ooi-Pedersen; Greyserman-Kaminski) | 0.2–0.4 (L/F/S); ~0 (crisis-only) | **Strongly negative in sustained bears**; ~0 on isolated spike days | 29yr testable, 4 real bears; canonical crisis alpha; 3 integration modes dodge the #83 dilution trap | 1d |
| 2 | **V2** Complacency-timed UVXY convexity (D-breadth inversion + VRP-proxy + ignition) | **Original synthesis** | ~0 (skew book) | **Negative when in-market** — long the exact loss instrument | Only candidate paying off on the precise worst-5% profile (UVXY +7.5% days) with carry bleed capped ≈5bp/yr by budget | 1–2d |
| 3 | **V4** Residual-momentum re-rank of F | Direct (Blitz-Huij-Martens; Hanauer-Windmüller) | F ±0.1 vs 1.55 | Reduces champion's own stress beta (subtraction, not hedge) | No new slot, no dilution channel; de-betas the second-largest stress contributor at source | 1d |
| 4 | **V3** UVXY carry-regime map | Direct (Dew-Becker et al.; Whaley) | n/a (study) | n/a | Upgrades V2's gate; falsifiable conditional-drift table for the verifier | 0.5d |
| 5 | **V5** Utilities/SPY beta rotation | Direct (Gayed-Bilello, Dow Award) | 0.3–0.6 | Flat-ish (damped), not convex | In-market always (no cash-drag dilution); flow-based leading indicator; hard corr gates vs low-vol adjacency | 1d |

**Recommended execution order:** V3 (half-day study) → V2 (uses V3's map) → V1 → V4 → V5. Total ≈ 5 days. Promotion gate for each: champion+candidate ≥ champion with Ledoit-Wolf bootstrap p<0.10 (#85 protocol), PLUS an explicit worst-5%-day table (mean P&L on champion's stress days before/after) — the mission metric, reported alongside Sharpe.

---

## For the registry (rows to merge into research/papers_read.md as tested)

| Paper (Author, Year, Journal) | Family | Status |
|---|---|---|
| Hurst, Ooi & Pedersen (2017), JPM | TS momentum/crisis alpha | pending — V1 |
| Greyserman & Kaminski (2014), Wiley book | TS momentum/crisis alpha | pending — V1 (context) |
| Cheng (2019), RFS 32(1) | Vol premium dynamics | pending — V2 gate design |
| Israelov & Nielsen (2015), JPM 41(4) | Convexity cost | pending — V2 counter-evidence, respected in spec |
| Whaley (2013), JPM | VIX ETP decay | pending — V2/V3 |
| Dew-Becker, Giglio, Le & Rodriguez (2017), JFE 123(2) | Variance term structure | pending — V3 (structural constraint: short end priciest) |
| Blitz, Huij & Martens (2011), JEF 18(3) | Residual momentum | pending — V4 |
| Hanauer & Windmüller (2023), JBF | Enhanced momentum | pending — V4 |
| Gayed & Bilello (2014), Dow Award/SSRN | Sector rotation/defensive | pending — V5 (practitioner tier) |
| Frazzini & Pedersen (2014), JFE 111(1) | BAB | **rejected** — R1 (borrow convention + NMV critique + funding-crash correlation) |
| Novy-Marx & Velikov (2022), JFE 143(1) | BAB critique | **rejected (context)** — kill evidence for R1 |

**Citation corrections vs prompt:** Israelov-Nielsen is *JPM* 2015 (not FAJ); "Dew-Becker vol term structure" is Dew-Becker, Giglio, Le & Rodriguez (2017) *JFE*; Blitz-Huij-Martens (2011) is *J. Empirical Finance* — the "half the crash risk" framing is supported by the follow-on literature (Hanauer-Windmüller 2023: residualization strips the time-varying beta that drives crashes) rather than stated in the 2011 original.
