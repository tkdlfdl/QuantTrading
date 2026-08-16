# New-Book Ideation Sweep — 2026-08

**Date:** 2026-08-15
**Mission:** Candidate books structurally orthogonal to existing momentum (A, F) and mean-reversion/contrarian (D, C) books.
**Hurdle:** standalone honest Sharpe > 0.8 after 0.2% round-trip costs, low corr to momentum AND reversal, tolerable DD.
**Data constraint:** daily OHLCV 1997+ (~523 large caps + SPY, UVXY, ^VIX, incl. VOLUME); hourly open/close 2019+ (515 names, SPY 2020-07+). No options, futures, fundamentals, short interest, tick data, small caps.

**Killed ideas (do not re-propose):** Reddit sentiment (all variants), universe L/S bubble, pairs trading, 52wk-high ranking, turnover-conditioned ST momentum, SPY first-hour→last-hour intraday momentum, residual/market-mode-stripped reversal, lead-lag networks, LPPLS, critical-slowing-down, conformal Kelly.

---

## Candidate Catalog (ALL considered)

### C1. Turn-of-Month Harvest + Dash-for-Cash Refinement — **ADVANCE (Top 5)**

- **Origin class:** Direct (literature)
- **Citations (verified):**
  - Lakonishok & Smidt (1988), RFS — original TOM documentation, 90 years of DJIA.
  - Etula, Rinne, Suominen & Vaittinen (2020), *"Dash for Cash: Monthly Market Impact of Institutional Liquidity Needs"*, **Review of Financial Studies** 33(1), 75–111 (NOT Journal of Finance as sometimes cited). https://academic.oup.com/rfs/article/33/1/75/5494694
  - Xu & McConnell (2008), "Equity Returns at the Turn of the Month", FAJ.
  - Recent survivals check: UK indices 1990–2025 still show positive TOM abnormal returns; seasonality-after-costs surveys find TOM among the few effects surviving realistic costs at low turnover (https://quantpedia.com/strategies/turn-of-the-month-in-equity-indexes, https://paperswithbacktest.com/blog/every-seasonality-effect-in-finance).
- **Mechanism / why it persists:** Institutional payment cycle — pension contributions, payroll flows, fund distributions create predictable month-end selling pressure (T-8..T-4) and month-turn buying pressure (T-1..T+3). The flow is structural (contractual payment dates), not information-driven; arbitrage capital is limited because the premium per event is small and calendar-locked. Etula et al. show the pattern is global and tied to measurable institutional liquidity needs.
- **Signal spec (no look-ahead):**
  - Calendar-only signal — computable arbitrarily far in advance, zero leakage risk by construction.
  - ENTER: buy SPY at the close of the 4th-to-last trading day of the month (T-3).
  - EXIT: sell at the close of the 3rd trading day of the next month (T+3). ~7 sessions in market per month; flat otherwise (~67% of days flat).
  - Refinement A (dash-for-cash): remain flat during T-8..T-4 (documented negative drift window). No short leg (8% borrow assumption kills it; keep long-only).
  - Refinement B (optional, test only): substitute equal-weight basket of our 523 large caps for SPY; institutional-flow effect is broad.
- **Expected corr:** ~0.10–0.20 to A (SPY beta overlap only ~33% of days, and A holds concentrated momentum names); ~0 to D/C (different trigger mechanism, different days). Structurally orthogonal: this is a FLOW/CALENDAR book — no price signal at all.
- **Expected Sharpe (post-haircut, post-cost):** Gross TOM premium ~0.4–0.6% per event × 12 = 5–7%/yr; costs 12 round trips × 0.2% = 2.4%/yr; net ~3–4%/yr at ~7–9% annualized vol (in-market only 1/3 of time) → **honest expectation 0.4–0.6**. Below the 0.8 hurdle standalone, BUT near-zero corr means portfolio math may still accept it; also capital is free 2/3 of the time (can sit in the same account funding other books).
- **Implementation:** trivial — daily SPY closes 1997+, ~30 lines. Effort: 0.5 day. 29 years of events (≈350 independent observations) = strong statistical power for a calendar test.

---

### C2. High-Volume Return Premium (GKM) — **ADVANCE (Top 5)**

- **Origin class:** Direct (literature), with an orthogonality-enhancing modification
- **Citations (verified):**
  - Gervais, Kaniel & Mingelgrin (2001), *"The High-Volume Return Premium"*, **Journal of Finance** 56(3), 877–919. https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00349
  - Kaniel, Li & Starks — international replication (effect present in nearly all developed + emerging markets). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=474100
  - Decay caveat: McLean & Pontiff (2016) — average anomaly loses ~26% out-of-sample, ~58% post-publication. https://www.hec.ca/finance/Fichier/McLean.pdf
- **Mechanism / why it persists:** Merton investor-recognition / visibility hypothesis: an extreme-volume day expands a stock's investor base; broader ownership lowers required return temporarily → price appreciates over the following weeks. GKM show the effect is NOT explained by return autocorrelation, announcements, market risk, or liquidity — i.e., it is not momentum and not reversal, which is exactly the structural orthogonality we want. Persists because it is small per name, needs breadth, and is attention-driven (attention frictions don't arbitrage away).
- **Signal spec (no look-ahead):**
  - For each stock, each day t: volume percentile of day-t volume vs its own trailing 50-day window [t-49, t] (rolling, own-history only).
  - High-volume shock: day-t volume in top decile of that window (GKM's "high-volume" classification).
  - **Orthogonality filter (our modification):** exclude names whose |day-t return z-score| > 3 (vs 20d window) — removes overlap with Book C's 4-sigma events and with news-driven momentum; isolates the pure volume/visibility shock.
  - Portfolio: at close t+1 (execution shifted 1 bar), go long equal-weight the top-N (N=10–20) volume-shock names from day t; hold 20 trading days; run 4 weekly-staggered tranches to smooth entry timing.
- **Expected corr:** low to A/F (GKM orthogonal to momentum controls); low to D/C after the price-move filter (D/C trigger on price extremes, this triggers on volume extremes with price explicitly muted). Estimate 0.1–0.25 vs each. First book to use the VOLUME column at all — a genuinely unused data dimension.
- **Expected Sharpe (post-haircut, post-cost):** GKM premium is weaker in large caps (~0.2–0.5%/month for large-cap terciles vs >1% small caps); after 50% haircut and ~2.5%/yr costs (monthly turnover): **honest expectation 0.4–0.7**. Wide error bars — the large-cap-only restriction is the main risk; the international replication breadth is the main hope.
- **Implementation:** daily OHLCV 1997+ suffices. Effort: 1–2 days. Grid: ref window {30,50,70}, decile cutoff {80,90,95pct}, hold {10,20,40d}, N {10,20}.

---

### C3. Overnight-Share Cross-Sectional Persistence — **ADVANCE (Top 5, with corr gate)**

- **Origin class:** Direct (literature) — cross-sectional version, NOT the killed timing overlay
- **Citations (verified):**
  - Lou, Polk & Skouras (2019), *"A Tug of War: Overnight versus Intraday Expected Returns"*, **Journal of Financial Economics** 134, 192–213. https://personal.lse.ac.uk/polk/research/TugOfWar.pdf
  - Key finding: strong firm-level overnight return continuation (persists for years), offset by cross-period reversal; heterogeneous-clientele mechanism.
- **Mechanism / why it persists:** Different investor clienteles trade at open vs during the day (retail/sentiment demand concentrates near the open; institutions trade intraday). A stock's persistent overnight-return tilt reveals which clientele "owns" it; clientele demand is sticky for months–years. Persists because exploiting it fully requires trading at the open daily (costly), but the CROSS-SECTIONAL slow version (monthly holds) captures the clientele premium at low turnover.
- **Signal spec (no look-ahead):**
  - Overnight return: r_on(t) = open_t / close_{t-1} − 1 (daily OHLCV has open — computable 1997+).
  - Signal at month-end m: mean overnight return over trailing 12 months, skip most recent 5 days (avoid ST reversal contamination). All rolling, own-history.
  - Long top-decile (~50 names, or top-20 for concentration) by trailing overnight return; execute at next session's close (shift ≥ 1 bar); hold 1 month; rebalance monthly.
- **Expected corr:** **0.3–0.45 to A/F — this is the weak point.** LPS show momentum profits accrue overnight, so past-overnight-winners overlap with past-total-return winners. Mitigation: rank on overnight MINUS intraday return (the "tug of war" spread) instead of raw overnight — LPS show these components have OPPOSITE-signed premia, so the spread is more orthogonal to total-return momentum. Near-zero expected corr to D/C.
- **Expected Sharpe (post-haircut, post-cost):** LPS decile spreads ~2%/month gross include shorts and small caps; long-only large-cap version after haircut and 2.4%/yr costs: **honest expectation 0.5–0.7**.
- **HARD GATE:** compute realized daily-return corr to Book F in backtest; **kill if > 0.40** (we killed turnover-conditioned ST momentum at 0.55 — same rule).
- **Implementation:** daily OHLCV 1997+. Effort: 1–2 days.

---

### C4. Oversold-Breadth Capitulation → Index Timing ("Book D at market level") — **ADVANCE (Top 5, ranked #1)**

- **Origin class:** Original synthesis (Book D's confirmed signal + market timing). No Sharpe credit claimed.
- **Citations:** none for the exact construct (original). Adjacent, verified: Zweig Breadth Thrust literature — breadth-derived thrust/washout signals are rare, long-horizon, historically bull-market-initiating (e.g., March 2009); documented as low-frequency but reliable (https://www.quantifiedstrategies.com/zweig-breadth-thrust-indicator-strategy/, https://trendspider.com/learning-center/zweig-breadth-thrust/). Our construct differs: we aggregate a PROVEN proprietary signal (D's bubble score, 300/300 grid combos positive) rather than advance/decline lines.
- **Mechanism / why it persists:** Book D proves single-name hourly oversold (bubble < −0.8) carries a large premium (Sharpe 2.71, 8/8 years). When MANY names are simultaneously oversold, the cause is indiscriminate index-level liquidation (margin calls, risk-parity deleveraging, fund outflows) — forced selling, not information. Forced selling mean-reverts at the INDEX level over days–weeks, a horizon D (8h holds, top-20 rotation) does not monetize. The edge persists for the same reason D's does, plus institutional constraints prevent buying during drawdowns.
- **Signal spec (no look-ahead):**
  - Breadth_t = count of universe names with bubble score < −0.8 at hourly bar t, using D's exact bubble computation (already shifted 1 bar in production — inherit that). Aggregate to daily: max over day's bars.
  - Percentile of Breadth_t vs trailing 504-day (2yr) rolling window, own history only.
  - ENTER: breadth percentile crosses above 95th → long QQQ (or SPY) at NEXT day's open (shift 1 bar).
  - EXIT: fixed hold H ∈ {5, 10, 15, 20} trading days (grid), or breadth percentile < 50th, whichever first. Re-trigger allowed after exit.
  - Variant C4b (test as alternative/composite trigger): ^VIX percentile > 95th over 2yr window as trigger — compare and combine (breadth AND/OR VIX spike).
  - Flat otherwise (expected in-market ~10–25% of days).
- **Expected corr:** to D: 0.2–0.35 — both are long during selloff recoveries, BUT D rotates 8h positions in single names while this holds the index for weeks; time-scale separation caps overlap. To A/F: near zero or negative (momentum books are getting hurt exactly when this triggers). To C: near zero (C is single-name 4-sigma events). Must be measured — gate at 0.45 vs D.
- **Expected Sharpe:** **no credit — original.** Prior beliefs: entries at extreme capitulation historically favorable (2020-03, 2022-06, 2022-10 analogs in our data window); low trade count (~5–15 events/yr) means wide confidence intervals — 1997+ daily variant (using a daily-bar bubble-score port) should be built for statistical power alongside the 2019+ hourly version.
- **Implementation:** highest leverage of any candidate — reuses D's tested bubble infra (`strategies/contrarian_bubble_hourly.py`), single-instrument execution, ~10–20 round trips/yr → costs negligible (0.2% × 15 = 0.3%/yr). Effort: 1 day hourly version; +1 day for the 1997+ daily-bubble variant.
- **Failure mode to test explicitly:** 2008-style regime where oversold keeps getting more oversold — measure MaxDD of entries during sustained bear legs (hence the fixed-hold exit, no averaging down, no re-entry until flat).

---

### C5. VRP-Proxy Regime Timing for SPY — **ADVANCE (Top 5, ranked #5, with alpha-vs-beta gate)**

- **Origin class:** Direct (literature) + adaptation to our data (no options → VIX-minus-realized proxy)
- **Citations (verified):**
  - Bollerslev, Tauchen & Zhou (2009), *"Expected Stock Returns and Variance Risk Premia"*, **RFS** 22(11), 4463–4492 — VRP (implied minus realized variance) predicts aggregate returns; strongest at quarterly horizon; dominates P/E, default spread, CAY. https://academic.oup.com/rfs/article-abstract/22/11/4463/1565787
  - Moreira & Muir (2017), *"Volatility-Managed Portfolios"*, **JF** 72(4) — scaling exposure inverse to vol raises Sharpe. https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513
  - **Counter-evidence (must respect):** Cederburg, O'Doherty, Wang & Yan (2020), *"On the performance of volatility-managed portfolios"*, **JFE** — real-time implementable versions generally FAIL out-of-sample; spanning-regression alphas are not tradeable. https://www.sciencedirect.com/science/article/abs/pii/S0304405X2030132X
- **Mechanism / why it persists:** VRP = compensation for bearing variance risk; when the market pays a high premium for insurance (VIX >> realized), subsequent equity returns are high. Persists as a genuine risk premium (compensation for crash exposure), not an inefficiency.
- **Signal spec (no look-ahead):**
  - VRP_t = VIX_t² − RV_t², where RV = annualized 21-day realized vol of SPY daily returns through close t (rolling, no forward data).
  - Percentile of VRP_t vs trailing 2yr window.
  - Exposure at t+1 close (shift 1): 100% SPY if VRP pctile > 70; 50% if 30–70; 0% if < 30 (i.e., realized vol spiking ABOVE implied = regime stress → stand aside). Discrete tiers, NOT regression weights — direct response to Cederburg critique (structural instability of spanning regressions).
- **Expected corr:** 0.3–0.5 to A in bull markets (both are long-equity-beta when things are calm) — the worst orthogonality profile of the top 5. Near zero to C/D.
- **Expected Sharpe (post-haircut, post-cost):** SPY B&H Sharpe ~0.6 on this window; BTZ predictability suggests timed version 0.7–0.9 gross; after haircut: **honest expectation 0.5–0.7**, and only the increment over beta counts.
- **HARD GATE (beta-not-alpha rule, learned from Reddit book):** must beat a constant-exposure SPY benchmark with the SAME average exposure on Sharpe AND MaxDD, and corr to A must come in < 0.45. If it's repackaged beta, kill it.
- **Implementation:** ^VIX + SPY daily 1997+ (VIX full history). Effort: 1 day. Cheap to test, cheap to kill.

---

### C6. Long UVXY Crisis Convexity at Low-VIX Regimes — **REJECT**

- **Origin class:** Direct (direction 1 in the mandate).
- **Why rejected:** UVXY (1.5x VIX futures) bleeds roll-down at roughly −6 to −10%/month in contango regimes — which is precisely when the "low VIX percentile" trigger would have you long. Standalone expected Sharpe is deeply NEGATIVE; it is insurance, not a book, and violates the standalone Sharpe > 0.8 mandate by construction. Book A already carries a UVXY hedge — adding a second UVXY-long book duplicates that exposure. Shorting UVXY correctly remains banned (unbounded risk). **Disposition:** if crisis convexity is wanted, it belongs as a portfolio-level overlay sized off Book A's existing hedge logic, not as a new book. No further work.

---

### C7. Defensive Low-Vol / Low-Idio-Vol Basket (Baker-Haugen; Ang et al.) — **REJECT (park)**

- **Origin class:** Direct (directions 2 and 7 collapse into this — "long low-idio-vol as the defensive book" IS the low-vol book).
- **Citations (verified):** Long-run premium real (1940–2023: ~6.4%/yr, CAPM alpha 6.3%, t=5.3) BUT recent evidence is adverse: documented underperformance 2015–2024, and the premium is valuation-conditional — "low-volatility only works when it's cheap" (https://rpc.cfainstitute.org/blogs/enterprising-investor/2024/the-low-volatility-factor-and-occams-razor, https://www.acadian-asset.com/investment-insights/managing-risk/low-volatility-investing-welcoming-the-elephant-into-the-room, https://alphaarchitect.com/low-volatility-strategies/).
- **Why rejected:** (1) The anomaly is a LOW-BETA effect; long-only implementation still carries ~0.7–0.9 market beta → correlates with A in bulls through the beta channel, violating the orthogonality mandate. Beta-neutral implementation needs shorts (8% borrow kills it). (2) A decade of underperformance + crowding + valuation-dependence means honest post-haircut Sharpe estimate is 0.3–0.5 with high regime risk. (3) Within our 523 LARGE-CAP universe the vol spread is compressed vs the full-market studies. **Disposition:** parked; revisit only with a valuation-conditioning signal we currently lack (no fundamentals).

---

### C8. Index Reconstitution / S&P Inclusion-Exclusion — **REJECT**

- **Origin class:** Direct (direction 6).
- **Citations (verified):** Greenwood & Sammon (2025), *"The Disappearing Index Effect"*, **Journal of Finance** — S&P 500 inclusion abnormal return fell from +7.4% (1990s) to <1% (2010s); deletion effect −0.1% 2010–2020, despite index-fund growth. Markets now anticipate changes. https://onlinelibrary.wiley.com/doi/10.1111/jofi.13410
- **Why rejected:** (1) The edge is documented as gone in the modern sample — the only sample we could trade. (2) Data: our universe lists are current snapshots, not timestamped membership histories; detecting historical add/drop dates from them is unreliable and survivorship-contaminated. Double kill (no edge + no data). No further work.

---

### C9. Cross-Sectional Seasonality (Heston-Sadka same-calendar-month) — **BENCH (2nd tier, not rejected)**

- **Origin class:** Direct (not in the mandate's list; adjacent to direction 3).
- **Citations (verified):** Heston & Sadka (2008), *"Seasonality in the Cross-Section of Stock Returns"*, **JFE** — stocks with high returns in a given calendar month repeat in that same month up to 20 annual lags; strategy ~13%/yr; independent of size, industry, earnings, dividends. International replication (2010). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=687022
- **Mechanism:** recurring seasonal flows/attention (fiscal calendars, window dressing, mood seasonality per Hirshleifer et al.).
- **Spec sketch:** at month-end, rank names by average same-calendar-month return over lags 1–20 years (rolling, own history; needs ≥5yr history per name — fine, data from 1997); long top-20 equal weight; hold 1 month; execute next close.
- **Why benched (not top 5):** headline 13%/yr is LONG-SHORT full-universe; long-only large-cap post-publication estimate degrades to ~0.3–0.5 Sharpe after 2.4%/yr costs. Orthogonality is genuinely good (lag-12+ signal, near-zero overlap with our lookbacks), so it's a legitimate 6th candidate if a top-5 dies at the gate. Effort: 1 day.

---

### C10. Long-Only Dispersion / Idio-Vol Trades — **REJECT**

- **Origin class:** Direct (direction 7).
- **Why rejected:** Real dispersion trades need options (index vs single-name vol) — we have none. Long-only proxies collapse to either (a) long high-idio-vol — documented NEGATIVE premium (Ang, Hodrick, Xing, Zhang 2006), i.e., wrong sign by construction; or (b) long low-idio-vol — which is C7, already rejected via the beta channel. Realized-correlation/dispersion as a REGIME conditioner for sizing D is plausible but is an overlay on an existing book, not a new book. No further work as a book.

---

### C11. Month-End Institutional-Flow Reversal — **MERGED into C1**

The Etula et al. dash-for-cash pre-month-end weakness (T-8..T-4) and post-turn strength are two halves of the same flow cycle; implemented as C1's stay-flat window rather than a separate short book (8% borrow + thin premium kills the short half standalone).

---

## Top 5 Ranking (orthogonality × expected Sharpe × implementability / effort)

| Rank | Candidate | Origin | Why this rank |
|------|-----------|--------|---------------|
| 1 | **C4 Oversold-Breadth Capitulation Index Timing** | Original synthesis | Reuses D's proven infra (1 day effort); trigger fires when momentum books bleed (negative conditional corr to A/F); negligible costs; the only candidate extending a CONFIRMED in-house edge to a new time scale. Risk: low event count — mitigate with 1997 daily-bubble variant. |
| 2 | **C2 High-Volume Return Premium (GKM)** | Direct | First use of the volume column — structural orthogonality by data dimension; JF-published, internationally replicated; clean spec on 29 years of data. Risk: large-cap attenuation. |
| 3 | **C1 Turn-of-Month + Dash-for-Cash** | Direct | Near-zero corr by construction (flat 2/3 of days, calendar trigger); RFS-grade mechanism; half-day of work; 29yr × 12 events of statistical power. Risk: modest Sharpe ceiling (0.4–0.6) — accepted for its diversification-per-unit-effort. |
| 4 | **C3 Overnight-Share Cross-Section (LPS)** | Direct | Strong JFE mechanism, perfect data fit (daily opens 1997+), monthly turnover. Ranked below C1/C2 solely for corr risk to momentum — hard gate at 0.40 vs F; prefer the overnight-minus-intraday spread ranking. |
| 5 | **C5 VRP-Proxy SPY Regime Timing** | Direct | Full VIX history, 1-day build, RFS-backed premium. Last because it must survive the beta-not-alpha gate (Cederburg critique + our Reddit lesson) and has the worst expected corr to A (0.3–0.5). Cheap to test, cheap to kill. |

**Build order:** C4 and C1 first (2 days combined, both nearly free option value), then C2, then C3/C5 gated.

---

## For the Registry

| ID | Name | Class | Key citation | Signal family | Freq / Hold | E[Sharpe] net | E[corr] A/F | E[corr] D/C | Effort | Status |
|----|------|-------|--------------|---------------|-------------|---------------|-------------|-------------|--------|--------|
| C4 | Oversold-Breadth Capitulation | Synthesis/Original | none (Zweig-adjacent) | Breadth of proprietary bubble score → index long | Event, ~5–15/yr, hold 5–20d | no credit — original | ~0 / neg | 0.2–0.35 (D) | 1–2d | ADVANCE #1 |
| C2 | High-Volume Return Premium | Direct | Gervais-Kaniel-Mingelgrin JF 2001 | Volume shock (price-muted) | Weekly form, 20d hold | 0.4–0.7 | 0.1–0.2 | 0.1–0.25 | 1–2d | ADVANCE #2 |
| C1 | Turn-of-Month + Dash-for-Cash | Direct | Etula et al. RFS 2020; Lakonishok-Smidt 1988 | Calendar/flow | Monthly, 7-session hold | 0.4–0.6 | 0.1–0.2 | ~0 | 0.5d | ADVANCE #3 |
| C3 | Overnight-Share Cross-Section | Direct | Lou-Polk-Skouras JFE 2019 | Overnight-vs-intraday clientele | Monthly, 1mo hold | 0.5–0.7 | 0.3–0.45 (GATE 0.40) | ~0 | 1–2d | ADVANCE #4 (gated) |
| C5 | VRP-Proxy SPY Timing | Direct | Bollerslev-Tauchen-Zhou RFS 2009; Cederburg JFE 2020 (counter) | VIX − realized vol regime | Daily signal, tiered exposure | 0.5–0.7 (increment over beta only) | 0.3–0.5 (GATE 0.45 + beta test) | ~0 | 1d | ADVANCE #5 (gated) |
| C9 | Same-Calendar-Month Seasonality | Direct | Heston-Sadka JFE 2008 | Lag-12k seasonal | Monthly | 0.3–0.5 | 0.1–0.2 | ~0 | 1d | BENCH |
| C7 | Low-Vol Defensive Basket | Direct | Baker-Haugen; Ang et al. 2006 | Low beta/idio-vol | Monthly | 0.3–0.5 | beta channel ~0.4+ | ~0 | — | REJECT (parked: valuation-conditional) |
| C6 | Long UVXY Convexity | Direct | — | Vol carry (wrong side) | Event | negative | neg | ~0 | — | REJECT (overlay only, not a book) |
| C8 | Index Reconstitution | Direct | Greenwood-Sammon JF 2025 | Flow event | Event | ~0 (effect gone) | — | — | — | REJECT (edge dead + no membership history) |
| C10 | Long-Only Dispersion | Direct | Ang et al. 2006 | Idio-vol | — | wrong sign / needs options | — | — | — | REJECT |

**Gates applying to all advances:** shift ≥ 1 bar between signal and execution; rolling-window stats only; 0.2% round-trip cost; 8% borrow on any short (none of the top 5 shorts); full-history backtest (1997+ where data allows, per CLAUDE.md rule 2); realized-corr kill thresholds as specified; long-only unless stated. All Sharpe figures above are pre-backtest EXPECTATIONS, not results — per project rules, no number enters CLAUDE.md until an actual backtest produces it.

---

## Verified Source Links

- GKM 2001: https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00349 ; intl replication https://papers.ssrn.com/sol3/papers.cfm?abstract_id=474100
- Lou-Polk-Skouras 2019: https://personal.lse.ac.uk/polk/research/TugOfWar.pdf ; https://www.sciencedirect.com/science/article/abs/pii/S0304405X19300650
- Etula et al. 2020 (RFS, not JF): https://academic.oup.com/rfs/article/33/1/75/5494694
- Greenwood-Sammon 2025: https://onlinelibrary.wiley.com/doi/10.1111/jofi.13410
- Moreira-Muir 2017: https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513 ; Cederburg et al. 2020 counter: https://www.sciencedirect.com/science/article/abs/pii/S0304405X2030132X
- Bollerslev-Tauchen-Zhou 2009: https://academic.oup.com/rfs/article-abstract/22/11/4463/1565787
- Heston-Sadka 2008: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=687022
- Low-vol recent evidence: https://rpc.cfainstitute.org/blogs/enterprising-investor/2024/the-low-volatility-factor-and-occams-razor ; https://www.acadian-asset.com/investment-insights/managing-risk/low-volatility-investing-welcoming-the-elephant-into-the-room
- TOM persistence/costs: https://quantpedia.com/strategies/turn-of-the-month-in-equity-indexes ; https://paperswithbacktest.com/blog/every-seasonality-effect-in-finance
- McLean-Pontiff decay: https://www.hec.ca/finance/Fichier/McLean.pdf
- Zweig breadth thrust background: https://www.quantifiedstrategies.com/zweig-breadth-thrust-indicator-strategy/
