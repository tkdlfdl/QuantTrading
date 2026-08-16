# Backtest Performance Reference

Documentation of actual backtest results for all strategies (Books A–F) and the
combined portfolios. **Data As Of:** 2026-06-18 (retest: 2026-07-08).

---

## 2026-08-15 ALL-BOOKS RETEST (locked params, daily-convention scoring)

Every live book re-run with its locked `live/config.py` params on data through
2026-07-08, scored with the project-standard daily Sharpe, recorded to
`strategies/performance/book_*_retest_*`. Runner: `retest_all_books.py` (+
`retest_fix_bc.py`, `retest_universe_ls.py`).

| Book | Documented | Retest | Δ Sharpe | Verdict |
|------|-----------|--------|----------|---------|
| D Contrarian | 2.710 / -6.4% | **2.740 / -6.4%** | +0.03 | ✅ CONFIRMED |
| A Momentum+UVXY | 1.410 / -65.3% | **1.301 / -65.4%** | -0.11 | ✅ confirmed |
| F Hourly Mom | 1.642 / -39.6% | **1.548 / -39.8%** | -0.09 | ✅ confirmed |
| C Intraday MR | 0.927 / -20.8% | **0.592 / -53.7%** | **-0.34** | ❌ INFLATED — daily MTM rewrite (`retest_book_c_mtm.py`) reveals hidden intra-trade drawdowns (2021: -53.7%); block convention showed 0.853/-30.0% |
| B QQQ Bubble | 1.106 / -5.9% | **0.775 / -8.8%** | **-0.33** | ❌ INFLATED — docs used entry-date attribution; proper bar-by-bar daily gives ~0.78 |
| E Reddit Sent. | 1.97 (old) | **0.625 / -43.7%** | -1.35 | ❌ see Strategy 4 audit |
| Universe L/S | 0.454 / -14.0% | **-0.959 / -80.2%** full hist; **-0.123** even on doc window | — | ❌ FAILED — does not reproduce; deprecated |

**Corrected correlation matrix (retest series, 1,450 common days):**
```
        A      B      C      D      F      E
A   1.000  0.196 -0.002  0.255  0.588  0.186
B   0.196  1.000  0.008  0.371  0.270  0.244
C  -0.002  0.008  1.000 -0.032  0.000 -0.023
D   0.255  0.371 -0.032  1.000  0.392  0.519
F   0.588  0.270  0.000  0.392  1.000  0.375
E   0.186  0.244 -0.023  0.519  0.375  1.000
```
Key changes vs the documented matrix: **B↔D 0.371 (was -0.043)** and **D↔E 0.519**
— the old near-zero correlations were partly an artifact of entry-date block
attribution misaligning P&L in time. True diversification is lower than
documented. C remains genuinely uncorrelated.

### Honest combined portfolio (re-estimated from retest series)

Runner: `retest_portfolio_honest.py` (mirrors `run_portfolio_abdc.py` allocator
incl. 50% cap). Inputs: `book_*_retest` daily series, C = MTM rewrite.
Recorded: `portfolio_fixed_ew_honest_*`, `portfolio_momalloc_honest_*`.

| Config (2019-01..2026-07) | Sharpe | CAGR | MaxDD | Documented |
|---------------------------|--------|------|-------|------------|
| Fixed EW A+F+D+C+B | **2.060** | 48.0% | -19.0% | 2.344 / -18.3% |
| MomAlloc lb20 rb60 | 1.970 | 54.9% | -24.5% | 2.170 / -28.8% |
| Fixed EW **without B** | 2.013 | **57.3%** | -21.6% | — |
| MomAlloc **without B** | 2.013 | **62.9%** | -26.7% | — |

The portfolio effect is real but smaller than documented (2.06 vs 2.34).

**Book B decision (2026-08-15):** B's marginal Sharpe contribution is ≈ zero
(FixedEW +0.047, MomAlloc **-0.044**) while its equal-weight slot costs ~9pp
CAGR (48.0% vs 57.3% without it). At an honest standalone Sharpe of 0.775 and
CAGR 7.4%, B does not earn its allocation → **recommended: DROP Book B from the
live rotation** (fold its capital into the remaining books).

**Book C decision:** honest profile is Sharpe 0.592 / MaxDD **-53.7%** (2021
intra-trade collapse hidden by entry-date attribution; 2026 running -10.4% with
-33% MDD). C stays uniquely uncorrelated (~0 to all books), which is its only
case for a reduced allocation — treat as high-risk satellite, not core.

---

## 2026-08-15 LITERATURE PROGRAM (60yr survey → tested improvements)

Surveys: `research/briefs/strategies_60yr_survey.md`,
`research/briefs/construction_60yr_survey.md`. Paper tracking (read / IMPROVED
marks — never reread listed papers): **`research/papers_read.md`** (40 papers).

### Confirmed improvements (recorded in strategies/performance/)

| Config | Sharpe | CAGR | MaxDD | vs baseline | Paper |
|--------|--------|------|-------|-------------|-------|
| **ivol + 15% vol-target, no B** | **2.501** | 45.2% | **-8.9%** | EW 2.060/-19.0% | Moreira-Muir 2017 JF; Harvey+ 2018; Qian 2005 |
| ivol + vol-target, all books | 2.420 | 35.7% | -9.6% | EW 2.059/-19.0% | same |
| EW + vol-target, all | 2.206 | 37.3% | -11.5% | same | Moreira-Muir 2017 |
| champion + DM panic gate | 2.514 | 45.3% | -8.9% | champ 2.501 | Daniel-Moskowitz 2016 (marginal; 11 panic days) |
| quarter-Kelly shrunk, no B | 2.098 | 44.7% | -19.8% | EW-noB 2.013/-21.6% | Kelly 1956; MacLean+ 2011 |
| Book F risk-managed (10% tgt) | 1.609 | 18.4% | -10.3% | F 1.548/-39.8% | Barroso-Santa-Clara 2015 |

**New recommended portfolio: inverse-vol weights + de-risk-only 15% vol-target
overlay on A+C+D+F (no B) — Sharpe 2.501, MaxDD -8.9%** vs prior Fixed EW
2.060/-19.0%. Optional DM panic gate adds insurance at no cost.

### Tested, no gain (kept for the record)
ERC (1.57), HRP (1.82), Sharpe-weighting (1.80), plain inverse-vol (DD breach),
Nagel VIX-scaling of D (2.479 < champion), Barroso overlay on Book A (UVXY hedge
already covers it). DeMiguel 1/N null result empirically confirmed at N=5.

### Highest-value pending (specced, not yet tested)
1. Lou-Polk-Skouras 2019: re-time Book F entries/exits using overnight vs
   intraday return split (needs engine change; we have the hourly data).
2. Medhat-Schmeling 2022: turnover-conditioned short-term momentum.
3. Faber 2007 / TSMOM trend gate per book.
4. George-Hwang 2004: 52-week-high ranking variant for Book F.

All figures come from actual backtests. Methodology, cost model, and calculation
procedures live in `.claude/agents/backtest.md`. When a backtest changes any
parameter or metric here, update this file so it stays the source of truth.

---

## Part 1: Individual Strategies

### Strategy 1: QQQ Hourly Bubble (PRODUCTION — Book B)

**⚠️ 2026-08-15 retest:** documented Sharpe 1.106 came from entry-date block
attribution; proper bar-by-bar daily attribution on the same locked params gives
**Sharpe 0.775, MaxDD -8.8%** (engine-matched timing verified). Treat 0.775 as
the honest figure.

**Type:** Mean Reversion, Hourly, Single-Stock

**Universe:** QQQ (Invesco QQQ Trust)

**Signals:**
- LONG: Bubble score < -0.8 (oversold)
- Hold: 24 hours

**Backtest Period:** 2020-07-27 to 2026-06-18 (11,775 bars, ~6yr)
**Data Source:** Alpaca IEX (2020-2024) + yfinance (2024-2026)
**Note:** Alpaca IEX QQQ data starts 2020-07-27; earlier history unavailable.

**Best Parameters (6yr grid search):** ma=200h, z=100h, buy_thresh=0.8, hold=24h

| Metric | Value |
|--------|-------|
| Sharpe | 1.106 |
| Sortino | 0.594 |
| CAGR | ~9.4% |
| Total Return | +68.9% (6yr) |
| Max DD | -5.94% |
| Win Rate | 63.3% |
| Trades | 90 |

**Yearly:**
```
2021: +17.06% (Sharpe 1.672, MDD -2.98%)
2022: +2.93%  (Sharpe 0.354, MDD -4.23%)
2023: -2.05%  (Sharpe -0.277, MDD -5.94%)  <- QQQ +55% bull run hurt signals
2024: +13.11% (Sharpe 2.074, MDD -3.99%)
2025: +13.02% (Sharpe 1.611, MDD -4.65%)
2026: +8.82%  (Sharpe 1.468, MDD -3.88%)
```

**Robustness (240 grid combos):**
```
Positive Sharpe: 180/240 (75%)
Sharpe > 1.0:    9/240   (4%)    <- low robustness
Sharpe > 1.5:    0/240
```

**vs Benchmark (QQQ B&H, same period):**
```
                    Sharpe    Return    Max_DD
QQQ Bubble          1.106     +68.9%    -5.9%
QQQ Buy-Hold        0.913    +194.9%   -35.0%

Advantage: 1.2x better Sharpe, 6x smaller drawdown
Trade-off: Significantly lower absolute return
```

**WARNING — Prior 2yr Backtest Was Overfitted:**
```
Prior documented result (2024-2026 only, 15 trades):
  Sharpe 1.967, MaxDD -0.72%  <- cherry-picked 2yr window
Full 6yr result (2020-2026, 90 trades):
  Sharpe 1.106, MaxDD -5.94%  <- correct, includes 2023 failure year
```

**Characteristics:**
- ✅ Better Sharpe than QQQ buy-hold
- ✅ Much lower drawdown than buy-hold
- ⚠️ Low parameter robustness (only 4% of combos > Sharpe 1.0)
- ⚠️ 2023 was negative (-2%) during QQQ +55% bull run
- ⚠️ Thin signal frequency (90 trades over 6yr)
- ⚠️ Much weaker than Book D (Sharpe 1.1 vs 2.71, robustness 4% vs 100%)

---

### Strategy 2: Universe Hourly Bubble (DEPRECATED — failed 2026-08-15 retest)

**Retest:** same params on the same documented window now give Sharpe **-0.123**
(-7.1%), and full 2019-2026 history gives Sharpe **-0.959**, -77.5% total,
MaxDD -80.2% (2022: -62%). The documented +19.9% does not reproduce on current
data. Do not allocate. Original documentation retained below for the record.

**Type:** Mean Reversion, Hourly, Multi-Stock

**Universe:** S&P 500 + NASDAQ 100 (515 stocks)

**Signals:**
- LONG: Top 10 oversold (bubble < -0.8)
- SHORT: Top 10 overbought (bubble > +0.95)
- Hold: 8 hours
- Portfolio: 50% long + 50% short

**Backtest Period:** 2024-06-20 to 2026-06-18 (3,479 bars, 515 tickers)

**Best Parameters:** ma=100h, z=100h, buy_thresh=0.8, short_thresh=0.95, top_n=10

| Metric | Value |
|--------|-------|
| Sharpe | 0.454 |
| Sortino | 0.654 |
| Return | +19.85% |
| Max DD | -14.00% |
| Win Rate | 50.88% |
| Trades | 397 periods |

**Long vs Short Decomposition:**

LONG Side (3,347 trades):
```
Return: -43.9%
Sharpe: 0.428
Max DD: -86.2%
Avg Trade: +0.039%
Win Rate: 50.8%
```

SHORT Side (346 trades):
```
Return: +1.7%
Sharpe: 0.159
Max DD: -33.4%
Avg Trade: +0.027%
Win Rate: 49.7%
```

**Critical Insight: Hedging Effect**
```
Pure Long:      -43.9% (UNACCEPTABLE)
Pure Short:     +1.7% (MEDIOCRE)
50/50 Combined: +19.9% (OPTIMAL, Sharpe 0.454)

The 50/50 hedged portfolio is ESSENTIAL. Long bubbles are too volatile,
shorts provide necessary downside protection creating synergistic effect.
```

**Characteristics:**
- ✅ Higher absolute return (+19.9%)
- ✅ Moderate drawdown control
- ⚠️ Lower Sharpe than QQQ (0.454 vs 1.967)
- ⚠️ Larger drawdown (-14% vs -0.7%)
- ⚠️ High trade frequency requires active monitoring
- ⚠️ Long-only component severely underperforms

---

### Strategy 3: Intraday Mean Reversion (PRODUCTION — Book C)

**⚠️ 2026-08-15 retest (daily MTM rewrite):** documented Sharpe 0.927 / MaxDD
-20.8% used entry-date block attribution. Proper mark-to-market on the same
locked params gives **Sharpe 0.592 / MaxDD -53.7%** — a 2021 intra-trade
drawdown was entirely hidden by the block convention. Treat 0.592/-53.7% as the
honest figures; high-risk satellite only.

**Verifier finding (2026-08-15, `verification_log.csv`):** the -53.7% MaxDD is
one event — the HOOD post-IPO squeeze (2021-08-04 +50%, 08-05 -28%). 4-sigma
signals fired on consecutive days (8/2, 8/3), both flip-long HOOD, and
overlapping trades stack additively → **~2x implicit leverage on a single meme
stock** (engine-faithful, verified against raw prices; +93%/-54% portfolio
days). Structural risk: single-position concentration + signal stacking. A
per-ticker overlap cap would remove this exposure mode.

**Type:** Mean Reversion + Momentum Flip, Hourly/Daily, Multi-Stock

**Universe:** S&P 500 + NASDAQ 100 (404 tickers with full hourly history)

**Logic (two-phase):**
1. **Signal:** Daily return Z-score > 4.0 (extreme move vs 20-day rolling window)
2. **Phase 1 (next bar):** Fade the move for 1 hour (short after surge, long after crash)
3. **Phase 2 (next 3 days):** Flip direction — ride the momentum continuation

**Backtest Period:** 2019-01-02 to 2026-06-18 (7.46 years)
**Data Source:** Alpaca hourly (2019-2024) + yfinance (2024-2026), merged

**Best Parameters:** sigma=4.0, lookback=20d, flip_hold=3d, top_n=5

| Metric | Value |
|--------|-------|
| Sharpe | 0.927 |
| Sortino | 0.599 |
| CAGR | 27.18% |
| Total Return | +501% |
| Max DD | -20.81% |
| Win Rate | 58.3% |
| Trades | 127 (7.46yr) |
| Positive Years | 7/8 (88%) |

**Yearly:**
```
2019:  -0.30% (Sharpe -1.15, 1 trade only — near flat)
2020: +11.52% (Sharpe 1.13,  MDD -4.27%)
2021:+104.72% (Sharpe 1.37,  MDD -12.05%) <- COVID volatility spike
2022: +30.60% (Sharpe 1.16,  MDD -10.67%)
2023: +27.49% (Sharpe 1.59,  MDD -9.06%)
2024: +18.92% (Sharpe 0.59,  MDD -20.81%) <- low volatility, fewer signals
2025: +21.68% (Sharpe 1.07,  MDD -10.50%)
2026:  +9.56% (Sharpe 0.82,  MDD -14.24%) <- partial year
```

**Robustness (540 grid combos):**
```
Positive Sharpe: 34/540  (6.3%)   <- very fragile
Sharpe > 0.5:    18/540  (3.3%)
Sharpe > 1.0:    0/540            <- no combo exceeds 1.0
```

**WARNING — Extremely Narrow Parameter Sensitivity:**
```
Only sigma=4.0 + lookback=20d works. All other sigma values average deeply negative Sharpe.
top_n has zero impact (5, 10, 20 produce identical results).
The strategy is essentially a single point in parameter space.
```

**Characteristics:**
- ✅ Strong CAGR (+27%) driven by high-volatility years
- ✅ 7/8 positive years (COVID crash 2020 included)
- ⚠️ High max drawdown (-20.81%) — worst of all live books
- ⚠️ Extremely fragile: only 6.3% of grid combos positive Sharpe
- ⚠️ Regime-dependent: ~17 trades/yr, only fires on 4-sigma events
- ⚠️ 2024 near-zero Sharpe (0.59) in low-volatility environment

---

### Strategy 4: Reddit Sentiment Long (EXPERIMENTAL — Book E)

**Type:** Sentiment Analysis, Daily

**Status:** Experimental — **documented metrics found unreliable (2026-08-15 reliability audit)**

**Locked live params:** ma=15, zw=40, mild=0.5, extreme=0.6, hold=8d, top_n=5, min_mentions=5

**Reliability audit (backtest_book_e_reliability.py):** the same locked strategy,
re-scored three ways:

| Scoring | Window | Sharpe | Ann/CAGR | MaxDD |
|---------|--------|--------|----------|-------|
| Original (per-trade blocks, RF=2%) | 2024-03..2026-06 | 1.968 | 37.0% | -12.2% |
| Project-standard daily Sharpe | 2024-03..2026-06 | 1.253 | 32.6% | -22.0% |
| Project-standard daily, full history | 2019-01..2026-06 | **0.625** | 12.6% | **-43.7%** |

**Why the documented 1.97 was inflated:**
1. **Per-block Sharpe:** returns were recorded once per 8-day holding block (63
   observations), hiding intra-block volatility and drawdown. Daily attribution
   on the identical trades gives 1.25, and true MaxDD doubles (-12% → -22%).
2. **Short favorable window:** 2024-03 start (data available at the time)
   excludes 2022 (-24%) and the 2020 -44% drawdown. Full backfilled history
   (2019-2026) gives Sharpe 0.63.
3. **Thin evidence:** 63 trades; sentiment coverage degrades sharply post-2024
   (2025: only 86 symbols, whole months missing).
4. **No out-of-sample confirmation:** live paper trading (42 days,
   2026-06-09..2026-08-14) has held **zero positions** — sentiment data ends
   2026-06-01 and the 5-day staleness guard drops the book daily.

**Related variants (all recorded in strategies/performance/):** MA-momentum
(Sharpe 1.13), MA-contrarian (0.98), 4-zone capitulation (0.92), locked-params
full history (0.63). All variants are +0.79..0.88 correlated with each other —
the returns are dominated by long-only beta to a heavily-discussed high-beta
basket, not by the sentiment signal itself.

**Notes:**
- Treat all Reddit sentiment variants as supplementary at most
- 2022 is negative for every variant (-20% to -36%)
- Literature review: `research/briefs/reddit-sentiment-ma.md`

---

### Strategy 5: Contrarian Bubble (PRODUCTION — Book D)

**Type:** Mean Reversion, Hourly, Multi-Stock

**Universe:** S&P 500 + NASDAQ 100 (515 stocks)

**Signals:**
- LONG: Top 20 most oversold (bubble < -0.8)
- Hold: 8 hours

**Backtest Period:** 2019-01-02 to 2026-06-18 (12,378 bars, 515 tickers)
**Data Source:** Alpaca hourly (2019-2024) + yfinance (2024-2026), merged

**Best Parameters:** ma=104h, buy_thresh=0.8, hold=8h, top_n=20

**Daily Attribution:** Actual intraday (entry open->day close, middle days close->close, exit prev_close->exit close)

| Metric | Value |
|--------|-------|
| Sharpe | 2.71 |
| Sortino | 4.34 |
| CAGR (Ann Return) | 30.09% |
| Total Return | +611% |
| Max DD | -6.41% |
| Win Rate | 54.7% |
| Active Trade Days | 1,478 |
| Positive Years | 8/8 (100%) |

**Yearly (all 8 years positive):**
```
2019: +15.06% (Ann +20.1%, Sharpe 2.16, MDD -2.29%)
2020: +13.93% (Ann +34.5%, Sharpe 2.92, MDD -3.72%)
2021: +45.98% (Ann +46.0%, Sharpe 4.02, MDD -2.54%)
2022: +46.16% (Ann +46.4%, Sharpe 2.54, MDD -6.41%)  <- bear market best year
2023: +27.33% (Ann +27.6%, Sharpe 2.43, MDD -5.70%)
2024: +23.38% (Ann +24.8%, Sharpe 2.45, MDD -3.16%)
2025: +33.72% (Ann +34.0%, Sharpe 2.47, MDD -5.85%)
2026: +21.04% (Ann +51.4%, Sharpe 3.54, MDD -2.37%)
```

**vs Other Live Strategies:**
```
                    Sharpe    CAGR     Max_DD    Period         Robustness
Book D Contrarian   2.71     30.1%    -6.4%     2019-2026      300/300 combos > 0
Book E Reddit       1.97     ~37%     ~-10%     ~1yr (exp)     data-limited
Book F Hourly Mom   1.64     78.8%    -39.6%    2019-2026      74/216 combos > 1.0
Book A Momentum     1.41     ~85%*    -65.3%    1997-2026      (*uses leverage)
Book B QQQ Bubble   1.11      9.4%    -5.9%     2020-2026      9/240 combos > 1.0
Book C Intraday MR  0.93     27.2%    -20.8%    2019-2026      34/540 combos > 0
```

**Characteristics:**
- ✅ Long-only (no shorts)
- ✅ Highest Sharpe of all live books (2.71)
- ✅ Lowest Max Drawdown of all live books (-6.41%)
- ✅ All 300 grid combinations positive Sharpe (extremely robust signal)
- ✅ Thrives in bear markets: 2022 +46% as sell-offs create abundant oversold signals
- ✅ 8/8 positive years including COVID crash (2020) and bear market (2022)
- ⚠️ Lower absolute return than Book A (no leverage)

---

### Strategy 6: Universe Hourly Momentum (PRODUCTION — Book F)

**Type:** Momentum, Hourly Signal, Multi-Stock, Long-Only

**Universe:** S&P 500 + NASDAQ 100 (404 tickers with full hourly history)

**Signals:**
- LONG: Top 5 stocks by cumulative return over past 750 hourly bars (107 trading days, ~5 months)
- Hold: 200 hourly bars (29 trading days, ~6 weeks)
- Non-overlapping: rebalance only after current hold fully closes

**Backtest Period:** 2019-01-02 to 2026-06-18 (7.46 years)
**Data Source:** Alpaca hourly (2019-2024) + yfinance (2024-2026), merged
**Note:** 750h warmup = 107 trading days; 2019 returns are 0% (warmup year).

**Best Parameters (216-combo grid search):** lb=750h, hold=200h, top_n=5

| Metric | Value |
|--------|-------|
| Sharpe | 1.642 |
| CAGR | 78.8% |
| Total Return | +4512% |
| Max DD | -39.6% |
| Trades | 56 (~7/yr) |

**Yearly:**
```
2019:   0.0%  (warmup — 750h lookback = 107 trading day warmup)
2020: +18.3%
2021: +26.0%
2022: +17.1%  (positive in bear market — momentum rotates to defensive winners)
2023: +107.3%
2024: +128.8%
2025:  +86.6%
2026: +177.5%  (partial year)
```

**Robustness (216 grid combos):**
```
Positive Sharpe:  166/216 (77%)
Sharpe > 1.0:      74/216 (34%)
Sharpe > 2.0:       0/216  (0%)
```

**Sensitivity (avg Sharpe across all other params):**
```
Hold period (decisive — monotonic improvement):
    4h: -1.857    13h:  0.361    40h:  0.888    120h: 1.130
    8h: -0.278    20h:  0.608    80h:  1.120    200h: 1.217  <- best tested

Lookback (also monotonic at long end):
   20h:  0.249   120h:  0.421   500h:  0.470
   40h:  0.240   200h:  0.342   750h:  0.533   <- best tested
   80h:  0.322   300h:  0.327  1000h:  0.682   <- still improving
```

**Critical Insight — Converges to Daily Momentum:**
```
750h / 7 bars per trading day = 107 trading days (~5-month lookback)
200h / 7 bars per trading day =  29 trading days (~6-week hold)
Book A uses: lookback=140d, hold=40d (nearly identical parameters)

The optimal hourly momentum IS daily momentum on hourly bars.
The hourly granularity provides no signal advantage at these parameters.
```

**vs Book A (aligned 2019-2026 window):**
```
                       Sharpe    CAGR     Max_DD
Book F Hourly Mom       1.619   77.0%    -39.6%
Book A Daily Mom(1.25x) 1.369   68.2%    -54.3%

Book F: higher Sharpe (no leverage = lower vol), better MaxDD
Book A: UVXY hedge advantage in late bull market (2025-2026)
```

**Characteristics:**
- ✅ Long-only, no leverage, no shorts
- ✅ Higher Sharpe than Book A on aligned window (1.62 vs 1.37)
- ✅ Better MaxDD than Book A (-39.6% vs -54.3%, no leverage)
- ✅ Positive in 2022 bear market (+17.1%)
- ✅ 77% of combos positive Sharpe
- ⚠️ Only 34% of combos > Sharpe 1.0 (lower robustness vs Book D 100%)
- ⚠️ Very few trades: ~7 rebalances/year (thin statistical evidence at optimal params)
- ⚠️ Sensitivity axes still monotonically improving at grid boundary
- ⚠️ 2019 fully absorbed by warmup (effectively 6-year live history)

---

## Part 2: Combined Portfolio Results (A+F+D+C+B)

**Runner:** `run_portfolio_abdc.py`
**Backtest Period:** 2019-01-02 to 2026-06-18 (7.73 years)
**Book availability:** D+C from 2019, A+F from 2019 (F warmup absorbs first ~107 days), B from 2020-07-27

### Portfolio Configurations

Two combined portfolios are tracked:

**1. Fixed Equal Weight (Fixed EW)**
- Equal weight among all active books on each day
- D+C always active; A+F active from start; B active from 2020-07-27
- No rebalancing signal — pure diversification

**2. Momentum Allocation (MomAlloc)**
- Weight proportional to rolling return over lookback window
- Max 50% cap per strategy; excess redistributed to remaining books
- Best params: lb=20d, rebalance=60d

### Overall Metrics (2019-2026, 7.73yr)

| Strategy | Sharpe | CAGR | Total Return | Max DD |
|----------|--------|------|-------------|--------|
| Fixed EW (A+F+D+C+B) | **2.344** | 47.5% | +1912% | -18.3% |
| MomAlloc lb=20d rb=60d | 2.170 | **61.9%** | +4045% | -28.8% |
| Fixed 50/50 D+C | 1.605 | 29.2% | +626% | -10.9% |
| Book A only | 1.627 | 93.9% | +16583% | -54.3% |
| Book F only | 1.516 | 64.2% | +4512% | -39.6% |
| Book D only | 2.528 | 29.3% | +629% | -6.5% |
| Book C only | 0.855 | 26.1% | +501% | -20.8% |
| Book B only | 0.961 | 6.9% | +67% | -5.9% |

**Key finding:** Fixed EW achieves the highest Sharpe (2.344) of any configuration — diversification across low-correlation books does most of the work. MomAlloc yields higher raw CAGR (61.9%) but more drawdown (-28.8%).

### Yearly Returns

```
Year   MomAlloc   FixedEW   Book A   Book F   Book D   Book C   Book B
2019    +10.27%   +11.04%  +27.14%   +0.00%  +15.06%   -0.30%   +0.00%
2020   +146.35%   +75.42% +501.25%  +18.32%  +13.93%  +11.52%   +1.67%
2021    +56.93%   +34.52%  -20.35%  +25.95%  +45.98% +104.72%  +17.06%
2022    +40.93%   +26.40%  +20.86%  +17.06%  +46.43%  +30.60%   +2.93%
2023    +42.43%   +40.91%  +52.07% +107.33%  +27.38%  +27.49%   -2.05%
2024    +49.52%   +51.63%  +69.05% +128.79%  +23.76%  +18.92%  +13.11%
2025    +46.68%   +63.18% +167.22%  +86.55%  +38.49%  +21.68%  +13.02%
2026   +120.86%   +74.28% +230.01% +198.77%  +19.13%   +9.56%   +8.82%
```
MomAlloc: 0 negative years across full 7.73yr period.

### Yearly Max Drawdown

```
Year   MomAlloc   FixedEW   Book A   Book F   Book D   Book C   Book B
2019     -6.80%    -4.38%  -17.27%    0.00%   -2.29%   -0.30%    0.00%
2020    -10.27%    -7.49%  -28.01%   -4.77%   -3.72%   -4.27%   -1.16%
2021    -28.82%   -12.34%  -48.95%  -29.45%   -2.54%  -12.05%   -2.98%
2022     -8.19%    -6.57%  -19.63%  -28.95%   -6.52%  -10.67%   -4.23%
2023    -12.03%    -8.56%  -30.26%  -18.89%   -5.70%   -9.06%   -5.94%
2024    -16.14%   -13.34%  -20.43%  -24.49%   -3.33%  -20.81%   -3.99%
2025    -15.35%   -18.33%  -38.71%  -39.65%   -5.64%  -10.50%   -4.65%
2026     -8.90%    -5.65%  -10.02%  -18.38%   -2.35%  -14.24%   -3.88%
```
2021 is the worst year for MomAlloc (-28.82%) — Book A dropped -48.95% (UVXY drag in calm bull market).
Book D has never exceeded -6.52% intra-year drawdown in any single year.

### Correlation Matrix (daily returns, 2019-2026)

```
        A      F      D      C      B
A   1.000  0.486  0.213 -0.004  0.009
F   0.486  1.000  0.387  0.001 -0.029
D   0.213  0.387  1.000 -0.039 -0.043
C  -0.004  0.001 -0.039  1.000  0.021
B   0.009 -0.029 -0.043  0.021  1.000
```

**Pairwise (sorted):**
```
A vs F: +0.486  (highest — both momentum strategies, same universe)
F vs D: +0.387  (moderate — both hourly, same universe, different signals)
A vs D: +0.213  (low)
All other pairs: < 0.05  (near-zero, essentially independent)
```

**Diversification insight:**
- A and F share momentum signal → moderate correlation (0.49)
- C (intraday reversal on 4-sigma events) is uncorrelated with everything
- B (QQQ only) is uncorrelated with the multi-stock books
- D's bubble score is structurally different from momentum → low corr with A/F
- Near-zero average cross-correlation explains why Fixed EW Sharpe (2.34) beats all individual books

### MomAlloc Sensitivity

```
Lookback sensitivity (avg Sharpe):  20d=1.922  30d=2.049  60d=2.092  90d=1.999  120d=1.851
Rebalance sensitivity (avg Sharpe):  1d=1.945   5d=1.941  10d=2.014   20d=2.029   60d=1.985
Best combo: lb=20d, rb=60d  (Sharpe 2.170, CAGR 61.9%, MaxDD -28.8%)
```
Results are fairly robust across the grid — all 25 combos show Sharpe > 1.7.
