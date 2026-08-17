# Strategy Brief — Structural-Hole Sweep (Cycle 24, 2026-08-17)

Queue-refill round after Cycle 23 emptied the queue. Selection principle:
the full-stack saturation lesson says correlated candidates die on the
marginal test, so every candidate here occupies a cell the current stack
leaves empty (structure, not parameter variation).

## Current stack coverage map

| Cell | Occupied by |
|------|-------------|
| Single-name hourly reversal | D-complex (8h/14h/DU) |
| Single-name daily/hourly momentum | A (40d), F (750h) |
| Cross-asset monthly momentum | X |
| Overnight-clientele cross-section | G |
| Intraday MR + flip | C |
| **Market-residual momentum** | — |
| **Defensive / low-vol** | — |
| **Sector-level weekly reversal** | — |
| **Earnings events** | — (data?) |

## Candidates

### (a) Residual momentum — Blitz, Huij & Martens (2011), J. Empirical Finance
- **Signal:** 252d rolling beta vs SPY; momentum = sum of residual returns
  over months t-12..t-2, scaled by residual vol; top-20, monthly, long-only.
- **Claim:** similar premium to raw momentum at roughly half the factor
  exposure -> lower corr to A/F is the entire case.
- **Kill risk:** it is still momentum; corr(F) decides.

### (b) Low-volatility long-only — Blitz & van Vliet (2007), JPM
- **Signal:** bottom-50 by 252d realized vol, monthly, equal weight.
- **Claim:** flat-to-market return at ~70% of the vol; anti-cyclical tilt.
- **Case:** only structurally defensive equity book available; stress-day
  behavior is the metric that matters.

### (c) Sector weekly reversal — short-horizon reversal literature at the
  aggregate level (Lehmann 1990 horizon, sector aggregation)
- **Signal:** long worst-2 of 11 SPDRs by trailing 5d return, hold 5d,
  non-overlapping.
- **Case:** empty frequency x aggregation cell; sector ETFs are cheap to
  trade; capacity effectively unlimited.

### (d) Earnings-announcement premium — Frazzini & Lamont (2007)
- **Signal:** hold names through scheduled announcements.
- **Gate:** yfinance earnings-date history depth probe; if median start
  > 2023 -> DATA-BLOCKED, no approximation.

### (e) Book F overnight timing — Lou, Polk & Skouras (2019) applied to F
  (the pending item from the literature program)
- **Diagnostic only:** split F's holding-period log return into overnight vs
  intraday. An overnight-only variant pays ~2x daily costs, so the split has
  to be lopsided before any trading test is justified.

## Evaluation plan

Standard gates, quiet-D full stack as the redundancy bar: standalone from
earliest data (min 2yr), corr vs champion, stress-day mean, full-stack
marginal @0.25 shares via engine.ivol_voltgt, LW p<0.10.
Costs 0.1%/side research basis; signals shifted; splice guard (>100%
trailing-year move exclusion) on all single-name screens.
