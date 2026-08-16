# Improvements Log — what improved performance, and by how much

One entry per verified improvement: the idea/paper responsible and the
exact metric deltas vs its baseline. Appended automatically by
`tools.record.record_improvement`; newest entries at the bottom.

### 2026-08-15 — Portfolio vol-target overlay on inverse-vol weights (drop Book B)
- **Source:** Moreira & Muir 2017 JF; Harvey et al. 2018 JPM; Qian 2005 (registry #10, #11, #13)
- **Sharpe:** 2.060 → 2.501  (**+0.441**)
- **CAGR:**   48.0% → 45.2%  (-2.8%)
- **MaxDD:**  -19.0% → -8.9%  (+10.1%)
- **Records:** portfolio_ivol_voltgt_nob
- **Notes:** Baseline = honest Fixed EW A+F+D+C+B. Single biggest improvement to date: +0.44 Sharpe, half the drawdown.

### 2026-08-15 — Daniel-Moskowitz panic gate on champion
- **Source:** Daniel & Moskowitz 2016 JFE (registry #27)
- **Sharpe:** 2.501 → 2.514  (**+0.013**)
- **CAGR:**   45.2% → 45.3%  (+0.1%)
- **MaxDD:**  -8.9% → -8.9%  (+0.0%)
- **Records:** portfolio_champ_panic_nob
- **Notes:** Marginal (+0.013): only 11 panic days in 2019-2026 window; kept as insurance.

### 2026-08-15 — Quarter-Kelly with double shrinkage (no B)
- **Source:** Kelly 1956; MacLean-Thorp-Ziemba 2011; Ledoit-Wolf 2004 (registry #17, #18)
- **Sharpe:** 2.013 → 2.098  (**+0.085**)
- **CAGR:**   57.3% → 44.7%  (-12.6%)
- **MaxDD:**  -21.6% → -19.8%  (+1.8%)
- **Records:** portfolio_qkelly_nob
- **Notes:** Baseline = Fixed EW no-B. Improves Sharpe/DD but dominated by the vol-target champion.

### 2026-08-15 — Risk-managed momentum overlay on Book F (10% vol target)
- **Source:** Barroso & Santa-Clara 2015 JFE (registry #12)
- **Sharpe:** 1.548 → 1.609  (**+0.061**)
- **CAGR:**   75.7% → 18.4%  (-57.3%)
- **MaxDD:**  -39.8% → -10.3%  (+29.5%)
- **Records:** book_f_riskmanaged
- **Notes:** Standalone F. Sharpe up, DD quartered, but CAGR collapses 76%->18% — portfolio-level overlay is the better home for this mechanism.

### 2026-08-15 — Book C per-ticker overlap cap (Improvement Cycle 1)
- **Source:** ORIGINAL — verifier finding, no paper (registry #41)
- **Sharpe:** 0.592 → 0.675  (**+0.083**)
- **CAGR:**   21.0% → 18.8%  (-2.2%)
- **MaxDD:**  -53.7% → -33.0%  (+20.7%)
- **Records:** book_c_overlap_cap, portfolio_champion_cappedC
- **Notes:** Champion effect: 2.514 -> 2.580 Sharpe at unchanged -8.9% MaxDD. Applied to live C engine 2026-08-15.

### 2026-08-16 — NEW BOOK candidate: oversold-breadth capitulation index timing
- **Source:** ORIGINAL — no paper; synthesis of Book D bubble score at market level
- **Sharpe:** 0.000 → 0.896  (**+0.896**)
- **CAGR:**   0.0% → 12.5%  (+12.5%)
- **MaxDD:**  0.0% → -23.4%  (-23.4%)
- **Records:** book_h_breadth_timing
- **Notes:** hold 20d; corr {'A': 0.25861050889914233, 'C': 0.02390735944951421, 'D': 0.3776107927926821, 'F': 0.2958182303808347}; PENDING: incubation per Phase 2, human sign-off

### 2026-08-16 — HAR-X vol forecast (SPY intraday RV) for champion overlay
- **Source:** Corsi 2009 + own intraday RV (registry #60 revisit)
- **Sharpe:** 2.580 → 2.632  (**+0.052**)
- **CAGR:**   43.5% → 38.9%  (-4.6%)
- **MaxDD:**  -8.9% → -8.9%  (+0.0%)
- **Records:** portfolio_champ_harx

### 2026-08-16 — D14 aggressive sleeve added as fifth book
- **Source:** Own grid finding (registry #45 note)
- **Sharpe:** 2.580 → 2.781  (**+0.201**)
- **CAGR:**   43.5% → 43.9%  (+0.4%)
- **MaxDD:**  -8.9% → -8.2%  (+0.7%)
- **Records:** portfolio_champ_d14
