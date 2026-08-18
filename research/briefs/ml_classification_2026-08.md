# Strategy Brief — ML Next-Day Classification Program (Cycle 27, 2026-08-17)

User-directed: predict next-day return CLASS from historical features.

## Label (user spec, fixed)

For instrument i, day t: with trailing 63d mean mu and std sigma of daily
returns (through t, no look-ahead),
  y = +1  if r[t+1] >  mu + k*sigma     (BUY)
  y = -1  if r[t+1] <  mu - k*sigma     (SHORT)
  y =  0  otherwise                     (FLAT/EXIT)
k in {1.0, 1.5, 2.0} — all three reported, no argmax adoption. Position on
day t+1 = predicted class at t. Costs 0.1%/side on position changes; 8%/yr
borrow while short.

## Literature anchors (CS/ML + finance-ML)

- Krauss, Do & Huck (2017, EJOR): DNN/GBT/RF ensembles, daily S&P 500
  classification — edge strong pre-2001, decayed after.
- Fischer & Krauss (2018, EJOR): LSTM daily direction classification —
  outperformed memory-free models; returns decay post-2010.
- Gu, Kelly & Xiu (2020, RFS): trees + shallow nets dominate linear models
  for return prediction; signal mostly in longer horizons.
- Lopez de Prado (2018): threshold/triple-barrier labeling, purged CV —
  labeling here is the fixed-horizon threshold variant.
Expectation set by this literature: daily classification is largely dead
after costs in the modern era. The test's job is to confirm or surprise.

## Model ladder (simple -> complex, all pre-registered)

M1  Multinomial logistic. Features: 5 lagged returns + trailing 21d vol. SPY.
M2  Multinomial logistic, rich features (12): lagged returns (1,2,3,5d),
    momentum (21/63/126/252d), vol 21d, vol-of-vol, ^VIX level z + 5d change,
    universe breadth (% above 20d MA), ToM flag. SPY.
M3  Random Forest (400 trees, depth 6), M2 features. SPY.
M4  HistGradientBoosting, M2 features. SPY.
M5  Pooled cross-sectional HistGB on the full daily panel (500+ names,
    2000+): per-stock features (lags, momentum set, vol, gap, volume z,
    market features), per-stock class prediction; portfolio = equal-weight
    long all +1 / short all -1 predictions, daily.

## Feature set v2 (user-directed revision, 2026-08-17)

After M1-M4 v1 (base features) confirmed the literature's negative result,
the user directed a richer feature set before the M5 retry. Added per
instrument: price/MA ratios (20/50/200), MA20/50 cross, RSI(14), Bollinger
z(20), MACD(12,26,9) normalized, stochastic %K(14), 52-week high/low
distance, ATR14/price, vol5/vol60 ratio, 63d skew. Added market context:
^VIX9D/^VIX and ^VIX/^VIX3M term-structure ratios, 10y yield 21d change,
10y-13w curve slope, breadth(50d), ToM flag. Same models, same walk-forward,
same no-tuning rule; both the SPY ladder and M5 rerun on v2 features.

## Validation protocol

Walk-forward: expanding window, refit each Jan (M5: every 2 years, train
capped at trailing 8y for tractability), first prediction year 2005 (SPY) /
2012 (panel). NO hyperparameter search — fixed a-priori settings above.
Report per model x k: class base rates, out-of-sample accuracy vs majority
class, after-cost Sharpe/CAGR/MaxDD, turnover, 2015+ subperiod.
Gates for any survivor: full-stack marginal @0.25 + LW p<0.10.
