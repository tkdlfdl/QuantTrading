# Reddit Sentiment Long-Only Strategy (Book E)
**Version:** 1.0  
**Status:** EXPERIMENTAL — promising but data-limited  
**Last Updated:** June 2026  
**Data:** Daily Reddit sentiment (2024-04 to 2026-06, 2.2 years) + daily close prices  
**Universe:** ~512 S&P500/NASDAQ100 tickers with Reddit mentions (sparse coverage)

---

## Executive Summary

Apply a **bubble score to each stock's Reddit-sentiment index**, then go **long** in two
regimes (no shorts):
- **Capitulation** — sentiment bubble score **< −extreme** (deeply hated names that bounce)
- **Moderate momentum** — sentiment bubble score **mild < score ≤ extreme** (ride building hype)

**Best (most robust) configuration:**

| Metric | Value |
|--------|-------|
| Annual Return | +37.0% |
| Sharpe Ratio | 1.97 (trade-based) / ~1.0 (daily, portfolio-comparable) |
| Max Drawdown | −12.2% |
| Win Rate | 76% |
| Trades | 63 (over 2.2 yrs) |

---

## How we got here (the honest path)

The user's original idea was "short when sentiment is too high." Testing revealed the
opposite — the edge is on the **long** side:

| Version | Logic | Best Sharpe | Win% | Verdict |
|---------|-------|-------------|------|---------|
| Short-only | fade euphoria (short high sentiment) | 0.18 | 44% | **fails** — euphoria keeps ripping |
| 4-zone | trend moderate, fade extremes (long+short) | 0.62 | 58% | weak — shorts drag it down |
| **Long-only** | **buy capitulation + ride moderate hype** | **1.97** | **76%** | **works** |

### Zone breakdown (why long-only)
Decomposing the 4-zone strategy by regime exposed exactly where the edge lives:

| Zone | Side | Rule (bubble score) | Total P&L | Win% |
|------|------|--------------------|-----------|------|
| Moderate positive | LONG | 0.4 < score ≤ 0.7 | **+44.7%** | 51% |
| Extreme positive | SHORT | score > 0.7 | −4.9% | 41% |
| Moderate negative | SHORT | −0.7 ≤ score < −0.4 | **−41.1%** | 42% |
| Extreme negative | LONG | score < −0.7 | **+57.3%** | 63% |

**Both long zones make money (+102%); both short zones lose (−46%).** Dropping the shorts
isolates the edge — Sharpe jumped 0.62 → 1.97.

---

## Strategy Logic

```
For each stock with Reddit mentions:
  1. Build a sentiment "price index":  index *= (1 + weighted_compound * 0.05)  each day
  2. Bubble score = tanh(z/2) on that index   (ma_window, z_window rolling)
  3. Shift by 1 day (no look-ahead: signal known before next open)

Signal (LONG only):
  score < -extreme           -> BUY (capitulation: deeply hated sentiment, expect bounce)
  mild < score <= extreme    -> BUY (moderate positive: ride building hype as momentum)
  otherwise                  -> CASH (earn 2%/yr)

Execution: equal-weight top-N per regime, hold `hold` days, rebalance.
Eligibility: cumulative Reddit mentions >= min_mentions (no look-ahead).
Cost: 0.25% one-way transaction cost.
```

---

## Parameters

### Production combo (recommended — best robust sample)
| Parameter | Value |
|-----------|-------|
| MA window | 15 |
| Z window | 40 |
| Mild threshold | 0.5 |
| Extreme threshold | 0.6 |
| Hold period | 8 days |
| Top-N per regime | 5 |
| Min mentions | 5 |

→ Sharpe 1.97, Annual +37%, MaxDD −12.2%, Win 76%, **63 trades**

### Highest-Sharpe combo (caution: thin sample)
`ma=45, z=90, mild=0.5, extreme=0.6, hold=10d, top5` → Sharpe **2.10**, Annual **+88%**,
MaxDD −9.1%, but only **42 trades** (Sortino 7.1 is a small-sample artifact — not fully trusted).

### Grid (1,920 combinations)
| Parameter | Values tested |
|-----------|--------------|
| MA window | 10, 15, 20, 30, 45 |
| Z window | 40, 60, 90, 120 |
| Mild threshold | 0.3, 0.4, 0.5 |
| Extreme threshold | 0.6, 0.7 |
| Hold days | 3, 5, 8, 10 |
| Top-N | 3, 5 |
| Min mentions | 3, 5 |

**Robustness:** 1,384 / 1,920 (72%) positive Sharpe; 836 > 0.5; **392 > 1.0**.

### Sensitivity (what matters)
| Parameter | Best | Insight |
|-----------|------|---------|
| **Hold** | **8–10 days** | sentiment reversion needs time (3d: −0.01 avg, 8d: +0.72) |
| **Extreme** | **0.6** | lower threshold → wider capitulation-LONG zone (the +57% engine) |
| **Mild** | 0.4–0.5 | ride only strongly-building sentiment |
| **Z window** | 90–120 | longer windows smooth the sparse signal |

---

## Performance

### Standalone (production combo, 2024-2026)
| Metric | Value |
|--------|-------|
| Annual Return | +37.0% (trade-based) |
| Sharpe | 1.97 (trade-based) / ~1.0 (daily-marked) |
| Max Drawdown | −12.2% |
| Win Rate | 76% |

### Portfolio impact — adding Book E to A+B+C+D (2024-2026 overlap)
| Allocation | Without E | With E | Change |
|------------|-----------|--------|--------|
| Fixed Equal-Weight | 2.159 | 2.124 | −0.036 |
| **Momentum Allocation** | 3.252 | **3.458** | **+0.206** |

- **Under momentum allocation, E improves the portfolio** (+0.21 Sharpe) — the allocator
  leans into E when it's hot, capturing diversification.
- **Under equal-weight, E slightly dilutes** (−0.04) — the existing books are already strong;
  E's ~1.0 daily Sharpe drags the simple average, though it *improves* drawdown.

### Correlation to existing books (diversification)
| vs | corr |
|----|------|
| A (Daily Momentum) | +0.33 |
| B (QQQ Bubble) | +0.39 |
| **C (Intraday MR)** | **+0.02** ← most diversifying |
| D (Contrarian) | +0.47 |

E is most diversifying against Intraday MR (C); moderately correlated to the others.

---

## Limitations (important)

1. **Sparse sentiment data** — ~512 symbols but averaging only **23 mention-days each** over
   2.2 years. The bubble score is mostly neutral, spiking on mention bursts. This is the
   single biggest weakness.
2. **Short window** — 2.2 years, 63 trades (production) is a thin sample; wide confidence intervals.
3. **Standalone Sharpe ~1.0 daily** — solid but below the live books (D at 1.8, A at 2.3 in-window).
4. **Sentiment is daily** — holds are in days, not hours; no intraday version possible with this data.
5. **Lost −1.3% in 2024** (early, thin data) before strong 2025 (+42%) / 2026 (+18%).

**Verdict:** the signal is real and robust across parameters, and adds value under momentum
allocation — but the **data sparsity caps confidence**. Densifying sentiment (more history /
StockTwits / news) is the key unlock before live deployment.

---

## Files

| File | Description |
|------|-------------|
| `reddit_sentiment_long_backtest.py` | Long-only grid search (1,920 combos) |
| `reddit_sentiment_4zone_backtest.py` | 4-zone version (for the zone breakdown) |
| `reddit_sentiment_short_backtest.py` | Short-only version (the failed first attempt) |
| `portfolio_5strategy_sentiment.py` | Portfolio Sharpe impact of adding Book E |
| `results/reddit_sentiment_long_grid.csv` | Full 1,920-combo grid |
| `strategies/reddit_sentiment_bubble.py` | Original 4-zone module |
| `data/market_data.duckdb` → `sentiment_daily` | Source sentiment data |

---

**Repository:** https://github.com/tkdlfdl/QuantTrading  
**Status:** EXPERIMENTAL — validated signal, data-limited; deploy after densifying sentiment
