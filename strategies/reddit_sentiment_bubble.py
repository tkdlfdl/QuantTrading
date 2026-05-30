"""
Reddit Sentiment Bubble Strategy.

Signal: apply bubble score proxy to a sentiment "price index" per stock.
  - Sentiment too HIGH (bubble score > threshold) → SHORT
  - Sentiment too LOW  (bubble score < threshold) → LONG

Grid search: holding_period, short_threshold, long_threshold, top_n,
             ma_window, z_window, sentiment_scale.

Backtest uses actual price returns (from yfinance panel).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

from backtest.metrics import performance_stats, yearly_performance_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def calculate_bubble_score_proxy(
    price: pd.Series,
    ma_window: int = 252,
    z_window: int = 252,
) -> pd.Series:
    price = price.replace(0, np.nan).ffill()
    log_price     = np.log(price)
    fair_value    = price.rolling(ma_window).mean()
    log_fair_value = np.log(fair_value)
    residual      = log_price - log_fair_value
    z = (residual - residual.rolling(z_window).mean()) / residual.rolling(z_window).std()
    return np.tanh(z / 2)


def sentiment_to_price_index(sentiment: pd.Series, scale: float = 0.05, base: float = 100.0) -> pd.Series:
    """
    Convert a bounded sentiment series (-1 to 1) to a price-like index.
    Each day: index *= (1 + sentiment * scale)
    Missing sentiment → 0 return (neutral).
    """
    daily_ret = sentiment.fillna(0).clip(-1, 1) * scale
    return base * (1 + daily_ret).cumprod()


# ---------------------------------------------------------------------------
# Main strategy
# ---------------------------------------------------------------------------

def run_reddit_sentiment_bubble(
    sentiment_panel: pd.DataFrame,   # wide: date × symbol, values = daily compound score
    price_panel: pd.DataFrame,       # wide: date × symbol, values = close prices

    # Grid search parameters
    holding_period_grid: list = [5, 10, 20, 40],
    short_threshold_grid: list = [0.5, 0.6, 0.7],
    long_threshold_grid:  list = [-0.5, -0.6, -0.7],
    top_n_grid:           list = [5, 10, 20],
    ma_window_grid:       list = [30, 60],
    z_window_grid:        list = [60, 120],
    sentiment_scale_grid: list = [0.05],

    min_mentions: int = 5,           # min total mentions for a symbol to be included
    trading_days: int = 252,
) -> tuple:
    """
    Backtest a contrarian Reddit sentiment bubble strategy.

    Returns:
        analysis, yearly_analysis, best_wealth, best_ret, grid_result_df, best_params
    """
    # ── Align date ranges ──────────────────────────────────────────────────
    common_dates = sentiment_panel.index.intersection(price_panel.index)
    if len(common_dates) < max(max(ma_window_grid), max(z_window_grid)) + 10:
        raise ValueError("Insufficient overlapping dates between sentiment and price panels.")

    sent  = sentiment_panel.loc[common_dates]
    price = price_panel.loc[common_dates]

    # Candidate symbols present in both panels
    active_symbols = [s for s in sent.columns if s in price.columns]
    sent  = sent[active_symbols]
    price = price[active_symbols]

    # Cumulative mention counts per symbol over time (used for live universe filter)
    # sent != 0 means that day had at least one post mentioning the symbol
    cumulative_mentions = (sent != 0).cumsum()

    # Daily price returns — computed once, no lookahead (pct_change is backward-looking)
    price_ret = price.pct_change().ffill().fillna(0)

    grid_results = []
    best_sharpe  = -np.inf
    best_ret     = None
    best_wealth  = None
    best_params  = None

    total_combos = (len(holding_period_grid) * len(short_threshold_grid) *
                    len(long_threshold_grid) * len(top_n_grid) *
                    len(ma_window_grid) * len(z_window_grid) * len(sentiment_scale_grid))
    print(f"Grid search: {total_combos} combinations...")

    for (holding_period, short_thresh, long_thresh, top_n,
         ma_window, z_window, sent_scale) in product(
        holding_period_grid, short_threshold_grid, long_threshold_grid,
        top_n_grid, ma_window_grid, z_window_grid, sentiment_scale_grid,
    ):
        # ── Compute bubble scores (all backward-looking rolling stats) ────────
        bubble_scores_raw = pd.DataFrame(index=common_dates, columns=active_symbols, dtype=float)
        for sym in active_symbols:
            price_idx = sentiment_to_price_index(sent[sym], scale=sent_scale)
            bubble_scores_raw[sym] = calculate_bubble_score_proxy(
                price_idx, ma_window=ma_window, z_window=z_window
            )

        # NO-LOOKAHEAD: shift scores by 1 day so that:
        #   signal_scores.loc[T] = bubble score computed at END of day T-1
        #   → signal known BEFORE market open on day T
        #   → trade executes at OPEN of day T (using day T's return)
        # Rule: "after signal detected → buy/sell NEXT day"
        signal_scores = bubble_scores_raw.shift(1)

        # ── Generate daily returns ─────────────────────────────────────────
        daily_returns = []
        warmup = z_window + ma_window + 1  # +1 for the shift

        for i in range(warmup, len(common_dates) - holding_period, holding_period):
            signal_date = common_dates[i]

            # Universe filter: only include symbols that had >= min_mentions
            # posts UP TO (and not including) signal_date → no lookahead in universe
            eligible = cumulative_mentions.loc[common_dates[i - 1]]
            eligible = eligible[eligible >= min_mentions].index
            scores = signal_scores.loc[signal_date, eligible].dropna()

            if scores.empty:
                continue

            short_candidates = scores[scores >  short_thresh].nlargest(top_n)
            long_candidates  = scores[scores <  long_thresh].nsmallest(top_n)

            # Trade from day i (signal was from day i-1 via shift)
            # price_ret[i] = (close_i - close_{i-1}) / close_{i-1}
            # → captures return of the day AFTER the signal date
            hold_end = min(i + holding_period, len(common_dates))

            for j in range(i, hold_end):
                trade_date = common_dates[j]
                ret = 0.0

                if len(long_candidates) > 0:
                    long_ret = price_ret.loc[trade_date, long_candidates.index].mean()
                    ret += long_ret * 0.5

                if len(short_candidates) > 0:
                    short_ret = -price_ret.loc[trade_date, short_candidates.index].mean()
                    ret += short_ret * 0.5

                daily_returns.append({"date": trade_date, "ret": ret})

        if not daily_returns:
            continue

        ret_series = pd.DataFrame(daily_returns).set_index("date")["ret"]
        ret_series = ret_series[~ret_series.index.duplicated(keep="last")]

        wealth = (1 + ret_series).cumprod()
        wealth = wealth / wealth.iloc[0]

        std    = ret_series.std()
        sharpe = np.sqrt(trading_days) * ret_series.mean() / std if std > 0 else np.nan
        downside = ret_series[ret_series < 0].std()
        sortino  = np.sqrt(trading_days) * ret_series.mean() / downside if downside > 0 else np.nan
        mdd      = (wealth / wealth.cummax() - 1).min()
        tot_ret  = wealth.iloc[-1] - 1

        row = {
            "holding_period":  holding_period,
            "short_threshold": short_thresh,
            "long_threshold":  long_thresh,
            "top_n":           top_n,
            "ma_window":       ma_window,
            "z_window":        z_window,
            "sentiment_scale": sent_scale,
            "Sharpe Ratio":    sharpe,
            "Sortino Ratio":   sortino,
            "Total Return":    tot_ret,
            "Max Drawdown":    mdd,
            "Final Wealth":    wealth.iloc[-1],
            "n_trades":        len(daily_returns),
        }
        grid_results.append(row)

        if pd.notna(sharpe) and sharpe > best_sharpe:
            best_sharpe  = sharpe
            best_ret     = ret_series.rename("SentimentBubble")
            best_wealth  = wealth.rename("SentimentBubble")
            best_params  = row

    if not grid_results:
        raise RuntimeError("No valid grid results — check data coverage and thresholds.")

    grid_df = pd.DataFrame(grid_results).sort_values("Sharpe Ratio", ascending=False)

    # ── Performance summary ────────────────────────────────────────────────
    ret_df_summary   = best_ret.to_frame()
    wealth_df_summary = best_wealth.to_frame()
    analysis         = performance_stats(ret_df_summary, wealth_df_summary, trading_days)
    yearly_analysis  = yearly_performance_stats(ret_df_summary, wealth_df_summary, trading_days)

    print(f"\nBest Parameters:")
    print(pd.Series(best_params).to_string())

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 14),
                             gridspec_kw={"height_ratios": [3, 1, 1]})

    axes[0].plot(best_wealth.index, best_wealth.values, linewidth=2, label="Sentiment Bubble Strategy")
    axes[0].set_title(f"Reddit Sentiment Bubble | top_n={best_params['top_n']} | "
                      f"hold={best_params['holding_period']}d | "
                      f"short>{best_params['short_threshold']} long<{best_params['long_threshold']}")
    axes[0].set_ylabel("Cumulative Wealth")
    axes[0].legend()
    axes[0].grid(True)

    # Sample bubble scores for top symbol by mention count (shifted — same as signal)
    sample_sym = (sent != 0).sum().idxmax()
    sample_price_idx = sentiment_to_price_index(
        sent[sample_sym],
        scale=best_params["sentiment_scale"]
    )
    sample_bubble = calculate_bubble_score_proxy(
        sample_price_idx,
        ma_window=best_params["ma_window"],
        z_window=best_params["z_window"],
    ).shift(1)  # match signal_scores shift
    axes[1].plot(sample_bubble.index, sample_bubble.values, label=f"Sentiment Bubble Score: {sample_sym}")
    axes[1].axhline(best_params["short_threshold"], linestyle="--", color="red",  label="Short threshold")
    axes[1].axhline(best_params["long_threshold"],  linestyle="--", color="green", label="Long threshold")
    axes[1].axhline(0, linestyle="--", linewidth=0.8)
    axes[1].set_ylim(-1, 1)
    axes[1].set_ylabel("Bubble Score")
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    dd = (best_wealth / best_wealth.cummax() - 1)
    axes[2].fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
    axes[2].set_ylabel("Drawdown")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    return analysis, yearly_analysis, best_wealth, best_ret, grid_df, best_params
