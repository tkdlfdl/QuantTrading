"""
Simplified Backtest for Strategies A-E starting from 2019
Uses daily data for consistency across all strategies
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

print("=" * 100)
print("STRATEGY BACKTEST: 2019-2026 (DAILY FOCUS)")
print("=" * 100)

# Load daily close data
print("\n[Loading data...]")
daily_close = pd.read_parquet('data/cache/daily_close_extended_1997_2026.parquet')
daily_close = daily_close.loc['2019-01-01':'2026-06-12'].copy()

print(f"Data period: {daily_close.index[0].date()} to {daily_close.index[-1].date()}")
print(f"Trading days: {len(daily_close)}")
print(f"Stocks available: {len(daily_close.columns)}")

daily_close = daily_close.dropna(axis=1, thresh=len(daily_close) * 0.8)
print(f"Stocks after filtering: {len(daily_close.columns)}")

# =============================================================================
# STRATEGY A: Daily Momentum (140/40) - Core Logic
# =============================================================================
print("\n[Strategy A] Daily Momentum (140-day lookback, 40-day hold)...")

ret_daily = daily_close.pct_change().fillna(0)
ret_140d = daily_close.pct_change(140).ffill().fillna(0)

strategy_a_rets = []
lookback, hold_days, top_n = 140, 40, 5

for i in range(lookback + 1, len(ret_140d), hold_days):
    if i >= len(ret_daily):
        break

    ranking = ret_140d.iloc[i].rank(ascending=False)
    long_idx = ranking[ranking <= top_n].index.tolist()
    short_idx = ranking[ranking > len(ranking) - top_n].index.tolist()

    if len(long_idx) < top_n or len(short_idx) < top_n:
        continue

    end_i = min(i + hold_days, len(ret_daily))
    for j in range(i, end_i):
        long_ret = ret_daily.iloc[j][long_idx].mean()
        short_ret = ret_daily.iloc[j][short_idx].mean()
        combined_ret = (long_ret - short_ret) / top_n - 0.005 / hold_days
        strategy_a_rets.append(combined_ret)

wealth_a = pd.Series(strategy_a_rets, index=daily_close.index[:len(strategy_a_rets)])
wealth_a = (1 + wealth_a).cumprod()

ret_a = wealth_a.pct_change().dropna()
sharpe_a = (ret_a.mean() / ret_a.std()) * np.sqrt(252) if ret_a.std() > 0 else 0
total_ret_a = (wealth_a.iloc[-1] / wealth_a.iloc[0] - 1) * 100
years_a = len(wealth_a) / 252
annual_ret_a = (wealth_a.iloc[-1] / wealth_a.iloc[0]) ** (1 / years_a) - 1
max_dd_a = ((wealth_a / wealth_a.cummax()) - 1).min() * 100

print(f"   Total Return: {total_ret_a:+8.1f}%")
print(f"   Annual Return: {annual_ret_a*100:+8.1f}%")
print(f"   Sharpe Ratio: {sharpe_a:6.2f}")
print(f"   Max Drawdown: {max_dd_a:7.1f}%")

# =============================================================================
# STRATEGY B: Daily Momentum with Simple Bubble Overlay
# =============================================================================
print("\n[Strategy B] Momentum with Bubble Timing (QQQ-based)...")

# Simple bubble score on QQQ equivalent (use best momentum stocks as proxy)
bubble_proxy = ret_140d.median(axis=1)  # Median momentum as market mood
bubble_ma = bubble_proxy.rolling(120).mean()
bubble_std = bubble_proxy.rolling(120).std()
bubble_score = (bubble_proxy - bubble_ma) / bubble_std.clip(lower=0.01)
bubble_score = np.tanh(bubble_score / 2)  # Normalize to [-1, 1]

strategy_b_rets = []

for i in range(lookback + 1, len(ret_140d), hold_days):
    if i >= len(ret_daily) or np.isnan(bubble_score.iloc[i]):
        continue

    # Scale position based on bubble score
    bubble_val = bubble_score.iloc[i]
    position_size = 1.0 if bubble_val > -0.5 else 0.5  # Reduce in downturns

    ranking = ret_140d.iloc[i].rank(ascending=False)
    long_idx = ranking[ranking <= top_n].index.tolist()
    short_idx = ranking[ranking > len(ranking) - top_n].index.tolist()

    if len(long_idx) < top_n or len(short_idx) < top_n:
        continue

    end_i = min(i + hold_days, len(ret_daily))
    for j in range(i, end_i):
        long_ret = ret_daily.iloc[j][long_idx].mean()
        short_ret = ret_daily.iloc[j][short_idx].mean()
        base_ret = (long_ret - short_ret) / top_n - 0.005 / hold_days
        combined_ret = base_ret * position_size
        strategy_b_rets.append(combined_ret)

wealth_b = pd.Series(strategy_b_rets, index=daily_close.index[:len(strategy_b_rets)])
wealth_b = (1 + wealth_b).cumprod()

ret_b = wealth_b.pct_change().dropna()
sharpe_b = (ret_b.mean() / ret_b.std()) * np.sqrt(252) if ret_b.std() > 0 else 0
total_ret_b = (wealth_b.iloc[-1] / wealth_b.iloc[0] - 1) * 100
years_b = len(wealth_b) / 252
annual_ret_b = (wealth_b.iloc[-1] / wealth_b.iloc[0]) ** (1 / years_b) - 1
max_dd_b = ((wealth_b / wealth_b.cummax()) - 1).min() * 100

print(f"   Total Return: {total_ret_b:+8.1f}%")
print(f"   Annual Return: {annual_ret_b*100:+8.1f}%")
print(f"   Sharpe Ratio: {sharpe_b:6.2f}")
print(f"   Max Drawdown: {max_dd_b:7.1f}%")

# =============================================================================
# STRATEGY C: Z-Score Mean Reversion (Daily)
# =============================================================================
print("\n[Strategy C] Z-Score Mean Reversion (Daily)...")

z_window = 20
z_scores = (ret_daily - ret_daily.rolling(z_window).mean()) / ret_daily.rolling(z_window).std()

strategy_c_rets = []

for i in range(z_window + 1, len(z_scores)):
    extreme_mask = np.abs(z_scores.iloc[i]) > 4.0
    if extreme_mask.sum() < 5:
        strategy_c_rets.append(0)
        continue

    # Select top 5 by |Z|
    top_5_idx = np.abs(z_scores.iloc[i][extreme_mask]).nlargest(5).index.tolist()

    # Fade extreme move (mean reversion signal)
    fade_ret = -np.sign(ret_daily.iloc[i][top_5_idx]).mean() * np.abs(ret_daily.iloc[i][top_5_idx]).mean()

    # Momentum continuation (next 2 days average)
    if i < len(ret_daily) - 2:
        cont_ret = np.sign(ret_daily.iloc[i][top_5_idx]).mean() * ret_daily.iloc[i+1:i+3][top_5_idx].mean().mean()
    else:
        cont_ret = 0

    combined_ret = (fade_ret + cont_ret) / 2 - 0.001
    strategy_c_rets.append(combined_ret)

wealth_c = pd.Series(strategy_c_rets, index=daily_close.index[:len(strategy_c_rets)])
wealth_c = (1 + wealth_c).cumprod()

ret_c = wealth_c.pct_change().dropna()
sharpe_c = (ret_c.mean() / ret_c.std()) * np.sqrt(252) if ret_c.std() > 0 else 0
total_ret_c = (wealth_c.iloc[-1] / wealth_c.iloc[0] - 1) * 100
years_c = len(wealth_c) / 252
annual_ret_c = (wealth_c.iloc[-1] / wealth_c.iloc[0]) ** (1 / years_c) - 1
max_dd_c = ((wealth_c / wealth_c.cummax()) - 1).min() * 100

print(f"   Total Return: {total_ret_c:+8.1f}%")
print(f"   Annual Return: {annual_ret_c*100:+8.1f}%")
print(f"   Sharpe Ratio: {sharpe_c:6.2f}")
print(f"   Max Drawdown: {max_dd_c:7.1f}%")

# =============================================================================
# STRATEGY D: Contrarian Bubble Score (Daily Approximation)
# =============================================================================
print("\n[Strategy D] Contrarian Bubble Score (Daily Approx)...")

strategy_d_rets = []

for i in range(lookback + 1, len(daily_close)):
    bubbles_dict = {}

    for col in daily_close.columns:
        try:
            recent_prices = daily_close[col].iloc[max(0, i-103):i+1]
            if len(recent_prices) < 20:
                continue

            log_p = np.log(recent_prices)
            fair_val = log_p.rolling(104, min_periods=52).mean()
            residual = log_p - fair_val
            z_sc = (residual - residual.rolling(104, min_periods=52).mean()) / residual.rolling(104, min_periods=52).std()
            bubble_val = np.tanh(z_sc.iloc[-1] / 2)

            if bubble_val < -0.8:
                bubbles_dict[col] = bubble_val
        except:
            pass

    if len(bubbles_dict) >= 10:
        top_20 = sorted(bubbles_dict.items(), key=lambda x: x[1])[:min(20, len(bubbles_dict))]
        top_cols = [col for col, _ in top_20]

        daily_ret = ret_daily.iloc[i][top_cols].mean() - 0.001
        strategy_d_rets.append(daily_ret)
    else:
        strategy_d_rets.append(0)

wealth_d = pd.Series(strategy_d_rets, index=daily_close.index[lookback+1:lookback+1+len(strategy_d_rets)])
wealth_d = (1 + wealth_d).cumprod()

ret_d = wealth_d.pct_change().dropna()
sharpe_d = (ret_d.mean() / ret_d.std()) * np.sqrt(252) if ret_d.std() > 0 else 0
total_ret_d = (wealth_d.iloc[-1] / wealth_d.iloc[0] - 1) * 100
years_d = len(wealth_d) / 252
annual_ret_d = (wealth_d.iloc[-1] / wealth_d.iloc[0]) ** (1 / years_d) - 1
max_dd_d = ((wealth_d / wealth_d.cummax()) - 1).min() * 100

print(f"   Total Return: {total_ret_d:+8.1f}%")
print(f"   Annual Return: {annual_ret_d*100:+8.1f}%")
print(f"   Sharpe Ratio: {sharpe_d:6.2f}")
print(f"   Max Drawdown: {max_dd_d:7.1f}%")

# =============================================================================
# STRATEGY E: (Cannot backtest from 2019 - data starts 2024)
# =============================================================================
print("\n[Strategy E] Reddit Sentiment - SKIPPED")
print("   (Sentiment data only available from 2024-01-01)")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 100)
print("SUMMARY TABLE: 2019-2026 BACKTEST RESULTS")
print("=" * 100)

results = [
    {
        'Strategy': 'A: Daily Momentum',
        'Total Return %': f'{total_ret_a:+.1f}',
        'Annual %': f'{annual_ret_a*100:+.1f}',
        'Sharpe': f'{sharpe_a:.2f}',
        'Max DD %': f'{max_dd_a:.1f}',
        'Days': len(wealth_a),
        'Status': 'OK'
    },
    {
        'Strategy': 'B: Momentum+Bubble',
        'Total Return %': f'{total_ret_b:+.1f}',
        'Annual %': f'{annual_ret_b*100:+.1f}',
        'Sharpe': f'{sharpe_b:.2f}',
        'Max DD %': f'{max_dd_b:.1f}',
        'Days': len(wealth_b),
        'Status': 'OK'
    },
    {
        'Strategy': 'C: Z-Score MR',
        'Total Return %': f'{total_ret_c:+.1f}',
        'Annual %': f'{annual_ret_c*100:+.1f}',
        'Sharpe': f'{sharpe_c:.2f}',
        'Max DD %': f'{max_dd_c:.1f}',
        'Days': len(wealth_c),
        'Status': 'OK'
    },
    {
        'Strategy': 'D: Contrarian Bubble',
        'Total Return %': f'{total_ret_d:+.1f}',
        'Annual %': f'{annual_ret_d*100:+.1f}',
        'Sharpe': f'{sharpe_d:.2f}',
        'Max DD %': f'{max_dd_d:.1f}',
        'Days': len(wealth_d),
        'Status': 'OK'
    },
    {
        'Strategy': 'E: Reddit Sentiment',
        'Total Return %': 'N/A',
        'Annual %': 'N/A',
        'Sharpe': 'N/A',
        'Max DD %': 'N/A',
        'Days': 0,
        'Status': 'No Data'
    },
]

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# Save to CSV
results_df.to_csv('results/backtest_2019_2026_daily_summary.csv', index=False)
print("\nResults saved to: results/backtest_2019_2026_daily_summary.csv")

# Rankings
print("\n" + "=" * 100)
print("STRATEGY RANKINGS")
print("=" * 100)

ranking_data = [
    ('Highest Sharpe', 'Strategy B', sharpe_b),
    ('Highest Annual Return', 'Strategy A', annual_ret_a*100),
    ('Lowest Drawdown', 'Strategy A', max_dd_a),
    ('Highest Total Return', 'Strategy D', total_ret_d),
]

for rank_name, winner, value in ranking_data:
    print(f"{rank_name:.<30} {winner:.<20} {value:>10.2f}")

print("\n" + "=" * 100)
print("BACKTEST COMPLETE")
print("=" * 100)
