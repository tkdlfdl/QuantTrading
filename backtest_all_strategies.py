"""
Complete Strategy Backtests - All Strategies A-E
Longest available data for each
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Load data
daily_close = pd.read_parquet('data/cache/daily_close_extended_1997_2026.parquet')
daily_close = daily_close.dropna(axis=1, thresh=len(daily_close) * 0.80)
daily_ret = daily_close.pct_change().fillna(0)

# ============================================================================
# STRATEGY B: QQQ Bubble (2020-2026 - hourly data availability)
# ============================================================================
print("\n" + "="*120)
print("STRATEGY B: QQQ Bubble Hourly Momentum (2020-2026)")
print("="*120)

daily_close_b = daily_close.loc['2020-07-27':]
daily_ret_b = daily_close_b.pct_change().fillna(0)

# Create bubble score approximation
ret_140d_b = daily_close_b.pct_change(140).ffill().fillna(0)
bubble_proxy = ret_140d_b.median(axis=1)
bubble_ma = bubble_proxy.rolling(100).mean()  # Approximate 500h with 100d
bubble_std = bubble_proxy.rolling(100).std()
bubble_score = np.tanh((bubble_proxy - bubble_ma) / bubble_std.clip(lower=0.01) / 2)

strategy_b_daily = []
for i in range(100, len(daily_close_b)):
    if bubble_score.iloc[i] < -0.8 and i < len(daily_ret_b):
        # Select top 5 by momentum
        momentum = daily_ret_b.iloc[i]
        if len(momentum) >= 5:
            top_5_idx = momentum.nlargest(5).index.tolist()
            daily_return = daily_ret_b.iloc[i][top_5_idx].mean() - 0.001
        else:
            daily_return = 0
    else:
        daily_return = 0
    strategy_b_daily.append(daily_return)

daily_returns_b = pd.Series(strategy_b_daily, index=daily_close_b.index[100:])
wealth_b = (1 + daily_returns_b).cumprod()

print(f"Period: {daily_returns_b.index[0].date()} to {daily_returns_b.index[-1].date()}")
print(f"Annual Return: {((wealth_b.iloc[-1] / wealth_b.iloc[0]) ** (252 / len(daily_returns_b)) - 1):+.2%}")
print(f"Sharpe: {(daily_returns_b.mean() / daily_returns_b.std() * np.sqrt(252) if daily_returns_b.std() > 0 else 0):.2f}")

daily_returns_b.to_csv('results/Strategy_B_Daily_Returns_2020_2026.csv', header=['Daily_Return'])
print("  [OK] Strategy_B_Daily_Returns_2020_2026.csv saved")

# ============================================================================
# STRATEGY C: Intraday MR (2019-2026)
# ============================================================================
print("\n" + "="*120)
print("STRATEGY C: Intraday Mean Reversion (2019-2026)")
print("="*120)

daily_close_c = daily_close.loc['2019-01-01':]
daily_ret_c = daily_close_c.pct_change().fillna(0)

z_window = 20
strategy_c_daily = []

for i in range(z_window + 5, len(daily_ret_c)):
    z_scores = (daily_ret_c.iloc[i] - daily_ret_c.iloc[i-z_window:i].mean()) / daily_ret_c.iloc[i-z_window:i].std()
    extreme_mask = np.abs(z_scores) > 4.0
    
    if extreme_mask.sum() < 5:
        strategy_c_daily.append(0)
        continue
    
    top_5_idx = np.abs(z_scores[extreme_mask]).nlargest(5).index.tolist()
    fade_ret = -np.sign(daily_ret_c.iloc[i][top_5_idx]).mean() * np.abs(daily_ret_c.iloc[i][top_5_idx]).mean()
    
    if i < len(daily_ret_c) - 3:
        cont_ret = np.sign(daily_ret_c.iloc[i][top_5_idx]).mean() * daily_ret_c.iloc[i+1:i+4][top_5_idx].mean().mean()
    else:
        cont_ret = 0
    
    daily_return = (fade_ret + cont_ret) / 2 - 0.001
    strategy_c_daily.append(daily_return)

daily_returns_c = pd.Series(strategy_c_daily, index=daily_close_c.index[z_window+5:z_window+5+len(strategy_c_daily)])
wealth_c = (1 + daily_returns_c).cumprod()

print(f"Period: {daily_returns_c.index[0].date()} to {daily_returns_c.index[-1].date()}")
print(f"Annual Return: {((wealth_c.iloc[-1] / wealth_c.iloc[0]) ** (252 / len(daily_returns_c)) - 1):+.2%}")
print(f"Sharpe: {(daily_returns_c.mean() / daily_returns_c.std() * np.sqrt(252) if daily_returns_c.std() > 0 else 0):.2f}")

daily_returns_c.to_csv('results/Strategy_C_Daily_Returns_2019_2026.csv', header=['Daily_Return'])
print("  [OK] Strategy_C_Daily_Returns_2019_2026.csv saved")

# ============================================================================
# STRATEGY D: Contrarian Bubble (2019-2026)
# ============================================================================
print("\n" + "="*120)
print("STRATEGY D: Contrarian Bubble (2019-2026)")
print("="*120)

daily_close_d = daily_close.loc['2019-01-01':]
daily_ret_d = daily_close_d.pct_change().fillna(0)

ma_window = 21  # Approximate 104h as 21 days
strategy_d_daily = []

for i in range(ma_window + 1, len(daily_close_d)):
    bubbles_dict = {}
    
    for col in daily_close_d.columns:
        try:
            recent = daily_close_d[col].iloc[max(0, i-ma_window):i+1]
            if len(recent) < 10:
                continue
            
            log_p = np.log(recent)
            fair = log_p.rolling(ma_window, min_periods=ma_window//2).mean()
            res = log_p - fair
            z_sc = (res - res.rolling(ma_window, min_periods=ma_window//2).mean()) / res.rolling(ma_window, min_periods=ma_window//2).std()
            bubble = np.tanh(z_sc.iloc[-1] / 2)
            
            if bubble < -0.8:
                bubbles_dict[col] = bubble
        except:
            pass
    
    if len(bubbles_dict) >= 10:
        top_20 = sorted(bubbles_dict.items(), key=lambda x: x[1])[:min(20, len(bubbles_dict))]
        top_cols = [col for col, _ in top_20]
        daily_return = daily_ret_d.iloc[i][top_cols].mean() - 0.001
    else:
        daily_return = 0
    
    strategy_d_daily.append(daily_return)

daily_returns_d = pd.Series(strategy_d_daily, index=daily_close_d.index[ma_window+1:ma_window+1+len(strategy_d_daily)])
wealth_d = (1 + daily_returns_d).cumprod()

print(f"Period: {daily_returns_d.index[0].date()} to {daily_returns_d.index[-1].date()}")
print(f"Annual Return: {((wealth_d.iloc[-1] / wealth_d.iloc[0]) ** (252 / len(daily_returns_d)) - 1):+.2%}")
print(f"Sharpe: {(daily_returns_d.mean() / daily_returns_d.std() * np.sqrt(252) if daily_returns_d.std() > 0 else 0):.2f}")

daily_returns_d.to_csv('results/Strategy_D_Daily_Returns_2019_2026.csv', header=['Daily_Return'])
print("  [OK] Strategy_D_Daily_Returns_2019_2026.csv saved")

# ============================================================================
# STRATEGY E: Reddit Sentiment (2024-2026 - data limited)
# ============================================================================
print("\n" + "="*120)
print("STRATEGY E: Reddit Sentiment (2024-2026) - Data Limited")
print("="*120)
print("NOTE: Requires sentiment API - cannot generate from price data alone")
print("SKIPPED")

print("\n" + "="*120)
print("ALL DAILY RETURNS GENERATED")
print("="*120)

