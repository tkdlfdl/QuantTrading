"""
Backtest All Strategies (A-E) Based on strat.md Parameters
Generates daily returns and verifies against documented results
"""

import numpy as np
import pandas as pd
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

print("=" * 120)
print("STRATEGY BACKTEST VERIFICATION: 2019-2026")
print("Based on strat.md Parameters")
print("=" * 120)

# Load data
print("\n[1/6] Loading data...")
daily_close = pd.read_parquet('data/cache/daily_close_extended_1997_2026.parquet')
daily_close = daily_close.loc['2019-01-01':'2026-06-12'].copy()
daily_close = daily_close.dropna(axis=1, thresh=len(daily_close) * 0.85)

print(f"Period: {daily_close.index[0].date()} to {daily_close.index[-1].date()}")
print(f"Stocks: {len(daily_close.columns)}")

daily_ret = daily_close.pct_change().fillna(0)

# Store all results
all_daily_returns = {}
all_yearly_metrics = {}

# =============================================================================
# STRATEGY A: Daily Momentum + Leverage + UVXY (Parameters from strat.md)
# =============================================================================
print("\n[2/6] Strategy A: Daily Momentum + Leverage + UVXY")
print("  Parameters: Lookback=140d, Hold=40d, Long=5, Short=5, Leverage=1.25x")

ret_140d = daily_close.pct_change(140).ffill().fillna(0)

strategy_a_daily = []
lookback, hold_days, top_n = 140, 40, 5

for i in range(lookback, len(daily_close)):
    # Get momentum ranking
    if i < len(ret_140d):
        ranking = ret_140d.iloc[i].rank(ascending=False)
        long_idx = ranking[ranking <= top_n].index.tolist()
        short_idx = ranking[ranking > len(ranking) - top_n].index.tolist()

        if len(long_idx) >= top_n and len(short_idx) >= top_n:
            long_ret = daily_ret.iloc[i][long_idx].mean()
            short_ret = daily_ret.iloc[i][short_idx].mean()

            # Base momentum (0.5% cost per 40-day cycle = 0.0125% per day)
            daily_return = (long_ret - short_ret) / top_n - 0.005 / hold_days

            # Simplified: Add leverage/UVXY logic (would need bubble score)
            strategy_a_daily.append(daily_return)
        else:
            strategy_a_daily.append(0)
    else:
        strategy_a_daily.append(0)

all_daily_returns['A_Momentum_Leverage_UVXY'] = pd.Series(
    strategy_a_daily,
    index=daily_close.index[lookback:]
)

# =============================================================================
# STRATEGY B: QQQ Bubble Hourly (Can't implement hourly from daily data)
# =============================================================================
print("\n[3/6] Strategy B: QQQ Bubble Hourly Momentum")
print("  NOTE: Requires hourly data (not available for full 2019-2026)")
print("  Using daily approximation with bubble timing on momentum median")

# Simple proxy: use momentum-based bubble
bubble_proxy = ret_140d.median(axis=1)
bubble_ma = bubble_proxy.rolling(120).mean()  # Approximate 500h as 120d
bubble_std = bubble_proxy.rolling(120).std()
bubble_score = np.tanh((bubble_proxy - bubble_ma) / bubble_std.clip(lower=0.01) / 2)

strategy_b_daily = []
for i in range(120, len(daily_close)):
    # Buy when bubble < -0.8
    if bubble_score.iloc[i] < -0.8:
        # Select top 5 momentum
        momentum_20h_approx = daily_ret.iloc[i:i+1].mean(axis=0)
        if len(momentum_20h_approx) >= 5:
            top_5_idx = momentum_20h_approx.nlargest(5).index.tolist()
            daily_return = daily_ret.iloc[i][top_5_idx].mean() - 0.001
        else:
            daily_return = 0
    else:
        daily_return = 0

    strategy_b_daily.append(daily_return)

all_daily_returns['B_QQQ_Bubble_Hourly'] = pd.Series(
    strategy_b_daily,
    index=daily_close.index[120:]
)

# =============================================================================
# STRATEGY C: Intraday Mean Reversion + Momentum Flip
# =============================================================================
print("\n[4/6] Strategy C: Intraday Mean Reversion + Momentum Flip")
print("  Parameters: Lookback=20d, Sigma=4.0, Hold=3d, Top-N=5")

z_window = 20
strategy_c_daily = []

for i in range(z_window + 5, len(daily_ret)):
    # Calculate z-scores
    z_scores = (daily_ret.iloc[i] - daily_ret.iloc[i-z_window:i].mean()) / \
               daily_ret.iloc[i-z_window:i].std()

    # Find extreme moves (Z > 4.0)
    extreme_mask = np.abs(z_scores) > 4.0
    if extreme_mask.sum() < 5:
        strategy_c_daily.append(0)
        continue

    # Select top 5 by |Z|
    top_5_idx = np.abs(z_scores[extreme_mask]).nlargest(5).index.tolist()

    # Phase 1 (1 day): fade the move (mean reversion)
    fade_ret = -np.sign(daily_ret.iloc[i][top_5_idx]).mean() * \
              np.abs(daily_ret.iloc[i][top_5_idx]).mean()

    # Phase 2 (3 days): momentum flip
    if i < len(daily_ret) - 3:
        cont_ret = np.sign(daily_ret.iloc[i][top_5_idx]).mean() * \
                  daily_ret.iloc[i+1:i+4][top_5_idx].mean().mean()
    else:
        cont_ret = 0

    # Combine phases, subtract costs
    daily_return = (fade_ret + cont_ret) / 2 - 0.001
    strategy_c_daily.append(daily_return)

all_daily_returns['C_Intraday_MR'] = pd.Series(
    strategy_c_daily,
    index=daily_close.index[z_window+5:z_window+5+len(strategy_c_daily)]
)

# =============================================================================
# STRATEGY D: Contrarian Bubble Score
# =============================================================================
print("\n[5/6] Strategy D: Contrarian Bubble Score")
print("  Parameters: MA=104h (approx daily), Threshold=-0.8, Hold=13h (approx daily), Top-N=20")

strategy_d_daily = []
ma_window = 104  # Approximate 104h as ~21 days

for i in range(ma_window + 1, len(daily_close)):
    bubbles_dict = {}

    for col in daily_close.columns:
        try:
            recent = daily_close[col].iloc[max(0, i-ma_window):i+1]
            if len(recent) < 20:
                continue

            log_p = np.log(recent)
            fair = log_p.rolling(ma_window, min_periods=ma_window//2).mean()
            res = log_p - fair
            z_sc = (res - res.rolling(ma_window, min_periods=ma_window//2).mean()) / \
                   res.rolling(ma_window, min_periods=ma_window//2).std()
            bubble = np.tanh(z_sc.iloc[-1] / 2)

            if bubble < -0.8:
                bubbles_dict[col] = bubble
        except:
            pass

    if len(bubbles_dict) >= 10:
        top_20 = sorted(bubbles_dict.items(), key=lambda x: x[1])[:min(20, len(bubbles_dict))]
        top_cols = [col for col, _ in top_20]
        daily_return = daily_ret.iloc[i][top_cols].mean() - 0.001
    else:
        daily_return = 0

    strategy_d_daily.append(daily_return)

all_daily_returns['D_Contrarian_Bubble'] = pd.Series(
    strategy_d_daily,
    index=daily_close.index[ma_window+1:ma_window+1+len(strategy_d_daily)]
)

# =============================================================================
# STRATEGY E: Reddit Sentiment (Can't implement - need sentiment data)
# =============================================================================
print("\n[6/6] Strategy E: Reddit Sentiment Long")
print("  NOTE: No sentiment data available for 2019-2026")
print("  Skipping E for this backtest")

# =============================================================================
# Calculate Metrics
# =============================================================================

print("\n" + "=" * 120)
print("BACKTEST RESULTS VERIFICATION")
print("=" * 120)

results_comparison = []

for strategy_name, daily_series in all_daily_returns.items():
    ret = daily_series.dropna()

    if len(ret) == 0:
        continue

    # Overall metrics
    wealth = (1 + ret).cumprod()
    total_ret = wealth.iloc[-1] / wealth.iloc[0] - 1
    years = len(ret) / 252
    annual_ret = (wealth.iloc[-1] / wealth.iloc[0]) ** (1 / years) - 1
    sharpe = (ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
    sortino = (ret.mean() / ret[ret < 0].std() * np.sqrt(252)) if ret[ret < 0].std() > 0 else 0
    max_dd = ((wealth / wealth.cummax()) - 1).min()
    win_rate = (ret > 0).sum() / len(ret)

    strategy_id = strategy_name.split('_')[0]

    # Get documented metrics from strat.md
    doc_metrics = {
        'A': {'sharpe': 1.41, 'annual': 85, 'maxdd': -65, 'period': '30yr'},
        'B': {'sharpe': 1.66, 'annual': 17.19, 'maxdd': -16.25, 'period': '5.85yr'},
        'C': {'sharpe': 0.98, 'annual': 26.8, 'maxdd': -20.8, 'period': '7.4yr'},
        'D': {'sharpe': 2.65, 'annual': 38.08, 'maxdd': -10.09, 'period': '7.4yr'},
    }

    if strategy_id in doc_metrics:
        doc = doc_metrics[strategy_id]
        sharpe_diff = sharpe - doc['sharpe']
        annual_diff = (annual_ret * 100) - doc['annual']
        maxdd_diff = (max_dd * 100) - doc['maxdd']

        results_comparison.append({
            'Strategy': strategy_name,
            'Actual_Annual_%': annual_ret * 100,
            'Doc_Annual_%': doc['annual'],
            'Annual_Diff_%': annual_diff,
            'Actual_Sharpe': sharpe,
            'Doc_Sharpe': doc['sharpe'],
            'Sharpe_Diff': sharpe_diff,
            'Actual_MaxDD_%': max_dd * 100,
            'Doc_MaxDD_%': doc['maxdd'],
            'MaxDD_Diff_%': maxdd_diff,
            'Days': len(ret),
            'Total_Return_%': total_ret * 100,
        })

    print(f"\n{strategy_name}")
    print(f"  Days: {len(ret):,}")
    print(f"  Total Return: {total_ret:+.2%}")
    print(f"  Annual Return: {annual_ret:+.2%}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_dd:.2%}")
    print(f"  Win Rate: {win_rate:.1%}")

# =============================================================================
# Yearly Breakdown
# =============================================================================

print("\n" + "=" * 120)
print("YEARLY BREAKDOWN")
print("=" * 120)

yearly_results = []

for strategy_name, daily_series in all_daily_returns.items():
    ret = daily_series.dropna()

    for year in range(2019, 2027):
        year_mask = ret.index.year == year
        year_ret = ret[year_mask]

        if len(year_ret) == 0:
            continue

        year_wealth = (1 + year_ret).cumprod()
        year_total = year_wealth.iloc[-1] / year_wealth.iloc[0] - 1
        year_sharpe = (year_ret.mean() / year_ret.std() * np.sqrt(252)) if year_ret.std() > 0 else 0
        year_maxdd = ((year_wealth / year_wealth.cummax()) - 1).min()

        yearly_results.append({
            'Year': year,
            'Strategy': strategy_name.split('_')[0],
            'Return_%': year_total * 100,
            'Sharpe': year_sharpe,
            'MaxDD_%': year_maxdd * 100,
            'Days': len(year_ret),
        })

yearly_df = pd.DataFrame(yearly_results)

# Pivot for easier viewing
for strategy in ['A', 'B', 'C', 'D']:
    print(f"\n{strategy}:")
    strategy_yearly = yearly_df[yearly_df['Strategy'] == strategy]
    print(strategy_yearly[['Year', 'Return_%', 'Sharpe', 'MaxDD_%']].to_string(index=False))

# =============================================================================
# Save to Excel
# =============================================================================

print("\n" + "=" * 120)
print("SAVING RESULTS TO EXCEL")
print("=" * 120)

# Create Excel writer
with pd.ExcelWriter('results/Backtest_Verification_2019_2026.xlsx', engine='openpyxl') as writer:
    # Sheet 1: Daily Returns
    print("\n[1] Creating Daily Returns sheet...")
    daily_df = pd.DataFrame(all_daily_returns)
    daily_df.to_excel(writer, sheet_name='Daily Returns')
    print(f"  Daily Returns: {daily_df.shape[0]} days x {daily_df.shape[1]} strategies")

    # Sheet 2: Comparison with Documented
    print("[2] Creating Results Comparison sheet...")
    comp_df = pd.DataFrame(results_comparison)
    comp_df.to_excel(writer, sheet_name='Results Comparison', index=False)

    # Sheet 3: Yearly Metrics
    print("[3] Creating Yearly Metrics sheet...")
    yearly_df.to_excel(writer, sheet_name='Yearly Metrics', index=False)

    # Sheet 4: Summary Statistics
    print("[4] Creating Summary Statistics sheet...")
    summary_stats = []
    for strategy_name, daily_series in all_daily_returns.items():
        ret = daily_series.dropna()
        wealth = (1 + ret).cumprod()
        total_ret = wealth.iloc[-1] / wealth.iloc[0] - 1
        years = len(ret) / 252
        annual_ret = (wealth.iloc[-1] / wealth.iloc[0]) ** (1 / years) - 1
        sharpe = (ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0
        sortino = (ret.mean() / ret[ret < 0].std() * np.sqrt(252)) if ret[ret < 0].std() > 0 else 0
        max_dd = ((wealth / wealth.cummax()) - 1).min()

        summary_stats.append({
            'Strategy': strategy_name,
            'Total_Return_%': total_ret * 100,
            'Annual_Return_%': annual_ret * 100,
            'Sharpe': sharpe,
            'Sortino': sortino,
            'Max_DD_%': max_dd * 100,
            'Trading_Days': len(ret),
        })

    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)

    # Sheet 5: Wealth Curves
    print("[5] Creating Wealth Curves sheet...")
    wealth_curves = {}
    for strategy_name, daily_series in all_daily_returns.items():
        ret = daily_series.dropna()
        wealth = (1 + ret).cumprod()
        wealth_curves[f'{strategy_name}_Wealth'] = wealth

    wealth_df = pd.DataFrame(wealth_curves)
    wealth_df.to_excel(writer, sheet_name='Wealth Curves')

print("\n✓ Excel file saved: results/Backtest_Verification_2019_2026.xlsx")

print("\n" + "=" * 120)
print("VERIFICATION SUMMARY")
print("=" * 120)

print("\nCompare 'Actual' vs 'Doc' columns in 'Results Comparison' sheet:")
print("- If differences are small (< 5%), backtest parameters are correct")
print("- If differences are large (> 10%), there may be implementation issues")
print("\nNote:")
print("- Strategy B & E: Limited backtest scope (hourly data/sentiment data not available)")
print("- This is a daily approximation using available daily data")

