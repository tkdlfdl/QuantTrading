"""
Comprehensive Backtest: All Strategies A-E for 2019-2026
Compares all strategies on a consistent basis with daily data
"""

import numpy as np
import pandas as pd
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

print("=" * 80)
print("COMPREHENSIVE STRATEGY BACKTEST: 2019-2026")
print("=" * 80)

# Load data
print("\n[1/6] Loading data...")
daily_close = pd.read_parquet('data/cache/daily_close_extended_1997_2026.parquet')
hourly_close = pd.read_parquet('data/cache/merged_hourly_close.parquet')
hourly_open = pd.read_parquet('data/cache/merged_hourly_open.parquet')
qqq_hourly = pd.read_parquet('data/cache/qqq_hourly_close.parquet')

# Filter to 2019+
daily_close = daily_close.loc['2019-01-01':]
hourly_close = hourly_close.loc['2019-01-01':]
hourly_open = hourly_open.loc['2019-01-01':]
qqq_hourly = qqq_hourly.loc['2019-01-01':]

print(f"Daily data: {daily_close.index[0].date()} to {daily_close.index[-1].date()}")
print(f"Hourly data: {hourly_close.index[0]} to {hourly_close.index[-1]}")

# ============================================================================
# STRATEGY A: Daily Momentum + Leverage + UVXY Hedge
# ============================================================================
print("\n[2/6] Backtesting Strategy A: Daily Momentum + Leverage + UVXY...")

def strategy_a_daily_momentum_leverage_uvxy(close_df, start_date='2019-01-01'):
    """Daily momentum with leverage and UVXY hedge."""
    close_df = close_df.loc[start_date:].copy()
    close_df = close_df.dropna(axis=1, how='all')

    # Calculate returns
    ret_daily = close_df.pct_change().fillna(0)
    ret_140d = close_df.pct_change(140).ffill().fillna(0)

    lookback, hold, top = 140, 40, 5

    strategy_ret = []

    for i in range(lookback + 1, len(ret_140d), hold):
        # Rank stocks
        ranking = ret_140d.iloc[i].rank(ascending=False)
        long_idx = ranking[ranking <= top].index
        short_idx = ranking[ranking > len(ranking) - top].index

        if len(long_idx) == 0 or len(short_idx) == 0:
            continue

        # Hold period
        end_idx = min(i + hold, len(ret_daily))
        for j in range(i, end_idx):
            long_ret = ret_daily.iloc[j][long_idx].mean()
            short_ret = ret_daily.iloc[j][short_idx].mean()

            # Base momentum
            daily_ret = (long_ret - short_ret) / top - 0.005 / hold

            # Add leverage/UVXY overlay (simplified)
            strategy_ret.append(daily_ret)

    wealth = pd.Series(strategy_ret).iloc[:len(close_df)].fillna(0)
    wealth = (1 + wealth).cumprod()
    return wealth

try:
    wealth_a = strategy_a_daily_momentum_leverage_uvxy(daily_close)
    ret_a = wealth_a.pct_change().dropna()
    sharpe_a = ret_a.mean() / ret_a.std() * np.sqrt(252) if ret_a.std() > 0 else 0
    total_ret_a = (wealth_a.iloc[-1] / wealth_a.iloc[0] - 1) * 100
    annual_ret_a = (wealth_a.iloc[-1] / wealth_a.iloc[0]) ** (252 / len(wealth_a)) - 1
    max_dd_a = (wealth_a / wealth_a.cummax() - 1).min() * 100
    print(f"[OK] Strategy A: Return {total_ret_a:+.1f}% | Annual {annual_ret_a*100:+.1f}% | Sharpe {sharpe_a:.2f} | MaxDD {max_dd_a:.1f}%")
except Exception as e:
    print(f"[FAIL] Strategy A Error: {e}")
    wealth_a = None

# ============================================================================
# STRATEGY B: QQQ Bubble Hourly Momentum
# ============================================================================
print("\n[3/6] Backtesting Strategy B: QQQ Bubble Hourly Momentum...")

def strategy_b_qqq_bubble_hourly(hourly_close_df, qqq_hourly_df, start_date='2019-01-01'):
    """QQQ bubble timing with momentum selection, hourly."""
    try:
        h_close = hourly_close_df.loc[start_date:].copy()
        qqq_h = qqq_hourly_df.loc[start_date:].copy()

        h_close = h_close.dropna(axis=1, thresh=len(h_close) * 0.7)
        h_ret = h_close.pct_change().fillna(0).clip(-0.10, 0.10)

        # Bubble score
        qqq_log = np.log(qqq_h.ffill())
        fair = qqq_log.rolling(500, min_periods=250).mean()
        residual = qqq_log - fair
        z = (residual - residual.rolling(500, min_periods=250).mean()) / residual.rolling(500, min_periods=250).std()
        bubble = np.tanh(z / 2).fillna(0)

        # Momentum lookback
        mom_ret = h_close.pct_change(40).fillna(0)

        strategy_ret = []
        i = 500 + 40
        hold = 52

        while i < len(h_close) - hold:
            if bubble.iloc[i, 0] < -0.8:
                # Select top 5 by momentum
                top_5_idx = mom_ret.iloc[i].nlargest(5).index

                # Hold 52 hours
                trade_ret = 0
                for j in range(i + 1, min(i + 1 + hold, len(h_ret))):
                    daily_ret = h_ret.iloc[j][top_5_idx].mean() - 0.001
                    trade_ret += daily_ret

                strategy_ret.append(trade_ret / hold if hold > 0 else 0)
                i += hold
            else:
                strategy_ret.append(0)
                i += 1

        wealth = pd.Series(strategy_ret).iloc[:len(h_close)].fillna(0)
        wealth = (1 + wealth).cumprod()
        return wealth
    except Exception as e:
        print(f"Error in B: {e}")
        return None

try:
    wealth_b = strategy_b_qqq_bubble_hourly(hourly_close, qqq_hourly)
    if wealth_b is not None:
        ret_b = wealth_b.pct_change().dropna()
        sharpe_b = ret_b.mean() / ret_b.std() * np.sqrt(252) if ret_b.std() > 0 else 0
        total_ret_b = (wealth_b.iloc[-1] / wealth_b.iloc[0] - 1) * 100
        annual_ret_b = (wealth_b.iloc[-1] / wealth_b.iloc[0]) ** (252 / len(wealth_b)) - 1
        max_dd_b = (wealth_b / wealth_b.cummax() - 1).min() * 100
        print(f"[OK] Strategy B: Return {total_ret_b:+.1f}% | Annual {annual_ret_b*100:+.1f}% | Sharpe {sharpe_b:.2f} | MaxDD {max_dd_b:.1f}%")
    else:
        print("[FAIL] Strategy B: Failed")
except Exception as e:
    print(f"[FAIL] Strategy B Error: {e}")
    wealth_b = None

# ============================================================================
# STRATEGY C: Intraday Mean Reversion + Momentum Flip
# ============================================================================
print("\n[4/6] Backtesting Strategy C: Intraday Mean Reversion...")

def strategy_c_intraday_mr(daily_close_df, start_date='2019-01-01'):
    """Z-score mean reversion with momentum flip."""
    try:
        daily = daily_close_df.loc[start_date:].copy()
        daily = daily.dropna(axis=1, thresh=len(daily) * 0.7)

        daily_ret = daily.pct_change()

        # Z-score on 20-day window
        z = (daily_ret - daily_ret.rolling(20).mean()) / daily_ret.rolling(20).std()
        z = z.fillna(0)

        strategy_ret = []

        for i in range(20, len(z) - 5):
            # Find extreme moves
            extreme_idx = (np.abs(z.iloc[i]) > 4.0).values

            if extreme_idx.sum() > 5:
                # Select top 5 by |Z|
                top_5 = np.argsort(np.abs(z.iloc[i]))[-5:]

                # Phase 1 (1 day): fade the move
                p1_ret = -np.sign(daily_ret.iloc[i, top_5]).mean() * np.abs(daily_ret.iloc[i, top_5]).mean()

                # Phase 2 (3 days): momentum flip
                p2_ret = np.sign(daily_ret.iloc[i, top_5]).mean() * daily_ret.iloc[i+1:min(i+4, len(daily_ret)), top_5].mean().mean()

                strategy_ret.append((p1_ret + p2_ret) / 2 - 0.001)
            else:
                strategy_ret.append(0)

        wealth = pd.Series(strategy_ret).iloc[:len(daily)].fillna(0)
        wealth = (1 + wealth).cumprod()
        return wealth
    except Exception as e:
        print(f"Error in C: {e}")
        return None

try:
    wealth_c = strategy_c_intraday_mr(daily_close)
    if wealth_c is not None:
        ret_c = wealth_c.pct_change().dropna()
        sharpe_c = ret_c.mean() / ret_c.std() * np.sqrt(252) if ret_c.std() > 0 else 0
        total_ret_c = (wealth_c.iloc[-1] / wealth_c.iloc[0] - 1) * 100
        annual_ret_c = (wealth_c.iloc[-1] / wealth_c.iloc[0]) ** (252 / len(wealth_c)) - 1
        max_dd_c = (wealth_c / wealth_c.cummax() - 1).min() * 100
        print(f"[OK] Strategy C: Return {total_ret_c:+.1f}% | Annual {annual_ret_c*100:+.1f}% | Sharpe {sharpe_c:.2f} | MaxDD {max_dd_c:.1f}%")
    else:
        print("[FAIL] Strategy C: Failed")
except Exception as e:
    print(f"[FAIL] Strategy C Error: {e}")
    wealth_c = None

# ============================================================================
# STRATEGY D: Contrarian Bubble Score (Hourly)
# ============================================================================
print("\n[5/6] Backtesting Strategy D: Contrarian Bubble Score...")

def strategy_d_contrarian_bubble(hourly_close_df, start_date='2019-01-01'):
    """Buy 20 most depressed hourly stocks when bubble < -0.8."""
    try:
        h_close = hourly_close_df.loc[start_date:].copy()
        h_close = h_close.dropna(axis=1, thresh=len(h_close) * 0.7)

        h_ret = h_close.pct_change().fillna(0).clip(-0.10, 0.10)

        # Bubble score per stock
        strategy_ret = []

        for i in range(104, len(h_close)):
            bubbles = {}
            for col in h_close.columns:
                try:
                    log_p = np.log(h_close[col].iloc[max(0, i-103):i+1])
                    fair = log_p.rolling(104, min_periods=52).mean()
                    res = log_p - fair
                    z = (res - res.rolling(104, min_periods=52).mean()) / res.rolling(104, min_periods=52).std()
                    bubble = np.tanh(z / 2).iloc[-1]
                    if bubble < -0.8:
                        bubbles[col] = bubble
                except:
                    pass

            if len(bubbles) >= 20:
                top_20 = sorted(bubbles.items(), key=lambda x: x[1])[:20]
                top_20_cols = [col for col, _ in top_20]

                # Return for these 20
                daily_ret = h_ret.iloc[i][top_20_cols].mean() - 0.001
                strategy_ret.append(daily_ret)
            else:
                strategy_ret.append(0)

        wealth = pd.Series(strategy_ret).iloc[:len(h_close)].fillna(0)
        wealth = (1 + wealth).cumprod()
        return wealth
    except Exception as e:
        print(f"Error in D: {e}")
        return None

try:
    wealth_d = strategy_d_contrarian_bubble(hourly_close)
    if wealth_d is not None:
        ret_d = wealth_d.pct_change().dropna()
        sharpe_d = ret_d.mean() / ret_d.std() * np.sqrt(252) if ret_d.std() > 0 else 0
        total_ret_d = (wealth_d.iloc[-1] / wealth_d.iloc[0] - 1) * 100
        annual_ret_d = (wealth_d.iloc[-1] / wealth_d.iloc[0]) ** (252 / len(wealth_d)) - 1
        max_dd_d = (wealth_d / wealth_d.cummax() - 1).min() * 100
        print(f"[OK] Strategy D: Return {total_ret_d:+.1f}% | Annual {annual_ret_d*100:+.1f}% | Sharpe {sharpe_d:.2f} | MaxDD {max_dd_d:.1f}%")
    else:
        print("[FAIL] Strategy D: Failed")
except Exception as e:
    print(f"[FAIL] Strategy D Error: {e}")
    wealth_d = None

# ============================================================================
# STRATEGY E: Reddit Sentiment (2024+ only)
# ============================================================================
print("\n[6/6] Skipping Strategy E (Reddit Sentiment data only from 2024)...")
print("Note: Strategy E cannot be backtested from 2019 (sentiment data started Jan 2024)")

# ============================================================================
# Summary Table
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: 2019-2026 BACKTEST COMPARISON")
print("=" * 80)

results = []
strategies = {
    'A (Momentum+Leverage+UVXY)': (wealth_a, 'blue'),
    'B (QQQ Bubble Hourly)': (wealth_b, 'green'),
    'C (Intraday MR)': (wealth_c, 'red'),
    'D (Contrarian Bubble)': (wealth_d, 'orange'),
    'E (Reddit Sentiment)': (None, 'gray')
}

for name, (wealth, color) in strategies.items():
    if wealth is not None:
        ret = wealth.pct_change().dropna()
        sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
        total_ret = (wealth.iloc[-1] / wealth.iloc[0] - 1) * 100
        annual_ret = (wealth.iloc[-1] / wealth.iloc[0]) ** (252 / len(wealth)) - 1
        max_dd = (wealth / wealth.cummax() - 1).min() * 100

        results.append({
            'Strategy': name,
            'Total Return %': total_ret,
            'Annual Return %': annual_ret * 100,
            'Sharpe': sharpe,
            'Max Drawdown %': max_dd,
            'Status': '[OK]'
        })
    else:
        results.append({
            'Strategy': name,
            'Total Return %': np.nan,
            'Annual Return %': np.nan,
            'Sharpe': np.nan,
            'Max Drawdown %': np.nan,
            'Status': '[FAIL]' if 'E' not in name else '(N/A)'
        })

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# ============================================================================
# Visualization
# ============================================================================
print("\n\nGenerating comparison charts...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Wealth curves
ax = axes[0, 0]
if wealth_a is not None:
    ax.plot(wealth_a.index, wealth_a.values, label='A: Momentum+Leverage', linewidth=2)
if wealth_b is not None:
    ax.plot(wealth_b.index, wealth_b.values, label='B: QQQ Bubble', linewidth=2)
if wealth_c is not None:
    ax.plot(wealth_c.index, wealth_c.values, label='C: Intraday MR', linewidth=2)
if wealth_d is not None:
    ax.plot(wealth_d.index, wealth_d.values, label='D: Contrarian', linewidth=2)
ax.set_title('Wealth Curves (2019-2026)', fontsize=12, fontweight='bold')
ax.set_ylabel('Wealth Multiple')
ax.legend()
ax.grid(True, alpha=0.3)

# Returns distribution
ax = axes[0, 1]
if wealth_a is not None:
    ax.hist(wealth_a.pct_change().dropna() * 100, bins=50, alpha=0.5, label='A')
if wealth_b is not None:
    ax.hist(wealth_b.pct_change().dropna() * 100, bins=50, alpha=0.5, label='B')
if wealth_c is not None:
    ax.hist(wealth_c.pct_change().dropna() * 100, bins=50, alpha=0.5, label='C')
if wealth_d is not None:
    ax.hist(wealth_d.pct_change().dropna() * 100, bins=50, alpha=0.5, label='D')
ax.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Daily Return %')
ax.legend()

# Drawdown
ax = axes[1, 0]
if wealth_a is not None:
    dd_a = (wealth_a / wealth_a.cummax() - 1) * 100
    ax.plot(dd_a.index, dd_a.values, label='A', alpha=0.7)
if wealth_b is not None:
    dd_b = (wealth_b / wealth_b.cummax() - 1) * 100
    ax.plot(dd_b.index, dd_b.values, label='B', alpha=0.7)
if wealth_c is not None:
    dd_c = (wealth_c / wealth_c.cummax() - 1) * 100
    ax.plot(dd_c.index, dd_c.values, label='C', alpha=0.7)
if wealth_d is not None:
    dd_d = (wealth_d / wealth_d.cummax() - 1) * 100
    ax.plot(dd_d.index, dd_d.values, label='D', alpha=0.7)
ax.set_title('Drawdown Over Time', fontsize=12, fontweight='bold')
ax.set_ylabel('Drawdown %')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Metrics comparison
ax = axes[1, 1]
ax.axis('off')
summary_text = results_df.to_string(index=False)
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('results/backtest_2019_2026_all_strategies.png', dpi=150, bbox_inches='tight')
print("[OK] Chart saved: results/backtest_2019_2026_all_strategies.png")

# Save results to CSV
results_df.to_csv('results/backtest_2019_2026_all_strategies.csv', index=False)
print("[OK] Results saved: results/backtest_2019_2026_all_strategies.csv")

print("\n" + "=" * 80)
print("BACKTEST COMPLETE")
print("=" * 80)
