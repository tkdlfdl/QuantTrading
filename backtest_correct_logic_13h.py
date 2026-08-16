"""
CORRECT IMPLEMENTATION: Check EVERY hour for signals, hold 13 hours
===================================================================

Key fix: Loop through EVERY bar, not just every 13-hour interval.

Only trade when bubble < -0.8, then hold for exactly 13 hours.
No overlapping trades - when one exits, the next can enter.
"""

import numpy as np
import pandas as pd
from data.universe import get_universe
from data.intraday_loader import load_hourly_bars
from strategies.universe_bubble_hourly import _bubble_scores_matrix

TRADING_DAYS = 252

print("=" * 100)
print("CORRECT IMPLEMENTATION: Check EVERY hour for signals, hold 13 hours")
print("=" * 100)

# Load data
universe = get_universe()
ho, hc = load_hourly_bars(universe, use_cache=True)

print(f"\nDATA VALIDATION:")
print(f"  Tickers: {hc.shape[1]}")
print(f"  Total bars: {hc.shape[0]}")
print(f"  Period: {hc.index[0]} to {hc.index[-1]}")
print(f"  Bar frequency: {(hc.index[1] - hc.index[0]).total_seconds() / 3600:.1f} hours")
print(f"  Index type: {type(hc.index[0])}")

# Verify hourly data
time_diffs = hc.index.to_series().diff()
typical_diff = time_diffs[time_diffs > pd.Timedelta(0)].median()
print(f"  Typical time between bars: {typical_diff}")

if typical_diff == pd.Timedelta(hours=1):
    print(f"  [OK] Data is hourly")
else:
    print(f"  [WARNING] Data may not be hourly! Typical interval: {typical_diff}")

# Parameters (CORRECT: 13h hold, not 104h)
ma = 104
z = 104
buy_thresh = 0.8
hold = 13  # CRITICAL: 13 hours, not 104
top_n = 20
transaction_cost = 0.001
short_borrow_rate = 0.08

print(f"\nPARAMETERS:")
print(f"  MA Window: {ma}h")
print(f"  Z Window: {z}h")
print(f"  Buy Threshold: -{buy_thresh}")
print(f"  Hold Period: {hold}h")
print(f"  Top N: {top_n}")

# Compute bubble scores ONCE
print(f"\nComputing bubble scores for all bars...")
raw = _bubble_scores_matrix(hc, ma, z)
scores = raw.shift(1)  # No lookahead

n = len(hc)
print(f"  Bubble scores computed: {n} bars")

# Track trades
trades = []
in_trade_until = -1  # Bar index where current trade(s) exit

print(f"\nRunning backtest (checking EVERY hour for signals)...")

# KEY FIX: Loop through EVERY bar, check for entry signals
for i in range(max(ma, z) + 1, n - hold):
    # Skip if we're still in an active trade
    if i <= in_trade_until:
        continue

    sig = scores.iloc[i].dropna()
    if sig.empty:
        continue

    # Find top-N oversold stocks
    long_cands = sig[sig < -buy_thresh].nsmallest(top_n)
    if len(long_cands) == 0:
        continue

    # Entry: open of bar i
    # Exit: close of bar i+hold-1
    exit_i = min(i + hold - 1, n - 1)

    # Record all trades from this rebalance
    entry_dt = hc.index[i]
    exit_dt = hc.index[exit_i]
    entry_date = entry_dt.normalize()

    for tkr in long_cands.index:
        ep = ho.iat[i, ho.columns.get_loc(tkr)]
        xp = hc.iat[exit_i, hc.columns.get_loc(tkr)]

        if pd.isna(ep) or pd.isna(xp) or ep <= 0:
            continue

        price_return = (xp / ep - 1)
        net_ret = price_return - transaction_cost

        trades.append({
            "entry_bar": i,
            "exit_bar": exit_i,
            "entry_dt": entry_dt,
            "exit_dt": exit_dt,
            "entry_date": entry_date,
            "ticker": tkr,
            "entry_price": ep,
            "exit_price": xp,
            "price_return": price_return,
            "cost": transaction_cost,
            "net_ret": net_ret,
        })

    # Update: no new trades until this batch exits
    in_trade_until = exit_i

print(f"  Total trades: {len(trades)}")

if len(trades) < 5:
    print(f"\n[ERROR] Too few trades ({len(trades)}). Cannot calculate metrics.")
else:
    df = pd.DataFrame(trades)

    # Aggregate returns by entry date
    daily = df.groupby("entry_date")["net_ret"].sum()
    data_end = hc.index[-1].normalize()
    all_dates = pd.date_range(daily.index.min(), data_end, freq="B")
    daily_full = daily.reindex(all_dates, fill_value=0.0)

    # Calculate metrics
    wealth = (1 + daily_full).cumprod()
    wealth = wealth / wealth.iloc[0]

    mean_ret = daily_full.mean()
    std_ret = daily_full.std()
    sharpe = (mean_ret * TRADING_DAYS) / (std_ret * np.sqrt(TRADING_DAYS)) if std_ret > 0 else np.nan

    ds = daily_full[daily_full < 0].std()
    sortino = (mean_ret * TRADING_DAYS) / (ds * np.sqrt(TRADING_DAYS)) if ds > 0 else np.nan

    mdd = (wealth / wealth.cummax() - 1).min()
    total_ret = wealth.iloc[-1] - 1
    wr = (df["net_ret"] > 0).sum() / len(df) if len(df) > 0 else 0

    print("\n" + "=" * 100)
    print("BACKTEST RESULTS (Correct Logic: Check Every Hour, Hold 13h)")
    print("=" * 100)

    print(f"\nData Period: {hc.index[0].date()} to {hc.index[-1].date()}")
    print(f"  Sharpe Ratio:     {sharpe:>8.3f}")
    print(f"  Sortino Ratio:    {sortino:>8.3f}")
    print(f"  Total Return:     {total_ret:>8.1%}")
    print(f"  Max Drawdown:     {mdd:>8.1%}")
    print(f"  Win Rate:         {wr:>8.1%}")
    print(f"  Total Trades:     {len(df):>8}")
    print(f"  Avg Trade Return: {df['net_ret'].mean():>8.3%}")
    print(f"  Std Dev:          {df['net_ret'].std():>8.3%}")

    print(f"\n" + "=" * 100)
    print("COMPARISON TO DOCUMENTED (2019-2026, 8-year period):")
    print("=" * 100)

    doc_sharpe = 2.648
    doc_return = 0.381  # Annual
    doc_mdd = -0.101
    doc_wr = 0.566

    print(f"\n{'Metric':<25} {'Documented':<20} {'Our Result':<20} {'Ratio':<20}")
    print("-" * 85)
    print(f"{'Sharpe Ratio':<25} {doc_sharpe:<20.3f} {sharpe:<20.3f} {sharpe/doc_sharpe:<20.1%}")
    print(f"{'Total Return':<25} {doc_return:<20.1%} {total_ret:<20.1%} N/A")
    print(f"{'Max Drawdown':<25} {doc_mdd:<20.1%} {mdd:<20.1%} N/A")
    print(f"{'Win Rate':<25} {doc_wr:<20.1%} {wr:<20.1%} {wr/doc_wr:<20.1%}")

    print(f"\n" + "=" * 100)
    print("ANALYSIS:")
    print("=" * 100)

    if sharpe > 0 and total_ret > -0.5:
        print(f"\n[GOOD] Strategy shows positive Sharpe and reasonable returns")
        print(f"  2024-2026 is a tougher market than 2019-2026 average")
        print(f"  Documented period included 2020-2021 COVID boom and 2022 bear market")
    else:
        print(f"\n[CONCERN] Results still poor despite correct implementation")
        print(f"  This suggests:")
        print(f"    1. 2024-2026 market regime is very unfavorable for this strategy")
        print(f"    2. Strategy may be curve-fit to 2019-2026 period")
        print(f"    3. Market conditions have changed (less mean reversion, more trending)")

    print(f"\n" + "=" * 100)
    print("TRADE STATISTICS:")
    print("=" * 100)

    # Yearly breakdown
    df_copy = df.copy()
    df_copy["year"] = pd.to_datetime(df_copy["entry_date"]).dt.year

    print(f"\nYearly Performance:")
    for year in sorted(df_copy["year"].unique()):
        year_trades = df_copy[df_copy["year"] == year]
        year_daily = year_trades.groupby("entry_date")["net_ret"].sum()
        year_wealth = (1 + year_daily).cumprod() / (1 + year_daily).cumprod().iloc[0]

        year_ret = year_wealth.iloc[-1] - 1
        year_sharpe = (year_daily.mean() * TRADING_DAYS) / (year_daily.std() * np.sqrt(TRADING_DAYS)) if year_daily.std() > 0 else 0
        year_mdd = (year_wealth / year_wealth.cummax() - 1).min()
        year_wr = (year_trades["net_ret"] > 0).sum() / len(year_trades)

        print(f"  {year}: Ret={year_ret:+7.1%}  Sharpe={year_sharpe:6.3f}  MDD={year_mdd:7.1%}  "
              f"WR={year_wr:6.1%}  Trades={len(year_trades):4}")

    # Top and bottom trades
    print(f"\nTop 10 Trades:")
    top_trades = df.nlargest(10, "net_ret")[["entry_date", "ticker", "entry_price", "exit_price", "net_ret"]]
    for idx, row in top_trades.iterrows():
        print(f"  {row['entry_date'].date()} {row['ticker']:6} ${row['entry_price']:7.2f} -> ${row['exit_price']:7.2f} "
              f"Return={row['net_ret']:+7.3%}")

    print(f"\nBottom 10 Trades:")
    bot_trades = df.nsmallest(10, "net_ret")[["entry_date", "ticker", "entry_price", "exit_price", "net_ret"]]
    for idx, row in bot_trades.iterrows():
        print(f"  {row['entry_date'].date()} {row['ticker']:6} ${row['entry_price']:7.2f} -> ${row['exit_price']:7.2f} "
              f"Return={row['net_ret']:+7.3%}")

print(f"\n" + "=" * 100)
