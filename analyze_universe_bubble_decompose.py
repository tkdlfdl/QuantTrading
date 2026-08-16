import numpy as np
import pandas as pd
from data.universe import get_universe
from data.intraday_loader import load_hourly_bars
from strategies.universe_bubble_hourly import run_universe_bubble_hourly

TRADING_DAYS = 252

print("Loading universe hourly bars...")
universe = get_universe()
ho, hc = load_hourly_bars(universe, use_cache=True)
print(f"Data: {hc.shape[1]} tickers × {hc.shape[0]} bars  ({hc.index[0].date()} → {hc.index[-1].date()})")

# Run with smaller grid for speed
print("\nRunning backtest...")
best_ret, best_params, grid_df = run_universe_bubble_hourly(
    hourly_open  = ho,
    hourly_close = hc,
    ma_window_grid   = [50, 100],
    z_window_grid    = [100, 200],
    buy_threshold_grid   = [0.7, 0.8, 0.9],
    short_threshold_grid = [0.85, 0.95],
    hold_hours_grid  = [2, 4, 8],
    top_n_grid       = [10, 20],
    transaction_cost  = 0.001,
    short_borrow_rate = 0.08,
)

print(f"\nBest Params: ma={best_params['ma_window']} z={best_params['z_window']} "
      f"buy_thresh={best_params['buy_threshold']} short_thresh={best_params['short_threshold']} "
      f"hold={best_params['hold_hours']} top_n={best_params['top_n']}")

# Now re-run the best params manually to decompose long/short
from strategies.universe_bubble_hourly import _bubble_scores_matrix
from itertools import product

ho_copy = ho.copy()
hc_copy = hc.copy()
tickers = hc_copy.columns.tolist()
n = len(hc_copy)

ma = best_params['ma_window']
z = best_params['z_window']
buy_thresh = best_params['buy_threshold']
short_thresh = best_params['short_threshold']
hold = best_params['hold_hours']
top_n = best_params['top_n']
transaction_cost = 0.001
short_borrow_rate = 0.08

hourly_borrow = short_borrow_rate / (TRADING_DAYS * 6.5)
borrow = hourly_borrow * hold

# Compute bubble scores
raw = _bubble_scores_matrix(hc_copy, ma, z)
scores = raw.shift(1)

# Separate tracking for long and short
long_trades = []
short_trades = []

for i in range(max(ma, z) + 1, n - hold, hold):
    sig = scores.iloc[i].dropna()
    if sig.empty:
        continue

    long_cands  = sig[sig < -buy_thresh].nsmallest(top_n)
    short_cands = sig[sig >  short_thresh].nlargest(top_n)

    has_long  = len(long_cands)  > 0
    has_short = len(short_cands) > 0
    if not has_long and not has_short:
        continue

    weight = 0.5 if (has_long and has_short) else 1.0
    exit_i = min(i + hold - 1, n - 1)

    # LONG trades
    if has_long:
        for tkr in long_cands.index:
            ep = ho_copy.iat[i,      ho_copy.columns.get_loc(tkr)]
            xp = hc_copy.iat[exit_i, hc_copy.columns.get_loc(tkr)]
            if pd.isna(ep) or pd.isna(xp) or ep <= 0:
                continue
            ret = ((xp/ep - 1) - transaction_cost) * weight
            long_trades.append({
                "date": hc_copy.index[i].normalize(),
                "ticker": tkr,
                "entry": ep,
                "exit": xp,
                "ret": ret,
            })

    # SHORT trades
    if has_short:
        for tkr in short_cands.index:
            ep = ho_copy.iat[i,      ho_copy.columns.get_loc(tkr)]
            xp = hc_copy.iat[exit_i, hc_copy.columns.get_loc(tkr)]
            if pd.isna(ep) or pd.isna(xp) or ep <= 0:
                continue
            ret = (-(xp/ep - 1) - transaction_cost - borrow) * weight
            short_trades.append({
                "date": hc_copy.index[i].normalize(),
                "ticker": tkr,
                "entry": ep,
                "exit": xp,
                "ret": ret,
            })

# Convert to DataFrames
long_df = pd.DataFrame(long_trades)
short_df = pd.DataFrame(short_trades)

# Aggregate by date
if len(long_df) > 0:
    long_daily = long_df.groupby("date")["ret"].sum()
    data_end = hc_copy.index[-1].normalize()
    all_dates = pd.date_range(long_daily.index.min(), data_end, freq="B")
    long_daily_full = long_daily.reindex(all_dates, fill_value=0.0)

    long_wealth = (1 + long_daily_full).cumprod(); long_wealth = long_wealth / long_wealth.iloc[0]
    long_std = long_daily_full.std()
    long_sharpe = float(np.sqrt(TRADING_DAYS) * long_daily_full.mean() / long_std) if long_std > 0 else 0
    long_ds = long_daily_full[long_daily_full < 0].std()
    long_sortino = float(np.sqrt(TRADING_DAYS) * long_daily_full.mean() / long_ds) if long_ds > 0 else 0
    long_mdd = float((long_wealth / long_wealth.cummax() - 1).min())
    long_total_ret = float(long_wealth.iloc[-1] - 1)
    long_wr = float((long_df["ret"] > 0).sum() / len(long_df)) if len(long_df) > 0 else 0
else:
    long_sharpe = long_sortino = long_mdd = long_total_ret = long_wr = 0

if len(short_df) > 0:
    short_daily = short_df.groupby("date")["ret"].sum()
    data_end = hc_copy.index[-1].normalize()
    all_dates = pd.date_range(short_daily.index.min(), data_end, freq="B")
    short_daily_full = short_daily.reindex(all_dates, fill_value=0.0)

    short_wealth = (1 + short_daily_full).cumprod(); short_wealth = short_wealth / short_wealth.iloc[0]
    short_std = short_daily_full.std()
    short_sharpe = float(np.sqrt(TRADING_DAYS) * short_daily_full.mean() / short_std) if short_std > 0 else 0
    short_ds = short_daily_full[short_daily_full < 0].std()
    short_sortino = float(np.sqrt(TRADING_DAYS) * short_daily_full.mean() / short_ds) if short_ds > 0 else 0
    short_mdd = float((short_wealth / short_wealth.cummax() - 1).min())
    short_total_ret = float(short_wealth.iloc[-1] - 1)
    short_wr = float((short_df["ret"] > 0).sum() / len(short_df)) if len(short_df) > 0 else 0
else:
    short_sharpe = short_sortino = short_mdd = short_total_ret = short_wr = 0

print("\n" + "="*80)
print("LONG vs SHORT DECOMPOSITION")
print("="*80)
print(f"\n{'Metric':<25} {'LONG':<15} {'SHORT':<15} {'COMBINED':<15}")
print("-" * 70)
print(f"{'Sharpe Ratio':<25} {long_sharpe:>14.3f} {short_sharpe:>14.3f} {best_params['Sharpe']:>14.3f}")
print(f"{'Sortino Ratio':<25} {long_sortino:>14.3f} {short_sortino:>14.3f} {best_params['Sortino']:>14.3f}")
print(f"{'Total Return':<25} {long_total_ret:>13.1%} {short_total_ret:>13.1%} {best_params['Total_Return']:>13.1%}")
print(f"{'Max Drawdown':<25} {long_mdd:>13.1%} {short_mdd:>13.1%} {best_params['Max_DD']:>13.1%}")
print(f"{'Win Rate':<25} {long_wr:>13.1%} {short_wr:>13.1%} {best_params['Win_Rate']:>13.1%}")
print(f"{'Number of Trades':<25} {len(long_df):>15} {len(short_df):>15} {best_params['n_trades']:>15}")

print("\n" + "="*80)
print("TRADE STATISTICS")
print("="*80)
if len(long_df) > 0:
    print(f"\nLONG TRADES:")
    print(f"  Total:        {len(long_df)}")
    print(f"  Avg Return:   {long_df['ret'].mean():+.3%}")
    print(f"  Std Dev:      {long_df['ret'].std():.3%}")
    print(f"  Min Return:   {long_df['ret'].min():+.3%}")
    print(f"  Max Return:   {long_df['ret'].max():+.3%}")
    print(f"  Positive:     {(long_df['ret'] > 0).sum()} ({(long_df['ret'] > 0).sum()/len(long_df):.1%})")

if len(short_df) > 0:
    print(f"\nSHORT TRADES:")
    print(f"  Total:        {len(short_df)}")
    print(f"  Avg Return:   {short_df['ret'].mean():+.3%}")
    print(f"  Std Dev:      {short_df['ret'].std():.3%}")
    print(f"  Min Return:   {short_df['ret'].min():+.3%}")
    print(f"  Max Return:   {short_df['ret'].max():+.3%}")
    print(f"  Positive:     {(short_df['ret'] > 0).sum()} ({(short_df['ret'] > 0).sum()/len(short_df):.1%})")

print("\n" + "="*80)
