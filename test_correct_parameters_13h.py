"""
TEST CORRECT DOCUMENTED PARAMETERS: 13h hold (NOT 104h!)
=========================================================
"""
import numpy as np
import pandas as pd
from data.universe import get_universe
from data.intraday_loader import load_hourly_bars
from strategies.universe_bubble_hourly import _bubble_scores_matrix

TRADING_DAYS = 252

print("=" * 100)
print("CORRECTED: TESTING ACTUAL DOCUMENTED PARAMETERS")
print("=" * 100)

universe = get_universe()
ho, hc = load_hourly_bars(universe, use_cache=True)
print(f"\nData: {hc.shape[1]} tickers x {hc.shape[0]} bars")
print(f"Period: {hc.index[0].date()} to {hc.index[-1].date()}")

# CORRECT DOCUMENTED PARAMETERS (from grid search peak)
ma = 104
z = 104
buy_thresh = 0.8
hold = 13  # NOT 104! THIS WAS THE ERROR!
top_n = 20
transaction_cost = 0.001

print(f"\nCORRECT PARAMETERS (from grid search peak):")
print(f"  MA Window: {ma}h")
print(f"  Z Window: {z}h")
print(f"  Buy Threshold: -{buy_thresh} (oversold only)")
print(f"  Hold Period: {hold}h  [CRITICAL: 13h NOT 104h!]")
print(f"  Top N: {top_n}")

# Compute bubble scores
print("\nComputing bubble scores...")
raw = _bubble_scores_matrix(hc, ma, z)
scores = raw.shift(1)

long_trades = []
n = len(hc)

print("Running backtest...")
for i in range(max(ma, z) + 1, n - hold, hold):
    sig = scores.iloc[i].dropna()
    if sig.empty:
        continue

    long_cands = sig[sig < -buy_thresh].nsmallest(top_n)
    if len(long_cands) == 0:
        continue

    exit_i = min(i + hold - 1, n - 1)
    entry_dt = hc.index[i].normalize()

    for tkr in long_cands.index:
        ep = ho.iat[i, ho.columns.get_loc(tkr)]
        xp = hc.iat[exit_i, hc.columns.get_loc(tkr)]
        if pd.isna(ep) or pd.isna(xp) or ep <= 0:
            continue

        price_return = (xp / ep - 1)
        net_ret = price_return - transaction_cost

        long_trades.append({
            "date": entry_dt,
            "ret": net_ret,
            "ticker": tkr,
        })

print(f"Total trades: {len(long_trades)}")

if len(long_trades) > 5:
    df = pd.DataFrame(long_trades)
    daily = df.groupby("date")["ret"].sum()
    data_end = hc.index[-1].normalize()
    all_dates = pd.date_range(daily.index.min(), data_end, freq="B")
    daily_full = daily.reindex(all_dates, fill_value=0.0)

    wealth = (1 + daily_full).cumprod()
    wealth = wealth / wealth.iloc[0]

    mean_ret = daily_full.mean()
    std_ret = daily_full.std()
    sharpe = (mean_ret * TRADING_DAYS) / (std_ret * np.sqrt(TRADING_DAYS)) if std_ret > 0 else np.nan

    ds = daily_full[daily_full < 0].std()
    sortino = (mean_ret * TRADING_DAYS) / (ds * np.sqrt(TRADING_DAYS)) if ds > 0 else np.nan

    mdd = (wealth / wealth.cummax() - 1).min()
    total_ret = wealth.iloc[-1] - 1
    wr = (df["ret"] > 0).sum() / len(df)

    print("\n" + "=" * 100)
    print("RESULTS WITH CORRECT PARAMETERS (13h hold):")
    print("=" * 100)
    print(f"\nBacktest Period: {hc.index[0].date()} to {hc.index[-1].date()}")
    print(f"  Sharpe Ratio:     {sharpe:.3f}")
    print(f"  Sortino Ratio:    {sortino:.3f}")
    print(f"  Total Return:     {total_ret:+.1%}")
    print(f"  Max Drawdown:     {mdd:.1%}")
    print(f"  Win Rate:         {wr:.1%}")
    print(f"  Trades:           {len(df)}")

    print(f"\n" + "=" * 100)
    print("COMPARISON TO DOCUMENTED (2019-2026, 8-year backtest):")
    print("=" * 100)
    print(f"  Documented Sharpe:      2.648  (vs our {sharpe:.3f})")
    print(f"  Documented Return:      +38.1% annual (vs our {total_ret:+.1%} on 2024-2026)")
    print(f"  Documented Max_DD:      -10.1% (vs our {mdd:.1%})")
    print(f"  Documented Win Rate:    56.6% (vs our {wr:.1%})")

    print(f"\n" + "=" * 100)
    print("INTERPRETATION:")
    print("=" * 100)
    print(f"""
The 2024-2026 market is harder than 2019-2026 average.
Documented results used 8-year period including:
  - 2020-2021: COVID boom (great for contrarian)
  - 2022: Bear market (exceptional for contrarian)
  - 2023-2024: Recovery years

Our 2024-2026 backtest is on recent/current market only.
    """)
