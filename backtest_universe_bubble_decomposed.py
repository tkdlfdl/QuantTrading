"""
Universe Bubble Strategy - Long vs Short Decomposed Performance Analysis
=========================================================================

Backtests the universe bubble strategy on individual stocks and provides
detailed decomposition of long and short performance metrics.

No approximations - all metrics from actual backtest data.
Backtest period: 2024-06-20 to 2026-06-18
"""

import numpy as np
import pandas as pd
from data.universe import get_universe
from data.intraday_loader import load_hourly_bars
from strategies.universe_bubble_hourly import _bubble_scores_matrix
from pathlib import Path

TRADING_DAYS = 252
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("UNIVERSE BUBBLE STRATEGY - LONG vs SHORT DECOMPOSED BACKTEST")
print("=" * 80)

# Load data
print("\nLoading universe hourly bars...")
universe = get_universe()
ho, hc = load_hourly_bars(universe, use_cache=True)
print(f"Data: {hc.shape[1]} tickers × {hc.shape[0]} bars")
print(f"Period: {hc.index[0].date()} to {hc.index[-1].date()}")

# Strategy parameters (best from grid search)
ma = 100
z = 100
buy_thresh = 0.8
short_thresh = 0.95
hold = 8
top_n = 10
transaction_cost = 0.001
short_borrow_rate = 0.08

hourly_borrow = short_borrow_rate / (TRADING_DAYS * 6.5)
borrow = hourly_borrow * hold

print(f"\nParameters: ma={ma}h, z={z}h, buy_thresh={buy_thresh}, "
      f"short_thresh={short_thresh}, hold={hold}h, top_n={top_n}")

# Compute bubble scores
print("\nComputing bubble scores...")
raw = _bubble_scores_matrix(hc, ma, z)
scores = raw.shift(1)  # No lookahead bias

# Track trades separately
long_trades = []
short_trades = []
all_trades = []

n = len(hc)
tickers = hc.columns.tolist()

print("Running backtest...")
for i in range(max(ma, z) + 1, n - hold, hold):
    sig = scores.iloc[i].dropna()
    if sig.empty:
        continue

    long_cands  = sig[sig < -buy_thresh].nsmallest(top_n)
    short_cands = sig[sig >  short_thresh].nlargest(top_n)

    has_long  = len(long_cands) > 0
    has_short = len(short_cands) > 0

    if not has_long and not has_short:
        continue

    weight = 0.5 if (has_long and has_short) else 1.0
    exit_i = min(i + hold - 1, n - 1)
    entry_dt = hc.index[i].normalize()
    exit_dt = hc.index[exit_i].normalize()

    # LONG trades
    if has_long:
        for tkr in long_cands.index:
            ep = ho.iat[i, ho.columns.get_loc(tkr)]
            xp = hc.iat[exit_i, hc.columns.get_loc(tkr)]
            if pd.isna(ep) or pd.isna(xp) or ep <= 0:
                continue

            price_return = (xp / ep - 1)
            net_ret = (price_return - transaction_cost) * weight

            trade = {
                "date": entry_dt,
                "exit_date": exit_dt,
                "ticker": tkr,
                "side": "LONG",
                "entry_price": ep,
                "exit_price": xp,
                "price_return": price_return,
                "cost": transaction_cost * weight,
                "net_ret": net_ret,
            }
            long_trades.append(trade)
            all_trades.append(trade)

    # SHORT trades
    if has_short:
        for tkr in short_cands.index:
            ep = ho.iat[i, ho.columns.get_loc(tkr)]
            xp = hc.iat[exit_i, hc.columns.get_loc(tkr)]
            if pd.isna(ep) or pd.isna(xp) or ep <= 0:
                continue

            price_return = (xp / ep - 1)
            net_ret = (-(price_return) - transaction_cost - borrow) * weight

            trade = {
                "date": entry_dt,
                "exit_date": exit_dt,
                "ticker": tkr,
                "side": "SHORT",
                "entry_price": ep,
                "exit_price": xp,
                "price_return": price_return,
                "cost": (transaction_cost + borrow) * weight,
                "net_ret": net_ret,
            }
            short_trades.append(trade)
            all_trades.append(trade)

print(f"Total trades: {len(all_trades)} ({len(long_trades)} long, {len(short_trades)} short)")

# Convert to DataFrames
long_df = pd.DataFrame(long_trades) if long_trades else pd.DataFrame()
short_df = pd.DataFrame(short_trades) if short_trades else pd.DataFrame()
all_df = pd.DataFrame(all_trades)

# Helper function to calculate metrics
def calculate_metrics(df, name="Strategy"):
    if df.empty or len(df) < 5:
        return {
            "Side": name,
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "Total_Return": np.nan,
            "Max_DD": np.nan,
            "Win_Rate": np.nan,
            "Avg_Return": np.nan,
            "Std_Dev": np.nan,
            "Min_Return": np.nan,
            "Max_Return": np.nan,
            "n_trades": len(df),
        }

    # Aggregate by date
    daily = df.groupby("date")["net_ret"].sum()
    data_end = hc.index[-1].normalize()
    all_dates = pd.date_range(daily.index.min(), data_end, freq="B")
    daily_full = daily.reindex(all_dates, fill_value=0.0)

    # Calculate metrics
    wealth = (1 + daily_full).cumprod()
    wealth = wealth / wealth.iloc[0]

    mean_ret = daily_full.mean()
    std_ret = daily_full.std()
    sharpe = (mean_ret * TRADING_DAYS) / (std_ret * np.sqrt(TRADING_DAYS)) if std_ret > 0 else np.nan

    downside_rets = daily_full[daily_full < 0]
    downside_std = downside_rets.std()
    sortino = (mean_ret * TRADING_DAYS) / (downside_std * np.sqrt(TRADING_DAYS)) if downside_std > 0 else np.nan

    mdd = (wealth / wealth.cummax() - 1).min()
    total_ret = wealth.iloc[-1] - 1
    wr = (df["net_ret"] > 0).sum() / len(df) if len(df) > 0 else 0

    return {
        "Side": name,
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "Total_Return": float(total_ret),
        "Max_DD": float(mdd),
        "Win_Rate": float(wr),
        "Avg_Return": float(df["net_ret"].mean()),
        "Std_Dev": float(df["net_ret"].std()),
        "Min_Return": float(df["net_ret"].min()),
        "Max_Return": float(df["net_ret"].max()),
        "n_trades": len(df),
    }

# Calculate metrics for each side
long_metrics = calculate_metrics(long_df, "LONG")
short_metrics = calculate_metrics(short_df, "SHORT")
all_metrics = calculate_metrics(all_df, "COMBINED")

# Print summary
print("\n" + "=" * 80)
print("PERFORMANCE SUMMARY")
print("=" * 80)

summary_df = pd.DataFrame([long_metrics, short_metrics, all_metrics])
print(summary_df[["Side", "Sharpe", "Sortino", "Total_Return", "Max_DD", "Win_Rate", "n_trades"]].to_string(index=False))

# Detailed breakdown
print("\n" + "=" * 80)
print("DETAILED METRICS BY SIDE")
print("=" * 80)

for side, metrics in [("LONG", long_metrics), ("SHORT", short_metrics), ("COMBINED", all_metrics)]:
    print(f"\n{side}:")
    print(f"  Sharpe Ratio:     {metrics['Sharpe']:>10.3f}")
    print(f"  Sortino Ratio:    {metrics['Sortino']:>10.3f}")
    print(f"  Total Return:     {metrics['Total_Return']:>10.1%}")
    print(f"  Max Drawdown:     {metrics['Max_DD']:>10.1%}")
    print(f"  Win Rate:         {metrics['Win_Rate']:>10.1%}")
    print(f"  Avg Return/Trade: {metrics['Avg_Return']:>10.3%}")
    print(f"  Std Dev:          {metrics['Std_Dev']:>10.3%}")
    print(f"  Min Return:       {metrics['Min_Return']:>10.3%}")
    print(f"  Max Return:       {metrics['Max_Return']:>10.3%}")
    print(f"  # Trades:         {metrics['n_trades']:>10}")

# Yearly breakdown
print("\n" + "=" * 80)
print("YEARLY BREAKDOWN")
print("=" * 80)

def yearly_breakdown(df, side_name=""):
    if df.empty:
        return

    df_copy = df.copy()
    df_copy["year"] = pd.to_datetime(df_copy["date"]).dt.year

    print(f"\n{side_name}:")
    for year in sorted(df_copy["year"].unique()):
        year_trades = df_copy[df_copy["year"] == year]
        year_daily = year_trades.groupby("date")["net_ret"].sum()

        if len(year_daily) > 0:
            wealth = (1 + year_daily).cumprod()
            wealth = wealth / wealth.iloc[0]

            mean_ret = year_daily.mean()
            std_ret = year_daily.std()
            sharpe = (mean_ret * TRADING_DAYS) / (std_ret * np.sqrt(TRADING_DAYS)) if std_ret > 0 else 0

            mdd = (wealth / wealth.cummax() - 1).min()
            total_ret = wealth.iloc[-1] - 1
            wr = (year_trades["net_ret"] > 0).sum() / len(year_trades)

            print(f"  {year}: Return={total_ret:>7.1%}  Sharpe={sharpe:>6.3f}  Max_DD={mdd:>8.1%}  "
                  f"Win_Rate={wr:>6.1%}  Trades={len(year_trades):>4}")

yearly_breakdown(long_df, "LONG")
yearly_breakdown(short_df, "SHORT")
yearly_breakdown(all_df, "COMBINED")

# Top and bottom trades
print("\n" + "=" * 80)
print("TOP 10 & BOTTOM 10 TRADES")
print("=" * 80)

if len(long_df) > 0:
    print("\nLONG - Top 10 Trades:")
    top_long = long_df.nlargest(10, "net_ret")[["date", "ticker", "entry_price", "exit_price", "net_ret"]]
    for idx, row in top_long.iterrows():
        print(f"  {row['date'].date()} {row['ticker']:6} Entry=${row['entry_price']:7.2f} "
              f"Exit=${row['exit_price']:7.2f} Return={row['net_ret']:+7.3%}")

    print("\nLONG - Bottom 10 Trades:")
    bot_long = long_df.nsmallest(10, "net_ret")[["date", "ticker", "entry_price", "exit_price", "net_ret"]]
    for idx, row in bot_long.iterrows():
        print(f"  {row['date'].date()} {row['ticker']:6} Entry=${row['entry_price']:7.2f} "
              f"Exit=${row['exit_price']:7.2f} Return={row['net_ret']:+7.3%}")

if len(short_df) > 0:
    print("\nSHORT - Top 10 Trades:")
    top_short = short_df.nlargest(10, "net_ret")[["date", "ticker", "entry_price", "exit_price", "net_ret"]]
    for idx, row in top_short.iterrows():
        print(f"  {row['date'].date()} {row['ticker']:6} Entry=${row['entry_price']:7.2f} "
              f"Exit=${row['exit_price']:7.2f} Return={row['net_ret']:+7.3%}")

    print("\nSHORT - Bottom 10 Trades:")
    bot_short = short_df.nsmallest(10, "net_ret")[["date", "ticker", "entry_price", "exit_price", "net_ret"]]
    for idx, row in bot_short.iterrows():
        print(f"  {row['date'].date()} {row['ticker']:6} Entry=${row['entry_price']:7.2f} "
              f"Exit=${row['exit_price']:7.2f} Return={row['net_ret']:+7.3%}")

# Save results
print("\n" + "=" * 80)
all_df.to_csv(OUT_DIR / "universe_bubble_all_trades.csv", index=False)
long_df.to_csv(OUT_DIR / "universe_bubble_long_trades.csv", index=False)
short_df.to_csv(OUT_DIR / "universe_bubble_short_trades.csv", index=False)
summary_df.to_csv(OUT_DIR / "universe_bubble_summary.csv", index=False)
print(f"Results saved to {OUT_DIR}/")
print("=" * 80)
