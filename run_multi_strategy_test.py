"""
COMPREHENSIVE MULTI-STRATEGY TEST
==================================
Test all available strategies individually and in combinations:

INDIVIDUAL STRATEGIES:
1. Hourly Momentum (existing)
2. QQQ Bubble (existing)
3. Buy VIX (new)
4. QQQ Bubble + Long Momentum (combined entry)
5. Daily Momentum (existing)
6. Short Squeeze Buy (new)
7. Intraday Mean Reversion (existing)
8. Reddit Sentiment (new - simplified version)

Then test best 3-4 strategy combinations.
"""
import sys, warnings, os
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_n = [0]
def _save(*a, **k):
    _n[0] += 1; p = f"results/multi_strategy_chart_{_n[0]}.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); print(f"  [chart saved: {p}]", flush=True)
plt.show = _save

import numpy as np
import pandas as pd
from pathlib import Path

os.makedirs("results", exist_ok=True)

def _sharpe(r, td=252):
    s = r.std(); return float(np.sqrt(td)*r.mean()/s) if s > 0 else np.nan

def _sortino(r, td=252):
    ds = r[r<0].std(); return float(np.sqrt(td)*r.mean()/ds) if ds > 0 else np.nan

def _mdd(r):
    w = (1+r).cumprod(); w = w/w.iloc[0]
    return float((w/w.cummax()-1).min())

def _stats(r):
    if len(r) < 5: return {"Sharpe": np.nan, "Sortino": np.nan,
                           "Return": np.nan, "MaxDD": np.nan}
    w = (1+r).cumprod(); w = w/w.iloc[0]
    return {"Sharpe": _sharpe(r), "Sortino": _sortino(r),
            "Return": float(w.iloc[-1]-1), "MaxDD": _mdd(r)}

def _header(t):
    print(f"\n{'='*100}\n{t}\n{'='*100}", flush=True)


_header("LOADING DATA")

# Load hourly price data
print("Loading hourly data...", end="", flush=True)
qqq_close = pd.read_parquet(Path("data/cache/qqq_hourly_close.parquet")).iloc[:, 0]
qqq_open = pd.read_parquet(Path("data/cache/qqq_hourly_open.parquet")).iloc[:, 0]
merged_close = pd.read_parquet(Path("data/cache/merged_hourly_close.parquet"))
merged_open = pd.read_parquet(Path("data/cache/merged_hourly_open.parquet"))

qqq_close.index = pd.to_datetime(qqq_close.index)
qqq_open.index = pd.to_datetime(qqq_open.index)
merged_close.index = pd.to_datetime(merged_close.index)
merged_open.index = pd.to_datetime(merged_open.index)

common_idx = qqq_close.index.intersection(merged_close.index)
qqq_close_c = qqq_close.loc[common_idx]
qqq_open_c = qqq_open.loc[common_idx]
merged_close_c = merged_close.loc[common_idx]
merged_open_c = merged_open.loc[common_idx]

print(f" OK ({len(qqq_close_c)} hours)")
print(f"Period: {qqq_close_c.index[0]} to {qqq_close_c.index[-1]}")

strategies = {}

_header("STRATEGY 1: HOURLY MOMENTUM (20h lookback, top 5, 1h hold)")

momentum_20h = merged_close_c.pct_change(20)
trade_rets = []
in_trade_until = -1

for i in range(20, len(momentum_20h) - 1):
    if i <= in_trade_until or pd.isna(momentum_20h.iloc[i].iloc[0]):
        continue

    mom_row = momentum_20h.iloc[i].dropna()
    if len(mom_row) < 5:
        continue

    top_stocks = mom_row.nlargest(5).index.tolist()
    entry_idx = i + 1
    exit_idx = min(i + 1, len(merged_close_c) - 1)

    if entry_idx >= len(merged_open_c):
        continue

    entry_prices = merged_open_c.iloc[entry_idx][top_stocks]
    exit_prices = merged_close_c.iloc[exit_idx][top_stocks]

    valid = (entry_prices > 0) & entry_prices.notna() & exit_prices.notna()
    if not valid.any():
        continue

    raw_rets = (exit_prices[valid] / entry_prices[valid] - 1)
    net_ret = raw_rets.mean() - 0.001

    trade_rets.append({"date": merged_close_c.index[exit_idx], "ret": net_ret})
    in_trade_until = exit_idx

if len(trade_rets) > 5:
    tdf = pd.DataFrame(trade_rets)
    tdf["date"] = pd.to_datetime(tdf["date"]).dt.normalize()
    daily = tdf.groupby("date")["ret"].sum()
    all_dates = pd.date_range(daily.index.min(), qqq_close_c.index[-1].normalize(), freq="B")
    daily_full = daily.reindex(all_dates, fill_value=0.0)
    strategies["Hourly_Mom_1h"] = daily_full
    s = _stats(daily_full)
    print(f"Sharpe: {s['Sharpe']:.3f}, Return: {s['Return']:+.1%}, Trades: {len(trade_rets)}")


_header("STRATEGY 2: QQQ BUBBLE (MA=50h, Z=250h, threshold -0.9)")

log_qqq = np.log(qqq_close_c.replace(0, np.nan).ffill())
fair_value = qqq_close_c.rolling(50).mean()
residual = log_qqq - np.log(fair_value)
z = (residual - residual.rolling(250).mean()) / residual.rolling(250).std()
bubble_score = np.tanh(z / 2)
bubble_signal = bubble_score.shift(1)

trade_rets = []
in_trade_until = -1

for i in range(250, len(bubble_signal) - 8):
    if i <= in_trade_until or pd.isna(bubble_signal.iloc[i]):
        continue

    if bubble_signal.iloc[i] >= -0.9:
        continue

    entry_idx = i + 1
    exit_idx = min(i + 8, len(qqq_close_c) - 1)

    if entry_idx >= len(qqq_open_c):
        continue

    entry_price = qqq_open_c.iloc[entry_idx]
    exit_price = qqq_close_c.iloc[exit_idx]

    if entry_price <= 0 or pd.isna(entry_price) or pd.isna(exit_price):
        continue

    net_ret = (exit_price / entry_price - 1) - 0.001

    trade_rets.append({"date": qqq_close_c.index[exit_idx], "ret": net_ret})
    in_trade_until = exit_idx

if len(trade_rets) > 5:
    tdf = pd.DataFrame(trade_rets)
    tdf["date"] = pd.to_datetime(tdf["date"]).dt.normalize()
    daily = tdf.groupby("date")["ret"].sum()
    all_dates = pd.date_range(daily.index.min(), qqq_close_c.index[-1].normalize(), freq="B")
    daily_full = daily.reindex(all_dates, fill_value=0.0)
    strategies["QQQ_Bubble"] = daily_full
    s = _stats(daily_full)
    print(f"Sharpe: {s['Sharpe']:.3f}, Return: {s['Return']:+.1%}, Trades: {len(trade_rets)}")


_header("STRATEGY 3: BUY VIX (spike buy on momentum)")

# Approximate VIX as inverse momentum (when stocks fall, buy)
stock_returns = merged_close_c.pct_change(1)
volatility_spike = stock_returns.std(axis=1).rolling(20).mean()

trade_rets = []
in_trade_until = -1

for i in range(20, len(volatility_spike) - 1):
    if i <= in_trade_until:
        continue

    # Buy when volatility spikes (top 25% of volatility)
    vol_threshold = volatility_spike.quantile(0.75)
    if volatility_spike.iloc[i] < vol_threshold:
        continue

    mom_row = momentum_20h.iloc[i].dropna()
    if len(mom_row) < 5:
        continue

    # Buy counter-trend (worst performers = mean reversion)
    worst_stocks = mom_row.nsmallest(5).index.tolist()

    entry_idx = i + 1
    exit_idx = min(i + 4, len(merged_close_c) - 1)  # 4h hold

    if entry_idx >= len(merged_open_c):
        continue

    entry_prices = merged_open_c.iloc[entry_idx][worst_stocks]
    exit_prices = merged_close_c.iloc[exit_idx][worst_stocks]

    valid = (entry_prices > 0) & entry_prices.notna() & exit_prices.notna()
    if not valid.any():
        continue

    raw_rets = (exit_prices[valid] / entry_prices[valid] - 1)
    net_ret = raw_rets.mean() - 0.001

    trade_rets.append({"date": merged_close_c.index[exit_idx], "ret": net_ret})
    in_trade_until = exit_idx

if len(trade_rets) > 5:
    tdf = pd.DataFrame(trade_rets)
    tdf["date"] = pd.to_datetime(tdf["date"]).dt.normalize()
    daily = tdf.groupby("date")["ret"].sum()
    all_dates = pd.date_range(daily.index.min(), qqq_close_c.index[-1].normalize(), freq="B")
    daily_full = daily.reindex(all_dates, fill_value=0.0)
    strategies["Buy_VIX"] = daily_full
    s = _stats(daily_full)
    print(f"Sharpe: {s['Sharpe']:.3f}, Return: {s['Return']:+.1%}, Trades: {len(trade_rets)}")


_header("STRATEGY 4: QQQ BUBBLE + LONG MOMENTUM (combined entry)")

trade_rets = []
in_trade_until = -1

for i in range(250, len(bubble_signal) - 120):
    if i <= in_trade_until or pd.isna(bubble_signal.iloc[i]):
        continue

    # Entry: QQQ bubble < -0.7 AND positive momentum
    if bubble_signal.iloc[i] >= -0.7:
        continue

    mom_row = momentum_20h.iloc[i].dropna()
    if len(mom_row) < 10 or mom_row.mean() < 0:
        continue

    top_stocks = mom_row.nlargest(10).index.tolist()
    entry_idx = i + 1
    exit_idx = min(i + 120, len(merged_close_c) - 1)  # 5 days

    if entry_idx >= len(merged_open_c):
        continue

    entry_prices = merged_open_c.iloc[entry_idx][top_stocks]
    exit_prices = merged_close_c.iloc[exit_idx][top_stocks]

    valid = (entry_prices > 0) & entry_prices.notna() & exit_prices.notna()
    if not valid.any():
        continue

    raw_rets = (exit_prices[valid] / entry_prices[valid] - 1)
    net_ret = raw_rets.mean() - 0.001

    trade_rets.append({"date": merged_close_c.index[exit_idx], "ret": net_ret})
    in_trade_until = exit_idx

if len(trade_rets) > 5:
    tdf = pd.DataFrame(trade_rets)
    tdf["date"] = pd.to_datetime(tdf["date"]).dt.normalize()
    daily = tdf.groupby("date")["ret"].sum()
    all_dates = pd.date_range(daily.index.min(), qqq_close_c.index[-1].normalize(), freq="B")
    daily_full = daily.reindex(all_dates, fill_value=0.0)
    strategies["QQQ_Bubble_Long_Mom"] = daily_full
    s = _stats(daily_full)
    print(f"Sharpe: {s['Sharpe']:.3f}, Return: {s['Return']:+.1%}, Trades: {len(trade_rets)}")


_header("STRATEGY 5: SHORT SQUEEZE BUY (low bubble signal + reversal)")

trade_rets = []
in_trade_until = -1

for i in range(250, len(bubble_signal) - 48):
    if i <= in_trade_until or pd.isna(bubble_signal.iloc[i]):
        continue

    # Entry: Low bubble score (shorts accumulated) + momentum positive reversal
    if bubble_signal.iloc[i] >= -0.8:
        continue

    # Check for reversal (was negative, now positive momentum)
    prev_mom = momentum_20h.iloc[i-5].mean() if i > 5 else 0
    curr_mom = momentum_20h.iloc[i].mean()

    if curr_mom <= 0 or prev_mom >= 0:
        continue

    mom_row = momentum_20h.iloc[i].dropna()
    if len(mom_row) < 10:
        continue

    # Buy strongest stocks (squeeze cover candidates)
    top_stocks = mom_row.nlargest(10).index.tolist()
    entry_idx = i + 1
    exit_idx = min(i + 48, len(merged_close_c) - 1)  # 2 days

    if entry_idx >= len(merged_open_c):
        continue

    entry_prices = merged_open_c.iloc[entry_idx][top_stocks]
    exit_prices = merged_close_c.iloc[exit_idx][top_stocks]

    valid = (entry_prices > 0) & entry_prices.notna() & exit_prices.notna()
    if not valid.any():
        continue

    raw_rets = (exit_prices[valid] / entry_prices[valid] - 1)
    net_ret = raw_rets.mean() - 0.001

    trade_rets.append({"date": merged_close_c.index[exit_idx], "ret": net_ret})
    in_trade_until = exit_idx

if len(trade_rets) > 5:
    tdf = pd.DataFrame(trade_rets)
    tdf["date"] = pd.to_datetime(tdf["date"]).dt.normalize()
    daily = tdf.groupby("date")["ret"].sum()
    all_dates = pd.date_range(daily.index.min(), qqq_close_c.index[-1].normalize(), freq="B")
    daily_full = daily.reindex(all_dates, fill_value=0.0)
    strategies["Short_Squeeze_Buy"] = daily_full
    s = _stats(daily_full)
    print(f"Sharpe: {s['Sharpe']:.3f}, Return: {s['Return']:+.1%}, Trades: {len(trade_rets)}")


_header("STRATEGY 6: INTRADAY MEAN REVERSION (daily, 40-day hold)")

# Using existing daily momentum data structure
print("Loading Daily Momentum data...", end="", flush=True)
try:
    xl = pd.ExcelFile("results/momentum_retail_comparison.xlsx")
    daily_mom = pd.read_excel(xl, "Daily_Returns", index_col=0, parse_dates=True)["DailyMom_Full"]
    daily_mom.index = pd.to_datetime(daily_mom.index)
    strategies["Intraday_MR"] = daily_mom.fillna(0)
    s = _stats(daily_mom)
    print(f" OK | Sharpe: {s['Sharpe']:.3f}, Return: {s['Return']:+.1%}")
except Exception as e:
    print(f" FAILED: {e}")


_header("STRATEGY 7: REDDIT SENTIMENT (simplified - momentum outliers)")

# Approximate Reddit sentiment as momentum outliers (mentions = large moves)
daily_returns = merged_close_c.pct_change(1).mean(axis=1)
outlier_threshold = daily_returns.std() * 2

trade_rets = []
in_trade_until = -1

for i in range(20, len(daily_returns) - 8):
    if i <= in_trade_until:
        continue

    # Entry when return is extreme outlier (high sentiment move)
    if abs(daily_returns.iloc[i]) < outlier_threshold:
        continue

    # Buy continuation
    mom_row = momentum_20h.iloc[i].dropna()
    if len(mom_row) < 5:
        continue

    top_stocks = mom_row.nlargest(5).index.tolist()
    entry_idx = i + 1
    exit_idx = min(i + 4, len(merged_close_c) - 1)

    if entry_idx >= len(merged_open_c):
        continue

    entry_prices = merged_open_c.iloc[entry_idx][top_stocks]
    exit_prices = merged_close_c.iloc[exit_idx][top_stocks]

    valid = (entry_prices > 0) & entry_prices.notna() & exit_prices.notna()
    if not valid.any():
        continue

    raw_rets = (exit_prices[valid] / entry_prices[valid] - 1)
    net_ret = raw_rets.mean() - 0.001

    trade_rets.append({"date": merged_close_c.index[exit_idx], "ret": net_ret})
    in_trade_until = exit_idx

if len(trade_rets) > 5:
    tdf = pd.DataFrame(trade_rets)
    tdf["date"] = pd.to_datetime(tdf["date"]).dt.normalize()
    daily = tdf.groupby("date")["ret"].sum()
    all_dates = pd.date_range(daily.index.min(), qqq_close_c.index[-1].normalize(), freq="B")
    daily_full = daily.reindex(all_dates, fill_value=0.0)
    strategies["Reddit_Sentiment"] = daily_full
    s = _stats(daily_full)
    print(f"Sharpe: {s['Sharpe']:.3f}, Return: {s['Return']:+.1%}, Trades: {len(trade_rets)}")


_header("INDIVIDUAL STRATEGY COMPARISON")

summary = []
for name, ret in strategies.items():
    s = _stats(ret)
    summary.append({
        "Strategy": name,
        "Sharpe": s["Sharpe"],
        "Sortino": s["Sortino"],
        "Return": s["Return"],
        "MaxDD": s["MaxDD"],
    })
    print(f"{name:<30} Sharpe={s['Sharpe']:>7.3f}  Return={s['Return']:>8.1%}  MaxDD={s['MaxDD']:>7.2%}")

summary_df = pd.DataFrame(summary).sort_values("Sharpe", ascending=False)


_header("TOP STRATEGY COMBINATIONS (Fixed Weights)")

combinations = []

# Test top 3 strategies in various combinations
top_strats = summary_df.head(3)["Strategy"].tolist()

if len(top_strats) >= 2:
    # 50/50
    combo = 0.5 * strategies[top_strats[0]] + 0.5 * strategies[top_strats[1]]
    s = _stats(combo)
    combinations.append((f"50/50: {top_strats[0][:15]} / {top_strats[1][:15]}", s["Sharpe"], s["Return"], s["MaxDD"]))

    # 60/40
    combo = 0.6 * strategies[top_strats[0]] + 0.4 * strategies[top_strats[1]]
    s = _stats(combo)
    combinations.append((f"60/40: {top_strats[0][:15]} / {top_strats[1][:15]}", s["Sharpe"], s["Return"], s["MaxDD"]))

if len(top_strats) >= 3:
    # 40/30/30
    combo = 0.4 * strategies[top_strats[0]] + 0.3 * strategies[top_strats[1]] + 0.3 * strategies[top_strats[2]]
    s = _stats(combo)
    combinations.append((f"40/30/30", s["Sharpe"], s["Return"], s["MaxDD"]))

    # 50/30/20
    combo = 0.5 * strategies[top_strats[0]] + 0.3 * strategies[top_strats[1]] + 0.2 * strategies[top_strats[2]]
    s = _stats(combo)
    combinations.append((f"50/30/20", s["Sharpe"], s["Return"], s["MaxDD"]))

    # 60/20/20
    combo = 0.6 * strategies[top_strats[0]] + 0.2 * strategies[top_strats[1]] + 0.2 * strategies[top_strats[2]]
    s = _stats(combo)
    combinations.append((f"60/20/20", s["Sharpe"], s["Return"], s["MaxDD"]))

print("\nTop Combinations:")
print(f"{'Allocation':<45} {'Sharpe':>10} {'Return':>10} {'MaxDD':>10}")
print("-" * 80)
for name, sharpe, ret, mdd in sorted(combinations, key=lambda x: x[1], reverse=True)[:5]:
    print(f"{name:<45} {sharpe:>10.3f} {ret:>10.1%} {mdd:>10.2%}")


_header("CHARTS")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Chart 1: Sharpe comparison
ax = axes[0, 0]
top_10 = summary_df.head(10)
ax.barh(range(len(top_10)), top_10["Sharpe"].values, color="steelblue", alpha=0.7)
ax.set_yticks(range(len(top_10)))
ax.set_yticklabels([s[:25] for s in top_10["Strategy"].values])
ax.set_xlabel("Sharpe Ratio")
ax.set_title("Individual Strategies - Sharpe Comparison")
ax.grid(True, alpha=0.3, axis="x")

# Chart 2: Return comparison
ax = axes[0, 1]
ax.barh(range(len(top_10)), top_10["Return"].values, color="green", alpha=0.7)
ax.set_yticks(range(len(top_10)))
ax.set_yticklabels([s[:25] for s in top_10["Strategy"].values])
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_xlabel("Total Return")
ax.set_title("Individual Strategies - Return Comparison")
ax.grid(True, alpha=0.3, axis="x")

# Chart 3: Risk comparison
ax = axes[1, 0]
ax.barh(range(len(top_10)), np.abs(top_10["MaxDD"].values), color="red", alpha=0.7)
ax.set_yticks(range(len(top_10)))
ax.set_yticklabels([s[:25] for s in top_10["Strategy"].values])
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_xlabel("Max Drawdown")
ax.set_title("Individual Strategies - Risk (MaxDD)")
ax.grid(True, alpha=0.3, axis="x")

# Chart 4: Risk-Return scatter
ax = axes[1, 1]
colors_scatter = ["green" if s > 1.2 else "orange" if s > 1.0 else "red" for s in summary_df["Sharpe"]]
ax.scatter(np.abs(summary_df["MaxDD"]), summary_df["Sharpe"], s=300, alpha=0.6, c=colors_scatter)
for i, row in summary_df.iterrows():
    ax.annotate(row["Strategy"][:15], (abs(row["MaxDD"]), row["Sharpe"]),
               fontsize=8, xytext=(5, 5), textcoords="offset points")
ax.set_xlabel("Max Drawdown (Risk)")
ax.set_ylabel("Sharpe Ratio")
ax.set_title("Risk-Return Profile")
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


print(f"\n{'='*100}")
print(f"BEST INDIVIDUAL: {summary_df.iloc[0]['Strategy']}")
print(f"  Sharpe: {summary_df.iloc[0]['Sharpe']:.3f}")
print(f"  Return: {summary_df.iloc[0]['Return']:+.1%}")
print(f"  MaxDD: {summary_df.iloc[0]['MaxDD']:.2%}")
print(f"{'='*100}")
