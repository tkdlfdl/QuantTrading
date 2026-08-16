"""
Strategy 2 — Corrected MaxDD using HOURLY wealth curve
=======================================================
Bug: MaxDD was computed on sparse net-trade returns only.
     A winning trade (net +5%) can still have -3% intra-trade drawdown.
Fix: Build full hourly wealth series (holding during trade, cash otherwise).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings("ignore")

# ── provided formula ─────────────────────────────────────────
def calculate_performance_metrics(returns_series, risk_free_annual=0.02, trading_days=252):
    daily_rf         = risk_free_annual / trading_days
    excess_returns   = returns_series - daily_rf
    mean_excess      = excess_returns.mean()
    std_total        = returns_series.std()
    daily_sharpe     = mean_excess / std_total if std_total != 0 else 0
    annual_sharpe    = daily_sharpe * np.sqrt(trading_days)

    downside_returns = returns_series[returns_series < 0]
    std_downside     = downside_returns.std(ddof=0) if len(downside_returns) > 0 else 0
    daily_sortino    = mean_excess / std_downside if std_downside != 0 else 0
    annual_sortino   = daily_sortino * np.sqrt(trading_days)

    wealth_index   = (1 + returns_series).cumprod()
    historical_max = wealth_index.cummax()
    drawdowns      = (wealth_index - historical_max) / historical_max
    max_drawdown   = drawdowns.min()

    return dict(sharpe=annual_sharpe, sortino=annual_sortino, max_drawdown=max_drawdown,
                wealth=wealth_index)

# ── data ─────────────────────────────────────────────────────
print("Loading data...")
qqq_raw    = pd.read_parquet("data/cache/qqq_hourly_close.parquet")["QQQ"]
stocks_raw = pd.read_parquet("data/cache/merged_hourly_close.parquet")

qqq_h = qqq_raw.copy();    qqq_h.index = qqq_h.index.floor("h")
stk_h = stocks_raw.copy(); stk_h.index = stk_h.index.floor("h")
qqq_h = qqq_h[~qqq_h.index.duplicated(keep="last")]
stk_h = stk_h[~stk_h.index.duplicated(keep="last")]
idx   = qqq_h.index.intersection(stk_h.index)
qqq   = qqq_h.loc[idx]
stocks = stk_h.loc[idx]

daily_cols = set(pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet").columns)
sp_cols    = [c for c in stocks.columns if c in daily_cols]
valid      = stocks[sp_cols].columns[stocks[sp_cols].isna().mean() < 0.30]
stocks     = stocks[valid].ffill()

ret_hourly = stocks.pct_change().clip(-0.10, 0.10).fillna(0)  # hourly returns DataFrame
ret_np     = ret_hourly.values.astype(np.float32)

log_ret = np.log1p(ret_np.clip(-0.10, 0.10))
cumlog  = np.cumsum(log_ret, axis=0)
n       = len(idx)
years   = (idx[-1] - idx[0]).days / 365.25

# ── strategy params ──────────────────────────────────────────
MA = 500; THR = -0.8; LB = 40; HOLD = 52; TOPN = 5; RF = 0.02

def bubble_score(price, ma_w):
    log_p = np.log(price)
    fair  = price.rolling(ma_w).mean()
    res   = log_p - np.log(fair)
    z     = (res - res.rolling(ma_w).mean()) / res.rolling(ma_w).std()
    return np.tanh(z / 2)

bub     = bubble_score(qqq, MA).fillna(0).values
mom_ret = stocks.pct_change(LB).fillna(0).values.astype(np.float32)
warmup  = max(MA, LB) + 5

# Pre-compute compound fwd return (for trade-level P&L)
fwd = np.zeros((n, stocks.shape[1]), dtype=np.float32)
fwd[:n-HOLD] = np.expm1(cumlog[HOLD:] - cumlog[:n-HOLD])

# ── RUN STRATEGY ─────────────────────────────────────────────
# Build TWO wealth series:
#   1. hourly_ret_series: hourly return at each bar (0 in cash, stock avg while holding)
#   2. trade_series:      one return per trade (net compound return over hold period)

hourly_rets = np.zeros(n, dtype=np.float64)   # 0 = cash by default
trade_rets, trade_dates = [], []

i = warmup
while i < n - HOLD:
    if bub[i] < THR:
        # Select top-N stocks by momentum at entry
        top_idx = np.argpartition(mom_ret[i], -TOPN)[-TOPN:]

        # Record hourly returns while in this trade
        for j in range(i + 1, i + 1 + HOLD):
            if j < n:
                hourly_rets[j] = ret_np[j, top_idx].mean()

        # Net compound return for this trade
        net_ret = float(fwd[i, top_idx].mean()) - 0.001
        net_ret = max(-0.50, min(2.00, net_ret))
        trade_rets.append(net_ret)
        trade_dates.append(idx[i])
        i += HOLD
    else:
        i += 1

# ── HOURLY WEALTH SERIES ─────────────────────────────────────
hourly_series = pd.Series(hourly_rets, index=idx)
hourly_wealth = (1 + hourly_series).cumprod()

# ── TRADE SERIES ─────────────────────────────────────────────
trade_series  = pd.Series(trade_rets, index=pd.DatetimeIndex(trade_dates))
trade_wealth  = (1 + trade_series).cumprod()

# ── OVERALL METRICS ──────────────────────────────────────────
total_ret  = trade_wealth.iloc[-1] - 1
annual_ret = (1 + total_ret) ** (1 / years) - 1
actual_tpy = len(trade_series) / years

# Sharpe/Sortino: use trade-level returns + trade frequency
m_overall = calculate_performance_metrics(trade_series, RF, actual_tpy)

# MaxDD from HOURLY wealth (correct)
hourly_dd   = (hourly_wealth / hourly_wealth.cummax() - 1).min()

print(f"\n{'='*90}")
print("OVERALL PERFORMANCE — CORRECTED MaxDD (from hourly wealth)")
print(f"{'='*90}")
print(f"  Total Return:    {total_ret:.2%}")
print(f"  Annual Return:   {annual_ret:.2%}")
print(f"  Sharpe Ratio:    {m_overall['sharpe']:.4f}")
print(f"  Sortino Ratio:   {m_overall['sortino']:.4f}")
print(f"  MaxDD (WRONG):   {m_overall['max_drawdown']:.2%}   <- from net trade returns only")
print(f"  MaxDD (CORRECT): {hourly_dd:.2%}   <- from hourly wealth (includes intra-trade moves)")
print(f"  Total Trades:    {len(trade_series)}")
print(f"  Win Rate:        {(trade_series > 0).mean():.1%}")

# ── YEARLY BREAKDOWN ─────────────────────────────────────────
print(f"\n{'='*90}")
print("YEARLY PERFORMANCE — with CORRECTED MaxDD")
print(f"{'='*90}")
print(f"  {'Year':<6}  {'Return':>9}  {'Ann Ret':>9}  {'Sharpe':>8}  {'Sortino':>9}  {'MaxDD(net)':>11}  {'MaxDD(hrly)':>12}  {'Trades':>7}")
print(f"  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*11}  {'-'*12}  {'-'*7}")

yearly_rows = []
for yr in sorted(trade_series.index.year.unique()):
    # Trade-level for the year
    yt = trade_series[trade_series.index.year == yr]
    yw = trade_wealth[trade_wealth.index.year == yr]
    # Hourly-level for the year
    yh = hourly_series[hourly_series.index.year == yr]
    yhw = hourly_wealth[hourly_wealth.index.year == yr]

    if len(yt) == 0:
        continue

    y_total = (1 + yt).prod() - 1

    # Annualised return
    days_in_year = 366 if yr % 4 == 0 else 365
    active_years = max((yt.index[-1] - yt.index[0]).days / days_in_year,
                       len(yt) / 260)
    y_annual = (1 + y_total) ** (1 / max(active_years, 0.01)) - 1 if y_total > -1 else -1

    # Sharpe/Sortino from trade returns
    y_tpy = len(yt)
    if len(yt) >= 2:
        m = calculate_performance_metrics(yt, RF, y_tpy)
        y_sh, y_so = m['sharpe'], m['sortino']
    else:
        y_sh, y_so = 0.0, 0.0

    # MaxDD from sparse trade wealth (old wrong way)
    y_dd_sparse = (yw / yw.cummax() - 1).min() if len(yw) > 1 else min(yt.iloc[0], 0)

    # MaxDD from hourly wealth (correct)
    y_dd_hourly = (yhw / yhw.cummax() - 1).min() if len(yhw) > 0 else 0.0

    print(f"  {yr:<6}  {y_total:>9.2%}  {y_annual:>9.2%}  {y_sh:>8.3f}  {y_so:>9.3f}  {y_dd_sparse:>11.2%}  {y_dd_hourly:>12.2%}  {len(yt):>7}")
    yearly_rows.append(dict(Year=yr, Total_Return=y_total, Annual_Return=y_annual,
                            Sharpe=y_sh, Sortino=y_so,
                            MaxDD_sparse=y_dd_sparse, MaxDD_hourly=y_dd_hourly,
                            Trades=len(yt)))

print(f"\n  Note: MaxDD(net) was 0.00% in years where all trades were individually profitable")
print(f"        MaxDD(hrly) captures the actual intra-trade hourly drawdowns")

# ── COMPARISON ───────────────────────────────────────────────
print(f"\n{'='*90}")
print("MaxDD COMPARISON — sparse vs hourly")
print(f"{'='*90}")
ydf = pd.DataFrame(yearly_rows)
print(f"\n  {'Year':<6}  {'MaxDD(net,WRONG)':>17}  {'MaxDD(hourly,CORRECT)':>22}  {'Difference':>12}")
print(f"  {'-'*6}  {'-'*17}  {'-'*22}  {'-'*12}")
for _, row in ydf.iterrows():
    diff = row['MaxDD_hourly'] - row['MaxDD_sparse']
    print(f"  {int(row['Year']):<6}  {row['MaxDD_sparse']:>17.2%}  {row['MaxDD_hourly']:>22.2%}  {diff:>12.2%}")

# ── PLOTS ────────────────────────────────────────────────────
qqq_ret   = qqq.pct_change().fillna(0)
qqq_wealth = (1 + qqq_ret.loc[hourly_wealth.index[0]:]).cumprod()
qqq_wealth /= qqq_wealth.iloc[0]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Strategy 2: Corrected MaxDD (Hourly Wealth Curve)", fontsize=13, fontweight="bold")

# 1. Wealth
axes[0,0].plot(hourly_wealth.index, hourly_wealth.values, lw=1.5, label="Strategy (hourly)", color="steelblue")
axes[0,0].plot(qqq_wealth.index, qqq_wealth.values, lw=1.2, ls="--", label="QQQ", color="gray")
axes[0,0].set_title("Hourly Wealth Curve", fontweight="bold")
axes[0,0].set_ylabel("Wealth Multiple")
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

# 2. Drawdown from hourly wealth
dd_series = hourly_wealth / hourly_wealth.cummax() - 1
axes[0,1].fill_between(dd_series.index, dd_series.values, 0, alpha=0.6, color="red")
axes[0,1].set_title("Drawdown from Hourly Wealth (correct)", fontweight="bold")
axes[0,1].set_ylabel("Drawdown")
axes[0,1].grid(True, alpha=0.3)

# 3. MaxDD comparison
if len(ydf) > 0:
    x = np.arange(len(ydf))
    w_bar = 0.35
    axes[1,0].bar(x - w_bar/2, ydf['MaxDD_sparse']*100, w_bar, label='MaxDD (sparse, wrong)', color='orange', alpha=0.7)
    axes[1,0].bar(x + w_bar/2, ydf['MaxDD_hourly']*100, w_bar, label='MaxDD (hourly, correct)', color='red', alpha=0.7)
    axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(ydf['Year'].astype(int).astype(str))
    axes[1,0].axhline(0, color='black', lw=0.8)
    axes[1,0].set_title("Yearly MaxDD: Sparse vs Hourly", fontweight="bold")
    axes[1,0].set_ylabel("MaxDD (%)"); axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3, axis='y')

# 4. Yearly returns
if len(ydf) > 0:
    clrs = ["green" if r >= 0 else "red" for r in ydf["Total_Return"]]
    axes[1,1].bar(ydf["Year"].astype(int).astype(str), ydf["Total_Return"]*100, color=clrs, alpha=0.7)
    axes[1,1].axhline(0, color='black', lw=0.8)
    axes[1,1].set_title("Yearly Total Returns (%)", fontweight="bold")
    axes[1,1].set_ylabel("Return (%)"); axes[1,1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/strategy2_corrected_maxdd.png", dpi=150, bbox_inches="tight")
print("\nSaved: results/strategy2_corrected_maxdd.png")

pd.DataFrame(yearly_rows).to_csv("results/strategy2_yearly_corrected.csv", index=False)
print("Saved: results/strategy2_yearly_corrected.csv")
print("\nDONE")
