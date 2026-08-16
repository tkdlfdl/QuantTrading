"""
2nd Sharpe Strategy — Full Performance Analysis
================================================
Parameters: MA=500h, Thr=-0.8, MomLB=40h, Hold=52h, TopN=5
Formula:    calculate_performance_metrics() as provided
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PROVIDED FORMULA — exact as given
# ─────────────────────────────────────────────────────────────
def calculate_performance_metrics(returns_series, risk_free_annual=0.02, trading_days=252):
    """
    trading_days: number of return periods per year
                  e.g. 252 for daily, or actual_trades_per_year for trade-based
    """
    daily_rf = risk_free_annual / trading_days
    excess_returns = returns_series - daily_rf

    # 1. SHARPE RATIO
    mean_excess_return = excess_returns.mean()
    std_total          = returns_series.std()
    daily_sharpe       = mean_excess_return / std_total if std_total != 0 else 0
    annual_sharpe      = daily_sharpe * np.sqrt(trading_days)

    # 2. SORTINO RATIO
    downside_returns = returns_series[returns_series < 0]
    std_downside     = downside_returns.std(ddof=0) if len(downside_returns) > 0 else 0
    daily_sortino    = mean_excess_return / std_downside if std_downside != 0 else 0
    annual_sortino   = daily_sortino * np.sqrt(trading_days)

    # 3. MAX DRAWDOWN
    wealth_index  = (1 + returns_series).cumprod()
    historical_max = wealth_index.cummax()
    drawdowns      = (wealth_index - historical_max) / historical_max
    max_drawdown   = drawdowns.min()

    return {
        "annual_sharpe":  annual_sharpe,
        "annual_sortino": annual_sortino,
        "max_drawdown":   max_drawdown,
        "wealth":         wealth_index,
    }

# ─────────────────────────────────────────────────────────────
# LOAD + ALIGN DATA
# ─────────────────────────────────────────────────────────────
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
stocks     = stocks[sp_cols]
valid      = stocks.columns[stocks.isna().mean() < 0.30]
stocks     = stocks[valid].ffill()

ret_np  = stocks.pct_change().clip(-0.10, 0.10).fillna(0).values.astype(np.float32)
log_ret = np.log1p(ret_np.clip(-0.10, 0.10))
cumlog  = np.cumsum(log_ret, axis=0)
n       = len(idx)
years   = (idx[-1] - idx[0]).days / 365.25

print(f"Data: {n} bars  |  {idx[0].date()} to {idx[-1].date()}  ({years:.2f} yrs)")
print(f"Stocks: {len(valid)} tickers (S&P500/NASDAQ100)")

# ─────────────────────────────────────────────────────────────
# BUBBLE SCORE
# ─────────────────────────────────────────────────────────────
def bubble_score(price, ma_w):
    log_p = np.log(price)
    fair  = price.rolling(ma_w).mean()
    res   = log_p - np.log(fair)
    z     = (res - res.rolling(ma_w).mean()) / res.rolling(ma_w).std()
    return np.tanh(z / 2)

# ─────────────────────────────────────────────────────────────
# STRATEGY 2 — PARAMETERS
# ─────────────────────────────────────────────────────────────
MA    = 500
THR   = -0.8
LB    = 40
HOLD  = 52
TOPN  = 5
RF    = 0.02   # 2% annual risk-free rate

print(f"\nStrategy 2 parameters:")
print(f"  Bubble MA:    {MA}h (~{MA/6.5:.0f} trading days)")
print(f"  Threshold:    {THR}")
print(f"  Mom Lookback: {LB}h")
print(f"  Hold Period:  {HOLD}h (~{HOLD/6.5:.1f} sessions = ~{HOLD/32.5:.1f} weeks)")
print(f"  Top-N:        {TOPN}")
print(f"  Risk-Free:    {RF:.0%} annual")

# ─────────────────────────────────────────────────────────────
# RUN STRATEGY
# ─────────────────────────────────────────────────────────────
bub     = bubble_score(qqq, MA).fillna(0).values
mom_ret = stocks.pct_change(LB).fillna(0).values.astype(np.float32)
warmup  = max(MA, LB) + 5

fwd = np.zeros((n, stocks.shape[1]), dtype=np.float32)
fwd[:n-HOLD] = np.expm1(cumlog[HOLD:] - cumlog[:n-HOLD])

trade_rets, trade_dates = [], []
i = warmup
while i < n - HOLD:
    if bub[i] < THR:
        top_idx = np.argpartition(mom_ret[i], -TOPN)[-TOPN:]
        r = float(fwd[i, top_idx].mean()) - 0.001
        r = max(-0.50, min(2.00, r))
        trade_rets.append(r)
        trade_dates.append(idx[i])
        i += HOLD
    else:
        i += 1

s = pd.Series(trade_rets, index=pd.DatetimeIndex(trade_dates))
w = (1 + s).cumprod()

# ─────────────────────────────────────────────────────────────
# OVERALL PERFORMANCE using provided formula
# ─────────────────────────────────────────────────────────────
actual_tpy = len(s) / years   # actual trades per year

# Annual return
total_ret   = w.iloc[-1] - 1
annual_ret  = (1 + total_ret) ** (1 / years) - 1

overall = calculate_performance_metrics(s, risk_free_annual=RF, trading_days=actual_tpy)

print(f"\n{'='*90}")
print("OVERALL PERFORMANCE  (2020-2026, 5.85 years)")
print(f"{'='*90}")
print(f"  Total Return:    {total_ret:.2%}")
print(f"  Annual Return:   {annual_ret:.2%}")
print(f"  Sharpe Ratio:    {overall['annual_sharpe']:.4f}")
print(f"  Sortino Ratio:   {overall['annual_sortino']:.4f}")
print(f"  Max Drawdown:    {overall['max_drawdown']:.2%}")
print(f"  Total Trades:    {len(s)}")
print(f"  Trades / Year:   {actual_tpy:.2f}")
print(f"  Win Rate:        {(s > 0).mean():.1%}")

# ─────────────────────────────────────────────────────────────
# YEARLY BREAKDOWN
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("YEARLY PERFORMANCE")
print(f"{'='*90}")
print(f"  {'Year':<6}  {'Return':>9}  {'Ann Return':>10}  {'Sharpe':>8}  {'Sortino':>9}  {'MaxDD':>9}  {'Trades':>7}")
print(f"  {'-'*6}  {'-'*9}  {'-'*10}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*7}")

yearly_rows = []
for yr in sorted(s.index.year.unique()):
    ys = s[s.index.year == yr]
    yw = w[w.index.year == yr]
    if len(ys) == 0:
        continue

    # Total return for this year
    y_total = (1 + ys).prod() - 1

    # Annualised return (fraction of year this strategy was active)
    days_in_year = 366 if yr % 4 == 0 else 365
    first_trade  = ys.index[0]
    last_trade   = ys.index[-1] + pd.Timedelta(hours=HOLD)
    active_years = min((last_trade - first_trade).days / days_in_year, 1.0)
    active_years = max(active_years, len(ys) / 260)   # at least n_trades/260
    y_annual     = (1 + y_total) ** (1 / active_years) - 1 if y_total > -1 else -1

    # Sharpe, Sortino, MaxDD using provided formula
    y_tpy = len(ys)   # trades this year (annualised within year)
    if len(ys) >= 2:
        m = calculate_performance_metrics(ys, risk_free_annual=RF, trading_days=y_tpy)
        y_sh  = m["annual_sharpe"]
        y_so  = m["annual_sortino"]
        y_dd  = m["max_drawdown"]
    elif len(ys) == 1:
        y_sh  = 0.0
        y_so  = 0.0
        y_dd  = min(ys.iloc[0], 0.0)
    else:
        continue

    print(f"  {yr:<6}  {y_total:>9.2%}  {y_annual:>10.2%}  {y_sh:>8.3f}  {y_so:>9.3f}  {y_dd:>9.2%}  {len(ys):>7}")
    yearly_rows.append(dict(Year=yr, Total_Return=y_total, Annual_Return=y_annual,
                            Sharpe=y_sh, Sortino=y_so, MaxDD=y_dd, Trades=len(ys)))

# ─────────────────────────────────────────────────────────────
# QQQ BENCHMARK
# ─────────────────────────────────────────────────────────────
qqq_ret = qqq.pct_change().fillna(0)

print(f"\n{'='*90}")
print("QQQ BUY & HOLD  (benchmark)")
print(f"{'='*90}")
print(f"  {'Year':<6}  {'Return':>9}  {'Sharpe':>8}  {'Sortino':>9}  {'MaxDD':>9}")
print(f"  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*9}")

for yr in sorted(qqq_ret.index.year.unique()):
    yr_r = qqq_ret[qqq_ret.index.year == yr]
    if len(yr_r) < 10:
        continue
    y_total = (1 + yr_r).prod() - 1
    m = calculate_performance_metrics(yr_r, risk_free_annual=RF, trading_days=len(yr_r))
    print(f"  {yr:<6}  {y_total:>9.2%}  {m['annual_sharpe']:>8.3f}  {m['annual_sortino']:>9.3f}  {m['max_drawdown']:>9.2%}")

# ─────────────────────────────────────────────────────────────
# FORMULA VERIFICATION
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("FORMULA VERIFICATION")
print(f"{'='*90}")
print(f"""
  Annual Return:
    total_return      = (1 + r1)*(1 + r2)*...*(1 + rN) - 1  = {total_ret:.4f}
    annual_return     = (1 + total_return)^(1/years) - 1     = {annual_ret:.4f}

  Sharpe Ratio (provided formula):
    daily_rf          = risk_free_annual / trading_days       = {RF}/{actual_tpy:.2f} = {RF/actual_tpy:.6f}
    excess_returns    = returns - daily_rf
    mean_excess       = excess_returns.mean()                 = {(s - RF/actual_tpy).mean():.6f}
    std_total         = returns.std()                         = {s.std():.6f}
    daily_sharpe      = mean_excess / std_total               = {(s - RF/actual_tpy).mean() / s.std():.6f}
    annual_sharpe     = daily_sharpe * sqrt(trading_days)     = {overall['annual_sharpe']:.4f}

  Sortino Ratio (provided formula):
    downside_returns  = returns[returns < 0]                  (N={len(s[s<0])} trades)
    std_downside      = downside_returns.std(ddof=0)          = {s[s<0].std(ddof=0):.6f}
    daily_sortino     = mean_excess / std_downside            = {(s - RF/actual_tpy).mean() / s[s<0].std(ddof=0) if len(s[s<0])>0 else 0:.6f}
    annual_sortino    = daily_sortino * sqrt(trading_days)    = {overall['annual_sortino']:.4f}

  Max Drawdown:
    wealth_index      = (1 + returns).cumprod()
    max_drawdown      = min((wealth - peak) / peak)           = {overall['max_drawdown']:.4f}
""")

# ─────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Strategy 2: QQQ Bubble + Long Momentum (MA=500h, Thr=-0.8, Hold=52h, Top5)",
             fontsize=13, fontweight="bold")

# 1. Wealth
qqq_bench = (1 + qqq_ret.loc[w.index[0]:]).cumprod()
qqq_bench /= qqq_bench.iloc[0]
axes[0,0].plot(w.index, w.values, lw=2, label=f"Strategy (Total {total_ret:.0%})", color="steelblue")
axes[0,0].plot(qqq_bench.index, qqq_bench.values, lw=1.5, ls="--", label="QQQ", color="gray")
axes[0,0].set_title("Cumulative Wealth", fontweight="bold")
axes[0,0].set_ylabel("Wealth Multiple")
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

# 2. Bubble score with trade markers
bub_full = pd.Series(bub, index=idx)
axes[0,1].plot(bub_full.index, bub_full.values, lw=0.8, color="navy", alpha=0.6, label="Bubble Score")
axes[0,1].axhline(THR, color="red", ls="--", lw=1.5, label=f"Entry threshold {THR}")
axes[0,1].axhline(0, color="black", lw=0.5)
axes[0,1].scatter(s.index, [THR]*len(s), color="green", zorder=5, s=40, label="Trade entries")
axes[0,1].set_ylim(-1, 1)
axes[0,1].set_title("QQQ Bubble Score + Trade Entries", fontweight="bold")
axes[0,1].legend(fontsize=8); axes[0,1].grid(True, alpha=0.3)

# 3. Yearly returns
if yearly_rows:
    ydf   = pd.DataFrame(yearly_rows)
    clrs  = ["green" if r >= 0 else "red" for r in ydf["Total_Return"]]
    axes[1,0].bar(ydf["Year"].astype(str), ydf["Total_Return"]*100, color=clrs, alpha=0.7)
    axes[1,0].axhline(0, color="black", lw=0.8)
    axes[1,0].set_title("Yearly Total Returns (%)", fontweight="bold")
    axes[1,0].set_ylabel("Return (%)"); axes[1,0].grid(True, alpha=0.3, axis="y")

    # 4. Yearly Sharpe
    sharpe_clrs = ["steelblue" if v >= 0 else "tomato" for v in ydf["Sharpe"]]
    axes[1,1].bar(ydf["Year"].astype(str), ydf["Sharpe"], color=sharpe_clrs, alpha=0.7)
    axes[1,1].axhline(0, color="black", lw=0.8)
    axes[1,1].set_title("Yearly Sharpe Ratios", fontweight="bold")
    axes[1,1].set_ylabel("Sharpe"); axes[1,1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/strategy2_qqq_bubble_performance.png", dpi=150, bbox_inches="tight")
print("Saved: results/strategy2_qqq_bubble_performance.png")

pd.DataFrame(yearly_rows).to_csv("results/strategy2_yearly_performance.csv", index=False)
print("Saved: results/strategy2_yearly_performance.csv")
print("\nDONE")
