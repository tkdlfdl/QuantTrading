"""
Hybrid Strategy: QQQ Bubble Timing + Momentum Stock Selection
==============================================================
Use QQQ bubble score as ENTRY TRIGGER, then buy TOP MOMENTUM stocks

Strategy Logic:
1. Monitor QQQ bubble score (MA=50h, Z=250h)
2. When score < -0.9 (extreme undervaluation), SIGNAL to BUY
3. At signal, select top 5-20 momentum stocks from 516-ticker universe
4. Hold for specified period (4, 8, 24 hours)
5. Exit after hold period

Tests on 2020-2026 data (includes bear market 2022)
Compares with:
- QQQ Bubble only (single asset)
- Momentum only (no timing signal)
- Combined (best of both)
"""
import sys, warnings, os
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_n = [0]
def _save(*a, **k):
    _n[0] += 1; p = f"results/bubble_momentum_combo_chart_{_n[0]}.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); print(f"  [chart saved: {p}]", flush=True)
plt.show = _save

sys.path.insert(0, ".")
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
    print(f"\n{'='*90}\n{t}\n{'='*90}", flush=True)


_header("HYBRID STRATEGY: QQQ BUBBLE SIGNAL + MOMENTUM STOCKS")

# Load data
print("Loading data...")
qqq_close = pd.read_parquet(Path("data/cache/qqq_hourly_close.parquet")).iloc[:, 0]
qqq_open = pd.read_parquet(Path("data/cache/qqq_hourly_open.parquet")).iloc[:, 0]
merged_close = pd.read_parquet(Path("data/cache/merged_hourly_close.parquet"))
merged_open = pd.read_parquet(Path("data/cache/merged_hourly_open.parquet"))

qqq_close.index = pd.to_datetime(qqq_close.index)
qqq_open.index = pd.to_datetime(qqq_open.index)
merged_close.index = pd.to_datetime(merged_close.index)
merged_open.index = pd.to_datetime(merged_open.index)

# Align to common period
common_idx = qqq_close.index.intersection(merged_close.index)
qqq_close_c = qqq_close.loc[common_idx]
qqq_open_c = qqq_open.loc[common_idx]
merged_close_c = merged_close.loc[common_idx]
merged_open_c = merged_open.loc[common_idx]

print(f"Common period: {qqq_close_c.index[0]} to {qqq_close_c.index[-1]} ({len(qqq_close_c)} bars)")

# Compute QQQ bubble score (best params from 7-year test)
MA_QQQ = 50
Z_QQQ = 250
THRESH_QQQ = 0.9

log_qqq = np.log(qqq_close_c.replace(0, np.nan).ffill())
fair_value = qqq_close_c.rolling(MA_QQQ).mean()
residual = log_qqq - np.log(fair_value)
z = (residual - residual.rolling(Z_QQQ).mean()) / residual.rolling(Z_QQQ).std()
bubble_score = np.tanh(z / 2)
bubble_signal = bubble_score.shift(1)  # No lookahead

print(f"QQQ Bubble Score computed: MA={MA_QQQ}h, Z={Z_QQQ}h, Thresh={THRESH_QQQ}")

# Compute momentum for each bar (20-hour lookback)
LOOKBACK_HOURS = 20
momentum = merged_close_c.pct_change(LOOKBACK_HOURS)

print(f"Momentum computed: {LOOKBACK_HOURS}-hour lookback")

# Strategy: When bubble signal, buy top momentum stocks
TOP_N_VALUES = [5, 10, 20]
HOLD_HOURS_VALUES = [4, 8, 24]

results = []

for top_n in TOP_N_VALUES:
    for hold_hours in HOLD_HOURS_VALUES:
        print(f"\nTesting: Top {top_n} momentum, Hold {hold_hours}h...", end="", flush=True)

        trade_rets = []
        in_trade_until = -1

        for i in range(Z_QQQ + 1, len(bubble_signal) - hold_hours):
            # Check bubble signal
            if i <= in_trade_until or pd.isna(bubble_signal.iloc[i]):
                continue

            # Only enter on bubble undervaluation
            if bubble_signal.iloc[i] >= -THRESH_QQQ:
                continue

            # Get momentum at this bar
            mom_row = momentum.iloc[i].dropna()
            if len(mom_row) < top_n:
                continue

            # Select top N momentum stocks
            top_stocks = mom_row.nlargest(top_n).index.tolist()

            # Entry and exit
            entry_idx = i + 1
            exit_idx = min(i + hold_hours, len(merged_close_c) - 1)

            if entry_idx >= len(merged_open_c):
                continue

            # Get entry prices (open of next bar)
            entry_prices = merged_open_c.iloc[entry_idx][top_stocks]
            exit_prices = merged_close_c.iloc[exit_idx][top_stocks]

            # Calculate returns
            valid = (entry_prices > 0) & entry_prices.notna() & exit_prices.notna()
            if not valid.any():
                continue

            raw_rets = (exit_prices[valid] / entry_prices[valid] - 1)
            net_ret = raw_rets.mean() - 0.001  # TC

            trade_rets.append({
                "entry_dt": merged_open_c.index[entry_idx],
                "bubble_score": bubble_signal.iloc[i],
                "net_ret": net_ret,
            })
            in_trade_until = exit_idx

        if len(trade_rets) < 10:
            results.append({
                "top_n": top_n, "hold_hours": hold_hours,
                "Sharpe": np.nan, "Return": np.nan, "MaxDD": np.nan,
                "Trades": len(trade_rets),
            })
            print(f" {len(trade_rets)} trades (too few)")
            continue

        # Daily returns
        tdf = pd.DataFrame(trade_rets)
        tdf["date"] = pd.to_datetime(tdf["entry_dt"]).dt.normalize()
        daily = tdf.groupby("date")["net_ret"].sum()

        data_end = merged_close_c.index[-1].normalize()
        all_dates = pd.date_range(daily.index.min(), data_end, freq="B")
        daily_full = daily.reindex(all_dates, fill_value=0.0)

        s = _stats(daily_full)
        results.append({
            "top_n": top_n, "hold_hours": hold_hours,
            "Sharpe": s["Sharpe"], "Sortino": s["Sortino"],
            "Return": s["Return"], "MaxDD": s["MaxDD"],
            "Trades": len(trade_rets),
        })
        print(f" Sharpe={s['Sharpe']:.3f}")

results_df = pd.DataFrame(results).sort_values("Sharpe", ascending=False)


_header("RESULTS: BUBBLE SIGNAL + MOMENTUM STOCKS")

print("\n" + results_df.to_string(index=False))

best = results_df.iloc[0]

if pd.notna(best["Sharpe"]):
    # Recompute best strategy for detailed analysis
    top_n_best = int(best["top_n"])
    hold_hours_best = int(best["hold_hours"])

    trade_rets = []
    in_trade_until = -1

    for i in range(Z_QQQ + 1, len(bubble_signal) - hold_hours_best):
        if i <= in_trade_until or pd.isna(bubble_signal.iloc[i]):
            continue
        if bubble_signal.iloc[i] >= -THRESH_QQQ:
            continue

        mom_row = momentum.iloc[i].dropna()
        if len(mom_row) < top_n_best:
            continue

        top_stocks = mom_row.nlargest(top_n_best).index.tolist()
        entry_idx = i + 1
        exit_idx = min(i + hold_hours_best, len(merged_close_c) - 1)

        if entry_idx >= len(merged_open_c):
            continue

        entry_prices = merged_open_c.iloc[entry_idx][top_stocks]
        exit_prices = merged_close_c.iloc[exit_idx][top_stocks]
        valid = (entry_prices > 0) & entry_prices.notna() & exit_prices.notna()

        if not valid.any():
            continue

        raw_rets = (exit_prices[valid] / entry_prices[valid] - 1)
        net_ret = raw_rets.mean() - 0.001

        trade_rets.append({
            "entry_dt": merged_open_c.index[entry_idx],
            "bubble_score": bubble_signal.iloc[i],
            "net_ret": net_ret,
        })
        in_trade_until = exit_idx

    tdf = pd.DataFrame(trade_rets)
    tdf["date"] = pd.to_datetime(tdf["entry_dt"]).dt.normalize()
    daily = tdf.groupby("date")["net_ret"].sum()
    data_end = merged_close_c.index[-1].normalize()
    all_dates = pd.date_range(daily.index.min(), data_end, freq="B")
    best_ret = daily.reindex(all_dates, fill_value=0.0)

    s_best = _stats(best_ret)

    _header("BEST STRATEGY: BUBBLE SIGNAL + MOMENTUM")

    print(f"""
Strategy:       QQQ Bubble (MA={MA_QQQ}h, Z={Z_QQQ}h, T={THRESH_QQQ}) + Top {top_n_best} Momentum Stocks
Hold Period:    {hold_hours_best} hours
Lookback:       {LOOKBACK_HOURS} hours
Total Trades:   {len(trade_rets)}

Performance:
  Sharpe:       {s_best['Sharpe']:.4f}
  Sortino:      {s_best['Sortino']:.4f}
  Return:       {s_best['Return']:+.2%}
  Max DD:       {s_best['MaxDD']:.2%}
  Annual Return: {s_best['Return'] / (len(best_ret)/252):+.2%}

Entry Logic:
  1. Monitor QQQ bubble score
  2. When score < -0.9 (extreme undervaluation), trigger signal
  3. Select top {top_n_best} momentum stocks (highest {LOOKBACK_HOURS}-hour return)
  4. Buy equal-weight
  5. Hold {hold_hours_best} hours
  6. Exit and repeat

This combines:
  ✓ QQQ bubble for TIMING (when to enter)
  ✓ Momentum for STOCK SELECTION (which stocks to buy)
""")

    # Yearly breakdown
    yearly_rows = []
    for yr in sorted(best_ret.index.year.unique()):
        yr_ret = best_ret[best_ret.index.year == yr]
        if len(yr_ret) < 5:
            continue
        s = _stats(yr_ret)
        yearly_rows.append({
            "Year": yr,
            "Return": s["Return"],
            "Sharpe": s["Sharpe"],
            "MaxDD": s["MaxDD"],
            "Days": len(yr_ret),
        })

    yearly_df = pd.DataFrame(yearly_rows)

    _header("YEARLY BREAKDOWN")

    print("\n" + yearly_df.to_string(index=False))

    # Charts
    _header("CHARTS")

    fig, axes = plt.subplots(3, 1, figsize=(18, 14), gridspec_kw={"height_ratios": [3, 1.5, 1.5]})

    ax = axes[0]
    w = (1+best_ret).cumprod(); w = w/w.iloc[0]
    ax.plot(w.index, w.values, color="steelblue", linewidth=2)
    ax.axvline(pd.Timestamp("2022-01-01"), color="red", linestyle="--", alpha=0.5, label="Bear 2022")
    ax.set_title(f"Bubble Signal + Momentum Stocks | Top {top_n_best}, Hold {hold_hours_best}h\n"
                 f"Sharpe={s_best['Sharpe']:.3f}, Return={s_best['Return']:+.1%}, MaxDD={s_best['MaxDD']:.1%}, Trades={len(trade_rets)}",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Wealth"); ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.1f}x"))
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    w = (1+best_ret).cumprod(); w = w/w.iloc[0]
    dd = w/w.cummax()-1
    ax.fill_between(dd.index, dd.values, 0, alpha=0.5, color="crimson")
    ax.set_ylabel("Drawdown"); ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if len(yearly_df) > 0:
        ax.bar(yearly_df['Year'], yearly_df['Return']*100, alpha=0.7, color="coral")
        ax.set_xlabel("Year"); ax.set_ylabel("Return (%)")
        ax.set_xticks(yearly_df['Year'])
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()

    # Save
    xl = "results/bubble_momentum_combo_backtest.xlsx"
    try:
        with pd.ExcelWriter(xl, engine="openpyxl") as writer:
            results_df.to_excel(writer, sheet_name="Grid_Results", index=False)
            yearly_df.to_excel(writer, sheet_name="Yearly", index=False)
            best_ret.to_excel(writer, sheet_name="Daily_Returns")

            summary = pd.DataFrame({
                "Parameter": ["Top N", "Hold Hours", "Sharpe", "Return", "MaxDD", "Trades"],
                "Value": [top_n_best, hold_hours_best, f"{s_best['Sharpe']:.4f}",
                         f"{s_best['Return']:+.2%}", f"{s_best['MaxDD']:.2%}", len(trade_rets)]
            })
            summary.to_excel(writer, sheet_name="Summary", index=False)

        print(f"\nSaved: {xl}")
    except Exception as e:
        print(f"Save error: {e}")


_header("COMPARISON TABLE")

print("""
Strategy                          Sharpe    Return    MaxDD    Trades
────────────────────────────────────────────────────────────────────────
QQQ Bubble Only                   1.235    +19.94%   -1.57%      31
Momentum Only (best grid)         ~2.5     +600%     -20%        1500+
Bubble Signal + Momentum (COMBO)   """ + f"{s_best['Sharpe']:.3f}    {s_best['Return']:+.2%}   {s_best['MaxDD']:.2%}     {len(trade_rets)}")

print("""
KEY INSIGHT:
  Using QQQ bubble as ENTRY TIMING + momentum for STOCK SELECTION
  May provide:
  - Better risk-adjusted returns (selective entry)
  - Diversification (multiple stocks not just QQQ)
  - Lower drawdowns (timing-filtered entries)
""")

EOF
