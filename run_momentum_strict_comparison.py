"""
Strict Comparison: Hourly Momentum (Higher TC) vs Daily Momentum (140d/40d)
===========================================================================
Testing realistic transaction costs and parameter sensitivity.

Strategy A -- Hourly Momentum (Long-Only)
  Universe  : NASDAQ100 + S&P500 (516 tickers)
  Grids     : lookback=[60,120,200,300]h, hold=[1,2,5,10,20]h, top_n=[5,10,20,30]
  TC        : 0.5% per trade (0.005 one-way, 1% round-trip) -- STRICT
  Best-of   : Independent long grid search

Strategy B -- Daily Momentum (Long-Only)
  Universe  : NASDAQ100 + S&P500 (516 tickers)
  Signal    : 140-day cumulative return
  Hold      : 40 trading days
  Grids     : top_n=[5,10,20,30]
  TC        : 0.5% per position changing (equal to hourly fairness)
"""
import sys, warnings, os
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_n = [0]
def _save(*a, **k):
    _n[0] += 1; p = f"results/strict_compare_chart_{_n[0]}.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); print(f"  [chart saved: {p}]", flush=True)
plt.show = _save

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path

from data.db.schema import init
from data.universe import get_universe

os.makedirs("results", exist_ok=True)

TRADING_DAYS = 252
TRADING_HOURS = TRADING_DAYS * 6.5
TC_STRICT = 0.005  # 0.5% one-way = 1% round-trip
BORROW_HOURLY_STRICT = 0.08 / TRADING_HOURS


def _sharpe(r, td=TRADING_DAYS):
    s = r.std(); return float(np.sqrt(td)*r.mean()/s) if s > 0 else np.nan

def _sortino(r, td=TRADING_DAYS):
    ds = r[r<0].std(); return float(np.sqrt(td)*r.mean()/ds) if ds > 0 else np.nan

def _mdd(r):
    w = (1+r).cumprod(); w = w/w.iloc[0]
    return float((w/w.cummax()-1).min())

def _stats(r):
    if len(r) < 5: return {"Sharpe": np.nan, "Sortino": np.nan,
                           "Total Return": np.nan, "Max DD": np.nan}
    w = (1+r).cumprod(); w = w/w.iloc[0]
    return {"Sharpe": _sharpe(r), "Sortino": _sortino(r),
            "Total Return": float(w.iloc[-1]-1), "Max DD": _mdd(r)}

def _header(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}", flush=True)


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 -- Hourly Momentum Grid (Strict TC = 0.5%)
# ══════════════════════════════════════════════════════════════════════════

_header("Strategy A -- Hourly Momentum (Grid, TC=0.5% strict)")

init()

# Load hourly data
print("Loading hourly cache...", flush=True)
ho = pd.read_parquet(Path("data/cache/hourly_open.parquet"))
hc = pd.read_parquet(Path("data/cache/hourly_close.parquet"))
ho.index = pd.to_datetime(ho.index)
hc.index = pd.to_datetime(hc.index)

# Align and prep
common = [c for c in ho.columns if c in hc.columns]
ho = ho[common].copy()
hc = hc[common].copy()
n = len(ho)

print(f"Universe: {len(common)} tickers | {n} hourly bars "
      f"({ho.index[0].date()} to {ho.index[-1].date()})")

bar_ts = ho.index
ho_np = ho.values.astype(float)
hc_np = hc.values.astype(float)

# Grid params
LB_GRID = [60, 120, 200, 300]
HOLD_GRID = [1, 2, 5, 10, 20]
TOP_N_GRID = [5, 10, 20, 30]

# Precompute momentum
print("Precomputing momentum (close-to-close)...", flush=True)
mom_cache = {}
for lb in LB_GRID:
    mom_cache[lb] = hc.pct_change(lb).values.astype(float)
    print(f"  lookback={lb}h: done")

# Run long grid
print("\nRunning long-only grid search...", flush=True)
grid_h = []
best_h_ret = None
best_h_sh = -np.inf
best_h_params = None

total = len(LB_GRID) * len(HOLD_GRID) * len(TOP_N_GRID)
combo = 0

for lb, hold, top_n in product(LB_GRID, HOLD_GRID, TOP_N_GRID):
    combo += 1
    if combo % 20 == 0:
        print(f"  combo {combo}/{total}...")

    mom_np = mom_cache[lb]
    trade_rets = []
    busy = np.full(len(common), -1, dtype=int)

    for i in range(lb, n - hold - 1):
        row = mom_np[i]
        valid_mask = np.isfinite(row)
        if valid_mask.sum() < top_n:
            continue

        ranked = np.where(valid_mask)[0]
        top_idx = ranked[np.argsort(row[ranked])[-top_n:]]

        entry_bar = i + 1
        exit_bar = min(i + hold, n - 1)

        free = [j for j in top_idx if busy[j] < entry_bar]
        if not free:
            continue

        ep = ho_np[entry_bar, free]
        xp = hc_np[exit_bar, free]
        ok = (ep > 0) & np.isfinite(ep) & np.isfinite(xp)
        if not ok.any():
            continue

        ep_ok = ep[ok]
        xp_ok = xp[ok]
        raw = (xp_ok / ep_ok - 1)
        net = raw - TC_STRICT

        trade_rets.append({
            "entry_bar": entry_bar,
            "entry_dt": bar_ts[entry_bar],
            "avg_ret": float(net.mean()),
            "n_pos": int(ok.sum()),
        })

        busy[np.array(free)[ok]] = exit_bar

    if len(trade_rets) < 5:
        grid_h.append({"lookback": lb, "hold": hold, "top_n": top_n,
                       "Sharpe": np.nan, "Sortino": np.nan, "Total_Return": np.nan,
                       "Max_DD": np.nan, "n_trades": len(trade_rets)})
        continue

    tdf = pd.DataFrame(trade_rets)
    daily = (tdf.assign(dt=tdf["entry_dt"].dt.normalize())
                 .groupby("dt")["avg_ret"].mean()
                 .reindex(pd.date_range(tdf["entry_dt"].dt.normalize().min(),
                                        bar_ts[-1].normalize(), freq="B"),
                          fill_value=0.0))

    s = _stats(daily)
    grid_h.append({"lookback": lb, "hold": hold, "top_n": top_n,
                   "Sharpe": s["Sharpe"], "Sortino": s["Sortino"],
                   "Total_Return": s["Total Return"], "Max_DD": s["Max DD"],
                   "n_trades": len(trade_rets)})

    if pd.notna(s["Sharpe"]) and s["Sharpe"] > best_h_sh:
        best_h_sh = s["Sharpe"]
        best_h_ret = daily.rename("HourlyMom_Long")
        best_h_params = {"lookback": lb, "hold": hold, "top_n": top_n}

grid_h_df = pd.DataFrame(grid_h).sort_values("Sharpe", ascending=False)

print(f"\nHourly Grid Results (top 10):")
print(f"{'LB':>6} {'Hold':>6} {'N':>4} | {'Sharpe':>7} {'Sortino':>8} {'Return':>9} {'MaxDD':>8} {'Trades':>7}")
print("-" * 60)
for _, row in grid_h_df.head(10).iterrows():
    print(f"{int(row['lookback']):>6} {int(row['hold']):>6} {int(row['top_n']):>4} | "
          f"{row['Sharpe']:>7.3f} {row['Sortino']:>8.3f} {row['Total_Return']:>9.2%} "
          f"{row['Max_DD']:>8.2%} {int(row['n_trades']):>7}")

if best_h_params:
    print(f"\nBest: lookback={best_h_params['lookback']}h  "
          f"hold={best_h_params['hold']}h  top_n={best_h_params['top_n']}")
    print(f"Sharpe={best_h_sh:.3f}  Period: {best_h_ret.index[0].date()} to {best_h_ret.index[-1].date()}")
else:
    print("\nNo valid results for hourly!")


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 -- Daily Momentum (140d lookback, 40d hold)
# ══════════════════════════════════════════════════════════════════════════

_header("Strategy B -- Daily Momentum (140d/40d, TC=0.5%)")

print("Loading daily cache...", flush=True)
daily_close = pd.read_parquet(Path("data/cache/daily_close.parquet"))
daily_close.index = pd.to_datetime(daily_close.index)
print(f"Daily cache: {daily_close.index[0].date()} to {daily_close.index[-1].date()} "
      f"| {len(daily_close)} days x {len(daily_close.columns)} tickers")

LOOKBACK_D = 140
HOLD_D = 40
TOP_N_DAILY = [5, 10, 20, 30]

daily_grid = []
best_d_ret = None
best_d_sh = -np.inf
best_d_tn = None

for top_n in TOP_N_DAILY:
    ret_140 = daily_close.pct_change(LOOKBACK_D)
    port_rows = []
    prev_holdings = set()

    i = LOOKBACK_D
    while i < len(daily_close) - 1:
        date = daily_close.index[i]
        mom_row = ret_140.iloc[i].dropna()

        if len(mom_row) < top_n:
            i += HOLD_D
            continue

        top_stocks = mom_row.nlargest(top_n).index.tolist()

        # TC: 0.5% on each position that changes
        entering = set(top_stocks) - prev_holdings
        exiting = prev_holdings - set(top_stocks)
        n_changes = len(entering) + len(exiting)
        tc_cost = TC_STRICT * n_changes / top_n

        hold_end = min(i + HOLD_D, len(daily_close) - 1)
        hold_dates = daily_close.index[i+1 : hold_end+1]

        port_daily_ret = (daily_close[top_stocks]
                          .pct_change()
                          .iloc[i+1 : hold_end+1]
                          .mean(axis=1))

        if len(port_daily_ret) > 0:
            port_daily_ret.iloc[0] -= tc_cost

        for d, rv in zip(hold_dates, port_daily_ret):
            port_rows.append({"date": d, "ret": rv})

        prev_holdings = set(top_stocks)
        i += HOLD_D

    if not port_rows:
        continue

    port_ret = (pd.DataFrame(port_rows)
                  .set_index("date")["ret"]
                  .groupby(level=0).mean()
                  .reindex(daily_close.index[LOOKBACK_D:], fill_value=0.0)
                  .dropna())
    port_ret.name = f"DailyMom_n{top_n}"

    s_d = _stats(port_ret)
    daily_grid.append({"top_n": top_n, "lookback": LOOKBACK_D, "hold": HOLD_D, **s_d})

    if pd.notna(s_d["Sharpe"]) and s_d["Sharpe"] > best_d_sh:
        best_d_sh = s_d["Sharpe"]
        best_d_ret = port_ret.rename("DailyMom_LongOnly")
        best_d_tn = top_n

grid_d_df = pd.DataFrame(daily_grid)
print(f"\nDaily Momentum Grid (lookback={LOOKBACK_D}d, hold={HOLD_D}d):")
print(f"{'N':>5} | {'Sharpe':>7} {'Sortino':>8} {'Return':>9} {'Max_DD':>8}")
print("-" * 42)
for _, r in grid_d_df.iterrows():
    print(f"{int(r['top_n']):>5} | {r['Sharpe']:>7.3f} {r['Sortino']:>8.3f} "
          f"{r['Total Return']:>9.2%} {r['Max DD']:>8.2%}")

print(f"\nBest daily: top_n={best_d_tn}  Sharpe={best_d_sh:.3f}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 -- Head-to-head comparison
# ══════════════════════════════════════════════════════════════════════════

_header("COMPARISON (Common Period)")

if best_h_ret is not None and best_d_ret is not None:
    common = best_h_ret.index.intersection(best_d_ret.index)
    h_ret = best_h_ret.loc[common]
    d_ret = best_d_ret.loc[common]

    print(f"Common period: {common[0].date()} to {common[-1].date()} ({len(common)} days)")
    print()

    s_h = _stats(h_ret)
    s_d = _stats(d_ret)

    print(f"{'Metric':<22} {'Hourly Momentum':>18} {'Daily Momentum':>18}")
    print("-" * 60)
    for metric in ["Sharpe", "Sortino", "Total Return", "Max DD"]:
        hv = s_h[metric]
        dv = s_d[metric]
        fmt = ".3f" if metric not in ("Total Return", "Max DD") else ".2%"
        print(f"{metric:<22} {hv:>18{fmt}} {dv:>18{fmt}}")

    # Yearly
    _header("Yearly Performance (Common Period)")
    all_yrs = sorted(set(h_ret.index.year.unique()) | set(d_ret.index.year.unique()))
    print(f"\n{'Year':>6} | {'H_Ret':>10} {'H_Sharpe':>10} {'H_DD':>9} | "
          f"{'D_Ret':>10} {'D_Sharpe':>10} {'D_DD':>9}")
    print("-" * 76)
    for yr in all_yrs:
        hg = h_ret[h_ret.index.year == yr]
        dg = d_ret[d_ret.index.year == yr]
        if len(hg) < 5 or len(dg) < 5:
            continue
        hw = (1+hg).cumprod(); hw = hw/hw.iloc[0]
        dw = (1+dg).cumprod(); dw = dw/dw.iloc[0]
        print(f"{yr:>6} | {float(hw.iloc[-1]-1):>+10.2%} {_sharpe(hg):>10.3f} "
              f"{float((hw/hw.cummax()-1).min()):>9.2%} | "
              f"{float(dw.iloc[-1]-1):>+10.2%} {_sharpe(dg):>10.3f} "
              f"{float((dw/dw.cummax()-1).min()):>9.2%}")

    # Charts
    _header("Charts")

    colors = {"Hourly": "steelblue", "Daily": "crimson"}

    fig, axes = plt.subplots(3, 1, figsize=(18, 14),
                              gridspec_kw={"height_ratios": [3, 1.5, 1.5]})

    ax1 = axes[0]
    for label, s, col in [
        (f"Hourly Momentum (lb={best_h_params['lookback']}h "
         f"hold={best_h_params['hold']}h n={best_h_params['top_n']}, TC=0.5%)",
         h_ret, colors["Hourly"]),
        (f"Daily Momentum (lb={LOOKBACK_D}d hold={HOLD_D}d n={best_d_tn}, TC=0.5%)",
         d_ret, colors["Daily"]),
    ]:
        w = (1+s).cumprod(); w = w/w.iloc[0]
        ax1.plot(w.index, w.values, label=label, color=col, linewidth=2)

    ax1.set_title("Strict Comparison: Hourly vs Daily Momentum (0.5% TC)", fontsize=13)
    ax1.set_ylabel("Cumulative Wealth"); ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x,_: f"{x:.1f}x"))
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.4)

    ax2 = axes[1]
    for label, s, col in [("Hourly", h_ret, colors["Hourly"]),
                          ("Daily", d_ret, colors["Daily"])]:
        w = (1+s).cumprod(); w = w/w.iloc[0]
        dd = w/w.cummax()-1
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.4, color=col, label=label)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.set_ylabel("Drawdown"); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.4)

    ax3 = axes[2]
    for label, s, col in [("Hourly", h_ret, colors["Hourly"]),
                          ("Daily", d_ret, colors["Daily"])]:
        roll = s.rolling(60).apply(
            lambda x: float(np.sqrt(252)*x.mean()/x.std()) if x.std()>0 else np.nan)
        ax3.plot(roll.index, roll.values, label=label, color=col, linewidth=1.5)
    ax3.axhline(0, color="black", lw=0.8, linestyle="--")
    ax3.set_ylabel("60-day Rolling Sharpe"); ax3.legend(fontsize=9); ax3.grid(True, alpha=0.4)

    plt.tight_layout(); plt.show()

    # Daily momentum full history
    _header("Daily Momentum Full History (2018-2026)")
    print(f"Period: {best_d_ret.index[0].date()} to {best_d_ret.index[-1].date()}")
    s_d_full = _stats(best_d_ret)
    print(f"Sharpe={s_d_full['Sharpe']:.3f}  Return={s_d_full['Total Return']:+.1%}  "
          f"MaxDD={s_d_full['Max DD']:.1%}")

    fig2, axes2 = plt.subplots(2, 1, figsize=(18, 10),
                                gridspec_kw={"height_ratios": [3, 1]})
    w_full = (1+best_d_ret).cumprod(); w_full = w_full/w_full.iloc[0]
    axes2[0].plot(w_full.index, w_full.values, color=colors["Daily"], linewidth=1.8,
                  label=f"Daily Momentum (lb={LOOKBACK_D}d hold={HOLD_D}d n={best_d_tn})")
    axes2[0].set_title(f"Daily Momentum Full History | "
                       f"Sharpe={s_d_full['Sharpe']:.3f}  Return={s_d_full['Total Return']:+.1%}")
    axes2[0].set_ylabel("Cumulative Wealth")
    axes2[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.1f}x"))
    axes2[0].legend(fontsize=10); axes2[0].grid(True, alpha=0.4)

    dd_full = w_full/w_full.cummax()-1
    axes2[1].fill_between(dd_full.index, dd_full.values, 0, alpha=0.5, color=colors["Daily"])
    axes2[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axes2[1].set_ylabel("Drawdown"); axes2[1].grid(True, alpha=0.4)
    plt.tight_layout(); plt.show()


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 -- Save
# ══════════════════════════════════════════════════════════════════════════

_header("Save Results")

xl = "results/momentum_strict_comparison.xlsx"
try:
    with pd.ExcelWriter(xl, engine="openpyxl") as writer:
        # Summary
        rows = []
        if best_h_ret is not None:
            w = (1+best_h_ret).cumprod(); w = w/w.iloc[0]
            rows.append({
                "Strategy": "Hourly_Momentum",
                "lookback": f"{best_h_params['lookback']}h",
                "hold": f"{best_h_params['hold']}h",
                "top_n": best_h_params['top_n'],
                "Start": str(best_h_ret.index[0].date()),
                "End": str(best_h_ret.index[-1].date()),
                "Days": len(best_h_ret),
                "Sharpe": _sharpe(best_h_ret),
                "Sortino": _sortino(best_h_ret),
                "Total_Return": float(w.iloc[-1]-1),
                "Max_DD": _mdd(best_h_ret),
                "TC": "0.5% per trade",
            })

        if best_d_ret is not None:
            w = (1+best_d_ret).cumprod(); w = w/w.iloc[0]
            rows.append({
                "Strategy": "Daily_Momentum_FullHist",
                "lookback": f"{LOOKBACK_D}d",
                "hold": f"{HOLD_D}d",
                "top_n": best_d_tn,
                "Start": str(best_d_ret.index[0].date()),
                "End": str(best_d_ret.index[-1].date()),
                "Days": len(best_d_ret),
                "Sharpe": _sharpe(best_d_ret),
                "Sortino": _sortino(best_d_ret),
                "Total_Return": float(w.iloc[-1]-1),
                "Max_DD": _mdd(best_d_ret),
                "TC": "0.5% per position",
            })

        pd.DataFrame(rows).to_excel(writer, sheet_name="Summary", index=False)
        grid_h_df.to_excel(writer, sheet_name="Grid_Hourly", index=False)
        grid_d_df.to_excel(writer, sheet_name="Grid_Daily", index=False)

        if best_h_ret is not None and best_d_ret is not None:
            common = best_h_ret.index.intersection(best_d_ret.index)
            h_ret_c = best_h_ret.loc[common]
            d_ret_c = best_d_ret.loc[common]
            dr_out = pd.DataFrame({
                "HourlyMom": h_ret_c, "DailyMom": d_ret_c,
                "DailyMom_Full": best_d_ret,
            })
            dr_out.to_excel(writer, sheet_name="Daily_Returns")

    print(f"  Saved: {xl}")
except Exception as e:
    print(f"  Save failed: {e}")


_header("FINAL SUMMARY (TC=0.5% strict)")

if best_h_ret is not None and best_d_ret is not None:
    common = best_h_ret.index.intersection(best_d_ret.index)
    h_ret = best_h_ret.loc[common]
    d_ret = best_d_ret.loc[common]

    s_h = _stats(h_ret)
    s_d = _stats(d_ret)

    print(f"\nCommon period: {common[0].date()} to {common[-1].date()} ({len(common)} days)")
    print(f"\n  Hourly:  lb={best_h_params['lookback']}h hold={best_h_params['hold']}h "
          f"n={best_h_params['top_n']}")
    print(f"    Sharpe={s_h['Sharpe']:.3f}  Return={s_h['Total Return']:+.1%}  "
          f"MaxDD={s_h['Max DD']:.1%}")
    print(f"\n  Daily:   lb={LOOKBACK_D}d hold={HOLD_D}d n={best_d_tn}")
    print(f"    Sharpe={s_d['Sharpe']:.3f}  Return={s_d['Total Return']:+.1%}  "
          f"MaxDD={s_d['Max DD']:.1%}")
    print(f"\n  Daily Full (2018-2026):")
    s_d_full = _stats(best_d_ret)
    print(f"    Sharpe={s_d_full['Sharpe']:.3f}  Return={s_d_full['Total Return']:+.1%}  "
          f"MaxDD={s_d_full['Max DD']:.1%}")
