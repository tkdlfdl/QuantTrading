"""
Short Squeeze Backtest
=======================
1. Fetch current short interest (shortPercentOfFloat) for NASDAQ100 + S&P500
   via yfinance -> rank -> pick top-N as the "squeeze-prone" universe.
2. Load merged hourly Alpaca+yfinance cache (2019-2026).
3. Run grid search: signal_sigma x hold_hours x lookback_hours x top_n x min_ret.
4. Show yearly Return / Sharpe / Max Drawdown.

NOTE: Using CURRENT short interest as a static historical universe is a
simplification. Stocks with chronically high short interest (e.g. heavily
shorted tech, bio, retail) tend to remain so for years, making this a
reasonable first-pass proxy. A more rigorous version would use FINRA
bi-weekly historical short interest files.

Usage:
  python run_short_squeeze.py
"""
import sys, warnings, socket
socket.setdefaulttimeout(8)   # hard global limit on ALL socket ops (prevents yfinance hangs)
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_n = [0]
def _save(*a, **k):
    _n[0] += 1
    p = f"results/squeeze_chart_{_n[0]}.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    print(f"  [chart saved: {p}]")
plt.show = _save

sys.path.insert(0, ".")

import os, math, json
import numpy as np
import pandas as pd
import requests
from io import StringIO

import yfinance as yf

from data.db.schema import init
from data.universe import get_universe
from strategies.short_squeeze import run_short_squeeze

os.makedirs("results", exist_ok=True)

TRADING_DAYS  = 252
TOP_N_UNIVERSE = 80    # fetch SI for this many top-ranked stocks; strategy grid picks subset
TC_RATE       = 0.001  # 0.1% one-way transaction cost

SQUEEZE_GRID = dict(
    signal_sigma_grid   = [2.0, 2.5, 3.0, 3.5],
    hold_hours_grid     = [1, 2, 4],
    lookback_hours_grid = [20, 80, 130],
    top_n_grid          = [5, 20],
    min_ret_grid        = [0.003, 0.010],
    transaction_cost    = TC_RATE,
)  # 4 x 3 x 3 x 2 x 2 = 144 combos (down from 720)


# ── Helpers ────────────────────────────────────────────────────────────────

def _sharpe(r, td=TRADING_DAYS):
    s = r.std(); return float(np.sqrt(td) * r.mean() / s) if s > 0 else np.nan

def _sortino(r, td=TRADING_DAYS):
    ds = r[r < 0].std(); return float(np.sqrt(td) * r.mean() / ds) if ds > 0 else np.nan

def _mdd(r):
    w = (1 + r).cumprod(); w = w / w.iloc[0]
    return float((w / w.cummax() - 1).min())

def _header(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}")

def yearly_table(ret_dict):
    rows = []
    all_years = sorted({y for r in ret_dict.values() for y in r.index.year.unique()})
    for yr in all_years:
        row = {"Year": yr}
        for name, r in ret_dict.items():
            yr_r = r[r.index.year == yr]
            if yr_r.empty:
                row[f"{name}_Ret"] = np.nan; row[f"{name}_Sharpe"] = np.nan; row[f"{name}_MDD"] = np.nan
            else:
                w = (1 + yr_r).cumprod(); w = w / w.iloc[0]
                row[f"{name}_Ret"]    = round(float(w.iloc[-1] - 1), 4)
                row[f"{name}_Sharpe"] = round(_sharpe(yr_r), 3)
                row[f"{name}_MDD"]    = round(float((w / w.cummax() - 1).min()), 4)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Year")

def print_yearly(df, names):
    for name in names:
        cols = [f"{name}_Ret", f"{name}_Sharpe", f"{name}_MDD"]
        present = [c for c in cols if c in df.columns]
        if not present: continue
        print(f"\n  [{name}]")
        sub = df[present].copy(); sub.columns = ["Return","Sharpe","MaxDD"]
        for yr, row in sub.iterrows():
            r  = f"{row['Return']:>+8.2%}" if pd.notna(row['Return'])  else "       N/A"
            sh = f"{row['Sharpe']:>7.3f}"  if pd.notna(row['Sharpe'])  else "    N/A"
            md = f"{row['MaxDD']:>8.2%}"   if pd.notna(row['MaxDD'])   else "     N/A"
            print(f"    {yr}:  Return={r}  Sharpe={sh}  MaxDD={md}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 - Build high short-interest universe
# ══════════════════════════════════════════════════════════════════════════

init()
_header("STEP 1 - Build high short-interest universe")

universe = get_universe()
print(f"Full universe: {len(universe)} tickers")

_SI_CACHE = "results/short_interest_universe.csv"
if os.path.exists(_SI_CACHE):
    print(f"  Loading cached SI data from {_SI_CACHE}")
    si_df = pd.read_csv(_SI_CACHE).sort_values("shortPercentOfFloat", ascending=False)
    print(f"  Loaded {len(si_df)} tickers from cache.")
else:
    print("  No SI cache found. Run yfinance fetch manually first.")
    import sys; sys.exit(1)

print(f"\nTop 30 most shorted stocks (% of float):")
print(si_df.head(30)[["ticker","shortPercentOfFloat","shortRatio"]].to_string(index=False))

squeeze_universe = si_df["ticker"].head(TOP_N_UNIVERSE).tolist()
print(f"\nSqueeze universe (top {TOP_N_UNIVERSE} by short float %):")
print(", ".join(squeeze_universe))


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 - Load hourly price data
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 2 - Load hourly price data (from merged cache)")

# Use pre-built merged hourly cache (2019-2026, 7.4 years)
# Bypasses yfinance download (too slow for 516 tickers)
_MERGED_O = "data/cache/merged_hourly_open.parquet"
_MERGED_C = "data/cache/merged_hourly_close.parquet"

hourly_open  = pd.read_parquet(_MERGED_O)
hourly_close = pd.read_parquet(_MERGED_C)

# Normalize timestamps to hour-floor
hourly_open.index  = hourly_open.index.floor("h")
hourly_close.index = hourly_close.index.floor("h")
hourly_open  = hourly_open[~hourly_open.index.duplicated(keep="last")]
hourly_close = hourly_close[~hourly_close.index.duplicated(keep="last")]

# Align on common timestamps
idx = hourly_open.index.intersection(hourly_close.index)
hourly_open  = hourly_open.loc[idx].ffill()
hourly_close = hourly_close.loc[idx].ffill()

print(f"\nHourly data: {hourly_open.index[0].date()} to {hourly_open.index[-1].date()}")
print(f"  {len(hourly_open)} bars x {len(hourly_open.columns)} tickers")

in_cache = [t for t in squeeze_universe if t in hourly_open.columns]
missing  = [t for t in squeeze_universe if t not in hourly_open.columns]
print(f"\nSqueeze universe in hourly cache: {len(in_cache)}/{len(squeeze_universe)}")
if missing:
    print(f"  Missing from cache: {missing}")

si_in_cache = si_df[si_df["ticker"].isin(in_cache)].head(20)
print(f"\nTop 20 squeeze stocks (in cache) by short float %:")
print(si_in_cache[["ticker","shortPercentOfFloat","shortRatio"]].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 - Run short squeeze strategy grid search
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 3 - Short squeeze strategy grid search")

total_combos = (len(SQUEEZE_GRID["signal_sigma_grid"]) *
                len(SQUEEZE_GRID["hold_hours_grid"]) *
                len(SQUEEZE_GRID["lookback_hours_grid"]) *
                len(SQUEEZE_GRID["top_n_grid"]) *
                len(SQUEEZE_GRID["min_ret_grid"]))
print(f"Total grid combinations: {total_combos}")
print(f"Universe: top {TOP_N_UNIVERSE} highest-shorted stocks\n")

best_ret, best_params, grid_df = run_short_squeeze(
    hourly_open   = hourly_open,
    hourly_close  = hourly_close,
    short_universe = in_cache,
    **SQUEEZE_GRID,
)

if best_ret is None:
    print("ERROR: no valid grid results.")
    sys.exit(1)

# ── Top grid results ────────────────────────────────────────────────────────
_header("STEP 3 - Top 20 grid results by Sharpe")
print(f"{'Lookback':>9} {'Sigma':>6} {'Hold':>5} {'TopN':>6} {'MinRet':>8} | "
      f"{'Sharpe':>7} {'Sortino':>8} {'Return':>9} {'Max_DD':>8} {'nTrades':>8} {'WinRate':>8}")
print("-" * 85)
for _, r in grid_df.head(20).iterrows():
    print(f"{int(r['lookback']):>9} {r['sigma']:>6.1f} {int(r['hold_hours']):>5} "
          f"{int(r['top_n']):>6} {r['min_ret']:>8.3f} | "
          f"{r['Sharpe']:>7.3f} {r['Sortino']:>8.3f} {r['Total_Return']:>9.2%} "
          f"{r['Max_DD']:>8.2%} {int(r['n_trades']):>8} {r['Win_Rate']:>8.2%}")

p = best_params
print(f"\nBest: lookback={p['lookback']}h  sigma={p['sigma']}  "
      f"hold={p['hold_hours']}h  top_n={p['top_n']}  min_ret={p['min_ret']:.3f}")
print(f"  Sharpe={p['Sharpe']:.3f}  Sortino={p['Sortino']:.3f}  "
      f"Return={p['Total_Return']:+.1%}  MaxDD={p['Max_DD']:.1%}  "
      f"Trades={p['n_trades']}  WinRate={p['Win_Rate']:.1%}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 - Yearly breakdown
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 4 - Yearly performance")
yr = yearly_table({"ShortSqueeze": best_ret})
print_yearly(yr, ["ShortSqueeze"])


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 - Charts
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 5 - Charts")

wealth = (1 + best_ret).cumprod(); wealth = wealth / wealth.iloc[0]
dd = wealth / wealth.cummax() - 1

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                gridspec_kw={"height_ratios": [3, 1]})

ax1.plot(wealth.index, wealth.values, color="crimson", linewidth=2, label="Short Squeeze")
ax1.set_title(f"Short Squeeze Strategy | sigma={p['sigma']} | hold={p['hold_hours']}h | "
              f"lb={p['lookback']}h | top_n={p['top_n']} | min_ret={p['min_ret']:.3f}",
              fontsize=12)
ax1.set_ylabel("Cumulative Wealth")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}x"))
ax1.grid(True, alpha=0.4); ax1.legend()

ax2.fill_between(dd.index, dd.values, 0, alpha=0.5, color="crimson")
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax2.set_ylabel("Drawdown"); ax2.grid(True, alpha=0.4)

plt.tight_layout(); plt.show()

# ── Short interest distribution chart ───────────────────────────────────
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))

top30 = si_df.head(30)
ax3.barh(top30["ticker"][::-1], top30["shortPercentOfFloat"][::-1] * 100,
         color="steelblue", alpha=0.8)
ax3.set_xlabel("Short % of Float")
ax3.set_title("Top 30 Most Shorted Stocks (NASDAQ100 + S&P500)")
ax3.xaxis.set_major_formatter(mticker.PercentFormatter())
ax3.grid(True, axis="x", alpha=0.4)

# Sharpe heat by sigma x hold_hours for best lookback and top_n
pivot_data = grid_df[
    (grid_df["lookback"] == p["lookback"]) &
    (grid_df["top_n"]    == p["top_n"]) &
    (grid_df["min_ret"]  == p["min_ret"])
].pivot_table(index="sigma", columns="hold_hours", values="Sharpe")
if not pivot_data.empty:
    im = ax4.imshow(pivot_data.values, aspect="auto",
                    cmap="RdYlGn", vmin=-0.5, vmax=2.0)
    ax4.set_xticks(range(len(pivot_data.columns)))
    ax4.set_xticklabels([f"{h}h" for h in pivot_data.columns])
    ax4.set_yticks(range(len(pivot_data.index)))
    ax4.set_yticklabels([f"sigma={s}" for s in pivot_data.index])
    plt.colorbar(im, ax=ax4, label="Sharpe")
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            v = pivot_data.values[i, j]
            ax4.text(j, i, f"{v:.2f}" if pd.notna(v) else "N/A",
                     ha="center", va="center", fontsize=9,
                     color="black" if pd.notna(v) and -0.5 < v < 1.5 else "white")
    ax4.set_title(f"Sharpe: sigma x hold_hours\n(lb={p['lookback']}h, top_n={p['top_n']}, min_ret={p['min_ret']:.3f})")

plt.tight_layout(); plt.show()


# ══════════════════════════════════════════════════════════════════════════
# STEP 6 - Save to Excel
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 6 - Saving to Excel")
xl = "results/short_squeeze_backtest.xlsx"
try:
    with pd.ExcelWriter(xl, engine="openpyxl") as writer:
        # Summary
        pd.DataFrame([{
            "Strategy": "ShortSqueeze",
            "Start": str(best_ret.index[0].date()),
            "End":   str(best_ret.index[-1].date()),
            "Days":  len(best_ret),
            "Total Return": float(wealth.iloc[-1] - 1),
            "Sharpe":  _sharpe(best_ret),
            "Sortino": _sortino(best_ret),
            "Max DD":  _mdd(best_ret),
            **{k: p[k] for k in ["lookback","sigma","hold_hours","top_n","min_ret",
                                  "n_trades","Win_Rate"]},
        }]).to_excel(writer, sheet_name="Summary", index=False)

        # Yearly
        yr.reset_index().to_excel(writer, sheet_name="Yearly", index=False)

        # Full grid
        grid_df.to_excel(writer, sheet_name="Grid_Results", index=False)

        # Short interest universe
        si_df.to_excel(writer, sheet_name="Short_Interest", index=False)

        # Daily returns
        pd.DataFrame({"Date": best_ret.index, "Return": best_ret.values}) \
          .to_excel(writer, sheet_name="Daily_Returns", index=False)

    print(f"  Saved: {xl}")
except Exception as e:
    print(f"  Excel save failed: {e}")


_header("DONE")
print(f"\nShort Squeeze Strategy (top {TOP_N_UNIVERSE} most-shorted stocks)")
print(f"Best params: lookback={p['lookback']}h | sigma={p['sigma']} | "
      f"hold={p['hold_hours']}h | top_n={p['top_n']} | min_ret={p['min_ret']:.3f}")
print(f"  Sharpe={p['Sharpe']:.3f}  Return={p['Total_Return']:+.1%}  "
      f"MaxDD={p['Max_DD']:.1%}  Trades={p['n_trades']}  WinRate={p['Win_Rate']:.1%}")
print(f"\nNOTE: Universe built from CURRENT yfinance short interest data.")
print(f"  More rigorous: use FINRA bi-weekly historical SI files per settlement date.")
