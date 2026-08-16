"""
Short Squeeze Strategy v2 -- Point-in-time, No Lookahead
=========================================================
Universe  : Stocks with highest short interest from the MOST RECENT bi-weekly
            Nasdaq settlement report that is STRICTLY BEFORE each trade date.
            (12-month history from Nasdaq API -- Jun 2025 to Jun 2026)

Signal    : At hourly bar t:
              rolling_mean, rolling_std = computed over bars [t-lookback, t-1]
                                          (shift(1) guarantees no lookahead)
              signal = return[t] > rolling_mean + z * rolling_std
                       AND stock in high-SI universe at t

Entry     : LONG at open of bar t+1  (next bar after signal)
Exit      : close of bar t+hold_hours

Grid      : lookback_hours x z x hold_hours x top_n  (240 combos)

TC        : 0.1% round-trip deducted from each trade

Usage:
  python run_short_squeeze_v2.py
"""
import sys, warnings, socket, os, math, json
warnings.filterwarnings("ignore")
socket.setdefaulttimeout(8)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_n = [0]
def _save(*a, **k):
    _n[0] += 1
    p = f"results/squeeze_v2_chart_{_n[0]}.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    print(f"  [chart saved: {p}]", flush=True)
plt.show = _save

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import requests
from itertools import product
from pathlib import Path

import yfinance as yf
from data.db.schema import init
from data.universe import get_universe
from data.intraday_loader import load_hourly_bars

os.makedirs("results", exist_ok=True)

TRADING_DAYS = 252
TC_RATE      = 0.001   # 0.1% one-way per trade
TOP_N_UNIV   = 50      # fetch SI for top-50 ranked stocks in each period

GRID = dict(
    lookback_hours = [20, 40, 80, 120],
    z_grid         = [1.5, 2.0, 2.5, 3.0, 3.5],
    hold_hours     = [1, 2, 3, 4],
    top_n          = [10, 20, 30],
)

_NAS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept":     "application/json, text/plain, */*",
    "Referer":    "https://www.nasdaq.com/",
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _sharpe(r, td=TRADING_DAYS):
    s = r.std(); return float(np.sqrt(td)*r.mean()/s) if s > 0 else np.nan

def _sortino(r, td=TRADING_DAYS):
    ds = r[r<0].std(); return float(np.sqrt(td)*r.mean()/ds) if ds > 0 else np.nan

def _mdd(r):
    w = (1+r).cumprod(); w = w/w.iloc[0]
    return float((w/w.cummax()-1).min())

def _header(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}", flush=True)

def yearly_table(ret_dict):
    rows = []
    all_years = sorted({y for r in ret_dict.values() for y in r.index.year.unique()})
    for yr in all_years:
        row = {"Year": yr}
        for name, r in ret_dict.items():
            yr_r = r[r.index.year == yr]
            if yr_r.empty:
                row[f"{name}_Ret"] = np.nan
                row[f"{name}_Sharpe"] = np.nan
                row[f"{name}_MDD"] = np.nan
            else:
                w = (1+yr_r).cumprod(); w = w/w.iloc[0]
                row[f"{name}_Ret"]    = round(float(w.iloc[-1]-1), 4)
                row[f"{name}_Sharpe"] = round(_sharpe(yr_r), 3)
                row[f"{name}_MDD"]    = round(float((w/w.cummax()-1).min()), 4)
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
# STEP 1 -- Fetch 12-month bi-weekly short interest from Nasdaq API
# ══════════════════════════════════════════════════════════════════════════

init()
_header("STEP 1 -- Fetch 12-month bi-weekly short interest (Nasdaq API)")

universe = get_universe()
print(f"Universe: {len(universe)} tickers", flush=True)

_SI_PANEL_CACHE = Path("results/si_panel_12m.parquet")
sess = requests.Session(); sess.headers.update(_NAS_HEADERS)

def _fetch_ticker_si(ticker: str) -> list[dict]:
    """Fetch all available bi-weekly SI rows for one ticker from Nasdaq API."""
    try:
        url = (f"https://api.nasdaq.com/api/quote/{ticker}/short-interest"
               f"?type=COMMON&assetClass=stocks&offset=0&limit=200")
        r = sess.get(url, timeout=8)
        if r.status_code != 200:
            return []
        rows = (r.json().get("data", {})
                        .get("shortInterestTable", {})
                        .get("rows", []))
        result = []
        for row in rows:
            try:
                dt  = pd.to_datetime(row["settlementDate"], format="%m/%d/%Y")
                si  = int(str(row["interest"]).replace(",", ""))
                dtc = float(row["daysToCover"])
                result.append({"ticker": ticker, "settlementDate": dt,
                                "sharesShort": si, "daysToCover": dtc})
            except Exception:
                pass
        return result
    except Exception:
        return []


if _SI_PANEL_CACHE.exists():
    print(f"Loading cached SI panel from {_SI_PANEL_CACHE}", flush=True)
    si_raw = pd.read_parquet(_SI_PANEL_CACHE)
else:
    all_rows = []
    _CHUNK = 50
    for i in range(0, len(universe), _CHUNK):
        batch = universe[i:i+_CHUNK]
        print(f"  batch {i//_CHUNK+1}/{math.ceil(len(universe)/_CHUNK)} "
              f"({len(batch)} tickers)...", end=" ", flush=True)
        for t in batch:
            all_rows.extend(_fetch_ticker_si(t))
        print(f"ok ({len(all_rows)} rows so far)", flush=True)

    si_raw = pd.DataFrame(all_rows)
    si_raw.to_parquet(_SI_PANEL_CACHE)
    print(f"Saved SI panel: {len(si_raw)} rows")

print(f"\nSI panel: {len(si_raw)} rows")
print(f"  Tickers: {si_raw['ticker'].nunique()}")
print(f"  Date range: {si_raw['settlementDate'].min().date()} to "
      f"{si_raw['settlementDate'].max().date()}")
print(f"  Settlement dates: {si_raw['settlementDate'].nunique()}")

# Show top shorted stocks at most recent settlement
latest_date = si_raw["settlementDate"].max()
latest_si   = (si_raw[si_raw["settlementDate"] == latest_date]
               .sort_values("daysToCover", ascending=False)
               .head(20))
print(f"\nTop 20 by Days-to-Cover at {latest_date.date()}:")
print(latest_si[["ticker","sharesShort","daysToCover"]].to_string(index=False))


# ── Build point-in-time universe lookup ────────────────────────────────────
# For each settlement date, rank tickers by daysToCover
# si_timeline: sorted list of (settlementDate, set_of_top_tickers)

settlement_dates = sorted(si_raw["settlementDate"].unique())
print(f"\nBuilding point-in-time universe (top {TOP_N_UNIV} by daysToCover per period)...")

si_timeline = []   # list of (settlement_dt, {ticker: days_to_cover})
for sd in settlement_dates:
    period_df = (si_raw[si_raw["settlementDate"] == sd]
                 .sort_values("daysToCover", ascending=False))
    # dict: ticker -> daysToCover for this period
    top_dict = dict(zip(period_df["ticker"], period_df["daysToCover"]))
    si_timeline.append((sd, top_dict))

_PUB_LAG = pd.offsets.BusinessDay(8)  # FINRA publishes ~8 biz days after settlement

def get_universe_at(trade_dt: pd.Timestamp, n: int) -> set:
    """
    Return top-n tickers by DTC from the most recent settlement report whose
    PUBLICATION DATE (settlement + 8 business days) is strictly before trade_dt.
    """
    eligible = [(sd, d) for sd, d in si_timeline if sd + _PUB_LAG < trade_dt]
    if not eligible:
        return set()
    _, top_dict = eligible[-1]
    ranked = sorted(top_dict.items(), key=lambda x: x[1], reverse=True)[:n]
    return {t for t, _ in ranked}


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 -- Load hourly price data
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 2 -- Load hourly price data")

hourly_open, hourly_close = load_hourly_bars(universe, use_cache=True)
print(f"Hourly data: {hourly_open.index[0].date()} to {hourly_open.index[-1].date()}")
print(f"  {len(hourly_open)} bars x {len(hourly_open.columns)} tickers")

# Restrict to backtest period: first SI settlement date to end of hourly data
# Use data only AFTER the first settlement date so universe is always defined
backtest_start = si_timeline[0][0] + pd.Timedelta(days=1)
backtest_end   = hourly_open.index[-1]

mask = (hourly_open.index >= backtest_start) & (hourly_open.index <= backtest_end)
ho   = hourly_open[mask].copy()
hc   = hourly_close[mask].copy()
n    = len(ho)

print(f"\nBacktest window: {ho.index[0].date()} to {ho.index[-1].date()} ({n} hourly bars)")

# Restrict columns to tickers that appear in SI data (have SI coverage)
si_tickers = set(si_raw["ticker"].unique())
common_tickers = [t for t in si_tickers if t in ho.columns]
ho = ho[common_tickers]
hc = hc[common_tickers]
print(f"Tickers with both SI data and hourly bars: {len(common_tickers)}")

# Hourly returns (close/open - 1 for bar return, close-to-close for signal)
# Use close-to-close return for signal (standard); bar return for P&L
bar_ret    = (hc / ho - 1).replace([np.inf, -np.inf], np.nan)  # P&L ret per bar
ctc_ret    = hc.pct_change()                                    # close-to-close for signal


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 -- Precompute rolling stats for each lookback
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 3 -- Precompute rolling statistics (no lookahead)")

print("Computing rolling mean and std (shift(1) applied -- no lookahead)...", flush=True)

# roll_cache[lb] = (mean_shifted, std_shifted)
# shift(1): at bar t, these values are computed from bars [t-lb, t-1]
# -> uses ONLY past data to generate signal at bar t
roll_cache: dict[int, tuple] = {}
for lb in GRID["lookback_hours"]:
    mu  = ctc_ret.rolling(lb).mean().shift(1)
    sig = ctc_ret.rolling(lb).std().shift(1)
    roll_cache[lb] = (mu, sig)
    print(f"  lookback={lb}h: done (warmup bars needed = {lb+1})", flush=True)


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 -- Grid search
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 4 -- Grid search")

total_combos = (len(GRID["lookback_hours"]) * len(GRID["z_grid"]) *
                len(GRID["hold_hours"]) * len(GRID["top_n"]))
print(f"Grid: {len(GRID['lookback_hours'])} lookback x {len(GRID['z_grid'])} z x "
      f"{len(GRID['hold_hours'])} hold x {len(GRID['top_n'])} top_n = {total_combos} combos\n",
      flush=True)

ticker_list = list(ho.columns)
ticker_idx  = {t: i for i, t in enumerate(ticker_list)}

# Precompute universe membership at each bar (per top_n value)
# universe_at_bar[i] = set of tickers eligible at bar i (based on last SI report before bar[i])
print("Precomputing point-in-time universe membership...", flush=True)
bar_timestamps = ho.index

# For each top_n, precompute a boolean matrix [n_bars x n_tickers]
univ_cache: dict[int, np.ndarray] = {}
for tn in GRID["top_n"]:
    mat = np.zeros((n, len(ticker_list)), dtype=bool)
    for i, ts in enumerate(bar_timestamps):
        univ = get_universe_at(ts, tn)
        for t in univ:
            if t in ticker_idx:
                mat[i, ticker_idx[t]] = True
    univ_cache[tn] = mat
    print(f"  top_n={tn}: avg eligible tickers per bar = "
          f"{mat.sum(axis=1).mean():.1f}", flush=True)

# Convert price arrays to numpy for speed
ho_np = ho.values.astype(float)   # [n_bars x n_tickers]
hc_np = hc.values.astype(float)

grid_results = []
best_sharpe  = -np.inf
best_ret     = None
best_params  = None

combo_n = 0
for lb, z, hold, tn in product(
    GRID["lookback_hours"], GRID["z_grid"],
    GRID["hold_hours"],     GRID["top_n"]
):
    combo_n += 1
    if combo_n % 20 == 0:
        print(f"  combo {combo_n}/{total_combos}...", flush=True)

    mu_df, sig_df = roll_cache[lb]
    mu_np  = mu_df.values.astype(float)    # [n_bars x n_tickers]
    sig_np = sig_df.values.astype(float)

    univ_mat = univ_cache[tn]   # [n_bars x n_tickers] bool

    # ctc return numpy
    ctc_np = ctc_ret.values.astype(float)  # [n_bars x n_tickers]

    # Signal matrix: return[t] > mu[t] + z * sig[t]  (mu/sig already shifted)
    # AND ticker in universe at bar t
    # AND mu/sig are valid (not NaN) -- warmup guard
    valid   = np.isfinite(mu_np) & np.isfinite(sig_np) & np.isfinite(ctc_np)
    thresh  = mu_np + z * sig_np
    signal  = (ctc_np > thresh) & univ_mat & valid   # [n_bars x n_tickers]

    # For each signal at bar i: entry at open[i+1], exit at close[min(i+hold, n-1)]
    trade_rets   = []
    busy_until   = np.full(len(ticker_list), -1, dtype=int)  # per-ticker last exit bar

    for i in range(n - hold - 1):
        if not signal[i].any():
            continue

        entry_bar = i + 1
        exit_bar  = min(i + hold, n - 1)

        # Tickers with signal at bar i AND not already in a trade
        sig_tickers = np.where(signal[i])[0]
        eligible    = [j for j in sig_tickers if busy_until[j] < entry_bar]

        if not eligible:
            continue

        ep = ho_np[entry_bar, eligible]
        xp = hc_np[exit_bar,  eligible]

        valid_trade = (ep > 0) & np.isfinite(ep) & np.isfinite(xp)
        if not valid_trade.any():
            continue

        ep = ep[valid_trade]; xp = xp[valid_trade]
        net_rets = (xp / ep - 1) - TC_RATE

        trade_rets.append({
            "entry_bar": entry_bar,
            "entry_dt":  bar_timestamps[entry_bar],
            "avg_ret":   float(net_rets.mean()),
            "n_pos":     int(valid_trade.sum()),
        })

        # Mark tickers as busy until exit
        act = np.array(eligible)[valid_trade]
        busy_until[act] = exit_bar

    if len(trade_rets) < 5:
        grid_results.append(dict(lookback=lb, z=z, hold_hours=hold, top_n=tn,
                                 Sharpe=np.nan, Sortino=np.nan,
                                 Total_Return=np.nan, Max_DD=np.nan,
                                 n_trades=len(trade_rets), Win_Rate=np.nan))
        continue

    tdf = pd.DataFrame(trade_rets)
    tdf["date"] = tdf["entry_dt"].dt.normalize()
    daily = tdf.groupby("date")["avg_ret"].mean()

    all_dates  = pd.date_range(daily.index.min(), bar_timestamps[-1].normalize(), freq="B")
    daily_full = daily.reindex(all_dates, fill_value=0.0)

    wealth = (1 + daily_full).cumprod(); wealth = wealth / wealth.iloc[0]
    sh  = _sharpe(daily_full)
    so  = _sortino(daily_full)
    mdd = float((wealth / wealth.cummax() - 1).min())
    tot = float(wealth.iloc[-1] - 1)
    wr  = float((tdf["avg_ret"] > 0).mean())

    row = dict(lookback=lb, z=z, hold_hours=hold, top_n=tn,
               Sharpe=sh, Sortino=so, Total_Return=tot, Max_DD=mdd,
               n_trades=len(trade_rets), Win_Rate=wr)
    grid_results.append(row)

    if pd.notna(sh) and sh > best_sharpe:
        best_sharpe = sh
        best_ret    = daily_full.rename("ShortSqueeze_v2")
        best_params = row

grid_df = pd.DataFrame(grid_results).sort_values("Sharpe", ascending=False)


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 -- Results
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 5 -- Top 20 grid results")
print(f"{'Lookback':>9} {'Z':>5} {'Hold':>5} {'TopN':>6} | "
      f"{'Sharpe':>7} {'Sortino':>8} {'Return':>9} {'Max_DD':>8} "
      f"{'nTrades':>8} {'WinRate':>8}")
print("-" * 78)
for _, r in grid_df.head(20).iterrows():
    if pd.notna(r["Sharpe"]):
        print(f"{int(r['lookback']):>9} {r['z']:>5.1f} {int(r['hold_hours']):>5} "
              f"{int(r['top_n']):>6} | "
              f"{r['Sharpe']:>7.3f} {r['Sortino']:>8.3f} {r['Total_Return']:>9.2%} "
              f"{r['Max_DD']:>8.2%} {int(r['n_trades']):>8} {r['Win_Rate']:>8.2%}")

if best_params is None:
    print("No valid results -- insufficient signals in backtest window.")
    sys.exit(0)

p = best_params
print(f"\nBest: lookback={p['lookback']}h | z={p['z']} | "
      f"hold={p['hold_hours']}h | top_n={p['top_n']}")
print(f"  Sharpe={p['Sharpe']:.3f}  Sortino={p['Sortino']:.3f}  "
      f"Return={p['Total_Return']:+.1%}  MaxDD={p['Max_DD']:.1%}  "
      f"Trades={p['n_trades']}  WinRate={p['Win_Rate']:.1%}")


# ── Yearly breakdown ────────────────────────────────────────────────────────
_header("Yearly performance")
yr = yearly_table({"ShortSqueeze_v2": best_ret})
print_yearly(yr, ["ShortSqueeze_v2"])


# ── Sharpe heatmap: z x hold_hours (best lookback + top_n) ────────────────
_header("STEP 6 -- Charts")

wealth = (1 + best_ret).cumprod(); wealth = wealth / wealth.iloc[0]
dd     = wealth / wealth.cummax() - 1

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                gridspec_kw={"height_ratios": [3, 1]})
ax1.plot(wealth.index, wealth.values, color="steelblue", linewidth=2)
ax1.set_title(f"Short Squeeze v2 (Point-in-Time SI) | "
              f"lb={p['lookback']}h | z={p['z']} | hold={p['hold_hours']}h | "
              f"top_n={p['top_n']}", fontsize=11)
ax1.set_ylabel("Cumulative Wealth")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}x"))
ax1.grid(True, alpha=0.4)

ax2.fill_between(dd.index, dd.values, 0, alpha=0.5, color="steelblue")
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax2.set_ylabel("Drawdown"); ax2.grid(True, alpha=0.4)
plt.tight_layout(); plt.show()

# Heatmap: z x hold_hours for best lookback + top_n
sub = grid_df[(grid_df["lookback"] == p["lookback"]) & (grid_df["top_n"] == p["top_n"])]
pivot = sub.pivot_table(index="z", columns="hold_hours", values="Sharpe")
if not pivot.empty:
    fig2, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=2)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{h}h" for h in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"z={z}" for z in pivot.index])
    plt.colorbar(im, ax=ax, label="Sharpe")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.2f}" if pd.notna(v) else "N/A",
                    ha="center", va="center", fontsize=10)
    ax.set_title(f"Sharpe: z x hold_hours  (lookback={p['lookback']}h, top_n={p['top_n']})")
    plt.tight_layout(); plt.show()

# Win rate heatmap
pivot_wr = sub.pivot_table(index="z", columns="hold_hours", values="Win_Rate")
if not pivot_wr.empty:
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    im3 = ax3.imshow(pivot_wr.values, aspect="auto", cmap="RdYlGn", vmin=0.3, vmax=0.7)
    ax3.set_xticks(range(len(pivot_wr.columns)))
    ax3.set_xticklabels([f"{h}h" for h in pivot_wr.columns])
    ax3.set_yticks(range(len(pivot_wr.index)))
    ax3.set_yticklabels([f"z={z}" for z in pivot_wr.index])
    plt.colorbar(im3, ax=ax3, label="Win Rate")
    for i in range(len(pivot_wr.index)):
        for j in range(len(pivot_wr.columns)):
            v = pivot_wr.values[i, j]
            ax3.text(j, i, f"{v:.0%}" if pd.notna(v) else "N/A",
                     ha="center", va="center", fontsize=10)
    ax3.set_title(f"Win Rate: z x hold_hours  (lookback={p['lookback']}h, top_n={p['top_n']})")
    plt.tight_layout(); plt.show()


# ── Save to Excel ───────────────────────────────────────────────────────────
_header("STEP 7 -- Save to Excel")
xl = "results/short_squeeze_v2.xlsx"
try:
    with pd.ExcelWriter(xl, engine="openpyxl") as writer:
        pd.DataFrame([{
            "Strategy": "ShortSqueeze_v2",
            "Start": str(best_ret.index[0].date()),
            "End":   str(best_ret.index[-1].date()),
            "Days":  len(best_ret),
            "Total Return": float(wealth.iloc[-1]-1),
            "Sharpe":  _sharpe(best_ret), "Sortino": _sortino(best_ret),
            "Max DD":  _mdd(best_ret),
            **p
        }]).to_excel(writer, sheet_name="Summary", index=False)

        yr.reset_index().to_excel(writer, sheet_name="Yearly", index=False)
        grid_df.to_excel(writer, sheet_name="Grid_Results", index=False)

        pd.DataFrame({"Date": best_ret.index, "Return": best_ret.values}) \
          .to_excel(writer, sheet_name="Daily_Returns", index=False)

        # SI panel summary per settlement date
        (si_raw.sort_values(["settlementDate","daysToCover"], ascending=[True,False])
               .to_excel(writer, sheet_name="SI_Panel", index=False))

    print(f"  Saved: {xl}", flush=True)
except Exception as e:
    print(f"  Excel save failed: {e}", flush=True)


_header("DONE")
print(f"\nBacktest period: {best_ret.index[0].date()} to {best_ret.index[-1].date()}")
print(f"SI settlement dates used: {len(si_timeline)}")
print(f"Universe construction: point-in-time (no lookahead)")
print(f"\nBest params: lb={p['lookback']}h | z={p['z']} | hold={p['hold_hours']}h | top_n={p['top_n']}")
print(f"  Sharpe={p['Sharpe']:.3f}  Return={p['Total_Return']:+.1%}  "
      f"MaxDD={p['Max_DD']:.1%}  WinRate={p['Win_Rate']:.1%}")
print(f"\nNOTE: Backtest covers only the 12-month period where point-in-time SI data is available.")
