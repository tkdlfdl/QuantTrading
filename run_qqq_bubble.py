"""
Runner: QQQ Hourly Bubble Score Strategy
  - QQQ data: Alpaca IEX (2019-2024) + yfinance (2024-2026) merged
  - Cache: data/cache/qqq_hourly.parquet  (rebuilt if >2 days old)
  - Grid: ma_window x z_window x buy_threshold x hold_hours
  - Long-only
"""
import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

from strategies.qqq_bubble_hourly import run_qqq_bubble_hourly

OUT_DIR   = Path("results");            OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("data/cache");         CACHE_DIR.mkdir(exist_ok=True)
QQQ_CACHE = CACHE_DIR / "qqq_hourly.parquet"

ALPACA_START = "2019-01-01"
ALPACA_END   = "2024-06-20"   # hand-off point to yfinance
YF_START     = "2024-06-20"


def _load_alpaca_qqq() -> tuple[pd.Series, pd.Series]:
    """Fetch QQQ hourly open/close from Alpaca IEX (2019-2024)."""
    creds_path = Path("live/state/alpaca_creds.json")
    if not creds_path.exists():
        raise FileNotFoundError("Alpaca credentials not found at live/state/alpaca_creds.json")
    creds = json.loads(creds_path.read_text())
    api_key    = creds["api_key"]
    secret_key = creds["secret_key"]

    from data.fetchers.alpaca_fetcher import fetch_alpaca_bars
    print(f"  Fetching QQQ from Alpaca IEX ({ALPACA_START} -> {ALPACA_END})...")
    ho_df, hc_df = fetch_alpaca_bars(
        tickers    = ["QQQ"],
        start      = ALPACA_START,
        end        = ALPACA_END,
        timeframe  = "1Hour",
        feed       = "iex",
        api_key    = api_key,
        secret_key = secret_key,
    )
    if "QQQ" not in hc_df.columns:
        raise ValueError("Alpaca returned no QQQ data")
    return ho_df["QQQ"], hc_df["QQQ"]


def _load_yf_qqq() -> tuple[pd.Series, pd.Series]:
    """Fetch QQQ hourly open/close from yfinance (last 730d)."""
    start = (datetime.now() - timedelta(days=729)).strftime("%Y-%m-%d")
    print(f"  Fetching QQQ from yfinance ({start} -> now)...")
    raw = yf.download("QQQ", start=start, interval="1h",
                      progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError("yfinance returned no QQQ data")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    return raw["Open"].ffill(), raw["Close"].ffill()


def _build_qqq_cache() -> tuple[pd.Series, pd.Series]:
    """Build merged QQQ hourly series and save to parquet."""
    print("Building QQQ hourly cache (Alpaca 2019-2024 + yfinance 2024-2026)...")

    ho_a, hc_a = _load_alpaca_qqq()
    ho_y, hc_y = _load_yf_qqq()

    # Align to hourly-rounded timestamps, drop tz
    for s in [ho_a, hc_a, ho_y, hc_y]:
        s.index = pd.to_datetime(s.index).tz_localize(None)

    # Merge: Alpaca up to hand-off, yfinance after
    cutoff = pd.Timestamp(ALPACA_END)
    ho = pd.concat([ho_a[ho_a.index < cutoff], ho_y[ho_y.index >= cutoff]]).sort_index()
    hc = pd.concat([hc_a[hc_a.index < cutoff], hc_y[hc_y.index >= cutoff]]).sort_index()

    # Remove duplicates, forward-fill gaps
    ho = ho[~ho.index.duplicated(keep="last")].ffill()
    hc = hc[~hc.index.duplicated(keep="last")].ffill()

    # Save
    merged = pd.DataFrame({"open": ho, "close": hc})
    merged.to_parquet(QQQ_CACHE)
    print(f"  Saved: {QQQ_CACHE}  ({len(hc)} bars, "
          f"{hc.index[0].date()} -> {hc.index[-1].date()})")
    return ho, hc


def _load_qqq() -> tuple[pd.Series, pd.Series]:
    """Load QQQ from cache (rebuild if stale or missing)."""
    rebuild = True
    if QQQ_CACHE.exists():
        age_days = (datetime.now() - datetime.fromtimestamp(QQQ_CACHE.stat().st_mtime)).days
        if age_days <= 3:
            df = pd.read_parquet(QQQ_CACHE)
            ho, hc = df["open"], df["close"]
            print(f"Loaded QQQ cache: {len(hc)} bars  "
                  f"({hc.index[0].date()} -> {hc.index[-1].date()})  [age={age_days}d]")
            rebuild = False
    if rebuild:
        ho, hc = _build_qqq_cache()
    return ho, hc


# ── Load data ───────────────────────────────────────────────────────────────
ho, hc = _load_qqq()
print(f"QQQ: {len(hc)} bars  |  {hc.index[0].date()} -> {hc.index[-1].date()}")

# ── Grid search (long-only) ─────────────────────────────────────────────────
best_ret, best_params, grid_df = run_qqq_bubble_hourly(
    hourly_open  = ho,
    hourly_close = hc,
    ma_window_grid       = [20, 50, 100, 200],
    z_window_grid        = [50, 100, 200],
    buy_threshold_grid   = [0.6, 0.7, 0.8, 0.9],
    short_threshold_grid = [0.95],
    hold_hours_grid      = [4, 8, 13, 24, 52],
    transaction_cost     = 0.001,
    short_borrow_rate    = 0.08,
    enable_short         = False,
)

# ── Top-20 combos ────────────────────────────────────────────────────────────
S = "=" * 80
print(f"\n{S}\nTOP 20 PARAMETER COMBOS (by Sharpe) - QQQ Bubble Long-Only\n{S}")
cols = ["ma_window","z_window","buy_threshold","hold_hours",
        "Sharpe","Sortino","Total_Return","Max_DD","n_trades","Win_Rate"]
print(grid_df[cols].head(20).to_string(index=False))

print(f"\nPositive Sharpe: {(grid_df['Sharpe']>0).sum()}/{len(grid_df)}")
print(f"Sharpe > 1.0:   {(grid_df['Sharpe']>1.0).sum()}/{len(grid_df)}")
print(f"Sharpe > 1.5:   {(grid_df['Sharpe']>1.5).sum()}/{len(grid_df)}")
print(f"Sharpe > 2.0:   {(grid_df['Sharpe']>2.0).sum()}/{len(grid_df)}")

# ── Benchmark ────────────────────────────────────────────────────────────────
bh_ret    = hc.pct_change().dropna()
bh_daily  = bh_ret.resample("B").apply(lambda r: (1+r).prod()-1)
bh_wealth = (1+bh_daily).cumprod(); bh_wealth /= bh_wealth.iloc[0]
bh_sh     = float(np.sqrt(252)*bh_daily.mean()/bh_daily.std())
bh_tot    = float(bh_wealth.iloc[-1]-1)
bh_mdd    = float((bh_wealth/bh_wealth.cummax()-1).min())

print(f"\n{S}\nSTRATEGY vs BENCHMARK\n{S}")
print(f"  {'':30} {'Sharpe':>7} {'Return':>9} {'Max_DD':>8} {'Trades':>7}")
print(f"  {'QQQ Buy-and-Hold':30} {bh_sh:>7.3f} {bh_tot:>+9.1%} {bh_mdd:>8.1%} {'N/A':>7}")
if best_params:
    print(f"  {'QQQ Bubble (best)':30} {best_params['Sharpe']:>7.3f} "
          f"{best_params['Total_Return']:>+9.1%} {best_params['Max_DD']:>8.1%} "
          f"{best_params['n_trades']:>7}")

# ── Yearly breakdown ─────────────────────────────────────────────────────────
print(f"\n{S}\nYEARLY BREAKDOWN (best params)\n{S}")
print(f"  {'Year':>6} {'Return':>9} {'Sharpe':>8} {'Max_DD':>9} {'Sig_Days':>9}")
if best_ret is not None:
    for yr, grp in best_ret.groupby(best_ret.index.year):
        r   = float((1+grp).prod()-1)
        std = grp.std()
        sh  = float(np.sqrt(252)*grp.mean()/std) if std > 0 else np.nan
        w   = (1+grp).cumprod()
        mdd = float((w/w.cummax()-1).min())
        sig = int((grp != 0).sum())
        print(f"  {yr:>6} {r:>+9.2%} {sh:>8.3f} {mdd:>9.2%} {sig:>9}")

# ── Sensitivity ──────────────────────────────────────────────────────────────
print(f"\n{S}\nPARAMETER SENSITIVITY\n{S}")
valid = grid_df[grid_df["Sharpe"].notna()]
for col, label in [("ma_window","MA window (h)"), ("z_window","Z window (h)"),
                   ("buy_threshold","Buy threshold"), ("hold_hours","Hold (h)")]:
    print(f"\n  {label}:")
    for val, grp in valid.groupby(col):
        avg  = grp["Sharpe"].mean()
        best = grp["Sharpe"].max()
        print(f"    {val:<8} avg={avg:.3f}  best={best:.3f}")

# ── Chart ────────────────────────────────────────────────────────────────────
if best_ret is not None and best_params is not None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                              gridspec_kw={"height_ratios": [3, 1.5, 1]})

    strat_w = (1+best_ret).cumprod(); strat_w /= strat_w.iloc[0]
    common  = strat_w.index.intersection(bh_wealth.index)
    axes[0].plot(common, strat_w.loc[common], color="steelblue",
                 linewidth=2, label="QQQ Bubble Strategy")
    axes[0].plot(common, bh_wealth.loc[common], color="gray",
                 linewidth=1.2, linestyle="--", label="QQQ Buy-and-Hold")
    axes[0].set_title(
        f"QQQ Hourly Bubble (Long-Only, 2019-2026) | "
        f"MA={best_params['ma_window']}h  Z={best_params['z_window']}h  "
        f"buy<-{best_params['buy_threshold']}  hold={best_params['hold_hours']}h | "
        f"Sharpe={best_params['Sharpe']:.3f}  "
        f"Return={best_params['Total_Return']:+.1%}  "
        f"Trades={best_params['n_trades']}"
    )
    axes[0].set_ylabel("Cumulative Wealth"); axes[0].legend(); axes[0].grid(True, alpha=0.4)

    from strategies.qqq_bubble_hourly import calculate_bubble_score
    score = calculate_bubble_score(hc, best_params["ma_window"], best_params["z_window"])
    axes[1].plot(score.index, score.values, color="darkorange", linewidth=0.8, alpha=0.8)
    axes[1].axhline(-best_params["buy_threshold"], color="green", linewidth=1.2,
                    linestyle="--", label=f"Buy: score < -{best_params['buy_threshold']}")
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_ylabel("Bubble Score"); axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.4)
    axes[1].set_ylim(-1.1, 1.1)

    dd = strat_w/strat_w.cummax()-1
    axes[2].fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
    axes[2].set_ylabel("Drawdown"); axes[2].grid(True, alpha=0.4)

    plt.tight_layout()
    out = OUT_DIR / "qqq_bubble_hourly.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n[Chart saved: {out}]")

grid_out = OUT_DIR / "qqq_bubble_grid.xlsx"
grid_df.to_excel(grid_out, index=False)
print(f"[Grid saved: {grid_out}]")
