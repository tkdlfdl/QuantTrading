"""
Short Squeeze Screener + Backtest
===================================
Data sources:
  1. Nasdaq screener   -- all NASDAQ-listed stocks
  2. Nasdaq API        -- 12-month bi-weekly short interest (NASDAQ stocks only)
  3. FINRA RegSho      -- today's short sale volume ratio per ticker (all exchanges)
                         shortParQuantity / totalParQuantity = % of day's volume that was short

Workflow:
  Part A  Live screener (no backtest):
    - Fetch today's FINRA RegSho short volume for all stocks
    - Fetch Nasdaq SI for top short-pressure stocks
    - Rank by combined score: short_interest x short_volume_ratio -> prime squeeze list

  Part B  Expanded backtest:
    - Broader NASDAQ universe (all Nasdaq-listed stocks with SI data, not just S&P500/NDX100)
    - Re-run bubble score strategy on expanded universe
    - Compare Sharpe / return vs. old 145-ticker universe

Usage:
  python run_squeeze_screen.py
"""
import sys, warnings, socket, os, math, time
warnings.filterwarnings("ignore")
socket.setdefaulttimeout(8)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_n = [0]
def _save(*a, **k):
    _n[0] += 1; p = f"results/screen_chart_{_n[0]}.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); print(f"  [chart saved: {p}]", flush=True)
plt.show = _save

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from itertools import product

from data.db.schema import init
from data.intraday_loader import load_hourly_bars
from strategies.short_squeeze import run_short_squeeze
from strategies.qqq_bubble_hourly import run_qqq_bubble_hourly  # reuse bubble formula

os.makedirs("results", exist_ok=True)

TRADING_DAYS = 252
TC_RATE      = 0.001
TOP_N_UNIV   = 50    # top-N by DTC for universe at each settlement date
PUB_LAG      = pd.offsets.BusinessDay(8)   # FINRA publishes 8 biz days after settlement date

NAS_H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
          "Accept": "application/json, text/plain, */*",
          "Referer": "https://www.nasdaq.com/"}
FINRA_H = {"User-Agent": "research@example.com"}

GRID = dict(
    ma_window  = [20, 50, 100],
    z_window   = [50, 100, 200],
    threshold  = [0.3, 0.5, 0.7, 0.9],
    hold_hours = [1, 2, 4, 8, 13],
    top_n      = [10, 20, 30],
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _sharpe(r, td=TRADING_DAYS):
    s = r.std(); return float(np.sqrt(td)*r.mean()/s) if s > 0 else np.nan

def _mdd(r):
    w = (1+r).cumprod(); w = w/w.iloc[0]; return float((w/w.cummax()-1).min())

def _header(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}", flush=True)

def _fetch_si(ticker: str, sess: requests.Session) -> list:
    try:
        r = sess.get(
            f"https://api.nasdaq.com/api/quote/{ticker}/short-interest"
            f"?type=COMMON&assetClass=stocks&offset=0&limit=200",
            timeout=8)
        if r.status_code != 200: return []
        rows = (r.json().get("data") or {}).get("shortInterestTable", {}).get("rows", [])
        result = []
        for row in rows:
            try:
                dt  = pd.to_datetime(row["settlementDate"], format="%m/%d/%Y")
                si  = int(str(row["interest"]).replace(",", ""))
                dtc = float(row["daysToCover"])
                result.append({"ticker": ticker, "settlementDate": dt,
                                "sharesShort": si, "daysToCover": dtc})
            except Exception: pass
        return result
    except Exception: return []


# ══════════════════════════════════════════════════════════════════════════
# PART A -- Live Screener
# ══════════════════════════════════════════════════════════════════════════

init()
_header("PART A -- Live Screener: FINRA RegSho + Nasdaq SI")

# ── A1: All NASDAQ-listed stocks ───────────────────────────────────────────
print("\n[A1] Fetching all NASDAQ-listed stocks from Nasdaq screener...", flush=True)
sess = requests.Session(); sess.headers.update(NAS_H)
r = sess.get(
    "https://api.nasdaq.com/api/screener/stocks"
    "?tableonly=true&exchange=NASDAQ&download=true",
    timeout=15)
if r.status_code == 200:
    from io import StringIO
    nas_df = pd.read_csv(StringIO(r.text))
    # clean symbol col
    sym_col = [c for c in nas_df.columns if "symbol" in c.lower() or "ticker" in c.lower()][0]
    nas_tickers = (nas_df[sym_col].dropna().astype(str)
                   .str.strip().str.replace(r"[^\w]","",regex=True)
                   .unique().tolist())
    nas_tickers = [t for t in nas_tickers if t and len(t) <= 5 and t.isalpha()]
    print(f"  {len(nas_tickers)} NASDAQ-listed stocks found")
else:
    print(f"  Screener failed ({r.status_code}) -- using cached universe")
    nas_tickers = []

# ── A2: FINRA RegSho today ─────────────────────────────────────────────────
print("\n[A2] Fetching FINRA RegSho short volume (today)...", flush=True)
r2 = requests.get(
    "https://api.finra.org/data/group/otcmarket/name/regShoDaily?limit=50000",
    headers=FINRA_H, timeout=30)

finra_df = pd.DataFrame()
if r2.status_code == 200:
    from io import StringIO as _SI
    finra_raw = pd.read_csv(_SI(r2.text))
    finra_raw.columns = [c.strip() for c in finra_raw.columns]
    # Aggregate across market centers per ticker
    sym_col = "securitiesInformationProcessorSymbolIdentifier"
    finra_agg = (finra_raw.groupby(sym_col)
                 .agg(shortVol=("shortParQuantity","sum"),
                      totalVol=("totalParQuantity","sum"))
                 .reset_index()
                 .rename(columns={sym_col: "ticker"}))
    finra_agg["shortVolRatio"] = (finra_agg["shortVol"] / finra_agg["totalVol"]).clip(0, 1)
    finra_agg = finra_agg.sort_values("shortVolRatio", ascending=False)
    finra_df  = finra_agg.copy()
    date_col  = finra_raw["tradeReportDate"].iloc[0]
    print(f"  {len(finra_agg)} tickers | date: {date_col}")
    print(f"\n  Top 20 by short volume ratio (today):")
    print(finra_agg.head(20)[["ticker","shortVol","totalVol","shortVolRatio"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
else:
    print(f"  FINRA RegSho failed: {r2.status_code}")

# ── A3: Nasdaq SI for top short-pressure stocks ────────────────────────────
print("\n[A3] Fetching Nasdaq SI for top 100 short-pressure NASDAQ stocks...", flush=True)

_SI_EXPANDED = Path("results/si_panel_expanded.parquet")

# Identify NASDAQ stocks with high short pressure from FINRA
if not finra_df.empty and nas_tickers:
    # Filter to NASDAQ-listed stocks in FINRA data
    nas_set = set(nas_tickers)
    finra_nas = finra_df[finra_df["ticker"].isin(nas_set)].head(150)
    candidates = finra_nas["ticker"].tolist()
    print(f"  NASDAQ stocks in FINRA data: {len(finra_nas)}")
    print(f"  Fetching SI for top {len(candidates)} by short pressure...", flush=True)
else:
    # Fallback: use previously cached tickers
    candidates = []
    print("  Skipping SI fetch (FINRA data unavailable)")

if _SI_EXPANDED.exists():
    si_exp = pd.read_parquet(_SI_EXPANDED)
    existing_tickers = set(si_exp["ticker"].unique())
    new_candidates   = [t for t in candidates if t not in existing_tickers]
    print(f"  Cached: {len(existing_tickers)} tickers. New to fetch: {len(new_candidates)}")
else:
    si_exp = pd.DataFrame()
    new_candidates = candidates

new_rows = []
_CHUNK = 25
for i in range(0, len(new_candidates), _CHUNK):
    batch = new_candidates[i:i+_CHUNK]
    print(f"  batch {i//_CHUNK+1}/{math.ceil(len(new_candidates)/_CHUNK)} ({len(batch)})...",
          end=" ", flush=True)
    for t in batch:
        new_rows.extend(_fetch_si(t, sess))
    print(f"ok ({len(new_rows)} rows so far)", flush=True)

if new_rows:
    si_new = pd.DataFrame(new_rows)
    si_exp  = pd.concat([si_exp, si_new], ignore_index=True).drop_duplicates(
        subset=["ticker","settlementDate"])
    si_exp.to_parquet(_SI_EXPANDED)
    print(f"  Saved expanded SI panel: {len(si_exp)} rows, {si_exp['ticker'].nunique()} tickers")

# ── A4: Combined screener ranking ─────────────────────────────────────────
_header("A4 -- Combined Screener: High SI + High Short Pressure (TODAY)")

if not si_exp.empty and not finra_df.empty:
    # Most recent Nasdaq SI settlement
    latest_sd  = si_exp["settlementDate"].max()
    latest_si  = si_exp[si_exp["settlementDate"] == latest_sd].copy()

    # Merge with FINRA short pressure
    screen = latest_si.merge(finra_df[["ticker","shortVolRatio","shortVol","totalVol"]],
                              on="ticker", how="left")
    screen["shortVolRatio"] = screen["shortVolRatio"].fillna(0)

    # Normalize ranks (higher = worse squeeze potential, so invert)
    screen["si_rank"]  = screen["daysToCover"].rank(ascending=False)
    screen["svr_rank"] = screen["shortVolRatio"].rank(ascending=False)
    screen["combo_rank"] = screen["si_rank"] + screen["svr_rank"]

    screen = screen.sort_values("combo_rank")
    top_screen = screen.head(30)

    print(f"\nSettlement date: {latest_sd.date()}")
    print(f"Combined ranking: Days-to-Cover rank + Short-Volume-Ratio rank (lower = better candidate)\n")
    print(f"{'Ticker':>8} {'DTC':>8} {'ShortVolRatio':>14} {'SharesShort':>14} {'ComboRank':>10}")
    print("-" * 58)
    for _, row in top_screen.iterrows():
        print(f"{row['ticker']:>8} {row['daysToCover']:>8.2f} {row['shortVolRatio']:>14.3f} "
              f"{int(row['sharesShort'] or 0):>14,} {row['combo_rank']:>10.0f}")

    # Chart: SI vs Short Volume Ratio
    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(screen["daysToCover"], screen["shortVolRatio"]*100,
                    c=screen["combo_rank"], cmap="RdYlGn_r",
                    s=80, alpha=0.7, edgecolors="white", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="Combined Rank (lower = better)")

    # Label top 15
    for _, row in top_screen.head(15).iterrows():
        ax.annotate(row["ticker"],
                    (row["daysToCover"], row["shortVolRatio"]*100),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("Days to Cover (Nasdaq SI)", fontsize=11)
    ax.set_ylabel("Short Volume Ratio % (FINRA RegSho)", fontsize=11)
    ax.set_title(f"Squeeze Candidate Screener\n"
                 f"SI date: {latest_sd.date()}  |  Short volume date: {date_col if not finra_df.empty else 'N/A'}",
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    # Save screener output
    screen.sort_values("combo_rank").to_csv("results/squeeze_screener.csv", index=False)
    print(f"\nScreener saved: results/squeeze_screener.csv")
else:
    print("Insufficient data for combined screener.")
    top_screen = pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════
# PART B -- Expanded Backtest: broader NASDAQ universe
# ══════════════════════════════════════════════════════════════════════════

_header("PART B -- Backtest with Expanded NASDAQ Universe")

# Load SI panel (expanded + original combined)
_SI_ORIG = Path("results/si_panel_12m.parquet")
si_panels = []
for p in [_SI_ORIG, _SI_EXPANDED]:
    if p.exists():
        si_panels.append(pd.read_parquet(p))
if si_panels:
    si_all = pd.concat(si_panels, ignore_index=True).drop_duplicates(
        subset=["ticker","settlementDate"])
else:
    print("No SI data available."); sys.exit(0)

print(f"Combined SI panel: {si_all['ticker'].nunique()} tickers | "
      f"{si_all['settlementDate'].nunique()} settlement dates")
print(f"  Date range: {si_all['settlementDate'].min().date()} to "
      f"{si_all['settlementDate'].max().date()}")

# Build point-in-time timeline
settlement_dates = sorted(si_all["settlementDate"].unique())
si_timeline = []
for sd in settlement_dates:
    period = si_all[si_all["settlementDate"]==sd].sort_values("daysToCover", ascending=False)
    si_timeline.append((sd, dict(zip(period["ticker"], period["daysToCover"]))))

def get_universe_at(ts: pd.Timestamp, n: int) -> set:
    eligible = [(sd, d) for sd, d in si_timeline if sd + PUB_LAG < ts]
    if not eligible: return set()
    _, top_dict = eligible[-1]
    ranked = sorted(top_dict.items(), key=lambda x: x[1], reverse=True)[:n]
    return {t for t, _ in ranked}

# Load hourly data
from data.universe import get_universe
universe = get_universe()
hourly_open, hourly_close = load_hourly_bars(universe, use_cache=True)

backtest_start = si_timeline[0][0] + pd.Timedelta(days=1)
mask = hourly_open.index >= backtest_start
ho   = hourly_open[mask]; hc = hourly_close[mask]

si_tickers = set(si_all["ticker"].unique())
# Include tickers that are in BOTH hourly cache AND SI panel
common_tickers = [t for t in si_tickers if t in ho.columns]
# Also try to find new tickers from expanded SI (may not be in merged cache)
new_si_tickers = set(si_exp["ticker"].unique()) if not si_exp.empty else set()
available_new  = [t for t in new_si_tickers if t in ho.columns]
print(f"\nTickers with SI data in hourly cache: {len(common_tickers)}")
print(f"  From original universe (S&P500/NDX100): {len([t for t in common_tickers if t not in new_si_tickers])}")
print(f"  New from expanded NASDAQ search: {len([t for t in common_tickers if t in new_si_tickers])}")

ho = ho[common_tickers]; hc = hc[common_tickers]
n  = len(ho)

# ── Bubble score computation (same as run_short_squeeze_bubble.py) ─────────
print(f"\nComputing bubble scores for {len(common_tickers)} tickers...", flush=True)
ho_full = hourly_open[common_tickers]
hc_full = hourly_close[common_tickers]

def compute_bubble(close_df, ma_w, z_w):
    log_c = np.log(close_df.replace(0,np.nan).ffill())
    fv    = close_df.rolling(ma_w).mean()
    res   = log_c - np.log(fv)
    z     = (res - res.rolling(z_w).mean()) / res.rolling(z_w).std()
    return np.tanh(z/2).shift(1)

ma_z_pairs = list({(ma,zw) for ma,zw,_,_,_ in product(
    GRID["ma_window"],GRID["z_window"],GRID["threshold"],GRID["hold_hours"],GRID["top_n"])})

bubble_cache = {}
for ma, zw in ma_z_pairs:
    scores = compute_bubble(hc_full, ma, zw)
    bubble_cache[(ma,zw)] = scores.loc[ho.index].copy()
print(f"  Computed {len(bubble_cache)} (ma,z_window) combinations")

# ── Precompute universe membership ─────────────────────────────────────────
ticker_list = list(common_tickers); ticker_idx = {t:i for i,t in enumerate(ticker_list)}
bar_ts      = ho.index
univ_cache  = {}
for tn in GRID["top_n"]:
    mat = np.zeros((n,len(ticker_list)),dtype=bool)
    for i, ts in enumerate(bar_ts):
        for t in get_universe_at(ts, tn):
            if t in ticker_idx: mat[i,ticker_idx[t]] = True
    univ_cache[tn] = mat
    print(f"  top_n={tn}: avg {mat.sum(axis=1).mean():.1f} tickers/bar", flush=True)

ho_np = ho.values.astype(float); hc_np = hc.values.astype(float)

# ── Grid search ────────────────────────────────────────────────────────────
_header("Grid search: bubble score < -threshold on expanded NASDAQ SI universe")

total = (len(GRID["ma_window"])*len(GRID["z_window"])*len(GRID["threshold"])
         *len(GRID["hold_hours"])*len(GRID["top_n"]))
print(f"Grid: {len(GRID['ma_window'])} ma x {len(GRID['z_window'])} z_win x "
      f"{len(GRID['threshold'])} thresh x {len(GRID['hold_hours'])} hold x "
      f"{len(GRID['top_n'])} top_n = {total} combos\n", flush=True)

grid_results = []; best_sharpe = -np.inf; best_ret = None; best_params = None
combo_n = 0

for ma, zw, thresh, hold, tn in product(
    GRID["ma_window"],GRID["z_window"],GRID["threshold"],GRID["hold_hours"],GRID["top_n"]):
    combo_n += 1
    if combo_n % 60 == 0: print(f"  combo {combo_n}/{total}...", flush=True)

    scores_np = bubble_cache[(ma,zw)].values.astype(float)
    univ_mat  = univ_cache[tn]
    signal    = (scores_np < -thresh) & univ_mat & np.isfinite(scores_np)

    trade_rets = []; busy = np.full(len(ticker_list),-1,dtype=int)
    for i in range(n-hold-1):
        if not signal[i].any(): continue
        eb = i+1; xb = min(i+hold, n-1)
        sigs = np.where(signal[i])[0]
        elig = [j for j in sigs if busy[j] < eb]
        if not elig: continue
        ep = ho_np[eb,elig]; xp = hc_np[xb,elig]
        valid = (ep>0) & np.isfinite(ep) & np.isfinite(xp)
        if not valid.any(): continue
        net = (xp[valid]/ep[valid]-1) - TC_RATE
        trade_rets.append({"entry_dt":bar_ts[eb],"avg_ret":float(net.mean()),
                           "n_pos":int(valid.sum()),"score":float(scores_np[i][np.array(elig)[valid]].mean())})
        busy[np.array(elig)[valid]] = xb

    if len(trade_rets) < 5:
        grid_results.append(dict(ma=ma,z_window=zw,threshold=thresh,hold_hours=hold,top_n=tn,
                                 Sharpe=np.nan,Sortino=np.nan,Total_Return=np.nan,
                                 Max_DD=np.nan,n_trades=len(trade_rets),Win_Rate=np.nan))
        continue

    tdf    = pd.DataFrame(trade_rets)
    tdf["dt"] = tdf["entry_dt"].dt.normalize()
    daily  = tdf.groupby("dt")["avg_ret"].mean()
    dates  = pd.date_range(daily.index.min(), bar_ts[-1].normalize(), freq="B")
    daily  = daily.reindex(dates, fill_value=0.0)

    w   = (1+daily).cumprod(); w = w/w.iloc[0]
    sh  = _sharpe(daily); mdd = float((w/w.cummax()-1).min()); tot = float(w.iloc[-1]-1)
    so_d= daily[daily<0].std(); so = float(np.sqrt(TRADING_DAYS)*daily.mean()/so_d) if so_d>0 else np.nan
    wr  = float((tdf["avg_ret"]>0).mean())

    row = dict(ma=ma,z_window=zw,threshold=thresh,hold_hours=hold,top_n=tn,
               Sharpe=sh,Sortino=so,Total_Return=tot,Max_DD=mdd,
               n_trades=len(trade_rets),Win_Rate=wr,Avg_Score=float(tdf["score"].mean()))
    grid_results.append(row)
    if pd.notna(sh) and sh > best_sharpe:
        best_sharpe = sh; best_ret = daily.rename("SqueezeBubble_Exp"); best_params = row

grid_df = pd.DataFrame(grid_results).sort_values("Sharpe",ascending=False)

_header("Top 20 results")
print(f"{'MA':>5} {'ZWin':>5} {'Thresh':>7} {'Hold':>5} {'TopN':>5} | "
      f"{'Sharpe':>7} {'Sortino':>8} {'Return':>9} {'Max_DD':>8} {'nTrades':>8} {'WinRate':>8}")
print("-"*90)
for _, r in grid_df.head(20).iterrows():
    if pd.notna(r["Sharpe"]):
        print(f"{int(r['ma']):>5} {int(r['z_window']):>5} {r['threshold']:>7.1f} "
              f"{int(r['hold_hours']):>5} {int(r['top_n']):>5} | "
              f"{r['Sharpe']:>7.3f} {r['Sortino']:>8.3f} {r['Total_Return']:>9.2%} "
              f"{r['Max_DD']:>8.2%} {int(r['n_trades']):>8} {r['Win_Rate']:>8.2%}")

if best_params is None:
    print("No valid results."); sys.exit(0)

p = best_params
print(f"\nBest: ma={p['ma']}h | z_win={p['z_window']}h | thresh={p['threshold']} | "
      f"hold={p['hold_hours']}h | top_n={p['top_n']}")
print(f"  Sharpe={p['Sharpe']:.3f}  Return={p['Total_Return']:+.1%}  "
      f"MaxDD={p['Max_DD']:.1%}  WinRate={p['Win_Rate']:.1%}")

# ── Compare with original 145-ticker backtest ──────────────────────────────
_header("Comparison: Expanded vs Original universe")

# Load original backtest results
orig_xl = pd.ExcelFile("results/short_squeeze_bubble.xlsx")
orig_sum = pd.read_excel(orig_xl,"Summary")
print("\nOriginal (145 NASDAQ tickers from S&P500/NDX100):")
for _, r in orig_sum.iterrows():
    print(f"  Sharpe={r.get('Sharpe',r.get('Sharpe Ratio','N/A')):.3f}  "
          f"Return={float(str(r.get('Total Return','0')).replace('%',''))/100 if '%' in str(r.get('Total Return','')) else r.get('Total Return',0):+.1%}  "
          f"MaxDD={float(str(r.get('Max DD','0')).replace('%',''))/100 if '%' in str(r.get('Max DD','')) else r.get('Max DD',0):.1%}")

print(f"\nExpanded ({len(common_tickers)} tickers incl. broader NASDAQ):")
print(f"  Sharpe={p['Sharpe']:.3f}  Return={p['Total_Return']:+.1%}  MaxDD={p['Max_DD']:.1%}")

# Yearly
_header("Yearly performance (expanded universe)")
for yr, grp in best_ret.groupby(best_ret.index.year):
    w   = (1+grp).cumprod(); w = w/w.iloc[0]
    ret = float(w.iloc[-1]-1); mdd = float((w/w.cummax()-1).min())
    sh  = _sharpe(grp)
    print(f"  {yr}: Return={ret:>+8.2%}  Sharpe={sh:>6.3f}  MaxDD={mdd:>8.2%}  Days={len(grp)}")

# ── Charts ─────────────────────────────────────────────────────────────────
_header("Charts")

wealth = (1+best_ret).cumprod(); wealth = wealth/wealth.iloc[0]
dd     = wealth/wealth.cummax()-1

fig, (ax1,ax2) = plt.subplots(2,1,figsize=(16,10),gridspec_kw={"height_ratios":[3,1]})
ax1.plot(wealth.index, wealth.values, color="steelblue", linewidth=2)
ax1.set_title(f"Expanded NASDAQ Squeeze Bubble | ma={p['ma']}h z_win={p['z_window']}h "
              f"thresh={p['threshold']} hold={p['hold_hours']}h top_n={p['top_n']}", fontsize=11)
ax1.set_ylabel("Cumulative Wealth")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.2f}x"))
ax1.grid(True,alpha=0.4)

ax2.fill_between(dd.index, dd.values, 0, alpha=0.5, color="steelblue")
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax2.set_ylabel("Drawdown"); ax2.grid(True,alpha=0.4)
plt.tight_layout(); plt.show()

# Sharpe heatmap: threshold x hold_hours
sub = grid_df[(grid_df["ma"]==p["ma"]) &
              (grid_df["z_window"]==p["z_window"]) &
              (grid_df["top_n"]==p["top_n"])]
pivot = sub.pivot_table(index="threshold",columns="hold_hours",values="Sharpe")
if not pivot.empty:
    fig2, ax = plt.subplots(figsize=(10,6))
    im = ax.imshow(pivot.values,aspect="auto",cmap="RdYlGn",vmin=-1,vmax=3)
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels([f"{h}h" for h in pivot.columns])
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels([f"t={t}" for t in pivot.index])
    plt.colorbar(im,ax=ax,label="Sharpe")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i,j]
            ax.text(j,i,f"{v:.2f}" if pd.notna(v) else "N/A",ha="center",va="center",fontsize=10)
    ax.set_title(f"Sharpe: threshold x hold_hours (ma={p['ma']}, z_win={p['z_window']}, top_n={p['top_n']})")
    plt.tight_layout(); plt.show()

# ── Save ───────────────────────────────────────────────────────────────────
_header("Save to Excel")
xl = "results/squeeze_screen_backtest.xlsx"
try:
    with pd.ExcelWriter(xl,engine="openpyxl") as writer:
        pd.DataFrame([{
            "Universe": f"{len(common_tickers)} NASDAQ tickers",
            "Start": str(best_ret.index[0].date()), "End": str(best_ret.index[-1].date()),
            "Days": len(best_ret), "Total Return": p["Total_Return"],
            "Sharpe": p["Sharpe"], "Max DD": p["Max_DD"],
            **{k:p[k] for k in ["ma","z_window","threshold","hold_hours","top_n","n_trades","Win_Rate"]},
        }]).to_excel(writer,sheet_name="Summary",index=False)

        grid_df.to_excel(writer,sheet_name="Grid_Results",index=False)
        pd.DataFrame({"Date":best_ret.index,"Return":best_ret.values}).to_excel(
            writer,sheet_name="Daily_Returns",index=False)

        if not top_screen.empty:
            top_screen.to_excel(writer,sheet_name="Live_Screener",index=False)
        if not finra_df.empty:
            finra_df.head(100).to_excel(writer,sheet_name="FINRA_RegSho_Today",index=False)
    print(f"  Saved: {xl}", flush=True)
except Exception as e:
    print(f"  Save failed: {e}", flush=True)

_header("DONE")
print(f"\nPart A -- Live Screener:")
print(f"  NASDAQ stocks fetched: {len(nas_tickers) if nas_tickers else 'N/A'}")
print(f"  FINRA RegSho tickers: {len(finra_df) if not finra_df.empty else 'N/A'}")
print(f"  Screener output: results/squeeze_screener.csv")
print(f"\nPart B -- Expanded Backtest:")
print(f"  Universe: {len(common_tickers)} tickers")
print(f"  Best params: ma={p['ma']} z_win={p['z_window']} thresh={p['threshold']} hold={p['hold_hours']}h top_n={p['top_n']}")
print(f"  Sharpe={p['Sharpe']:.3f}  Return={p['Total_Return']:+.1%}  MaxDD={p['Max_DD']:.1%}")
