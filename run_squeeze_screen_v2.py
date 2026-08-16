"""
Short Squeeze Live Screener + Expanded Backtest
=================================================
Part A  -- Live Screener (today):
  1. Nasdaq screener  -> 4000+ NASDAQ-listed tickers
  2. FINRA RegSho     -> short sale volume ratio per ticker (today, all exchanges)
  3. Nasdaq SI API    -> 12-month bi-weekly short interest for top short-pressure NASDAQ stocks
  Combined rank: High DTC (days-to-cover) + High short volume ratio = prime squeeze candidates

Part B  -- Expanded Backtest:
  Bubble score strategy on expanded NASDAQ SI universe
  (S&P500/NDX100 145 tickers + new tickers found via FINRA pressure filter)

Usage:
  python run_squeeze_screen_v2.py
"""
import sys, warnings, socket, os, math
warnings.filterwarnings("ignore")
socket.setdefaulttimeout(8)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_n = [0]
def _save(*a, **k):
    _n[0] += 1; p = f"results/screen2_chart_{_n[0]}.png"
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
from data.universe import get_universe

os.makedirs("results", exist_ok=True)

TRADING_DAYS = 252
TC_RATE      = 0.001
TOP_N_UNIV   = 50
PUB_LAG      = pd.offsets.BusinessDay(8)

NAS_H   = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.nasdaq.com/"}
FINRA_H = {"User-Agent": "research@example.com"}


def _sharpe(r, td=TRADING_DAYS):
    s = r.std(); return float(np.sqrt(td)*r.mean()/s) if s > 0 else np.nan

def _sortino(r, td=TRADING_DAYS):
    ds = r[r<0].std(); return float(np.sqrt(td)*r.mean()/ds) if ds > 0 else np.nan

def _mdd(r):
    w = (1+r).cumprod(); w = w/w.iloc[0]; return float((w/w.cummax()-1).min())

def _header(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}", flush=True)

def _fetch_si(ticker: str, sess: requests.Session) -> list:
    try:
        r = sess.get(
            f"https://api.nasdaq.com/api/quote/{ticker}/short-interest"
            f"?type=COMMON&assetClass=stocks&offset=0&limit=200", timeout=8)
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
_header("PART A -- Live Screener")
sess = requests.Session(); sess.headers.update(NAS_H)

# ── A1: All NASDAQ-listed tickers ──────────────────────────────────────────
print("\n[A1] Nasdaq screener: all listed stocks...", flush=True)
r = sess.get(
    "https://api.nasdaq.com/api/screener/stocks"
    "?tableonly=true&exchange=NASDAQ&download=true", timeout=15)
d = r.json()
rows_nasdaq = d.get("data", {}).get("rows", [])
nas_tickers = [row["symbol"].strip() for row in rows_nasdaq
               if row.get("symbol") and str(row["symbol"]).isalpha() and len(row["symbol"]) <= 5]
print(f"  {len(nas_tickers)} NASDAQ-listed tickers found")

# Build a quick market cap lookup for ranking later
nas_mcap = {}
for row in rows_nasdaq:
    sym = row.get("symbol","").strip()
    try: nas_mcap[sym] = float(str(row.get("marketCap","0")).replace(",","")) or 0
    except: nas_mcap[sym] = 0

# ── A2: FINRA RegSho today ─────────────────────────────────────────────────
print("\n[A2] FINRA RegSho: today's short volume ratio...", flush=True)
r2 = requests.get(
    "https://api.finra.org/data/group/otcmarket/name/regShoDaily?limit=50000",
    headers=FINRA_H, timeout=30)
finra_df = pd.DataFrame()
if r2.status_code == 200:
    from io import StringIO
    finra_raw = pd.read_csv(StringIO(r2.text))
    sym_col   = "securitiesInformationProcessorSymbolIdentifier"
    date_val  = finra_raw["tradeReportDate"].iloc[0]
    finra_agg = (finra_raw.groupby(sym_col)
                 .agg(shortVol=("shortParQuantity","sum"),
                      totalVol=("totalParQuantity","sum"))
                 .reset_index().rename(columns={sym_col:"ticker"}))
    finra_agg["shortVolRatio"] = (finra_agg["shortVol"] / finra_agg["totalVol"]).clip(0, 1)
    # Filter: minimum volume threshold to avoid noise (at least 1000 shares total)
    finra_agg = finra_agg[finra_agg["totalVol"] >= 1000].copy()
    finra_agg = finra_agg.sort_values("shortVolRatio", ascending=False)
    finra_df  = finra_agg.copy()
    print(f"  {len(finra_agg)} tickers (totalVol >= 1000) | date: {date_val}")

    # Filter to NASDAQ-listed stocks
    nas_set = set(nas_tickers)
    finra_nas = finra_agg[finra_agg["ticker"].isin(nas_set)].copy()
    finra_nas["marketCap"] = finra_nas["ticker"].map(nas_mcap).fillna(0)
    # Filter to stocks with meaningful market cap (>$100M) for better quality
    finra_nas_sig = finra_nas[finra_nas["marketCap"] > 1e8].sort_values("shortVolRatio", ascending=False)
    print(f"\n  Top 30 NASDAQ stocks by short volume ratio today (mktcap>$100M):")
    print(f"  {'Ticker':>8} {'ShortVolRatio':>14} {'ShortVol':>12} {'TotalVol':>12} {'MktCap$B':>10}")
    print("  " + "-"*60)
    for _, row in finra_nas_sig.head(30).iterrows():
        print(f"  {row['ticker']:>8} {row['shortVolRatio']:>14.3f} "
              f"{int(row['shortVol']):>12,} {int(row['totalVol']):>12,} "
              f"{row['marketCap']/1e9:>10.2f}")
else:
    print(f"  FINRA RegSho failed: {r2.status_code}")
    finra_nas_sig = pd.DataFrame()
    date_val = "N/A"

# ── A3: Nasdaq SI for top short-pressure NASDAQ stocks ────────────────────
_SI_EXPANDED = Path("results/si_panel_expanded.parquet")
SI_FETCH_N   = 80    # fetch SI for top-80 NASDAQ stocks by short pressure

si_exp = pd.read_parquet(_SI_EXPANDED) if _SI_EXPANDED.exists() else pd.DataFrame()
existing = set(si_exp["ticker"].unique()) if not si_exp.empty else set()

# Load original SI panel
_SI_ORIG = Path("results/si_panel_12m.parquet")
si_orig  = pd.read_parquet(_SI_ORIG) if _SI_ORIG.exists() else pd.DataFrame()
orig_tickers = set(si_orig["ticker"].unique()) if not si_orig.empty else set()

# Which new NASDAQ tickers to fetch SI for
if not finra_nas_sig.empty:
    top_pressure_tickers = finra_nas_sig["ticker"].head(SI_FETCH_N).tolist()
    new_to_fetch = [t for t in top_pressure_tickers
                    if t not in existing and t not in orig_tickers]
    print(f"\n[A3] Fetching Nasdaq SI for {len(new_to_fetch)} new high-pressure tickers...", flush=True)

    new_rows = []
    _CHUNK = 20
    for i in range(0, len(new_to_fetch), _CHUNK):
        batch = new_to_fetch[i:i+_CHUNK]
        print(f"  batch {i//_CHUNK+1}/{math.ceil(len(new_to_fetch)/_CHUNK)} ({len(batch)})...",
              end=" ", flush=True)
        for t in batch:
            new_rows.extend(_fetch_si(t, sess))
        print(f"ok ({len(new_rows)} rows)", flush=True)

    if new_rows:
        si_new = pd.DataFrame(new_rows)
        si_exp = pd.concat([si_exp, si_new], ignore_index=True).drop_duplicates(
            subset=["ticker","settlementDate"])
        si_exp.to_parquet(_SI_EXPANDED)
        print(f"  Saved: {si_exp['ticker'].nunique()} tickers in expanded panel")
    else:
        print("  No new SI data returned (all tickers may be NYSE-listed)")
else:
    print("\n[A3] Skipping SI fetch (FINRA data unavailable)")

# ── A4: Combined Screener ──────────────────────────────────────────────────
_header("A4 -- Combined Screener: Short Interest x Short Volume Pressure")

# Combine: original SI + expanded SI
all_si = pd.concat([p for p in [si_orig, si_exp] if not p.empty],
                   ignore_index=True).drop_duplicates(subset=["ticker","settlementDate"])
latest_sd = all_si["settlementDate"].max()
latest_si = (all_si[all_si["settlementDate"] == latest_sd]
             .sort_values("daysToCover", ascending=False))

if not finra_df.empty and not latest_si.empty:
    screen = latest_si.merge(finra_df[["ticker","shortVolRatio","shortVol","totalVol"]],
                              on="ticker", how="left")
    screen["shortVolRatio"] = screen["shortVolRatio"].fillna(0)
    screen["marketCap"]     = screen["ticker"].map(nas_mcap).fillna(0)

    # Percentile ranks (0=best squeeze candidate)
    screen["dtc_pct"] = screen["daysToCover"].rank(pct=True, ascending=True)
    screen["svr_pct"] = screen["shortVolRatio"].rank(pct=True, ascending=True)
    screen["combo"]   = 1 - (screen["dtc_pct"] + screen["svr_pct"]) / 2  # higher = better
    screen = screen.sort_values("combo", ascending=False)

    print(f"\nSI settlement: {latest_sd.date()}  |  Short volume date: {date_val}")
    print(f"Combined score = avg(DTC percentile, ShortVolRatio percentile)\n")
    print(f"{'Rank':>5} {'Ticker':>8} {'DTC':>7} {'ShortVolRatio':>14} {'Score':>7} {'MktCap$B':>10}")
    print("-" * 57)
    for rank, (_, row) in enumerate(screen.head(25).iterrows(), 1):
        mc = row.get("marketCap", 0)
        print(f"{rank:>5} {row['ticker']:>8} {row['daysToCover']:>7.2f} "
              f"{row['shortVolRatio']:>14.3f} {row['combo']:>7.3f} "
              f"{mc/1e9:>10.2f}")

    screen.head(50).to_csv("results/squeeze_screener_v2.csv", index=False)
    print(f"\nScreener saved: results/squeeze_screener_v2.csv")

    # Chart: DTC vs Short Volume Ratio (bubble size = market cap)
    fig, ax = plt.subplots(figsize=(14, 9))
    mc_scaled = (screen["marketCap"].fillna(0) / 1e9).clip(0.1, 500)
    sc = ax.scatter(screen["daysToCover"], screen["shortVolRatio"]*100,
                    c=screen["combo"], cmap="RdYlGn",
                    s=(mc_scaled**0.4)*40, alpha=0.7,
                    edgecolors="white", linewidths=0.5, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="Combined Score (green=best squeeze candidate)")
    for _, row in screen.head(20).iterrows():
        ax.annotate(row["ticker"],
                    (row["daysToCover"], row["shortVolRatio"]*100),
                    fontsize=8, ha="left", va="bottom",
                    xytext=(4, 3), textcoords="offset points",
                    fontweight="bold" if row["combo"] > 0.7 else "normal")
    ax.set_xlabel("Days to Cover (Nasdaq SI, settlement: {})".format(latest_sd.date()), fontsize=11)
    ax.set_ylabel("Short Volume Ratio % (FINRA RegSho, date: {})".format(date_val), fontsize=11)
    ax.set_title("Short Squeeze Screener: High SI x High Short Selling Pressure\n"
                 "(bubble size = market cap, color = combined score)", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


# ══════════════════════════════════════════════════════════════════════════
# PART B -- Expanded Backtest
# ══════════════════════════════════════════════════════════════════════════

_header("PART B -- Expanded Backtest (original + new tickers)")

si_all = pd.concat([p for p in [si_orig, si_exp] if not p.empty],
                   ignore_index=True).drop_duplicates(subset=["ticker","settlementDate"])

print(f"Combined SI universe: {si_all['ticker'].nunique()} tickers | "
      f"{si_all['settlementDate'].nunique()} settlement dates", flush=True)

universe = get_universe()
hourly_open, hourly_close = load_hourly_bars(universe, use_cache=True)

backtest_start = sorted(si_all["settlementDate"].unique())[0] + pd.Timedelta(days=1)
mask  = hourly_open.index >= backtest_start
ho    = hourly_open[mask]; hc = hourly_close[mask]

si_tickers     = set(si_all["ticker"].unique())
common_tickers = [t for t in si_tickers if t in ho.columns]
ho = ho[common_tickers]; hc = hc[common_tickers]; n = len(ho)

# Show how many are new vs original
new_tickers = [t for t in common_tickers if t not in orig_tickers]
print(f"\nTickers in hourly cache: {len(common_tickers)}")
print(f"  Original S&P500/NDX100: {len(common_tickers) - len(new_tickers)}")
print(f"  NEW from NASDAQ pressure screen: {len(new_tickers)}")
if new_tickers:
    print(f"  New tickers: {new_tickers}")

# Build SI timeline
settlement_dates = sorted(si_all["settlementDate"].unique())
si_timeline = []
for sd in settlement_dates:
    period = si_all[si_all["settlementDate"]==sd].sort_values("daysToCover",ascending=False)
    si_timeline.append((sd, dict(zip(period["ticker"], period["daysToCover"]))))

def get_universe_at(ts, n_stocks):
    eligible = [(sd,d) for sd,d in si_timeline if sd+PUB_LAG < ts]
    if not eligible: return set()
    _, top_dict = eligible[-1]
    return {t for t,_ in sorted(top_dict.items(),key=lambda x:x[1],reverse=True)[:n_stocks]}

# Use best params from previous run (no need to re-grid if no new tickers)
BEST = dict(ma=50, z_window=100, threshold=0.5, hold_hours=13, top_n=20)

if not new_tickers:
    print("\nNo new tickers added to hourly cache -- backtest unchanged from previous run.")
    print(f"Best params remain: ma={BEST['ma']}h z_win={BEST['z_window']}h "
          f"thresh={BEST['threshold']} hold={BEST['hold_hours']}h top_n={BEST['top_n']}")
    print(f"Sharpe=2.086  Return=+86.0%  MaxDD=-13.3%")
else:
    # Run quick grid with best known params + nearby variants
    GRID_QUICK = dict(
        ma_window  = [50, 100],
        z_window   = [100, 50],
        threshold  = [0.5, 0.7],
        hold_hours = [8, 13],
        top_n      = [20, 30],
    )

    ho_full = hourly_open[common_tickers]; hc_full = hourly_close[common_tickers]
    bubble_cache = {}
    ma_z_pairs = list({(ma,zw) for ma,zw,_,_,_ in product(
        GRID_QUICK["ma_window"],GRID_QUICK["z_window"],GRID_QUICK["threshold"],
        GRID_QUICK["hold_hours"],GRID_QUICK["top_n"])})
    for ma, zw in ma_z_pairs:
        log_c  = np.log(hc_full.replace(0,np.nan).ffill())
        fv     = hc_full.rolling(ma).mean()
        res    = log_c - np.log(fv)
        z      = (res - res.rolling(zw).mean()) / res.rolling(zw).std()
        bubble_cache[(ma,zw)] = np.tanh(z/2).shift(1).loc[ho.index].copy()

    ticker_list = list(common_tickers); ticker_idx = {t:i for i,t in enumerate(ticker_list)}
    bar_ts      = ho.index
    univ_cache  = {}
    for tn in GRID_QUICK["top_n"]:
        mat = np.zeros((n,len(ticker_list)),dtype=bool)
        for i,ts in enumerate(bar_ts):
            for t in get_universe_at(ts,tn):
                if t in ticker_idx: mat[i,ticker_idx[t]] = True
        univ_cache[tn] = mat

    ho_np = ho.values.astype(float); hc_np = hc.values.astype(float)

    grid_results = []; best_sh = -np.inf; best_ret = None; best_p = None
    total = (len(GRID_QUICK["ma_window"])*len(GRID_QUICK["z_window"])*
             len(GRID_QUICK["threshold"])*len(GRID_QUICK["hold_hours"])*len(GRID_QUICK["top_n"]))
    print(f"\nRunning {total}-combo grid on expanded universe...", flush=True)

    for ma,zw,thresh,hold,tn in product(GRID_QUICK["ma_window"],GRID_QUICK["z_window"],
                                         GRID_QUICK["threshold"],GRID_QUICK["hold_hours"],
                                         GRID_QUICK["top_n"]):
        scores_np = bubble_cache[(ma,zw)].values.astype(float)
        univ_mat  = univ_cache[tn]
        signal    = (scores_np < -thresh) & univ_mat & np.isfinite(scores_np)
        trade_rets= []; busy = np.full(len(ticker_list),-1,dtype=int)
        for i in range(n-hold-1):
            if not signal[i].any(): continue
            eb=i+1; xb=min(i+hold,n-1)
            sigs=np.where(signal[i])[0]; elig=[j for j in sigs if busy[j]<eb]
            if not elig: continue
            ep=ho_np[eb,elig]; xp=hc_np[xb,elig]
            valid=(ep>0)&np.isfinite(ep)&np.isfinite(xp)
            if not valid.any(): continue
            net=(xp[valid]/ep[valid]-1)-TC_RATE
            trade_rets.append({"entry_dt":bar_ts[eb],"avg_ret":float(net.mean())})
            busy[np.array(elig)[valid]]=xb
        if len(trade_rets)<5: continue
        tdf=pd.DataFrame(trade_rets); tdf["dt"]=tdf["entry_dt"].dt.normalize()
        daily=tdf.groupby("dt")["avg_ret"].mean().reindex(
            pd.date_range(tdf["dt"].min(), bar_ts[-1].normalize(),freq="B"),fill_value=0.0)
        w=(1+daily).cumprod(); w=w/w.iloc[0]
        sh=_sharpe(daily); mdd=float((w/w.cummax()-1).min()); tot=float(w.iloc[-1]-1)
        so_d=daily[daily<0].std(); so=float(np.sqrt(252)*daily.mean()/so_d) if so_d>0 else np.nan
        wr=float((tdf["avg_ret"]>0).mean())
        row=dict(ma=ma,z_window=zw,threshold=thresh,hold_hours=hold,top_n=tn,
                 Sharpe=sh,Sortino=so,Total_Return=tot,Max_DD=mdd,
                 n_trades=len(trade_rets),Win_Rate=wr)
        grid_results.append(row)
        if pd.notna(sh) and sh>best_sh:
            best_sh=sh; best_ret=daily.rename("SqueezeBubble_Exp"); best_p=row

    if grid_results:
        gdf=pd.DataFrame(grid_results).sort_values("Sharpe",ascending=False)
        print(f"\n{'MA':>5} {'ZWin':>5} {'Thresh':>7} {'Hold':>5} {'TopN':>5} | "
              f"{'Sharpe':>7} {'Return':>9} {'MaxDD':>8}")
        for _,r in gdf.head(10).iterrows():
            if pd.notna(r["Sharpe"]):
                print(f"{int(r['ma']):>5} {int(r['z_window']):>5} {r['threshold']:>7.1f} "
                      f"{int(r['hold_hours']):>5} {int(r['top_n']):>5} | "
                      f"{r['Sharpe']:>7.3f} {r['Total_Return']:>9.2%} {r['Max_DD']:>8.2%}")
        print(f"\nBest: Sharpe={best_p['Sharpe']:.3f}  Return={best_p['Total_Return']:+.1%}")
        print(f"vs Original: Sharpe=2.086  Return=+86.0%")


# ── FINRA RegSho analysis ──────────────────────────────────────────────────
_header("FINRA RegSho Analysis: Short Pressure Distribution")

if not finra_df.empty:
    nas_set = set(nas_tickers)
    finra_nas_all = finra_df[finra_df["ticker"].isin(nas_set)].copy()
    finra_nas_all["marketCap"] = finra_nas_all["ticker"].map(nas_mcap).fillna(0)

    # Statistics
    print(f"\nFINRA RegSho date: {date_val}")
    print(f"Total tickers: {len(finra_df)}")
    print(f"NASDAQ-listed tickers: {len(finra_nas_all)}")
    print(f"\nShort Volume Ratio distribution (NASDAQ stocks with mktcap>$1B):")
    sig = finra_nas_all[finra_nas_all["marketCap"]>1e9].copy()
    print(sig["shortVolRatio"].describe().round(4).to_string())

    # Show percentile buckets
    print(f"\n  Percentile buckets:")
    for pct in [50, 75, 90, 95, 99]:
        v = sig["shortVolRatio"].quantile(pct/100)
        n_above = (sig["shortVolRatio"] >= v).sum()
        print(f"  {pct}th pct: SVR >= {v:.3f}  ({n_above} stocks above)")

    # Chart: histogram of short volume ratio
    fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax_l = axes[0]
    sig["shortVolRatio"].hist(bins=30, ax=ax_l, color="steelblue", alpha=0.8, edgecolor="white")
    ax_l.axvline(sig["shortVolRatio"].mean(), color="red", linestyle="--",
                 label=f"Mean: {sig['shortVolRatio'].mean():.3f}")
    ax_l.axvline(sig["shortVolRatio"].quantile(0.9), color="orange", linestyle="--",
                 label=f"90th pct: {sig['shortVolRatio'].quantile(0.9):.3f}")
    ax_l.set_xlabel("Short Volume Ratio")
    ax_l.set_ylabel("Count")
    ax_l.set_title(f"FINRA RegSho: Short Volume Distribution\nNASDAQ stocks (mktcap>$1B) | {date_val}")
    ax_l.legend(); ax_l.grid(True, alpha=0.3)

    # Top 20 by SVR
    ax_r = axes[1]
    top20 = sig.nlargest(20, "shortVolRatio")
    ax_r.barh(range(len(top20)), top20["shortVolRatio"]*100, color="steelblue", alpha=0.8)
    ax_r.set_yticks(range(len(top20)))
    ax_r.set_yticklabels(top20["ticker"], fontsize=9)
    ax_r.set_xlabel("Short Volume Ratio (%)")
    ax_r.set_title(f"Top 20 NASDAQ by Short Volume Ratio\n({date_val})")
    ax_r.grid(True, alpha=0.3, axis="x")
    plt.tight_layout(); plt.show()


_header("SUMMARY")
print(f"\nPart A -- Live Screener")
print(f"  NASDAQ stocks from screener: {len(nas_tickers)}")
print(f"  FINRA RegSho tickers (totalVol>=1000): {len(finra_df)}")
if not finra_df.empty:
    top5 = screen.head(5)["ticker"].tolist() if 'screen' in dir() else []
    print(f"  Top 5 combined squeeze candidates: {top5}")
print(f"  Screener file: results/squeeze_screener_v2.csv")
print(f"\nPart B -- Expanded Backtest")
print(f"  Total tickers: {len(common_tickers)} (new: {len(new_tickers)})")
print(f"  Original results (Sharpe=2.086) {'UNCHANGED' if not new_tickers else 'see grid above'}")
print(f"\nNote: FINRA RegSho date ({date_val}) is the latest available.")
print(f"  Without auth, the API only returns the most recent day in its dataset.")
