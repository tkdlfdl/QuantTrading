"""
Fetch hourly bars from Alpaca for new squeeze-universe tickers
(those in si_panel_expanded.parquet but not in merged_hourly cache).

Saves to:
  data/cache/squeeze_extra_hourly_open.parquet
  data/cache/squeeze_extra_hourly_close.parquet

Then re-runs the expanded bubble score backtest.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

from data.db.schema import init
from data.fetchers.alpaca_fetcher import fetch_alpaca_bars

os.makedirs("results", exist_ok=True)

ALPACA_KEY    = "PKKJQB5MR5L2IFIDZJAGWOMTRL"
ALPACA_SECRET = "6YVRc9gnm9Xkcb7kN1qSFmGkMcUN6LCJeFCyUHJCMCQi"

FETCH_START   = "2024-01-01"   # enough warmup for ma=100h bubble score
CACHE_O       = Path("data/cache/squeeze_extra_hourly_open.parquet")
CACHE_C       = Path("data/cache/squeeze_extra_hourly_close.parquet")

TRADING_DAYS  = 252
TC_RATE       = 0.001
PUB_LAG       = pd.offsets.BusinessDay(8)

GRID = dict(
    ma_window  = [20, 50, 100],
    z_window   = [50, 100, 200],
    threshold  = [0.3, 0.5, 0.7, 0.9],
    hold_hours = [1, 2, 4, 8, 13],
    top_n      = [10, 20, 30],
)


def _sharpe(r, td=TRADING_DAYS):
    s = r.std(); return float(np.sqrt(td)*r.mean()/s) if s > 0 else np.nan

def _sortino(r, td=TRADING_DAYS):
    ds = r[r<0].std(); return float(np.sqrt(td)*r.mean()/ds) if ds > 0 else np.nan

def _mdd(r):
    w = (1+r).cumprod(); w = w/w.iloc[0]; return float((w/w.cummax()-1).min())

def _header(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}", flush=True)


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 -- Identify new tickers needing Alpaca data
# ══════════════════════════════════════════════════════════════════════════

init()
_header("STEP 1 -- Identify new tickers")

si_exp  = pd.read_parquet("results/si_panel_expanded.parquet")
si_orig = pd.read_parquet("results/si_panel_12m.parquet")
orig_tickers = set(si_orig["ticker"].unique())
new_tickers  = sorted(set(si_exp["ticker"].unique()) - orig_tickers)
print(f"New tickers in expanded SI panel: {len(new_tickers)}")

# Check merged hourly cache
ho_merged = pd.read_parquet("data/cache/merged_hourly_open.parquet")
not_in_cache = [t for t in new_tickers if t not in ho_merged.columns]
print(f"Need Alpaca fetch: {len(not_in_cache)}")
print(f"Tickers: {not_in_cache}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 -- Fetch from Alpaca (or load existing cache)
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 2 -- Fetch hourly bars from Alpaca")

if CACHE_O.exists() and CACHE_C.exists():
    ho_extra = pd.read_parquet(CACHE_O)
    hc_extra = pd.read_parquet(CACHE_C)
    cached_tickers = list(ho_extra.columns)
    missing = [t for t in not_in_cache if t not in cached_tickers]
    data_age = (pd.Timestamp.now().date() - ho_extra.index[-1].date()).days
    print(f"Existing extra cache: {len(cached_tickers)} tickers "
          f"({ho_extra.index[0].date()} to {ho_extra.index[-1].date()}, "
          f"age={data_age} days)")
    if missing:
        print(f"Still missing: {len(missing)} tickers -- fetching...")
    elif data_age > 7:
        print(f"Cache stale ({data_age} days) -- re-fetching all...")
        missing = not_in_cache
    else:
        print("Cache is fresh and complete -- skipping fetch.")
        missing = []
else:
    ho_extra = pd.DataFrame()
    hc_extra = pd.DataFrame()
    missing  = not_in_cache

if missing:
    # Filter out likely non-tradable tickers (preferred shares, warrants, etc.)
    tradable = [t for t in missing
                if not any(c in t for c in ['P','W','R','Z','U'])
                or len(t) <= 4]
    # Also keep those ending in P only if they look like normal tickers
    clean_missing = []
    for t in missing:
        # Skip obvious preferred/warrant symbols
        if len(t) > 5: continue
        if t.endswith(('W','R','Z','U')): continue   # warrants, rights, units
        clean_missing.append(t)
    print(f"\nFetching {len(clean_missing)} tickers from Alpaca IEX "
          f"(start={FETCH_START})...", flush=True)

    ho_new, hc_new = fetch_alpaca_bars(
        tickers     = clean_missing,
        start       = FETCH_START,
        end         = None,
        timeframe   = "1Hour",
        feed        = "iex",
        api_key     = ALPACA_KEY,
        secret_key  = ALPACA_SECRET,
        chunk_size  = 20,
    )

    if ho_new.empty:
        print("No data returned from Alpaca.")
    else:
        print(f"\nAlpaca returned: {len(ho_new.columns)} tickers "
              f"({ho_new.index[0].date()} to {ho_new.index[-1].date()})")
        print(f"  Tickers with data: {list(ho_new.columns)}")
        skipped = [t for t in clean_missing if t not in ho_new.columns]
        if skipped:
            print(f"  No data for: {skipped}")

        # Merge with any existing extra cache
        if not ho_extra.empty:
            # Combine: reindex to union of timestamps
            all_idx = ho_extra.index.union(ho_new.index)
            ho_extra = ho_extra.reindex(all_idx)
            hc_extra = hc_extra.reindex(all_idx)
            for col in ho_new.columns:
                ho_extra[col] = ho_new[col]
                hc_extra[col] = hc_new[col] if col in hc_new.columns else np.nan
            ho_extra = ho_extra.sort_index().ffill()
            hc_extra = hc_extra.sort_index().ffill()
        else:
            ho_extra = ho_new
            hc_extra = hc_new

        ho_extra.to_parquet(CACHE_O)
        hc_extra.to_parquet(CACHE_C)
        print(f"\nSaved extra cache: {len(ho_extra.columns)} tickers, "
              f"{len(ho_extra)} hourly bars")
        print(f"  {CACHE_O}")

if ho_extra.empty:
    print("No extra hourly data available. Backtest will use original universe only.")


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 -- Load all hourly data (merged + extra)
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 3 -- Combine hourly data: merged cache + new tickers")

from data.intraday_loader import load_hourly_bars
from data.universe import get_universe

universe = get_universe()
ho_base, hc_base = load_hourly_bars(universe, use_cache=True)

if not ho_extra.empty:
    # Align timestamps: reindex extra to base index
    extra_tickers = [t for t in ho_extra.columns if t not in ho_base.columns]
    if extra_tickers:
        common_idx = ho_base.index.union(ho_extra.index)
        ho_all = ho_base.reindex(common_idx).sort_index()
        hc_all = hc_base.reindex(common_idx).sort_index()
        for t in extra_tickers:
            ho_all[t] = ho_extra[t].reindex(common_idx)
            hc_all[t] = hc_extra[t].reindex(common_idx) if t in hc_extra.columns else np.nan
        ho_all = ho_all.ffill()
        hc_all = hc_all.ffill()
        print(f"Combined: {len(ho_all.columns)} tickers "
              f"({ho_all.index[0].date()} to {ho_all.index[-1].date()})")
        print(f"  Original: {len(ho_base.columns)}, New: {len(extra_tickers)}")
        print(f"  New tickers: {extra_tickers}")
    else:
        ho_all = ho_base; hc_all = hc_base
        print("No new tickers added (all already in base cache or no Alpaca data).")
else:
    ho_all = ho_base; hc_all = hc_base


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 -- Rebuild SI panel + point-in-time universe
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 4 -- Build SI timeline (original + expanded)")

si_all = pd.concat([si_orig, si_exp], ignore_index=True).drop_duplicates(
    subset=["ticker","settlementDate"])
print(f"SI panel: {si_all['ticker'].nunique()} tickers | "
      f"{si_all['settlementDate'].nunique()} settlement dates")

# Restrict to backtest window
backtest_start = sorted(si_all["settlementDate"].unique())[0] + pd.Timedelta(days=1)
mask  = ho_all.index >= backtest_start
ho    = ho_all[mask].copy()
hc    = hc_all[mask].copy()

si_tickers     = set(si_all["ticker"].unique())
common_tickers = [t for t in si_tickers if t in ho.columns]
ho = ho[common_tickers]; hc = hc[common_tickers]; n = len(ho)

orig_in_bt = [t for t in common_tickers if t in orig_tickers]
new_in_bt  = [t for t in common_tickers if t not in orig_tickers]
print(f"\nTickers in backtest: {len(common_tickers)}")
print(f"  Original (S&P500/NDX100): {len(orig_in_bt)}")
print(f"  NEW from NASDAQ pressure screen: {len(new_in_bt)}")
if new_in_bt:
    print(f"  New tickers: {sorted(new_in_bt)}")

# Build timeline
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


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 -- Bubble score + grid search
# ══════════════════════════════════════════════════════════════════════════

_header("STEP 5 -- Precompute bubble scores")

ho_full = ho_all[common_tickers]
hc_full = hc_all[common_tickers]

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
    bubble_cache[(ma,zw)] = compute_bubble(hc_full, ma, zw).loc[ho.index].copy()
print(f"Computed {len(bubble_cache)} bubble score combinations for {len(common_tickers)} tickers",
      flush=True)

ticker_list = list(common_tickers); ticker_idx = {t:i for i,t in enumerate(ticker_list)}
bar_ts      = ho.index
univ_cache  = {}
for tn in GRID["top_n"]:
    mat = np.zeros((n,len(ticker_list)),dtype=bool)
    for i,ts in enumerate(bar_ts):
        for t in get_universe_at(ts,tn):
            if t in ticker_idx: mat[i,ticker_idx[t]] = True
    univ_cache[tn] = mat
    print(f"  top_n={tn}: avg {mat.sum(axis=1).mean():.1f} tickers/bar", flush=True)

ho_np = ho.values.astype(float); hc_np = hc.values.astype(float)

_header("STEP 6 -- Grid search (expanded universe)")

total = (len(GRID["ma_window"])*len(GRID["z_window"])*len(GRID["threshold"])
         *len(GRID["hold_hours"])*len(GRID["top_n"]))
print(f"Grid: {total} combos | Universe: {len(common_tickers)} tickers "
      f"({len(new_in_bt)} new)\n", flush=True)

grid_results = []; best_sharpe = -np.inf; best_ret = None; best_params = None
combo_n = 0

for ma,zw,thresh,hold,tn in product(
    GRID["ma_window"],GRID["z_window"],GRID["threshold"],GRID["hold_hours"],GRID["top_n"]):
    combo_n += 1
    if combo_n % 60 == 0: print(f"  combo {combo_n}/{total}...", flush=True)

    scores_np = bubble_cache[(ma,zw)].values.astype(float)
    univ_mat  = univ_cache[tn]
    signal    = (scores_np < -thresh) & univ_mat & np.isfinite(scores_np)

    trade_rets = []; busy = np.full(len(ticker_list),-1,dtype=int)
    for i in range(n-hold-1):
        if not signal[i].any(): continue
        eb=i+1; xb=min(i+hold,n-1)
        sigs=np.where(signal[i])[0]; elig=[j for j in sigs if busy[j]<eb]
        if not elig: continue
        ep=ho_np[eb,elig]; xp=hc_np[xb,elig]
        valid=(ep>0)&np.isfinite(ep)&np.isfinite(xp)
        if not valid.any(): continue
        net=(xp[valid]/ep[valid]-1)-TC_RATE
        trade_rets.append({"entry_dt":bar_ts[eb],"avg_ret":float(net.mean()),
                           "n_pos":int(valid.sum()),
                           "score":float(scores_np[i][np.array(elig)[valid]].mean())})
        busy[np.array(elig)[valid]]=xb

    if len(trade_rets)<5:
        grid_results.append(dict(ma=ma,z_window=zw,threshold=thresh,hold_hours=hold,top_n=tn,
                                 Sharpe=np.nan,Sortino=np.nan,Total_Return=np.nan,
                                 Max_DD=np.nan,n_trades=len(trade_rets),Win_Rate=np.nan))
        continue

    tdf   = pd.DataFrame(trade_rets)
    daily = (tdf.assign(dt=tdf["entry_dt"].dt.normalize())
               .groupby("dt")["avg_ret"].mean()
               .reindex(pd.date_range(tdf["entry_dt"].dt.normalize().min(),
                                      bar_ts[-1].normalize(),freq="B"), fill_value=0.0))
    w   = (1+daily).cumprod(); w=w/w.iloc[0]
    sh  = _sharpe(daily)
    so_d= daily[daily<0].std()
    so  = float(np.sqrt(TRADING_DAYS)*daily.mean()/so_d) if so_d>0 else np.nan
    mdd = float((w/w.cummax()-1).min())
    tot = float(w.iloc[-1]-1)
    wr  = float((tdf["avg_ret"]>0).mean())

    row = dict(ma=ma,z_window=zw,threshold=thresh,hold_hours=hold,top_n=tn,
               Sharpe=sh,Sortino=so,Total_Return=tot,Max_DD=mdd,
               n_trades=len(trade_rets),Win_Rate=wr,
               Avg_Score=float(tdf["score"].mean()))
    grid_results.append(row)
    if pd.notna(sh) and sh>best_sharpe:
        best_sharpe=sh; best_ret=daily.rename("SqueezeBubble_Expanded"); best_params=row

grid_df = pd.DataFrame(grid_results).sort_values("Sharpe",ascending=False)

_header("Top 20 results")
print(f"{'MA':>5} {'ZWin':>5} {'Thresh':>7} {'Hold':>5} {'TopN':>5} | "
      f"{'Sharpe':>7} {'Sortino':>8} {'Return':>9} {'Max_DD':>8} {'nTrades':>8} {'WinRate':>8}")
print("-"*90)
for _,r in grid_df.head(20).iterrows():
    if pd.notna(r["Sharpe"]):
        print(f"{int(r['ma']):>5} {int(r['z_window']):>5} {r['threshold']:>7.1f} "
              f"{int(r['hold_hours']):>5} {int(r['top_n']):>5} | "
              f"{r['Sharpe']:>7.3f} {r['Sortino']:>8.3f} {r['Total_Return']:>9.2%} "
              f"{r['Max_DD']:>8.2%} {int(r['n_trades']):>8} {r['Win_Rate']:>8.2%}")

p = best_params
print(f"\nBest: ma={p['ma']}h | z_win={p['z_window']}h | thresh={p['threshold']} | "
      f"hold={p['hold_hours']}h | top_n={p['top_n']}")
print(f"  Sharpe={p['Sharpe']:.3f}  Sortino={p['Sortino']:.3f}  "
      f"Return={p['Total_Return']:+.1%}  MaxDD={p['Max_DD']:.1%}  "
      f"WinRate={p['Win_Rate']:.1%}")

_header("Yearly performance")
for yr, grp in best_ret.groupby(best_ret.index.year):
    w   = (1+grp).cumprod(); w=w/w.iloc[0]
    print(f"  {yr}: Return={float(w.iloc[-1]-1):>+8.2%}  "
          f"Sharpe={_sharpe(grp):>6.3f}  MaxDD={float((w/w.cummax()-1).min()):>8.2%}  "
          f"Days={len(grp)}")

_header("Comparison: Original vs Expanded universe")
print(f"  Original (145 tickers):  Sharpe=2.086  Return=+86.0%  MaxDD=-13.3%")
print(f"  Expanded ({len(common_tickers)} tickers):  "
      f"Sharpe={p['Sharpe']:.3f}  Return={p['Total_Return']:+.1%}  MaxDD={p['Max_DD']:.1%}")
delta_sh = p['Sharpe'] - 2.086
print(f"  Delta Sharpe: {delta_sh:+.3f}")

# Save
xl = "results/squeeze_expanded_backtest.xlsx"
with pd.ExcelWriter(xl, engine="openpyxl") as writer:
    pd.DataFrame([{"Universe":f"{len(common_tickers)} tickers ({len(new_in_bt)} new)",
                   "NewTickers": str(sorted(new_in_bt)),
                   "Start":str(best_ret.index[0].date()), "End":str(best_ret.index[-1].date()),
                   "Total Return":p["Total_Return"],"Sharpe":p["Sharpe"],
                   "Max DD":p["Max_DD"],**p}]).to_excel(writer, sheet_name="Summary", index=False)
    grid_df.to_excel(writer, sheet_name="Grid_Results", index=False)
    pd.DataFrame({"Date":best_ret.index,"Return":best_ret.values}).to_excel(
        writer, sheet_name="Daily_Returns", index=False)
print(f"\nSaved: {xl}", flush=True)
