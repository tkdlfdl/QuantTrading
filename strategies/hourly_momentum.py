"""
Hourly Cross-Sectional Momentum Strategy
==========================================
Universe : NASDAQ100 + S&P500 (516 tickers)
Signal   : At hourly bar t, compute each stock's cumulative return over the
           past `lookback` hours using CLOSE prices:
             mom[t] = close[t] / close[t - lookback] - 1
           (no lookahead: signal uses only data through bar t's close)
Entry    : LONG / SHORT at open of bar t+1
Exit     : close of bar t + hold_hours
Long     : top-N by momentum (strongest up trend)
Short    : bottom-N by momentum (strongest down trend)

Long and short grids are searched INDEPENDENTLY.

Costs    : 0.1% round-trip transaction cost
           8%/yr short borrow rate on short positions
             = 8% / 252 / 6.5 per hour held short
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import product

TRADING_DAYS  = 252
TRADING_HOURS = TRADING_DAYS * 6.5   # ~1638 per year
TC            = 0.001                 # one-way 0.1%
BORROW_HOURLY = 0.08 / TRADING_HOURS  # 8%/yr → per hour


def _sharpe(r: pd.Series, td: int = TRADING_DAYS) -> float:
    s = r.std(); return float(np.sqrt(td) * r.mean() / s) if s > 0 else np.nan

def _sortino(r: pd.Series, td: int = TRADING_DAYS) -> float:
    ds = r[r < 0].std(); return float(np.sqrt(td) * r.mean() / ds) if ds > 0 else np.nan

def _mdd(r: pd.Series) -> float:
    w = (1 + r).cumprod(); w = w / w.iloc[0]
    return float((w / w.cummax() - 1).min())


def _run_side(
    ho_np: np.ndarray,          # [bars x tickers] hourly open
    hc_np: np.ndarray,          # [bars x tickers] hourly close
    bar_ts: pd.DatetimeIndex,
    lookback_grid: list,
    hold_grid: list,
    top_n_grid: list,
    direction: str,             # "long" or "short"
    mom_cache: dict,            # precomputed momentum arrays per lookback
) -> tuple[pd.DataFrame, pd.Series | None, dict | None]:
    """
    Run grid search for one direction (long or short).
    Returns (grid_df, best_daily_ret, best_params).
    """
    n       = len(bar_ts)
    n_tick  = ho_np.shape[1]
    is_long = (direction == "long")
    sign    = 1 if is_long else -1

    total = len(lookback_grid) * len(hold_grid) * len(top_n_grid)
    print(f"  [{direction}] Grid: {len(lookback_grid)} lookback x "
          f"{len(hold_grid)} hold x {len(top_n_grid)} top_n = {total} combos",
          flush=True)

    grid_results = []
    best_sh  = -np.inf
    best_ret = None
    best_p   = None

    combo_n = 0
    for lb, hold, top_n in product(lookback_grid, hold_grid, top_n_grid):
        combo_n += 1
        if combo_n % 20 == 0:
            print(f"    combo {combo_n}/{total}...", flush=True)

        mom_np = mom_cache[lb]   # [bars x tickers], NaN for first lb bars

        trade_rets = []
        busy       = np.full(n_tick, -1, dtype=int)   # per-ticker last exit bar

        for i in range(lb, n - hold - 1):
            row = mom_np[i]
            valid_mask = np.isfinite(row)
            if valid_mask.sum() < top_n:
                continue

            if is_long:
                # Top-N by momentum (most positive = best uptrend)
                ranked = np.where(valid_mask)[0]
                top_idx = ranked[np.argsort(row[ranked])[-top_n:]]
            else:
                # Bottom-N by momentum (most negative = weakest stocks)
                ranked = np.where(valid_mask)[0]
                top_idx = ranked[np.argsort(row[ranked])[:top_n]]

            entry_bar = i + 1
            exit_bar  = min(i + hold, n - 1)

            # Skip tickers already in a trade
            free = [j for j in top_idx if busy[j] < entry_bar]
            if not free:
                continue

            ep = ho_np[entry_bar, free]
            xp = hc_np[exit_bar,  free]
            ok = (ep > 0) & np.isfinite(ep) & np.isfinite(xp)
            if not ok.any():
                continue

            ep_ok = ep[ok]; xp_ok = xp[ok]
            raw   = (xp_ok / ep_ok - 1) * sign
            borrow = BORROW_HOURLY * hold if not is_long else 0.0
            net   = raw - TC - borrow

            trade_rets.append({
                "entry_bar": entry_bar,
                "entry_dt":  bar_ts[entry_bar],
                "avg_ret":   float(net.mean()),
                "n_pos":     int(ok.sum()),
                "avg_mom":   float(row[np.array(free)[ok]].mean()),
            })

            busy[np.array(free)[ok]] = exit_bar

        if len(trade_rets) < 5:
            grid_results.append(dict(lookback=lb, hold=hold, top_n=top_n, direction=direction,
                                     Sharpe=np.nan, Sortino=np.nan, Total_Return=np.nan,
                                     Max_DD=np.nan, n_trades=len(trade_rets), Win_Rate=np.nan))
            continue

        tdf   = pd.DataFrame(trade_rets)
        daily = (tdf.assign(dt=tdf["entry_dt"].dt.normalize())
                    .groupby("dt")["avg_ret"].mean()
                    .reindex(pd.date_range(tdf["entry_dt"].dt.normalize().min(),
                                           bar_ts[-1].normalize(), freq="B"),
                             fill_value=0.0))

        w   = (1 + daily).cumprod(); w = w / w.iloc[0]
        sh  = _sharpe(daily); mdd = float((w / w.cummax() - 1).min()); tot = float(w.iloc[-1] - 1)
        so_d= daily[daily < 0].std()
        so  = float(np.sqrt(TRADING_DAYS) * daily.mean() / so_d) if so_d > 0 else np.nan
        wr  = float((tdf["avg_ret"] > 0).mean())
        avg_hold = float(tdf["entry_bar"].sub(
            tdf["entry_bar"].shift(1).fillna(tdf["entry_bar"].iloc[0])).abs().mean())

        row_r = dict(lookback=lb, hold=hold, top_n=top_n, direction=direction,
                     Sharpe=sh, Sortino=so, Total_Return=tot, Max_DD=mdd,
                     n_trades=len(trade_rets), Win_Rate=wr,
                     Avg_Mom=float(tdf["avg_mom"].mean()))
        grid_results.append(row_r)

        if pd.notna(sh) and sh > best_sh:
            best_sh  = sh
            best_ret = daily.rename(f"Mom_{direction}")
            best_p   = row_r

    grid_df = pd.DataFrame(grid_results).sort_values("Sharpe", ascending=False)
    return grid_df, best_ret, best_p


def run_hourly_momentum(
    hourly_open:  pd.DataFrame,
    hourly_close: pd.DataFrame,
    lookback_grid: list = [60, 120, 200, 300],
    hold_grid:     list = [1, 2, 5, 10, 20],
    top_n_grid:    list = [5, 10, 20, 30],
) -> dict:
    """
    Run separate long and short grids, then combine.

    Returns dict with keys:
      long_grid, short_grid        -- full grid DataFrames
      best_long, best_short        -- best return Series for each side
      best_long_params, best_short_params
      combined_ret                 -- best_long + best_short (equal weight)
      combined_grid                -- grid of all long x short combinations
    """
    # Align columns (common tickers)
    common = [c for c in hourly_open.columns if c in hourly_close.columns]
    ho = hourly_open[common].copy()
    hc = hourly_close[common].copy()
    n  = len(ho)

    print(f"Universe: {len(common)} tickers | {n} hourly bars "
          f"({ho.index[0].date()} to {ho.index[-1].date()})", flush=True)

    bar_ts  = ho.index
    ho_np   = ho.values.astype(float)
    hc_np   = hc.values.astype(float)

    # Pre-compute close-to-close momentum for each lookback
    # mom[i] = hc[i] / hc[i - lb] - 1  (shift by 1 so signal is known before bar i+1 opens)
    # We shift the whole matrix by 1 so that at bar i we use the momentum visible after
    # bar i closes, which we act on at bar i+1 open.
    print("\nPrecomputing momentum (close-to-close, shift=0, entry next bar)...", flush=True)
    mom_cache: dict[int, np.ndarray] = {}
    for lb in lookback_grid:
        # At bar i: mom = hc[i] / hc[i-lb] - 1  (uses bar i close, enters bar i+1 open)
        mom_df = hc.pct_change(lb)   # = hc / hc.shift(lb) - 1
        mom_cache[lb] = mom_df.values.astype(float)
        print(f"  lookback={lb}h: done", flush=True)

    # ── Long grid ──────────────────────────────────────────────────────────
    print("\n--- LONG GRID ---", flush=True)
    long_grid, best_long, best_long_p = _run_side(
        ho_np, hc_np, bar_ts,
        lookback_grid, hold_grid, top_n_grid,
        direction="long", mom_cache=mom_cache,
    )

    # ── Short grid ─────────────────────────────────────────────────────────
    print("\n--- SHORT GRID ---", flush=True)
    short_grid, best_short, best_short_p = _run_side(
        ho_np, hc_np, bar_ts,
        lookback_grid, hold_grid, top_n_grid,
        direction="short", mom_cache=mom_cache,
    )

    # ── Combined: align + average ──────────────────────────────────────────
    combined_ret = None
    if best_long is not None and best_short is not None:
        idx = best_long.index.union(best_short.index)
        l   = best_long.reindex(idx, fill_value=0.0)
        s   = best_short.reindex(idx, fill_value=0.0)
        combined_ret = ((l + s) / 2).rename("Mom_Combined")

    # ── Combined grid: all long x short param combinations ─────────────────
    # Use top-5 long and top-5 short param combos to build combined grid
    combo_rows = []
    top_long_params  = long_grid.dropna(subset=["Sharpe"]).head(5)
    top_short_params = short_grid.dropna(subset=["Sharpe"]).head(5)

    for _, lrow in top_long_params.iterrows():
        for _, srow in top_short_params.iterrows():
            # Rebuild returns for this combo
            lret_rows = []; busy_l = np.full(n, -1, dtype=int)
            lb_l = int(lrow["lookback"]); hld_l = int(lrow["hold"]); tn_l = int(lrow["top_n"])
            mom_l = mom_cache[lb_l]
            for i in range(lb_l, n - hld_l - 1):
                row_ = mom_l[i]; valid_ = np.isfinite(row_)
                if valid_.sum() < tn_l: continue
                idxs = np.where(valid_)[0]
                top_ = idxs[np.argsort(row_[idxs])[-tn_l:]]
                eb = i+1; xb = min(i+hld_l, n-1)
                free_ = [j for j in top_ if busy_l[j] < eb]
                if not free_: continue
                ep_ = ho_np[eb, free_]; xp_ = hc_np[xb, free_]
                ok_ = (ep_>0)&np.isfinite(ep_)&np.isfinite(xp_)
                if not ok_.any(): continue
                net_ = (xp_[ok_]/ep_[ok_]-1) - TC
                lret_rows.append({"dt": bar_ts[eb].normalize(), "ret": float(net_.mean())})
                busy_l[np.array(free_)[ok_]] = xb

            sret_rows = []; busy_s = np.full(n, -1, dtype=int)
            lb_s = int(srow["lookback"]); hld_s = int(srow["hold"]); tn_s = int(srow["top_n"])
            mom_s = mom_cache[lb_s]
            for i in range(lb_s, n - hld_s - 1):
                row_ = mom_s[i]; valid_ = np.isfinite(row_)
                if valid_.sum() < tn_s: continue
                idxs = np.where(valid_)[0]
                bot_ = idxs[np.argsort(row_[idxs])[:tn_s]]
                eb = i+1; xb = min(i+hld_s, n-1)
                free_ = [j for j in bot_ if busy_s[j] < eb]
                if not free_: continue
                ep_ = ho_np[eb, free_]; xp_ = hc_np[xb, free_]
                ok_ = (ep_>0)&np.isfinite(ep_)&np.isfinite(xp_)
                if not ok_.any(): continue
                raw_ = -(xp_[ok_]/ep_[ok_]-1)
                net_ = raw_ - TC - BORROW_HOURLY * hld_s
                sret_rows.append({"dt": bar_ts[eb].normalize(), "ret": float(net_.mean())})
                busy_s[np.array(free_)[ok_]] = xb

            if not lret_rows or not sret_rows:
                continue

            ld = (pd.DataFrame(lret_rows).groupby("dt")["ret"].mean())
            sd = (pd.DataFrame(sret_rows).groupby("dt")["ret"].mean())
            all_dt = ld.index.union(sd.index)
            combo_d = ((ld.reindex(all_dt, fill_value=0.0) +
                        sd.reindex(all_dt, fill_value=0.0)) / 2)
            combo_d = combo_d.reindex(
                pd.date_range(combo_d.index.min(), bar_ts[-1].normalize(), freq="B"),
                fill_value=0.0)

            w   = (1+combo_d).cumprod(); w=w/w.iloc[0]
            sh  = _sharpe(combo_d); mdd = float((w/w.cummax()-1).min()); tot=float(w.iloc[-1]-1)
            so_d= combo_d[combo_d<0].std()
            so  = float(np.sqrt(TRADING_DAYS)*combo_d.mean()/so_d) if so_d>0 else np.nan

            combo_rows.append({
                "lb_long": lb_l, "hold_long": hld_l, "n_long": tn_l,
                "lb_short": lb_s, "hold_short": hld_s, "n_short": tn_s,
                "Sharpe": sh, "Sortino": so, "Total_Return": tot, "Max_DD": mdd,
            })

    combined_grid = pd.DataFrame(combo_rows).sort_values("Sharpe", ascending=False) \
        if combo_rows else pd.DataFrame()

    return dict(
        long_grid=long_grid, short_grid=short_grid,
        best_long=best_long, best_short=best_short,
        best_long_params=best_long_p, best_short_params=best_short_p,
        combined_ret=combined_ret, combined_grid=combined_grid,
    )
