"""
live/settle.py
==============
Per-book replay (EXECUTE + RECORD).  Each book is re-simulated with its LOCKED params
over the full cached window using the validated maths; the engine then keeps only the
live (>= inception) portion.  Returns daily-return series, open positions and closed
trades per book.

  replay_all(panels) -> (book_rets_df[A,B,C,D], positions_dict, trades_df)

Books B and D use the equal-weight daily-P&L engine (engine.equal_weight_daily_pnl).
Book C reuses the validated strategies/intraday_mean_reversion module (single param set).
Book A ports the daily Momentum + Leverage + UVXY construction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C
from . import signals as S
from . import engine as E


# ─────────────────────────────────────────────────────────────────────
# DAILY INFRASTRUCTURE for an hourly panel
# ─────────────────────────────────────────────────────────────────────
def _daily_infra(idx, prices):
    bar_day = idx.normalize().values
    tdays   = np.unique(bar_day)
    d2i     = {d: i for i, d in enumerate(tdays)}
    bdi     = np.array([d2i[d] for d in bar_day], dtype=np.int32)
    D       = len(tdays)
    day_last  = np.zeros(D, dtype=np.int32)
    day_first = np.zeros(D, dtype=np.int32)
    for t in range(len(idx)):       day_last[bdi[t]]  = t
    for t in range(len(idx) - 1, -1, -1): day_first[bdi[t]] = t
    daily_close = prices[day_last]
    daily_ret_cc = np.zeros_like(daily_close)
    daily_ret_cc[1:] = daily_close[1:] / np.maximum(daily_close[:-1], 1e-8) - 1
    daily_ret_cc = np.clip(daily_ret_cc, -0.20, 0.20)
    return tdays, bdi, day_last, day_first, daily_ret_cc


# ─────────────────────────────────────────────────────────────────────
# BOOK D — Contrarian Bubble (MA=104h, thr=-0.8, hold=13h, top=20)
# ─────────────────────────────────────────────────────────────────────
def replay_D(panels):
    p = C.PARAMS["D"]
    hc, ho, idx = panels["hourly_close"], panels["hourly_open"], panels["idx_h"]
    tickers = panels["tickers"]
    prices = hc.values.astype(np.float64)
    opens  = ho.values.astype(np.float64)
    U = len(tickers)
    T = len(idx)

    bub = S.bubble_score_hourly(hc, p["bubble_ma_hours"]).values
    tdays, bdi, day_last, day_first, dret = _daily_infra(idx, prices)

    warmup = p["bubble_ma_hours"] + 1
    thr, hold_h, top_n = p["threshold"], p["hold_hours"], p["top_n"]
    free_at = np.zeros(U, dtype=np.int64)
    trades, open_positions, closed = [], [], []

    last_bar = T - 1
    for t in range(warmup, T - 1):
        scores = bub[t]
        avail = (scores < thr) & (free_at <= t)
        if not avail.any():
            continue
        ai = np.where(avail)[0]
        npick = min(top_n, len(ai))
        chosen = ai[np.argpartition(scores[ai], npick - 1)[:npick]]
        eb = t + 1
        xb = min(t + hold_h, last_bar)
        trades.append((eb, xb, list(chosen), +1))
        for s in chosen:
            rec = dict(book="D", ticker=tickers[s], side=1,
                       entry_ts=str(idx[eb]), exit_ts=str(idx[xb]),
                       entry_px=float(opens[eb, s]), exit_px=float(prices[xb, s]))
            if xb >= last_bar:
                open_positions.append(rec)
            else:
                rec["ret"] = rec["exit_px"] / rec["entry_px"] - 1 if rec["entry_px"] > 0 else 0.0
                closed.append(rec)
        free_at[chosen] = xb

    tc_rt = 2 * p["tc_one_way"]
    port = E.equal_weight_daily_pnl(trades, prices, opens, idx, day_last, day_first,
                                    bdi, dret, U, tc_rt, hold_h)
    ser = pd.Series(port, index=pd.to_datetime(tdays))
    return ser, open_positions, closed


# ─────────────────────────────────────────────────────────────────────
# BOOK B — QQQ Bubble triggers top-5 momentum buys (hold 52h)
# ─────────────────────────────────────────────────────────────────────
def replay_B(panels):
    p = C.PARAMS["B"]
    hc, ho, idx = panels["hourly_close"], panels["hourly_open"], panels["idx_h"]
    tickers = panels["tickers"]
    qqq = panels["qqq_hourly"]
    prices = hc.values.astype(np.float64)
    opens  = ho.values.astype(np.float64)
    U = len(tickers); T = len(idx)

    bub_q = S.qqq_bubble(qqq, p["qqq_bubble_ma_hours"]).values
    mom   = S.momentum_hours(hc, p["mom_lookback_hours"]).values
    tdays, bdi, day_last, day_first, dret = _daily_infra(idx, prices)

    warmup = p["qqq_bubble_ma_hours"] + 5
    thr, hold_h, top_n = p["threshold"], p["hold_hours"], p["top_n"]
    trades, open_positions, closed = [], [], []
    last_bar = T - 1
    i = warmup
    while i < T - 1:
        if bub_q[i] < thr:
            row = mom[i]
            chosen = np.argpartition(row, -top_n)[-top_n:]
            eb = i + 1
            xb = min(i + hold_h, last_bar)
            trades.append((eb, xb, list(chosen), +1))
            for s in chosen:
                rec = dict(book="B", ticker=tickers[s], side=1,
                           entry_ts=str(idx[eb]), exit_ts=str(idx[xb]),
                           entry_px=float(opens[eb, s]), exit_px=float(prices[xb, s]))
                if xb >= last_bar:
                    open_positions.append(rec)
                else:
                    rec["ret"] = rec["exit_px"] / rec["entry_px"] - 1 if rec["entry_px"] > 0 else 0.0
                    closed.append(rec)
            i += hold_h
        else:
            i += 1

    tc_rt = 2 * p["tc_one_way"]
    port = E.equal_weight_daily_pnl(trades, prices, opens, idx, day_last, day_first,
                                    bdi, dret, U, tc_rt, hold_h)
    ser = pd.Series(port, index=pd.to_datetime(tdays))
    return ser, open_positions, closed


# ─────────────────────────────────────────────────────────────────────
# BOOK C — Intraday MR + Flip (reuse validated module, single param set)
# ─────────────────────────────────────────────────────────────────────
def replay_C(panels):
    p = C.PARAMS["C"]
    from strategies.intraday_mean_reversion import run_intraday_mean_reversion
    hc, ho, idx = panels["hourly_close"], panels["hourly_open"], panels["idx_h"]
    daily = panels["daily_close"]
    tickers = panels["tickers"]

    s_d, e_d = idx[0].date(), idx[-1].date()
    daily_c = daily[[t for t in tickers if t in daily.columns]].ffill().loc[str(s_d):str(e_d)]
    cols = list(daily_c.columns)
    best_ret, best_params, _ = run_intraday_mean_reversion(
        daily_close  = daily_c,
        hourly_open  = ho[cols].ffill(),
        hourly_close = hc[cols].ffill(),
        sigma_grid          = [p["sigma"]],
        flip_hold_days_grid = [p["flip_hold_days"]],
        lookback_grid       = [p["z_lookback_days"]],
        top_n_grid          = [p["top_n"]],
        transaction_cost    = p["tc_per_phase"],
        short_borrow_rate   = p["short_borrow_ann"],
    )
    ser = best_ret if best_ret is not None else pd.Series(dtype=float)
    ser.index = pd.to_datetime(ser.index)
    return ser, [], []   # positions/trades detail not tracked for C


# ─────────────────────────────────────────────────────────────────────
# BOOK A — Daily Momentum + 1.25x Leverage + UVXY hedge
# ─────────────────────────────────────────────────────────────────────
def replay_A(panels):
    p = C.PARAMS["A"]
    close = panels["daily_close"].copy()
    lookback, holding, top = p["lookback_days"], p["rebalance_days"], p["top_n"]
    ret_daily = close.pct_change().ffill().fillna(0)
    ret_mom   = close.pct_change(lookback).ffill().fillna(0)

    rows = []
    for i in range(lookback + 1, len(ret_mom), holding):
        ranking = ret_mom.iloc[i-1:i].rank(axis=1, ascending=False)
        ranked  = np.argsort(ranking.values[0])
        long_n  = top                      # Book A is long-only (no short leg)
        for j in range(i, min(i + holding, len(ret_mom))):
            date = ret_daily.index[j]
            ls = np.sign(ret_mom.iloc[:, ranked[:long_n]].iloc[i-1:i]).abs()
            lr = ls.mul(np.array(ret_daily.iloc[:, ranked[:long_n]].iloc[j:j+1])[0])
            lret = lr.values.mean() * long_n
            mom_r = lret / top - p["tc_per_cycle"] / holding
            h_ret = 0.0
            if "UVXY" in close.columns and pd.notna(close.loc[date, "UVXY"]):
                h_ret = ret_daily.loc[date, "UVXY"]
            elif "^VIX" in close.columns and pd.notna(close.loc[date, "^VIX"]):
                v = ret_daily.loc[date, "^VIX"]
                h_ret = (2.0*v - 0.002 - 0.25*v**2 if date < pd.Timestamp("2018-02-28")
                         else 1.5*v - 0.0015 - 0.25*v**2)
            rows.append({"Date": date, "Momentum": mom_r, "Hedge": h_ret})

    dfA = pd.DataFrame(rows).set_index("Date").dropna()
    if dfA.empty:
        return pd.Series(dtype=float), [], []

    base_w = (1 + dfA).cumprod() / (1 + dfA).cumprod().iloc[0]
    bub = S.daily_bubble_on_curve(base_w["Momentum"], p["bubble_ma_days"], p["bubble_z_days"])
    h_sig = (bub > p["hedge_threshold"]).shift(1).fillna(False)
    l_sig = (bub < p["lev_threshold"]).shift(1).fillna(False)
    lev_cost = p["lev_cost_ann"] / C.TRADING_DAYS

    out = []; h_rem = l_rem = 0
    for date in dfA.index:
        if h_rem == 0 and h_sig.loc[date]: h_rem = p["hedge_hold_days"]
        if l_rem == 0 and l_sig.loc[date]: l_rem = p["lev_hold_days"]
        base = dfA.loc[date, "Momentum"]
        if h_rem > 0:
            r = (1 - p["hedge_alloc"])*base + p["hedge_alloc"]*dfA.loc[date, "Hedge"]; h_rem -= 1
        elif l_rem > 0:
            r = base + p["lev_mult"]*base - p["lev_mult"]*lev_cost; l_rem -= 1
        else:
            r = base
        out.append(r)
    ser = pd.Series(out, index=dfA.index)
    ser.index = pd.to_datetime(ser.index)
    return ser, [], []


# ─────────────────────────────────────────────────────────────────────
# REPLAY ALL BOOKS
# ─────────────────────────────────────────────────────────────────────
def replay_all(panels):
    rA, _, _   = replay_A(panels)
    rB, posB, clB = replay_B(panels)
    rC, _, _   = replay_C(panels)
    rD, posD, clD = replay_D(panels)

    book_rets = pd.DataFrame({"A": rA, "B": rB, "C": rC, "D": rD}).sort_index()
    positions = {"A": [], "B": posB, "C": [], "D": posD}
    closed = clB + clD
    trades_df = pd.DataFrame(closed) if closed else pd.DataFrame(
        columns=["book", "ticker", "side", "entry_ts", "exit_ts", "entry_px", "exit_px", "ret"])
    return book_rets, positions, trades_df
