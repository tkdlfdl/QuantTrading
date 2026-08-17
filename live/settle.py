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
def replay_D(panels, book="D"):
    """Contrarian bubble replay. book="D" (hold 8h) or "D14" (hold 14h sleeve,
    promoted 2026-08-16: champion 2.580 -> 2.781 with both sleeves)."""
    p = C.PARAMS[book]
    hc, ho, idx = panels["hourly_close"], panels["hourly_open"], panels["idx_h"]
    tickers = panels["tickers"]
    prices = hc.values.astype(np.float64)
    opens  = ho.values.astype(np.float64)
    U = len(tickers)
    T = len(idx)

    bub = S.bubble_score_hourly(hc, p["bubble_ma_hours"]).values
    if p.get("mom_filter_days"):
        # DU sleeve: only "dips in uptrends" — mask scores where trailing
        # momentum <= 0 so those names can never trigger the -0.8 entry.
        mom_bars = int(p["mom_filter_days"]) * 7
        mom = hc.pct_change(mom_bars).values
        bub = np.where(mom > 0, bub, 0.0)
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
            rec = dict(book=book, ticker=tickers[s], side=1,
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
    # SPLICE GUARD (2026-08-17): daily panel's hourly-close extension can mix
    # adjustment bases (DD +203% 6/22, DELL +213% 6/30 phantom jumps) — mask
    # tickers with any 1-day |move| > 100% inside the lookback out of the
    # ranking. Root fix = consistent-basis daily refresh (queue #34).
    _jump   = close.pct_change().abs().rolling(lookback, min_periods=1).max()
    ret_mom = ret_mom.mask(_jump > 1.0, -9.99)   # flagged -> never top-ranked

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
# BOOK E — Reddit sentiment long-only (daily return series)
# ─────────────────────────────────────────────────────────────────────
def replay_E(panels):
    """Daily return series for Book E over the sentiment window (NaN elsewhere)."""
    p = C.PARAMS["E"]
    try:
        import duckdb
    except Exception:
        return pd.Series(dtype=float), [], []
    if not C.SENTIMENT_DB.exists():
        return pd.Series(dtype=float), [], []

    con = duckdb.connect(str(C.SENTIMENT_DB), read_only=True)
    sd = con.execute("select date,symbol,weighted_compound,mention_count from sentiment_daily").df()
    con.close()
    if sd.empty:
        return pd.Series(dtype=float), [], []
    sd["date"] = pd.to_datetime(sd["date"])
    sent = sd.pivot_table(index="date", columns="symbol", values="weighted_compound", aggfunc="mean")
    ment = sd.pivot_table(index="date", columns="symbol", values="mention_count", aggfunc="sum").fillna(0)

    prices = panels["daily_close"]
    cal = pd.bdate_range(sent.index.min(), sent.index.max())
    syms = [s for s in sent.columns if s in prices.columns]
    sent = sent.reindex(cal)[syms]
    cum = (ment.reindex(cal)[syms].fillna(0) > 0).cumsum()
    px = prices.reindex(cal)[syms].ffill(); ret = px.pct_change().fillna(0)
    n = len(cal)

    idx = pd.DataFrame(index=cal, columns=syms, dtype=float)
    for s in syms:
        si = 100.0 * (1 + sent[s].fillna(0).clip(-1, 1) * p["sentiment_scale"]).cumprod()
        lp = np.log(si.replace(0, np.nan).ffill()); fair = si.rolling(p["ma_window"]).mean()
        res = lp - np.log(fair)
        z = (res - res.rolling(p["z_window"]).mean()) / res.rolling(p["z_window"]).std()
        idx[s] = np.tanh(z / 2)
    sig = idx.shift(1); warm = p["ma_window"] + p["z_window"] + 1; tc = p["tc_one_way"]

    daily = pd.Series(np.nan, index=cal)
    i = warm
    while i < n - 1:
        elig = cum.iloc[i-1]; elig = elig[elig >= p["min_mentions"]].index
        sc = sig.iloc[i][elig].dropna()
        cap = sc[sc < -p["extreme"]].nsmallest(p["top_n"])
        momo = sc[(sc > p["mild"]) & (sc <= p["extreme"])].nlargest(p["top_n"])
        longs = pd.concat([cap, momo]).drop_duplicates()
        end = min(i + p["hold_days"], n)
        if len(longs) == 0:
            for j in range(i, end): daily.iloc[j] = 0.0
            i = end; continue
        for j in range(i, end):
            r = ret.iloc[j][longs.index].mean()
            if j == i: r -= tc
            daily.iloc[j] = r
        i = end
    daily.index = pd.to_datetime(daily.index)
    return daily, [], []


# ─────────────────────────────────────────────────────────────────────
# BOOK F — Universe Hourly Momentum Long (lb=750h, hold=200h, top=5)
# ─────────────────────────────────────────────────────────────────────
def replay_F(panels):
    """
    Non-overlapping hourly cross-sectional momentum.
    Signal: top-5 stocks by 750h cumulative return (107 trading days, ~5 months).
    Hold:   200h (29 trading days, ~6 weeks). Rebalance only after hold closes.
    """
    p = C.PARAMS["F"]
    hc, ho, idx = panels["hourly_close"], panels["hourly_open"], panels["idx_h"]
    tickers = panels["tickers"]
    prices = hc.values.astype(np.float64)
    opens  = ho.values.astype(np.float64)
    U = len(tickers); T = len(idx)

    mom = S.momentum_hours(hc, p["lookback_hours"]).values   # [T x U]
    tdays, bdi, day_last, day_first, dret = _daily_infra(idx, prices)

    # SPLICE GUARD (2026-08-16): exclude tickers with any single-bar move
    # > 100% inside the lookback — ticker-reuse/splice artifacts (e.g. the
    # "BNY" column spliced a ~$10 instrument onto BNY Mellon at +1265%/day,
    # which would have topped the ranking on a phantom return).
    jump = hc.pct_change().abs().rolling(
        p["lookback_hours"], min_periods=1).max().values      # [T x U]

    warmup = p["lookback_hours"] + 1
    hold_h = p["hold_hours"]
    top_n  = p["top_n"]
    trades, open_positions, closed = [], [], []
    last_bar = T - 1

    i = warmup
    while i + hold_h < T:
        row   = mom[i]
        valid = np.where(np.isfinite(row) & (row != 0)
                         & ~(jump[i] > 1.0))[0]
        if len(valid) >= top_n:
            chosen = valid[np.argpartition(row[valid], -top_n)[-top_n:]]
            eb = i + 1
            xb = min(i + hold_h, last_bar)
            trades.append((eb, xb, list(chosen), +1))
            for s in chosen:
                rec = dict(book="F", ticker=tickers[s], side=1,
                           entry_ts=str(idx[eb]), exit_ts=str(idx[xb]),
                           entry_px=float(opens[eb, s]), exit_px=float(prices[xb, s]))
                if xb >= last_bar:
                    open_positions.append(rec)
                else:
                    rec["ret"] = (rec["exit_px"] / rec["entry_px"] - 1
                                  if rec["entry_px"] > 0 else 0.0)
                    closed.append(rec)
        i += hold_h   # non-overlapping: step by full hold period

    tc_rt = 2 * p["tc_one_way"]
    port = E.equal_weight_daily_pnl(trades, prices, opens, idx, day_last, day_first,
                                    bdi, dret, U, tc_rt, hold_h)
    ser = pd.Series(port, index=pd.to_datetime(tdays))
    return ser, open_positions, closed


# ─────────────────────────────────────────────────────────────────────
# BOOK G — Overnight-Share Cross-Section (INCUBATING; zero champion weight)
# ─────────────────────────────────────────────────────────────────────
def replay_G(panels):
    """Overnight-share cross-section — INCUBATING. Rank by trailing 252d
    (overnight - intraday) return spread; long top-7, 21-trading-day holds.

    DATA BASIS IS CRITICAL (validated 2026-08-16): the signal requires OFFICIAL
    daily open prices from DuckDB. Hourly-panel first-bar opens destroy the
    edge (Sharpe 0.19 vs 0.77 — IEX-era opens too noisy for the open-auction
    microstructure this strategy harvests). Requires nightly daily-OHLCV
    refresh (prepare_data.refresh_daily_ohlcv)."""
    p = C.PARAMS["G"]
    import duckdb
    con = duckdb.connect(str(C.SENTIMENT_DB), read_only=True)
    px = con.execute(
        "SELECT ts, symbol, open, close FROM ohlcv WHERE interval='1d' AND ts >= '2018-06-01'"
    ).df()
    con.close()
    px["ts"] = pd.to_datetime(px["ts"])
    close = px.pivot_table(index="ts", columns="symbol", values="close").sort_index()
    openp = px.pivot_table(index="ts", columns="symbol", values="open").sort_index()
    ret1d = close.pct_change()
    spread_r = (openp/close.shift(1) - 1) - (close/openp - 1)
    # splice guard (BNY-type ticker reuse shows up as a giant overnight jump)
    jump = ret1d.abs().rolling(p["spread_window_days"], min_periods=1).max().shift(1)
    sp = spread_r.rolling(p["spread_window_days"], min_periods=p["spread_window_days"]).sum().shift(1)
    sp = sp.mask(jump > 1.0)

    dates = close.index
    n = len(dates)
    hold, top_n = p["hold_days"], p["top_n"]
    tc_rt = 2 * p["tc_one_way"]
    rows, open_positions = [], []
    i = p["spread_window_days"] + 5
    while i + 1 < n:
        d = dates[i]
        srow = sp.loc[d].dropna()
        if len(srow) >= p["min_names"]:
            picks = list(srow.nlargest(top_n).index)
            end = min(i + hold, n)
            fwd = ret1d.iloc[i:end][picks]
            pr = fwd.mean(axis=1).fillna(0.0)
            pr.iloc[0] -= tc_rt
            for dt, x in pr.items():
                rows.append((dt, float(x)))
            if end >= n - 1:
                for tick in picks:
                    open_positions.append(dict(book="G", ticker=tick, side=1,
                                               entry_ts=str(d), exit_ts="open",
                                               entry_px=float(close.loc[d, tick]),
                                               exit_px=float(close[tick].iloc[-1])))
            i = end
        else:
            rows.append((dates[i], 0.0))
            i += 1
    ser = pd.Series(dict(rows)).sort_index()
    ser = ser[~ser.index.duplicated(keep="last")]
    return ser, open_positions, []


# ─────────────────────────────────────────────────────────────────────
# BOOK X — Cross-Asset ETF Momentum (INCUBATING; zero champion weight)
# ─────────────────────────────────────────────────────────────────────
def replay_X(panels):
    """12-1 momentum top-3 of 15 ETFs, equal-weight, monthly, ungated.
    Data: data/cache/etf_daily_close.parquet (refreshed by
    prepare_data.refresh_etf_daily). v1 position bug (ffill accumulation)
    voided 2026-08-16 — this uses the explicit monthly-target construction."""
    p = C.PARAMS["X"]
    path = C.CACHE_DIR / "etf_daily_close.parquet"
    if not path.exists():
        return pd.Series(dtype=float), [], []
    raw = pd.read_parquet(path)
    raw.index = pd.to_datetime(raw.index)
    raw = raw[[t for t in p["etf_universe"] if t in raw.columns]]
    r = raw.pct_change()
    mom = raw.pct_change(p["mom_lookback_days"]).shift(p["mom_skip_days"])
    month_end = raw.index.to_series().dt.month.diff().fillna(1) != 0
    targets = pd.DataFrame(index=raw.index[month_end], columns=raw.columns, dtype=float)
    for d in targets.index:
        m_ = mom.loc[d].dropna()
        row = pd.Series(0.0, index=raw.columns)
        if len(m_) >= 5:
            for t in m_.nlargest(p["top_n"]).index:
                row[t] = 1.0 / p["top_n"]
        targets.loc[d] = row
    pos = targets.reindex(raw.index).ffill().fillna(0.0).shift(1)
    ser = ((r.fillna(0) * pos).sum(axis=1)
           - pos.diff().abs().sum(axis=1).fillna(0) * p["tc_one_way"])
    ser = ser.dropna()
    last = pos.iloc[-1]
    open_positions = [dict(book="X", ticker=t, side=1, entry_ts=str(pos.index[-1]),
                           exit_ts="open", entry_px=float(raw[t].iloc[-1]),
                           exit_px=float(raw[t].iloc[-1]))
                      for t in last[last > 0].index]
    return ser, open_positions, []


# ─────────────────────────────────────────────────────────────────────
# REPLAY ALL BOOKS
# ─────────────────────────────────────────────────────────────────────
def replay_all(panels):
    rA, _, _      = replay_A(panels)
    rB, posB, clB = replay_B(panels)
    rC, _, _      = replay_C(panels)
    rD8, posD8, clD8 = replay_D(panels)
    rD14, posD14, clD14 = replay_D(panels, book="D14")
    # Book D = 50/50 blend of the 8h and 14h sleeves (JT overlapping cohorts;
    # promoted 2026-08-16). Sleeve positions carry half sizing each.
    rD = 0.5 * rD8.add(rD14, fill_value=0.0)
    for rec in posD8:  rec["sleeve"] = "8h";  rec["sleeve_w"] = 0.5
    for rec in posD14: rec["sleeve"] = "14h"; rec["sleeve_w"] = 0.5
    posD = posD8 + posD14
    clD = clD8 + clD14
    rE, _, _      = replay_E(panels)
    rF, posF, clF = replay_F(panels)
    rG, posG, _   = replay_G(panels)
    rX, posX, _   = replay_X(panels)
    rDU, posDU, clDU = replay_D(panels, book="DU")

    book_rets = pd.DataFrame(
        {"A": rA, "B": rB, "C": rC, "D": rD, "E": rE, "F": rF, "G": rG,
         "X": rX, "DU": rDU}
    ).sort_index()
    positions = {"A": [], "B": posB, "C": [], "D": posD, "E": [], "F": posF,
                 "G": posG, "X": posX, "DU": posDU}
    closed = clB + clD + clF
    trades_df = pd.DataFrame(closed) if closed else pd.DataFrame(
        columns=["book", "ticker", "side", "entry_ts", "exit_ts", "entry_px", "exit_px", "ret"])
    return book_rets, positions, trades_df
