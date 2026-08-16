"""
live/engine.py
==============
Shared primitives for the paper-trading engine:

  StateStore            - load/save meta.json, positions.json, equity.csv, trades.csv
  metrics_from_returns  - Sharpe / Sortino / MaxDD / cumulative / annualised / win-rate
  fixed_ew / mom_alloc  - portfolio combiners from per-book daily-return series
  equal_weight_daily    - the validated equal-weight daily-P&L sizing (multi-day holds)

Design note (forward-only, replay-based):
  Each settled evening, settle.py re-replays each book with its locked params over the
  full cached window and extracts the per-day return series.  Only days >= inception count
  toward the track record.  This guarantees the live numbers use the exact same maths as
  the validated backtests with zero state-drift, while warmup is satisfied automatically by
  the cached history that precedes inception.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

from . import config as C


# ─────────────────────────────────────────────────────────────────────
# STATE STORE
# ─────────────────────────────────────────────────────────────────────
class StateStore:
    def __init__(self):
        C.ensure_dirs()
        self.meta = self._load_json(C.META_FILE, default=None)
        self.positions = self._load_json(C.POSITIONS_FILE, default={b: [] for b in C.BOOKS})
        self.equity = self._load_csv(C.EQUITY_FILE)
        self.trades = self._load_csv(C.TRADES_FILE)

    # ---- io helpers ----
    @staticmethod
    def _load_json(path, default):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return default

    @staticmethod
    def _load_csv(path):
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()

    def save(self):
        C.ensure_dirs()
        if self.meta is not None:
            C.META_FILE.write_text(json.dumps(self.meta, indent=2, default=str), encoding="utf-8")
        C.POSITIONS_FILE.write_text(json.dumps(self.positions, indent=2, default=str), encoding="utf-8")
        if not self.equity.empty:
            self.equity.to_csv(C.EQUITY_FILE, index=False)
        if not self.trades.empty:
            self.trades.to_csv(C.TRADES_FILE, index=False)

    # ---- meta ----
    def init_meta(self, inception: str):
        if self.meta is None:
            self.meta = dict(
                inception=inception,
                last_settled_date=inception,   # forward-only: nothing before inception
                last_planned_date=None,
                capital_per_book=C.CAPITAL_PER_BOOK,
            )

    @property
    def inception(self):
        return self.meta["inception"] if self.meta else None

    @property
    def last_settled(self):
        return self.meta["last_settled_date"] if self.meta else None


# ─────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────
def metrics_from_returns(returns: pd.Series, rf_ann: float = C.RISK_FREE_ANN) -> dict:
    """All inception-to-date metrics from a daily-return series."""
    r = returns.dropna()
    if len(r) == 0:
        return dict(n_days=0, cum_ret=0.0, ann_ret=0.0, sharpe=0.0,
                    sortino=0.0, maxdd=0.0, win_rate=0.0, vol_ann=0.0)
    rf_d = rf_ann / C.TRADING_DAYS
    exc  = r - rf_d
    std  = r.std()
    sharpe = float(exc.mean() / std * np.sqrt(C.TRADING_DAYS)) if std > 0 else 0.0
    dn   = r[r < 0].std(ddof=0)
    sortino = float(exc.mean() / dn * np.sqrt(C.TRADING_DAYS)) if dn and dn > 0 else 0.0
    w    = (1 + r).cumprod()
    cum  = float(w.iloc[-1] - 1)
    dd   = float((w / w.cummax() - 1).min())
    years = len(r) / C.TRADING_DAYS
    ann  = float((1 + cum) ** (1 / years) - 1) if (years > 0 and cum > -1) else 0.0
    active = r[r != 0]
    win  = float((active > 0).mean()) if len(active) else 0.0
    return dict(n_days=int(len(r)), cum_ret=cum, ann_ret=ann, sharpe=sharpe,
                sortino=sortino, maxdd=dd, win_rate=win,
                vol_ann=float(std * np.sqrt(C.TRADING_DAYS)))


def rolling_sharpe(returns: pd.Series, window: int = 30) -> pd.Series:
    rm = returns.rolling(window, min_periods=max(5, window // 3)).mean()
    rs = returns.rolling(window, min_periods=max(5, window // 3)).std()
    return (rm / rs * np.sqrt(C.TRADING_DAYS)).replace([np.inf, -np.inf], np.nan)


# ─────────────────────────────────────────────────────────────────────
# PORTFOLIO COMBINERS  (from per-book daily-return DataFrame [date x book])
# ─────────────────────────────────────────────────────────────────────
def fixed_ew(book_rets: pd.DataFrame) -> pd.Series:
    """Equal weight across books that are 'active' (non-NaN) each day."""
    avail = book_rets.notna()
    n = avail.sum(axis=1).replace(0, np.nan)
    w = avail.div(n, axis=0)
    return (book_rets.fillna(0) * w.fillna(0)).sum(axis=1)


def mom_alloc(book_rets: pd.DataFrame,
              window: int = C.MOM_ALLOC_WINDOW,
              min_days: int = C.MOM_ALLOC_MIN_DAYS) -> pd.Series:
    """
    60-day trailing-Sharpe weights (clip >=0, renormalise).
    Falls back to equal-weight until `min_days` of history exist or all Sharpes <=0.
    """
    rs = {}
    for b in book_rets.columns:
        rm = book_rets[b].rolling(window, min_periods=min_days).mean()
        sd = book_rets[b].rolling(window, min_periods=min_days).std()
        rs[b] = (rm / sd * np.sqrt(C.TRADING_DAYS)).clip(lower=0)
    sh = pd.DataFrame(rs).reindex(book_rets.index)

    avail = book_rets.notna()
    sh = sh.where(avail, 0.0)
    tot = sh.sum(axis=1)

    # equal-weight fallback weights
    n = avail.sum(axis=1).replace(0, np.nan)
    ew = avail.div(n, axis=0).fillna(0)

    w = sh.div(tot.replace(0, np.nan), axis=0)
    w = w.where(tot > 0, ew)            # fallback where no positive Sharpe yet
    return (book_rets.fillna(0) * w.fillna(0)).sum(axis=1)


# ─────────────────────────────────────────────────────────────────────
# CHAMPION COMBINER: inverse-vol weights + vol-target overlay (+ panic gate)
# Backtest: Sharpe 2.501-2.514, MaxDD -8.9% (2019-2026) vs Fixed EW 2.060/-19.0%.
# Papers: Moreira-Muir 2017 JF; Harvey+ 2018 JPM; Daniel-Moskowitz 2016 JFE.
# ─────────────────────────────────────────────────────────────────────
def _panic_state() -> pd.Series:
    """Daniel-Moskowitz panic state from SPY daily closes (lagged 1 day):
    trailing 24m return < 0 AND 63d realized vol > trailing-3yr 80th pct."""
    daily = pd.read_parquet(C.DAILY_CLOSE)
    daily.index = pd.to_datetime(daily.index)
    spy = daily["SPY"].dropna()
    ret24 = spy.pct_change(C.PANIC_RET_LOOKBACK)
    vol63 = spy.pct_change().rolling(C.PANIC_VOL_WINDOW).std() * np.sqrt(C.TRADING_DAYS)
    q = vol63.rolling(C.PANIC_VOL_QWINDOW).quantile(C.PANIC_VOL_QUANTILE)
    return ((ret24 < 0) & (vol63 > q)).shift(1).fillna(False)


def ivol_vt_weights(book_rets: pd.DataFrame, panic_today: bool = False) -> dict:
    """Current champion weights over C.ALLOC_BOOKS from trailing book returns.
    Inverse trailing-60d vol, optional panic gate (halve A/F -> D). Weights sum
    to 1 BEFORE the vol-target scale (applied by callers on the return series
    or exposure level). Falls back to equal-weight while history is short."""
    keys = [b for b in C.ALLOC_BOOKS if b in book_rets.columns]
    hist = book_rets[keys].tail(C.IVOL_WINDOW)
    vol = hist.std() * np.sqrt(C.TRADING_DAYS)
    iv = {}
    for b in keys:
        v = float(vol.get(b, np.nan))
        iv[b] = 1.0 / v if np.isfinite(v) and v > 1e-9 else 0.0
    tot = sum(iv.values())
    w = ({b: iv[b] / tot for b in keys} if tot > 0
         else {b: 1.0 / len(keys) for b in keys})
    if panic_today and C.PANIC_GATE:
        freed = 0.0
        for b in ("A", "F"):
            if b in w:
                freed += 0.5 * w[b]
                w[b] *= 0.5
        if "D" in w:
            w["D"] += freed
    return w


def ivol_voltgt(book_rets: pd.DataFrame) -> pd.Series:
    """Full champion daily-return series (for settle/replay): inverse-vol
    weights rebalanced every IVOL_REBAL_DAYS, Daniel-Moskowitz panic gate,
    then a de-risk-only vol-target overlay (min(1, 15% / realized 20d vol))."""
    keys = [b for b in C.ALLOC_BOOKS if b in book_rets.columns]
    R = book_rets[keys].fillna(0.0)
    panic = _panic_state().reindex(R.index).fillna(False) if C.PANIC_GATE \
        else pd.Series(False, index=R.index)
    n = len(R)
    port = np.zeros(n)
    w = None
    for t in range(n):
        if w is None or t % C.IVOL_REBAL_DAYS == 0:
            w = ivol_vt_weights(R.iloc[:t] if t else R.iloc[:1])
        wt = dict(w)
        if bool(panic.iloc[t]) and C.PANIC_GATE:
            freed = 0.0
            for b in ("A", "F"):
                if b in wt:
                    freed += 0.5 * wt[b]
                    wt[b] *= 0.5
            if "D" in wt:
                wt["D"] += freed
        port[t] = float(sum(wt[b] * R.iloc[t][b] for b in keys))
    ser = pd.Series(port, index=R.index)
    rv = ser.rolling(C.VT_VOL_WINDOW).std().shift(1) * np.sqrt(C.TRADING_DAYS)
    scale = (C.VOL_TARGET_ANN / rv).clip(upper=C.VT_MAX_SCALE).fillna(1.0)
    return ser * scale


def vt_scale_today(port_rets: pd.Series) -> float:
    """Current vol-target exposure scale from the champion portfolio's own
    trailing realized vol (de-risk only, <= VT_MAX_SCALE)."""
    rv = float(port_rets.tail(C.VT_VOL_WINDOW).std() * np.sqrt(C.TRADING_DAYS))
    if not np.isfinite(rv) or rv <= 1e-9:
        return 1.0
    return float(min(C.VT_MAX_SCALE, C.VOL_TARGET_ANN / rv))


# ─────────────────────────────────────────────────────────────────────
# EQUAL-WEIGHT DAILY P&L  (validated sizing — fixes the -99% MaxDD bug)
# ─────────────────────────────────────────────────────────────────────
def equal_weight_daily_pnl(trades, prices, opens, idx, day_last, day_first,
                           bar_day_int, daily_ret_cc, U, tc_round_trip, hold_h):
    """
    Given a list of trades (entry_bar, exit_bar, [stock_idx], side) build the daily
    portfolio-return series: each day = equal-weight mean over all positions open that day.

    Returns: pd.Series indexed by trading day (np.datetime64) of daily returns.
    """
    D = len(np.unique(bar_day_int))
    num = np.zeros(D, dtype=np.float64)
    den = np.zeros(D, dtype=np.float64)
    tc_day = tc_round_trip / max(hold_h, 1)

    for (entry_bar, exit_bar, stocks, side) in trades:
        ed = bar_day_int[entry_bar]
        xd = bar_day_int[exit_bar]
        days = np.arange(ed, xd + 1)
        for s in stocks:
            ep = opens[entry_bar, s]
            xp = prices[exit_bar, s]
            if ep <= 0 or xp <= 0 or not (np.isfinite(ep) and np.isfinite(xp)):
                continue
            dr = daily_ret_cc[days, s].copy()
            dc_entry = prices[day_last[ed], s]
            dr[0] = (dc_entry / ep - 1) if dc_entry > 0 else 0.0
            if len(days) > 1:
                pc = prices[day_last[xd - 1], s]
                dr[-1] = (xp / pc - 1) if pc > 0 else 0.0
            dr = np.clip(dr, -0.20, 0.20) * side
            num[days] += dr
            den[days] += 1.0

    active = den > 0
    port = np.zeros(D, dtype=np.float64)
    port[active] = num[active] / den[active] - tc_day
    return port
