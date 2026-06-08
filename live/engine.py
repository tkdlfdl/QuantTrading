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
