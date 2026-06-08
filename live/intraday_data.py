"""
live/intraday_data.py
=====================
Light real-time-ish bar refresh for the intraday loop.

Fetches TODAY's hourly bars for the universe from Alpaca (IEX feed, ~15min delayed) and
splices them onto the cached merged hourly panel in memory, so signals can be computed on
an up-to-the-hour panel.  Falls back to the cached panel alone if the fetch fails or no
credentials are present (dry-run still works on cached data).
"""
from __future__ import annotations
import datetime as dt
import pandas as pd

from . import config as C
from . import signals as S


def refresh_today(verbose=True):
    """
    Returns the panels dict (same shape as signals.load_panels) with today's bars
    spliced on when available.
    """
    panels = S.load_panels()
    key, sec = C.load_alpaca_creds()
    if not (key and sec):
        if verbose:
            print("  [intraday] no Alpaca creds -> using cached panel only.")
        return panels

    try:
        from data.fetchers.alpaca_fetcher import fetch_alpaca_bars
        today = dt.date.today().strftime("%Y-%m-%d")
        start = (dt.date.today() - dt.timedelta(days=3)).strftime("%Y-%m-%d")

        # Alpaca uses dotted class-share symbols (BRK.B) vs our dashed (BRK-B).
        fetch_syms = [alpaca_symbol(s) for s in (panels["tickers"] + ["QQQ"])]
        ho_new, hc_new = fetch_alpaca_bars(
            fetch_syms, start=start, end=today,
            timeframe="1Hour", feed="iex", api_key=key, secret_key=sec, chunk_size=50)
        if hc_new.empty:
            if verbose:
                print("  [intraday] no fresh bars returned -> cached panel.")
            _realign_qqq(panels)
            return panels

        # map Alpaca dotted columns back to our dashed convention
        hc_new = hc_new.rename(columns=panel_symbol_map(hc_new.columns))
        ho_new = ho_new.rename(columns=panel_symbol_map(ho_new.columns))

        qqq_fresh = hc_new["QQQ"] if "QQQ" in hc_new.columns else None

        hc = _splice(panels["hourly_close"], hc_new)
        ho = _splice(panels["hourly_open"], ho_new)
        idx = hc.index.intersection(ho.index)
        hc, ho = hc.loc[idx], ho.loc[idx]
        panels["hourly_close"] = hc.ffill()
        panels["hourly_open"]  = ho.ffill()
        panels["idx_h"] = idx

        # QQQ: splice fresh QQQ onto the original series, then realign to idx
        if qqq_fresh is not None and panels.get("qqq_hourly") is not None:
            qf = qqq_fresh.copy()
            qf.index = pd.DatetimeIndex(qf.index).floor("h")
            qf = qf[~qf.index.duplicated(keep="last")]
            merged = pd.concat([panels["qqq_hourly"], qf])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            panels["qqq_hourly"] = merged
        _realign_qqq(panels)

        if verbose:
            print(f"  [intraday] spliced fresh bars through {idx.max()}.")
    except Exception as e:
        if verbose:
            print(f"  [intraday] fresh-bar fetch failed ({e}) -> cached panel.")
        _realign_qqq(panels)
    return panels


def alpaca_symbol(sym: str) -> str:
    """Our dashed convention -> Alpaca dotted (BRK-B -> BRK.B)."""
    return sym.replace("-", ".")


def panel_symbol_map(cols) -> dict:
    """Alpaca dotted columns -> our dashed convention (BRK.B -> BRK-B)."""
    return {c: c.replace(".", "-") for c in cols if "." in c}


def _realign_qqq(panels):
    """Ensure qqq_hourly is aligned to idx_h (prevents iloc out-of-bounds)."""
    q = panels.get("qqq_hourly")
    if q is not None:
        panels["qqq_hourly"] = q.reindex(panels["idx_h"]).ffill()


def _splice(base: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    new = new.copy()
    new.index = pd.DatetimeIndex(new.index).floor("h")
    new = new[~new.index.duplicated(keep="last")]
    cols = [c for c in base.columns if c in new.columns]
    merged = pd.concat([base, new[cols].reindex(columns=base.columns)])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged
