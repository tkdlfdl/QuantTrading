"""Shared feature engineering for Cycle 27 v2 (see brief ml_classification_2026-08).

All features are causal (use data through day t only): rolling windows on
prices/returns, .shift(1) applied by CALLERS where the convention requires.
Works on a price DataFrame (columns = instruments) or a single Series.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def tech_features(px: pd.DataFrame | pd.Series) -> dict:
    """Per-instrument technical features. Returns dict name -> same-shape obj."""
    if isinstance(px, pd.Series):
        px = px.to_frame("_x")
        squeeze = True
    else:
        squeeze = False
    ret = px.pct_change()
    F = {}
    for l in (1, 3, 5):
        F[f"lag{l}"] = ret.rolling(l).sum()
    for w in (21, 63, 126, 252):
        F[f"mom{w}"] = px.pct_change(w)
    F["vol21"] = ret.rolling(21).std()
    F["volratio"] = ret.rolling(5).std() / ret.rolling(60).std()
    F["skew63"] = ret.rolling(63).skew()
    mu63 = ret.rolling(63).mean()
    sd63 = ret.rolling(63).std()
    F["zself"] = (ret - mu63) / sd63
    ma20 = px.rolling(20).mean()
    ma50 = px.rolling(50).mean()
    ma200 = px.rolling(200).mean()
    F["pma20"] = px / ma20 - 1
    F["pma50"] = px / ma50 - 1
    F["pma200"] = px / ma200 - 1
    F["macross"] = ma20 / ma50 - 1
    F["bollz"] = (px - ma20) / px.rolling(20).std()
    delta = px.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    F["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    ema12 = px.ewm(span=12, adjust=False).mean()
    ema26 = px.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26) / px
    F["macd"] = macd
    F["macd_sig"] = macd - macd.ewm(span=9, adjust=False).mean()
    hi252 = px.rolling(252).max()
    lo252 = px.rolling(252).min()
    F["hi52"] = px / hi252 - 1
    F["lo52"] = px / lo252 - 1
    lo14 = px.rolling(14).min()
    hi14 = px.rolling(14).max()
    F["stoch14"] = (px - lo14) / (hi14 - lo14).replace(0, np.nan)
    F["atr14"] = ret.abs().rolling(14).mean()
    if squeeze:
        F = {k: v["_x"] for k, v in F.items()}
    return F


def market_features(index_like: pd.DatetimeIndex, daily: pd.DataFrame,
                    breadth_px: pd.DataFrame | None = None) -> pd.DataFrame:
    """Market-context features aligned to index_like."""
    mk = pd.DataFrame(index=index_like)
    spy = daily["SPY"].dropna()
    spy_r = spy.pct_change()
    mk["spy_mom21"] = spy.pct_change(21).reindex(index_like)
    mk["spy_mom63"] = spy.pct_change(63).reindex(index_like)
    mk["spy_vol21"] = spy_r.rolling(21).std().reindex(index_like)
    if "^VIX" in daily.columns:
        vix = daily["^VIX"].dropna()
        mk["vix_z"] = ((vix - vix.rolling(252).mean())
                       / vix.rolling(252).std()).reindex(index_like).ffill()
        mk["vix_ch5"] = vix.pct_change(5).reindex(index_like).ffill()
    try:
        vts = pd.read_parquet("data/cache/vix_term_structure.parquet")
        vts.index = pd.to_datetime(vts.index)
        mk["vix_ts"] = (vts["^VIX"] / vts["^VIX3M"]).reindex(index_like).ffill()
        mk["vix_front"] = (vts["^VIX9D"] / vts["^VIX"]).reindex(index_like).ffill()
    except Exception:
        pass
    try:
        yl = pd.read_parquet("data/cache/yields_daily.parquet")
        yl.index = pd.to_datetime(yl.index)
        mk["tnx_ch21"] = yl["^TNX"].diff(21).reindex(index_like).ffill()
        mk["slope"] = (yl["^TNX"] - yl["^IRX"]).reindex(index_like).ffill()
    except Exception:
        pass
    if breadth_px is not None:
        mk["breadth20"] = (breadth_px > breadth_px.rolling(20).mean()).mean(axis=1).reindex(index_like)
        mk["breadth50"] = (breadth_px > breadth_px.rolling(50).mean()).mean(axis=1).reindex(index_like)
    month = index_like.to_period("M")
    tleft = pd.Series(0, index=index_like)
    for m_, grp in pd.Series(index_like, index=index_like).groupby(month):
        tleft.loc[grp.index] = np.arange(len(grp) - 1, -1, -1)
    tin = pd.Series(0, index=index_like)
    for m_, grp in pd.Series(index_like, index=index_like).groupby(month):
        tin.loc[grp.index] = np.arange(len(grp))
    mk["tom"] = ((tleft <= 3) | (tin <= 2)).astype(float)
    return mk
