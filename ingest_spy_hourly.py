"""
Data-engineer job (queue #8b-data): add SPY to the merged hourly caches.

Alpaca IEX (2019 -> ~730d ago) + yfinance (last 730d), merged on the existing
merged-cache bar index, appended as a new SPY column to
merged_hourly_open/close.parquet. Backs up the caches first.
Then queue item 8b (Gao et al. SPY intraday momentum) becomes testable.
"""
from __future__ import annotations
import sys, shutil, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

CACHE = "data/cache"
MO, MC = f"{CACHE}/merged_hourly_open.parquet", f"{CACHE}/merged_hourly_close.parquet"

ho = pd.read_parquet(MO); hc = pd.read_parquet(MC)
ho.index = pd.to_datetime(ho.index); hc.index = pd.to_datetime(hc.index)
if "SPY" in hc.columns:
    print("SPY already present — nothing to do"); sys.exit(0)

# ── yfinance leg (last ~730d) ──
yf_start = (datetime.now() - timedelta(days=729)).strftime("%Y-%m-%d")
spy_yf = yf.download("SPY", start=yf_start, interval="1h", auto_adjust=True, progress=False)
if isinstance(spy_yf.columns, pd.MultiIndex):
    spy_yf.columns = spy_yf.columns.get_level_values(0)
spy_yf.index = pd.to_datetime(spy_yf.index).tz_localize(None)
print(f"yfinance leg: {len(spy_yf)} bars {spy_yf.index.min()} .. {spy_yf.index.max()}")

# ── Alpaca leg (2019 -> yf start) ──
o_al = c_al = None
try:
    from data.fetchers.alpaca_fetcher import fetch_alpaca_bars
    o_al, c_al = fetch_alpaca_bars(["SPY"], start="2019-01-01", end=yf_start, feed="iex")
    o_al.index = pd.to_datetime(o_al.index); c_al.index = pd.to_datetime(c_al.index)
    print(f"alpaca leg: {len(c_al)} bars {c_al.index.min()} .. {c_al.index.max()}")
except Exception as e:
    print(f"alpaca leg unavailable ({e}) — proceeding with yfinance only (2 recent yrs)")

# ── build SPY open/close series on the merged index ──
def leg_series(df, col):
    if df is None: return pd.Series(dtype=float)
    s = df["SPY"] if "SPY" in df.columns else df[col]
    return s.dropna()

spy_o = pd.concat([leg_series(o_al, "Open"), spy_yf["Open"]]).sort_index()
spy_c = pd.concat([leg_series(c_al, "Close"), spy_yf["Close"]]).sort_index()
spy_o = spy_o[~spy_o.index.duplicated(keep="last")]
spy_c = spy_c[~spy_c.index.duplicated(keep="last")]

# align to merged bar index (nearest within 31 min — yf uses :30 offsets, merged may too)
spy_o_al = spy_o.reindex(ho.index, method="nearest", tolerance=pd.Timedelta("31min"))
spy_c_al = spy_c.reindex(hc.index, method="nearest", tolerance=pd.Timedelta("31min"))
cov = spy_c_al.notna().mean()
print(f"coverage on merged index: {cov:.1%} "
      f"({spy_c_al.notna().sum()}/{len(hc)} bars)")

for path in (MO, MC):
    shutil.copy2(path, path + ".bak")
ho["SPY"] = spy_o_al; hc["SPY"] = spy_c_al
ho.to_parquet(MO); hc.to_parquet(MC)
print(f"SPY appended to merged caches (backups: *.bak). "
      f"first bar {spy_c_al.first_valid_index()}, last {spy_c_al.last_valid_index()}")
