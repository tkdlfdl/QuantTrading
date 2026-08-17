"""Universe-wide earnings-date acquisition (rate-limit tolerant).
Checkpoints to data/cache/earnings_dates.parquet every 40 tickers; resumable."""
from __future__ import annotations
import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import pandas as pd
import yfinance as yf
from pathlib import Path

CACHE = Path("data/cache/earnings_dates.parquet")
daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
stocks = [c for c in daily.columns
          if not c.startswith("^") and c not in {"SPY", "UVXY", "QQQ"}]

done = set()
frames = []
if CACHE.exists():
    prev = pd.read_parquet(CACHE)
    frames.append(prev)
    done = set(prev["symbol"].unique())
    print(f"resume: {len(done)} tickers already cached")

todo = [t for t in stocks if t not in done]
print(f"fetching {len(todo)} tickers (throttled)...")
t0 = time.time()
new_frames = []
fail = 0
for k, t in enumerate(todo):
    got = None
    for attempt, pause in enumerate((0, 3, 10)):
        if pause:
            time.sleep(pause)
        try:
            e = yf.Ticker(t).get_earnings_dates(limit=120)
            if e is not None and len(e):
                idx_ = pd.to_datetime(e.index).tz_localize(None).normalize()
                got = pd.DataFrame({"symbol": t, "date": idx_.unique()})
            break
        except Exception:
            continue
    if got is not None:
        new_frames.append(got)
    else:
        fail += 1
    time.sleep(0.4)
    if (k + 1) % 40 == 0 or k == len(todo) - 1:
        allf = frames + new_frames
        if allf:
            pd.concat(allf, ignore_index=True).to_parquet(CACHE)
        print(f"  {k+1}/{len(todo)} ({fail} empty/fail, {time.time()-t0:.0f}s)",
              flush=True)

out = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame()
if len(out):
    per_yr = out.groupby(out["date"].dt.year).size()
    print(f"DONE: {len(out)} rows, {out['symbol'].nunique()} tickers")
    print("rows/yr:", {int(y): int(n) for y, n in per_yr.items() if n > 100})
else:
    print("DONE: nothing fetched")
