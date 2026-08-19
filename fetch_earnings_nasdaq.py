"""Queue #46 alternate route: nasdaq.com per-DAY earnings calendar.
One call per trading day 2018-06..2026-07 (FL test needs coverage-floor
years); filter to universe tickers; checkpoint every 100 days; resumable.
Output: data/cache/earnings_dates.parquet (symbol, date)."""
from __future__ import annotations
import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import pandas as pd
import requests
from pathlib import Path

CACHE = Path("data/cache/earnings_dates.parquet")
daily = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
daily.index = pd.to_datetime(daily.index)
universe = {c for c in daily.columns if not c.startswith("^")}
days = [d for d in daily.index if pd.Timestamp("2018-06-01") <= d <= pd.Timestamp("2026-07-10")]

done_days = set()
frames = []
if CACHE.exists():
    prev = pd.read_parquet(CACHE)
    frames.append(prev)
    done_days = set(pd.to_datetime(prev["date"]).dt.normalize())
    print(f"resume: {len(done_days)} days cached")

HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36",
       "Accept": "application/json"}
t0 = time.time()
new_rows = []
fail = 0
todo = [d for d in days if d.normalize() not in done_days]
print(f"fetching {len(todo)} trading days...")
for k, d in enumerate(todo):
    got = False
    for pause in (0, 2, 6):
        if pause:
            time.sleep(pause)
        try:
            r = requests.get("https://api.nasdaq.com/api/calendar/earnings",
                             params={"date": d.strftime("%Y-%m-%d")},
                             headers=HDR, timeout=15)
            j = r.json()
            rows = (j.get("data") or {}).get("rows") or []
            for row in rows:
                sym = str(row.get("symbol", "")).strip()
                if sym in universe:
                    new_rows.append((sym, d.normalize()))
            got = True
            break
        except Exception:
            continue
    if not got:
        fail += 1
    time.sleep(0.35)
    if (k + 1) % 100 == 0 or k == len(todo) - 1:
        allf = frames + ([pd.DataFrame(new_rows, columns=["symbol", "date"])]
                         if new_rows else [])
        if allf:
            out = pd.concat(allf, ignore_index=True).drop_duplicates()
            out.to_parquet(CACHE)
        print(f"  {k+1}/{len(todo)} days ({len(new_rows)} rows, {fail} fails, "
              f"{time.time()-t0:.0f}s)", flush=True)

out = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame()
if len(out):
    per_yr = out.groupby(pd.to_datetime(out["date"]).dt.year).size()
    print(f"DONE: {len(out)} events, {out['symbol'].nunique()} tickers; "
          f"per-yr {dict(per_yr)}")
else:
    print("DONE: nothing fetched")
