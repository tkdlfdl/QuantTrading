# Live Paper-Trading Track-Record Engine

Closed-loop daily paper trading for the 4 validated strategies + 2 portfolios.
**Forward-only** (starts flat at inception), **0.25% one-way / 0.5% round-trip** TC.

## The daily loop (runs each weekday evening)
```
0. PREPARE  prepare_data.py  — refresh daily+hourly caches (self-stale-managed)
1. SETTLE   settle.py        — replay all books, rebuild live equity + metrics
2. PLAN     plan.py          — write next session's plan (no look-ahead)
3. REPORT   report.py        — render track_record.md + dashboard.png
```
Orchestrated by `run_daily.py` (idempotent — safe to re-run).

## Books
| Book | Strategy | Locked params |
|------|----------|---------------|
| A | Daily Momentum + Leverage + UVXY | 140d mom, 40d rebalance, top5, 1.25x lev / UVXY hedge |
| B | QQQ Bubble Hourly Momentum | QQQ bubble<-0.8 (500h), top5 by 40h mom, hold 52h |
| C | Intraday MR + Momentum Flip | Z>4 (20d), fade 1h then flip 3d, top5 |
| D | Contrarian Bubble | per-stock bubble<-0.8 (104h), hold 13h, top20 |
| FixedEW | Equal-weight portfolio | 1/N of available books |
| MomAlloc | Momentum-allocation portfolio | 60d trailing-Sharpe weights |

## Run manually
```
python -m live.run_daily                 # normal daily cycle (refreshes data)
python -m live.run_daily --no-refresh     # use existing cache
python -m live.run_daily --replay-last 60 # seed a 60-day history (sanity check) then --reset
python -m live.run_daily --reset          # wipe state, start flat (forward-only)
```

## Scheduled automation (Windows Task Scheduler)
Registered task **PaperTradingDaily** runs `live/run_daily.bat` weekdays at 18:07 local,
logging to `live/state/cron.log`. Runs unattended even when Claude/terminal is closed.
```
schtasks /Query  /TN PaperTradingDaily          # check status / next run
schtasks /Run    /TN PaperTradingDaily          # trigger once now
schtasks /Delete /TN PaperTradingDaily /F       # remove the schedule
```

## Outputs
- `live/reports/track_record.md`  — current standings table (per book + portfolios)
- `live/reports/dashboard.png`    — equity curves, drawdown, rolling Sharpe
- `live/state/equity.csv`         — per (date, book): daily_ret, equity, cum_ret, drawdown, Sharpe/Sortino ITD
- `live/state/trades.csv`         — closed-trade log (books B + D)
- `live/state/positions.json`     — current open positions
- `live/state/plans/<date>.json`  — nightly plan per book
- `live/state/meta.json`          — inception, last_settled, capital

## Notes
- Metrics are inception-to-date; short early windows annualize aggressively (expected).
- Data refresh re-downloads when the cache is >7 days stale (slow for ~516 tickers) —
  this is the one fragile step; the evening schedule absorbs it.
- True forward-only inception is recorded in `meta.json`; do not `--replay-last` on the
  live book unless you intend to reseed history.

---

# Alpaca Paper-Broker Execution (intraday, dry-run first)

A parallel live path that mirrors the **MomAlloc** blend into a real Alpaca **paper**
account. Separate from the nightly track record above.

```
broker.py        Alpaca TradingClient wrapper (account/positions/reconcile), dry-run gate
live_book.py     forward intraday position book blending A/B/C/D -> per-symbol target shares
intraday_data.py today's IEX bars spliced onto the cached panel (cached-only if no keys)
run_intraday.py  hourly tick: gate -> refresh -> tick book -> reconcile -> log
```

## Execution model
Each market hour: recompute each sub-book's open positions (D=13h, B=52h, C=3d flip,
A=40d basket), weight by the live 60-day-Sharpe MomAlloc weights, net to per-symbol
targets, cap gross at `GROSS_CAP` (1.0x equity), convert to shares, reconcile the account
with **market** orders. State persists in `live/state/live_positions.json`.

## Safety — double gate
Orders are **logged, never submitted** unless BOTH:
1. credentials present (env `ALPACA_API_KEY`/`ALPACA_SECRET_KEY`, or `state/alpaca_creds.json`)
2. `--live` flag passed
Default is dry-run; `orders.log` records every intended order with `mode=DRYRUN`.

## Run
```
python -m live.run_intraday --once --force      # dry-run one tick (ignore market gate)
python -m live.run_intraday --once              # respect ET market-hours gate
python -m live.run_intraday --once --live       # submit real paper orders (needs keys)
```

## Go live
1. Add paper keys:  `setx ALPACA_API_KEY "PK..."` / `setx ALPACA_SECRET_KEY "..."`
   (or create `live/state/alpaca_creds.json` = `{"api_key":"PK...","secret_key":"..."}`)
2. Dry-run a few ticks, inspect `live/state/orders.log`.
3. When satisfied, run with `--live` (or add `--live` to `run_intraday.bat`).

## Scheduled automation
Task **PaperTradingIntraday** runs `run_intraday.bat` hourly (dry-run, market-gated),
logging to `live/state/intraday.log`. To trade live, edit the bat to append `--live`.
```
schtasks /Query|/Run|/Delete /TN PaperTradingIntraday
```

## Outputs
- `live/state/orders.log`        — every intended/submitted order
- `live/state/broker_equity.csv` — per tick: equity, gross, target/order counts, open A/B/C/D
- `live/state/live_positions.json` — forward intraday position book

## Caveats
- Free IEX feed is ~15 min delayed and thinner than SIP — intraday signals are noisier
  than the backtest. Acceptable for paper.
- Book C is the approximate book (phase-2 flip only; the 1h phase-1 fade is omitted live).
- In dry-run, `get_positions()` returns empty so every tick re-logs the full target as new
  orders; in live mode held positions net out and only deltas are sent.
