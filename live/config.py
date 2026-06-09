"""
live/config.py
==============
Central configuration for the daily paper-trading track-record engine.

All locked best-parameters come from the validated strategy .md files:
  - CONTRARIAN_BUBBLE_STRATEGY.md          (Book D)
  - QQQ_BUBBLE_MOMENTUM_HOURLY_STRATEGY.md (Book B)
  - INTRADAY_MEAN_REVERSION_STRATEGY.md    (Book C)
  - MOMENTUM_LEVERAGE_UVXY_COMPLETE_STRATEGY.md (Book A)

Transaction cost = 0.25% one-way (0.5% round-trip), per the user's latest setting.
"""
from __future__ import annotations
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent          # Trading/
LIVE_DIR    = ROOT / "live"
STATE_DIR   = LIVE_DIR / "state"
PLANS_DIR   = STATE_DIR / "plans"
REPORTS_DIR = LIVE_DIR / "reports"
CACHE_DIR   = ROOT / "data" / "cache"

# Data caches (refreshed by prepare_data.py via data/intraday_loader.py)
MERGED_HOURLY_CLOSE = CACHE_DIR / "merged_hourly_close.parquet"
MERGED_HOURLY_OPEN  = CACHE_DIR / "merged_hourly_open.parquet"
DAILY_CLOSE         = CACHE_DIR / "daily_close_extended_1997_2026.parquet"
QQQ_HOURLY_CLOSE    = CACHE_DIR / "qqq_hourly_close.parquet"

# State files
META_FILE      = STATE_DIR / "meta.json"
POSITIONS_FILE = STATE_DIR / "positions.json"
EQUITY_FILE    = STATE_DIR / "equity.csv"
TRADES_FILE    = STATE_DIR / "trades.csv"

# Reports
TRACK_RECORD_MD = REPORTS_DIR / "track_record.md"
DASHBOARD_PNG   = REPORTS_DIR / "dashboard.png"

# ── Capital & cost model ────────────────────────────────────────────
CAPITAL_PER_BOOK = 100_000.0     # notional $ per book (cosmetic; ratios are return-based)
TC_ONE_WAY       = 0.0025        # 0.25% one-way  → 0.5% round-trip
SHORT_BORROW_ANN = 0.08          # 8%/yr borrow on short legs (Book C)
TRADING_DAYS     = 252
TRADING_HOURS    = 6.5           # regular-session hours per day
NAN_MAX          = 0.30          # drop tickers with >30% NaN (per-book where relevant)

# ── Books ───────────────────────────────────────────────────────────
# Individual strategy books + two derived portfolios.
BOOKS = ["A", "B", "C", "D", "E"]
PORTFOLIOS = ["FixedEW", "MomAlloc"]
ALL_BOOKS = BOOKS + PORTFOLIOS

BOOK_LABELS = {
    "A": "Daily Momentum + Leverage + UVXY",
    "B": "QQQ Bubble Hourly Momentum",
    "C": "Intraday MR + Momentum Flip",
    "D": "Contrarian Bubble Score",
    "E": "Reddit Sentiment Long-Only",
    "FixedEW": "Fixed Equal-Weight Portfolio",
    "MomAlloc": "Momentum-Allocation Portfolio",
}

# ── Locked parameters per book ──────────────────────────────────────
PARAMS = {
    # A: Daily Momentum + 1.25x Leverage + UVXY hedge
    "A": dict(
        lookback_days=140, rebalance_days=40, top_n=5,
        bubble_ma_days=120, bubble_z_days=240,
        lev_threshold=-0.88, lev_mult=0.25, lev_hold_days=50,
        hedge_threshold=0.85, hedge_alloc=0.50, hedge_hold_days=40,
        tc_per_cycle=0.010,                 # 0.5% round-trip per 40-day cycle
        lev_cost_ann=0.10,
    ),
    # B: QQQ bubble triggers top-5 momentum stock buys
    "B": dict(
        qqq_bubble_ma_hours=500, threshold=-0.8,
        mom_lookback_hours=40, hold_hours=52, top_n=5,
        tc_one_way=TC_ONE_WAY,
    ),
    # C: Intraday mean-reversion + momentum flip
    "C": dict(
        z_lookback_days=20, sigma=4.0, top_n=5,
        phase1_hold_hours=1, flip_hold_days=3,
        tc_per_phase=TC_ONE_WAY, short_borrow_ann=SHORT_BORROW_ANN,
    ),
    # D: Contrarian bubble — buy deeply depressed stocks
    "D": dict(
        bubble_ma_hours=104, threshold=-0.8,
        hold_hours=13, top_n=20,
        tc_one_way=TC_ONE_WAY,
    ),
    # E: Reddit sentiment long-only — buy capitulation + moderate hype (no shorts)
    "E": dict(
        ma_window=15, z_window=40, mild=0.5, extreme=0.6,
        hold_days=8, top_n=5, min_mentions=5,
        sentiment_scale=0.05, tc_one_way=TC_ONE_WAY,
    ),
}

# Sentiment data source (DuckDB) for Book E
SENTIMENT_DB = ROOT / "data" / "market_data.duckdb"
REDDIT_CREDS_FILE = STATE_DIR / "reddit_creds.json"


def load_reddit_creds():
    """Reddit API creds for PRAW. env REDDIT_CLIENT_ID/SECRET -> reddit_creds.json -> (None,None)."""
    import os, json
    cid = os.environ.get("REDDIT_CLIENT_ID")
    csec = os.environ.get("REDDIT_CLIENT_SECRET")
    if cid and csec:
        return cid, csec
    if REDDIT_CREDS_FILE.exists():
        try:
            d = json.loads(REDDIT_CREDS_FILE.read_text(encoding="utf-8"))
            return d.get("client_id"), d.get("client_secret")
        except Exception:
            pass
    return None, None

# Momentum-allocation portfolio settings
MOM_ALLOC_WINDOW   = 60     # trailing days for rolling Sharpe weighting
MOM_ALLOC_MIN_DAYS = 10     # fall back to equal-weight until this much history exists

# Book availability (forward-only, but kept for completeness/benchmarks)
# In forward-only mode all four books start together on inception day.

RISK_FREE_ANN = 0.02

# ── Broker / live execution (Alpaca paper) ──────────────────────────
LIVE_BOOK     = "MomAlloc"        # which book the Alpaca account mirrors
DRY_RUN       = True              # default: log orders, never submit (also needs --live)
GROSS_CAP     = 1.0               # max gross exposure as a fraction of account equity
MIN_ORDER_USD = 50.0              # skip reconciling deltas below this notional
STUB_EQUITY   = 100_000.0         # equity used in dry-run when no broker connection

ALPACA_PAPER_ENDPOINT = "https://paper-api.alpaca.markets"
ALPACA_CREDS_FILE     = STATE_DIR / "alpaca_creds.json"

# Intraday live-execution state
LIVE_POSITIONS_FILE = STATE_DIR / "live_positions.json"
ORDERS_LOG          = STATE_DIR / "orders.log"
BROKER_EQUITY_FILE  = STATE_DIR / "broker_equity.csv"
INTRADAY_CACHE      = CACHE_DIR / "intraday_today.parquet"   # today's spliced bars

# US market session (Eastern) — the gate is enforced in run_intraday
MARKET_OPEN_ET  = (9, 30)
MARKET_CLOSE_ET = (16, 0)


def load_alpaca_creds():
    """
    Resolve Alpaca paper credentials.
    Order: env vars (ALPACA_API_KEY / ALPACA_SECRET_KEY) -> alpaca_creds.json -> (None,None).
    """
    import os, json
    key = os.environ.get("ALPACA_API_KEY")
    sec = os.environ.get("ALPACA_SECRET_KEY")
    if key and sec:
        return key, sec
    if ALPACA_CREDS_FILE.exists():
        try:
            d = json.loads(ALPACA_CREDS_FILE.read_text(encoding="utf-8"))
            return d.get("api_key"), d.get("secret_key")
        except Exception:
            pass
    return None, None


def ensure_dirs() -> None:
    """Create state/report directories if missing."""
    for d in (STATE_DIR, PLANS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
