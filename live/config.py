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
SHARES_OUTSTANDING  = CACHE_DIR / "shares_outstanding.parquet"

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
# Individual strategy books + derived portfolios.
BOOKS = ["A", "B", "C", "D", "E", "F", "G", "X", "DU"]
# Books eligible for CHAMPION (IvolVT) capital — audit + promotion decisions:
#   B dropped  : honest Sharpe 0.775, ~zero marginal portfolio contribution (2026-08-15)
#   E excluded : audit Sharpe 0.63 full-history; sentiment feed stale since 2026-06
#   D merged   : Book D is a SINGLE BLENDED book of two internal sleeves
#                (hold=8h and 14h, 50/50 — JT overlapping cohorts). Promoted
#                2026-08-16: champion 2.580 -> ~2.78 / MaxDD -8.2%. The blend
#                carries TWO allocation shares (ALLOC_SHARES) to preserve the
#                two-cohort risk budget — validated identical to separate books
#                (corr 0.9996). Do NOT reduce to one share: that reverts the
#                gain (-0.20 Sharpe, p=0.004 — tested 2026-08-16).
#   G INCUBATING: overnight-share cross-section (LPS 2019), added 2026-08-16
#                at ZERO champion weight for a 60-trading-day Phase-2 incubation.
#                Backtest case: standalone 1.05 (top-7), corr 0.13 to champion;
#                at 0.25 ALLOC_SHARES champion 2.788 -> ~2.85 (+0.07, 32/32
#                perturbation cells positive). Promotion to 0.25 shares is
#                MECHANICAL after clean tracking + human sign-off.
# B and E keep running as paper track records; they receive no champion capital.
ALLOC_BOOKS = ["A", "C", "D", "F"]
G_INCUBATION_START = "2026-08-17"    # first settle date of the incubation window
G_INCUBATION_DAYS  = 60              # trading days before promotion review
#   X INCUBATING: cross-asset ETF momentum (Faber GTAA/dual-momentum lineage),
#                added 2026-08-17 at ZERO weight. First non-single-stock book:
#                15 ETFs (bonds/gold/cmdty/intl/REIT/credit), 12-1 mom top-3,
#                UNGATED (grid: ungated uniformly beats gated — champion's
#                vol-target already de-risks), monthly. Backtest: standalone
#                ~0.6, corr 0.22, positive 2008/2020/2022; champ+X @0.25
#                shares +0.115 (p=0.034), 12/12 perturbation cells positive.
X_INCUBATION_START = "2026-08-17"
X_INCUBATION_DAYS  = 60
#   DU INCUBATING: D x momentum "dips in uptrends" (Cycle 15 synthesis) —
#                D signal masked to names with positive 126-trading-day
#                momentum. Standalone 2.57/-6.8%; as 0.25-share third D-family
#                sleeve champ +0.046 at p=0.010 (strongest in program),
#                robustness 3/3 monotone. Zero weight until review.
DU_INCUBATION_START = "2026-08-18"
DU_INCUBATION_DAYS  = 60
# ── Re-baseline + rollback calibration (2026-08-17, post-mortem #35) ──
# Track record re-baselined: inception 2026-08-18 (prior record measured 4
# engine defects, archived in live/state/archive_incident_2026-08/).
# SETTLE-BASIS champion expectation (live costs 0.25%/side + live mechanics,
# recorded portfolio_ivolvt_settle_basis): Sharpe 1.43 / CAGR 24.3% / MaxDD
# -30.9%. Mechanical rollback bound (1.25x settle-basis MaxDD):
ROLLBACK_MAXDD_BOUND = -0.386   # NOTE: mechanically derived; consider a
                                # tighter absolute override per risk appetite
ROLLBACK_SHARPE_FLOOR = -2.37   # settle-basis worst rolling-60d Sharpe
ALLOC_SHARES = {"D": 2.0}   # blended D = two cohort shares in inverse-vol weighting
PORTFOLIOS = ["FixedEW", "MomAlloc", "IvolVT"]
ALL_BOOKS = BOOKS + PORTFOLIOS

BOOK_LABELS = {
    "A": "Daily Momentum + Leverage + UVXY",
    "B": "QQQ Bubble Hourly Momentum",
    "C": "Intraday MR + Momentum Flip",
    "D": "Contrarian Bubble Score",
    "E": "Reddit Sentiment Long-Only",
    "F": "Universe Hourly Momentum Long",
    "G": "Overnight-Share Cross-Section (INCUBATING)",
    "X": "Cross-Asset ETF Momentum (INCUBATING)",
    "DU": "Contrarian Dips-in-Uptrends (INCUBATING)",
    "FixedEW": "Fixed Equal-Weight Portfolio",
    "MomAlloc": "Momentum-Allocation Portfolio",
    "IvolVT": "Champion: Inverse-Vol + 15% Vol-Target (A/C/D/F, D=2-sleeve blend)",
}

# ── Champion (IvolVT) allocator settings ────────────────────────────
# Backtest (retest_construction_methods.py + retest_regime_overlays.py,
# recorded portfolio_ivol_voltgt_nob / portfolio_champ_panic_nob):
#   Sharpe 2.501-2.514, CAGR ~45%, MaxDD -8.9% (2019-2026)
# vs Fixed EW honest baseline 2.060 / -19.0%.  Papers: Moreira-Muir 2017 JF;
# Harvey et al. 2018 JPM; Qian 2005; Daniel-Moskowitz 2016 JFE (panic gate).
IVOL_WINDOW       = 60      # trailing days for book-vol estimation
IVOL_REBAL_DAYS   = 21      # recompute inverse-vol weights every N trading days
VOL_TARGET_ANN    = 0.15    # 15% annualized portfolio vol target
VT_VOL_WINDOW     = 20      # trailing days for realized portfolio vol
VT_MAX_SCALE      = 1.0     # de-risk only — never lever above 1.0
PANIC_GATE        = True    # Daniel-Moskowitz panic state: halve A/F, redeploy to D
PANIC_RET_LOOKBACK = 504    # SPY trailing 24-month return < 0 ...
PANIC_VOL_WINDOW   = 63     # ... AND SPY 63d realized vol ...
PANIC_VOL_QWINDOW  = 756    # ... above its trailing-3yr ...
PANIC_VOL_QUANTILE = 0.80   # ... 80th percentile

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
    # B: QQQ bubble — direct QQQ trade (long-only, mean reversion)
    # Backtested 6yr (2020-2026): MA=200h, Z=100h, buy<-0.8, hold=24h
    # NOTE: live engine (settle.py/plan.py/live_book.py) still uses old
    #       "QQQ bubble -> momentum stocks" logic and needs a full rewrite
    #       to match the backtested strategy (direct QQQ trades, 2-window bubble).
    "B": dict(
        qqq_bubble_ma_hours=200, z_window_hours=100, threshold=-0.8,
        hold_hours=24, top_n=5,        # top_n unused in new strategy; kept for live compat
        mom_lookback_hours=40,         # unused in new strategy; kept for live compat
        tc_one_way=TC_ONE_WAY,
    ),
    # C: Intraday mean-reversion + momentum flip
    "C": dict(
        z_lookback_days=20, sigma=4.0, top_n=5,
        phase1_hold_hours=1, flip_hold_days=3,
        tc_per_phase=TC_ONE_WAY, short_borrow_ann=SHORT_BORROW_ANN,
    ),
    # D: Contrarian bubble — buy deeply depressed stocks.
    # rank="quiet": QUIET-CAPITULATION ordering (registry #127, champion +0.087
    # p=0.022, 5/5 perturbation) — among gate-eligible names prefer LOW
    # turnover-spike (5d/60d turnover ratio, TRUE shares outstanding).
    # D8 sleeve only — quiet on D14 failed rule 4b (+0.036 p=0.175, #130).
    # Fail-soft: if the turnover feed is unavailable, settle falls back to raw
    # score ordering (the pre-quiet behavior) and logs a warning.
    "D": dict(
        bubble_ma_hours=104, threshold=-0.8,
        hold_hours=8, top_n=20,
        rank="quiet", quiet_spike_days=5, quiet_base_days=60,
        tc_one_way=TC_ONE_WAY,
    ),
    # DU: D x momentum double-sort — INCUBATING third D-family sleeve.
    # Same signal/params as D but eligible set masked to stocks with positive
    # trailing 126-trading-day momentum ("dips in uptrends").
    "DU": dict(
        bubble_ma_hours=104, threshold=-0.8,
        hold_hours=8, top_n=20, mom_filter_days=126,
        tc_one_way=TC_ONE_WAY,
    ),
    # D14: INTERNAL sleeve params for blended Book D (not a standalone book).
    # Same signal as D, 14h harvest horizon; settle averages the two sleeves.
    "D14": dict(
        bubble_ma_hours=104, threshold=-0.8,
        hold_hours=14, top_n=20,
        tc_one_way=TC_ONE_WAY,
    ),
    # F: Universe Hourly Momentum Long — top-5 by 750h (~107d) return, hold 200h (~29d)
    # Non-overlapping rebalance. No leverage, no shorts.
    # Backtest 2019-2026: Sharpe 1.642, CAGR 78.8%, MaxDD -39.6%, 74/216 combos > 1.0
    "F": dict(
        lookback_hours=750,   # 750h / 7 bars/day = 107 trading days (~5 months)
        hold_hours=200,       # 200h / 7 bars/day =  29 trading days (~6 weeks)
        top_n=5,
        tc_one_way=TC_ONE_WAY,
    ),
    # G: Overnight-share cross-section (Lou-Polk-Skouras 2019) — INCUBATING.
    # Rank by trailing 252d (overnight - intraday) return spread; long top-7,
    # 21-trading-day holds, non-overlapping. Spec chosen mid-plateau (top-7 =
    # standalone Sharpe peak, NOT the p-argmax) per anti-selection rules.
    "G": dict(
        spread_window_days=252, hold_days=21, top_n=7, min_names=50,
        tc_one_way=TC_ONE_WAY,
    ),
    # X: Cross-asset ETF momentum — INCUBATING. 12-1 momentum, top-3 of 15
    # ETFs, equal-weight, monthly, NO trend gate (mid-plateau spec).
    "X": dict(
        etf_universe=["TLT","IEF","GLD","SLV","DBC","USO","UUP","EEM","EFA",
                      "IWM","VNQ","HYG","LQD","SPY","QQQ"],
        mom_lookback_days=252, mom_skip_days=21, top_n=3,
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
# Book E is DROPPED (no positions) when the latest sentiment data is older than this
# many business days — so a stale signal never trades.
SENTIMENT_MAX_STALE_DAYS = 5


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
LIVE_BOOK     = "IvolVT"          # which book the Alpaca account mirrors (champion, 2026-08-15)
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
