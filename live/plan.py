"""
live/plan.py
============
Phase 1 — generate the next session's PLAN per book, using data strictly through the
latest completed close (NO look-ahead).  Written to state/plans/<next_session>.json
BEFORE that session trades.

  - B (QQQ Bubble): watchlist rule + whether QQQ is currently in a buy regime, and the
    top-5 momentum names that would be bought if the regime holds at next open.
  - D (Contrarian): the stocks currently below the bubble threshold (would be bought
    next open), top-20 most depressed.
  - C (Intraday MR): concrete — stocks whose latest daily |Z|>4 (faded then flipped).
  - A (Daily Mom): current long/short basket + leverage/hedge regime flags.

The plan is intent/expectation; realised fills are produced by settle.py once the
session's bars exist.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

from . import config as C
from . import signals as S


def _next_session(last_date: pd.Timestamp) -> pd.Timestamp:
    return (last_date + pd.offsets.BDay(1)).normalize()


def generate_plan(panels, last_date: pd.Timestamp) -> dict:
    hc = panels["hourly_close"]; tickers = panels["tickers"]
    idx = panels["idx_h"]
    nxt = _next_session(last_date)
    plan = {"for_session": str(nxt.date()), "generated_from_close": str(last_date.date()),
            "books": {}}

    # latest hourly bar index
    t = len(idx) - 1

    # ── D: Contrarian bubble watchlist ──────────────────────────────
    pD = C.PARAMS["D"]
    bubD = S.bubble_score_hourly(hc, pD["bubble_ma_hours"]).iloc[t]
    depressed = bubD[bubD < pD["threshold"]].sort_values()
    pickD = depressed.head(pD["top_n"]).index.tolist()
    plan["books"]["D"] = dict(
        rule=f"buy stocks with bubble<{pD['threshold']} (MA={pD['bubble_ma_hours']}h), "
             f"hold {pD['hold_hours']}h, top {pD['top_n']}",
        watch_now=pickD,
        n_signals=int((bubD < pD["threshold"]).sum()),
    )

    # ── B: QQQ bubble regime + momentum picks ───────────────────────
    pB = C.PARAMS["B"]
    qqq = panels["qqq_hourly"]
    bubQ = float(S.qqq_bubble(qqq, pB["qqq_bubble_ma_hours"]).iloc[t]) if qqq is not None else 0.0
    in_regime = bubQ < pB["threshold"]
    mom = S.momentum_hours(hc, pB["mom_lookback_hours"]).iloc[t]
    pickB = mom.sort_values(ascending=False).head(pB["top_n"]).index.tolist()
    plan["books"]["B"] = dict(
        rule=f"if QQQ bubble<{pB['threshold']} buy top {pB['top_n']} by {pB['mom_lookback_hours']}h "
             f"momentum, hold {pB['hold_hours']}h",
        qqq_bubble_now=round(bubQ, 4),
        buy_regime=bool(in_regime),
        momentum_picks=pickB if in_regime else [],
    )

    # ── C: daily Z-score signal (concrete) ──────────────────────────
    pC = C.PARAMS["C"]
    daily = panels["daily_close"]
    dcols = [tk for tk in tickers if tk in daily.columns]
    dret = daily[dcols].pct_change()
    z = S.zscore_daily(dret, pC["z_lookback_days"]).iloc[-1]
    z = z.dropna()
    longs  = z[z < -pC["sigma"]].nsmallest(pC["top_n"]).index.tolist()
    shorts = z[z >  pC["sigma"]].nlargest(pC["top_n"]).index.tolist()
    plan["books"]["C"] = dict(
        rule=f"|Z|>{pC['sigma']} (lb={pC['z_lookback_days']}d): fade hour-1 then flip {pC['flip_hold_days']}d",
        fade_long_then_short=shorts,   # surged today -> phase1 short, phase2 long? (per module)
        fade_short_then_long=longs,
        n_signals=int(len(longs) + len(shorts)),
    )

    # ── A: current basket + regime (informational) ──────────────────
    pA = C.PARAMS["A"]
    mom_d = S.daily_momentum_rank(daily[dcols], pA["lookback_days"]).iloc[-1].dropna()
    top = mom_d.sort_values(ascending=False)
    plan["books"]["A"] = dict(
        rule=f"140d momentum, rebalance {pA['rebalance_days']}d, top {pA['top_n']} long/short; "
             f"1.25x lev when curve-bubble<{pA['lev_threshold']}, UVXY hedge when >{pA['hedge_threshold']}",
        longs=top.head(pA["top_n"]).index.tolist(),
        shorts=top.tail(pA["top_n"]).index.tolist(),
    )

    return plan


def write_plan(plan: dict) -> str:
    C.ensure_dirs()
    path = C.PLANS_DIR / f"{plan['for_session']}.json"
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return str(path)
