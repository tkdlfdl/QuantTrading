"""
live/live_book.py
=================
Forward INTRADAY position book for the blended MomAlloc target.

Maintains, per sub-strategy, the set of currently-open virtual positions with hold-expiry
timers (persisted in live_positions.json so holds survive between hourly ticks).  Each tick:

  1. expire positions whose hold has elapsed
  2. open new positions per each book's rule from the latest bar's signals
  3. weight each book equally across its open positions (signed)
  4. blend by the MomAlloc per-book weights
  5. net to per-symbol target weights, apply the gross-exposure cap
  6. convert to target share counts at the latest price

Fidelity notes: B and D are faithful (long, fixed hour-holds). A holds a 140d-momentum
long/short basket refreshed on a 40-trading-day anchor with leverage/UVXY overlays.
C is the approximate book — implemented as the dominant phase-2 flip (3-day hold, signed by
the daily Z move); the 1-hour phase-1 fade is omitted for live execution.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

from . import config as C
from . import signals as S


def _empty_state():
    return {"last_tick": None,
            "books": {"A": {"longs": [], "shorts": [], "anchor": None},
                      "B": [], "C": [], "D": [],
                      "E": {"longs": [], "anchor": None}}}


class LiveBook:
    def __init__(self):
        if C.LIVE_POSITIONS_FILE.exists():
            self.state = json.loads(C.LIVE_POSITIONS_FILE.read_text(encoding="utf-8"))
        else:
            self.state = _empty_state()

    def save(self):
        C.ensure_dirs()
        C.LIVE_POSITIONS_FILE.write_text(json.dumps(self.state, indent=2, default=str),
                                         encoding="utf-8")

    # ─────────────────────────────────────────────────────────────────
    def tick(self, panels, now_ts, book_weights, equity):
        hc, tickers, idx = panels["hourly_close"], panels["tickers"], panels["idx_h"]
        # locate current bar
        pos = idx.searchsorted(pd.Timestamp(now_ts), side="right") - 1
        pos = int(max(0, min(pos, len(idx) - 1)))
        now = idx[pos]
        last_px = hc.iloc[pos]

        self._tick_D(panels, pos, now)
        self._tick_B(panels, pos, now)
        self._tick_C(panels, pos, now)
        self._tick_A(panels, now)
        self._tick_E(panels, now)

        # ── per-book signed weights ──────────────────────────────────
        book_sym_w = {
            "A": self._weights_A(),
            "B": self._weights_list("B"),
            "C": self._weights_list("C"),
            "D": self._weights_list("D"),
            "E": self._weights_E(),
        }

        # ── blend by MomAlloc book weights ───────────────────────────
        target_w = {}
        for b in C.BOOKS:
            bw = float(book_weights.get(b, 0.0))
            for sym, w in book_sym_w[b].items():
                target_w[sym] = target_w.get(sym, 0.0) + bw * w

        # gross-exposure cap
        gross = sum(abs(v) for v in target_w.values())
        if gross > C.GROSS_CAP and gross > 0:
            scale = C.GROSS_CAP / gross
            target_w = {k: v * scale for k, v in target_w.items()}

        # ── to target shares ─────────────────────────────────────────
        prices, shares = {}, {}
        for sym, w in target_w.items():
            px = float(last_px.get(sym, np.nan))
            if not np.isfinite(px) or px <= 0:
                continue
            prices[sym] = px
            shares[sym] = int(round(w * equity / px))
        shares = {k: v for k, v in shares.items() if v != 0}

        attribution = self._attribution(panels, pos)

        self.state["last_tick"] = str(now)
        diag = dict(now=str(now), n_target=len(shares),
                    gross=round(sum(abs(v) for v in target_w.values()), 3),
                    n_open=dict(A=len(self.state["books"]["A"]["longs"]) +
                                  len(self.state["books"]["A"]["shorts"]),
                                B=len(self.state["books"]["B"]),
                                C=len(self.state["books"]["C"]),
                                D=len(self.state["books"]["D"]),
                                E=len(self.state["books"]["E"]["longs"])))
        return target_w, shares, prices, diag, attribution

    # ─────────────────────────────────────────────────────────────────
    # ATTRIBUTION: for each held symbol -> (book, signal value, description)
    # ─────────────────────────────────────────────────────────────────
    def _attribution(self, panels, pos):
        hc, daily, tickers = panels["hourly_close"], panels["daily_close"], panels["tickers"]
        attr = {}

        # Book A — 140d momentum rank
        a = self.state["books"]["A"]
        if a.get("longs") or a.get("shorts"):
            dcols = [t for t in tickers if t in daily.columns]
            mom = S.daily_momentum_rank(daily[dcols], C.PARAMS["A"]["lookback_days"]).iloc[-1]
            ranks = {s: i + 1 for i, s in enumerate(mom.sort_values(ascending=False).index)}
            for s in a.get("longs", []):
                attr[s] = dict(book="A", signal=round(float(mom.get(s, float("nan"))), 4),
                               desc=f"140d momentum #{ranks.get(s,'?')} ({mom.get(s,0)*100:.0f}%)")
            for s in a.get("shorts", []):
                attr[s] = dict(book="A", signal=round(float(mom.get(s, float("nan"))), 4),
                               desc=f"140d momentum bottom ({mom.get(s,0)*100:.0f}%)")

        # Book D — current bubble score
        if self.state["books"]["D"]:
            bub = S.bubble_score_hourly(hc, C.PARAMS["D"]["bubble_ma_hours"]).iloc[pos]
            for x in self.state["books"]["D"]:
                s = x["symbol"]
                attr[s] = dict(book="D", signal=round(float(bub.get(s, float("nan"))), 4),
                               desc=f"bubble {bub.get(s,0):.2f} (buy dip <-0.8)")

        # Book B — momentum picks under QQQ regime
        for x in self.state["books"]["B"]:
            attr.setdefault(x["symbol"], dict(book="B", signal=None, desc="QQQ bubble<-0.8 momentum buy"))

        # Book C — Z-score flip
        for x in self.state["books"]["C"]:
            attr.setdefault(x["symbol"], dict(book="C", signal=x.get("side"),
                                              desc=f"Z-flip side={x.get('side')}"))

        # Book E — Reddit sentiment long
        for s in self.state["books"].get("E", {}).get("longs", []):
            attr.setdefault(s, dict(book="E", signal=None,
                                    desc="reddit sentiment long (capitulation/hype)"))
        return attr

    # ─────────────────────────────────────────────────────────────────
    # BOOK D — long contrarian, hold 13h
    # ─────────────────────────────────────────────────────────────────
    def _tick_D(self, panels, pos, now):
        p = C.PARAMS["D"]
        hc, tickers = panels["hourly_close"], panels["tickers"]
        bub = S.bubble_score_hourly(hc, p["bubble_ma_hours"]).iloc[pos]
        book = [x for x in self.state["books"]["D"]
                if pd.Timestamp(x["exit_ts"]) > now]      # expire
        held = {x["symbol"] for x in book}
        depressed = bub[bub < p["threshold"]].sort_values()
        for sym in depressed.index:
            if len(book) >= p["top_n"]:
                break
            if sym in held:
                continue
            exit_ts = now + pd.Timedelta(hours=p["hold_hours"])
            book.append(dict(symbol=sym, side=1, entry_ts=str(now), exit_ts=str(exit_ts)))
            held.add(sym)
        self.state["books"]["D"] = book

    # ─────────────────────────────────────────────────────────────────
    # BOOK B — QQQ bubble triggers top-5 momentum longs, hold 52h
    # ─────────────────────────────────────────────────────────────────
    def _tick_B(self, panels, pos, now):
        p = C.PARAMS["B"]
        hc = panels["hourly_close"]; qqq = panels["qqq_hourly"]
        book = [x for x in self.state["books"]["B"]
                if pd.Timestamp(x["exit_ts"]) > now]
        if len(book) == 0 and qqq is not None and pos < len(qqq):
            bq = float(S.qqq_bubble(qqq, p["qqq_bubble_ma_hours"]).iloc[pos])
            if bq < p["threshold"]:
                mom = S.momentum_hours(hc, p["mom_lookback_hours"]).iloc[pos]
                picks = mom.sort_values(ascending=False).head(p["top_n"]).index
                exit_ts = now + pd.Timedelta(hours=p["hold_hours"])
                book = [dict(symbol=s, side=1, entry_ts=str(now), exit_ts=str(exit_ts))
                        for s in picks]
        self.state["books"]["B"] = book

    # ─────────────────────────────────────────────────────────────────
    # BOOK C — approximate phase-2 flip, hold 3d, signed by daily Z
    # ─────────────────────────────────────────────────────────────────
    def _tick_C(self, panels, pos, now):
        p = C.PARAMS["C"]
        book = [x for x in self.state["books"]["C"]
                if pd.Timestamp(x["exit_ts"]) > now]
        last_date = self.state.get("last_tick")
        is_new_session = (last_date is None) or (pd.Timestamp(last_date).date() != now.date())
        if is_new_session:
            daily = panels["daily_close"]; tickers = panels["tickers"]
            dcols = [t for t in tickers if t in daily.columns]
            dret = daily[dcols].pct_change()
            z = S.zscore_daily(dret, p["z_lookback_days"]).iloc[-1].dropna()
            big = z[z.abs() > p["sigma"]]
            big = big.reindex(big.abs().sort_values(ascending=False).index).head(p["top_n"])
            held = {x["symbol"] for x in book}
            exit_ts = now + pd.Timedelta(days=p["flip_hold_days"])
            for sym, zv in big.items():
                if sym in held:
                    continue
                book.append(dict(symbol=sym, side=int(np.sign(zv)),
                                 entry_ts=str(now), exit_ts=str(exit_ts)))
        self.state["books"]["C"] = book

    # ─────────────────────────────────────────────────────────────────
    # BOOK A — 140d momentum basket on a 40-day anchor + lev/hedge state
    # ─────────────────────────────────────────────────────────────────
    def _tick_A(self, panels, now):
        p = C.PARAMS["A"]
        a = self.state["books"]["A"]
        anchor = a.get("anchor")
        need = anchor is None
        if anchor is not None:
            held_days = np.busday_count(pd.Timestamp(anchor).date(), now.date())
            need = held_days >= p["rebalance_days"]
        if need:
            daily = panels["daily_close"]; tickers = panels["tickers"]
            dcols = [t for t in tickers if t in daily.columns]
            mom = S.daily_momentum_rank(daily[dcols], p["lookback_days"]).iloc[-1].dropna()
            ranked = mom.sort_values(ascending=False)
            a["longs"]  = ranked.head(p["top_n"]).index.tolist()
            a["shorts"] = []   # Book A is long-only
            a["anchor"] = str(now.date())
        self.state["books"]["A"] = a

    # ─────────────────────────────────────────────────────────────────
    # per-book signed weight vectors
    # ─────────────────────────────────────────────────────────────────
    def _weights_list(self, book_key):
        book = self.state["books"][book_key]
        if not book:
            return {}
        w = 1.0 / len(book)
        out = {}
        for x in book:
            out[x["symbol"]] = out.get(x["symbol"], 0.0) + x["side"] * w
        return out

    def _weights_A(self):
        a = self.state["books"]["A"]
        longs, shorts = a.get("longs", []), a.get("shorts", [])
        n = len(longs) + len(shorts)
        if n == 0:
            return {}
        w = 1.0 / n
        out = {}
        for s in longs:
            out[s] = out.get(s, 0.0) + w
        for s in shorts:
            out[s] = out.get(s, 0.0) - w
        return out

    # ─────────────────────────────────────────────────────────────────
    # BOOK E — Reddit sentiment long-only basket (8-day hold)
    # ─────────────────────────────────────────────────────────────────
    def _tick_E(self, panels, now):
        p = C.PARAMS["E"]
        e = self.state["books"].get("E", {"longs": [], "anchor": None})
        anchor = e.get("anchor")
        need = anchor is None
        if anchor is not None:
            held = np.busday_count(pd.Timestamp(anchor).date(), now.date())
            need = held >= p["hold_days"]
        if need:
            longs, _info = S.sentiment_longs(set(panels["tickers"]))
            e["longs"] = longs
            e["anchor"] = str(now.date())
        self.state["books"]["E"] = e

    def _weights_E(self):
        longs = self.state["books"].get("E", {}).get("longs", [])
        if not longs:
            return {}
        w = 1.0 / len(longs)
        return {s: w for s in longs}   # long-only
