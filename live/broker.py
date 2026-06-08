"""
live/broker.py
==============
Thin wrapper over Alpaca's paper TradingClient with a hard dry-run safety gate.

Submission requires ALL of:
  - credentials present (env or alpaca_creds.json)
  - live=True passed to Broker(...)         (the --live flag)
  - config.DRY_RUN is False OR live override

Otherwise every order is LOGGED to orders.log with mode=DRYRUN and NOT submitted.
"""
from __future__ import annotations
import json
import datetime as dt

from . import config as C


class Broker:
    def __init__(self, live: bool = False):
        self.key, self.sec = C.load_alpaca_creds()
        self.have_creds = bool(self.key and self.sec)
        # live submission only when explicitly requested AND creds exist
        self.live = bool(live and self.have_creds)
        self.client = None
        self._connect_error = None
        if self.live:
            try:
                from alpaca.trading.client import TradingClient
                self.client = TradingClient(self.key, self.sec, paper=True)
            except Exception as e:
                self._connect_error = str(e)
                self.client = None
                self.live = False

    # ── account / positions ─────────────────────────────────────────
    def get_account(self) -> dict:
        if self.client is not None:
            try:
                a = self.client.get_account()
                return dict(equity=float(a.equity), cash=float(a.cash),
                            buying_power=float(a.buying_power), live=True)
            except Exception as e:
                self._connect_error = str(e)
        # dry-run / no connection → stubbed account
        return dict(equity=C.STUB_EQUITY, cash=C.STUB_EQUITY,
                    buying_power=C.STUB_EQUITY * 2, live=False)

    def get_positions(self) -> dict:
        if self.client is not None:
            try:
                pos = self.client.get_all_positions()
                return {p.symbol: float(p.qty) for p in pos}
            except Exception as e:
                self._connect_error = str(e)
        return {}

    # ── order logging ────────────────────────────────────────────────
    @staticmethod
    def _log(rows):
        C.ensure_dirs()
        with open(C.ORDERS_LOG, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    # ── reconcile target shares vs held ──────────────────────────────
    def reconcile(self, target_shares: dict, prices: dict) -> list:
        """
        target_shares : {symbol: signed int shares}  (negative = short)
        prices        : {symbol: last price} for min-notional checks
        Returns the list of order dicts (submitted or dry-run logged).
        """
        held = self.get_positions()
        symbols = set(target_shares) | set(held)
        ts = dt.datetime.now().isoformat(timespec="seconds")
        orders = []

        for sym in sorted(symbols):
            tgt = int(round(target_shares.get(sym, 0)))
            cur = int(round(held.get(sym, 0)))
            delta = tgt - cur
            if delta == 0:
                continue
            px = float(prices.get(sym, 0) or 0)
            if px > 0 and abs(delta) * px < C.MIN_ORDER_USD:
                continue
            side = "buy" if delta > 0 else "sell"
            qty = abs(delta)
            mode = "LIVE" if self.live else "DRYRUN"
            order = dict(ts=ts, symbol=sym, side=side, qty=qty,
                         target=tgt, current=cur, price=px,
                         notional=round(qty * px, 2), mode=mode)

            if self.live and self.client is not None:
                try:
                    from alpaca.trading.requests import MarketOrderRequest
                    from alpaca.trading.enums import OrderSide, TimeInForce
                    req = MarketOrderRequest(
                        symbol=sym.replace("-", "."),   # Alpaca dotted class-share format
                        qty=qty,
                        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                    resp = self.client.submit_order(req)
                    order["order_id"] = str(getattr(resp, "id", ""))
                    order["status"] = "submitted"
                except Exception as e:
                    order["status"] = f"error: {e}"
            else:
                order["status"] = "dryrun"
            orders.append(order)

        if orders:
            self._log(orders)
        return orders

    def status_line(self) -> str:
        if self.live:
            return "LIVE (paper account connected)"
        if not self.have_creds:
            return "DRY-RUN (no credentials)"
        if self._connect_error:
            return f"DRY-RUN (connect failed: {self._connect_error})"
        return "DRY-RUN (--live not set)"
