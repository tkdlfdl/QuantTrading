#!/usr/bin/env python
"""Check Alpaca live account balance and positions."""
import os
import json
from pathlib import Path
from alpaca.trading.client import TradingClient

try:
    # Load credentials from config
    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')

    if not (api_key and secret_key):
        creds_file = Path(__file__).parent / "live" / "state" / "alpaca_creds.json"
        if creds_file.exists():
            creds = json.loads(creds_file.read_text())
            api_key = creds.get('api_key')
            secret_key = creds.get('secret_key')

    if not api_key or not secret_key:
        print("Error: ALPACA_API_KEY or ALPACA_SECRET_KEY not found")
        exit(1)

    client = TradingClient(api_key, secret_key)
    account = client.get_account()

    print("=" * 70)
    print("ALPACA LIVE ACCOUNT STATUS")
    print("=" * 70)
    print(f"Account Status:      {account.status}")
    print(f"Portfolio Value:     ${float(account.portfolio_value):,.2f}")
    print(f"Cash Available:      ${float(account.cash):,.2f}")
    print(f"Long Value:          ${float(account.long_market_value):,.2f}")
    print(f"Short Value:         ${float(account.short_market_value):,.2f}")
    print(f"Buying Power:        ${float(account.buying_power):,.2f}")
    print(f"Multiplier:          {account.multiplier}x")
    print()

    if hasattr(account, 'last_equity') and account.last_equity:
        last_equity = float(account.last_equity)
        current_equity = float(account.equity)
        daily_pl = current_equity - last_equity
        daily_ret = (daily_pl / last_equity) * 100 if last_equity != 0 else 0
        print(f"Last Equity:         ${last_equity:,.2f}")
        print(f"Current Equity:      ${current_equity:,.2f}")
        print(f"Daily P&L:           ${daily_pl:,.2f} ({daily_ret:+.2f}%)")

    positions = client.get_all_positions()

    if positions:
        print()
        print("=" * 70)
        print("OPEN POSITIONS")
        print("=" * 70)
        total_pl = 0
        for pos in positions[:30]:
            qty = int(float(pos.qty))
            price = float(pos.current_price)
            market_val = float(pos.market_value)
            pl = float(pos.unrealized_pl)
            pl_pct = float(pos.unrealized_plpc) * 100
            total_pl += pl

            status = "UP" if pl > 0 else "DN" if pl < 0 else "--"
            print(f"[{status}] {pos.symbol:6} {qty:>6} shares @ ${price:>8.2f}  Value: ${market_val:>12,.2f}  P&L: ${pl:>10,.2f} ({pl_pct:>+7.2f}%)")

        print()
        print(f"Total Unrealized P&L: ${total_pl:,.2f}")
        if len(positions) > 30:
            print(f"\n(Showing 30 of {len(positions)} positions)")
    else:
        print("\nNo open positions")

except Exception as e:
    print(f"Error connecting to Alpaca: {e}")
    import traceback
    traceback.print_exc()
