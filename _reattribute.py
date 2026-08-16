"""Current live per-book P&L from Alpaca fills (FIFO realized + open unrealized) + since-inception days."""
import json, datetime as dt
from collections import defaultdict, deque
from live import config as C

oid2book, sym2book = {}, {}
with open("live/state/orders.log", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        b, oid = r.get("book"), r.get("order_id")
        if b and b != "?":
            if oid:
                oid2book[oid] = b
            sym2book.setdefault(r["symbol"], b)

k, s = C.load_alpaca_creds()
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
cli = TradingClient(k, s, paper=True)

orders, until = [], None
while True:
    batch = cli.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500,
                                            until=until, direction="desc", nested=False))
    if not batch:
        break
    orders.extend(batch)
    if len(batch) < 500:
        break
    until = batch[-1].submitted_at
    if len(orders) > 12000:
        break

seen = {}
for o in orders:
    if str(o.status).split(".")[-1].lower() == "filled" and o.filled_qty and float(o.filled_qty) > 0:
        seen[o.id] = o
fills = sorted(seen.values(), key=lambda o: o.filled_at or o.submitted_at)

def book_of(o):
    return oid2book.get(str(o.id)) or sym2book.get(o.symbol, "?")

lots, realized, ntr = defaultdict(deque), defaultdict(float), defaultdict(int)
for o in fills:
    b, sym = book_of(o), o.symbol
    qty, px = float(o.filled_qty), float(o.filled_avg_price)
    side = str(o.side).split(".")[-1].lower()
    key = (b, sym)
    if side == "buy":
        lots[key].append([qty, px])
    else:
        remaining = qty
        for src in [key, ("?", sym)] + [(bb, sym) for (bb, ss) in list(lots) if ss == sym]:
            dq = lots.get(src)
            while remaining > 1e-9 and dq:
                lot = dq[0]
                take = min(remaining, lot[0])
                realized[b] += take * (px - lot[1])
                lot[0] -= take
                remaining -= take
                if lot[0] <= 1e-9:
                    dq.popleft()
            if remaining <= 1e-9:
                break
        ntr[b] += 1

pos = cli.get_all_positions()
st = json.loads(C.LIVE_POSITIONS_FILE.read_text(encoding="utf-8")).get("books", {})
bm = {}
for x in st.get("A", {}).get("longs", []):  bm[x] = "A"
for x in st.get("A", {}).get("shorts", []): bm[x] = "A"
for x in st.get("B", []): bm[x["symbol"]] = "B"
for x in st.get("C", []): bm[x["symbol"]] = "C"
for x in st.get("D", []): bm[x["symbol"]] = "D"
for x in st.get("E", {}).get("longs", []): bm[x] = "E"
unreal, nopen = defaultdict(float), defaultdict(int)
for p in pos:
    b = bm.get(p.symbol) or sym2book.get(p.symbol, "?")
    unreal[b] += float(p.unrealized_pl)
    nopen[b] += 1

acct = cli.get_account()
eq = float(acct.equity)
# days since first fill
first = min((o.filled_at for o in fills if o.filled_at), default=None)
days = (dt.datetime.now(dt.timezone.utc) - first).days if first else 0

LAB = {"A":"Momentum","B":"QQQ Bubble","C":"Intraday MR","D":"Contrarian",
       "E":"Reddit Sentiment","F":"Hourly Momentum","?":"Untagged(early)"}
books = sorted(set(list(realized)+list(unreal)), key=lambda x:(x=="?", x))
print("="*70)
print(f"  LIVE PERFORMANCE  equity ${eq:,.0f}  total {eq-100000:+,.0f} ({(eq/100000-1)*100:+.2f}%)  {days}d live")
print("="*70)
print(f"{'Bk':<3}{'Strategy':<17}{'Realized':>11}{'Unreal':>10}{'Total':>11}{'Sells':>7}{'Open':>6}")
print("-"*70)
tr=tu=0.0
for b in books:
    r,u = realized.get(b,0.0), unreal.get(b,0.0)
    tr+=r; tu+=u
    print(f"{b:<3}{LAB.get(b,b):<17}{r:>+11,.0f}{u:>+10,.0f}{r+u:>+11,.0f}{ntr.get(b,0):>7}{nopen.get(b,0):>6}")
print("-"*70)
print(f"{'':<3}{'TOTAL':<17}{tr:>+11,.0f}{tu:>+10,.0f}{tr+tu:>+11,.0f}")
print(f"\nAccount P&L {eq-100000:+,.2f} | reattributed {tr+tu:+,.2f} | gap {(eq-100000)-(tr+tu):+,.2f}")
print(f"First fill: {first}  ->  {days} calendar days")
