"""
REDDIT SENTIMENT EUPHORIA -> SHORT  (backtest)
================================================
Thesis: when a stock's Reddit sentiment gets euphoric (sentiment bubble score too high),
fade it — SHORT for several days, betting the hype mean-reverts.

Signal:
  1. Build a per-symbol sentiment "price index" from daily weighted_compound score.
  2. Bubble score = tanh(z/2) on that index (same proxy as the price-bubble strategies).
  3. If bubble score > threshold AND recent mentions >= min_mentions  -> SHORT next day.
  4. Hold `hold_days`, equal-weight across the top-N most-euphoric names.

Costs: 0.25% one-way TC + 8%/yr short borrow.
Data : sentiment_daily (2024-03-31..2026-06-01) + daily close prices.
"""
import warnings, os, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, duckdb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product

t0 = time.time()
RF = 0.02; BORROW = 0.08; TC = 0.0025; TD = 252

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
print("Loading sentiment + price data...")
con = duckdb.connect("data/market_data.duckdb", read_only=True)
sd = con.execute("select date, symbol, weighted_compound, mention_count from sentiment_daily").df()
sd["date"] = pd.to_datetime(sd["date"])

sent = sd.pivot_table(index="date", columns="symbol", values="weighted_compound", aggfunc="mean")
ment = sd.pivot_table(index="date", columns="symbol", values="mention_count", aggfunc="sum").fillna(0)

prices = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
prices.index = pd.to_datetime(prices.index)

# Align to a daily business calendar over the sentiment window
cal = pd.bdate_range(sent.index.min(), sent.index.max())
syms = [s for s in sent.columns if s in prices.columns]
sent = sent.reindex(cal)[syms]
ment = ment.reindex(cal)[syms].fillna(0)
px   = prices.reindex(cal)[syms].ffill()
ret  = px.pct_change().fillna(0)
n    = len(cal)
years = (cal[-1] - cal[0]).days / 365.25
print(f"Universe: {len(syms)} symbols  |  {cal[0].date()} -> {cal[-1].date()}  ({years:.2f} yrs)")

# Cumulative mentions (eligibility, no look-ahead)
cum_ment = (ment > 0).cumsum()

# ─────────────────────────────────────────────────────────────
# SENTIMENT BUBBLE SCORE
# ─────────────────────────────────────────────────────────────
def sentiment_index(s, scale=0.05, base=100.0):
    return base * (1 + s.fillna(0).clip(-1, 1) * scale).cumprod()

def bubble(price, ma, z):
    lp = np.log(price.replace(0, np.nan).ffill())
    fair = price.rolling(ma).mean()
    res = lp - np.log(fair)
    zz = (res - res.rolling(z).mean()) / res.rolling(z).std()
    return np.tanh(zz / 2)

# ─────────────────────────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────────────────────────
# Precompute bubble scores ONCE per (ma, zw) pair — reused across the grid
_BUB_CACHE = {}
def _bubble_panel(ma, zw):
    key = (ma, zw)
    if key not in _BUB_CACHE:
        bub = pd.DataFrame(index=cal, columns=syms, dtype=float)
        for s in syms:
            bub[s] = bubble(sentiment_index(sent[s]), ma, zw)
        _BUB_CACHE[key] = bub.shift(1)   # no look-ahead
    return _BUB_CACHE[key]


def run(ma, zw, thr, hold, top_n, min_ment):
    sig = _bubble_panel(ma, zw)
    warmup = ma + zw + 1
    rets, dates = [], []
    i = warmup
    while i < n - hold:
        elig = cum_ment.iloc[i-1]
        elig = elig[elig >= min_ment].index
        sc = sig.iloc[i][elig].dropna()
        hot = sc[sc > thr].nlargest(top_n)   # most euphoric
        if len(hot) == 0:
            i += 1
            continue
        # short return over hold window
        fwd = ret.iloc[i:i+hold][hot.index]
        short_ret = -(fwd.mean(axis=1)).sum()         # equal-weight short, summed over hold days
        short_ret -= BORROW / TD * hold               # borrow cost
        short_ret -= TC                               # round-trip cost
        rets.append(short_ret)
        dates.append(cal[i])
        i += hold
    if len(rets) < 5:
        return None
    return pd.Series(rets, index=pd.DatetimeIndex(dates))

def perf(s):
    if s is None or len(s) < 5: return None
    yrs = (s.index[-1]-s.index[0]).days/365.25
    tpy = len(s)/yrs
    w = (1+s).cumprod()
    tot = w.iloc[-1]-1
    ann = (1+tot)**(1/yrs)-1 if tot>-1 else -1
    sh = (s.mean()-RF/tpy)/s.std()*np.sqrt(tpy) if s.std()>0 else 0
    dn = s[s<0].std(ddof=0)
    so = (s.mean()-RF/tpy)/dn*np.sqrt(tpy) if dn>0 else 0
    dd = (w/w.cummax()-1).min()
    return dict(ann=ann, sharpe=sh, sortino=so, maxdd=dd, total=tot,
                trades=len(s), win=(s>0).mean())

# ─────────────────────────────────────────────────────────────
# GRID
# ─────────────────────────────────────────────────────────────
MA=[20,30]; ZW=[30,60]; THR=[0.5,0.6,0.7,0.8]; HOLD=[1,2,3,5]; TOPN=[3,5]; MINM=[3,5]
total=len(MA)*len(ZW)*len(THR)*len(HOLD)*len(TOPN)*len(MINM)
print(f"\nGrid: {total} combinations (short-on-euphoria)...")
rows=[]
for ma,zw,thr,hold,top_n,mm in product(MA,ZW,THR,HOLD,TOPN,MINM):
    s=run(ma,zw,thr,hold,top_n,mm)
    p=perf(s)
    if p: rows.append(dict(ma=ma,zw=zw,thr=thr,hold=hold,top_n=top_n,min_ment=mm,**p))

if not rows:
    print("No valid results — sentiment data too sparse for these params.")
    raise SystemExit

df=pd.DataFrame(rows).sort_values("sharpe",ascending=False).reset_index(drop=True)
print(f"Done in {time.time()-t0:.0f}s  ({len(df)} valid combos)")
print(f"\n{'='*100}\nTOP 15 (short when sentiment bubble > threshold)\n{'='*100}")
fmt={"ann":"{:.1%}".format,"sharpe":"{:.3f}".format,"sortino":"{:.3f}".format,
     "maxdd":"{:.1%}".format,"total":"{:.1%}".format,"win":"{:.0%}".format}
print(df.head(15).to_string(index=False,formatters=fmt))
print(f"\nPositive Sharpe: {(df.sharpe>0).sum()}/{len(df)}   Sharpe>0.5: {(df.sharpe>0.5).sum()}   Sharpe>1: {(df.sharpe>1).sum()}")

# sensitivity
print(f"\n{'='*60}\nSENSITIVITY (avg Sharpe)\n{'='*60}")
for c,l in [("thr","Threshold"),("hold","Hold days"),("top_n","Top-N"),("min_ment","Min mentions")]:
    g=df.groupby(c)["sharpe"].agg(["mean","max"]).round(3)
    print(f"\n{l}:\n{g.to_string()}")

# best yearly
best=df.iloc[0]
s=run(int(best.ma),int(best.zw),best.thr,int(best.hold),int(best.top_n),int(best.min_ment))
w=(1+s).cumprod()
print(f"\n{'='*60}\nBEST: ma={int(best.ma)} z={int(best.zw)} thr={best.thr} hold={int(best.hold)}d "
      f"top{int(best.top_n)} minMent={int(best.min_ment)}\n{'='*60}")
print(f"  Annual {best.ann:.1%}  Sharpe {best.sharpe:.3f}  Sortino {best.sortino:.3f}  "
      f"MaxDD {best.maxdd:.1%}  Trades {int(best.trades)}  Win {best.win:.0%}")
print(f"\n  Year     Return   Trades")
for yr in sorted(s.index.year.unique()):
    ys=s[s.index.year==yr]
    print(f"  {yr}   {(1+ys).prod()-1:>+8.1%}   {len(ys):>4}")

os.makedirs("results",exist_ok=True)
df.to_csv("results/reddit_sentiment_short_grid.csv",index=False)
fig,ax=plt.subplots(2,1,figsize=(14,9))
ax[0].plot(w.index,w.values,color="crimson",lw=2)
ax[0].set_title(f"Reddit Sentiment Euphoria->Short  (Ann {best.ann:.0%}, Sharpe {best.sharpe:.2f}, MaxDD {best.maxdd:.0%})",fontweight="bold")
ax[0].set_ylabel("Cumulative wealth"); ax[0].grid(True,alpha=0.3)
dd=w/w.cummax()-1
ax[1].fill_between(dd.index,dd.values*100,0,color="red",alpha=0.5)
ax[1].set_ylabel("Drawdown %"); ax[1].grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig("results/reddit_sentiment_short.png",dpi=140,bbox_inches="tight")
print(f"\nSaved: results/reddit_sentiment_short_grid.csv + .png")
