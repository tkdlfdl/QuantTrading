"""
REDDIT SENTIMENT 4-ZONE REGIME  (backtest)
============================================
Bubble score on a per-symbol sentiment index decides the regime:

  score >  extreme              -> SHORT   (sentiment TOO HIGH: fade the euphoria)
  mild  <  score <=  extreme    -> LONG    (positive, not too high: ride the trend)
  -mild <= score <= mild        -> CASH    (neutral: no trade)
  -extreme <= score < -mild     -> SHORT   (negative, not too low: ride the downtrend)
  score < -extreme              -> LONG    (TOO LOW: buy the capitulation)

Costs: 0.25% one-way TC + 8%/yr short borrow + 2%/yr cash yield when flat.
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

# ── DATA ─────────────────────────────────────────────────────
print("Loading sentiment + price data...")
con = duckdb.connect("data/market_data.duckdb", read_only=True)
sd = con.execute("select date, symbol, weighted_compound, mention_count from sentiment_daily").df()
sd["date"] = pd.to_datetime(sd["date"])
sent = sd.pivot_table(index="date", columns="symbol", values="weighted_compound", aggfunc="mean")
ment = sd.pivot_table(index="date", columns="symbol", values="mention_count", aggfunc="sum").fillna(0)
prices = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
prices.index = pd.to_datetime(prices.index)

cal = pd.bdate_range(sent.index.min(), sent.index.max())
syms = [s for s in sent.columns if s in prices.columns]
sent = sent.reindex(cal)[syms]
ment = ment.reindex(cal)[syms].fillna(0)
px   = prices.reindex(cal)[syms].ffill()
ret  = px.pct_change().fillna(0)
n    = len(cal); years = (cal[-1]-cal[0]).days/365.25
cum_ment = (ment > 0).cumsum()
print(f"Universe: {len(syms)} symbols  |  {cal[0].date()} -> {cal[-1].date()}  ({years:.2f} yrs)")

# ── SENTIMENT BUBBLE SCORE ───────────────────────────────────
def sentiment_index(s, scale=0.05, base=100.0):
    return base * (1 + s.fillna(0).clip(-1, 1) * scale).cumprod()
def bubble(price, ma, z):
    lp = np.log(price.replace(0, np.nan).ffill()); fair = price.rolling(ma).mean()
    res = lp - np.log(fair); zz = (res - res.rolling(z).mean()) / res.rolling(z).std()
    return np.tanh(zz/2)

_BUB = {}
def bub_panel(ma, zw):
    if (ma, zw) not in _BUB:
        b = pd.DataFrame(index=cal, columns=syms, dtype=float)
        for s in syms:
            b[s] = bubble(sentiment_index(sent[s]), ma, zw)
        _BUB[(ma, zw)] = b.shift(1)   # no look-ahead
    return _BUB[(ma, zw)]

# ── 4-ZONE BACKTEST ──────────────────────────────────────────
def run(ma, zw, mild, extreme, hold, top_n, min_ment):
    sig = bub_panel(ma, zw); warmup = ma + zw + 1
    rets, dates = [], []
    i = warmup
    while i < n - hold:
        elig = cum_ment.iloc[i-1]; elig = elig[elig >= min_ment].index
        sc = sig.iloc[i][elig].dropna()
        if sc.empty:
            i += 1; continue
        # zones
        mild_long     = sc[(sc > mild) & (sc <= extreme)].nlargest(top_n)     # positive, not too high -> LONG
        extreme_short = sc[sc > extreme].nlargest(top_n)                      # too high -> SHORT
        mild_short    = sc[(sc < -mild) & (sc >= -extreme)].nsmallest(top_n)  # negative, not too low -> SHORT
        extreme_long  = sc[sc < -extreme].nsmallest(top_n)                    # too low -> LONG
        longs  = pd.concat([mild_long, extreme_long]).drop_duplicates()
        shorts = pd.concat([extreme_short, mild_short]).drop_duplicates()
        has_l, has_s = len(longs) > 0, len(shorts) > 0
        if not (has_l or has_s):
            i += 1; continue
        w = 0.5 if (has_l and has_s) else 1.0
        fwd = ret.iloc[i:i+hold]
        r = 0.0; npos = 0
        if has_l:
            r += w * fwd[longs.index].mean(axis=1).sum(); npos += len(longs)
        if has_s:
            r += w * (-fwd[shorts.index].mean(axis=1).sum())
            r -= w * BORROW/TD*hold; npos += len(shorts)
        r -= TC                                  # round-trip cost
        rets.append(r); dates.append(cal[i]); i += hold
    if len(rets) < 5: return None
    return pd.Series(rets, index=pd.DatetimeIndex(dates))

def perf(s):
    if s is None or len(s) < 5: return None
    yrs = (s.index[-1]-s.index[0]).days/365.25; tpy = len(s)/yrs
    w = (1+s).cumprod(); tot = w.iloc[-1]-1
    ann = (1+tot)**(1/yrs)-1 if tot>-1 else -1
    sh = (s.mean()-RF/tpy)/s.std()*np.sqrt(tpy) if s.std()>0 else 0
    dn = s[s<0].std(ddof=0); so = (s.mean()-RF/tpy)/dn*np.sqrt(tpy) if dn>0 else 0
    dd = (w/w.cummax()-1).min()
    return dict(ann=ann, sharpe=sh, sortino=so, maxdd=dd, total=tot, trades=len(s), win=(s>0).mean())

# ── GRID ─────────────────────────────────────────────────────
MA=[20,30]; ZW=[30,60]; MILD=[0.2,0.3,0.4]; EXT=[0.8,0.85,0.9]; HOLD=[2,3,5]; TOPN=[3,5]; MINM=[3,5]
combos=[(ma,zw,m,e,h,t,mm) for ma,zw,m,e,h,t,mm in product(MA,ZW,MILD,EXT,HOLD,TOPN,MINM) if m<e]
print(f"\nGrid: {len(combos)} combinations (4-zone)...")
rows=[]
for ma,zw,m,e,h,t,mm in combos:
    p=perf(run(ma,zw,m,e,h,t,mm))
    if p: rows.append(dict(ma=ma,zw=zw,mild=m,extreme=e,hold=h,top_n=t,min_ment=mm,**p))
df=pd.DataFrame(rows).sort_values("sharpe",ascending=False).reset_index(drop=True)
print(f"Done in {time.time()-t0:.0f}s  ({len(df)} valid combos)")

fmt={"ann":"{:.1%}".format,"sharpe":"{:.3f}".format,"sortino":"{:.3f}".format,
     "maxdd":"{:.1%}".format,"total":"{:.1%}".format,"win":"{:.0%}".format}
print(f"\n{'='*110}\nTOP 15 (4-zone: trend moderate, fade extremes)\n{'='*110}")
print(df.head(15).to_string(index=False,formatters=fmt))
print(f"\nPositive Sharpe: {(df.sharpe>0).sum()}/{len(df)}   >0.5: {(df.sharpe>0.5).sum()}   >1.0: {(df.sharpe>1).sum()}")

print(f"\n{'='*60}\nSENSITIVITY (avg / best Sharpe)\n{'='*60}")
for c,l in [("mild","Mild thr"),("extreme","Extreme thr"),("hold","Hold days"),("top_n","Top-N")]:
    print(f"\n{l}:\n{df.groupby(c)['sharpe'].agg(['mean','max']).round(3).to_string()}")

best=df.iloc[0]
s=run(int(best.ma),int(best.zw),best.mild,best.extreme,int(best.hold),int(best.top_n),int(best.min_ment))
w=(1+s).cumprod()
print(f"\n{'='*60}\nBEST: ma={int(best.ma)} z={int(best.zw)} mild={best.mild} extreme={best.extreme} "
      f"hold={int(best.hold)}d top{int(best.top_n)} minMent={int(best.min_ment)}\n{'='*60}")
print(f"  Annual {best.ann:.1%}  Sharpe {best.sharpe:.3f}  Sortino {best.sortino:.3f}  "
      f"MaxDD {best.maxdd:.1%}  Trades {int(best.trades)}  Win {best.win:.0%}")
print(f"\n  Year     Return   Trades")
for yr in sorted(s.index.year.unique()):
    ys=s[s.index.year==yr]; print(f"  {yr}   {(1+ys).prod()-1:>+8.1%}   {len(ys):>4}")

os.makedirs("results",exist_ok=True)
df.to_csv("results/reddit_sentiment_4zone_grid.csv",index=False)
fig,ax=plt.subplots(2,1,figsize=(14,9))
ax[0].plot(w.index,w.values,color="darkgreen",lw=2)
ax[0].set_title(f"Reddit Sentiment 4-Zone  (Ann {best.ann:.0%}, Sharpe {best.sharpe:.2f}, MaxDD {best.maxdd:.0%})",fontweight="bold")
ax[0].set_ylabel("Cumulative wealth"); ax[0].grid(True,alpha=0.3)
dd=w/w.cummax()-1; ax[1].fill_between(dd.index,dd.values*100,0,color="red",alpha=0.5)
ax[1].set_ylabel("Drawdown %"); ax[1].grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig("results/reddit_sentiment_4zone.png",dpi=140,bbox_inches="tight")
print(f"\nSaved: results/reddit_sentiment_4zone_grid.csv + .png")
