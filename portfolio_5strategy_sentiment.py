"""
PORTFOLIO IMPACT OF ADDING BOOK E (Reddit Sentiment Long-Only)
================================================================
Builds daily returns for A/B/C/D + the new E, restricts to E's data window
(2024-04..2026-06), and compares Fixed-EW and Momentum-Allocation portfolio
Sharpe WITH vs WITHOUT E — to quantify how the sentiment book changes Sharpe.

E (production combo): ma=15, z=40, mild=0.5, extreme=0.6, hold=8d, top5, minMent=5
  Long-only: buy capitulation (score<-extreme) + mild positive (mild<score<=extreme)
"""
import warnings, sys, os
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, duckdb

RF = 0.02; TD = 252; TC = 0.0025

def metrics(s):
    s = s.dropna()
    if len(s) < 20: return dict(ann=0, sharpe=0, sortino=0, maxdd=0)
    yrs = len(s)/TD
    w = (1+s).cumprod(); tot = w.iloc[-1]-1
    ann = (1+tot)**(1/yrs)-1 if tot>-1 else -1
    sh = (s-RF/TD).mean()/s.std()*np.sqrt(TD) if s.std()>0 else 0
    dn = s[s<0].std(ddof=0); so = (s-RF/TD).mean()/dn*np.sqrt(TD) if dn>0 else 0
    dd = (w/w.cummax()-1).min()
    return dict(ann=ann, sharpe=sh, sortino=so, maxdd=dd)

# ══════════════════════════════════════════════════════════════════
# BOOK E — Reddit Sentiment Long-Only (daily return series)
# ══════════════════════════════════════════════════════════════════
print("Building Book E: Reddit Sentiment Long-Only...")
con = duckdb.connect("data/market_data.duckdb", read_only=True)
sd = con.execute("select date,symbol,weighted_compound,mention_count from sentiment_daily").df()
sd["date"] = pd.to_datetime(sd["date"])
sent = sd.pivot_table(index="date", columns="symbol", values="weighted_compound", aggfunc="mean")
ment = sd.pivot_table(index="date", columns="symbol", values="mention_count", aggfunc="sum").fillna(0)
prices_all = pd.read_parquet("data/cache/daily_close_extended_1997_2026.parquet")
prices_all.index = pd.to_datetime(prices_all.index)
calE = pd.bdate_range(sent.index.min(), sent.index.max())
symsE = [s for s in sent.columns if s in prices_all.columns]
sentE = sent.reindex(calE)[symsE]; mentE = ment.reindex(calE)[symsE].fillna(0)
pxE = prices_all.reindex(calE)[symsE].ffill(); retE = pxE.pct_change().fillna(0)
cumE = (mentE > 0).cumsum(); nE = len(calE)

def sidx(s): return 100.0*(1+s.fillna(0).clip(-1,1)*0.05).cumprod()
def bub(p, ma, z):
    lp=np.log(p.replace(0,np.nan).ffill()); fair=p.rolling(ma).mean()
    res=lp-np.log(fair); zz=(res-res.rolling(z).mean())/res.rolling(z).std(); return np.tanh(zz/2)

MA,ZW,MILD,EXT,HOLD,TOPN,MINM = 15,40,0.5,0.6,8,5,5
Bp = pd.DataFrame(index=calE, columns=symsE, dtype=float)
for s in symsE: Bp[s] = bub(sidx(sentE[s]), MA, ZW)
sig = Bp.shift(1); warm = MA+ZW+1

# daily return series (mark the basket each day during the hold)
dailyE = pd.Series(0.0, index=calE)
i = warm
while i < nE-1:
    elig = cumE.iloc[i-1]; elig = elig[elig>=MINM].index
    sc = sig.iloc[i][elig].dropna()
    cap = sc[sc < -EXT].nsmallest(TOPN); momo = sc[(sc>MILD)&(sc<=EXT)].nlargest(TOPN)
    longs = pd.concat([cap,momo]).drop_duplicates()
    end = min(i+HOLD, nE)
    if len(longs)==0:
        i = end; continue
    for j in range(i, end):
        r = retE.iloc[j][longs.index].mean()
        if j==i: r -= TC          # entry cost
        dailyE.iloc[j] = r
    i = end
dailyE.name = "E"
print(f"  E window: {calE[warm].date()} -> {calE[-1].date()}  (standalone metrics below)")
mE = metrics(dailyE[dailyE.index>=calE[warm]])
print(f"  E standalone: Ann {mE['ann']:.1%}  Sharpe {mE['sharpe']:.3f}  Sortino {mE['sortino']:.2f}  MaxDD {mE['maxdd']:.1%}")

# ══════════════════════════════════════════════════════════════════
# BOOKS A/B/C/D — reuse validated constructions
# ══════════════════════════════════════════════════════════════════
print("\nBuilding Books A/B/C/D...")
close_data = prices_all
# --- A: Daily Momentum + Leverage + UVXY (long-only per latest change) ---
lookback,holding,top=140,40,5
ret_daily=close_data.pct_change().ffill().fillna(0); ret_mom=close_data.pct_change(lookback).ffill().fillna(0)
rawA=[]
for i in range(lookback+1,len(ret_mom),holding):
    ranked=np.argsort(ret_mom.iloc[i-1:i].rank(axis=1,ascending=False).values[0])
    for j in range(i,min(i+holding,len(ret_mom))):
        d=ret_daily.index[j]
        ls=np.sign(ret_mom.iloc[:,ranked[:top]].iloc[i-1:i]).abs()
        lr=ls.mul(np.array(ret_daily.iloc[:,ranked[:top]].iloc[j:j+1])[0]); lret=lr.values.mean()*top
        mom_r=lret/top-0.010/holding
        h=0.0
        if "UVXY" in close_data.columns and pd.notna(close_data.loc[d,"UVXY"]): h=ret_daily.loc[d,"UVXY"]
        elif "^VIX" in close_data.columns and pd.notna(close_data.loc[d,"^VIX"]):
            v=ret_daily.loc[d,"^VIX"]; h=1.5*v-0.0015-0.25*v**2
        rawA.append({"Date":d,"Momentum":mom_r,"Hedge":h})
dfA=pd.DataFrame(rawA).set_index("Date").dropna()
def cbub(p,ma=120,z=240):
    lp=np.log(np.maximum(p,1e-6)); f=lp.rolling(ma).mean(); r=lp-f; return np.tanh(((r-r.rolling(z).mean())/r.rolling(z).std())/2)
bw=(1+dfA).cumprod()/(1+dfA).cumprod().iloc[0]; bub_A=cbub(bw["Momentum"])
hs=(bub_A>0.85).shift(1).fillna(False); ls_=(bub_A<-0.88).shift(1).fillna(False)
outA=[]; hr=lr2=0
for d in dfA.index:
    if hr==0 and hs.loc[d]: hr=40
    if lr2==0 and ls_.loc[d]: lr2=50
    base=dfA.loc[d,"Momentum"]
    if hr>0: r=0.5*base+0.5*dfA.loc[d,"Hedge"]; hr-=1
    elif lr2>0: r=base+0.25*base-0.25*(0.1/252); lr2-=1
    else: r=base
    outA.append(r)
A=pd.Series(outA,index=dfA.index); A.index=pd.to_datetime(A.index)

# --- B/C/D via the live settle replays (validated) ---
from live import signals as LS, settle as LSET
panels = LS.load_panels()
B,_,_=LSET.replay_B(panels); B.index=pd.to_datetime(B.index)
rC,_,_=LSET.replay_C(panels); rC.index=pd.to_datetime(rC.index)
rD,_,_=LSET.replay_D(panels); rD.index=pd.to_datetime(rD.index)
print(f"  A:{A.index[0].date()}-{A.index[-1].date()}  B:{B.index[0].date()}-  C:{rC.index[0].date()}-  D:{rD.index[0].date()}-")

# ══════════════════════════════════════════════════════════════════
# RESTRICT TO E'S WINDOW + COMPARE
# ══════════════════════════════════════════════════════════════════
start = pd.Timestamp("2024-04-01")
idx = pd.bdate_range(start, min(A.index.max(), calE[-1]))
books = pd.DataFrame(index=idx)
books["A"]=A.reindex(idx).fillna(0); books["B"]=B.reindex(idx).fillna(0)
books["C"]=rC.reindex(idx).fillna(0); books["D"]=rD.reindex(idx).fillna(0)
books["E"]=dailyE.reindex(idx).fillna(0)

def fixed_ew(bk):
    av=bk.notna()&(bk!=0);
    # equal weight across all listed books (always available in this window)
    return bk.mean(axis=1)
def mom_alloc(bk,win=60,mind=10):
    sh={c:(bk[c].rolling(win,min_periods=mind).mean()/bk[c].rolling(win,min_periods=mind).std()*np.sqrt(TD)).clip(lower=0) for c in bk.columns}
    sh=pd.DataFrame(sh); tot=sh.sum(axis=1)
    w=sh.div(tot.replace(0,np.nan),axis=0).fillna(1.0/len(bk.columns))
    return (bk*w).sum(axis=1)

print(f"\n{'='*78}")
print(f"PORTFOLIO SHARPE IMPACT  (window {idx[0].date()} -> {idx[-1].date()}, {len(idx)} days)")
print(f"{'='*78}")

print(f"\n  Per-book over this window:")
print(f"  {'Book':<6}{'Ann':>9}{'Sharpe':>9}{'Sortino':>9}{'MaxDD':>9}")
for c in ["A","B","C","D","E"]:
    m=metrics(books[c]); print(f"  {c:<6}{m['ann']:>8.1%}{m['sharpe']:>9.3f}{m['sortino']:>9.2f}{m['maxdd']:>8.1%}")

for label, cols in [("WITHOUT E  (A+B+C+D)",["A","B","C","D"]), ("WITH E     (A+B+C+D+E)",["A","B","C","D","E"])]:
    bk=books[cols]
    few=fixed_ew(bk); mom=mom_alloc(bk)
    mf=metrics(few); mm=metrics(mom)
    print(f"\n  {label}")
    print(f"    Fixed EW   : Sharpe {mf['sharpe']:.3f}  Ann {mf['ann']:.1%}  MaxDD {mf['maxdd']:.1%}")
    print(f"    Mom Alloc  : Sharpe {mm['sharpe']:.3f}  Ann {mm['ann']:.1%}  MaxDD {mm['maxdd']:.1%}")

# delta
f4=metrics(fixed_ew(books[["A","B","C","D"]])); f5=metrics(fixed_ew(books[["A","B","C","D","E"]]))
m4=metrics(mom_alloc(books[["A","B","C","D"]])); m5=metrics(mom_alloc(books[["A","B","C","D","E"]]))
print(f"\n{'='*78}")
print("SHARPE CHANGE FROM ADDING BOOK E")
print(f"{'='*78}")
print(f"  Fixed EW   Sharpe: {f4['sharpe']:.3f} -> {f5['sharpe']:.3f}   ({f5['sharpe']-f4['sharpe']:+.3f})")
print(f"  Mom Alloc  Sharpe: {m4['sharpe']:.3f} -> {m5['sharpe']:.3f}   ({m5['sharpe']-m4['sharpe']:+.3f})")
print(f"  Fixed EW   MaxDD : {f4['maxdd']:.1%} -> {f5['maxdd']:.1%}")
print(f"  Mom Alloc  MaxDD : {m4['maxdd']:.1%} -> {m5['maxdd']:.1%}")

# correlation of E to others
print(f"\n  Correlation of E to other books (diversification check):")
for c in ["A","B","C","D"]:
    print(f"    corr(E,{c}) = {books['E'].corr(books[c]):+.2f}")
books.to_csv("results/portfolio_5book_daily.csv")
print("\nSaved: results/portfolio_5book_daily.csv")
