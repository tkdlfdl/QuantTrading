"""
Queue #49 + #50 acquisition.

(1) SEC structured insider datasets: quarterly form345 zips 2018q2..2026q2.
    Extract NONDERIV_TRANS (open-market purchases, code P) joined with
    REPORTINGOWNER (officer/director flags) and SUBMISSION (issuer symbol).
    Cache: data/cache/insider_purchases.parquet
    (symbol, trans_date, shares, value, is_officer_dir).
(2) FINRA bi-monthly short interest: settlement files at
    cdn.finra.org/equity/otcmarket/biweekly/shrtYYYYMMDD.csv. Candidate
    dates = 15th & month-end (business-adjusted +/-3d probing) 2019-01..now.
    Cache: data/cache/short_interest.parquet (symbol, date, si_shares,
    avg_daily_vol, days_to_cover).
Both checkpoint/resume-safe; SEC fair-use UA; 0.3s spacing.
"""
from __future__ import annotations
import io, sys, time, warnings, zipfile
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import pandas as pd
import requests
from pathlib import Path

HDR = {"User-Agent": "research sailkim41@gmail.com"}
t0 = time.time()

# ══════════ (1) SEC insider datasets ══════════
INS = Path("data/cache/insider_purchases.parquet")
if INS.exists():
    print(f"insider cache exists: {len(pd.read_parquet(INS))} rows — skipping")
else:
    frames = []
    quarters = [f"{y}q{q}" for y in range(2018, 2027) for q in range(1, 5)]
    quarters = [q for q in quarters if q <= "2026q2" and q >= "2018q2"]
    for qt in quarters:
        url = ("https://www.sec.gov/files/structureddata/data/"
               f"insider-transactions-data-sets/{qt}_form345.zip")
        try:
            r = requests.get(url, headers=HDR, timeout=120)
            if r.status_code != 200:
                print(f"  {qt}: HTTP {r.status_code}", flush=True)
                continue
            z = zipfile.ZipFile(io.BytesIO(r.content))
            names = {n.upper().split(".")[0]: n for n in z.namelist()}
            sub = pd.read_csv(z.open(names["SUBMISSION"]), sep="\t",
                              low_memory=False,
                              usecols=lambda c: c.upper() in
                              {"ACCESSION_NUMBER", "ISSUERTRADINGSYMBOL"})
            sub.columns = [c.upper() for c in sub.columns]
            nd = pd.read_csv(z.open(names["NONDERIV_TRANS"]), sep="\t",
                             low_memory=False,
                             usecols=lambda c: c.upper() in
                             {"ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE",
                              "TRANS_SHARES", "TRANS_PRICEPERSHARE",
                              "TRANS_ACQUIRED_DISP_CD"})
            nd.columns = [c.upper() for c in nd.columns]
            ro = pd.read_csv(z.open(names["REPORTINGOWNER"]), sep="\t",
                             low_memory=False,
                             usecols=lambda c: c.upper() in
                             {"ACCESSION_NUMBER", "RPTOWNER_RELATIONSHIP"})
            ro.columns = [c.upper() for c in ro.columns]
            buys = nd[(nd["TRANS_CODE"] == "P")
                      & (nd["TRANS_ACQUIRED_DISP_CD"] == "A")].copy()
            buys = buys.merge(sub, on="ACCESSION_NUMBER", how="left")
            rel = ro.groupby("ACCESSION_NUMBER")["RPTOWNER_RELATIONSHIP"] \
                    .apply(lambda s: ",".join(map(str, s))).reset_index()
            buys = buys.merge(rel, on="ACCESSION_NUMBER", how="left")
            buys["is_officer_dir"] = buys["RPTOWNER_RELATIONSHIP"].fillna("") \
                .str.contains("Officer|Director", case=False)
            out = pd.DataFrame({
                "symbol": buys["ISSUERTRADINGSYMBOL"].astype(str).str.strip().str.upper(),
                "trans_date": pd.to_datetime(buys["TRANS_DATE"], errors="coerce"),
                "shares": pd.to_numeric(buys["TRANS_SHARES"], errors="coerce"),
                "price": pd.to_numeric(buys["TRANS_PRICEPERSHARE"], errors="coerce"),
                "is_officer_dir": buys["is_officer_dir"],
            }).dropna(subset=["trans_date"])
            frames.append(out)
            print(f"  {qt}: {len(out)} open-market buys "
                  f"({time.time()-t0:.0f}s)", flush=True)
            time.sleep(0.3)
        except Exception as e:
            print(f"  {qt}: failed ({type(e).__name__}: {str(e)[:60]})", flush=True)
    if frames:
        allb = pd.concat(frames, ignore_index=True)
        allb.to_parquet(INS)
        print(f"INSIDER DONE: {len(allb)} buys, "
              f"{allb['symbol'].nunique()} symbols", flush=True)
    else:
        print("INSIDER: nothing fetched")

# ══════════ (2) FINRA short interest ══════════
SI = Path("data/cache/short_interest.parquet")
if SI.exists():
    print(f"SI cache exists: {len(pd.read_parquet(SI))} rows — skipping")
else:
    frames = []
    got_dates = []
    months = pd.period_range("2019-01", "2026-08", freq="M")
    cands = []
    for m in months:
        for base in (pd.Timestamp(m.year, m.month, 15),
                      m.to_timestamp("M")):
            for off in (0, -1, -2, 1, -3):
                cands.append(base + pd.Timedelta(days=off))
    seen_settle = set()
    for c in cands:
        key = (c.year, c.month, c.day <= 20)
        if key in seen_settle:
            continue
        url = ("https://cdn.finra.org/equity/otcmarket/biweekly/"
               f"shrt{c.strftime('%Y%m%d')}.csv")
        try:
            r = requests.get(url, headers=HDR, timeout=60)
            if r.status_code != 200 or len(r.content) < 10000:
                continue
            df = pd.read_csv(io.BytesIO(r.content), sep="|")
            df.columns = [str(x).strip() for x in df.columns]
            sym_c = [x for x in df.columns if "symbol" in x.lower()][0]
            si_c = [x for x in df.columns if "currentshortpositionquantity" in
                    x.lower().replace("_", "") or x.lower() == "shortinterest"]
            adv_c = [x for x in df.columns if "averagedailyvolume" in
                     x.lower().replace("_", "")]
            dtc_c = [x for x in df.columns if "daystocover" in
                     x.lower().replace("_", "")]
            out = pd.DataFrame({
                "symbol": df[sym_c].astype(str).str.strip().str.upper(),
                "date": c,
                "si_shares": pd.to_numeric(df[si_c[0]], errors="coerce")
                             if si_c else pd.NA,
                "adv": pd.to_numeric(df[adv_c[0]], errors="coerce")
                       if adv_c else pd.NA,
                "dtc": pd.to_numeric(df[dtc_c[0]], errors="coerce")
                       if dtc_c else pd.NA,
            })
            frames.append(out)
            got_dates.append(c)
            seen_settle.add(key)
            print(f"  SI {c.date()}: {len(out)} rows ({time.time()-t0:.0f}s)",
                  flush=True)
            time.sleep(0.3)
        except Exception:
            continue
    if frames:
        alls = pd.concat(frames, ignore_index=True)
        alls.to_parquet(SI)
        print(f"SI DONE: {len(alls)} rows, {len(got_dates)} settlement dates "
              f"({min(got_dates).date()}..{max(got_dates).date()})", flush=True)
    else:
        print("SI: nothing fetched")
print(f"total {time.time()-t0:.0f}s")
