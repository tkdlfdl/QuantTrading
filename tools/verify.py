"""Backtest result verifier — sanity checks before results are trusted/recorded.

Checks a recorded strategy (strategies/performance/<name>_daily.csv +
<name>_history.json) or a raw return series for impossible/suspicious results:

  C1  sign-consistency : total_return < 0 while Sharpe > 0 (or the reverse).
      Genuinely impossible ONLY if arithmetic and geometric means agree in sign;
      under high volatility arithmetic mean can be positive while compounded
      return is negative (volatility drag). The check flags, then auto-diagnoses:
      if vol drag fully explains the sign split -> FLAG-EXPLAINED, else FAIL.
  C2  maxdd-zero        : MaxDD == 0 with >20 observations — no real strategy
      has zero drawdown. FAIL unless the series is all-zero (inactive book).
  C3  metrics-match     : recompute Sharpe/MaxDD/total from the daily CSV and
      compare to history.json (catches recording drift / wrong convention).
  C4  data-integrity    : NaN/inf, duplicate dates, |daily ret| > 50%,
      zero-fraction > 95% (block-attribution smell), non-monotonic dates.

Verdicts: PASS | FLAG-EXPLAINED (benign, explanation logged) | FAIL.
Every run appends one row per check to strategies/performance/verification_log.csv.

CLI:
  python -m tools.verify book_d_retest            # verify one recorded strategy
  python -m tools.verify --all                    # verify everything recorded
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from tools.metrics import metrics_from_returns

PERF_DIR = os.path.join("strategies", "performance")
LOG_PATH = os.path.join(PERF_DIR, "verification_log.csv")
LOG_COLS = ["timestamp", "name", "check", "verdict", "detail"]


def _log_rows(rows: list[dict]) -> None:
    exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_COLS)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def verify_series(name: str, rets: np.ndarray, history: dict | None = None,
                  periods_per_year: int = 252) -> dict:
    """Run all checks. Returns {"verdict": ..., "checks": [...]} and logs them."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    checks = []
    m = metrics_from_returns(rets, periods_per_year)

    def add(check, verdict, detail):
        checks.append({"timestamp": ts, "name": name, "check": check,
                       "verdict": verdict, "detail": detail})

    # ── C1: sign consistency (Sharpe vs total return) ──────────────────────
    sh, tot = m["sharpe"], m["total_return"]
    if np.isfinite(sh) and np.isfinite(tot) and sh * tot < 0:
        arith = float(np.mean(rets))
        geom_neg_arith_pos = (tot < 0 < sh and arith > 0)
        vol_drag = float(0.5 * np.var(rets))
        if geom_neg_arith_pos and arith - vol_drag < 0:
            add("C1-sign", "FLAG-EXPLAINED",
                f"total {tot:.2%} < 0 but Sharpe {sh:.2f} > 0: volatility drag "
                f"(arith mean {arith:.5f}/day > 0, drag {vol_drag:.5f}/day). "
                f"Math consistent, but treat performance as NEGATIVE.")
        else:
            add("C1-sign", "FAIL",
                f"Sharpe {sh:.2f} and total return {tot:.2%} have opposite signs "
                f"and volatility drag does NOT explain it — recheck the backtest.")
    else:
        add("C1-sign", "PASS", f"Sharpe {sh:.2f}, total {tot:.2%} — signs consistent")

    # ── C2: MaxDD cannot be zero ────────────────────────────────────────────
    dd = m["max_dd"]
    n = len(rets)
    if n > 20 and dd > -1e-9:
        if np.allclose(rets, 0.0):
            add("C2-maxdd", "FLAG-EXPLAINED",
                f"MaxDD == 0 but series is all-zero ({n} days) — inactive book, not a bug")
        else:
            add("C2-maxdd", "FAIL",
                f"MaxDD == 0 over {n} non-zero days — impossible for a real strategy; "
                f"recheck P&L attribution")
    else:
        add("C2-maxdd", "PASS", f"MaxDD {dd:.2%} over {n} days")

    # ── C3: metrics match recorded history ──────────────────────────────────
    if history is not None:
        hm = history.get("metrics", {})
        mism = []
        for k, tol in [("sharpe", 0.02), ("max_dd", 0.005), ("total_return", 0.01)]:
            a, b = m.get(k), hm.get(k)
            if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                if abs(a - b) > tol * max(1.0, abs(b)):
                    mism.append(f"{k}: recomputed {a:.4f} vs recorded {b:.4f}")
        if mism:
            add("C3-metrics", "FAIL", "; ".join(mism))
        else:
            add("C3-metrics", "PASS", "recomputed metrics match history.json")
    else:
        add("C3-metrics", "PASS", "no history.json provided (raw-series mode)")

    # ── C4: data integrity ──────────────────────────────────────────────────
    issues = []
    if not np.all(np.isfinite(rets)):
        issues.append(f"{int(np.sum(~np.isfinite(rets)))} NaN/inf returns")
    if np.max(np.abs(rets), initial=0.0) > 0.5:
        issues.append(f"extreme daily return {np.max(np.abs(rets)):.1%} (>50%)")
    zero_frac = float(np.mean(rets == 0.0)) if n else 0.0
    if zero_frac > 0.95:
        issues.append(f"{zero_frac:.0%} of days are exactly 0 — block-attribution smell")
    if issues:
        add("C4-integrity", "FAIL", "; ".join(issues))
    else:
        add("C4-integrity", "PASS",
            f"n={n}, zero-days {zero_frac:.0%}, max |ret| {np.max(np.abs(rets), initial=0.0):.1%}")

    verdicts = [c["verdict"] for c in checks]
    overall = ("FAIL" if "FAIL" in verdicts
               else "FLAG-EXPLAINED" if "FLAG-EXPLAINED" in verdicts
               else "PASS")
    add("OVERALL", overall, f"{verdicts.count('PASS')}/{len(verdicts)-1} checks clean")
    _log_rows(checks)
    return {"verdict": overall, "checks": checks, "metrics": m}


def verify_recorded(name: str) -> dict:
    """Verify a strategy recorded under strategies/performance/."""
    daily = os.path.join(PERF_DIR, f"{name}_daily.csv")
    hist_p = os.path.join(PERF_DIR, f"{name}_history.json")
    df = pd.read_csv(daily, parse_dates=["date"])
    if df["date"].duplicated().any():
        _log_rows([{"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "name": name, "check": "C4-integrity", "verdict": "FAIL",
                    "detail": f"{int(df['date'].duplicated().sum())} duplicate dates in daily CSV"}])
    history = json.load(open(hist_p)) if os.path.exists(hist_p) else None
    ppy = (history or {}).get("periods_per_year", 252)
    return verify_series(name, df["ret"].astype(float).values, history, ppy)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify recorded backtest results.")
    ap.add_argument("name", nargs="?", help="recorded strategy name (without _daily.csv)")
    ap.add_argument("--all", action="store_true", help="verify every recorded strategy")
    args = ap.parse_args(argv)

    names = ([os.path.basename(f)[:-10] for f in sorted(glob.glob(os.path.join(PERF_DIR, "*_daily.csv")))]
             if args.all else [args.name])
    if not names or names == [None]:
        ap.error("give a name or --all")

    worst = "PASS"
    for nm in names:
        r = verify_recorded(nm)
        v = r["verdict"]
        mark = {"PASS": "ok ", "FLAG-EXPLAINED": "flg", "FAIL": "XXX"}[v]
        print(f"[{mark}] {nm:<45} {v}")
        for c in r["checks"]:
            if c["verdict"] != "PASS" and c["check"] != "OVERALL":
                print(f"      {c['check']}: {c['verdict']} — {c['detail']}")
        if v == "FAIL" or (v == "FLAG-EXPLAINED" and worst == "PASS"):
            worst = v
    print(f"\nlog -> {LOG_PATH}")
    return 1 if worst == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
