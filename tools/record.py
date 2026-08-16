"""Performance recorder — persists a strategy's daily and historical performance.

Every backtest records its results under `strategies/performance/` in two files
with a canonical format so the correlation tool and future backtests can reuse
them:

  strategies/performance/<name>_daily.csv     — DAILY performance
      columns: date, ret, wealth   (ret = daily return, wealth = compounded, start 1.0)

  strategies/performance/<name>_history.json  — HISTORICAL performance
      { name, params, data_period, metrics{sharpe,max_dd,cagr,total_return,...},
        yearly{ <year>: {ret, sharpe, max_dd} }, n_days, recorded_at }

Usage:

    from tools.record import record_performance
    record_performance(
        name="book_d_contrarian",
        dates=dates,              # list/Index of daily dates
        returns=daily_rets,       # per-day returns aligned to dates
        params={"ma": 104, "hold": 8, "top_n": 20},
        data_period="2019-01-02..2026-06-18",
        periods_per_year=252,
    )

This does not replace BACKTEST_PERFORMANCE.md (human-readable documentation); it
is the machine-readable record kept alongside the strategies for reuse.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from tools.metrics import metrics_from_returns, sharpe_ratio, max_drawdown

PERF_DIR = os.path.join("strategies", "performance")
IMPROVEMENTS_LOG = os.path.join("research", "improvements_log.md")


def record_improvement(idea: str, source: str, baseline: dict, improved: dict,
                       records: list[str], notes: str = "") -> None:
    """Append an attributed improvement entry: WHICH idea/paper improved
    performance and BY HOW MUCH (Sharpe/CAGR/MaxDD deltas vs baseline).

    baseline/improved: dicts with keys sharpe, cagr (or ann), max_dd.
    records: recorded series names in strategies/performance/.
    """
    from datetime import date
    def g(d, k):
        v = d.get(k)
        return float(v) if v is not None else float("nan")
    b_sh, i_sh = g(baseline, "sharpe"), g(improved, "sharpe")
    b_cg, i_cg = g(baseline, "cagr"), g(improved, "cagr")
    b_dd, i_dd = g(baseline, "max_dd"), g(improved, "max_dd")
    entry = (
        f"\n### {date.today().isoformat()} — {idea}\n"
        f"- **Source:** {source}\n"
        f"- **Sharpe:** {b_sh:.3f} → {i_sh:.3f}  (**{i_sh-b_sh:+.3f}**)\n"
        f"- **CAGR:**   {b_cg:.1%} → {i_cg:.1%}  ({i_cg-b_cg:+.1%})\n"
        f"- **MaxDD:**  {b_dd:.1%} → {i_dd:.1%}  ({i_dd-b_dd:+.1%})\n"
        f"- **Records:** {', '.join(records)}\n"
        + (f"- **Notes:** {notes}\n" if notes else "")
    )
    os.makedirs(os.path.dirname(IMPROVEMENTS_LOG), exist_ok=True)
    if not os.path.exists(IMPROVEMENTS_LOG):
        with open(IMPROVEMENTS_LOG, "w", encoding="utf-8") as fh:
            fh.write(
                "# Improvements Log — what improved performance, and by how much\n\n"
                "One entry per verified improvement: the idea/paper responsible and the\n"
                "exact metric deltas vs its baseline. Appended automatically by\n"
                "`tools.record.record_improvement`; newest entries at the bottom.\n"
            )
    with open(IMPROVEMENTS_LOG, "a", encoding="utf-8") as fh:
        fh.write(entry)


def _ensure_dir() -> None:
    os.makedirs(PERF_DIR, exist_ok=True)


def daily_path(name: str) -> str:
    return os.path.join(PERF_DIR, f"{name}_daily.csv")


def history_path(name: str) -> str:
    return os.path.join(PERF_DIR, f"{name}_history.json")


def _yearly_breakdown(dates, returns, periods_per_year: int) -> dict:
    s = pd.Series(np.asarray(returns, dtype=float), index=pd.to_datetime(list(dates)))
    out = {}
    for year, grp in s.groupby(s.index.year):
        r = grp.values
        wealth = np.cumprod(1.0 + r)
        out[str(int(year))] = {
            "ret": float(wealth[-1] - 1.0),
            "sharpe": sharpe_ratio(r, periods_per_year),
            "max_dd": max_drawdown(r),
            "n_days": int(r.size),
        }
    return out


def record_performance(
    name: str,
    dates,
    returns,
    params: dict | None = None,
    data_period: str | None = None,
    periods_per_year: int = 252,
    extra: dict | None = None,
) -> dict:
    """Write <name>_daily.csv and <name>_history.json. Returns the history dict."""
    _ensure_dir()
    dates = list(dates)
    returns = [float(x) for x in returns]
    if len(dates) != len(returns):
        raise ValueError(f"dates ({len(dates)}) and returns ({len(returns)}) length mismatch")

    # --- daily performance ---
    wealth = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    with open(daily_path(name), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "ret", "wealth"])
        for d, r, wv in zip(dates, returns, wealth):
            d_str = pd.Timestamp(d).date().isoformat()
            w.writerow([d_str, f"{r:.10g}", f"{wv:.10g}"])

    # --- historical performance ---
    history = {
        "name": name,
        "params": params or {},
        "data_period": data_period,
        "periods_per_year": periods_per_year,
        "metrics": metrics_from_returns(returns, periods_per_year),
        "yearly": _yearly_breakdown(dates, returns, periods_per_year),
        "n_days": len(returns),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        history["extra"] = extra
    with open(history_path(name), "w") as fh:
        json.dump(history, fh, indent=2)

    return history


if __name__ == "__main__":
    # tiny self-test
    import pandas as pd

    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    rets = [0.01, -0.02, 0.005, 0.03, -0.01, 0.0, 0.015, -0.005, 0.02, -0.03]
    h = record_performance("selftest", idx, rets, params={"demo": True},
                           data_period="2020-01-01..2020-01-10")
    print(json.dumps(h, indent=2))
