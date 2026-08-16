"""Backtest metrics tool — Sharpe ratio and Max Drawdown.

Single source of truth for the two core risk/return metrics, implemented
exactly per the project methodology (CLAUDE.md / .claude/agents/backtest.md):

    Sharpe (annualized) = (mean(rets) * P) / (std(rets) * sqrt(P))
    Max Drawdown        = min((wealth - cummax(wealth)) / cummax(wealth))

where P = periods per year (252 for daily bars, 252*6.5 for hourly bars, etc.).

Use as a library:

    from tools.metrics import sharpe_ratio, max_drawdown, metrics_from_returns
    s  = sharpe_ratio(daily_rets, periods_per_year=252, rf=0.0)
    dd = max_drawdown(daily_rets)          # accepts returns...
    dd = max_drawdown_from_wealth(wealth)  # ...or a wealth/price curve

Use as a CLI tool (reads a column of daily returns or a price/wealth series):

    python -m tools.metrics --file rets.csv --returns-col ret
    python -m tools.metrics --file prices.csv --price-col close --periods 252
    echo "0.01\\n-0.02\\n0.005" | python -m tools.metrics --stdin

Output is JSON: {"sharpe", "max_dd", "n", "periods_per_year", ...}.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _as_array(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    a = a[np.isfinite(a)]
    return a


def sharpe_ratio(returns, periods_per_year: int = 252, rf: float = 0.0) -> float:
    """Annualized Sharpe ratio from a series of per-period returns.

    rf is the annual risk-free rate (0% for this project). It is converted to a
    per-period rate and subtracted from each return before annualizing.
    Returns nan if fewer than 2 points or zero volatility.
    """
    r = _as_array(returns)
    if r.size < 2:
        return float("nan")
    rf_per = rf / periods_per_year
    excess = r - rf_per
    sd = excess.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float((excess.mean() * periods_per_year) / (sd * np.sqrt(periods_per_year)))


def wealth_from_returns(returns) -> np.ndarray:
    """Compound a per-period return series into a wealth curve starting at 1.0."""
    r = _as_array(returns)
    return np.cumprod(1.0 + r)


def max_drawdown_from_wealth(wealth) -> float:
    """Max drawdown of a wealth/price curve. Negative number (0 = no drawdown)."""
    w = _as_array(wealth)
    if w.size == 0:
        return float("nan")
    running_max = np.maximum.accumulate(w)
    drawdown = (w - running_max) / running_max
    return float(drawdown.min())


def max_drawdown(returns) -> float:
    """Max drawdown from a per-period return series (compounds internally)."""
    return max_drawdown_from_wealth(wealth_from_returns(returns))


def metrics_from_returns(returns, periods_per_year: int = 252, rf: float = 0.0) -> dict:
    """Compute Sharpe + Max DD (and a few free extras) from a return series."""
    r = _as_array(returns)
    wealth = wealth_from_returns(r)
    total_return = float(wealth[-1] - 1.0) if r.size else float("nan")
    cagr = (
        float(wealth[-1] ** (periods_per_year / r.size) - 1.0)
        if r.size
        else float("nan")
    )
    return {
        "sharpe": sharpe_ratio(r, periods_per_year, rf),
        "max_dd": max_drawdown(r),
        "total_return": total_return,
        "cagr": cagr,
        "n": int(r.size),
        "periods_per_year": periods_per_year,
    }


def metrics_from_wealth(wealth, periods_per_year: int = 252, rf: float = 0.0) -> dict:
    """Same, but the input is a price/wealth curve rather than returns."""
    w = _as_array(wealth)
    rets = np.diff(w) / w[:-1] if w.size >= 2 else np.array([])
    out = metrics_from_returns(rets, periods_per_year, rf)
    # override with exact drawdown/return computed on the given curve
    out["max_dd"] = max_drawdown_from_wealth(w)
    if w.size:
        out["total_return"] = float(w[-1] / w[0] - 1.0)
    return out


def _load_series(args) -> np.ndarray:
    if args.stdin:
        raw = sys.stdin.read().replace(",", "\n").split()
        return _as_array([float(x) for x in raw])
    # CSV path
    import csv

    rows = []
    with open(args.file, newline="") as fh:
        reader = csv.DictReader(fh) if (args.returns_col or args.price_col) else csv.reader(fh)
        if args.returns_col or args.price_col:
            col = args.returns_col or args.price_col
            for row in reader:
                rows.append(float(row[col]))
        else:
            for row in reader:
                rows.append(float(row[0]))
    return _as_array(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compute Sharpe ratio and Max Drawdown.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="CSV file with a returns or price column")
    src.add_argument("--stdin", action="store_true", help="read numbers from stdin")
    p.add_argument("--returns-col", help="column name holding per-period returns")
    p.add_argument("--price-col", help="column name holding a price/wealth series")
    p.add_argument(
        "--periods", type=int, default=252,
        help="periods per year for annualization (252 daily, 1638 hourly)",
    )
    p.add_argument("--rf", type=float, default=0.0, help="annual risk-free rate (default 0)")
    args = p.parse_args(argv)

    series = _load_series(args)
    if args.price_col:
        out = metrics_from_wealth(series, args.periods, args.rf)
    else:
        out = metrics_from_returns(series, args.periods, args.rf)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
