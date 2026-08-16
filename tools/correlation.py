"""Strategy correlation tool.

Computes the Pearson correlation matrix of daily returns across strategies,
using the recorded daily-performance files in `strategies/performance/`.

Each strategy records a daily file `strategies/performance/<name>_daily.csv`
with at least columns `date,ret` (see tools/record.py / the backtest agent).
This tool loads several such files, aligns them on their common dates (inner
join — correlation requires overlapping observations), and reports:

    - the full correlation matrix
    - the pairwise correlations sorted by magnitude

Use as a library:

    from tools.correlation import correlation_matrix, load_returns_frame
    frame = load_returns_frame(dir="strategies/performance")  # cols = strategy names
    corr  = correlation_matrix(frame)

Use as a CLI tool:

    # all *_daily.csv in the default folder:
    python -m tools.correlation
    # a specific folder:
    python -m tools.correlation --dir strategies/performance
    # explicit files (labels taken from filename stem, minus _daily):
    python -m tools.correlation --files bookD_daily.csv bookF_daily.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

DEFAULT_DIR = os.path.join("strategies", "performance")


def _label_from_path(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem[:-6] if stem.endswith("_daily") else stem


def load_returns_frame(
    files: list[str] | None = None,
    dir: str = DEFAULT_DIR,
    date_col: str = "date",
    ret_col: str = "ret",
) -> pd.DataFrame:
    """Build a wide DataFrame: index = date, one column of daily returns per strategy.

    If `files` is None, loads every `*_daily.csv` in `dir`.
    Columns are NOT aligned here (outer union of dates); align at correlation time.
    """
    if files is None:
        files = sorted(glob.glob(os.path.join(dir, "*_daily.csv")))
    if not files:
        raise FileNotFoundError(f"No daily-performance files found (dir={dir!r}, files={files!r})")

    series = {}
    for path in files:
        df = pd.read_csv(path, parse_dates=[date_col])
        s = df.set_index(date_col)[ret_col].astype(float)
        series[_label_from_path(path)] = s
    return pd.DataFrame(series)


def correlation_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of daily returns on the common (inner) date range."""
    aligned = frame.dropna(how="any")
    return aligned.corr()


def pairwise_sorted(corr: pd.DataFrame) -> list[tuple[str, str, float]]:
    """Unique off-diagonal pairs sorted by descending correlation."""
    cols = list(corr.columns)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], float(corr.iloc[i, j])))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Correlation matrix of strategy daily returns.")
    p.add_argument("--dir", default=DEFAULT_DIR, help="folder of *_daily.csv files")
    p.add_argument("--files", nargs="+", help="explicit daily-return CSV files")
    p.add_argument("--date-col", default="date")
    p.add_argument("--ret-col", default="ret")
    p.add_argument("--json", action="store_true", help="emit JSON instead of tables")
    args = p.parse_args(argv)

    frame = load_returns_frame(args.files, args.dir, args.date_col, args.ret_col)
    corr = correlation_matrix(frame)
    n_common = int(frame.dropna(how="any").shape[0])
    pairs = pairwise_sorted(corr)

    if args.json:
        print(json.dumps({
            "strategies": list(corr.columns),
            "common_days": n_common,
            "matrix": corr.round(4).to_dict(),
            "pairwise_sorted": [[a, b, round(c, 4)] for a, b, c in pairs],
        }, indent=2))
    else:
        print(f"Common overlapping days: {n_common}\n")
        print("Correlation matrix:")
        print(corr.round(3).to_string())
        print("\nPairwise (sorted):")
        for a, b, c in pairs:
            print(f"  {a} vs {b}: {c:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
