import numpy as np
import pandas as pd


def performance_stats(ret_df: pd.DataFrame, wealth_df: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    def sharpe_ratio(r):
        return np.sqrt(trading_days) * r.mean() / r.std() if r.std() != 0 else np.nan

    def sortino_ratio(r):
        downside = r[r < 0]
        downside_std = downside.std()
        return np.sqrt(trading_days) * r.mean() / downside_std if downside_std != 0 else np.nan

    def max_drawdown(w):
        return (w / w.cummax() - 1).min()

    analysis = pd.DataFrame(index=ret_df.columns)
    analysis["Total Return"] = wealth_df.iloc[-1] - 1
    analysis["Final Wealth"] = wealth_df.iloc[-1]
    analysis["Sharpe Ratio"] = ret_df.apply(sharpe_ratio)
    analysis["Sortino Ratio"] = ret_df.apply(sortino_ratio)
    analysis["Max Drawdown"] = wealth_df.apply(max_drawdown)
    return analysis


def yearly_performance_stats(ret_df: pd.DataFrame, wealth_df: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    def sortino_ratio(r):
        downside = r[r < 0]
        downside_std = downside.std()
        return np.sqrt(trading_days) * r.mean() / downside_std if downside_std != 0 else np.nan

    rows = []
    for year, yearly_ret in ret_df.groupby(ret_df.index.year):
        yearly_wealth = wealth_df.loc[yearly_ret.index]
        row = {"Year": year}
        for col in ret_df.columns:
            r = yearly_ret[col]
            w = yearly_wealth[col]
            row[f"{col}_Return"] = w.iloc[-1] / w.iloc[0] - 1
            row[f"{col}_Sharpe"] = (
                np.sqrt(trading_days) * r.mean() / r.std() if r.std() != 0 else np.nan
            )
            row[f"{col}_Sortino"] = sortino_ratio(r)
            row[f"{col}_MaxDrawdown"] = (w / w.cummax() - 1).min()
        rows.append(row)
    return pd.DataFrame(rows).set_index("Year")


def compute_metrics(returns: pd.Series, equity: pd.Series, periods_per_year: int = 252) -> dict:
    r = returns.dropna()
    if r.empty:
        return {}

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n_years = len(r) / periods_per_year

    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    sharpe = (r.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() > 0 else 0.0

    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    wins = r[r > 0]
    losses = r[r < 0]
    win_rate = len(wins) / len(r[r != 0]) if len(r[r != 0]) > 0 else 0.0

    return {
        "Total Return":   f"{total_return:.2%}",
        "CAGR":           f"{cagr:.2%}",
        "Sharpe Ratio":   f"{sharpe:.3f}",
        "Max Drawdown":   f"{max_dd:.2%}",
        "Calmar Ratio":   f"{calmar:.3f}",
        "Win Rate":       f"{win_rate:.2%}",
        "Total Bars":     len(r),
    }
