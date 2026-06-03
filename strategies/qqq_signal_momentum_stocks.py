"""
QQQ Signal → Momentum Stock Selection Strategy
================================================
Signal: QQQ hourly bubble score (same formula as before)
  bubble_score < -threshold  →  market is extreme oversold

Execution: buy the TOP N momentum stocks from the universe
  (ranked by recent momentum_lookback_hours price return)
  Enter at next bar open, hold for hold_hours bars, then exit.

Rationale: QQQ extreme oversold = broad market dip → buy the
strongest individual stocks that tend to snap back hardest.

No-lookahead:
  - QQQ score at bar i computed using data through bar i
  - Signal fires at end of bar i → trade executes at bar i+1 open
  - Momentum ranking uses returns up to and including bar i

Grid:
  ma_window              QQQ MA window (hours)
  z_window               QQQ z-score window (hours)
  threshold              Extreme oversold threshold
  hold_hours             Hours to hold stock positions
  momentum_lookback_hours Hours of lookback to rank stocks
  top_n                  Number of stocks to buy
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import product

from strategies.qqq_bubble_hourly import calculate_bubble_score

TRADING_HOURS_PER_YEAR = 252 * 6.5


def run_qqq_signal_momentum_stocks(
    qqq_open:   pd.Series,          # QQQ hourly open prices
    qqq_close:  pd.Series,          # QQQ hourly close prices
    stock_open:  pd.DataFrame,      # universe hourly open  (index=timestamp, cols=tickers)
    stock_close: pd.DataFrame,      # universe hourly close (index=timestamp, cols=tickers)

    # QQQ signal params
    ma_window_grid:  list = [50, 100],
    z_window_grid:   list = [100, 200],
    threshold_grid:  list = [0.7, 0.8, 0.9],

    # Execution params
    hold_hours_grid:             list = [1, 2, 4, 8],
    momentum_lookback_hours_grid: list = [6, 24, 48],
    top_n_grid:                  list = [5, 10, 20],

    transaction_cost: float = 0.001,   # 0.1% round-trip per position
) -> tuple[pd.Series, dict, pd.DataFrame]:
    """
    Returns
    -------
    best_daily_ret : daily return Series
    best_params    : dict of best parameters + metrics
    grid_df        : full grid sorted by Sharpe
    """
    # Align QQQ and stock timestamps
    common_ts = qqq_close.index.intersection(stock_close.index)
    qqq_c   = qqq_close.loc[common_ts]
    qqq_o   = qqq_open.loc[common_ts]
    stk_c   = stock_close.loc[common_ts]
    stk_o   = stock_open.loc[common_ts]
    n       = len(common_ts)

    tickers = stk_c.columns.tolist()
    print(f"QQQ signal bars: {n}  |  Universe: {len(tickers)} tickers")

    total = (len(ma_window_grid) * len(z_window_grid) * len(threshold_grid) *
             len(hold_hours_grid) * len(momentum_lookback_hours_grid) * len(top_n_grid))
    print(f"Grid search: {total} combinations...")

    # Pre-compute QQQ bubble scores for each (ma, z)
    score_cache: dict = {}
    for ma, z in product(ma_window_grid, z_window_grid):
        raw = calculate_bubble_score(qqq_c, ma, z)
        score_cache[(ma, z)] = raw.shift(1)   # shift: no lookahead

    # Pre-compute stock returns matrix (rolling returns for momentum ranking)
    # stk_ret[i] = close[i] / close[i-1] - 1
    stk_pct = stk_c.pct_change().fillna(0)

    grid_results = []
    best_sharpe  = -np.inf
    best_daily   = None
    best_params  = None

    for ma, z, thresh, hold, mom_lb, top_n in product(
        ma_window_grid, z_window_grid, threshold_grid,
        hold_hours_grid, momentum_lookback_hours_grid, top_n_grid,
    ):
        signal_scores = score_cache[(ma, z)]
        trades: list[dict] = []
        in_trade_until = -1

        for i in range(mom_lb + 1, n - hold):
            if i <= in_trade_until:
                continue

            sig = signal_scores.iloc[i]
            if pd.isna(sig) or sig >= -thresh:
                continue

            # Momentum ranking: cumulative return over last mom_lb bars (no lookahead)
            lb_start = max(0, i - mom_lb)
            cum_ret  = (1 + stk_pct.iloc[lb_start:i]).prod() - 1
            cum_ret  = cum_ret.dropna()

            if len(cum_ret) == 0:
                continue

            # Top N by recent momentum
            selected = cum_ret.nlargest(top_n).index.tolist()
            if not selected:
                continue

            # Entry: bar i open, Exit: bar i+hold-1 close
            entry_i  = i
            exit_i   = min(i + hold - 1, n - 1)
            entry_ts = common_ts[entry_i]
            exit_ts  = common_ts[exit_i]

            pos_rets = []
            for tkr in selected:
                ep = stk_o.at[entry_ts, tkr]
                xp = stk_c.at[exit_ts,  tkr]
                if pd.isna(ep) or pd.isna(xp) or ep <= 0:
                    continue
                pos_rets.append((xp / ep - 1) - transaction_cost)

            if not pos_rets:
                continue

            port_ret = float(np.mean(pos_rets))
            trades.append({
                "entry_dt":   entry_ts,
                "exit_dt":    exit_ts,
                "qqq_score":  sig,
                "n_stocks":   len(pos_rets),
                "net_ret":    port_ret,
            })
            in_trade_until = exit_i

        if len(trades) < 3:
            grid_results.append(dict(
                ma_window=ma, z_window=z, threshold=thresh,
                hold_hours=hold, mom_lookback=mom_lb, top_n=top_n,
                Sharpe=np.nan, Sortino=np.nan, Total_Return=np.nan,
                Max_DD=np.nan, n_trades=len(trades), Win_Rate=np.nan,
            ))
            continue

        tdf   = pd.DataFrame(trades)
        tdf["date"] = pd.to_datetime(tdf["entry_dt"]).dt.normalize()
        daily = tdf.groupby("date")["net_ret"].sum()

        data_end  = hc.index[-1].normalize()
        all_dates  = pd.date_range(daily.index.min(), data_end, freq="B")
        daily_full = daily.reindex(all_dates, fill_value=0.0)

        wealth = (1 + daily_full).cumprod(); wealth = wealth / wealth.iloc[0]
        mdd    = float((wealth / wealth.cummax() - 1).min())
        sh     = float(np.sqrt(252) * daily_full.mean() / daily_full.std()) if daily_full.std() > 0 else np.nan
        ds     = daily_full[daily_full < 0].std()
        so     = float(np.sqrt(252) * daily_full.mean() / ds) if ds > 0 else np.nan
        wr     = float((tdf["net_ret"] > 0).mean())
        tot    = float(wealth.iloc[-1] - 1)

        row = dict(
            ma_window=ma, z_window=z, threshold=thresh,
            hold_hours=hold, mom_lookback=mom_lb, top_n=top_n,
            Sharpe=sh, Sortino=so, Total_Return=tot, Max_DD=mdd,
            n_trades=len(trades), Win_Rate=wr,
            Avg_Net_Ret=float(tdf["net_ret"].mean()),
        )
        grid_results.append(row)

        if pd.notna(sh) and sh > best_sharpe:
            best_sharpe = sh
            best_daily  = daily_full.rename("QQQ_MomStocks")
            best_params = row

    grid_df = pd.DataFrame(grid_results).sort_values("Sharpe", ascending=False)

    if best_params:
        print(f"\nBest Parameters:")
        for k, v in best_params.items():
            print(f"  {k:<22} {v}")

    return best_daily, best_params, grid_df
