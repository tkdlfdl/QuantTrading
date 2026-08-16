from data.universe import get_universe
from data.intraday_loader import load_hourly_bars
from strategies.universe_bubble_hourly import run_universe_bubble_hourly

print("Loading universe hourly bars...")
universe = get_universe()
ho, hc = load_hourly_bars(universe, use_cache=True)
print(f"Data: {hc.shape[1]} tickers × {hc.shape[0]} bars  ({hc.index[0].date()} → {hc.index[-1].date()})")

# Run with smaller grid for speed
print("\nRunning backtest with reduced grid...")
best_ret, best_params, grid_df = run_universe_bubble_hourly(
    hourly_open  = ho,
    hourly_close = hc,
    ma_window_grid   = [50, 100],
    z_window_grid    = [100, 200],
    buy_threshold_grid   = [0.7, 0.8, 0.9],
    short_threshold_grid = [0.85, 0.95],
    hold_hours_grid  = [2, 4, 8],
    top_n_grid       = [10, 20],
    transaction_cost  = 0.001,
    short_borrow_rate = 0.08,
)

print("\n" + "="*80)
print("BEST PARAMETERS")
print("="*80)
if best_params:
    for k, v in best_params.items():
        print(f"  {k:<18} {v}")

print("\n" + "="*80)
print("TOP 15 COMBOS (by Sharpe)")
print("="*80)
cols = ["ma_window","z_window","buy_threshold","short_threshold","hold_hours","top_n",
        "Sharpe","Sortino","Total_Return","Max_DD","n_trades","Win_Rate"]
print(grid_df[cols].head(15).to_string(index=False))
