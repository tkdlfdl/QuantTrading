"""
FULL STRATEGY REVIEW — Annual Return, Sharpe, MaxDD (all strategies)
"""
import pandas as pd, numpy as np, warnings, os
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# LOAD YEARLY DATA
# ─────────────────────────────────────────────────────────────
yr = pd.read_csv("results/portfolio_full_yearly_stats.csv")   # A,C,B,Fixed,Mom,QQQ,SPY
b  = pd.read_csv("results/strategy2_yearly_corrected.csv")    # B QQQ Bubble
c  = pd.read_csv("results/intraday_mr_yearly.csv")            # C Intraday MR
f  = pd.read_csv("results/short_squeeze_correct_yearly.csv")  # F Short Squeeze

# Map column names
b = b.rename(columns={"MaxDD_hourly":"MaxDD_B","Sharpe":"Sharpe_B",
                       "Total_Return":"Ret_B","Year":"Year"})
c = c.rename(columns={"MaxDD":"MaxDD_C","Sharpe":"Sharpe_C",
                       "Return":"Ret_C","Year":"Year"})
f = f.rename(columns={"MaxDD":"MaxDD_F","Sharpe":"Sharpe_F",
                       "Return":"Ret_F","Year":"Year"})

# Merge everything on Year
base = yr[["Year","Ret_A","Sharpe_A","MaxDD_A",
            "Ret_Fixed_EW","Sharpe_Fixed_EW","MaxDD_Fixed_EW",
            "Ret_Mom_Alloc","Sharpe_Mom_Alloc","MaxDD_Mom_Alloc",
            "Ret_QQQ","Sharpe_QQQ","MaxDD_QQQ",
            "Ret_SPY","Sharpe_SPY","MaxDD_SPY"]].copy()

all_df = base.merge(b[["Year","Ret_B","Sharpe_B","MaxDD_B"]], on="Year", how="left") \
             .merge(c[["Year","Ret_C","Sharpe_C","MaxDD_C"]], on="Year", how="left") \
             .merge(f[["Year","Ret_F","Sharpe_F","MaxDD_F"]], on="Year", how="left")
all_df = all_df.sort_values("Year").reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
# PRINT FULL YEARLY TABLE
# ─────────────────────────────────────────────────────────────
STRATS = {
    "A: Daily Momentum":     ("Ret_A",      "Sharpe_A",      "MaxDD_A"),
    "B: QQQ Bubble":         ("Ret_B",      "Sharpe_B",      "MaxDD_B"),
    "C: Intraday MR":        ("Ret_C",      "Sharpe_C",      "MaxDD_C"),
    "F: Short Squeeze":      ("Ret_F",      "Sharpe_F",      "MaxDD_F"),
    "Fixed EW (A+B+C)":      ("Ret_Fixed_EW","Sharpe_Fixed_EW","MaxDD_Fixed_EW"),
    "Mom Alloc (A+B+C)":     ("Ret_Mom_Alloc","Sharpe_Mom_Alloc","MaxDD_Mom_Alloc"),
    "QQQ B&H":               ("Ret_QQQ",    "Sharpe_QQQ",    "MaxDD_QQQ"),
    "SPY B&H":               ("Ret_SPY",    "Sharpe_SPY",    "MaxDD_SPY"),
}

def pct(v):  return f"{v*100:>7.1f}%" if pd.notna(v) else "    n/a"
def sh(v):   return f"{v:>7.3f}"       if pd.notna(v) else "    n/a"

# Print per-strategy yearly breakdown
print("=" * 110)
print("FULL STRATEGY REVIEW — Annual Return | Sharpe | Max Drawdown (1997-2026)")
print("=" * 110)

for name, (rc, sc, dc) in STRATS.items():
    sub = all_df[["Year", rc, sc, dc]].dropna(subset=[rc])
    if sub.empty:
        continue
    print(f"\n{'─'*90}")
    print(f"  {name}")
    print(f"{'─'*90}")
    print(f"  {'Year':<6}  {'Return':>9}  {'Sharpe':>8}  {'MaxDD':>9}")
    for _, row in sub.iterrows():
        yr_val = int(row["Year"])
        ret    = row[rc];   s_val = row[sc];   dd = row[dc]
        flag = ""
        if pd.notna(ret) and ret < 0: flag = " <<"
        print(f"  {yr_val:<6}  {pct(ret)}  {sh(s_val)}  {pct(dd)}{flag}")

# ─────────────────────────────────────────────────────────────
# OVERALL METRICS TABLE
# ─────────────────────────────────────────────────────────────
print("\n\n" + "=" * 110)
print("OVERALL PERFORMANCE SUMMARY — ALL STRATEGIES")
print("=" * 110)

# Load overall from CSV
ov = pd.read_csv("results/portfolio_overall_stats.csv")
print(f"\n{'Strategy':<30}  {'Ann Ret':>9}  {'Sharpe':>8}  {'Sortino':>9}  {'MaxDD':>9}  {'Total Ret':>12}  {'Pos Yrs':>9}  {'Win Rate':>9}")
print("  " + "-"*108)
for _, r in ov.iterrows():
    print(f"  {r.Strategy:<28}  {r.Ann_Ret:>9.2%}  {r.Sharpe:>8.4f}  "
          f"{r.Sortino:>9.4f}  {r.MaxDD:>9.2%}  {r.Total_Ret:>12.2%}  "
          f"{r.Pos_Years:>9}  {r.Win_Rate:>9.1%}")

# Add short squeeze
# from backtest: Ann=44.46%, Sharpe=2.51, Sortino=4.38, MaxDD=-9.08%, 2020-2026
f_ov = all_df[["Ret_F","MaxDD_F"]].dropna()
f_total = (1 + f_ov["Ret_F"]).prod() - 1
f_years = 6  # 2020-2026
f_ann   = (1 + f_total) ** (1/f_years) - 1
print(f"  {'F: Short Squeeze (2020-2026)':<28}  {f_ann:>9.2%}  {'2.5108':>8}  "
      f"{'4.3765':>9}  {'-9.08%':>9}  {f_total:>12.2%}  {'7/7':>9}  {'55.1%':>9}")

# ─────────────────────────────────────────────────────────────
# SUMMARY COMPARISON TABLE (2020-2026 overlap)
# ─────────────────────────────────────────────────────────────
print("\n\n" + "=" * 110)
print("OVERLAP PERIOD (2020-2026) — ALL STRATEGIES SIDE BY SIDE")
print("=" * 110)
print(f"\n  {'Year':<6}  {'A:DailyMom':>11}  {'B:QQQBub':>10}  {'C:IntraDA':>10}  "
      f"{'F:Squeeze':>10}  {'FixedEW':>9}  {'MomAlloc':>9}  {'QQQ':>8}")
print(f"  {'-'*6}  {'-'*11}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*8}")

for _, row in all_df[all_df.Year >= 2020].iterrows():
    yr_v = int(row.Year)
    def p(c): return f"{row[c]*100:>9.1f}%" if pd.notna(row.get(c)) else "      n/a"
    print(f"  {yr_v:<6}  {p('Ret_A'):>11}  {p('Ret_B'):>10}  {p('Ret_C'):>10}  "
          f"{p('Ret_F'):>10}  {p('Ret_Fixed_EW'):>9}  {p('Ret_Mom_Alloc'):>9}  {p('Ret_QQQ'):>8}")

print(f"\n  {'Sharpe':>6}:")
print(f"  {'Year':<6}  {'A:DailyMom':>11}  {'B:QQQBub':>10}  {'C:IntraDA':>10}  "
      f"{'F:Squeeze':>10}  {'FixedEW':>9}  {'MomAlloc':>9}  {'QQQ':>8}")
print(f"  {'-'*6}  {'-'*11}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*8}")

for _, row in all_df[all_df.Year >= 2020].iterrows():
    yr_v = int(row.Year)
    def s(c): return f"{row[c]:>9.3f}" if pd.notna(row.get(c)) else "      n/a"
    print(f"  {yr_v:<6}  {s('Sharpe_A'):>11}  {s('Sharpe_B'):>10}  {s('Sharpe_C'):>10}  "
          f"{s('Sharpe_F'):>10}  {s('Sharpe_Fixed_EW'):>9}  {s('Sharpe_Mom_Alloc'):>9}  {s('Sharpe_QQQ'):>8}")

print(f"\n  {'MaxDD':>6}:")
print(f"  {'Year':<6}  {'A:DailyMom':>11}  {'B:QQQBub':>10}  {'C:IntraDA':>10}  "
      f"{'F:Squeeze':>10}  {'FixedEW':>9}  {'MomAlloc':>9}  {'QQQ':>8}")
print(f"  {'-'*6}  {'-'*11}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*8}")

for _, row in all_df[all_df.Year >= 2020].iterrows():
    yr_v = int(row.Year)
    def d(c): return f"{row[c]*100:>9.1f}%" if pd.notna(row.get(c)) else "      n/a"
    print(f"  {yr_v:<6}  {d('MaxDD_A'):>11}  {d('MaxDD_B'):>10}  {d('MaxDD_C'):>10}  "
          f"{d('MaxDD_F'):>10}  {d('MaxDD_Fixed_EW'):>9}  {d('MaxDD_Mom_Alloc'):>9}  {d('MaxDD_QQQ'):>8}")

# ─────────────────────────────────────────────────────────────
# RANKING TABLE
# ─────────────────────────────────────────────────────────────
print("\n\n" + "=" * 90)
print("STRATEGY RANKINGS (2020-2026 period, except A which uses 1997-2026)")
print("=" * 90)

summary = [
    ("A: Daily Momentum+Lev+UVXY", "1997-2026", 63.95,  1.33, 2.06, -65.3),
    ("B: QQQ Bubble Hourly",       "2020-2026", 17.19,  1.66, 8.44, -16.3),
    ("C: Intraday MR+Flip",        "2019-2026", 26.78,  0.98, 0.65, -20.8),
    ("F: Short Squeeze+Bubble",    "2020-2026", 44.46,  2.51, 4.38,  -9.1),
    ("Fixed EW (A+B+C)",           "1997-2026", 52.74,  1.29, 1.87, -65.3),
    ("Momentum Alloc (A+B+C)",     "1997-2026", 72.29,  1.54, 2.29, -65.3),
    ("QQQ Buy & Hold",             "1997-2026", 10.18,  0.50, 0.64, -83.0),
    ("SPY Buy & Hold",             "1997-2026",  9.29,  0.55, 0.71, -55.2),
]

print(f"\n  {'Strategy':<35}  {'Period':<12}  {'Ann Ret':>9}  {'Sharpe':>8}  {'Sortino':>9}  {'MaxDD':>9}")
print("  " + "-"*95)
for name, period, ann, sh_v, so, dd in summary:
    print(f"  {name:<35}  {period:<12}  {ann:>8.2f}%  {sh_v:>8.4f}  {so:>9.4f}  {dd:>8.1f}%")

# Rankings
print("\n  RANKING BY SHARPE:    ", end="")
ranked_sh = sorted(summary, key=lambda x: x[3], reverse=True)
print(" > ".join([f"{n.split(':')[0].split('(')[0].strip()}({s:.2f})" for n,_,_,s,*_ in ranked_sh]))

print("  RANKING BY ANN RET:  ", end="")
ranked_ret = sorted(summary, key=lambda x: x[2], reverse=True)
print(" > ".join([f"{n.split(':')[0].split('(')[0].strip()}({r:.0f}%)" for n,_,r,*_ in ranked_ret]))

print("  RANKING BY MaxDD:    ", end="")
ranked_dd = sorted(summary, key=lambda x: x[5], reverse=True)  # less negative = better
print(" > ".join([f"{n.split(':')[0].split('(')[0].strip()}({d:.0f}%)" for n,_,_,_,_,d in ranked_dd]))

# ─────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(18, 16))
fig.suptitle("All Strategies Review — Annual Return | Sharpe | MaxDD (2020-2026)",
             fontsize=14, fontweight="bold")

strat_cols = {
    "A: Daily Mom": ("Ret_A", "Sharpe_A", "MaxDD_A", "steelblue"),
    "B: QQQ Bubble": ("Ret_B", "Sharpe_B", "MaxDD_B", "green"),
    "C: Intraday MR": ("Ret_C", "Sharpe_C", "MaxDD_C", "orange"),
    "F: Short Squeeze": ("Ret_F", "Sharpe_F", "MaxDD_F", "crimson"),
    "Fixed EW": ("Ret_Fixed_EW","Sharpe_Fixed_EW","MaxDD_Fixed_EW","darkred"),
    "Mom Alloc": ("Ret_Mom_Alloc","Sharpe_Mom_Alloc","MaxDD_Mom_Alloc","purple"),
    "QQQ": ("Ret_QQQ","Sharpe_QQQ","MaxDD_QQQ","gray"),
}

ov_yrs = all_df[all_df.Year >= 2020]["Year"].values
x = np.arange(len(ov_yrs)); w = 0.12

for k, (label, (rc, sc, dc, color)) in enumerate(strat_cols.items()):
    sub = all_df[all_df.Year >= 2020][rc].fillna(0).values
    sh_sub = all_df[all_df.Year >= 2020][sc].fillna(0).values
    dd_sub = all_df[all_df.Year >= 2020][dc].fillna(0).values
    axes[0].bar(x + k*w - 3*w, sub*100, w, label=label, color=color, alpha=0.8)
    axes[1].bar(x + k*w - 3*w, sh_sub,  w, label=label, color=color, alpha=0.8)
    axes[2].bar(x + k*w - 3*w, dd_sub*100, w, label=label, color=color, alpha=0.8)

for ax, title, ylabel in [
    (axes[0], "Annual Return (%)", "Return (%)"),
    (axes[1], "Sharpe Ratio", "Sharpe"),
    (axes[2], "Max Drawdown (%)", "MaxDD (%)")]:
    ax.set_xticks(x); ax.set_xticklabels(ov_yrs.astype(str))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(title, fontweight="bold"); ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, ncol=4); ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/all_strategies_review.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: results/all_strategies_review.png")
print("\nDONE")
