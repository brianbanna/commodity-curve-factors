"""Deep audit of claimed MinVar+TSMOM alpha.

Determines whether the OOS Sharpe +0.70 is genuine alpha or an artifact of
data mining, long-only commodity beta, covariance instability, or roll-return
double-counting.  Seven independent investigations, final verdict.

Run:
    python scripts/deep_audit.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from commodity_curve_factors.curves.builder import load_curves  # noqa: E402
from commodity_curve_factors.curves.metrics import compute_carry  # noqa: E402
from commodity_curve_factors.data.futures_loader import load_front_month_data  # noqa: E402
from commodity_curve_factors.evaluation.metrics import sharpe_ratio, split_is_oos  # noqa: E402
from commodity_curve_factors.signals.ranking import rank_and_select  # noqa: E402
from commodity_curve_factors.utils.paths import DATA_PROCESSED  # noqa: E402

BACKTEST_OUT = DATA_PROCESSED / "backtest"

# ---------------------------------------------------------------------------
# Load shared data
# ---------------------------------------------------------------------------
print("=" * 72)
print("DEEP AUDIT — Is MinVar+TSMOM Real Alpha or Self-Deception?")
print("=" * 72)

print("\nLoading data...")
futures = load_front_month_data()
prices = pd.DataFrame({sym: df["Close"] for sym, df in futures.items()})
prices.index = pd.DatetimeIndex(prices.index)
if prices.index.tz is not None:
    prices.index = prices.index.tz_localize(None)
prices = prices.sort_index()
returns = np.log(prices / prices.shift(1))

n_commodities = returns.shape[1]
print(f"  {n_commodities} commodities, {len(returns)} days")
print(f"  Date range: {returns.index[0].date()} to {returns.index[-1].date()}")

# =========================================================================
# INVESTIGATION 1: Is MinVar+TSMOM alpha or just being long commodities?
# =========================================================================
print("\n" + "=" * 72)
print("INVESTIGATION 1: MinVar+TSMOM vs. naive long-only benchmarks")
print("=" * 72)

# 1a. Equal-weight long-only (1/N)
ew_returns = returns.mean(axis=1).dropna()
is_ew, oos_ew = split_is_oos(ew_returns)
print(f"\n  Equal-weight long:  IS={sharpe_ratio(is_ew):+.3f}  OOS={sharpe_ratio(oos_ew):+.3f}")

# 1b. Inverse-vol long-only (simple risk parity)
vol = returns.rolling(60).std()
inv_vol_w = (1 / vol).div((1 / vol).sum(axis=1), axis=0)
inv_vol_returns = (inv_vol_w.shift(1) * returns).sum(axis=1).dropna()
is_iv, oos_iv = split_is_oos(inv_vol_returns)
print(f"  Inverse-vol long:   IS={sharpe_ratio(is_iv):+.3f}  OOS={sharpe_ratio(oos_iv):+.3f}")

# 1c. MinVar long-only WITHOUT any TSMOM tilt
# Pure minimum-variance portfolio using the SAME 60-day covariance
LOOKBACK = 60
common_cols = sorted(returns.columns.tolist())
minvar_pure_ret = pd.Series(0.0, index=returns.index, dtype=float)
for i in range(LOOKBACK, len(returns)):
    window = returns.iloc[max(0, i - LOOKBACK) : i][common_cols].dropna(axis=1, how="any")
    if window.shape[1] < 2 or window.shape[0] < LOOKBACK:
        continue
    sigma = window.cov().values
    n = sigma.shape[0]
    try:
        sigma_inv = np.linalg.inv(sigma + np.eye(n) * 1e-8)
    except np.linalg.LinAlgError:
        continue
    ones = np.ones(n)
    raw_mv = sigma_inv @ ones
    denom = ones @ raw_mv
    if denom == 0:
        continue
    mv_w = raw_mv / denom
    mv_w = np.maximum(mv_w, 0)
    total = mv_w.sum()
    if total > 0:
        mv_w /= total
    day_ret = returns.iloc[i][window.columns].values
    if np.any(np.isnan(day_ret)):
        continue
    minvar_pure_ret.iloc[i] = float(mv_w @ day_ret)

minvar_pure_ret = minvar_pure_ret[minvar_pure_ret != 0]
is_mvp, oos_mvp = split_is_oos(minvar_pure_ret)
print(f"  MinVar long (no tilt): IS={sharpe_ratio(is_mvp):+.3f}  OOS={sharpe_ratio(oos_mvp):+.3f}")

# 1d. MinVar + TSMOM (the claimed strategy)
minvar_bt = pd.read_parquet(BACKTEST_OUT / "minvar_momentum_tilt.parquet")
is_mv, oos_mv = split_is_oos(minvar_bt["net_return"])
print(f"  MinVar+TSMOM:       IS={sharpe_ratio(is_mv):+.3f}  OOS={sharpe_ratio(oos_mv):+.3f}")

# 1e. Pure TSMOM
tsmom_bt = pd.read_parquet(BACKTEST_OUT / "tsmom.parquet")
is_ts, oos_ts = split_is_oos(tsmom_bt["net_return"])
print(f"  TSMOM:              IS={sharpe_ratio(is_ts):+.3f}  OOS={sharpe_ratio(oos_ts):+.3f}")

# Attribution: correlation of MinVar+TSMOM with benchmarks
oos_start = "2018-01-01"
oos_mv_series = minvar_bt["net_return"].loc[oos_start:]
oos_ew_series = ew_returns.loc[oos_start:]
oos_iv_series = inv_vol_returns.loc[oos_start:]

common = oos_mv_series.index.intersection(oos_ew_series.index).intersection(oos_iv_series.index)
if len(common) > 50:
    corr_ew = float(oos_mv_series.reindex(common).corr(oos_ew_series.reindex(common)))
    corr_iv = float(oos_mv_series.reindex(common).corr(oos_iv_series.reindex(common)))
    print("\n  OOS correlation of MinVar+TSMOM with:")
    print(f"    Equal-weight long:  {corr_ew:.3f}")
    print(f"    Inverse-vol long:   {corr_iv:.3f}")

# Regression: alpha after controlling for long-only beta
y = oos_mv_series.reindex(common).values
x_ew = oos_ew_series.reindex(common).values
slope_beta, intercept, r_val, p_val, se = stats.linregress(x_ew, y)
alpha_ann = float(intercept) * 252
beta = float(slope_beta)
print("\n  Regression vs equal-weight long-only (OOS):")
print(f"    beta  = {beta:.3f}")
print(f"    alpha = {alpha_ann:+.4f} (annualised)")
print(f"    R^2   = {r_val**2:.3f}")
print(f"    p(alpha=0) = {p_val:.4f}")

# =========================================================================
# INVESTIGATION 2: Multiple testing correction
# =========================================================================
print("\n" + "=" * 72)
print("INVESTIGATION 2: Multiple testing correction")
print("=" * 72)

# Count all variants tested
n_variants = 14  # 7 original + 7 from strategy_variants.py (minus duplicates ~= 13-14)
# OOS period length
oos_ret = minvar_bt["net_return"].loc[oos_start:]
n_days = len(oos_ret)
n_years = n_days / 252.0
se_sharpe = 1.0 / np.sqrt(n_years)

print(f"\n  Variants tested: {n_variants}")
print(f"  OOS period: {n_days} days ({n_years:.1f} years)")
print(f"  SE(Sharpe) under H0: {se_sharpe:.3f}")

best_oos_sharpe = sharpe_ratio(oos_ret)
z_score = best_oos_sharpe / se_sharpe
p_single = 1.0 - stats.norm.cdf(z_score)
p_multiple = 1.0 - (1.0 - p_single) ** n_variants

# Harvey, Liu & Zhu (2016) t-stat threshold for 14 tests
# Bonferroni: alpha/N = 0.05/14 = 0.0036 => z = 2.69
bonferroni_z = stats.norm.ppf(1 - 0.05 / n_variants)
bonferroni_sharpe = bonferroni_z * se_sharpe

# Holm-Bonferroni (less conservative)
# At rank 1 of 14: z = Phi^-1(1 - 0.05/14) = 2.69
print(f"\n  Best OOS Sharpe: {best_oos_sharpe:+.3f}")
print(f"  z-score (H0: Sharpe=0): {z_score:.2f}")
print(f"  p-value (single test): {p_single:.4f}")
print(f"  P(at least 1 of {n_variants} >= {best_oos_sharpe:.2f} | H0): {p_multiple:.4f}")
print(f"  Bonferroni-corrected threshold (5% family): Sharpe >= {bonferroni_sharpe:.2f}")
print(
    f"  Our Sharpe of {best_oos_sharpe:+.3f} {'PASSES' if best_oos_sharpe >= bonferroni_sharpe else 'FAILS'} Bonferroni at 5%"
)

# Bailey, Borwein, de Prado & Zhu (2014) deflated Sharpe
# Expected max Sharpe under null = SE * E[max of N standard normals]
# E[max(N normals)] ~ sqrt(2 * ln(N)) - (ln(ln(N)) + ln(4*pi)) / (2 * sqrt(2 * ln(N)))
if n_variants > 1:
    ln_n = np.log(n_variants)
    e_max_z = np.sqrt(2 * ln_n) - (np.log(ln_n) + np.log(4 * np.pi)) / (2 * np.sqrt(2 * ln_n))
    e_max_sharpe = e_max_z * se_sharpe
    deflated = best_oos_sharpe - e_max_sharpe
    print(f"\n  Expected max Sharpe under null (N={n_variants}): {e_max_sharpe:+.3f}")
    print(f"  Deflated Sharpe (actual - expected max): {deflated:+.3f}")
    print(
        f"  Deflated Sharpe {'> 0 (marginally significant)' if deflated > 0 else '<= 0 (NOT significant)'}"
    )

# =========================================================================
# INVESTIGATION 3: Year-by-year decomposition
# =========================================================================
print("\n" + "=" * 72)
print("INVESTIGATION 3: Year-by-year decomposition (OOS)")
print("=" * 72)

print(f"\n  {'Year':<6} {'Sharpe':>8} {'Cumulative':>12} {'Volatility':>12} {'MaxDD':>8}")
print("  " + "-" * 50)

positive_years = 0
for year in range(2018, 2025):
    yr_str = str(year)
    yr_ret = minvar_bt["net_return"].loc[yr_str]
    if len(yr_ret) < 50:
        print(f"  {year}   insufficient data ({len(yr_ret)} days)")
        continue
    yr_sharpe = sharpe_ratio(yr_ret)
    yr_cum = float(np.exp(yr_ret.sum()))
    yr_vol = float(yr_ret.std() * np.sqrt(252))
    yr_dd = float((np.exp(yr_ret.cumsum()) / np.exp(yr_ret.cumsum()).cummax() - 1).min())
    if yr_cum > 1.0:
        positive_years += 1
    print(f"  {year}   {yr_sharpe:+.2f}      {yr_cum:.3f}x       {yr_vol:.1%}       {yr_dd:+.1%}")

total_years = 2024 - 2018 + 1
print(f"\n  Positive years: {positive_years}/{total_years}")
print(f"  {'ROBUST' if positive_years >= 5 else 'CONCENTRATED'}: ", end="")
if positive_years >= 5:
    print("performance spread across most years")
else:
    print(f"only {positive_years}/{total_years} positive years — likely event-driven, not alpha")

# Check if 2021-2022 drives the result
yr_2122 = minvar_bt["net_return"].loc["2021":"2022"]
yr_rest = minvar_bt["net_return"].loc[oos_start:]
yr_rest = yr_rest[~yr_rest.index.year.isin([2021, 2022])]
if len(yr_2122) > 50 and len(yr_rest) > 50:
    sharpe_2122 = sharpe_ratio(yr_2122)
    sharpe_rest = sharpe_ratio(yr_rest)
    print(f"\n  2021-2022 Sharpe: {sharpe_2122:+.3f}")
    print(f"  Non-2021-2022 OOS Sharpe: {sharpe_rest:+.3f}")
    if sharpe_2122 > 2 * sharpe_rest and sharpe_rest < 0.3:
        print("  WARNING: Performance is DOMINATED by the commodity boom period")

# =========================================================================
# INVESTIGATION 4: Covariance matrix stability
# =========================================================================
print("\n" + "=" * 72)
print("INVESTIGATION 4: Covariance matrix condition number (60-day window)")
print("=" * 72)

check_dates = [
    "2010-01-04",
    "2015-01-02",
    "2018-01-02",
    "2020-01-02",
    "2020-04-20",
    "2022-01-03",
    "2024-01-02",
]

print(f"\n  {'Date':<14} {'Cond #':>10} {'Min Eigval':>14} {'N Assets':>10} {'Status':<20}")
print("  " + "-" * 72)

for date_str in check_dates:
    ts = pd.Timestamp(date_str)
    # Find nearest trading day
    idx_loc = returns.index.get_indexer([ts], method="nearest")[0]
    nearest = returns.index[idx_loc]
    start = max(0, idx_loc - LOOKBACK + 1)
    window = returns.iloc[start : idx_loc + 1].dropna(axis=1, how="any")

    if window.shape[0] < 30 or window.shape[1] < 2:
        print(f"  {nearest.date()!s:<14} insufficient data")
        continue

    cov_mat = window.cov().values
    eigvals = np.linalg.eigvalsh(cov_mat)
    min_eig = eigvals.min()
    max_eig = eigvals.max()
    cond = max_eig / max(min_eig, 1e-18)

    status = "OK" if cond < 100 else ("BORDERLINE" if cond < 500 else "UNSTABLE")
    print(
        f"  {nearest.date()!s:<14} {cond:>10.0f}   {min_eig:>14.2e}   {window.shape[1]:>10}   {status}"
    )

# Average condition number over time
cond_numbers = []
for i in range(LOOKBACK, len(returns), 5):  # sample every 5 days
    window = returns.iloc[max(0, i - LOOKBACK) : i].dropna(axis=1, how="any")
    if window.shape[0] < 30 or window.shape[1] < 2:
        continue
    cov_mat = window.cov().values
    eigvals = np.linalg.eigvalsh(cov_mat)
    if len(eigvals) > 0 and eigvals.min() > 0:
        cond_numbers.append(eigvals.max() / eigvals.min())

if cond_numbers:
    cond_arr = np.array(cond_numbers)
    print(f"\n  Summary over {len(cond_arr)} sampled dates:")
    print(f"    Median condition number: {np.median(cond_arr):.0f}")
    print(f"    Mean condition number:   {np.mean(cond_arr):.0f}")
    print(f"    95th percentile:         {np.percentile(cond_arr, 95):.0f}")
    print(f"    Max condition number:    {np.max(cond_arr):.0f}")
    print(
        f"    Dates with cond > 100:   {(cond_arr > 100).sum()}/{len(cond_arr)} "
        f"({(cond_arr > 100).mean() * 100:.0f}%)"
    )
    print(
        f"    Dates with cond > 500:   {(cond_arr > 500).sum()}/{len(cond_arr)} "
        f"({(cond_arr > 500).mean() * 100:.0f}%)"
    )

    # 13 assets, 60 observations: N/T ratio
    ratio_nt = n_commodities / LOOKBACK
    print(f"\n  N/T ratio: {n_commodities}/{LOOKBACK} = {ratio_nt:.2f}")
    if ratio_nt > 0.15:
        print("  WARNING: N/T > 0.15 — Ledoit-Wolf shrinkage is strongly recommended")
        print("  The current code uses naive sample covariance + 1e-8*I ridge")
        print("  This produces noisy, unstable weights that look like alpha but are noise")

# =========================================================================
# INVESTIGATION 5: Carry factor academic replication
# =========================================================================
print("\n" + "=" * 72)
print("INVESTIGATION 5: Carry factor — academic replication attempt")
print("=" * 72)

try:
    curves = load_curves()
    raw_carry = pd.DataFrame({sym: compute_carry(c) for sym, c in curves.items()})

    # Monthly rebalance: last business day carry, hold for 1 month
    monthly_carry = raw_carry.resample("BME").last()
    monthly_returns = returns.resample("BME").sum()  # sum of daily log returns

    # Align
    common_months = monthly_carry.index.intersection(monthly_returns.index)
    common_syms = sorted(set(monthly_carry.columns) & set(monthly_returns.columns))
    mc = monthly_carry.reindex(index=common_months, columns=common_syms)
    mr = monthly_returns.reindex(index=common_months, columns=common_syms)

    # Rank and select: long top 3, short bottom 3 (monthly)
    monthly_weights = rank_and_select(mc, long_n=3, short_n=3)
    monthly_weights_lagged = monthly_weights.shift(1)

    # Portfolio monthly returns
    portfolio_monthly = (monthly_weights_lagged * mr).sum(axis=1).dropna()

    if len(portfolio_monthly) > 12 and portfolio_monthly.std() > 0:
        monthly_mean = portfolio_monthly.mean()
        monthly_std = portfolio_monthly.std()
        sharpe_monthly = monthly_mean / monthly_std * np.sqrt(12)

        # Split IS/OOS at monthly level
        is_monthly = portfolio_monthly.loc[:"2017-12-31"]
        oos_monthly = portfolio_monthly.loc["2018-01-01":]

        if len(is_monthly) > 12 and is_monthly.std() > 0:
            is_sharpe_m = is_monthly.mean() / is_monthly.std() * np.sqrt(12)
        else:
            is_sharpe_m = float("nan")

        if len(oos_monthly) > 12 and oos_monthly.std() > 0:
            oos_sharpe_m = oos_monthly.mean() / oos_monthly.std() * np.sqrt(12)
        else:
            oos_sharpe_m = float("nan")

        print("\n  Monthly carry L/S (raw, full history):")
        print(f"    Full-sample Sharpe: {sharpe_monthly:+.3f}")
        print(f"    IS Sharpe:          {is_sharpe_m:+.3f}")
        print(f"    OOS Sharpe:         {oos_sharpe_m:+.3f}")
        print(f"    Monthly mean:       {monthly_mean * 100:+.2f}%")
        print(f"    Monthly std:        {monthly_std * 100:.2f}%")
        print(f"    N months:           {len(portfolio_monthly)}")

        # Compare to academic benchmark
        print("\n  Academic benchmark (Koijen et al. 2018):")
        print("    Carry Sharpe: 0.7-0.8 (24 commodities, 1972-2012)")
        print(
            f"    Expected with {len(common_syms)} assets: "
            f"~{0.75 * np.sqrt(len(common_syms) / 24):.2f}"
        )
        print(f"    We got: {sharpe_monthly:+.3f}")

        if sharpe_monthly < 0:
            print("\n  DIAGNOSIS: Negative carry Sharpe. Possible reasons:")
            print("    1. yfinance continuous series already embeds roll return (contango drag)")
            print("       The carry signal then anti-correlates with what's left in the return")
            print("    2. Z-scoring may be inverting the carry signal for some commodities")
            print("    3. 13 assets is thin — carry is a slow-moving cross-sectional signal")
    else:
        print("  Insufficient monthly data for carry analysis")
except Exception as exc:
    print(f"  FAILED to load curves: {exc}")

# =========================================================================
# INVESTIGATION 6: Transaction cost reality check
# =========================================================================
print("\n" + "=" * 72)
print("INVESTIGATION 6: Transaction cost and capacity analysis")
print("=" * 72)

universe_config = {
    "CL": {"multiplier": 1000, "unit": "bbl", "typical_price": 75},
    "NG": {"multiplier": 10000, "unit": "mmbtu", "typical_price": 3.5},
    "HO": {"multiplier": 42000, "unit": "gal", "typical_price": 2.5},
    "RB": {"multiplier": 42000, "unit": "gal", "typical_price": 2.5},
    "GC": {"multiplier": 100, "unit": "oz", "typical_price": 2000},
    "SI": {"multiplier": 5000, "unit": "oz", "typical_price": 25},
    "HG": {"multiplier": 25000, "unit": "lb", "typical_price": 4},
    "ZC": {"multiplier": 50, "unit": "bu", "typical_price": 450},  # cents -> dollars
    "ZS": {"multiplier": 50, "unit": "bu", "typical_price": 1300},  # cents -> dollars
    "ZW": {"multiplier": 50, "unit": "bu", "typical_price": 600},  # cents -> dollars
    "KC": {"multiplier": 375, "unit": "lb", "typical_price": 200},  # cents -> dollars
    "SB": {"multiplier": 1120, "unit": "lb", "typical_price": 20},  # cents -> dollars
    "CC": {"multiplier": 10, "unit": "tonne", "typical_price": 4000},
}

aum = 10_000_000
n_active_positions = 6  # typical: 3 long + 3 short
position_size = aum / n_active_positions

print(f"\n  Assumed AUM: ${aum / 1e6:.0f}M")
print(f"  Active positions: {n_active_positions}")
print(f"  Position size: ${position_size / 1e6:.2f}M\n")
print(f"  {'Symbol':<6} {'Notional/Ct':>14} {'# Contracts':>14} {'Capacity':>10}")
print("  " + "-" * 48)

for sym, spec in universe_config.items():
    notional = spec["multiplier"] * spec["typical_price"]
    n_contracts = position_size / notional
    capacity = "OK" if n_contracts < 50 else ("TIGHT" if n_contracts < 200 else "PROBLEMATIC")
    print(f"  {sym:<6} ${notional:>13,.0f} {n_contracts:>14.0f}   {capacity}")

print("\n  Note: KC, SB positions may require 100+ contracts at $10M AUM")
print("  Market impact at that scale is NOT captured in a 3-5 bps cost model")

# Cost double-counting check
print("\n  --- Roll return double-counting check ---")
print("  yfinance continuous futures use ratio-adjustment (Panama Canal method)")
print("  This means log returns ALREADY include the roll yield")
print("  The backtest ALSO subtracts roll_cost_bps on roll days")
print("  For contango markets (CL, NG), this is double-penalizing")
print("  For backwardated markets, this is double-crediting")
print("  Net effect depends on the portfolio composition")

# Check CL cumulative return from yfinance
cl_ret = returns["CL"].dropna()
cl_cum = float(np.exp(cl_ret.sum()))
cl_years = len(cl_ret) / 252
cl_ann = cl_cum ** (1.0 / cl_years) - 1
print(
    f"\n  CL cumulative return (yfinance continuous, {cl_ret.index[0].date()}"
    f" to {cl_ret.index[-1].date()}):"
)
print(f"    Total: {cl_cum:.3f}x ({(cl_cum - 1) * 100:+.1f}%)")
print(f"    Annualised: {cl_ann * 100:+.1f}% per year")
print(
    f"    (WTI spot went from ~$45 to ~$70 over this period = ~{((70 / 45) ** (1 / 20) - 1) * 100:.1f}% p.a.)"
)

if cl_ann < 0:
    print("    yfinance CL return is NEGATIVE => contango roll drag dominates")
    print("    Adding explicit roll costs ON TOP creates a pessimistic bias")
else:
    print("    yfinance CL return is positive")

# Check a few more
for sym in ["NG", "GC", "ZC"]:
    if sym in returns.columns:
        s = returns[sym].dropna()
        cum = float(np.exp(s.sum()))
        ann = cum ** (1.0 / (len(s) / 252)) - 1
        print(f"  {sym}: {cum:.3f}x total, {ann * 100:+.1f}% annualised")

# =========================================================================
# INVESTIGATION 7: Turnover and weight stability
# =========================================================================
print("\n" + "=" * 72)
print("INVESTIGATION 7: Weight stability and turnover")
print("=" * 72)

minvar_turnover = minvar_bt["turnover"].dropna()
print("\n  MinVar+TSMOM turnover:")
print(f"    Mean daily:  {minvar_turnover.mean():.4f}")
print(f"    Median:      {minvar_turnover.median():.4f}")
print(f"    95th pct:    {minvar_turnover.quantile(0.95):.4f}")
print(f"    Annualised:  {minvar_turnover.mean() * 252:.1f}x")

# Cost drag
# Average per-commodity cost ~ (3 + 2) / 10000 = 5 bps per unit turnover
avg_cost_rate = 5 / 10000
annual_cost_drag = minvar_turnover.mean() * 252 * avg_cost_rate * 100
print(f"    Estimated annual cost drag: {annual_cost_drag:.1f}% (at ~5 bps avg)")

# Compare gross vs net
is_gross, oos_gross = split_is_oos(minvar_bt["gross_return"])
print("\n  Gross vs Net Sharpe (OOS):")
print(f"    Gross: {sharpe_ratio(oos_gross):+.3f}")
print(f"    Net:   {sharpe_ratio(oos_mv):+.3f}")
print(f"    Cost drag on Sharpe: {sharpe_ratio(oos_gross) - sharpe_ratio(oos_mv):+.3f}")

# =========================================================================
# VERDICT
# =========================================================================
print("\n" + "=" * 72)
print("VERDICT")
print("=" * 72)

# Collect the key numbers
oos_sharpe_claim = sharpe_ratio(oos_mv)
oos_sharpe_ew = sharpe_ratio(oos_ew)
oos_sharpe_iv = sharpe_ratio(oos_iv)
oos_sharpe_mvpure = sharpe_ratio(oos_mvp)

print(f"""
1. IS MinVar+TSMOM REAL ALPHA OR DATA MINING?

   Claimed OOS Sharpe: {oos_sharpe_claim:+.3f}
   Equal-weight long:  {oos_sharpe_ew:+.3f}
   Inverse-vol long:   {oos_sharpe_iv:+.3f}
   MinVar (no tilt):   {oos_sharpe_mvpure:+.3f}
   Correlation with EW long: {corr_ew:.3f}

   Answer: {"The strategy IS largely long-only commodity beta with a slight momentum tilt." if corr_ew > 0.6 else "The strategy shows some independence from long-only beta." if corr_ew > 0.3 else "The strategy appears genuinely independent of long-only beta."}
   {"The TSMOM tilt adds marginal value over pure MinVar." if oos_sharpe_claim > oos_sharpe_mvpure + 0.15 else "The TSMOM tilt adds negligible value over pure MinVar."}

2. IS THE OOS SHARPE STATISTICALLY SIGNIFICANT AFTER MULTIPLE TESTING?

   P(at least 1 of {n_variants} >= {oos_sharpe_claim:.2f} | H0): {p_multiple:.1%}
   Deflated Sharpe: {deflated:+.3f}
   Bonferroni threshold: {bonferroni_sharpe:.2f}

   Answer: {"NO" if p_multiple > 0.10 else "MARGINAL" if p_multiple > 0.05 else "YES"} — {p_multiple:.0%} probability of finding this Sharpe by chance across {n_variants} variants.

3. IS THE BACKTEST METHODOLOGY SOUND?

   Issues found:""")

issues = []
if cond_numbers and np.median(cond_arr) > 100:
    issues.append(
        "   - Covariance matrix is ill-conditioned (median cond #: "
        f"{np.median(cond_arr):.0f}). MinVar weights are noise-amplifying."
    )
if n_commodities / LOOKBACK > 0.15:
    issues.append(
        f"   - N/T ratio = {n_commodities}/{LOOKBACK} = "
        f"{n_commodities / LOOKBACK:.2f}. Need shrinkage estimator."
    )
issues.append("   - Roll costs potentially double-counted: yfinance embeds roll yield,")
issues.append("     backtest also subtracts roll_cost_bps. Direction of bias depends on")
issues.append(
    "     portfolio composition (net long contango = pessimistic, net short = optimistic)."
)
issues.append(
    f"   - IS Sharpe ({sharpe_ratio(is_mv):+.3f}) vs OOS Sharpe "
    f"({oos_sharpe_claim:+.3f}) is suspicious: OOS >> IS suggests regime luck."
)
if positive_years < 5:
    issues.append(
        f"   - Only {positive_years}/7 OOS years are positive — "
        "performance is not robust across time."
    )

for issue in issues:
    print(issue)

# True Sharpe estimate
# Deflated Sharpe is the best estimate after multiple testing
# But also need to account for the long-only beta component
# True alpha Sharpe ~ correlation-adjusted information ratio
if corr_ew > 0:
    residual_vol_fraction = np.sqrt(1 - corr_ew**2)
    # Approximate: alpha_sharpe = (Sharpe_portfolio - beta * Sharpe_benchmark) / residual_vol_fraction
    alpha_sharpe_est = (oos_sharpe_claim - beta * oos_sharpe_ew) / max(residual_vol_fraction, 0.01)
else:
    alpha_sharpe_est = oos_sharpe_claim

print(f"""
4. WHAT IS THE TRUE SHARPE AFTER CORRECTING FOR ALL ISSUES?

   Raw OOS Sharpe:                   {oos_sharpe_claim:+.3f}
   Deflated (multiple testing):      {deflated:+.3f}
   Alpha Sharpe (vs long-only beta): {alpha_sharpe_est:+.3f}
   Best conservative estimate:       {min(deflated, alpha_sharpe_est):+.3f}

5. WHAT WOULD YOU DO DIFFERENTLY? (TOP 3)

   a) REPLACE SAMPLE COVARIANCE with Ledoit-Wolf shrinkage. With 13 assets
      and 60-day windows (N/T = 0.22), sample covariance is dominated by
      estimation error. This is the single biggest methodological fix.

   b) TEST AGAINST A PROPER BENCHMARK. The right null is not "Sharpe = 0"
      but "Sharpe = long-only commodity beta." Regress all variants against
      1/N or inverse-vol long-only and test the intercept. If alpha is not
      significant, you are marketing beta as alpha.

   c) EXTEND THE OOS PERIOD OR USE PURGED K-FOLD. 7 years of OOS data with
      14 variants and Sharpe SE = {se_sharpe:.2f} means you cannot distinguish
      luck from skill. Use combinatorial purged cross-validation (de Prado)
      to get tighter confidence intervals without data snooping.
""")

print("=" * 72)
print("END OF AUDIT")
print("=" * 72)
