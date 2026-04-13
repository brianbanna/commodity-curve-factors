"""Signal recovery investigation for carry factor and contamination audit.

Phases 2–3 of the signal recovery sprint:
  Phase 2: Carry signal recovery (raw vs z-scored, monthly vs weekly rebal,
           within-sector, carry definition variants, contract-level carry,
           universe expansion check, inventory overlay)
  Phase 3: Contamination audit for all curve and non-curve factors

Run: python scripts/signal_recovery.py
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Silence noisy library output
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from commodity_curve_factors.backtest.engine import run_backtest  # noqa: E402
from commodity_curve_factors.curves.builder import load_curves  # noqa: E402
from commodity_curve_factors.curves.metrics import (  # noqa: E402
    compute_carry,
    compute_curvature,
    compute_slope,
    compute_term_carry,
)
from commodity_curve_factors.curves.roll_calendar import _active_contracts_from_group  # noqa: E402
from commodity_curve_factors.data.futures_loader import load_front_month_data  # noqa: E402
from commodity_curve_factors.evaluation.metrics import sharpe_ratio, split_is_oos  # noqa: E402
from commodity_curve_factors.factors.transforms import expanding_zscore_df  # noqa: E402
from commodity_curve_factors.signals.ranking import rank_and_select, resample_weights_weekly  # noqa: E402
from commodity_curve_factors.utils.config import load_config  # noqa: E402
from commodity_curve_factors.utils.paths import DATA_PROCESSED  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BACKTEST_CFG = load_config("backtest")
UNIVERSE_CFG = load_config("universe")
STRATEGY_CFG = {
    "constraints": {
        "vol_target": 0.10,
        "max_position_weight": 0.40,  # relaxed for investigation
        "max_sector_weight": 0.60,
        "max_leverage": 3.0,
    },
    "execution": {"lag_days": 1},
}
COST_CONFIG = BACKTEST_CFG["costs"]


def _build_returns() -> pd.DataFrame:
    """Load front-month prices and compute daily log returns."""
    raw = load_front_month_data()
    closes: dict[str, pd.Series] = {}
    for sym, df in raw.items():
        if "Close" in df.columns:
            closes[sym] = df["Close"]
        elif "close" in df.columns:
            closes[sym] = df["close"]
    prices = pd.DataFrame(closes)
    prices.index = pd.DatetimeIndex(prices.index)
    prices = prices.sort_index()
    # Remove timezone if present
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    returns = np.log(prices).diff()
    return returns


def resample_weights_monthly(weights: pd.DataFrame) -> pd.DataFrame:
    """Resample to monthly: take last business day of month, forward-fill."""
    monthly = weights.resample("BME").last()
    result: pd.DataFrame = monthly.reindex(weights.index, method="ffill")
    return result


def _run_pipeline(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    long_n: int = 3,
    short_n: int = 3,
    freq: str = "weekly",
    label: str = "",
) -> dict:
    """Rank → resample → portfolio → backtest → IS/OOS Sharpe."""
    w = rank_and_select(scores, long_n=long_n, short_n=short_n)
    if freq == "monthly":
        w = resample_weights_monthly(w)
    else:
        w = resample_weights_weekly(w)

    # Apply execution lag (1 day) manually — skip full build_portfolio to avoid
    # heavy vol-scaling for this investigation (we want cleaner signal comparison)
    w = w.shift(1)

    bt = run_backtest(w, returns, COST_CONFIG)
    if bt.empty:
        return {"IS": np.nan, "OOS": np.nan, "turnover": np.nan, "label": label}

    is_r, oos_r = split_is_oos(bt["net_return"])
    mean_to = float(bt["turnover"].mean(skipna=True))
    return {
        "IS": sharpe_ratio(is_r),
        "OOS": sharpe_ratio(oos_r),
        "turnover": mean_to,
        "label": label,
    }


def _compute_ic_table(factor: pd.DataFrame, returns: pd.DataFrame, n_days: int = 1000) -> dict:
    """Spearman IC at lag 0, 1, 5, 21 using the last n_days common dates."""
    common = factor.index.intersection(returns.index)
    if len(common) == 0:
        return {lag: (np.nan, np.nan, 0) for lag in ["lag0", "lag1", "lag5", "lag21"]}

    recent = common[-n_days:]
    results = {}
    for lag_name, lag_val in [("lag0", 0), ("lag1", 1), ("lag5", 5), ("lag21", 21)]:
        shifted_returns = returns.shift(-lag_val)
        ics = []
        for d in recent:
            f = factor.loc[d].dropna()
            r = shifted_returns.loc[d].reindex(f.index).dropna()
            c = f.index.intersection(r.index)
            if len(c) >= 5:
                ic, _ = spearmanr(f[c], r[c])
                if not np.isnan(ic):
                    ics.append(ic)
        if ics:
            arr = np.array(ics)
            results[lag_name] = (float(np.mean(arr)), float(np.mean(arr > 0)), len(arr))
        else:
            results[lag_name] = (np.nan, np.nan, 0)
    return results


# ---------------------------------------------------------------------------
# Load base data
# ---------------------------------------------------------------------------
print("=" * 70)
print("LOADING BASE DATA")
print("=" * 70)

curves = load_curves()
print(f"Curves loaded: {sorted(curves.keys())}")

returns = _build_returns()
print(f"Returns shape: {returns.shape}, {returns.index[0].date()} to {returns.index[-1].date()}")

# Raw carry (pre-z-score)
raw_carry = pd.DataFrame({sym: compute_carry(c) for sym, c in curves.items()})
print(f"Raw carry shape: {raw_carry.shape}")

# Z-scored carry (already saved)
carry_z = pd.read_parquet(DATA_PROCESSED / "factors" / "carry.parquet")
print(f"Z-scored carry shape: {carry_z.shape}")


# ===========================================================================
# PHASE 2: CARRY SIGNAL RECOVERY
# ===========================================================================
print("\n" + "=" * 70)
print("PHASE 2: CARRY SIGNAL RECOVERY")
print("=" * 70)

phase2_results: list[dict] = []

# ---------------------------------------------------------------------------
# 2.1  Raw carry levels vs z-scored carry (weekly rebal)
# ---------------------------------------------------------------------------
print("\n--- 2.1 Raw carry vs Z-scored carry (weekly) ---")

res_raw_weekly = _run_pipeline(raw_carry, returns, freq="weekly", label="Raw carry, weekly")
print(
    f"  Raw carry  weekly:   IS={res_raw_weekly['IS']:+.3f}  OOS={res_raw_weekly['OOS']:+.3f}"
    f"  turnover={res_raw_weekly['turnover']:.4f}"
)

res_z_weekly = _run_pipeline(carry_z, returns, freq="weekly", label="Z-scored carry, weekly")
print(
    f"  Z-scored   weekly:   IS={res_z_weekly['IS']:+.3f}  OOS={res_z_weekly['OOS']:+.3f}"
    f"  turnover={res_z_weekly['turnover']:.4f}"
)

phase2_results.extend([res_raw_weekly, res_z_weekly])

# IC at various lags — raw carry
ic_raw = _compute_ic_table(raw_carry, returns)
ic_z = _compute_ic_table(carry_z, returns)
print("\n  IC table — Raw carry (last 1000 days):")
for lag in ["lag0", "lag1", "lag5", "lag21"]:
    ic, hit, n = ic_raw[lag]
    print(f"    {lag}: IC={ic:+.4f}  hit={hit:.0%}  n={n}")
print("\n  IC table — Z-scored carry (last 1000 days):")
for lag in ["lag0", "lag1", "lag5", "lag21"]:
    ic, hit, n = ic_z[lag]
    print(f"    {lag}: IC={ic:+.4f}  hit={hit:.0%}  n={n}")

# ---------------------------------------------------------------------------
# 2.2  Monthly rebalancing
# ---------------------------------------------------------------------------
print("\n--- 2.2 Monthly rebalancing ---")

res_raw_monthly = _run_pipeline(raw_carry, returns, freq="monthly", label="Raw carry, monthly")
print(
    f"  Raw carry  monthly:  IS={res_raw_monthly['IS']:+.3f}  OOS={res_raw_monthly['OOS']:+.3f}"
    f"  turnover={res_raw_monthly['turnover']:.4f}"
)

res_z_monthly = _run_pipeline(carry_z, returns, freq="monthly", label="Z-scored carry, monthly")
print(
    f"  Z-scored   monthly:  IS={res_z_monthly['IS']:+.3f}  OOS={res_z_monthly['OOS']:+.3f}"
    f"  turnover={res_z_monthly['turnover']:.4f}"
)

phase2_results.extend([res_raw_monthly, res_z_monthly])

# ---------------------------------------------------------------------------
# 2.3  Within-sector carry
# ---------------------------------------------------------------------------
print("\n--- 2.3 Within-sector carry ---")

from commodity_curve_factors.utils.constants import SECTORS  # noqa: E402

sector_weights_list = []
for sector, syms in SECTORS.items():
    sector_cols = [s for s in syms if s in raw_carry.columns]
    if len(sector_cols) >= 2:
        sector_carry = raw_carry[sector_cols]
        sw = rank_and_select(sector_carry, long_n=1, short_n=1)
        sector_weights_list.append(sw)

if sector_weights_list:
    combined = pd.concat(sector_weights_list, axis=1).fillna(0.0)
    # Avoid duplicate columns in case any symbol appears twice
    combined = combined.loc[:, ~combined.columns.duplicated()]
    res_sector = _run_pipeline(combined, returns, freq="monthly", label="Within-sector carry, monthly")
    print(
        f"  Within-sector monthly: IS={res_sector['IS']:+.3f}  OOS={res_sector['OOS']:+.3f}"
        f"  turnover={res_sector['turnover']:.4f}"
    )
    phase2_results.append(res_sector)
else:
    print("  Not enough sectors with >= 2 commodities")

# ---------------------------------------------------------------------------
# 2.4  Carry definition variants
# ---------------------------------------------------------------------------
print("\n--- 2.4 Carry definition variants ---")

carry_variants: dict[str, pd.DataFrame] = {}

# (a) (F1M - F3M) / F3M * 4
carry_a = pd.DataFrame(
    {sym: (c["F1M"] - c["F3M"]) / c["F3M"] * 4 for sym, c in curves.items() if "F3M" in c.columns}
)
carry_variants["F1M-F3M (*4)"] = carry_a

# (b) log(F1M / F2M) * 12
carry_b_rows = {}
for sym, c in curves.items():
    if "F1M" in c.columns and "F2M" in c.columns:
        ratio = c["F1M"] / c["F2M"]
        # Guard against non-positive prices (WTI April 2020)
        ratio_safe = ratio.where(ratio > 0)
        carry_b_rows[sym] = np.log(ratio_safe) * 12
carry_b = pd.DataFrame(carry_b_rows)
carry_variants["log(F1M/F2M)*12"] = carry_b

# (c) (F1M - F6M) / F6M * 2
carry_c = pd.DataFrame(
    {sym: (c["F1M"] - c["F6M"]) / c["F6M"] * 2 for sym, c in curves.items() if "F6M" in c.columns}
)
carry_variants["F1M-F6M (*2)"] = carry_c

# (d) term carry: (F1M - F12M) / F12M
carry_d = pd.DataFrame({sym: compute_term_carry(c) for sym, c in curves.items()})
carry_variants["term_carry"] = carry_d

for vname, vc in carry_variants.items():
    # IC at key lags
    ic_v = _compute_ic_table(vc, returns)
    ic1 = ic_v["lag1"][0]
    ic5 = ic_v["lag5"][0]
    ic21 = ic_v["lag21"][0]
    print(f"\n  Variant: {vname}")
    print(f"    IC lag1={ic1:+.4f}  lag5={ic5:+.4f}  lag21={ic21:+.4f}")

    res_v = _run_pipeline(vc, returns, freq="monthly", label=f"Carry {vname}, monthly")
    print(
        f"    IS={res_v['IS']:+.3f}  OOS={res_v['OOS']:+.3f}"
        f"  turnover={res_v['turnover']:.4f}"
    )
    phase2_results.append(res_v)

# ---------------------------------------------------------------------------
# 2.5  Contract-level carry (no interpolation)
# ---------------------------------------------------------------------------
print("\n--- 2.5 Contract-level carry (no interpolation) ---")

from commodity_curve_factors.data.wrds_loader import load_contracts  # noqa: E402
from commodity_curve_factors.utils.config import load_config as _load_config  # noqa: E402

curve_cfg = _load_config("curve")
roll_rules = curve_cfg["roll_rules"]


def compute_contract_level_carry(symbol: str) -> pd.Series:
    """Compute carry from actual front and second contracts (no interpolation).

    carry = (front_settle - second_settle) / second_settle * 12
    """
    try:
        contracts = load_contracts(symbol)
    except FileNotFoundError:
        return pd.Series(dtype=float, name=symbol)

    roll_cfg = roll_rules.get(symbol, roll_rules["default"])
    roll_days = int(roll_cfg["roll_days_before_expiry"])

    records = []
    for ts_key, day_df in contracts.groupby("trade_date", sort=True):
        ts = pd.Timestamp(ts_key)
        active = _active_contracts_from_group(day_df, ts, roll_days)
        if len(active) < 2:
            continue
        # Sort by lasttrddate to get front and second
        sorted_active = active.sort_values("lasttrddate")
        front_settle = float(sorted_active.iloc[0]["settlement"])
        second_settle = float(sorted_active.iloc[1]["settlement"])
        if front_settle <= 0 or second_settle <= 0:
            continue
        carry_val = (front_settle - second_settle) / second_settle * 12
        records.append({"trade_date": ts, "carry": carry_val})

    if not records:
        return pd.Series(dtype=float, name=symbol)

    df_out = pd.DataFrame(records).set_index("trade_date")["carry"]
    df_out.name = symbol
    return df_out


contract_carry_series = {}
for sym in sorted(curves.keys()):
    s = compute_contract_level_carry(sym)
    if len(s) > 0:
        contract_carry_series[sym] = s
        print(f"  {sym}: {len(s)} days of contract-level carry")

if contract_carry_series:
    contract_carry = pd.DataFrame(contract_carry_series)
    ic_cc = _compute_ic_table(contract_carry, returns)
    print("\n  IC table — Contract-level carry (last 1000 days):")
    for lag in ["lag0", "lag1", "lag5", "lag21"]:
        ic, hit, n = ic_cc[lag]
        print(f"    {lag}: IC={ic:+.4f}  hit={hit:.0%}  n={n}")

    res_cc = _run_pipeline(
        contract_carry, returns, freq="monthly", label="Contract-level carry, monthly"
    )
    print(
        f"\n  Contract-level carry monthly: IS={res_cc['IS']:+.3f}  OOS={res_cc['OOS']:+.3f}"
        f"  turnover={res_cc['turnover']:.4f}"
    )
    phase2_results.append(res_cc)
else:
    print("  No contract-level carry data available")

# ---------------------------------------------------------------------------
# 2.6  Universe expansion feasibility
# ---------------------------------------------------------------------------
print("\n--- 2.6 Universe expansion feasibility ---")

contracts_dir = Path(PROJECT_ROOT) / "data" / "raw" / "futures" / "contracts"
if contracts_dir.exists():
    for d in sorted(contracts_dir.iterdir()):
        if d.is_dir() and d.name != "__pycache__":
            pq = d / "all_contracts.parquet"
            if pq.exists():
                df_check = pd.read_parquet(pq)
                td_min = pd.Timestamp(df_check['trade_date'].min()).date()
                td_max = pd.Timestamp(df_check['trade_date'].max()).date()
                print(
                    f"  {d.name}: {len(df_check):,} rows, "
                    f"{df_check['futcode'].nunique()} contracts, "
                    f"dates {td_min} to {td_max}"
                )
else:
    print(f"  contracts_dir not found: {contracts_dir}")

print("\n  Additional commodities feasible with WRDS access (not in current 13):")
additional = [
    ("Platinum",     "NYMEX",  "quarterly"),
    ("Palladium",    "NYMEX",  "quarterly"),
    ("Cotton",       "ICE",    "monthly"),
    ("Live Cattle",  "CME",    "bi-monthly"),
    ("Lean Hogs",    "CME",    "bi-monthly"),
    ("Brent Crude",  "ICE",    "monthly"),
    ("Soybean Oil",  "CBOT",   "monthly"),
    ("Soybean Meal", "CBOT",   "monthly"),
]
for name, exch, freq_note in additional:
    print(f"    {name:<16} ({exch}, {freq_note}) — requires new WRDS wrds_contrcode lookup")

# ---------------------------------------------------------------------------
# 2.7  Carry with inventory overlay
# ---------------------------------------------------------------------------
print("\n--- 2.7 Carry + inventory overlay (CL, NG, HO, RB) ---")

inv_path = DATA_PROCESSED / "factors" / "inventory.parquet"
if inv_path.exists():
    inventory = pd.read_parquet(inv_path)
    energy_syms = [s for s in ["CL", "NG", "HO", "RB"] if s in raw_carry.columns]

    conditional = raw_carry.copy()
    for sym in energy_syms:
        if sym in inventory.columns:
            inv = inventory[sym].reindex(conditional.index, method="ffill")
            # Long carry (backwardation) but inventory building → zero out
            conditional.loc[(conditional[sym] > 0) & (inv > 0), sym] = 0.0
            # Short carry (contango) but inventory drawing → zero out
            conditional.loc[(conditional[sym] < 0) & (inv < 0), sym] = 0.0

    ic_cond = _compute_ic_table(conditional, returns)
    print("  IC table — Carry + inventory overlay (last 1000 days):")
    for lag in ["lag0", "lag1", "lag5", "lag21"]:
        ic, hit, n = ic_cond[lag]
        print(f"    {lag}: IC={ic:+.4f}  hit={hit:.0%}  n={n}")

    res_cond = _run_pipeline(
        conditional, returns, freq="monthly", label="Carry + inventory overlay, monthly"
    )
    print(
        f"\n  Carry + inventory overlay monthly: IS={res_cond['IS']:+.3f}"
        f"  OOS={res_cond['OOS']:+.3f}  turnover={res_cond['turnover']:.4f}"
    )
    phase2_results.append(res_cond)
else:
    print("  Inventory factor not found — skipping overlay")

# ===========================================================================
# PHASE 3: CONTAMINATION AUDIT FOR ALL CURVE FACTORS
# ===========================================================================
print("\n" + "=" * 70)
print("PHASE 3: CONTAMINATION AUDIT")
print("=" * 70)

# Build raw metrics
raw_slope = pd.DataFrame({sym: compute_slope(c) for sym, c in curves.items()})
raw_curvature = pd.DataFrame({sym: compute_curvature(c) for sym, c in curves.items()})
raw_term_carry = pd.DataFrame({sym: compute_term_carry(c) for sym, c in curves.items()})

# Compute log(F1M/F2M)*12 as a "log carry" metric for auditing
raw_log_carry_rows = {}
for sym, c in curves.items():
    if "F1M" in c.columns and "F2M" in c.columns:
        ratio = c["F1M"] / c["F2M"]
        ratio_safe = ratio.where(ratio > 0)
        raw_log_carry_rows[sym] = np.log(ratio_safe) * 12
raw_log_carry = pd.DataFrame(raw_log_carry_rows)

factors_to_check: dict[str, pd.DataFrame] = {
    "carry_raw": raw_carry,
    "slope_raw": raw_slope,
    "curvature_raw": raw_curvature,
    "term_carry_raw": raw_term_carry,
}

# Add z-scored versions from disk
for name in ["carry", "slope", "curvature", "curve_momentum"]:
    fpath = DATA_PROCESSED / "factors" / f"{name}.parquet"
    if fpath.exists():
        factors_to_check[f"{name}_z"] = pd.read_parquet(fpath)

# Non-curve factors
for name in ["tsmom", "xsmom", "positioning", "inventory", "macro", "volatility"]:
    fpath = DATA_PROCESSED / "factors" / f"{name}.parquet"
    if fpath.exists():
        factors_to_check[f"{name}_z"] = pd.read_parquet(fpath)

print(f"\nFactors to audit: {sorted(factors_to_check.keys())}\n")

audit_results: dict[str, dict] = {}
print(
    f"{'Factor':<22}  {'lag0_IC':>8}  {'lag0_hit':>8}  {'lag1_IC':>8}  "
    f"{'lag1_hit':>8}  {'lag5_IC':>8}  {'lag21_IC':>8}  {'n':>5}"
)
print("-" * 90)

for fname, fdf in sorted(factors_to_check.items()):
    if fdf.empty:
        continue
    ics = _compute_ic_table(fdf, returns)
    audit_results[fname] = ics

    ic0, hit0, n0 = ics["lag0"]
    ic1, hit1, n1 = ics["lag1"]
    ic5, hit5, n5 = ics["lag5"]
    ic21, hit21, n21 = ics["lag21"]

    print(
        f"{fname:<22}  {ic0:+8.4f}  {hit0:8.1%}  {ic1:+8.4f}  "
        f"{hit1:8.1%}  {ic5:+8.4f}  {ic21:+8.4f}  {n1:5d}"
    )

# ---------------------------------------------------------------------------
# Decontamination: shift raw metrics by 1 day before z-scoring
# ---------------------------------------------------------------------------
print("\n--- Decontamination (shift raw metric by 1 day before z-scoring) ---\n")

raw_to_decontaminate = {
    "carry": raw_carry,
    "slope": raw_slope,
    "curvature": raw_curvature,
    "term_carry": raw_term_carry,
}

print(
    f"{'Factor':<22}  {'orig_lag0':>10}  {'orig_lag1':>10}  "
    f"{'decon_lag0':>11}  {'decon_lag1':>11}  {'Contaminated?':>14}"
)
print("-" * 90)

decon_results: dict[str, dict] = {}
for fname, raw_metric in raw_to_decontaminate.items():
    decontaminated = raw_metric.shift(1)
    decon_z = expanding_zscore_df(decontaminated, min_periods=252)

    ic_orig = _compute_ic_table(raw_metric, returns)
    ic_decon = _compute_ic_table(decon_z, returns)

    orig_lag0 = ic_orig["lag0"][0]
    orig_lag1 = ic_orig["lag1"][0]
    decon_lag0 = ic_decon["lag0"][0]
    decon_lag1 = ic_decon["lag1"][0]

    # Contaminated if lag0 IC materially exceeds lag1 IC for the raw metric
    contaminated = (
        not np.isnan(orig_lag0)
        and not np.isnan(orig_lag1)
        and abs(orig_lag0) > abs(orig_lag1) * 1.5
        and abs(orig_lag0) > 0.02
    )

    decon_results[fname] = {
        "orig_lag0": orig_lag0,
        "orig_lag1": orig_lag1,
        "decon_lag0": decon_lag0,
        "decon_lag1": decon_lag1,
        "contaminated": contaminated,
    }

    cont_str = "YES" if contaminated else "no"
    print(
        f"{fname:<22}  {orig_lag0:+10.4f}  {orig_lag1:+10.4f}  "
        f"{decon_lag0:+11.4f}  {decon_lag1:+11.4f}  {cont_str:>14}"
    )

# Decontaminated carry backtest
print("\n--- Decontaminated carry (shift-1 then z-score) backtest ---")
decon_carry_z = expanding_zscore_df(raw_carry.shift(1), min_periods=252)
res_decon_weekly = _run_pipeline(
    decon_carry_z, returns, freq="weekly", label="Decontaminated carry, weekly"
)
res_decon_monthly = _run_pipeline(
    decon_carry_z, returns, freq="monthly", label="Decontaminated carry, monthly"
)
print(
    f"  Decontaminated carry weekly:  IS={res_decon_weekly['IS']:+.3f}"
    f"  OOS={res_decon_weekly['OOS']:+.3f}  turnover={res_decon_weekly['turnover']:.4f}"
)
print(
    f"  Decontaminated carry monthly: IS={res_decon_monthly['IS']:+.3f}"
    f"  OOS={res_decon_monthly['OOS']:+.3f}  turnover={res_decon_monthly['turnover']:.4f}"
)
phase2_results.extend([res_decon_weekly, res_decon_monthly])

# ===========================================================================
# SUMMARY TABLES
# ===========================================================================
print("\n" + "=" * 70)
print("=== PHASE 2 CARRY RECOVERY SUMMARY ===")
print("=" * 70)
print(
    f"{'Variant':<40}  {'IS_Sharpe':>10}  {'OOS_Sharpe':>10}  {'Turnover':>9}"
)
print("-" * 75)
for r in phase2_results:
    is_s = f"{r['IS']:+.3f}" if not np.isnan(r["IS"]) else "  n/a"
    oos_s = f"{r['OOS']:+.3f}" if not np.isnan(r["OOS"]) else "  n/a"
    to_s = f"{r['turnover']:.4f}" if not np.isnan(r["turnover"]) else "  n/a"
    print(f"{r['label']:<40}  {is_s:>10}  {oos_s:>10}  {to_s:>9}")

print("\n" + "=" * 70)
print("=== PHASE 3 CONTAMINATION SUMMARY ===")
print("=" * 70)
print(
    f"{'Factor':<22}  {'lag0_IC':>8}  {'lag1_IC':>8}  "
    f"{'Contaminated?':>14}  {'Decon_lag1_IC':>14}"
)
print("-" * 75)
for fname, res in sorted(decon_results.items()):
    ic0 = res["orig_lag0"]
    ic1 = res["orig_lag1"]
    dcl1 = res["decon_lag1"]
    cont = "YES" if res["contaminated"] else "no"
    ic0s = f"{ic0:+.4f}" if not np.isnan(ic0) else "    n/a"
    ic1s = f"{ic1:+.4f}" if not np.isnan(ic1) else "    n/a"
    dcl1s = f"{dcl1:+.4f}" if not np.isnan(dcl1) else "    n/a"
    print(f"{fname:<22}  {ic0s:>8}  {ic1s:>8}  {cont:>14}  {dcl1s:>14}")

# Non-curve factors (no decontamination, just IC)
print("\nNon-curve factor ICs (from audit):")
print(f"{'Factor':<22}  {'lag0_IC':>8}  {'lag1_IC':>8}  {'lag5_IC':>8}  {'lag21_IC':>8}")
print("-" * 60)
non_curve = [k for k in audit_results if k not in decon_results and k not in
             {"carry_raw", "slope_raw", "curvature_raw", "term_carry_raw"}]
for fname in sorted(non_curve):
    ics = audit_results[fname]
    ic0 = ics["lag0"][0]
    ic1 = ics["lag1"][0]
    ic5 = ics["lag5"][0]
    ic21 = ics["lag21"][0]
    def _fmt(v: float) -> str:
        return f"{v:+.4f}" if not np.isnan(v) else "    n/a"

    print(f"{fname:<22}  {_fmt(ic0):>8}  {_fmt(ic1):>8}  {_fmt(ic5):>8}  {_fmt(ic21):>8}")

# ---------------------------------------------------------------------------
# Save investigation output to data/processed/investigation/
# ---------------------------------------------------------------------------
inv_dir = DATA_PROCESSED / "investigation"
inv_dir.mkdir(parents=True, exist_ok=True)

# Phase 2 results
phase2_df = pd.DataFrame(phase2_results)
phase2_df.to_parquet(inv_dir / "phase2_carry_recovery.parquet", index=False)
print(f"\nPhase 2 results saved → {inv_dir / 'phase2_carry_recovery.parquet'}")

# Phase 3 contamination audit
audit_rows = []
for fname, ics in audit_results.items():
    row = {"factor": fname}
    for lag in ["lag0", "lag1", "lag5", "lag21"]:
        row[f"{lag}_ic"] = ics[lag][0]
        row[f"{lag}_hit"] = ics[lag][1]
        row[f"{lag}_n"] = ics[lag][2]
    audit_rows.append(row)
audit_df = pd.DataFrame(audit_rows)
audit_df.to_parquet(inv_dir / "phase3_contamination_audit.parquet", index=False)
print(f"Phase 3 results saved → {inv_dir / 'phase3_contamination_audit.parquet'}")

print("\nDone.")
