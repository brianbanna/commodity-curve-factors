"""Strategy variant investigation — Phases 4 & 5 of the signal recovery sprint.

Builds and backtests 6 new strategy variants using ONLY factors with confirmed
lag=1 predictive power (TSMOM, XSMOM) and carry as a regime/direction filter.

Variants:
  4.1  TSMOM + XSMOM equal-weight momentum composite
  4.2  Carry-conditioned TSMOM (carry as conviction filter)
  4.3  Pure time-series carry (backwardation/contango signal, no ranking)
  4.4  Carry regime switch (VIX: calm=carry, turbulent=TSMOM, moderate=blend)
  4.5  Calendar spread with TSMOM overlay
  4.6  Minimum-variance + momentum tilt (long-only)

Run:
    python scripts/strategy_variants.py
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Silence INFO/DEBUG noise during the investigation run
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from commodity_curve_factors.backtest.engine import run_backtest  # noqa: E402
from commodity_curve_factors.data.futures_loader import load_front_month_data  # noqa: E402
from commodity_curve_factors.evaluation.metrics import (  # noqa: E402
    max_drawdown,
    sharpe_ratio,
    split_is_oos,
)
from commodity_curve_factors.signals.calendar_spreads import calendar_spread_signal  # noqa: E402
from commodity_curve_factors.signals.portfolio import build_portfolio  # noqa: E402
from commodity_curve_factors.signals.ranking import rank_and_select, resample_weights_weekly  # noqa: E402
from commodity_curve_factors.signals.regime import classify_regime  # noqa: E402
from commodity_curve_factors.signals.threshold import threshold_signal  # noqa: E402
from commodity_curve_factors.utils.config import load_config  # noqa: E402
from commodity_curve_factors.utils.paths import DATA_PROCESSED, DATA_RAW  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BACKTEST_CFG = load_config("backtest")
UNIVERSE_CFG = load_config("universe")
COST_CONFIG = BACKTEST_CFG["costs"]
STRATEGY_CFG = load_config("strategy")

# Standardised constraint block for all variants
_VARIANT_STRATEGY_CFG: dict = {
    "constraints": {
        "vol_target": 0.10,
        "max_position_weight": 0.20,
        "max_sector_weight": 0.40,
        "max_leverage": 2.0,
    },
    "execution": {"lag_days": 1},
}

BACKTEST_OUT = DATA_PROCESSED / "backtest"
BACKTEST_OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _build_returns() -> pd.DataFrame:
    """Load front-month prices and compute daily log returns."""
    raw = load_front_month_data()
    closes: dict[str, pd.Series] = {}
    for sym, df in raw.items():
        col = "Close" if "Close" in df.columns else "close"
        if col in df.columns:
            closes[sym] = df[col]
    prices = pd.DataFrame(closes)
    prices.index = pd.DatetimeIndex(prices.index)
    prices = prices.sort_index()
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    returns = np.log(prices).diff()
    return returns


def _load_vix() -> pd.Series:
    """Load VIX Close series from raw macro parquet."""
    vix_path = DATA_RAW / "macro" / "vix.parquet"
    vix_df = pd.read_parquet(vix_path)
    vix = vix_df["Close"]
    vix.index = pd.DatetimeIndex(vix.index)
    if vix.index.tz is not None:
        vix.index = vix.index.tz_localize(None)
    return vix


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------


def _run_variant(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    name: str,
    save_name: str,
) -> dict:
    """Run backtest for a given raw signal weight frame.

    Applies full build_portfolio chain (vol-target, position limits, sector
    limits, 1-day execution lag) before passing to run_backtest.

    Returns a dict with IS_Sharpe, OOS_Sharpe, Turnover, MaxDD, HitRate.
    """
    try:
        portfolio_weights = build_portfolio(
            weights,
            returns,
            _VARIANT_STRATEGY_CFG,
            UNIVERSE_CFG,
        )
        bt = run_backtest(portfolio_weights, returns, COST_CONFIG)

        if bt.empty:
            return _failed(name, "empty backtest result")

        bt.to_parquet(BACKTEST_OUT / f"{save_name}.parquet")

        is_r, oos_r = split_is_oos(bt["net_return"])
        is_sharpe = sharpe_ratio(is_r) if len(is_r) > 0 else float("nan")
        oos_sharpe = sharpe_ratio(oos_r) if len(oos_r) > 0 else float("nan")
        turnover = float(bt["turnover"].mean(skipna=True))
        max_dd = max_drawdown(bt["net_return"])
        hit_rate = float((bt["net_return"] > 0).mean())

        return {
            "Strategy": name,
            "IS_Sharpe": is_sharpe,
            "OOS_Sharpe": oos_sharpe,
            "Turnover": turnover,
            "MaxDD": max_dd,
            "HitRate": hit_rate,
            "Error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return _failed(name, str(exc))


def _failed(name: str, reason: str) -> dict:
    return {
        "Strategy": name,
        "IS_Sharpe": float("nan"),
        "OOS_Sharpe": float("nan"),
        "Turnover": float("nan"),
        "MaxDD": float("nan"),
        "HitRate": float("nan"),
        "Error": reason,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("=" * 70)
print("STRATEGY VARIANT INVESTIGATION — Phases 4 & 5")
print("=" * 70)

# ---- Load base data --------------------------------------------------------
print("\nLoading base data...")

tsmom = pd.read_parquet(DATA_PROCESSED / "factors" / "tsmom.parquet")
xsmom = pd.read_parquet(DATA_PROCESSED / "factors" / "xsmom.parquet")
carry_z = pd.read_parquet(DATA_PROCESSED / "factors" / "carry.parquet")
raw_carry = pd.read_parquet(DATA_PROCESSED / "curve_metrics" / "carry.parquet")

returns = _build_returns()
vix = _load_vix()

print(f"  TSMOM : {tsmom.shape}  {tsmom.index[0].date()} – {tsmom.index[-1].date()}")
print(f"  XSMOM : {xsmom.shape}  {xsmom.index[0].date()} – {xsmom.index[-1].date()}")
print(f"  Carry z: {carry_z.shape}")
print(f"  Raw carry: {raw_carry.shape}")
print(f"  Returns: {returns.shape}")
print(f"  VIX: {vix.shape[0]} obs")

results: list[dict] = []

# ===========================================================================
# 4.1  TSMOM + XSMOM equal-weight momentum composite
# ===========================================================================
print("\n--- 4.1 TSMOM + XSMOM equal-weight composite ---")

try:
    common_idx = tsmom.index.intersection(xsmom.index)
    common_cols = sorted(set(tsmom.columns) & set(xsmom.columns))
    t = tsmom.reindex(index=common_idx, columns=common_cols)
    x = xsmom.reindex(index=common_idx, columns=common_cols)
    momentum_composite = (t + x) / 2
    # Where at least one is non-NaN, keep value; where both NaN, set NaN
    both_nan = t.isna() & x.isna()
    momentum_composite = momentum_composite.where(~both_nan, np.nan)

    # Rank → weekly rebal → portfolio → backtest
    raw_w = rank_and_select(momentum_composite, long_n=3, short_n=3)
    raw_w = resample_weights_weekly(raw_w)
    r41 = _run_variant(
        raw_w, returns, "4.1 Momentum Composite (TSMOM+XSMOM EW)", "momentum_composite"
    )
    results.append(r41)
    print(f"  IS={r41['IS_Sharpe']:+.3f}  OOS={r41['OOS_Sharpe']:+.3f}  MaxDD={r41['MaxDD']:+.3f}")
except Exception as exc:  # noqa: BLE001
    r41 = _failed("4.1 Momentum Composite (TSMOM+XSMOM EW)", str(exc))
    results.append(r41)
    print(f"  FAILED: {exc}")

# ===========================================================================
# 4.2  Carry-conditioned TSMOM (carry as conviction filter)
# ===========================================================================
print("\n--- 4.2 Carry-conditioned TSMOM ---")

try:
    common_idx = raw_carry.index.intersection(tsmom.index)
    cols = sorted(set(raw_carry.columns) & set(tsmom.columns))
    carry_aligned = raw_carry.reindex(index=common_idx, columns=cols)
    tsmom_aligned = tsmom.reindex(index=common_idx, columns=cols)

    signal_42 = pd.DataFrame(0.0, index=common_idx, columns=cols)
    # Long when backwardated AND positive momentum
    signal_42[(carry_aligned > 0) & (tsmom_aligned > 0)] = 1.0
    # Short when contango AND negative momentum
    signal_42[(carry_aligned < 0) & (tsmom_aligned < 0)] = -1.0
    # Normalize by number of active positions
    n_active = (signal_42 != 0).sum(axis=1).replace(0, 1)
    signal_42 = signal_42.div(n_active, axis=0)

    raw_w42 = resample_weights_weekly(signal_42)
    r42 = _run_variant(raw_w42, returns, "4.2 Carry-Conditioned TSMOM", "carry_conditioned_tsmom")
    results.append(r42)
    print(f"  IS={r42['IS_Sharpe']:+.3f}  OOS={r42['OOS_Sharpe']:+.3f}  MaxDD={r42['MaxDD']:+.3f}")
except Exception as exc:  # noqa: BLE001
    r42 = _failed("4.2 Carry-Conditioned TSMOM", str(exc))
    results.append(r42)
    print(f"  FAILED: {exc}")

# ===========================================================================
# 4.3  Pure time-series carry (backwardation/contango signal, no ranking)
# ===========================================================================
print("\n--- 4.3 Pure time-series carry ---")

try:
    signal_43 = pd.DataFrame(0.0, index=raw_carry.index, columns=raw_carry.columns)
    signal_43[raw_carry > 0] = 1.0
    signal_43[raw_carry < 0] = -1.0
    # NaN where carry is NaN
    signal_43 = signal_43.where(raw_carry.notna(), np.nan)
    # Equal-weight: 1/N active positions
    n_active_43 = (signal_43 != 0).sum(axis=1).replace(0, 1)
    signal_43 = signal_43.div(n_active_43, axis=0)

    raw_w43 = resample_weights_weekly(signal_43)
    r43 = _run_variant(raw_w43, returns, "4.3 Pure TS Carry", "ts_carry")
    results.append(r43)
    print(f"  IS={r43['IS_Sharpe']:+.3f}  OOS={r43['OOS_Sharpe']:+.3f}  MaxDD={r43['MaxDD']:+.3f}")
except Exception as exc:  # noqa: BLE001
    r43 = _failed("4.3 Pure TS Carry", str(exc))
    results.append(r43)
    print(f"  FAILED: {exc}")

# ===========================================================================
# 4.4  Carry regime switch (VIX-based)
# ===========================================================================
print("\n--- 4.4 Carry regime switch (VIX) ---")

try:
    regimes = classify_regime(vix)

    # Build the two component signals on a shared index
    common_idx44 = raw_carry.index.intersection(tsmom.index).intersection(regimes.index)
    cols44 = sorted(set(raw_carry.columns) & set(tsmom.columns))

    # TS carry signal (same as 4.3)
    ts_carry_sig = pd.DataFrame(0.0, index=common_idx44, columns=cols44)
    rc = raw_carry.reindex(index=common_idx44, columns=cols44)
    ts_carry_sig[rc > 0] = 1.0
    ts_carry_sig[rc < 0] = -1.0
    ts_carry_sig = ts_carry_sig.where(rc.notna(), 0.0)
    n44c = (ts_carry_sig != 0).sum(axis=1).replace(0, 1)
    ts_carry_sig = ts_carry_sig.div(n44c, axis=0)

    # TSMOM threshold signal
    ts_mom_sig = threshold_signal(tsmom.reindex(index=common_idx44, columns=cols44))
    n44m = (ts_mom_sig != 0).sum(axis=1).replace(0, 1)
    ts_mom_sig_norm = ts_mom_sig.div(n44m, axis=0).fillna(0.0)

    regime_series = regimes.reindex(common_idx44)
    signal_44 = pd.DataFrame(0.0, index=common_idx44, columns=cols44)

    # Calm → carry
    calm_mask = regime_series == "calm"
    signal_44[calm_mask] = ts_carry_sig[calm_mask]

    # Turbulent → TSMOM
    turb_mask = regime_series == "turbulent"
    signal_44[turb_mask] = ts_mom_sig_norm[turb_mask]

    # Moderate → 50/50 blend
    mod_mask = regime_series == "moderate"
    signal_44[mod_mask] = (ts_carry_sig[mod_mask] + ts_mom_sig_norm[mod_mask]) / 2

    raw_w44 = resample_weights_weekly(signal_44)
    r44 = _run_variant(raw_w44, returns, "4.4 Carry Regime Switch", "carry_regime_switch")
    results.append(r44)
    print(f"  IS={r44['IS_Sharpe']:+.3f}  OOS={r44['OOS_Sharpe']:+.3f}  MaxDD={r44['MaxDD']:+.3f}")
except Exception as exc:  # noqa: BLE001
    r44 = _failed("4.4 Carry Regime Switch", str(exc))
    results.append(r44)
    print(f"  FAILED: {exc}")

# ===========================================================================
# 4.5  Calendar spread with TSMOM overlay
# ===========================================================================
print("\n--- 4.5 Calendar spread with TSMOM overlay ---")

try:
    # Build carry z-score aligned with tsmom
    common_idx45 = carry_z.index.intersection(tsmom.index)
    cols45 = sorted(set(carry_z.columns) & set(tsmom.columns))
    carry_z_aligned = carry_z.reindex(index=common_idx45, columns=cols45)
    tsmom_45 = tsmom.reindex(index=common_idx45, columns=cols45)

    # Calendar spread signal using threshold ±0.5
    spread = calendar_spread_signal(carry_z_aligned, long_threshold=0.5, short_threshold=-0.5)

    # TSMOM filter: agree on direction (long spread → tsmom > 0; short → tsmom < 0)
    # The spread result has MultiIndex columns (commodity, leg)
    # We need to zero out spread positions where TSMOM disagrees
    for comm in cols45:
        if (comm, "front") in spread.columns:
            ts_comm = tsmom_45[comm]
            front_w = spread[(comm, "front")]
            back_w = spread[(comm, "back")]

            # Long spread (front=+1, back=-1) → only keep when tsmom > 0
            long_spread_mask = front_w > 0
            tsmom_disagrees_long = ts_comm <= 0
            spread.loc[long_spread_mask & tsmom_disagrees_long, (comm, "front")] = 0.0
            spread.loc[long_spread_mask & tsmom_disagrees_long, (comm, "back")] = 0.0

            # Short spread (front=-1, back=+1) → only keep when tsmom < 0
            short_spread_mask = front_w < 0
            tsmom_disagrees_short = ts_comm >= 0
            spread.loc[short_spread_mask & tsmom_disagrees_short, (comm, "front")] = 0.0
            spread.loc[short_spread_mask & tsmom_disagrees_short, (comm, "back")] = 0.0

    # Flatten: sum front+back per commodity to get net position
    # For calendar spreads the net of front+back is 0 (self-financing within comm);
    # report net weight as the average of abs front leg (as an approximation for PnL)
    flat_signal = pd.DataFrame(index=common_idx45, columns=cols45, dtype=float)
    for comm in cols45:
        if (comm, "front") in spread.columns:
            # Net position = front leg weight (back is opposite, so this captures direction)
            flat_signal[comm] = spread[(comm, "front")]
        else:
            flat_signal[comm] = 0.0

    # Normalize by number of active
    n45 = (flat_signal != 0).sum(axis=1).replace(0, 1)
    flat_signal = flat_signal.div(n45, axis=0)

    raw_w45 = resample_weights_weekly(flat_signal)
    r45 = _run_variant(
        raw_w45, returns, "4.5 Calendar Spread + TSMOM Overlay", "calendar_tsmom_overlay"
    )
    results.append(r45)
    print(f"  IS={r45['IS_Sharpe']:+.3f}  OOS={r45['OOS_Sharpe']:+.3f}  MaxDD={r45['MaxDD']:+.3f}")
except Exception as exc:  # noqa: BLE001
    r45 = _failed("4.5 Calendar Spread + TSMOM Overlay", str(exc))
    results.append(r45)
    print(f"  FAILED: {exc}")

# ===========================================================================
# 4.6  Minimum-variance + momentum tilt (long-only)
# ===========================================================================
print("\n--- 4.6 Minimum-variance + momentum tilt ---")

try:
    LOOKBACK = 60
    ALPHA = 0.5  # tilt strength

    common_idx46 = tsmom.index.intersection(returns.index)
    cols46 = sorted(set(tsmom.columns) & set(returns.columns))
    tsmom_46 = tsmom.reindex(index=common_idx46, columns=cols46)
    ret_46 = returns.reindex(index=common_idx46, columns=cols46)

    n_assets = len(cols46)
    signal_46 = pd.DataFrame(0.0, index=common_idx46, columns=cols46)

    for i, date in enumerate(common_idx46):
        if i < LOOKBACK:
            continue
        window_ret = ret_46.iloc[max(0, i - LOOKBACK) : i].dropna(axis=1, how="any")
        if window_ret.shape[1] < 2:
            continue

        sigma = window_ret.cov().values
        n = sigma.shape[0]
        present_cols = window_ret.columns.tolist()

        try:
            sigma_inv = np.linalg.inv(sigma + np.eye(n) * 1e-8)
        except np.linalg.LinAlgError:
            continue

        ones = np.ones(n)
        raw_mv = sigma_inv @ ones
        denom = ones @ raw_mv
        if denom == 0:
            continue
        mv_w = raw_mv / denom  # minimum-variance weights (sum to 1)

        # Momentum tilt: scale by (1 + alpha * rank-normalized tsmom)
        tsmom_cross = tsmom_46.loc[date, present_cols].fillna(0.0)
        # Cross-sectional rank to [-1, +1]
        valid_ts = tsmom_cross.dropna()
        if len(valid_ts) > 1:
            ranked = valid_ts.rank(pct=True) * 2 - 1  # map [0,1] → [-1,+1]
            tsmom_rank = tsmom_cross.copy()
            tsmom_rank[valid_ts.index] = ranked
        else:
            tsmom_rank = pd.Series(0.0, index=present_cols)

        tilt = 1.0 + ALPHA * tsmom_rank.values
        tilted_w = mv_w * tilt
        tilted_w = np.maximum(tilted_w, 0)  # long-only
        total = tilted_w.sum()
        if total > 0:
            tilted_w /= total

        row = pd.Series(0.0, index=cols46)
        for col_name, w_val in zip(present_cols, tilted_w):
            row[col_name] = float(w_val)
        signal_46.loc[date] = row

    raw_w46 = resample_weights_weekly(signal_46)
    r46 = _run_variant(raw_w46, returns, "4.6 MinVar + Momentum Tilt", "minvar_momentum_tilt")
    results.append(r46)
    print(f"  IS={r46['IS_Sharpe']:+.3f}  OOS={r46['OOS_Sharpe']:+.3f}  MaxDD={r46['MaxDD']:+.3f}")
except Exception as exc:  # noqa: BLE001
    r46 = _failed("4.6 MinVar + Momentum Tilt", str(exc))
    results.append(r46)
    print(f"  FAILED: {exc}")

# ===========================================================================
# Phase 5 — Baseline strategies from existing backtest files
# ===========================================================================
print("\n--- Phase 5: Existing strategy metrics ---")

EXISTING_STRATEGIES = {
    "TSMOM (original)": "tsmom",
    "XS Carry (original)": "xs_carry",
    "Multi-Factor EW (original)": "multi_factor_ew",
    "Multi-Factor IC (original)": "multi_factor_ic",
    "Regime Conditioned (original)": "regime_conditioned",
    "Sector Neutral (original)": "sector_neutral",
    "Calendar Spread (original)": "calendar_spread",
}

for strat_name, fname in EXISTING_STRATEGIES.items():
    fpath = BACKTEST_OUT / f"{fname}.parquet"
    if not fpath.exists():
        results.append(_failed(strat_name, "parquet not found"))
        continue
    try:
        bt = pd.read_parquet(fpath)
        is_r, oos_r = split_is_oos(bt["net_return"])
        results.append(
            {
                "Strategy": strat_name,
                "IS_Sharpe": sharpe_ratio(is_r) if len(is_r) > 0 else float("nan"),
                "OOS_Sharpe": sharpe_ratio(oos_r) if len(oos_r) > 0 else float("nan"),
                "Turnover": float(bt["turnover"].mean(skipna=True)),
                "MaxDD": max_drawdown(bt["net_return"]),
                "HitRate": float((bt["net_return"] > 0).mean()),
                "Error": None,
            }
        )
        print(
            f"  {strat_name:<38}  IS={results[-1]['IS_Sharpe']:+.3f}"
            f"  OOS={results[-1]['OOS_Sharpe']:+.3f}"
        )
    except Exception as exc:  # noqa: BLE001
        results.append(_failed(strat_name, str(exc)))
        print(f"  {strat_name}: FAILED: {exc}")

# ===========================================================================
# Results table
# ===========================================================================
print("\n" + "=" * 70)
print("FULL RESULTS TABLE")
print("=" * 70)

header = f"{'Strategy':<42}  {'IS_Sharpe':>10}  {'OOS_Sharpe':>10}  {'Turnover':>9}  {'MaxDD':>8}  {'HitRate':>8}"
print(header)
print("-" * len(header))

for r in results:

    def _fmt(v: float, prefix: str = "") -> str:
        if v != v:  # NaN check
            return "     n/a"
        return f"{prefix}{v:+.3f}"

    err_note = f"  [FAILED: {r['Error'][:40]}]" if r["Error"] else ""
    strat_display = r["Strategy"][:42]
    print(
        f"{strat_display:<42}  {_fmt(r['IS_Sharpe']):>10}  {_fmt(r['OOS_Sharpe']):>10}"
        f"  {_fmt(r['Turnover']):>9}  {_fmt(r['MaxDD']):>8}  {_fmt(r['HitRate']):>8}{err_note}"
    )

# ===========================================================================
# Analysis summary
# ===========================================================================
print("\n" + "=" * 70)
print("ANALYSIS SUMMARY")
print("=" * 70)

oos_positive = [
    (r["Strategy"], r["OOS_Sharpe"])
    for r in results
    if r["OOS_Sharpe"] == r["OOS_Sharpe"] and r["OOS_Sharpe"] > 0.40
]
oos_positive.sort(key=lambda x: x[1], reverse=True)

print("\nVariants with OOS Sharpe > 0.40:")
if oos_positive:
    for name, sr in oos_positive:
        print(f"  {name:<42}  OOS={sr:+.3f}")
else:
    print("  None achieved OOS Sharpe > 0.40")

# Best OOS overall
valid_oos = [
    (r["Strategy"], r["OOS_Sharpe"])
    for r in results
    if r["OOS_Sharpe"] == r["OOS_Sharpe"] and not r["Error"]
]
if valid_oos:
    best_strat, best_oos = max(valid_oos, key=lambda x: x[1])
    print(f"\nBest overall (OOS Sharpe): {best_strat}  ({best_oos:+.3f})")

print("\nDone.")
