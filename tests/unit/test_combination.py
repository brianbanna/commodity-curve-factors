"""Tests for the factor combination module."""

import numpy as np
import pandas as pd

from commodity_curve_factors.factors.combination import (
    equal_weight_composite,
    ic_weighted_composite,
    regime_conditioned_composite,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_factor(
    dates: pd.DatetimeIndex,
    cols: list[str],
    seed: int = 0,
) -> pd.DataFrame:
    """Return a random factor DataFrame."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.standard_normal((len(dates), len(cols))),
        index=dates,
        columns=cols,
    )


# ---------------------------------------------------------------------------
# equal_weight_composite
# ---------------------------------------------------------------------------


def test_equal_weight_is_nanmean() -> None:
    """Composite must equal manual nanmean across factors per (date, col)."""
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    cols = ["CL", "GC", "NG"]
    rng = np.random.default_rng(7)
    a = pd.DataFrame(rng.standard_normal((50, 3)), index=dates, columns=cols)
    b = pd.DataFrame(rng.standard_normal((50, 3)), index=dates, columns=cols)
    factors = {"carry": a, "slope": b}

    result = equal_weight_composite(factors)

    expected = (a + b) / 2  # no NaNs → simple mean
    pd.testing.assert_frame_equal(result, expected[sorted(cols)], check_names=False)


def test_equal_weight_handles_different_nan_patterns() -> None:
    """When factor A is NaN for GC, composite should use factor B's value only."""
    dates = pd.date_range("2020-01-01", periods=10, freq="B")

    a = pd.DataFrame(
        {"CL": [1.0] * 10, "GC": [np.nan] * 10},
        index=dates,
    )
    b = pd.DataFrame(
        {"CL": [2.0] * 10, "GC": [3.0] * 10},
        index=dates,
    )
    result = equal_weight_composite({"a": a, "b": b})

    # CL: mean(1, 2) = 1.5
    assert np.allclose(result["CL"].to_numpy(), 1.5)
    # GC: only b contributes → 3.0
    assert np.allclose(result["GC"].to_numpy(), 3.0)


def test_equal_weight_all_nan_stays_nan() -> None:
    """Where every factor is NaN, the composite must also be NaN."""
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    a = pd.DataFrame({"CL": [np.nan] * 5}, index=dates)
    b = pd.DataFrame({"CL": [np.nan] * 5}, index=dates)
    result = equal_weight_composite({"a": a, "b": b})
    assert result["CL"].isna().all()


# ---------------------------------------------------------------------------
# ic_weighted_composite
# ---------------------------------------------------------------------------


def test_ic_weighted_shape() -> None:
    """IC-weighted composite must have the same date/column shape as the union."""
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    cols = ["CL", "GC", "NG", "ZC"]
    factors = {
        "carry": _make_factor(dates, cols, seed=1),
        "slope": _make_factor(dates, cols, seed=2),
        "tsmom": _make_factor(dates, cols, seed=3),
    }
    fwd = _make_factor(dates, cols, seed=99)

    result = ic_weighted_composite(factors, fwd, lookback=60, min_observations=10)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (len(dates), len(cols))
    assert set(result.columns) == set(cols)


def test_ic_weighted_falls_back_to_ew_with_insufficient_data() -> None:
    """With fewer rows than min_observations, fall back to equal weight."""
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    cols = ["CL", "GC"]
    rng = np.random.default_rng(42)
    a = pd.DataFrame(rng.standard_normal((30, 2)), index=dates, columns=cols)
    b = pd.DataFrame(rng.standard_normal((30, 2)), index=dates, columns=cols)
    fwd = pd.DataFrame(rng.standard_normal((30, 2)), index=dates, columns=cols)

    # With min_observations=252, every rebalance date falls back to equal weight.
    ic_result = ic_weighted_composite({"a": a, "b": b}, fwd, lookback=252, min_observations=252)
    ew_result = equal_weight_composite({"a": a, "b": b})

    # At equal weight, IC result == EW result.
    pd.testing.assert_frame_equal(ic_result, ew_result, check_names=False)


def test_ic_weighted_returns_dataframe_with_datetime_index() -> None:
    """Return type is a DataFrame with DatetimeIndex."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    cols = ["CL"]
    rng = np.random.default_rng(0)
    f = pd.DataFrame(rng.standard_normal((100, 1)), index=dates, columns=cols)
    fwd = pd.DataFrame(rng.standard_normal((100, 1)), index=dates, columns=cols)

    result = ic_weighted_composite({"f": f}, fwd, lookback=20, min_observations=5)
    assert isinstance(result, pd.DataFrame)
    assert isinstance(result.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# regime_conditioned_composite
# ---------------------------------------------------------------------------


def test_regime_conditioned_uses_correct_weights() -> None:
    """In calm regime (VIX < 15), weights should reflect the calm weight dict."""
    dates = pd.date_range("2020-01-01", periods=20, freq="B")

    carry = pd.DataFrame({"CL": [2.0] * 20}, index=dates)
    slope = pd.DataFrame({"CL": [1.0] * 20}, index=dates)
    vix = pd.Series([10.0] * 20, index=dates)  # all calm

    weights_by_regime = {
        "calm": {"carry": 0.8, "slope": 0.2},
        "moderate": {"carry": 0.5, "slope": 0.5},
        "turbulent": {"carry": 0.2, "slope": 0.8},
    }

    result = regime_conditioned_composite(
        {"carry": carry, "slope": slope},
        vix,
        vix_thresholds=[15.0, 25.0],
        weights_by_regime=weights_by_regime,
    )

    # Expected: 0.8 * 2.0 + 0.2 * 1.0 = 1.8
    expected_val = 0.8 * 2.0 + 0.2 * 1.0
    assert np.allclose(result["CL"].to_numpy(), expected_val), (
        f"Expected {expected_val}, got {result['CL'].to_numpy()}"
    )


def test_regime_conditioned_switches_on_vix_threshold() -> None:
    """Regime changes when VIX crosses the threshold."""
    dates = pd.date_range("2020-01-01", periods=20, freq="B")

    carry = pd.DataFrame({"CL": [4.0] * 20}, index=dates)
    slope = pd.DataFrame({"CL": [1.0] * 20}, index=dates)

    # First 10 days: VIX = 10 (calm), next 10: VIX = 30 (turbulent)
    vix_vals = [10.0] * 10 + [30.0] * 10
    vix = pd.Series(vix_vals, index=dates)

    weights_by_regime = {
        "calm": {"carry": 1.0, "slope": 0.0},  # only carry
        "moderate": {"carry": 0.5, "slope": 0.5},
        "turbulent": {"carry": 0.0, "slope": 1.0},  # only slope
    }

    result = regime_conditioned_composite(
        {"carry": carry, "slope": slope},
        vix,
        vix_thresholds=[15.0, 25.0],
        weights_by_regime=weights_by_regime,
    )

    calm_vals = result["CL"].iloc[:10].to_numpy()
    turbulent_vals = result["CL"].iloc[10:].to_numpy()

    # Calm: only carry (4.0); turbulent: only slope (1.0)
    assert np.allclose(calm_vals, 4.0), f"Calm period should be 4.0, got {calm_vals}"
    assert np.allclose(turbulent_vals, 1.0), f"Turbulent period should be 1.0, got {turbulent_vals}"


def test_regime_conditioned_nan_factor_excluded() -> None:
    """If a factor is NaN, its weight is redistributed to the remaining factors."""
    dates = pd.date_range("2020-01-01", periods=5, freq="B")

    carry = pd.DataFrame({"CL": [np.nan] * 5}, index=dates)
    slope = pd.DataFrame({"CL": [2.0] * 5}, index=dates)
    vix = pd.Series([10.0] * 5, index=dates)  # calm

    weights_by_regime = {
        "calm": {"carry": 0.6, "slope": 0.4},
    }

    result = regime_conditioned_composite(
        {"carry": carry, "slope": slope},
        vix,
        vix_thresholds=[15.0, 25.0],
        weights_by_regime=weights_by_regime,
    )

    # carry is NaN → only slope contributes → value = 2.0
    assert np.allclose(result["CL"].to_numpy(), 2.0)


# ---------------------------------------------------------------------------
# No lookahead tests (applies to all composite methods)
# ---------------------------------------------------------------------------


def test_all_composite_methods_no_lookahead() -> None:
    """Changing the last row must not affect any earlier composite values."""
    rng = np.random.default_rng(99)
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    cols = ["CL", "GC"]

    a = pd.DataFrame(rng.standard_normal((100, 2)), index=dates, columns=cols)
    b = pd.DataFrame(rng.standard_normal((100, 2)), index=dates, columns=cols)
    factors = {"a": a, "b": b}

    # --- equal_weight ---
    ew1 = equal_weight_composite(factors)

    a2, b2 = a.copy(), b.copy()
    a2.iloc[-1] = 999.0
    b2.iloc[-1] = 999.0
    ew2 = equal_weight_composite({"a": a2, "b": b2})

    pd.testing.assert_frame_equal(ew1.iloc[:-1], ew2.iloc[:-1])

    # --- ic_weighted ---
    fwd = pd.DataFrame(rng.standard_normal((100, 2)), index=dates, columns=cols)
    ic1 = ic_weighted_composite(factors, fwd, lookback=20, min_observations=5)

    fwd2 = fwd.copy()
    fwd2.iloc[-1] = 999.0
    ic2 = ic_weighted_composite({"a": a2, "b": b2}, fwd2, lookback=20, min_observations=5)

    pd.testing.assert_frame_equal(ic1.iloc[:-1], ic2.iloc[:-1])

    # --- regime_conditioned ---
    vix = pd.Series(rng.uniform(5, 35, 100), index=dates)
    weights = {
        "calm": {"a": 0.6, "b": 0.4},
        "moderate": {"a": 0.5, "b": 0.5},
        "turbulent": {"a": 0.3, "b": 0.7},
    }
    reg1 = regime_conditioned_composite(factors, vix, [15, 25], weights)
    reg2 = regime_conditioned_composite({"a": a2, "b": b2}, vix, [15, 25], weights)

    pd.testing.assert_frame_equal(reg1.iloc[:-1], reg2.iloc[:-1])


# ---------------------------------------------------------------------------
# Edge-case: empty factor dict
# ---------------------------------------------------------------------------


def test_equal_weight_empty_factors_returns_empty() -> None:
    """Empty factor dict returns an empty DataFrame without raising."""
    result = equal_weight_composite({})
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_ic_weighted_empty_factors_returns_empty() -> None:
    """Empty factor dict returns an empty DataFrame without raising."""
    fwd = pd.DataFrame()
    result = ic_weighted_composite({}, fwd)
    assert isinstance(result, pd.DataFrame)
    assert result.empty
