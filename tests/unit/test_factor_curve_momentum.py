"""Tests for the curve momentum factor module."""

import numpy as np
import pandas as pd

from commodity_curve_factors.factors.curve_momentum import compute_curve_momentum_factor


def _make_curve(
    dates: pd.DatetimeIndex,
    f1m: list[float],
    f12m: list[float],
    n: int | None = None,
) -> pd.DataFrame:
    """Build a minimal curve DataFrame with all required tenor columns."""
    if n is None:
        n = len(dates)
    return pd.DataFrame(
        {
            "F1M": f1m,
            "F2M": [99.5] * n,
            "F3M": [99.0] * n,
            "F6M": [98.0] * n,
            "F9M": [97.0] * n,
            "F12M": f12m,
        },
        index=dates,
    )


def test_curve_momentum_positive_when_slope_increasing() -> None:
    """When slope increases at an accelerating rate, curve momentum z-score is positive.

    A linearly rising F12M produces a constant slope diff, which the expanding
    z-score maps to 0.  We therefore use a quadratically rising F12M so the
    60-day slope change itself grows over time, giving a positive z-score once
    enough history accumulates.
    """
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    f1m = [100.0] * 300
    # Quadratic rise: slope diff(60) increases over time → positive z-score
    f12m = [100.0 + (i**1.5) * 0.01 for i in range(300)]
    curve = _make_curve(dates, f1m, f12m, n=300)
    factor = compute_curve_momentum_factor({"CL": curve}, lookback_days=60, min_periods=20)
    cl = factor["CL"].dropna()
    # After warmup, slope change is growing → most z-scores should be positive
    assert (cl > 0).mean() > 0.5


def test_curve_momentum_lookback_respected() -> None:
    """Changing lookback_days should produce different results."""
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    f12m = [100.0 + np.sin(i / 30) * 5 for i in range(300)]
    curve = _make_curve(dates, [100.0] * 300, f12m, n=300)
    f20 = compute_curve_momentum_factor({"CL": curve}, lookback_days=20, min_periods=20)
    f60 = compute_curve_momentum_factor({"CL": curve}, lookback_days=60, min_periods=20)
    valid = pd.DataFrame({"f20": f20["CL"], "f60": f60["CL"]}).dropna()
    corr = valid.corr().iloc[0, 1]
    assert corr < 0.99  # different lookbacks → not identical


def test_curve_momentum_shape_and_columns() -> None:
    """Output has same shape as input curves, columns = commodity symbols."""
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(30)
    curve = pd.DataFrame(
        {
            "F1M": rng.normal(100, 5, 300),
            "F2M": rng.normal(99, 5, 300),
            "F3M": rng.normal(98, 5, 300),
            "F6M": rng.normal(97, 5, 300),
            "F9M": rng.normal(96, 5, 300),
            "F12M": rng.normal(95, 5, 300),
        },
        index=dates,
    )
    curves = {"CL": curve, "GC": curve.copy()}
    factor = compute_curve_momentum_factor(curves, lookback_days=20, min_periods=20)
    assert set(factor.columns) == {"CL", "GC"}
    assert len(factor) == 300


def test_curve_momentum_no_lookahead() -> None:
    """Factor at time t should not use data from t+1 (no lookahead)."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    f1m = [100.0] * 100
    f12m = [105.0 + i * 0.05 for i in range(100)]
    curve = _make_curve(dates, f1m, f12m, n=100)
    factor = compute_curve_momentum_factor({"CL": curve}, lookback_days=10, min_periods=5)

    # Spike the last F12M — earlier z-scores must be unaffected
    curve2 = curve.copy()
    curve2.iloc[-1, curve2.columns.get_loc("F12M")] = 9999.0
    factor2 = compute_curve_momentum_factor({"CL": curve2}, lookback_days=10, min_periods=5)

    pd.testing.assert_series_equal(factor["CL"].iloc[:-1], factor2["CL"].iloc[:-1])


def test_curve_momentum_all_nan_when_insufficient_history() -> None:
    """All rows are NaN when the series is shorter than min_periods."""
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    curve = _make_curve(dates, [100.0] * 10, [105.0] * 10, n=10)
    factor = compute_curve_momentum_factor({"CL": curve}, lookback_days=5, min_periods=252)
    assert factor["CL"].isna().all()


def test_curve_momentum_returns_dataframe_with_datetime_index() -> None:
    """Return type is a DataFrame with DatetimeIndex."""
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    curve = _make_curve(dates, [100.0] * 50, [105.0] * 50, n=50)
    factor = compute_curve_momentum_factor({"CL": curve}, lookback_days=10, min_periods=5)
    assert isinstance(factor, pd.DataFrame)
    assert isinstance(factor.index, pd.DatetimeIndex)
