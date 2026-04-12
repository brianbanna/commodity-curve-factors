"""Tests for the carry factor module."""

import numpy as np
import pandas as pd

from commodity_curve_factors.factors.carry import compute_carry_factor


def _make_curve(
    dates: pd.DatetimeIndex,
    f1m: list[float],
    f2m: list[float],
    n: int | None = None,
) -> pd.DataFrame:
    """Build a minimal curve DataFrame with all required tenor columns."""
    if n is None:
        n = len(dates)
    return pd.DataFrame(
        {
            "F1M": f1m,
            "F2M": f2m,
            "F3M": [98.0] * n,
            "F6M": [97.0] * n,
            "F9M": [96.0] * n,
            "F12M": [95.0] * n,
        },
        index=dates,
    )


def test_carry_factor_shape_and_columns() -> None:
    """Output has same shape as input curves, columns = commodity symbols."""
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(42)
    curve = pd.DataFrame(
        {
            "F1M": rng.normal(100, 5, 300),
            "F2M": np.random.default_rng(43).normal(99, 5, 300),
            "F3M": np.random.default_rng(44).normal(98, 5, 300),
            "F6M": np.random.default_rng(45).normal(97, 5, 300),
            "F9M": np.random.default_rng(46).normal(96, 5, 300),
            "F12M": np.random.default_rng(47).normal(95, 5, 300),
        },
        index=dates,
    )
    curves = {"CL": curve, "GC": curve.copy()}
    factor = compute_carry_factor(curves, min_periods=20)
    assert set(factor.columns) == {"CL", "GC"}
    assert len(factor) == 300
    assert factor.iloc[:19].isna().all().all()  # min_periods warmup
    assert factor.iloc[20:].notna().any().any()  # some values after warmup


def test_carry_factor_sign_transition() -> None:
    """Carry should be positive for backwardation, negative for contango."""
    dates = pd.date_range("2020-01-01", periods=600, freq="B")
    # First 300 days: contango (F1M < F2M) → negative carry
    # Last 300 days: backwardation (F1M > F2M) → positive carry
    f1m = [95.0] * 300 + [105.0] * 300
    f2m = [100.0] * 600
    curve = _make_curve(dates, f1m, f2m, n=600)
    factor = compute_carry_factor({"CL": curve}, min_periods=20)
    cl = factor["CL"].dropna()
    mid = len(cl) // 2
    assert cl.iloc[:mid].mean() < cl.iloc[mid:].mean()


def test_carry_factor_uses_expanding_zscore() -> None:
    """Factor at time t should not use data from t+1 (no lookahead)."""
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    f1m = list(range(100, 150))
    f2m = list(range(99, 149))
    curve = _make_curve(dates, f1m, f2m, n=50)
    factor = compute_carry_factor({"CL": curve}, min_periods=5)

    # Changing the last value should NOT affect earlier z-scores
    curve2 = curve.copy()
    curve2.iloc[-1, curve2.columns.get_loc("F1M")] = 999.0  # spike F1M on last day
    factor2 = compute_carry_factor({"CL": curve2}, min_periods=5)

    pd.testing.assert_series_equal(factor["CL"].iloc[:-1], factor2["CL"].iloc[:-1])


def test_carry_factor_all_nan_when_insufficient_history() -> None:
    """All rows are NaN when the series is shorter than min_periods."""
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    curve = _make_curve(dates, [100.0] * 10, [99.0] * 10, n=10)
    factor = compute_carry_factor({"CL": curve}, min_periods=252)
    assert factor["CL"].isna().all()


def test_carry_factor_returns_dataframe_with_datetime_index() -> None:
    """Return type is a DataFrame with DatetimeIndex."""
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    curve = _make_curve(dates, [100.0] * 30, [99.0] * 30, n=30)
    factor = compute_carry_factor({"CL": curve}, min_periods=5)
    assert isinstance(factor, pd.DataFrame)
    assert isinstance(factor.index, pd.DatetimeIndex)
