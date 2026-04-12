"""Tests for the curvature factor module."""

import numpy as np
import pandas as pd

from commodity_curve_factors.factors.curvature import compute_curvature_factor


def _make_curve(
    dates: pd.DatetimeIndex,
    f1m: list[float],
    f6m: list[float],
    f12m: list[float],
    n: int | None = None,
) -> pd.DataFrame:
    """Build a minimal curve DataFrame with all required tenor columns."""
    if n is None:
        n = len(dates)
    return pd.DataFrame(
        {
            "F1M": f1m,
            "F2M": [99.0] * n,
            "F3M": [98.0] * n,
            "F6M": f6m,
            "F9M": [96.0] * n,
            "F12M": f12m,
        },
        index=dates,
    )


def test_curvature_factor_shape_and_columns() -> None:
    """Output has same shape as input curves, columns = commodity symbols."""
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(20)
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
    factor = compute_curvature_factor(curves, min_periods=20)
    assert set(factor.columns) == {"CL", "GC"}
    assert len(factor) == 300
    assert factor.iloc[:19].isna().all().all()  # min_periods warmup
    assert factor.iloc[20:].notna().any().any()  # some values after warmup


def test_curvature_factor_sign_convention() -> None:
    """Positive curvature = F6M below F1M–F12M line (concave-up); negative = humped."""
    dates = pd.date_range("2020-01-01", periods=600, freq="B")
    f1m = [100.0] * 600
    f12m = [100.0] * 600
    # First 300: F6M high (above midpoint) → curvature = 100 - 2*105 + 100 = -10 (concave-down)
    # Last 300: F6M low (below midpoint) → curvature = 100 - 2*95 + 100 = 10 (concave-up)
    f6m = [105.0] * 300 + [95.0] * 300
    curve = _make_curve(dates, f1m, f6m, f12m, n=600)
    factor = compute_curvature_factor({"CL": curve}, min_periods=20)
    cl = factor["CL"].dropna()
    mid = len(cl) // 2
    # Concave-down half should have lower z-scores than concave-up half
    assert cl.iloc[:mid].mean() < cl.iloc[mid:].mean()


def test_curvature_factor_no_lookahead() -> None:
    """Factor at time t should not use data from t+1 (no lookahead)."""
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    f1m = [100.0] * 50
    f6m = [98.0 + i * 0.05 for i in range(50)]
    f12m = [96.0] * 50
    curve = _make_curve(dates, f1m, f6m, f12m, n=50)
    factor = compute_curvature_factor({"CL": curve}, min_periods=5)

    # Spike the last F6M — earlier z-scores must be unaffected
    curve2 = curve.copy()
    curve2.iloc[-1, curve2.columns.get_loc("F6M")] = 9999.0
    factor2 = compute_curvature_factor({"CL": curve2}, min_periods=5)

    pd.testing.assert_series_equal(factor["CL"].iloc[:-1], factor2["CL"].iloc[:-1])


def test_curvature_factor_all_nan_when_insufficient_history() -> None:
    """All rows are NaN when the series is shorter than min_periods."""
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    curve = _make_curve(dates, [100.0] * 10, [98.0] * 10, [96.0] * 10, n=10)
    factor = compute_curvature_factor({"CL": curve}, min_periods=252)
    assert factor["CL"].isna().all()


def test_curvature_factor_returns_dataframe_with_datetime_index() -> None:
    """Return type is a DataFrame with DatetimeIndex."""
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    curve = _make_curve(dates, [100.0] * 30, [98.0] * 30, [96.0] * 30, n=30)
    factor = compute_curvature_factor({"CL": curve}, min_periods=5)
    assert isinstance(factor, pd.DataFrame)
    assert isinstance(factor.index, pd.DatetimeIndex)
