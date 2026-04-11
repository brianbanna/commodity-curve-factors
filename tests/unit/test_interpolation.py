"""Tests for log-linear curve interpolation kernel."""

import datetime

import numpy as np
import pandas as pd


def test_time_to_expiry_years_basic() -> None:
    """time_to_expiry_years from 2020-01-01 to 2020-07-01 ≈ 0.4983."""
    from commodity_curve_factors.curves.interpolation import time_to_expiry_years

    result = time_to_expiry_years(
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-07-01"),
    )
    assert abs(result - 0.4983) < 0.001, f"Expected ~0.4983, got {result}"


def test_time_to_expiry_years_with_date_object() -> None:
    """time_to_expiry_years accepts a datetime.date as expiry_date."""
    from commodity_curve_factors.curves.interpolation import time_to_expiry_years

    result = time_to_expiry_years(
        pd.Timestamp("2020-01-01"),
        datetime.date(2020, 7, 1),
    )
    assert abs(result - 0.4983) < 0.001


def test_time_to_expiry_years_nat_returns_nan() -> None:
    """NaT on either side should return NaN."""
    from commodity_curve_factors.curves.interpolation import time_to_expiry_years

    assert np.isnan(time_to_expiry_years(pd.NaT, pd.Timestamp("2020-07-01")))
    assert np.isnan(time_to_expiry_years(pd.Timestamp("2020-01-01"), pd.NaT))


def test_log_linear_interpolate_monotone_upward() -> None:
    """Interpolation between valid points produces a result in the expected range."""
    from commodity_curve_factors.curves.interpolation import log_linear_interpolate

    tenors = np.array([0.1, 0.2, 0.3, 0.5])
    prices = np.array([50.0, 51.0, 52.0, 55.0])
    target = np.array([0.25])

    result = log_linear_interpolate(tenors, prices, target)

    assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"
    assert np.isfinite(result[0]), f"Expected finite result, got {result[0]}"
    assert 51.0 < result[0] < 53.0, (
        f"Expected result between 51 and 53 (interpolation), got {result[0]}"
    )


def test_log_linear_interpolate_handles_negative_prices() -> None:
    """Negative front price is dropped; fit on positive back-months produces finite result.

    This is a regression guard for WTI 2020-04-20 where the front month settled
    at -$37.63 while back months were all positive.
    """
    from commodity_curve_factors.curves.interpolation import log_linear_interpolate

    tenors = np.array([0.1, 0.2, 0.5, 1.0])
    prices = np.array([-37.63, 20.0, 25.0, 30.0])
    target = np.array([0.25])

    result = log_linear_interpolate(tenors, prices, target, min_points=3)

    assert np.isfinite(result[0]), (
        f"Expected finite result after dropping negative price, got {result[0]}"
    )
    assert result[0] > 0, f"Interpolated price must be positive, got {result[0]}"


def test_log_linear_interpolate_too_few_points() -> None:
    """Fewer than min_points valid points → all-NaN result."""
    from commodity_curve_factors.curves.interpolation import log_linear_interpolate

    tenors = np.array([0.1, 0.5])
    prices = np.array([50.0, 55.0])
    target = np.array([0.3, 0.4])

    result = log_linear_interpolate(tenors, prices, target, min_points=3)

    assert all(np.isnan(result)), f"Expected all NaN when fewer than min_points, got {result}"


def test_log_linear_interpolate_extrapolation_within_limit() -> None:
    """Target just outside observed range but within 45-day limit → finite result."""
    from commodity_curve_factors.curves.interpolation import DAYS_PER_YEAR, log_linear_interpolate

    tenors = np.array([0.5, 0.7, 1.0])
    prices = np.array([50.0, 52.0, 55.0])

    # Gap of 30 days → within the 45-day limit
    gap_years = 30.0 / DAYS_PER_YEAR
    target_inside = np.array([1.0 + gap_years])
    result_inside = log_linear_interpolate(tenors, prices, target_inside, extrapolation_max_days=45)
    assert np.isfinite(result_inside[0]), (
        f"Expected finite result within extrapolation limit, got {result_inside[0]}"
    )


def test_log_linear_interpolate_extrapolation_beyond_limit() -> None:
    """Target beyond observed range by more than 45 days → NaN."""
    from commodity_curve_factors.curves.interpolation import DAYS_PER_YEAR, log_linear_interpolate

    tenors = np.array([0.5, 0.7, 1.0])
    prices = np.array([50.0, 52.0, 55.0])

    # Gap of 60 days → exceeds the 45-day limit
    gap_years = 60.0 / DAYS_PER_YEAR
    target_outside = np.array([1.0 + gap_years])
    result_outside = log_linear_interpolate(
        tenors, prices, target_outside, extrapolation_max_days=45
    )
    assert np.isnan(result_outside[0]), (
        f"Expected NaN beyond extrapolation limit, got {result_outside[0]}"
    )


def test_interpolate_curve_day_returns_labeled_series() -> None:
    """Build a synthetic single-day DataFrame; result has correct labels and finite values."""
    from commodity_curve_factors.curves.interpolation import interpolate_curve_day

    trade_date = pd.Timestamp("2020-06-15")
    # Five contracts expiring at 1m, 2m, 3m, 6m, 9m, 12m from trade_date
    import datetime

    def _expiry(months: int) -> datetime.date:
        offset = pd.DateOffset(months=months)
        return (trade_date + offset).date()

    standard_tenors_months = [1, 3, 6, 12]

    rows = []
    for m, price in [(2, 50.0), (3, 51.0), (6, 53.0), (9, 55.0), (12, 57.0)]:
        rows.append(
            {
                "trade_date": trade_date,
                "lasttrddate": _expiry(m),
                "settlement": price,
            }
        )
    df = pd.DataFrame(rows)

    result = interpolate_curve_day(df, standard_tenors_months)

    expected_index = ["F1M", "F3M", "F6M", "F12M"]
    assert list(result.index) == expected_index, (
        f"Expected index {expected_index}, got {list(result.index)}"
    )

    # F3M through F12M should be finite (within observed range)
    for label in ["F3M", "F6M", "F12M"]:
        assert np.isfinite(result[label]), f"Expected finite value for {label}, got {result[label]}"


def test_interpolate_curve_day_empty_returns_nan_series() -> None:
    """Empty contracts_day → all-NaN Series with correct labels."""
    from commodity_curve_factors.curves.interpolation import interpolate_curve_day

    empty_df = pd.DataFrame(columns=["trade_date", "lasttrddate", "settlement"])
    result = interpolate_curve_day(empty_df, [1, 3, 6, 12])

    assert list(result.index) == ["F1M", "F3M", "F6M", "F12M"]
    assert result.isna().all()
