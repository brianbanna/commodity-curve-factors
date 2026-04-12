"""Unit tests for validate.detect_gaps."""

import pandas as pd

from commodity_curve_factors.data.validate import detect_gaps


def test_detect_gaps_finds_long_nan_run():
    """A 7-day NaN gap should be detected with max_consecutive_missing=5."""
    dates = pd.bdate_range("2020-01-01", periods=20)
    vals = [1.0] * 5 + [float("nan")] * 7 + [1.0] * 8
    df = pd.DataFrame({"Close": vals}, index=dates)
    gaps = detect_gaps(df, max_consecutive_missing=5, col="Close")
    assert len(gaps) == 1
    assert gaps[0][2] == 7  # gap length


def test_detect_gaps_ignores_short_gaps():
    """A 3-day NaN gap should NOT be detected with max_consecutive_missing=5."""
    dates = pd.bdate_range("2020-01-01", periods=10)
    vals = [1.0] * 3 + [float("nan")] * 3 + [1.0] * 4
    df = pd.DataFrame({"Close": vals}, index=dates)
    gaps = detect_gaps(df, max_consecutive_missing=5, col="Close")
    assert len(gaps) == 0


def test_detect_gaps_no_column_checks_all_nan_rows():
    """With col=None, detect rows where ALL columns are NaN."""
    dates = pd.bdate_range("2020-01-01", periods=10)
    df = pd.DataFrame(
        {"A": [1] * 3 + [float("nan")] * 6 + [1], "B": [1] * 3 + [float("nan")] * 6 + [1]},
        index=dates,
    )
    gaps = detect_gaps(df, max_consecutive_missing=5, col=None)
    assert len(gaps) == 1
    assert gaps[0][2] == 6


def test_detect_gaps_exact_threshold_inclusive():
    """A gap of exactly max_consecutive_missing should be detected (inclusive)."""
    dates = pd.bdate_range("2020-01-01", periods=12)
    vals = [1.0] * 2 + [float("nan")] * 5 + [1.0] * 5
    df = pd.DataFrame({"Close": vals}, index=dates)
    gaps = detect_gaps(df, max_consecutive_missing=5, col="Close")
    assert len(gaps) == 1
    assert gaps[0][2] == 5


def test_detect_gaps_no_nans():
    """DataFrame with no NaN values should return an empty list."""
    dates = pd.bdate_range("2020-01-01", periods=10)
    df = pd.DataFrame({"Close": [1.0] * 10}, index=dates)
    gaps = detect_gaps(df, max_consecutive_missing=5, col="Close")
    assert gaps == []


def test_detect_gaps_multiple_gaps_sorted():
    """Multiple gaps should all be detected and sorted by start date."""
    dates = pd.bdate_range("2020-01-01", periods=30)
    vals = [1.0] * 2 + [float("nan")] * 6 + [1.0] * 2 + [float("nan")] * 7 + [1.0] * 13
    df = pd.DataFrame({"Close": vals}, index=dates)
    gaps = detect_gaps(df, max_consecutive_missing=5, col="Close")
    assert len(gaps) == 2
    assert gaps[0][2] == 6
    assert gaps[1][2] == 7
    assert gaps[0][0] < gaps[1][0]


def test_detect_gaps_col_only_checks_that_column():
    """With col specified, only that column's NaNs are used (not all-NaN rows)."""
    dates = pd.bdate_range("2020-01-01", periods=12)
    # Column A has a 6-day gap; column B is fully populated
    a_vals = [1.0] * 2 + [float("nan")] * 6 + [1.0] * 4
    b_vals = [1.0] * 12
    df = pd.DataFrame({"A": a_vals, "B": b_vals}, index=dates)
    gaps_a = detect_gaps(df, max_consecutive_missing=5, col="A")
    gaps_none = detect_gaps(df, max_consecutive_missing=5, col=None)
    assert len(gaps_a) == 1
    assert gaps_none == []  # col=None requires ALL cols to be NaN
