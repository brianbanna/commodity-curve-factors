"""Tests for signals/directional.py."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.signals.directional import (
    apply_trend_filter,
    build_directional_weights,
    resample_weights_monthly,
)


@pytest.fixture()
def monthly_dates():
    return pd.date_range("2015-01-31", periods=60, freq="ME")


@pytest.fixture()
def daily_index():
    return pd.bdate_range("2015-01-01", periods=500)


@pytest.fixture()
def monthly_cy(monthly_dates):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "CL": 0.10 + rng.standard_normal(60) * 0.02,
            "NG": -0.02 + rng.standard_normal(60) * 0.02,
            "GC": 0.04 + rng.standard_normal(60) * 0.02,
        },
        index=monthly_dates,
    )


# ---------------------------------------------------------------------------
# apply_trend_filter
# ---------------------------------------------------------------------------


def test_apply_trend_filter_long_with_positive_trend():
    """CL long + TSMOM > 0 stays long."""
    dates = pd.bdate_range("2020-01-01", periods=10)
    positions = pd.DataFrame({"CL": [1.0] * 10}, index=dates)
    tsmom = pd.DataFrame({"CL": [0.5] * 10}, index=dates)
    result = apply_trend_filter(positions, tsmom)
    assert (result["CL"] == 1.0).all(), "Long position with positive TSMOM should remain 1.0"


def test_apply_trend_filter_short_with_negative_trend():
    """NG short + TSMOM < 0 stays short."""
    dates = pd.bdate_range("2020-01-01", periods=10)
    positions = pd.DataFrame({"NG": [-1.0] * 10}, index=dates)
    tsmom = pd.DataFrame({"NG": [-0.5] * 10}, index=dates)
    result = apply_trend_filter(positions, tsmom)
    assert (result["NG"] == -1.0).all(), "Short position with negative TSMOM should remain -1.0"


def test_apply_trend_filter_overrides_long_when_trend_negative():
    """Long position is zeroed when TSMOM <= 0."""
    dates = pd.bdate_range("2020-01-01", periods=10)
    positions = pd.DataFrame({"CL": [1.0] * 10}, index=dates)
    tsmom = pd.DataFrame({"CL": [-0.3] * 10}, index=dates)
    result = apply_trend_filter(positions, tsmom)
    assert (result["CL"] == 0.0).all(), "Long position should be zeroed when TSMOM <= 0"


def test_apply_trend_filter_overrides_short_when_trend_positive():
    """Short position is zeroed when TSMOM > 0."""
    dates = pd.bdate_range("2020-01-01", periods=10)
    positions = pd.DataFrame({"NG": [-0.5] * 10}, index=dates)
    tsmom = pd.DataFrame({"NG": [0.8] * 10}, index=dates)
    result = apply_trend_filter(positions, tsmom)
    assert (result["NG"] == 0.0).all(), "Short position should be zeroed when TSMOM > 0"


# ---------------------------------------------------------------------------
# resample_weights_monthly
# ---------------------------------------------------------------------------


def test_resample_weights_monthly():
    """Forward-fills monthly weights to daily index with correct shape and values."""
    monthly_idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    weights = pd.DataFrame({"CL": [1.0, 0.5, 0.0, -0.5, 1.0, 0.5]}, index=monthly_idx)
    daily_idx = pd.bdate_range("2020-01-01", "2020-06-30")
    result = resample_weights_monthly(weights, daily_idx)

    # Jan 31 (month-end) should be 1.0; Feb 3 (first Feb bday) should forward-fill to 1.0
    # until Feb 28 which is 0.5
    assert result.loc[pd.Timestamp("2020-01-31"), "CL"] == 1.0
    assert result.loc[pd.Timestamp("2020-02-03"), "CL"] == 1.0, (
        "Feb business days before Feb month-end should carry Jan 31 weight"
    )
    # 2020 is a leap year: Feb ME is Feb 29 (Saturday) → first bday in March
    # gets the Feb weight (0.5) via forward-fill
    assert result.loc[pd.Timestamp("2020-02-28"), "CL"] == 1.0, (
        "Feb 28 should still carry Jan 31 weight (Feb ME is Feb 29, a Saturday)"
    )
    assert result.loc[pd.Timestamp("2020-03-02"), "CL"] == 0.5, (
        "First March bday should carry Feb weight 0.5 (forward-filled from Feb 29 ME)"
    )
    assert result.shape[0] == len(daily_idx)


# ---------------------------------------------------------------------------
# build_directional_weights
# ---------------------------------------------------------------------------


def test_build_directional_weights_shape(monthly_cy, daily_index):
    """Output should have exactly 500 rows (= daily_index length)."""
    result = build_directional_weights(monthly_cy, monthly_cy, daily_index)
    assert result.shape[0] == 500, f"Expected 500 rows, got {result.shape[0]}"
    assert set(result.columns) == set(monthly_cy.columns)
