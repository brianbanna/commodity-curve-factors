"""Tests for signals/directional.py (long-biased regime tilt with trend tilt)."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.signals.directional import (
    apply_trend_tilt,
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
# apply_trend_tilt
# ---------------------------------------------------------------------------


def test_trend_tilt_scales_up_with_positive_tsmom():
    """Positive TSMOM should multiply positions by trend_up_mult."""
    dates = pd.bdate_range("2020-01-01", periods=10)
    positions = pd.DataFrame({"CL": [1.0] * 10}, index=dates)
    tsmom = pd.DataFrame({"CL": [0.5] * 10}, index=dates)
    result = apply_trend_tilt(positions, tsmom, trend_up_mult=1.2, trend_down_mult=0.7)
    assert result["CL"].iloc[0] == pytest.approx(1.2)


def test_trend_tilt_scales_down_with_negative_tsmom():
    """Negative TSMOM should multiply positions by trend_down_mult, not zero."""
    dates = pd.bdate_range("2020-01-01", periods=10)
    positions = pd.DataFrame({"CL": [1.0] * 10}, index=dates)
    tsmom = pd.DataFrame({"CL": [-0.5] * 10}, index=dates)
    result = apply_trend_tilt(positions, tsmom, trend_up_mult=1.2, trend_down_mult=0.7)
    assert result["CL"].iloc[0] == pytest.approx(0.7), (
        "Position should be scaled to 0.7x, not zeroed"
    )


def test_trend_tilt_preserves_zero_positions():
    """Zero positions should stay zero regardless of TSMOM."""
    dates = pd.bdate_range("2020-01-01", periods=10)
    positions = pd.DataFrame({"CL": [0.0] * 10}, index=dates)
    tsmom = pd.DataFrame({"CL": [1.0] * 10}, index=dates)
    result = apply_trend_tilt(positions, tsmom)
    assert (result["CL"] == 0.0).all()


# ---------------------------------------------------------------------------
# resample_weights_monthly
# ---------------------------------------------------------------------------


def test_resample_weights_monthly():
    """Forward-fills monthly weights to daily index with correct shape and values."""
    monthly_idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    weights = pd.DataFrame({"CL": [1.0, 0.5, 0.0, -0.5, 1.0, 0.5]}, index=monthly_idx)
    daily_idx = pd.bdate_range("2020-01-01", "2020-06-30")
    result = resample_weights_monthly(weights, daily_idx)

    assert result.loc[pd.Timestamp("2020-01-31"), "CL"] == 1.0
    assert result.loc[pd.Timestamp("2020-02-03"), "CL"] == 1.0
    assert result.shape[0] == len(daily_idx)


# ---------------------------------------------------------------------------
# build_directional_weights
# ---------------------------------------------------------------------------


def test_build_directional_weights_shape(monthly_cy, daily_index):
    """Output should have exactly 500 rows (= daily_index length)."""
    result = build_directional_weights(monthly_cy, monthly_cy, daily_index)
    assert result.shape[0] == 500, f"Expected 500 rows, got {result.shape[0]}"
    assert set(result.columns) == set(monthly_cy.columns)


def test_build_directional_weights_mostly_positive(monthly_cy, daily_index):
    """Long-biased strategy should have mostly positive weights."""
    tsmom = pd.DataFrame(
        np.ones((500, 3)),
        index=daily_index,
        columns=monthly_cy.columns,
    )
    result = build_directional_weights(monthly_cy, tsmom, daily_index)
    # With all-positive TSMOM and varying CY, most non-NaN weights should be > 0
    non_nan = result.dropna()
    pos_frac = (non_nan > 0).mean().mean()
    assert pos_frac > 0.5, f"Expected mostly positive non-NaN weights, got {pos_frac:.1%} positive"
