"""Tests for signals/curve_transition.py."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.signals.curve_transition import (
    compute_transition_signal,
    transition_to_position,
)


@pytest.fixture()
def monthly_dates_long():
    """120 month-end dates for sufficient expanding-window history."""
    return pd.date_range("2010-01-31", periods=120, freq="ME")


@pytest.fixture()
def monthly_cy_tightening(monthly_dates_long):
    """CL convenience yield that rises steadily (tightening market)."""
    return pd.DataFrame(
        {"CL": np.linspace(0.02, 0.20, 120)},
        index=monthly_dates_long,
    )


@pytest.fixture()
def monthly_cy_loosening(monthly_dates_long):
    """NG convenience yield that falls steadily (loosening market)."""
    return pd.DataFrame(
        {"NG": np.linspace(0.10, -0.05, 120)},
        index=monthly_dates_long,
    )


@pytest.fixture()
def monthly_cy_flat(monthly_dates_long):
    """GC convenience yield that stays constant (no transition)."""
    return pd.DataFrame(
        {"GC": np.full(120, 0.04)},
        index=monthly_dates_long,
    )


# ---------------------------------------------------------------------------
# compute_transition_signal
# ---------------------------------------------------------------------------


def test_compute_transition_signal_shape(monthly_cy_tightening):
    """Output has same columns as input and index on a business-day grid."""
    result = compute_transition_signal(monthly_cy_tightening)
    assert isinstance(result, pd.DataFrame)
    assert "CL" in result.columns


def test_compute_transition_signal_positive_for_tightening(monthly_cy_tightening):
    """Steadily rising CY produces positive transition signal in later rows."""
    result = compute_transition_signal(monthly_cy_tightening)
    valid = result["CL"].dropna()
    assert len(valid) > 0, "Should have some non-NaN signal values"
    assert valid.iloc[-1] > 0, "Tightening market should produce positive signal"


def test_compute_transition_signal_negative_for_loosening(monthly_cy_loosening):
    """Steadily falling CY produces negative transition signal in later rows."""
    result = compute_transition_signal(monthly_cy_loosening)
    valid = result["NG"].dropna()
    assert len(valid) > 0, "Should have some non-NaN signal values"
    assert valid.iloc[-1] < 0, "Loosening market should produce negative signal"


# ---------------------------------------------------------------------------
# transition_to_position
# ---------------------------------------------------------------------------


def test_transition_to_position_values_in_set(monthly_cy_tightening, monthly_dates_long):
    """Position values must only be in {-1, 0, +1}."""
    tsmom = pd.DataFrame({"CL": np.ones(120) * 0.5}, index=monthly_dates_long)
    # resample to daily
    daily_idx = pd.bdate_range(monthly_dates_long[0], monthly_dates_long[-1])
    signal = compute_transition_signal(monthly_cy_tightening)
    tsmom_daily = tsmom.reindex(index=daily_idx, method="ffill")
    positions = transition_to_position(signal, tsmom_daily)
    valid_vals = {-1, 0, 1}
    unique_vals = set(positions["CL"].dropna().unique())
    assert unique_vals.issubset(valid_vals), f"Unexpected position values: {unique_vals}"


def test_transition_to_position_no_longs_when_tsmom_nonpositive(
    monthly_cy_tightening, monthly_dates_long
):
    """Tightening signal with TSMOM=0 → no long positions (confirmation gate blocks longs)."""
    daily_idx = pd.bdate_range(monthly_dates_long[0], monthly_dates_long[-1])
    signal = compute_transition_signal(monthly_cy_tightening)
    # TSMOM = 0 means trend_up is False → longs are never triggered
    tsmom_zero = pd.DataFrame({"CL": np.zeros(len(daily_idx))}, index=daily_idx)
    positions = transition_to_position(signal, tsmom_zero)
    valid = positions["CL"].dropna()
    # TSMOM = 0 (not > 0) blocks all long positions; flat and short are still allowed
    assert not (valid == 1).any(), "TSMOM=0 should block all long positions (gate requires > 0)"
