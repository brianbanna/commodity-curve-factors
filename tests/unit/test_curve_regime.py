"""Tests for signals/curve_regime.py."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.signals.curve_regime import (
    classify_regime,
    regime_to_position,
)


@pytest.fixture()
def monthly_cy():
    """Monthly convenience yield: CL high, NG low, GC mid."""
    dates = pd.date_range("2010-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "CL": 0.10 + rng.standard_normal(60) * 0.03,
            "NG": -0.02 + rng.standard_normal(60) * 0.03,
            "GC": 0.03 + rng.standard_normal(60) * 0.03,
        },
        index=dates,
    )


def test_classify_regime_returns_labels(monthly_cy):
    regimes = classify_regime(monthly_cy)
    assert isinstance(regimes, pd.DataFrame)
    assert set(regimes.columns) == {"CL", "NG", "GC"}
    valid_labels = {
        "crisis_backwardation",
        "mild_backwardation",
        "balanced",
        "mild_contango",
        "deep_contango",
    }
    for col in regimes.columns:
        unique_vals = set(regimes[col].dropna().unique())
        assert unique_vals.issubset(valid_labels), (
            f"{col} has unexpected labels: {unique_vals - valid_labels}"
        )


def test_classify_regime_no_lookahead(monthly_cy):
    """Adding future data should not change past regime labels."""
    regimes_short = classify_regime(monthly_cy.iloc[:30])
    regimes_full = classify_regime(monthly_cy)
    pd.testing.assert_frame_equal(
        regimes_short,
        regimes_full.iloc[:30],
    )


def test_classify_regime_custom_thresholds(monthly_cy):
    regimes = classify_regime(monthly_cy, thresholds=[20, 40, 60, 80])
    assert isinstance(regimes, pd.DataFrame)


def test_regime_to_position_values(monthly_cy):
    regimes = classify_regime(monthly_cy)
    positions = regime_to_position(regimes)
    assert isinstance(positions, pd.DataFrame)
    assert set(positions.columns) == {"CL", "NG", "GC"}
    valid_positions = {-0.5, 0.0, 0.5, 1.0}
    for col in positions.columns:
        unique_vals = set(positions[col].dropna().unique())
        assert unique_vals.issubset(valid_positions), (
            f"{col} has unexpected positions: {unique_vals}"
        )


def test_regime_to_position_custom_map(monthly_cy):
    regimes = classify_regime(monthly_cy)
    custom_map = {
        "crisis_backwardation": 1.0,
        "mild_backwardation": 0.5,
        "balanced": 0.0,
        "mild_contango": -0.25,
        "deep_contango": -1.0,
    }
    positions = regime_to_position(regimes, position_map=custom_map)
    valid_positions = set(custom_map.values())
    for col in positions.columns:
        unique_vals = set(positions[col].dropna().unique())
        assert unique_vals.issubset(valid_positions)
