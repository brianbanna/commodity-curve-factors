"""Tests for signals/seasonal.py."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.signals.seasonal import compute_seasonal_pattern, deseasonalise


@pytest.fixture()
def daily_index_5y():
    """Five years of business days."""
    return pd.bdate_range("2015-01-01", "2019-12-31")


@pytest.fixture()
def sin_series(daily_index_5y):
    """Daily series following a sinusoidal pattern peaking around week 13."""
    weeks = daily_index_5y.isocalendar().week.values.astype(float)
    # sin peaks at pi/2, mapping week 13 (~quarter) to peak
    return pd.Series(
        np.sin(2 * np.pi * (weeks - 1) / 52),
        index=daily_index_5y,
        name="test",
    )


@pytest.fixture()
def noisy_seasonal(daily_index_5y):
    """Seasonal series with Gaussian noise."""
    rng = np.random.default_rng(7)
    weeks = daily_index_5y.isocalendar().week.values.astype(float)
    signal = np.sin(2 * np.pi * weeks / 52)
    noise = rng.standard_normal(len(daily_index_5y)) * 0.2
    return pd.Series(signal + noise, index=daily_index_5y, name="noisy")


# ---------------------------------------------------------------------------
# compute_seasonal_pattern
# ---------------------------------------------------------------------------


def test_seasonal_pattern_length(sin_series):
    """Seasonal pattern covers 50-54 week numbers (ISO weeks can reach 53)."""
    pattern = compute_seasonal_pattern(sin_series)
    assert 50 <= len(pattern) <= 54, f"Pattern length {len(pattern)} outside [50, 54]"


def test_seasonal_pattern_captures_peak(sin_series):
    """Sine wave peaking near week 13: peak week in the pattern should be 13 ± 2."""
    pattern = compute_seasonal_pattern(sin_series)
    peak_week = int(pattern.idxmax())
    assert abs(peak_week - 13) <= 2, f"Peak week {peak_week} too far from expected 13"


def test_deseasonalise_reduces_variance(noisy_seasonal):
    """Removing seasonal component from a seasonal series reduces variance."""
    pattern = compute_seasonal_pattern(noisy_seasonal)
    deseasonalised = deseasonalise(noisy_seasonal, pattern)
    original_var = noisy_seasonal.var()
    deseas_var = deseasonalised.var()
    assert deseas_var < original_var, (
        f"Deseasonalised variance {deseas_var:.4f} not less than original {original_var:.4f}"
    )


def test_deseasonalise_preserves_length(noisy_seasonal):
    """Deseasonalised series has the same length as input."""
    pattern = compute_seasonal_pattern(noisy_seasonal)
    result = deseasonalise(noisy_seasonal, pattern)
    assert len(result) == len(noisy_seasonal)


def test_no_lookahead_different_lengths(daily_index_5y):
    """Different length inputs produce different (but both valid) patterns — no lookahead."""
    rng = np.random.default_rng(99)
    full = pd.Series(rng.standard_normal(len(daily_index_5y)), index=daily_index_5y)
    half_idx = daily_index_5y[: len(daily_index_5y) // 2]
    half = full.iloc[: len(half_idx)]

    pattern_full = compute_seasonal_pattern(full)
    pattern_half = compute_seasonal_pattern(half)

    # Both should be valid (no exceptions) and have correct lengths
    assert 50 <= len(pattern_full) <= 54
    assert 50 <= len(pattern_half) <= 54

    # Different data → different patterns (no lookahead)
    common_weeks = pattern_full.index.intersection(pattern_half.index)
    assert not pattern_full[common_weeks].equals(pattern_half[common_weeks]), (
        "Full and half series should produce different seasonal patterns"
    )
