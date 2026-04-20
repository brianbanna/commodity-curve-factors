"""Unit tests for signals/spreads.py — crack spread and livestock spread signals."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.signals.spreads import (
    compute_cy_crack,
    crack_spread_signal,
    inventory_overlay,
    livestock_spread_signal,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N = 1000
RNG = np.random.default_rng(42)
DATES = pd.bdate_range("2019-01-02", periods=N)


@pytest.fixture()
def cy_df() -> pd.DataFrame:
    """Synthetic convenience yield DataFrame with CL, RB, HO columns."""
    data = {
        "CL": RNG.standard_normal(N) * 0.5,
        "RB": RNG.standard_normal(N) * 0.5,
        "HO": RNG.standard_normal(N) * 0.5,
    }
    return pd.DataFrame(data, index=DATES)


@pytest.fixture()
def cy_crack_series(cy_df: pd.DataFrame) -> pd.Series:
    return compute_cy_crack(cy_df)


@pytest.fixture()
def crack_positions(cy_crack_series: pd.Series) -> pd.DataFrame:
    return crack_spread_signal(cy_crack_series, threshold=1.5)


# ---------------------------------------------------------------------------
# 1. test_compute_cy_crack_shape
# ---------------------------------------------------------------------------


def test_compute_cy_crack_shape(cy_df: pd.DataFrame) -> None:
    """compute_cy_crack returns a Series of length N."""
    result = compute_cy_crack(cy_df)
    assert isinstance(result, pd.Series)
    assert len(result) == N


# ---------------------------------------------------------------------------
# 2. test_compute_cy_crack_formula
# ---------------------------------------------------------------------------


def test_compute_cy_crack_formula(cy_df: pd.DataFrame) -> None:
    """cy_crack exactly equals cy[RB] + cy[HO] - cy[CL]."""
    result = compute_cy_crack(cy_df)
    expected = cy_df["RB"] + cy_df["HO"] - cy_df["CL"]
    pd.testing.assert_series_equal(result, expected, check_names=False)


# ---------------------------------------------------------------------------
# 3. test_crack_spread_signal_values
# ---------------------------------------------------------------------------


def test_crack_spread_signal_values(crack_positions: pd.DataFrame) -> None:
    """crack_spread_signal returns a DataFrame with columns {CL, RB, HO}."""
    assert set(crack_positions.columns) == {"CL", "RB", "HO"}


def test_crack_spread_signal_dollar_neutral(crack_positions: pd.DataFrame) -> None:
    """Each row sums to approximately 0 (dollar-neutral positions)."""
    row_sums = crack_positions.sum(axis=1)
    # Rows with non-NaN values should be ~0; NaN rows are allowed
    valid_mask = crack_positions.notna().all(axis=1)
    assert (row_sums[valid_mask].abs() < 1e-9).all(), "Non-neutral rows detected"


# ---------------------------------------------------------------------------
# 4. test_crack_spread_signal_direction
# ---------------------------------------------------------------------------


def test_crack_spread_signal_direction(cy_crack_series: pd.Series) -> None:
    """When z-score is strongly negative, CL position is -1.0 (crude tight)."""
    # Build a series that reliably produces a very negative z-score at the end:
    # long flat history then a large negative spike
    rng2 = np.random.default_rng(7)
    base = rng2.standard_normal(900)
    spike = np.array([-15.0] * 100)
    values = np.concatenate([base, spike])
    idx = pd.bdate_range("2015-01-02", periods=len(values))
    cy_series = pd.Series(values, index=idx)

    sig = crack_spread_signal(cy_series, threshold=1.5)

    # Last row should have CL short (crude tight → long products)
    last = sig.iloc[-1]
    assert last["CL"] == pytest.approx(-1.0)
    assert last["RB"] == pytest.approx(0.5)
    assert last["HO"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 5. test_inventory_overlay_amplifies
# ---------------------------------------------------------------------------


def test_inventory_overlay_amplifies() -> None:
    """Long CL + negative inventory surprise + rising CY → amplified to 1.5x."""
    dates = pd.bdate_range("2020-01-02", periods=5)
    positions = pd.DataFrame({"CL": [1.0, 1.0, 1.0, 1.0, 1.0]}, index=dates)
    inv_surprise = pd.Series([-1.0, -1.0, -1.0, -1.0, -1.0], index=dates)
    cy_change = pd.Series([0.1, 0.1, 0.1, 0.1, 0.1], index=dates)

    result = inventory_overlay(positions, inv_surprise, cy_change, amplification=1.5)

    assert result["CL"].iloc[0] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# 6. test_inventory_overlay_no_change_when_signals_disagree
# ---------------------------------------------------------------------------


def test_inventory_overlay_no_change_when_signals_disagree() -> None:
    """Positive inventory surprise on a long position → no amplification."""
    dates = pd.bdate_range("2020-01-02", periods=3)
    positions = pd.DataFrame({"CL": [1.0, 1.0, 1.0]}, index=dates)
    inv_surprise = pd.Series([1.0, 1.0, 1.0], index=dates)  # inventory build
    cy_change = pd.Series([0.1, 0.1, 0.1], index=dates)  # cy rising (disagrees)

    result = inventory_overlay(positions, inv_surprise, cy_change, amplification=1.5)

    # No amplify condition met → positions unchanged
    pd.testing.assert_frame_equal(result, positions)


# ---------------------------------------------------------------------------
# 7. test_livestock_spread_signal_shape
# ---------------------------------------------------------------------------


def test_livestock_spread_signal_shape() -> None:
    """livestock_spread_signal returns a DataFrame with columns {LC, LH}."""
    rng3 = np.random.default_rng(42)
    n = 1500  # > 5 * 252 for deseasonalisation branch
    idx = pd.bdate_range("2015-01-02", periods=n)
    lc = pd.Series(rng3.lognormal(5.0, 0.1, n), index=idx)
    lh = pd.Series(rng3.lognormal(4.5, 0.1, n), index=idx)

    sig = livestock_spread_signal(lc, lh, seasonal_years=5, threshold=1.5)

    assert isinstance(sig, pd.DataFrame)
    assert set(sig.columns) == {"LC", "LH"}
    assert len(sig) == n


# ---------------------------------------------------------------------------
# 8. test_livestock_spread_dollar_neutral
# ---------------------------------------------------------------------------


def test_livestock_spread_dollar_neutral() -> None:
    """Each active row in livestock signal sums to 0 (dollar-neutral)."""
    rng4 = np.random.default_rng(99)
    n = 1500
    idx = pd.bdate_range("2015-01-02", periods=n)
    lc = pd.Series(rng4.lognormal(5.0, 0.1, n), index=idx)
    lh = pd.Series(rng4.lognormal(4.5, 0.1, n), index=idx)

    sig = livestock_spread_signal(lc, lh, seasonal_years=5, threshold=1.5)

    valid_mask = sig.notna().all(axis=1)
    row_sums = sig[valid_mask].sum(axis=1)
    assert (row_sums.abs() < 1e-9).all(), "Non-neutral rows detected in livestock signal"
