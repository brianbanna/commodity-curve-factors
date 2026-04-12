"""Tests for the EIA inventory surprise factor module."""

import numpy as np
import pandas as pd

from commodity_curve_factors.factors.inventory import (
    compute_all_inventory_surprises,
    compute_inventory_surprise,
    compute_seasonal_expectation,
)


def _make_weekly_series(
    n_years: int = 8,
    start: str = "2010-01-01",
    seed: int = 42,
) -> pd.Series:
    """Build a synthetic weekly inventory-level series."""
    dates = pd.date_range(start, periods=n_years * 52, freq="W-FRI")
    rng = np.random.default_rng(seed)
    levels = 300_000 + np.cumsum(rng.normal(0, 1000, len(dates)))
    return pd.Series(levels, index=dates, name="value")


def _make_inventory_dict(series: pd.Series, key: str = "crude_stocks") -> dict[str, pd.DataFrame]:
    """Wrap a Series in a one-column DataFrame as load_inventory_data returns."""
    return {key: pd.DataFrame({"value": series})}


# ---------------------------------------------------------------------------
# compute_seasonal_expectation
# ---------------------------------------------------------------------------


def test_seasonal_expectation_uses_past_only() -> None:
    """Expectation at week W of year Y must use only years < Y (no lookahead)."""
    # Build a 7-year series; inject a big spike in year 6, week 10.
    dates = pd.date_range("2010-01-01", periods=7 * 52, freq="W-FRI")
    base = pd.Series(100_000.0, index=dates)

    # Find an index in year 2015 around week 10 and inject a large spike
    iso = base.index.isocalendar()
    spike_mask = (iso.year == 2015) & (iso.week == 10)
    spike_idx = base.index[spike_mask]
    assert len(spike_idx) > 0, "No spike index found; adjust test dates"

    modified = base.copy()
    modified.loc[spike_idx[0]] = 200_000.0  # big spike

    expected_modified = compute_seasonal_expectation(modified, years=5)
    expected_base = compute_seasonal_expectation(base, years=5)

    # The spike occurs in 2015-W10. The expectation at 2016-W10 uses
    # 2011-2015 (5 years) and should differ between modified and base.
    future_mask = (iso.year == 2016) & (iso.week == 10)
    future_idx = base.index[future_mask]
    assert len(future_idx) > 0, "No future index found; adjust test dates"

    # Modified series injects a huge change at the spike position.
    # The expectation after the spike year should be affected.
    val_base = expected_base.loc[future_idx[0]]
    val_modified = expected_modified.loc[future_idx[0]]
    assert val_base != val_modified, (
        "Seasonal expectation did not change after injecting spike in prior year — "
        "possible lookahead or the spike was not in the lookback window"
    )

    # The expectation at or BEFORE the spike date should not differ.
    pre_spike_mask = base.index < spike_idx[0]
    pd.testing.assert_series_equal(
        expected_base[pre_spike_mask],
        expected_modified[pre_spike_mask],
    )


def test_seasonal_expectation_returns_nan_for_insufficient_history() -> None:
    """First year of data has no prior history → expectation is NaN."""
    dates = pd.date_range("2015-01-01", periods=52, freq="W-FRI")
    series = pd.Series(np.linspace(100, 110, 52), index=dates)
    expected = compute_seasonal_expectation(series, years=5)
    assert expected.isna().all(), "First year (no prior history) should all be NaN"


# ---------------------------------------------------------------------------
# compute_inventory_surprise
# ---------------------------------------------------------------------------


def test_inventory_surprise_shape() -> None:
    """Output has the same index length as the input series."""
    series = _make_weekly_series(n_years=6)
    surprise = compute_inventory_surprise(series, years=5)
    assert len(surprise) == len(series)
    assert isinstance(surprise, pd.Series)


def test_surprise_positive_when_build_exceeds_expectation() -> None:
    """A larger-than-expected inventory BUILD should produce a positive surprise z-score.

    Sign convention: surprise = (actual_change - expected_change).
    When actual_change >> expected_change (unexpected build), raw_surprise > 0 → z > 0.
    When actual_change << expected_change (unexpected draw), raw_surprise < 0 → z < 0.
    """
    # Build 7 years of flat-ish weekly data so the seasonal expectation is near 0.
    dates = pd.date_range("2010-01-01", periods=7 * 52, freq="W-FRI")
    rng = np.random.default_rng(0)
    levels = 300_000 + np.cumsum(rng.normal(0, 50, len(dates)))
    series = pd.Series(levels, index=dates)

    # Inject a huge build in the LAST observation (well after warm-up).
    series_with_build = series.copy()
    series_with_build.iloc[-1] = series.iloc[-2] + 50_000  # big build

    surprise = compute_inventory_surprise(series_with_build, years=5)
    last_val = surprise.iloc[-1]

    # A huge build relative to history → large positive z-score
    assert last_val > 2.0, f"Expected large positive surprise for a big build, got {last_val:.3f}"

    # Conversely, a big draw should produce a negative z-score
    series_with_draw = series.copy()
    series_with_draw.iloc[-1] = series.iloc[-2] - 50_000  # big draw
    surprise_draw = compute_inventory_surprise(series_with_draw, years=5)
    assert surprise_draw.iloc[-1] < -2.0, (
        f"Expected large negative surprise for a big draw, got {surprise_draw.iloc[-1]:.3f}"
    )


# ---------------------------------------------------------------------------
# compute_all_inventory_surprises
# ---------------------------------------------------------------------------


def test_inventory_surprise_output_shape() -> None:
    """Output DataFrame has correct shape: all_commodities as columns."""
    all_comms = ["CL", "NG", "GC", "ZC"]
    cmap = {"CL": "crude_stocks"}
    inv_data = _make_inventory_dict(_make_weekly_series(n_years=6), "crude_stocks")

    result = compute_all_inventory_surprises(inv_data, cmap, all_comms, years=5)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == set(all_comms)


def test_non_energy_commodities_are_nan() -> None:
    """Commodities not in commodity_map must have all-NaN columns."""
    all_comms = ["CL", "GC", "SI", "HG", "ZC", "ZS", "ZW", "KC", "SB", "CC"]
    cmap = {"CL": "crude_stocks"}
    inv_data = _make_inventory_dict(_make_weekly_series(n_years=6), "crude_stocks")

    result = compute_all_inventory_surprises(inv_data, cmap, all_comms, years=5)

    non_energy = [c for c in all_comms if c != "CL"]
    for sym in non_energy:
        assert result[sym].isna().all(), f"Column {sym} should be all NaN"


def test_weekly_to_daily_forward_fill() -> None:
    """Output index must be daily (business-day) frequency, not weekly."""
    all_comms = ["CL"]
    cmap = {"CL": "crude_stocks"}
    weekly = _make_weekly_series(n_years=3)
    inv_data = _make_inventory_dict(weekly, "crude_stocks")

    result = compute_all_inventory_surprises(inv_data, cmap, all_comms, years=2)

    # Business-day index should have roughly 5x more rows than the weekly input
    n_weeks = len(weekly)
    n_days = len(result)
    assert n_days > n_weeks * 3, (
        f"Expected many more daily rows than weekly rows; got {n_days} vs {n_weeks}"
    )

    # Consecutive business-day gaps should be 1 or 3 days (Mon gaps)
    if len(result) > 1:
        gaps = pd.Series(result.index).diff().dropna()
        assert gaps.max() <= pd.Timedelta(days=3), "Daily index has unexpected large gap"


def test_missing_series_key_returns_nan() -> None:
    """Commodity whose series name is absent from inventory_data gets NaN."""
    all_comms = ["CL", "NG"]
    cmap = {"CL": "crude_stocks", "NG": "natural_gas_storage"}
    # Only provide crude_stocks; natural_gas_storage is missing
    inv_data = _make_inventory_dict(_make_weekly_series(n_years=6), "crude_stocks")

    result = compute_all_inventory_surprises(inv_data, cmap, all_comms, years=5)

    assert result["NG"].isna().all(), "NG column should be NaN when series is absent"
    assert result["CL"].notna().any(), "CL column should have some non-NaN values"
