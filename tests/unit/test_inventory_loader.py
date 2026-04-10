"""Tests for EIA inventory loader."""

import pandas as pd

from commodity_curve_factors.data.inventory_loader import (
    EIA_ROUTES,
    align_to_daily,
)


def test_eia_routes_covers_all_series() -> None:
    """Every series in inventory.yaml should have a route."""
    from commodity_curve_factors.utils.config import load_config

    inventory_config = load_config("inventory")
    configured_series = set(inventory_config["eia"]["series"].keys())
    assert configured_series == set(EIA_ROUTES.keys()), (
        f"Mismatch between inventory.yaml and EIA_ROUTES: "
        f"{configured_series} vs {set(EIA_ROUTES.keys())}"
    )


def test_align_to_daily_forward_fills_from_release_day() -> None:
    """Weekly observations should be visible only from the release day onward.

    Pre-release dates exist in the index with NaN values (not absent).
    """
    weekly = pd.DataFrame(
        {"value": [100.0, 105.0, 98.0]},
        index=pd.to_datetime(["2020-01-03", "2020-01-10", "2020-01-17"]),
    )
    daily = align_to_daily(weekly, release_day="wednesday")

    # Pre-release dates (before first release on 2020-01-08) should be NaN.
    # 2020-01-03 (Fri), 01-06 (Mon), 01-07 (Tue) are all before first release.
    assert pd.isna(daily.loc[pd.Timestamp("2020-01-03"), "value"])
    assert pd.isna(daily.loc[pd.Timestamp("2020-01-06"), "value"])
    assert pd.isna(daily.loc[pd.Timestamp("2020-01-07"), "value"])
    # 2020-01-08 (Wed) is the release day for the 2020-01-03 observation.
    assert daily.loc[pd.Timestamp("2020-01-08"), "value"] == 100.0
    # Forward fill continues through 2020-01-14 (Tue before next release).
    assert daily.loc[pd.Timestamp("2020-01-09"), "value"] == 100.0
    assert daily.loc[pd.Timestamp("2020-01-14"), "value"] == 100.0
    # 2020-01-15 (Wed) is the release day for the 2020-01-10 observation.
    assert daily.loc[pd.Timestamp("2020-01-15"), "value"] == 105.0


def test_align_to_daily_empty_input() -> None:
    """Empty weekly DataFrame should return empty daily DataFrame without crashing."""
    empty = pd.DataFrame({"value": []}, index=pd.to_datetime([]))
    result = align_to_daily(empty, release_day="wednesday")
    assert len(result) == 0


def test_load_inventory_data_empty_directory(tmp_path, monkeypatch) -> None:
    """Missing inventory directory returns empty dict, not error."""
    from commodity_curve_factors.data import inventory_loader

    monkeypatch.setattr(inventory_loader, "DATA_RAW", tmp_path)
    result = inventory_loader.load_inventory_data()
    assert result == {}


def test_load_inventory_data_roundtrip(tmp_path, monkeypatch) -> None:
    """save_inventory_data + load_inventory_data should be inverses."""
    from pandas.testing import assert_frame_equal

    from commodity_curve_factors.data import inventory_loader

    monkeypatch.setattr(inventory_loader, "DATA_RAW", tmp_path)

    df_in = pd.DataFrame(
        {"value": [432403.0, 435200.0]},
        index=pd.to_datetime(["2024-01-05", "2024-01-12"]),
    )
    inventory_loader.save_inventory_data({"crude_stocks": df_in})

    result = inventory_loader.load_inventory_data()
    assert "crude_stocks" in result
    assert_frame_equal(result["crude_stocks"], df_in)

