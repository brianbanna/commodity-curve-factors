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


def test_usda_stock_series_has_three_crops() -> None:
    """USDA_STOCK_SERIES should cover corn, soybeans, and wheat."""
    from commodity_curve_factors.data.inventory_loader import USDA_STOCK_SERIES

    assert set(USDA_STOCK_SERIES.keys()) == {"ZC", "ZS", "ZW"}


def test_usda_stock_series_uses_exact_short_desc() -> None:
    """short_desc values should be the specific total stocks measure, not aggregated."""
    from commodity_curve_factors.data.inventory_loader import USDA_STOCK_SERIES

    assert USDA_STOCK_SERIES["ZC"]["short_desc"] == "CORN, GRAIN - STOCKS, MEASURED IN BU"
    assert USDA_STOCK_SERIES["ZS"]["short_desc"] == "SOYBEANS - STOCKS, MEASURED IN BU"
    assert USDA_STOCK_SERIES["ZW"]["short_desc"] == "WHEAT - STOCKS, MEASURED IN BU"


def test_save_inventory_data_prefix(tmp_path, monkeypatch) -> None:
    """save_inventory_data with prefix should prepend to filenames."""
    from commodity_curve_factors.data import inventory_loader

    monkeypatch.setattr(inventory_loader, "DATA_RAW", tmp_path)
    df = pd.DataFrame({"value": [1.0]}, index=pd.to_datetime(["2024-01-01"]))

    inventory_loader.save_inventory_data({"ZC": df}, prefix="usda_")
    assert (tmp_path / "inventory" / "usda_ZC.parquet").exists()

    # Without prefix, saves as plain name (backwards-compatible default).
    inventory_loader.save_inventory_data({"crude_stocks": df})
    assert (tmp_path / "inventory" / "crude_stocks.parquet").exists()


def test_usda_value_parsing_handles_comma_strings(monkeypatch, tmp_path) -> None:
    """download_usda_stocks should parse 'Value' strings with commas to floats.

    Uses monkeypatched requests.get to return a fake NASS response. Also
    verifies that rows with non-matching ``short_desc`` are filtered out
    and that ``"(D)"`` (undisclosed) values become NaN.
    """
    from commodity_curve_factors.data import inventory_loader

    fake_response_data = {
        "data": [
            {
                "commodity_desc": "CORN",
                "short_desc": "CORN, GRAIN - STOCKS, MEASURED IN BU",
                "load_time": "2024-09-30 00:00:00",
                "reference_period_desc": "FIRST OF SEP",
                "year": "2024",
                "Value": "1,760,000,000",
            },
            {
                "commodity_desc": "CORN",
                "short_desc": "CORN, GRAIN - STOCKS, MEASURED IN BU",
                "load_time": "2024-06-28 00:00:00",
                "reference_period_desc": "FIRST OF JUN",
                "year": "2024",
                "Value": "5,200,000,000",
            },
            {
                # This row should be filtered out (wrong short_desc).
                "commodity_desc": "CORN",
                "short_desc": "CORN, GRAIN, ON FARM - STOCKS, MEASURED IN BU",
                "load_time": "2024-06-28 00:00:00",
                "reference_period_desc": "FIRST OF JUN",
                "year": "2024",
                "Value": "3,000,000,000",
            },
            {
                # Undisclosed value should become NaN.
                "commodity_desc": "CORN",
                "short_desc": "CORN, GRAIN - STOCKS, MEASURED IN BU",
                "load_time": "2024-03-28 00:00:00",
                "reference_period_desc": "FIRST OF MAR",
                "year": "2024",
                "Value": "(D)",
            },
        ]
    }

    class FakeResponse:
        status_code = 200

        def json(self) -> dict:
            return fake_response_data

        def raise_for_status(self) -> None:
            pass

    monkeypatch.setattr(
        inventory_loader.requests, "get", lambda *args, **kwargs: FakeResponse()
    )
    monkeypatch.setattr(inventory_loader, "DATA_CACHE", tmp_path)

    result = inventory_loader.download_usda_stocks(
        "ZC", "2024-01-01", "2024-12-31", "FAKE_KEY"
    )

    assert result is not None
    # Uniform schema: single 'value' column, DatetimeIndex = load_time.
    assert list(result.columns) == ["value"]
    # Three unique load_times (Mar, Jun, Sep); ON FARM row is filtered out.
    assert len(result) == 3
    assert result.loc[pd.Timestamp("2024-09-30"), "value"] == 1_760_000_000
    assert result.loc[pd.Timestamp("2024-06-28"), "value"] == 5_200_000_000
    assert pd.isna(result.loc[pd.Timestamp("2024-03-28"), "value"])
    # Index is unique so the shape matches the EIA loader's output.
    assert result.index.is_unique


def test_usda_keeps_latest_period_date_per_release(monkeypatch, tmp_path) -> None:
    """When NASS publishes multiple period_dates under one load_time (revisions),
    keep only the row with the LATEST period_date (the new observation).
    """
    from commodity_curve_factors.data import inventory_loader

    # One load_time with THREE period_date values — simulates a release that
    # includes a new Sep observation plus revisions to Jun and Mar.
    fake_response_data = {
        "data": [
            {
                "commodity_desc": "CORN",
                "short_desc": "CORN, GRAIN - STOCKS, MEASURED IN BU",
                "load_time": "2024-09-30 00:00:00",
                "reference_period_desc": "FIRST OF SEP",
                "year": "2024",
                "Value": "1,760,000,000",
            },
            {
                "commodity_desc": "CORN",
                "short_desc": "CORN, GRAIN - STOCKS, MEASURED IN BU",
                "load_time": "2024-09-30 00:00:00",
                "reference_period_desc": "FIRST OF JUN",
                "year": "2024",
                "Value": "5,100,000,000",
            },
            {
                "commodity_desc": "CORN",
                "short_desc": "CORN, GRAIN - STOCKS, MEASURED IN BU",
                "load_time": "2024-09-30 00:00:00",
                "reference_period_desc": "FIRST OF MAR",
                "year": "2024",
                "Value": "8,400,000,000",
            },
        ]
    }

    class FakeResponse:
        status_code = 200

        def json(self) -> dict:
            return fake_response_data

        def raise_for_status(self) -> None:
            pass

    monkeypatch.setattr(
        inventory_loader.requests, "get", lambda *args, **kwargs: FakeResponse()
    )
    monkeypatch.setattr(inventory_loader, "DATA_CACHE", tmp_path)

    result = inventory_loader.download_usda_stocks(
        "ZC", "2024-01-01", "2024-12-31", "FAKE_KEY"
    )

    assert result is not None
    # Three rows, one load_time → kept only the LATEST period_date (Sep).
    assert len(result) == 1
    assert result.loc[pd.Timestamp("2024-09-30"), "value"] == 1_760_000_000


def test_usda_error_body_returns_none(monkeypatch, tmp_path) -> None:
    """200-OK response with 'error' key should log and return None."""
    from commodity_curve_factors.data import inventory_loader

    class FakeResponse:
        status_code = 200

        def json(self) -> dict:
            return {"error": "invalid api key"}

        def raise_for_status(self) -> None:
            pass

    monkeypatch.setattr(
        inventory_loader.requests, "get", lambda *args, **kwargs: FakeResponse()
    )
    monkeypatch.setattr(inventory_loader, "DATA_CACHE", tmp_path)

    result = inventory_loader.download_usda_stocks(
        "ZC", "2024-01-01", "2024-12-31", "FAKE_KEY"
    )

    assert result is None
