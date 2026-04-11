"""Tests for futures roll calendar utilities."""

import datetime

import pandas as pd


def _make_contracts(specs: list[dict]) -> pd.DataFrame:
    """Build a minimal contracts DataFrame from a list of spec dicts.

    Each spec dict may contain:
        startdate, lasttrddate, trade_date, settlement, futcode, dsmnem
    """
    rows = []
    for i, spec in enumerate(specs):
        rows.append(
            {
                "futcode": spec.get("futcode", i + 1),
                "dsmnem": spec.get("dsmnem", f"NCL{i + 1:04d}"),
                "contrdate": spec.get("contrdate", "0120"),
                "startdate": spec.get("startdate", datetime.date(2019, 1, 1)),
                "lasttrddate": spec.get("lasttrddate"),
                "sttlmntdate": spec.get("sttlmntdate", spec.get("lasttrddate")),
                "isocurrcode": "USD",
                "ldb": "X",
                "trade_date": pd.Timestamp(spec.get("trade_date", "2020-06-01")),
                "open_price": spec.get("settlement", 50.0),
                "high_price": spec.get("settlement", 50.0),
                "low_price": spec.get("settlement", 50.0),
                "settlement": spec.get("settlement", 50.0),
                "volume": 1000.0,
                "openinterest": 5000.0,
            }
        )
    return pd.DataFrame(rows)


class TestActiveContractsOnDay:
    def test_filters_by_roll_offset(self) -> None:
        """Two of three contracts are within the active window; third is past roll date."""
        from commodity_curve_factors.curves.roll_calendar import active_contracts_on_day

        trade_date = pd.Timestamp("2020-06-01")
        # Contract 1: expires 2020-06-20, roll deadline = 2020-06-15 → active (6/1 ≤ 6/15)
        # Contract 2: expires 2020-07-20, roll deadline = 2020-07-15 → active
        # Contract 3: expires 2020-06-04, roll deadline = 2020-05-30 → NOT active (6/1 > 5/30)
        contracts = _make_contracts(
            [
                {"futcode": 1, "lasttrddate": datetime.date(2020, 6, 20), "settlement": 50.0},
                {"futcode": 2, "lasttrddate": datetime.date(2020, 7, 20), "settlement": 52.0},
                {"futcode": 3, "lasttrddate": datetime.date(2020, 6, 4), "settlement": 55.0},
            ]
        )

        result = active_contracts_on_day(contracts, trade_date, roll_days_before_expiry=5)

        assert set(result["futcode"].tolist()) == {1, 2}, (
            f"Expected futcodes {{1, 2}}, got {set(result['futcode'].tolist())}"
        )

    def test_excludes_nan_settlement(self) -> None:
        """Rows with NaN settlement are excluded even if within the active window."""
        from commodity_curve_factors.curves.roll_calendar import active_contracts_on_day

        trade_date = pd.Timestamp("2020-06-01")
        contracts = _make_contracts(
            [
                {"futcode": 1, "lasttrddate": datetime.date(2020, 6, 20), "settlement": 50.0},
                {
                    "futcode": 2,
                    "lasttrddate": datetime.date(2020, 7, 20),
                    "settlement": float("nan"),
                },
            ]
        )

        result = active_contracts_on_day(contracts, trade_date, roll_days_before_expiry=5)

        assert set(result["futcode"].tolist()) == {1}, (
            f"Expected only futcode 1 (settlement NaN for futcode 2), got {result['futcode'].tolist()}"
        )

    def test_no_rows_for_trade_date_returns_empty(self) -> None:
        """No rows matching trade_date → empty DataFrame."""
        from commodity_curve_factors.curves.roll_calendar import active_contracts_on_day

        contracts = _make_contracts(
            [
                {
                    "futcode": 1,
                    "lasttrddate": datetime.date(2020, 6, 20),
                    "trade_date": "2020-05-01",
                },
            ]
        )

        result = active_contracts_on_day(
            contracts, pd.Timestamp("2020-06-01"), roll_days_before_expiry=5
        )

        assert result.empty


class TestGetFrontContract:
    def test_returns_nearest_expiry(self) -> None:
        """Of three active contracts, the one with the smallest lasttrddate is front."""
        from commodity_curve_factors.curves.roll_calendar import get_front_contract

        trade_date = pd.Timestamp("2020-06-01")
        contracts = _make_contracts(
            [
                {"futcode": 10, "lasttrddate": datetime.date(2020, 8, 20), "settlement": 52.0},
                {"futcode": 11, "lasttrddate": datetime.date(2020, 6, 20), "settlement": 50.0},
                {"futcode": 12, "lasttrddate": datetime.date(2020, 7, 20), "settlement": 51.0},
            ]
        )

        result = get_front_contract(contracts, trade_date, roll_days_before_expiry=5)

        assert result is not None
        assert int(result["futcode"]) == 11, (
            f"Expected front = futcode 11 (earliest expiry), got {result['futcode']}"
        )

    def test_returns_none_when_all_rolled(self) -> None:
        """All contracts past roll date → get_front_contract returns None."""
        from commodity_curve_factors.curves.roll_calendar import get_front_contract

        # trade_date is 2020-06-10; lasttrddate=2020-06-12; roll_days=5 → deadline 2020-06-07
        trade_date = pd.Timestamp("2020-06-10")
        contracts = _make_contracts(
            [
                {"futcode": 1, "lasttrddate": datetime.date(2020, 6, 12), "settlement": 50.0},
                {"futcode": 2, "lasttrddate": datetime.date(2020, 6, 13), "settlement": 51.0},
            ]
        )

        result = get_front_contract(contracts, trade_date, roll_days_before_expiry=5)

        assert result is None, f"Expected None when all contracts are rolled, got {result}"


class TestBuildRollSchedule:
    def test_covers_all_trade_dates(self) -> None:
        """Row count matches the number of unique trade_dates in input."""
        from commodity_curve_factors.curves.roll_calendar import build_roll_schedule

        dates = pd.date_range("2020-01-02", "2020-01-31", freq="B")

        rows = []
        for d in dates:
            rows.append(
                {
                    "futcode": 1,
                    "dsmnem": "NCL0320",
                    "contrdate": "0320",
                    "startdate": datetime.date(2019, 1, 1),
                    "lasttrddate": datetime.date(2020, 3, 19),
                    "sttlmntdate": datetime.date(2020, 3, 19),
                    "isocurrcode": "USD",
                    "ldb": "X",
                    "trade_date": d,
                    "open_price": 50.0,
                    "high_price": 51.0,
                    "low_price": 49.0,
                    "settlement": 50.0,
                    "volume": 1000.0,
                    "openinterest": 5000.0,
                }
            )

        contracts = pd.DataFrame(rows)
        result = build_roll_schedule(contracts, roll_days_before_expiry=5)

        unique_dates = contracts["trade_date"].nunique()
        assert len(result) == unique_dates, f"Expected {unique_dates} rows, got {len(result)}"

    def test_sorted_by_trade_date(self) -> None:
        """Result is sorted by trade_date ascending."""
        from commodity_curve_factors.curves.roll_calendar import build_roll_schedule

        dates = [
            pd.Timestamp("2020-03-05"),
            pd.Timestamp("2020-03-03"),
            pd.Timestamp("2020-03-04"),
        ]
        rows = []
        for d in dates:
            rows.append(
                {
                    "futcode": 1,
                    "dsmnem": "NCL0620",
                    "contrdate": "0620",
                    "startdate": datetime.date(2019, 1, 1),
                    "lasttrddate": datetime.date(2020, 6, 22),
                    "sttlmntdate": datetime.date(2020, 6, 22),
                    "isocurrcode": "USD",
                    "ldb": "X",
                    "trade_date": d,
                    "open_price": 50.0,
                    "high_price": 51.0,
                    "low_price": 49.0,
                    "settlement": 50.0,
                    "volume": 1000.0,
                    "openinterest": 5000.0,
                }
            )
        contracts = pd.DataFrame(rows)
        result = build_roll_schedule(contracts, roll_days_before_expiry=5)

        assert result["trade_date"].is_monotonic_increasing

    def test_front_columns_present(self) -> None:
        """Result must have the required columns."""
        from commodity_curve_factors.curves.roll_calendar import build_roll_schedule

        rows = [
            {
                "futcode": 1,
                "dsmnem": "NCL0620",
                "contrdate": "0620",
                "startdate": datetime.date(2019, 1, 1),
                "lasttrddate": datetime.date(2020, 6, 22),
                "sttlmntdate": datetime.date(2020, 6, 22),
                "isocurrcode": "USD",
                "ldb": "X",
                "trade_date": pd.Timestamp("2020-03-01"),
                "open_price": 50.0,
                "high_price": 51.0,
                "low_price": 49.0,
                "settlement": 50.0,
                "volume": 1000.0,
                "openinterest": 5000.0,
            }
        ]
        contracts = pd.DataFrame(rows)
        result = build_roll_schedule(contracts, roll_days_before_expiry=5)

        for col in ("trade_date", "front_futcode", "front_dsmnem", "days_to_expiry", "settlement"):
            assert col in result.columns, f"Missing column: {col}"
