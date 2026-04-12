"""Tests for the curve builder using the committed WRDS fixture.

The fixture ``tests/fixtures/wrds_sample.parquet`` contains CL (WTI) contracts
for the full year 2020 (~145 unique contracts, 253 trading days).
"""

from pathlib import Path

import pandas as pd
import pytest

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "wrds_sample.parquet"


def _load_fixture() -> pd.DataFrame:
    """Load and normalise the CL 2020 fixture."""
    from commodity_curve_factors.data.wrds_loader import _normalize_dtypes

    df = pd.read_parquet(FIXTURE_PATH)
    return _normalize_dtypes(df)


def _curve_config() -> dict:
    """Load curve config from configs/curve.yaml."""
    from commodity_curve_factors.utils.config import load_config

    return load_config("curve")


class TestBuildCurveWti2020:
    def test_shape(self) -> None:
        """build_curve on CL 2020 fixture: ~253 rows, correct tenor columns."""
        from commodity_curve_factors.curves.builder import build_curve

        df = _load_fixture()
        cfg = _curve_config()
        curve = build_curve(df, "CL", cfg)

        standard_tenors = cfg["standard_tenors"]
        expected_cols = [f"F{m}M" for m in standard_tenors]

        assert list(curve.columns) == expected_cols, (
            f"Expected columns {expected_cols}, got {list(curve.columns)}"
        )
        assert len(curve) == df["trade_date"].nunique(), (
            f"Expected {df['trade_date'].nunique()} rows (one per trading day), got {len(curve)}"
        )
        assert curve.index.is_monotonic_increasing, "Curve index must be sorted ascending"
        assert curve.index.name == "trade_date"

    def test_2020_04_20_not_all_nan(self) -> None:
        """On 2020-04-20, back-month contracts are positive → curve has finite values.

        The May-2020 (NCL0520) contract at -$37.63 was already past its roll date
        (roll_deadline = 2020-04-16 with roll_days=5) so it is excluded by
        active_contracts_on_day before the interpolation step.  The remaining
        129 active contracts are all positive, so log-linear interpolation
        succeeds and produces finite F6M and F12M values.
        """
        from commodity_curve_factors.curves.builder import build_curve

        df = _load_fixture()
        cfg = _curve_config()
        curve = build_curve(df, "CL", cfg)

        row_2020_04_20 = curve.loc[pd.Timestamp("2020-04-20")]

        assert not row_2020_04_20.isna().all(), (
            "Expected at least some finite tenor values on 2020-04-20, but all are NaN"
        )
        assert pd.notna(row_2020_04_20.get("F6M")), (
            f"Expected finite F6M on 2020-04-20, got {row_2020_04_20.get('F6M')}"
        )
        assert pd.notna(row_2020_04_20.get("F12M")), (
            f"Expected finite F12M on 2020-04-20, got {row_2020_04_20.get('F12M')}"
        )

    def test_2020_backwardation_sign(self) -> None:
        """In June 2020 WTI was in contango (front < back). Verify F1M < F12M on a typical day."""
        from commodity_curve_factors.curves.builder import build_curve

        df = _load_fixture()
        cfg = _curve_config()
        curve = build_curve(df, "CL", cfg)

        # Pick a mid-year date where contango structure was clear
        sample_date = pd.Timestamp("2020-06-01")
        if sample_date not in curve.index:
            # find nearest available date
            idx = curve.index.get_indexer([sample_date], method="nearest")[0]
            sample_date = curve.index[idx]

        row = curve.loc[sample_date]
        if pd.notna(row.get("F1M")) and pd.notna(row.get("F12M")):
            # In June 2020 WTI was in contango: F1M < F12M
            assert row["F1M"] < row["F12M"], (
                f"Expected contango (F1M < F12M) in June 2020, got "
                f"F1M={row['F1M']:.2f}, F12M={row['F12M']:.2f}"
            )


class TestBuildAllCurves:
    def test_iterates_dict(self) -> None:
        """build_all_curves maps over a dict with 2 entries and returns both keys."""
        from commodity_curve_factors.curves.builder import build_all_curves

        df = _load_fixture()
        cfg = _curve_config()

        # Use two keys: "CL" and "CL_copy" (trimmed to first 5000 rows)
        contracts_dict = {
            "CL": df,
            "CL2": df.head(5000).copy(),
        }
        result = build_all_curves(contracts_dict, cfg)

        assert set(result.keys()) == {"CL", "CL2"}, (
            f"Expected keys {{'CL', 'CL2'}}, got {set(result.keys())}"
        )
        for key, curve in result.items():
            assert isinstance(curve, pd.DataFrame), f"Expected DataFrame for {key}"
            assert not curve.empty, f"Expected non-empty curve for {key}"


class TestSaveLoadCurves:
    def test_roundtrip(self, tmp_path: Path) -> None:
        """save_curves then load_curves returns identical DataFrames."""
        from commodity_curve_factors.curves.builder import build_curve, load_curves, save_curves

        df = _load_fixture()
        cfg = _curve_config()

        # Build a curve for one symbol
        curve = build_curve(df, "CL", cfg)
        curves = {"CL": curve}

        save_curves(curves, out_dir=tmp_path)

        loaded = load_curves(in_dir=tmp_path, symbols=["CL"])

        assert "CL" in loaded, "Expected CL in loaded curves"
        pd.testing.assert_frame_equal(
            curve,
            loaded["CL"],
            check_names=True,
        )

    def test_missing_file_skipped_not_raised(
        self, tmp_path: Path, caplog: "pytest.LogCaptureFixture"
    ) -> None:
        """load_curves logs a warning for missing symbols; does not raise."""
        import logging

        from commodity_curve_factors.curves.builder import load_curves

        with caplog.at_level(logging.WARNING):
            result = load_curves(in_dir=tmp_path, symbols=["ZZ"])

        assert "ZZ" not in result
        assert any("ZZ" in r.message for r in caplog.records), (
            f"Expected a warning about ZZ, got: {caplog.text!r}"
        )
