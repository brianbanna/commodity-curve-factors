"""Tests for WRDS Datastream futures contracts offline Parquet reader."""

import logging
from pathlib import Path

import pandas as pd
import pytest

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "wrds_sample.parquet"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _copy_fixture_to(tmp_path: Path, symbol: str = "CL") -> Path:
    """Copy the committed fixture into tmp_path/<symbol>/all_contracts.parquet."""
    dest_dir = tmp_path / symbol
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "all_contracts.parquet"
    dest.write_bytes(FIXTURE_PATH.read_bytes())
    return dest


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_contracts_reads_fixture_shape(tmp_path: Path) -> None:
    """Fixture should load with correct shape, column set, and dtype casts."""
    from commodity_curve_factors.data.wrds_loader import EXPECTED_COLUMNS, load_contracts

    _copy_fixture_to(tmp_path, "CL")
    df = load_contracts("CL", root=tmp_path)

    assert df.shape == (32500, 15)
    assert set(df.columns) == EXPECTED_COLUMNS
    assert df["futcode"].dtype == "int64", (
        f"futcode dtype should be int64, got {df['futcode'].dtype}"
    )
    assert df["trade_date"].dtype == "datetime64[ns]", (
        f"trade_date dtype should be datetime64[ns], got {df['trade_date'].dtype}"
    )
    assert df["futcode"].isna().sum() == 0, "futcode must not contain NaNs"


def test_load_contracts_captures_negative_wti_event(tmp_path: Path) -> None:
    """Settlement on 2020-04-20 must include the famous negative WTI price ~-37.63."""
    from commodity_curve_factors.data.wrds_loader import load_contracts

    _copy_fixture_to(tmp_path, "CL")
    df = load_contracts("CL", root=tmp_path)

    day = df[df["trade_date"] == pd.Timestamp("2020-04-20")]
    assert len(day) > 0, "No rows found for trade_date 2020-04-20"

    min_settlement = day["settlement"].min()
    assert abs(min_settlement - (-37.63)) < 0.1, (
        f"Expected min settlement near -37.63 on 2020-04-20, got {min_settlement}"
    )


def test_load_contracts_raises_on_missing_file(tmp_path: Path) -> None:
    """load_contracts should raise FileNotFoundError when the Parquet is absent."""
    from commodity_curve_factors.data.wrds_loader import load_contracts

    with pytest.raises(FileNotFoundError):
        load_contracts("ZZ", root=tmp_path)


def test_load_contracts_raises_on_bad_schema(tmp_path: Path) -> None:
    """load_contracts should raise ValueError listing missing columns."""
    from commodity_curve_factors.data.wrds_loader import load_contracts

    # Write a minimal two-column Parquet that is missing most expected columns.
    dest_dir = tmp_path / "XX"
    dest_dir.mkdir(parents=True, exist_ok=True)
    bad_df = pd.DataFrame({"futcode": [1], "trade_date": ["2020-01-02"]})
    bad_df.to_parquet(dest_dir / "all_contracts.parquet")

    with pytest.raises(ValueError) as exc_info:
        load_contracts("XX", root=tmp_path)

    # The error message should mention at least one missing column name.
    msg = str(exc_info.value)
    assert any(col in msg for col in ("settlement", "dsmnem", "open_price", "volume")), (
        f"ValueError message did not mention a missing column: {msg}"
    )


def test_filter_to_date_range_inclusive_bounds(tmp_path: Path) -> None:
    """filter_to_date_range should be inclusive on both ends and handle None bounds."""
    from commodity_curve_factors.data.wrds_loader import filter_to_date_range, load_contracts

    _copy_fixture_to(tmp_path, "CL")
    df = load_contracts("CL", root=tmp_path)

    # Both bounds
    filtered = filter_to_date_range(df, start="2020-06-01", end="2020-06-30")
    assert filtered["trade_date"].min() >= pd.Timestamp("2020-06-01")
    assert filtered["trade_date"].max() <= pd.Timestamp("2020-06-30")
    assert len(filtered) > 0

    # Start-only (open upper bound)
    from_june = filter_to_date_range(df, start="2020-12-01")
    assert from_june["trade_date"].min() >= pd.Timestamp("2020-12-01")
    assert from_june["trade_date"].max() == df["trade_date"].max()

    # End-only (open lower bound)
    to_jan = filter_to_date_range(df, end="2020-01-31")
    assert to_jan["trade_date"].max() <= pd.Timestamp("2020-01-31")
    assert to_jan["trade_date"].min() == df["trade_date"].min()

    # Neither bound — returns same row count
    unbounded = filter_to_date_range(df, start=None, end=None)
    assert len(unbounded) == len(df)

    # Does not mutate input
    assert len(df) == 32500


def test_get_contract_metadata_one_row_per_contract(tmp_path: Path) -> None:
    """get_contract_metadata should return one row per unique futcode."""
    from commodity_curve_factors.data.wrds_loader import get_contract_metadata, load_contracts

    _copy_fixture_to(tmp_path, "CL")
    df = load_contracts("CL", root=tmp_path)

    meta = get_contract_metadata(df)

    assert len(meta) == df["futcode"].nunique(), (
        f"Expected {df['futcode'].nunique()} rows, got {len(meta)}"
    )
    assert len(meta) == 145

    # Sorted by lasttrddate ascending
    assert list(meta["lasttrddate"]) == sorted(meta["lasttrddate"].tolist()), (
        "get_contract_metadata should be sorted by lasttrddate ascending"
    )

    # n_observations sums to original row count
    assert meta["n_observations"].sum() == len(df), (
        f"n_observations sum {meta['n_observations'].sum()} != {len(df)}"
    )

    # Required columns present
    for col in (
        "futcode",
        "dsmnem",
        "contrdate",
        "startdate",
        "lasttrddate",
        "sttlmntdate",
        "isocurrcode",
        "ldb",
        "first_trade_date",
        "last_trade_date",
        "n_observations",
    ):
        assert col in meta.columns, f"Missing column in metadata: {col}"


def test_load_all_contracts_skips_missing_without_raising(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """load_all_contracts should skip missing symbols with a warning, not raise."""
    from commodity_curve_factors.data.wrds_loader import load_all_contracts

    _copy_fixture_to(tmp_path, "CL")

    with caplog.at_level(logging.WARNING, logger="commodity_curve_factors.data.wrds_loader"):
        result = load_all_contracts(root=tmp_path, symbols=["CL", "NG"])

    assert set(result.keys()) == {"CL"}, f"Expected only CL, got {set(result.keys())}"
    assert len(result["CL"]) == 32500

    # A warning about NG should appear
    warning_text = caplog.text.lower()
    assert "ng" in warning_text or "missing" in warning_text or "not found" in warning_text, (
        f"Expected a warning about missing NG, got log: {caplog.text!r}"
    )
