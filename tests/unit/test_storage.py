"""Tests for storage helpers: save_parquet, load_parquet, build_catalog."""

import logging
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# save_parquet
# ---------------------------------------------------------------------------


def test_save_parquet_creates_file(tmp_path: Path) -> None:
    """save_parquet should create a Parquet file at the given path."""
    from commodity_curve_factors.data.storage import save_parquet

    df = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
    )
    out = tmp_path / "test.parquet"
    save_parquet(df, out)

    assert out.exists()


def test_save_parquet_sorts_datetime_index(tmp_path: Path) -> None:
    """save_parquet should sort a DatetimeIndex ascending before writing."""
    from commodity_curve_factors.data.storage import load_parquet, save_parquet

    # Deliberately out-of-order dates
    df = pd.DataFrame(
        {"value": [3.0, 1.0, 2.0]},
        index=pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"]),
    )
    out = tmp_path / "sorted.parquet"
    save_parquet(df, out)

    loaded = load_parquet(out)
    assert loaded.index.is_monotonic_increasing


def test_save_parquet_creates_parent_directories(tmp_path: Path) -> None:
    """save_parquet should create missing intermediate directories."""
    from commodity_curve_factors.data.storage import save_parquet

    df = pd.DataFrame({"a": [1, 2]})
    deep_path = tmp_path / "a" / "b" / "c.parquet"

    assert not (tmp_path / "a").exists()
    save_parquet(df, deep_path)
    assert deep_path.exists()


def test_save_parquet_non_datetime_index(tmp_path: Path) -> None:
    """save_parquet should save a DataFrame with RangeIndex without modification."""
    from commodity_curve_factors.data.storage import load_parquet, save_parquet

    df = pd.DataFrame({"x": [10, 20, 30], "y": ["a", "b", "c"]})
    out = tmp_path / "range_idx.parquet"
    save_parquet(df, out)

    loaded = load_parquet(out)
    # Values should be identical
    assert list(loaded["x"]) == [10, 20, 30]
    assert list(loaded["y"]) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# load_parquet
# ---------------------------------------------------------------------------


def test_load_parquet_raises_on_missing(tmp_path: Path) -> None:
    """load_parquet should raise FileNotFoundError for a non-existent file."""
    from commodity_curve_factors.data.storage import load_parquet

    with pytest.raises(FileNotFoundError):
        load_parquet(tmp_path / "nonexistent.parquet")


def test_load_parquet_warns_on_unsorted_datetime(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """load_parquet should log a WARNING if the DatetimeIndex is not sorted."""
    from commodity_curve_factors.data.storage import load_parquet

    # Write unsorted Parquet directly (bypassing save_parquet) to simulate
    # a file that arrived with an unsorted index.
    df = pd.DataFrame(
        {"value": [3.0, 1.0, 2.0]},
        index=pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"]),
    )
    out = tmp_path / "unsorted.parquet"
    df.to_parquet(out)  # bypass save_parquet intentionally

    with caplog.at_level(logging.WARNING, logger="commodity_curve_factors.data.storage"):
        loaded = load_parquet(out)

    # File loads successfully
    assert len(loaded) == 3

    # A WARNING was emitted
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) >= 1, f"Expected at least one WARNING, got: {caplog.text!r}"

    # Index is NOT re-sorted (load_parquet warns but doesn't fix)
    assert not loaded.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# build_catalog
# ---------------------------------------------------------------------------


def test_build_catalog_scans_directory(tmp_path: Path) -> None:
    """build_catalog should return one row per Parquet file with correct columns."""
    from commodity_curve_factors.data.storage import build_catalog, save_parquet

    df1 = pd.DataFrame(
        {"close": [100.0, 101.0]},
        index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
    )
    df2 = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"]),
    )
    save_parquet(df1, tmp_path / "prices.parquet")
    save_parquet(df2, tmp_path / "macro.parquet")

    cat = build_catalog(tmp_path)

    assert len(cat) == 2

    expected_cols = {"path", "rows", "start_date", "end_date", "columns", "size_mb"}
    assert expected_cols.issubset(set(cat.columns)), (
        f"Missing columns: {expected_cols - set(cat.columns)}"
    )

    # paths should be relative strings
    for p in cat["path"]:
        assert not Path(p).is_absolute(), f"Expected relative path, got absolute: {p}"

    # size_mb is a non-negative float (small test files may round to 0.00)
    assert (
        cat["size_mb"].dtype == float or cat["size_mb"].apply(lambda x: isinstance(x, float)).all()
    )
    assert (cat["size_mb"] >= 0).all()


def test_build_catalog_handles_corrupt_file(tmp_path: Path) -> None:
    """build_catalog should include corrupt files with rows=None and an error column."""
    from commodity_curve_factors.data.storage import build_catalog

    # Write plain text with a .parquet extension — not a valid Parquet file
    bad = tmp_path / "bad.parquet"
    bad.write_text("this is not parquet data")

    cat = build_catalog(tmp_path)

    assert len(cat) == 1
    row = cat.iloc[0]
    assert row["rows"] is None or (isinstance(row["rows"], float) and pd.isna(row["rows"]))
    assert "error" in cat.columns
    assert isinstance(row["error"], str) and len(row["error"]) > 0


def test_build_catalog_empty_directory(tmp_path: Path) -> None:
    """build_catalog on an empty directory should return an empty DataFrame with correct columns."""
    from commodity_curve_factors.data.storage import build_catalog

    cat = build_catalog(tmp_path)

    assert cat.empty
    expected_cols = {"path", "rows", "start_date", "end_date", "columns", "size_mb"}
    assert expected_cols.issubset(set(cat.columns)), (
        f"Missing columns: {expected_cols - set(cat.columns)}"
    )
