"""Tests for Nasdaq Data Link back-month contract loader."""

from pathlib import Path

import pandas as pd
import pytest

from commodity_curve_factors.data import nasdaq_loader
from commodity_curve_factors.data.nasdaq_loader import build_chris_symbol


def test_build_chris_symbol_cme() -> None:
    assert build_chris_symbol("CHRIS/CME_CL", 1) == "CHRIS/CME_CL1"
    assert build_chris_symbol("CHRIS/CME_CL", 3) == "CHRIS/CME_CL3"
    assert build_chris_symbol("CHRIS/CME_CL", 12) == "CHRIS/CME_CL12"


def test_build_chris_symbol_ice() -> None:
    assert build_chris_symbol("CHRIS/ICE_KC", 1) == "CHRIS/ICE_KC1"
    assert build_chris_symbol("CHRIS/ICE_CC", 6) == "CHRIS/ICE_CC6"


def test_build_chris_symbol_different_commodities() -> None:
    assert build_chris_symbol("CHRIS/CME_GC", 2) == "CHRIS/CME_GC2"
    assert build_chris_symbol("CHRIS/CME_C", 5) == "CHRIS/CME_C5"


def test_load_back_month_data_parses_filenames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_back_month_data should correctly parse {SYMBOL}_c{N}.parquet filenames."""
    # Point DATA_RAW at tmp_path
    monkeypatch.setattr(nasdaq_loader, "DATA_RAW", tmp_path)
    chris_dir = tmp_path / "futures" / "chris"
    chris_dir.mkdir(parents=True)

    # Create fake contract files
    df_cl1 = pd.DataFrame(
        {"Close": [70.0, 71.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"])
    )
    df_cl3 = pd.DataFrame(
        {"Close": [72.0, 73.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"])
    )
    df_gc2 = pd.DataFrame(
        {"Close": [2000.0, 2010.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"])
    )
    df_cl1.to_parquet(chris_dir / "CL_c1.parquet")
    df_cl3.to_parquet(chris_dir / "CL_c3.parquet")
    df_gc2.to_parquet(chris_dir / "GC_c2.parquet")

    result = nasdaq_loader.load_back_month_data()

    assert set(result.keys()) == {"CL", "GC"}
    assert set(result["CL"].keys()) == {1, 3}
    assert set(result["GC"].keys()) == {2}
    assert len(result["CL"][1]) == 2


def test_load_back_month_data_skips_malformed_filenames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Files without a valid _c{N} suffix should be skipped."""
    monkeypatch.setattr(nasdaq_loader, "DATA_RAW", tmp_path)
    chris_dir = tmp_path / "futures" / "chris"
    chris_dir.mkdir(parents=True)

    # Valid file
    df = pd.DataFrame({"Close": [70.0]}, index=pd.to_datetime(["2024-01-02"]))
    df.to_parquet(chris_dir / "CL_c1.parquet")
    # Malformed file (no numeric tenor)
    df.to_parquet(chris_dir / "CL_cfoo.parquet")

    result = nasdaq_loader.load_back_month_data()

    assert "CL" in result
    assert 1 in result["CL"]
    # Malformed should not appear
    assert len(result["CL"]) == 1


def test_load_back_month_data_empty_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing or empty chris/ directory should return empty dict, not crash."""
    monkeypatch.setattr(nasdaq_loader, "DATA_RAW", tmp_path)
    # Don't create the chris subdir

    result = nasdaq_loader.load_back_month_data()
    assert result == {}
