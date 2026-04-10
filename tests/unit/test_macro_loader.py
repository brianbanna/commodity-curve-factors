"""Tests for FRED + VIX + benchmarks macro data loader."""

from commodity_curve_factors.data.macro_loader import MACRO_SERIES


def test_macro_series_includes_required_keys() -> None:
    expected = {"usd_index", "dgs10", "t5yie", "dgs3mo"}
    assert expected.issubset(set(MACRO_SERIES.keys()))


def test_macro_series_fred_ids() -> None:
    assert MACRO_SERIES["usd_index"] == "DTWEXBGS"
    assert MACRO_SERIES["dgs10"] == "DGS10"
    assert MACRO_SERIES["t5yie"] == "T5YIE"
    assert MACRO_SERIES["dgs3mo"] == "DGS3MO"


def test_load_macro_data_empty_directory(tmp_path, monkeypatch) -> None:
    """Missing macro directory should return empty dict, not crash."""
    from commodity_curve_factors.data import macro_loader
    monkeypatch.setattr(macro_loader, "DATA_RAW", tmp_path)
    result = macro_loader.load_macro_data()
    assert result == {}


def test_load_macro_data_roundtrip(tmp_path, monkeypatch) -> None:
    """Saved series should be loadable by name."""
    import pandas as pd
    from commodity_curve_factors.data import macro_loader
    monkeypatch.setattr(macro_loader, "DATA_RAW", tmp_path)
    macro_dir = tmp_path / "macro"
    macro_dir.mkdir(parents=True)
    df = pd.DataFrame({"value": [1.0, 2.0]}, index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    df.to_parquet(macro_dir / "dgs10.parquet")
    result = macro_loader.load_macro_data()
    assert "dgs10" in result
    assert len(result["dgs10"]) == 2
