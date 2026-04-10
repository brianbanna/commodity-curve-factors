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
    """save_macro_data + load_macro_data should be inverses."""
    import pandas as pd
    from pandas.testing import assert_frame_equal

    from commodity_curve_factors.data import macro_loader

    monkeypatch.setattr(macro_loader, "DATA_RAW", tmp_path)

    df_in = pd.DataFrame(
        {"value": [4.04, 4.12, 4.08]},
        index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
    )
    macro_loader.save_macro_data({"dgs10": df_in})

    result = macro_loader.load_macro_data()

    assert "dgs10" in result
    assert len(result["dgs10"]) == 3
    assert_frame_equal(result["dgs10"], df_in)


def test_save_macro_data_creates_directory_and_saves_all(tmp_path, monkeypatch) -> None:
    """save_macro_data should create the directory and save every series."""
    import pandas as pd

    from commodity_curve_factors.data import macro_loader

    monkeypatch.setattr(macro_loader, "DATA_RAW", tmp_path)

    df_dgs10 = pd.DataFrame({"value": [4.0]}, index=pd.to_datetime(["2020-01-02"]))
    df_vix = pd.DataFrame(
        {"Open": [15.0], "High": [16.0], "Low": [14.5], "Close": [15.5], "Volume": [0]},
        index=pd.to_datetime(["2020-01-02"]),
    )

    macro_loader.save_macro_data({"dgs10": df_dgs10, "vix": df_vix})

    assert (tmp_path / "macro" / "dgs10.parquet").exists()
    assert (tmp_path / "macro" / "vix.parquet").exists()


def test_download_benchmarks_keys_parameter(monkeypatch) -> None:
    """download_benchmarks with keys=['spy'] should only fetch spy, not agg."""
    import tempfile
    from pathlib import Path

    import pandas as pd

    from commodity_curve_factors.data import macro_loader

    calls: list[str] = []

    def fake_yf_download(ticker, **kwargs):
        calls.append(ticker)
        return pd.DataFrame(
            {
                ("Open", ticker): [1.0],
                ("High", ticker): [1.0],
                ("Low", ticker): [1.0],
                ("Close", ticker): [1.0],
                ("Adj Close", ticker): [1.0],
                ("Volume", ticker): [0],
            },
            index=pd.to_datetime(["2020-01-02"]),
        )

    monkeypatch.setattr(macro_loader.yf, "download", fake_yf_download)

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(macro_loader, "DATA_CACHE", Path(tmp))
        result = macro_loader.download_benchmarks("2020-01-01", "2020-01-10", keys=["spy"])

    assert set(result.keys()) == {"spy"}
    assert "^GSPC" in calls
    assert "AGG" not in calls
