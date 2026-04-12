"""Unit tests for the CFTC COT loader.

All tests run offline against ``tests/fixtures/cftc_sample.csv``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.data import cftc_loader
from commodity_curve_factors.data.cftc_loader import (
    compute_net_speculative,
    download_cot_history,
    lag_to_release_date,
    load_cot_data,
    parse_cot_csv,
    save_cot_data,
)

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "cftc_sample.csv"

# Real contract market codes from the 2023 disaggregated futures-only file.
CL_CODE = "067651"
GC_CODE = "088691"


def _load_fixture() -> pd.DataFrame:
    """Load the hand-crafted COT fixture into a raw DataFrame."""
    return pd.read_csv(FIXTURE_PATH)


def test_parse_cot_csv_extracts_universe_commodities() -> None:
    """parse_cot_csv returns exactly the rows for the requested symbols."""
    raw = _load_fixture()
    commodity_codes = {"CL": CL_CODE, "GC": GC_CODE}

    parsed = parse_cot_csv(raw, commodity_codes)

    # 4 weeks x 2 commodities
    assert len(parsed) == 8
    assert set(parsed["commodity"].unique()) == {"CL", "GC"}
    assert list(parsed.columns) == [
        "commodity",
        "report_date",
        "mm_long",
        "mm_short",
        "mm_net",
        "open_interest",
    ]

    # mm_net == mm_long - mm_short on every non-NaN row.
    non_nan = parsed.dropna(subset=["mm_long", "mm_short"])
    assert np.allclose(
        non_nan["mm_net"].to_numpy(),
        (non_nan["mm_long"] - non_nan["mm_short"]).to_numpy(),
    )

    # Spot check the known GC 2023-12-26 row from the fixture.
    gc_last = parsed[(parsed["commodity"] == "GC") & (parsed["report_date"] == "2023-12-26")]
    assert len(gc_last) == 1
    assert gc_last["mm_long"].iloc[0] == 152801.0
    assert gc_last["mm_short"].iloc[0] == 46347.0
    assert gc_last["mm_net"].iloc[0] == 152801.0 - 46347.0
    assert gc_last["open_interest"].iloc[0] == 491343.0


def test_parse_cot_csv_drops_non_universe_commodities() -> None:
    """Rows for contract codes not in the mapping are dropped."""
    raw = _load_fixture()
    parsed = parse_cot_csv(raw, {"CL": CL_CODE})

    assert set(parsed["commodity"].unique()) == {"CL"}
    assert len(parsed) == 4  # four weekly rows for CL only


def test_parse_cot_csv_handles_dot_as_nan() -> None:
    """The '.' string that CFTC uses for missing values must become NaN."""
    raw = _load_fixture()
    parsed = parse_cot_csv(raw, {"CL": CL_CODE, "GC": GC_CODE})

    # The fixture has "." in CL mm_long on 2023-12-26.
    target = parsed[(parsed["commodity"] == "CL") & (parsed["report_date"] == "2023-12-26")]
    assert len(target) == 1
    assert pd.isna(target["mm_long"].iloc[0])
    # mm_short on that row is still a valid number.
    assert target["mm_short"].iloc[0] == 97235.0
    # mm_net should be NaN when either side is NaN (not silently 0).
    assert pd.isna(target["mm_net"].iloc[0])


def test_parse_cot_csv_raises_on_missing_columns() -> None:
    """Missing required columns should raise ValueError naming them."""
    raw = _load_fixture().drop(columns=["M_Money_Positions_Long_All"])
    with pytest.raises(ValueError, match="M_Money_Positions_Long_All"):
        parse_cot_csv(raw, {"CL": CL_CODE})


def test_compute_net_speculative_pivots_wide() -> None:
    """Long-format input pivots to report_date x commodity wide format."""
    raw = _load_fixture()
    long = parse_cot_csv(raw, {"CL": CL_CODE, "GC": GC_CODE})

    wide = compute_net_speculative(long)

    assert set(wide.columns) == {"CL", "GC"}
    assert len(wide) == 4
    assert wide.index.name == "report_date"

    # Spot-check one cell against the long-format mm_net.
    gc_dec26_long = long[(long["commodity"] == "GC") & (long["report_date"] == "2023-12-26")][
        "mm_net"
    ].iloc[0]
    assert wide.loc[pd.Timestamp("2023-12-26"), "GC"] == gc_dec26_long


def test_lag_to_release_date_friday_default() -> None:
    """Tuesday report_date should map to the following Friday release."""
    # 2020-06-16 is a Tuesday; its Friday is 2020-06-19.
    df = pd.DataFrame(
        {
            "commodity": ["CL", "CL", "GC"],
            "report_date": pd.to_datetime(["2020-06-16", "2020-06-23", "2020-06-16"]),
            "mm_long": [100.0, 110.0, 200.0],
            "mm_short": [50.0, 55.0, 80.0],
            "mm_net": [50.0, 55.0, 120.0],
            "open_interest": [1000.0, 1100.0, 2000.0],
        }
    )

    lagged = lag_to_release_date(df)

    assert "release_date" in lagged.columns
    # 2020-06-16 Tue -> 2020-06-19 Fri
    cl_first = lagged[(lagged["commodity"] == "CL") & (lagged["report_date"] == "2020-06-16")]
    assert cl_first["release_date"].iloc[0] == pd.Timestamp("2020-06-19")
    # 2020-06-23 Tue -> 2020-06-26 Fri
    cl_second = lagged[(lagged["commodity"] == "CL") & (lagged["report_date"] == "2020-06-23")]
    assert cl_second["release_date"].iloc[0] == pd.Timestamp("2020-06-26")

    # Edge case: report_date that is *itself* a Friday should roll to the
    # following Friday (strictly after), not the same day.
    edge = pd.DataFrame(
        {
            "commodity": ["CL"],
            "report_date": pd.to_datetime(["2020-06-19"]),  # a Friday
            "mm_long": [100.0],
            "mm_short": [50.0],
            "mm_net": [50.0],
            "open_interest": [1000.0],
        }
    )
    edge_lagged = lag_to_release_date(edge)
    assert edge_lagged["release_date"].iloc[0] == pd.Timestamp("2020-06-26")

    # Output is sorted by release_date ascending.
    assert lagged["release_date"].is_monotonic_increasing


def test_lag_to_release_date_rejects_bad_weekday() -> None:
    """A nonsense weekday string raises ValueError."""
    df = pd.DataFrame(
        {
            "commodity": ["CL"],
            "report_date": pd.to_datetime(["2020-06-16"]),
            "mm_long": [100.0],
            "mm_short": [50.0],
            "mm_net": [50.0],
            "open_interest": [1000.0],
        }
    )
    with pytest.raises(ValueError, match="release_weekday"):
        lag_to_release_date(df, release_weekday="funday")


def test_save_load_cot_data_roundtrip(tmp_path: Path) -> None:
    """save_cot_data + load_cot_data are inverses."""
    from pandas.testing import assert_frame_equal

    raw = _load_fixture()
    long = parse_cot_csv(raw, {"CL": CL_CODE, "GC": GC_CODE})

    save_cot_data(long, out_dir=tmp_path)
    loaded = load_cot_data(in_dir=tmp_path)

    assert_frame_equal(loaded, long)


def test_load_cot_data_raises_on_missing(tmp_path: Path) -> None:
    """An empty directory raises FileNotFoundError on load."""
    with pytest.raises(FileNotFoundError, match="cot_history.parquet"):
        load_cot_data(in_dir=tmp_path)


def test_download_cot_history_skips_missing_years(monkeypatch) -> None:
    """download_cot_history swallows None years without raising."""
    raw_2023 = _load_fixture()

    calls: list[int] = []

    def fake_download(year: int, *, use_cache: bool = True) -> pd.DataFrame | None:
        calls.append(year)
        if year == 2022:
            return None  # simulate 404 or network failure
        return raw_2023

    monkeypatch.setattr(cftc_loader, "download_cot_zip", fake_download)

    result = download_cot_history(2022, 2023)

    assert calls == [2022, 2023]
    # Only the 2023 year produced rows (8 = 2 commodities x 4 weeks).
    assert len(result) == 8
    assert set(result["commodity"].unique()) == {"CL", "GC"}


def test_download_cot_history_all_years_fail(monkeypatch) -> None:
    """If every year returns None, result is an empty DataFrame (not a crash)."""
    monkeypatch.setattr(
        cftc_loader,
        "download_cot_zip",
        lambda year, *, use_cache=True: None,
    )

    result = download_cot_history(2020, 2021)
    assert result.empty
    assert list(result.columns) == [
        "commodity",
        "report_date",
        "mm_long",
        "mm_short",
        "mm_net",
        "open_interest",
    ]
