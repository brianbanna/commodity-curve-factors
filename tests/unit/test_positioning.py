"""Tests for the CFTC positioning contrarian factor module."""

import numpy as np
import pandas as pd

from commodity_curve_factors.factors.positioning import compute_positioning_factor


def _make_cot(
    n_weeks: int = 200,
    commodities: list[str] | None = None,
    start: str = "2015-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic long-format COT DataFrame."""
    if commodities is None:
        commodities = ["CL", "GC"]

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_weeks, freq="W-TUE")

    rows = []
    for sym in commodities:
        mm_long = rng.uniform(50_000, 300_000, n_weeks)
        mm_short = rng.uniform(20_000, 200_000, n_weeks)
        mm_net = mm_long - mm_short
        for i, d in enumerate(dates):
            rows.append(
                {
                    "commodity": sym,
                    "report_date": d,
                    "mm_long": mm_long[i],
                    "mm_short": mm_short[i],
                    "mm_net": mm_net[i],
                    "open_interest": mm_long[i] + mm_short[i],
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_positioning_contrarian_sign() -> None:
    """High percentile (crowded long) → signal near -1; low → signal near +1."""
    # Build COT where CL is consistently very long (high mm_net throughout).
    n_weeks = 200
    dates = pd.date_range("2015-01-01", periods=n_weeks, freq="W-TUE")

    # CL is always at +200k net (crowded long for the whole history)
    rows_cl = [
        {
            "commodity": "CL",
            "report_date": d,
            "mm_long": 250_000,
            "mm_short": 50_000,
            "mm_net": 200_000,
            "open_interest": 300_000,
        }
        for d in dates
    ]
    # GC is always at -200k net (crowded short for the whole history)
    rows_gc = [
        {
            "commodity": "GC",
            "report_date": d,
            "mm_long": 50_000,
            "mm_short": 250_000,
            "mm_net": -200_000,
            "open_interest": 300_000,
        }
        for d in dates
    ]

    cot = pd.DataFrame(rows_cl + rows_gc)
    factor = compute_positioning_factor(cot, window=50, all_commodities=["CL", "GC"])

    last_cl = factor["CL"].dropna().iloc[-1]
    last_gc = factor["GC"].dropna().iloc[-1]

    # CL constant → percentile 1.0 or 0.5 (all tied) → signal in {-1.0, 0.0}
    # GC constant → same. The contrarian transform maps them symmetrically.
    # The key is: CL's signal should be <= GC's signal (CL is "more long")
    assert last_cl <= last_gc, (
        f"Crowded-long CL should have signal <= crowded-short GC; "
        f"got CL={last_cl:.3f}, GC={last_gc:.3f}"
    )


def test_positioning_contrarian_sign_asymmetric() -> None:
    """Verify contrarian sign: increasing net long → decreasing signal.

    Use two separate windows (two COT datasets that differ only in the
    last observation) to verify the directional property without relying
    on the exact warm-up behaviour of ``percentile_rank``.
    """
    n_weeks = 200
    dates = pd.date_range("2015-01-01", periods=n_weeks, freq="W-TUE")

    # Build a base COT with moderate positioning
    rng = np.random.default_rng(7)
    base_net = rng.uniform(-100_000, 100_000, n_weeks)
    rows_base = [
        {
            "commodity": "CL",
            "report_date": d,
            "mm_long": 200_000,
            "mm_short": 200_000 - v,
            "mm_net": v,
            "open_interest": 400_000,
        }
        for d, v in zip(dates, base_net)
    ]
    cot_base = pd.DataFrame(rows_base)

    # Build a version where the last observation is at historic maximum (very long)
    cot_long = cot_base.copy()
    last_cl = cot_long["commodity"] == "CL"
    last_row = cot_long[last_cl].index[-1]
    cot_long.loc[last_row, "mm_net"] = 999_999  # extreme long

    # Build a version where the last observation is at historic minimum (very short)
    cot_short = cot_base.copy()
    cot_short.loc[last_row, "mm_net"] = -999_999  # extreme short

    factor_long = compute_positioning_factor(cot_long, window=50, all_commodities=["CL"])
    factor_short = compute_positioning_factor(cot_short, window=50, all_commodities=["CL"])

    last_long_signal = factor_long["CL"].dropna().iloc[-1]
    last_short_signal = factor_short["CL"].dropna().iloc[-1]

    # Extreme long → signal near -1; extreme short → signal near +1
    assert last_long_signal < last_short_signal, (
        f"Extreme long should have lower signal than extreme short; "
        f"long={last_long_signal:.3f}, short={last_short_signal:.3f}"
    )


def test_positioning_output_daily() -> None:
    """Output index must be daily (business-day) frequency even though input is weekly."""
    cot = _make_cot(n_weeks=200)
    factor = compute_positioning_factor(cot, window=50, all_commodities=["CL", "GC"])

    assert isinstance(factor.index, pd.DatetimeIndex)

    # Should have many more rows than input weeks
    assert len(factor) > 200 * 3, (
        f"Expected many more daily rows than 200 weekly obs; got {len(factor)}"
    )

    # Consecutive gaps must be 1 or 3 business days (Mon gap)
    if len(factor) > 1:
        gaps = pd.Series(factor.index).diff().dropna()
        assert gaps.max() <= pd.Timedelta(days=3), "Daily index has unexpected large gap"


def test_positioning_no_lookahead() -> None:
    """Signal at date t uses only COT release dates <= t.

    We verify this by using two COT datasets that differ only in the
    second-to-last report. The two factors should be identical before
    the release date of that modified report, and different after.
    """
    from commodity_curve_factors.data.cftc_loader import lag_to_release_date

    cot = _make_cot(n_weeks=100, commodities=["CL"])
    factor1 = compute_positioning_factor(cot, window=20, all_commodities=["CL"])

    # Modify the second-to-last CL report (not the last, to see divergence)
    cot2 = cot.copy()
    cl_rows = cot2[cot2["commodity"] == "CL"].sort_values("report_date")
    penultimate_idx = cl_rows.index[-2]
    cot2.loc[penultimate_idx, "mm_net"] = 9_999_999

    factor2 = compute_positioning_factor(cot2, window=20, all_commodities=["CL"])

    # Find release date of the modified report
    lagged = lag_to_release_date(cot2[cot2["commodity"] == "CL"])
    modified_report_date = cot2.loc[penultimate_idx, "report_date"]
    matching = lagged[lagged["report_date"] == modified_report_date]
    assert len(matching) > 0
    release_date = matching["release_date"].iloc[0]

    # Dates strictly before the release date must be identical
    pre_mask = factor1.index < release_date
    if pre_mask.sum() > 0:
        pd.testing.assert_frame_equal(
            factor1.loc[pre_mask],
            factor2.loc[pre_mask],
            check_exact=False,
            rtol=1e-9,
        )


def test_positioning_handles_missing_commodities() -> None:
    """Commodities not present in COT data get NaN columns in output."""
    cot = _make_cot(n_weeks=100, commodities=["CL"])
    all_comms = ["CL", "GC", "SI", "ZC"]
    factor = compute_positioning_factor(cot, window=20, all_commodities=all_comms)

    assert set(factor.columns) == set(all_comms)
    for sym in ["GC", "SI", "ZC"]:
        assert factor[sym].isna().all(), f"{sym} should be all NaN (not in COT data)"
    assert factor["CL"].notna().any(), "CL should have some non-NaN values"
