"""Calendar spread signal.

Converts carry z-scores into front/back leg weights for calendar spread
trades. Strong backwardation signals long front / short back; strong
contango signals the reverse.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def calendar_spread_signal(
    carry_zscore: pd.DataFrame,
    long_threshold: float = 1.0,
    short_threshold: float = -1.0,
) -> pd.DataFrame:
    """Calendar spread signal based on carry z-score.

    When carry_z > long_threshold (strong backwardation): long front, short back.
    When carry_z < short_threshold (strong contango): short front, long back.
    Otherwise: flat.

    Parameters
    ----------
    carry_zscore : pd.DataFrame
        DatetimeIndex × commodity columns, values = carry z-scores.
    long_threshold : float
        Z-score strictly above this → long front leg. Default 1.0.
    short_threshold : float
        Z-score strictly below this → short front leg. Default -1.0.
        Must be <= long_threshold.

    Returns
    -------
    pd.DataFrame
        MultiIndex columns: (commodity, leg) where leg ∈ {"front", "back"}.
        Values are weights: +1, -1, or 0.
    """
    if short_threshold > long_threshold:
        raise ValueError(
            f"short_threshold ({short_threshold}) must be <= long_threshold ({long_threshold})"
        )

    commodities = carry_zscore.columns.tolist()
    legs = ["front", "back"]
    cols = pd.MultiIndex.from_product([commodities, legs], names=["commodity", "leg"])
    result = pd.DataFrame(0.0, index=carry_zscore.index, columns=cols)

    for comm in commodities:
        z = carry_zscore[comm]

        # Strong backwardation: long front, short back
        is_long = z > long_threshold
        result.loc[is_long, (comm, "front")] = 1.0
        result.loc[is_long, (comm, "back")] = -1.0

        # Strong contango: short front, long back
        is_short = z < short_threshold
        result.loc[is_short, (comm, "front")] = -1.0
        result.loc[is_short, (comm, "back")] = 1.0

    logger.info(
        "calendar_spread_signal: %d commodities, %d dates, "
        "long_threshold=%.2f, short_threshold=%.2f",
        len(commodities),
        len(carry_zscore),
        long_threshold,
        short_threshold,
    )

    return result
