"""Positioning filter.

Zeroes out momentum signals when speculative positioning is crowded —
i.e. when too many market participants are already on the same side.
This reduces the risk of entering a trade that is about to reverse due
to crowded positioning rather than a genuine signal.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def apply_positioning_filter(
    momentum_signal: pd.DataFrame,
    positioning_percentile: pd.DataFrame,
    crowded_threshold: float = 0.90,
) -> pd.DataFrame:
    """Zero out momentum signals when positioning is crowded.

    When momentum_signal > 0 and positioning_percentile > crowded_threshold
    (too many specs long → crowded), set signal to 0.
    When momentum_signal < 0 and positioning_percentile < (1 - crowded_threshold)
    (too many specs short → crowded), set signal to 0.
    Otherwise, pass momentum_signal through unchanged.

    Parameters
    ----------
    momentum_signal : pd.DataFrame
        DatetimeIndex × commodity columns, directional signal values.
    positioning_percentile : pd.DataFrame
        DatetimeIndex × commodity columns, values in [0, 1].
        Produced by ``factors.transforms.percentile_rank`` on CFTC net positions.
    crowded_threshold : float
        Upper percentile boundary for crowded-long detection.  The lower
        boundary for crowded-short is ``1 - crowded_threshold``.  Default 0.90.

    Returns
    -------
    pd.DataFrame
        Same shape, index, and columns as *momentum_signal*.
        NaN in either input → NaN in output.
    """
    if not (0.0 < crowded_threshold < 1.0):
        raise ValueError(
            f"crowded_threshold must be in (0, 1), got {crowded_threshold}"
        )

    # Align on common index/columns
    sig, pos = momentum_signal.align(positioning_percentile, join="left")

    result = sig.copy()

    # Crowded long: signal > 0 AND positioning_percentile > crowded_threshold
    crowded_long = (sig > 0) & (pos > crowded_threshold)
    result[crowded_long] = 0.0

    # Crowded short: signal < 0 AND positioning_percentile < (1 - crowded_threshold)
    crowded_short = (sig < 0) & (pos < (1.0 - crowded_threshold))
    result[crowded_short] = 0.0

    logger.info(
        "apply_positioning_filter: %d crowded-long, %d crowded-short signals zeroed out "
        "(threshold=%.2f)",
        int(crowded_long.sum().sum()),
        int(crowded_short.sum().sum()),
        crowded_threshold,
    )

    return result
