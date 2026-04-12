"""Cross-sectional ranking signal.

Converts daily factor z-scores into long/short portfolio weights by selecting
the top ``long_n`` and bottom ``short_n`` commodities on each date.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def rank_and_select(
    scores: pd.DataFrame,
    long_n: int = 3,
    short_n: int = 3,
) -> pd.DataFrame:
    """Per-day cross-sectional rank → long/short weights.

    Top ``long_n`` commodities get equal positive weight (+1/long_n each).
    Bottom ``short_n`` get equal negative weight (-1/short_n each).
    Middle commodities get 0. Weights sum to 0 (dollar-neutral).
    NaN scores are excluded from ranking.

    Parameters
    ----------
    scores : pd.DataFrame
        DatetimeIndex × commodity columns, values = factor z-scores.
    long_n : int
        Number of commodities to go long. Default 3.
    short_n : int
        Number of commodities to go short. Default 3.

    Returns
    -------
    pd.DataFrame
        Same index and columns as *scores*, values = portfolio weights.
        Long legs sum to +1.0, short legs sum to -1.0 (each side), net = 0.
    """
    if long_n < 1 or short_n < 1:
        raise ValueError("long_n and short_n must each be >= 1")

    weights = pd.DataFrame(np.nan, index=scores.index, columns=scores.columns)

    for date, row in scores.iterrows():
        valid = row.dropna()
        n_valid = len(valid)

        if n_valid < long_n + short_n:
            # Not enough valid observations — all weights remain 0
            weights.loc[date] = 0.0
            logger.debug(
                "rank_and_select: %s has only %d valid scores, need %d; setting all weights to 0",
                date,
                n_valid,
                long_n + short_n,
            )
            continue

        sorted_idx = valid.sort_values(ascending=False).index
        longs = sorted_idx[:long_n]
        shorts = sorted_idx[-short_n:]

        row_weights = pd.Series(0.0, index=scores.columns)
        row_weights[longs] = 1.0 / long_n
        row_weights[shorts] = -1.0 / short_n
        weights.loc[date] = row_weights

    logger.info(
        "rank_and_select: %d dates, long_n=%d, short_n=%d",
        len(scores),
        long_n,
        short_n,
    )

    return weights
