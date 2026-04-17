"""Block bootstrap confidence intervals for Sharpe ratio."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_samples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
    block_size: int = 20,
) -> tuple[float, float, float]:
    """Block bootstrap confidence interval for the annualised Sharpe ratio.

    Uses circular block bootstrap to preserve serial correlation structure
    in daily returns.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns.
    n_samples : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level (e.g. 0.95 for 95% CI).
    seed : int
        Random seed for reproducibility.
    block_size : int
        Block length for the circular block bootstrap.

    Returns
    -------
    tuple[float, float, float]
        (point_estimate, ci_lower, ci_upper) — all annualised.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(returns.dropna().values, dtype=np.float64)
    n = len(arr)

    if n < block_size * 2:
        logger.warning("Too few observations (%d) for block bootstrap", n)
        return 0.0, 0.0, 0.0

    n_blocks = int(np.ceil(n / block_size))

    sharpes = np.empty(n_samples)
    for i in range(n_samples):
        starts = rng.integers(0, n, size=n_blocks)
        blocks = []
        for s in starts:
            block = np.take(arr, range(s, s + block_size), mode="wrap")
            blocks.append(block)
        sample = np.concatenate(blocks)[:n]
        std = sample.std()
        if std > 0:
            sharpes[i] = (sample.mean() / std) * np.sqrt(252)
        else:
            sharpes[i] = 0.0

    arr_std = float(arr.std())
    point = float((float(arr.mean()) / arr_std) * np.sqrt(252)) if arr_std > 0 else 0.0
    alpha = (1 - confidence) / 2
    ci_lower = float(np.percentile(sharpes, alpha * 100))
    ci_upper = float(np.percentile(sharpes, (1 - alpha) * 100))

    logger.info(
        "bootstrap_sharpe_ci: point=%.3f, %.0f%% CI=[%.3f, %.3f], n_samples=%d",
        point,
        confidence * 100,
        ci_lower,
        ci_upper,
        n_samples,
    )
    return point, ci_lower, ci_upper
