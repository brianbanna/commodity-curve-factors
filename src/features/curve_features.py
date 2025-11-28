"""
Extract curve-based features: carry, slope, curvature, and roll yield.
"""

import pandas as pd
import numpy as np
from typing import Optional, List


def compute_carry(
    prices: pd.DataFrame,
    near_month: int = 1,
    far_month: int = 12
) -> pd.Series:
    """
    Compute carry factor as the log difference between near and far contracts.

    Parameters:
        prices: DataFrame with futures prices by maturity.
        near_month: Months to maturity for near contract.
        far_month: Months to maturity for far contract.

    Returns:
        Series of daily carry values.
    """
    pass


def compute_slope(
    prices: pd.DataFrame,
    maturities: Optional[List[int]] = None
) -> pd.Series:
    """
    Compute curve slope as first principal component or simple spread.

    Parameters:
        prices: DataFrame with futures prices by maturity.
        maturities: List of maturities to use for slope calculation.

    Returns:
        Series of daily slope values.
    """
    pass


def compute_curvature(
    prices: pd.DataFrame,
    short_month: int = 1,
    mid_month: int = 6,
    long_month: int = 12
) -> pd.Series:
    """
    Compute curve curvature using butterfly spread.

    Parameters:
        prices: DataFrame with futures prices by maturity.
        short_month: Short leg maturity in months.
        mid_month: Middle leg maturity in months.
        long_month: Long leg maturity in months.

    Returns:
        Series of daily curvature values.
    """
    pass


def compute_roll_yield(
    prices: pd.DataFrame,
    roll_dates: pd.DatetimeIndex
) -> pd.Series:
    """
    Compute roll yield from contract transitions.

    Parameters:
        prices: DataFrame with futures prices by contract.
        roll_dates: DatetimeIndex of roll dates.

    Returns:
        Series of roll yield values.
    """
    pass


def classify_regime(
    carry: pd.Series,
    threshold: float = 0.0
) -> pd.Series:
    """
    Classify market regime as backwardation or contango.

    Parameters:
        carry: Series of carry values.
        threshold: Threshold for regime classification.

    Returns:
        Series with 'backwardation' or 'contango' labels.
    """
    pass

