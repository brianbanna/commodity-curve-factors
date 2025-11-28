"""
Hidden Markov Models for regime classification.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def fit_hmm(
    returns: pd.Series,
    n_states: int = 2,
    n_iter: int = 100
) -> Tuple[object, np.ndarray]:
    """
    Fit Hidden Markov Model to returns for regime detection.

    Parameters:
        returns: Series of asset returns.
        n_states: Number of hidden states (regimes).
        n_iter: Maximum number of EM iterations.

    Returns:
        Tuple of (fitted_model, state_sequence).
    """
    pass


def classify_volatility_regimes(
    hmm_model: object,
    returns: pd.Series
) -> pd.Series:
    """
    Classify volatility regimes from HMM states.

    Parameters:
        hmm_model: Fitted HMM model.
        returns: Series of asset returns.

    Returns:
        Series with regime labels ('low_vol', 'high_vol', etc.).
    """
    pass


def compute_regime_transition_matrix(
    states: np.ndarray,
    n_states: int
) -> pd.DataFrame:
    """
    Compute transition probability matrix between regimes.

    Parameters:
        states: Array of state sequences.
        n_states: Number of states.

    Returns:
        DataFrame with transition probabilities.
    """
    pass

