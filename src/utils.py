"""
Utility functions for Social State Quantification.

Provides z-score normalization, LOCF imputation, and polarity alignment
as described in the paper.
"""

import numpy as np
import pandas as pd


def rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """
    Compute rolling z-score based on a moving window.

    Each value is standardized using the mean and standard deviation
    of the preceding `window` months.

    Parameters
    ----------
    series : pd.Series
        Monthly time series of a single indicator.
    window : int
        Rolling window size in months (default: 60).

    Returns
    -------
    pd.Series
        Z-score transformed series.
    """
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    return (series - rolling_mean) / rolling_std


def locf(series: pd.Series) -> pd.Series:
    """
    Last Observation Carried Forward (LOCF) imputation.

    Fills missing values by carrying the last observed value forward.

    Parameters
    ----------
    series : pd.Series
        Time series with potential missing values.

    Returns
    -------
    pd.Series
        Series with missing values filled.
    """
    return series.ffill()


def align_polarity(series: pd.Series, invert: bool = False) -> pd.Series:
    """
    Align indicator polarity toward the stress-increasing direction.

    For indicators where higher values mean LESS stress (e.g., wages),
    set invert=True to flip the sign.

    Parameters
    ----------
    series : pd.Series
        Z-score normalized indicator.
    invert : bool
        If True, multiply by -1 to align polarity.

    Returns
    -------
    pd.Series
        Polarity-aligned series.
    """
    if invert:
        return -series
    return series


# Default indicator configuration
# Each entry: (column_name, invert_polarity)
# invert=True means "higher raw value = lower stress" (e.g., wages)
DEFAULT_INDICATORS = [
    ("cpi", False),                  # CPI: higher = more stress
    ("total_cash_earnings", True),   # Wages: higher = less stress
    ("unemployment_rate", False),    # Unemployment: higher = more stress
    ("household_expenditure", True), # Expenditure capacity: higher = less stress
    ("food_engel_coefficient", False),  # Engel coefficient: higher = more stress
    ("working_hours", False),        # Overwork: higher = more stress
    ("part_time_ratio", False),      # Non-regular employment: higher = more stress
    ("consumer_confidence", True),   # Confidence: higher = less stress
    ("savings_rate", True),          # Savings: higher = less stress
    ("debt_ratio", False),           # Debt: higher = more stress
]
