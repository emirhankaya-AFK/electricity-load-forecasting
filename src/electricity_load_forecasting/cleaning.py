"""Data schema validation and cleaning pipeline."""

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CleaningReport(NamedTuple):
    initial_rows: int
    cleaned_rows: int
    missing_imputed: int
    negative_clipped: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp


def validate_schema(df: pd.DataFrame, target_col: str = "Global_active_power") -> None:
    """Validate DataFrame schema, index, and data types.

    Args:
        df: Input DataFrame.
        target_col: Target variable column name.

    Raises:
        ValueError: If schema constraints are violated.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pandas DatetimeIndex.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("DatetimeIndex must be strictly monotonic increasing.")

    if df.index.has_duplicates:
        raise ValueError("DatetimeIndex contains duplicate timestamps.")

    if target_col not in df.columns:
        raise ValueError(f"Required target column '{target_col}' is missing.")

    if not np.issubdtype(df[target_col].dtype, np.number):
        raise ValueError(
            f"Target column '{target_col}' must be numeric, got {df[target_col].dtype}"
        )


def clean_hourly_data(
    df: pd.DataFrame,
    target_col: str = "Global_active_power",
    min_value: float = 0.0,
    max_value: float = 20.0,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean and validate hourly electricity load time series.

    Steps:
    1. Validate DatetimeIndex structure.
    2. Ensure regular hourly frequency.
    3. Impute any missing intervals via time-weighted interpolation.
    4. Clip invalid values (negative or unrealistically high surges).

    Args:
        df: Input hourly DataFrame.
        target_col: Target column name.
        min_value: Minimum valid load value (kW).
        max_value: Maximum valid load value (kW).

    Returns:
        Tuple of (cleaned DataFrame, CleaningReport).
    """
    initial_rows = len(df)
    if initial_rows == 0:
        raise ValueError("Input DataFrame is empty.")

    df_clean = df.copy()

    # Ensure index is sorted
    if not df_clean.index.is_monotonic_increasing:
        df_clean = df_clean.sort_index()

    # Deduplicate index if any
    if df_clean.index.has_duplicates:
        df_clean = df_clean[~df_clean.index.duplicated(keep="last")]

    # Reindex to regular 1h frequency
    df_clean = df_clean.asfreq("1h")

    # Count missing values before imputation
    missing_count = int(df_clean[target_col].isna().sum())

    # Time-based interpolation followed by edge fills
    df_clean[target_col] = df_clean[target_col].interpolate(method="time").ffill().bfill()

    # Count and clip out-of-range values
    raw_vals = df_clean[target_col].to_numpy()
    out_of_bounds = np.sum((raw_vals < min_value) | (raw_vals > max_value))
    df_clean[target_col] = np.clip(df_clean[target_col], min_value, max_value)

    # Final schema check
    validate_schema(df_clean, target_col=target_col)

    report = CleaningReport(
        initial_rows=initial_rows,
        cleaned_rows=len(df_clean),
        missing_imputed=missing_count,
        negative_clipped=int(out_of_bounds),
        start_time=df_clean.index.min(),
        end_time=df_clean.index.max(),
    )
    logger.info("Cleaned data: %s", report)
    return df_clean, report
