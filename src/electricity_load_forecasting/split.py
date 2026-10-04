"""Leakage-safe chronological train/validation/test splitting."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import pandas as pd

T = TypeVar("T", pd.DataFrame, pd.Series)


@dataclass
class ChronologicalSplit(Generic[T]):
    train: T
    val: T
    test: T

    @property
    def train_start(self) -> pd.Timestamp:
        return self.train.index.min()

    @property
    def train_end(self) -> pd.Timestamp:
        return self.train.index.max()

    @property
    def val_start(self) -> pd.Timestamp:
        return self.val.index.min()

    @property
    def val_end(self) -> pd.Timestamp:
        return self.val.index.max()

    @property
    def test_start(self) -> pd.Timestamp:
        return self.test.index.min()

    @property
    def test_end(self) -> pd.Timestamp:
        return self.test.index.max()


def chronological_split(
    df: T,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> ChronologicalSplit[T]:
    """Split time-series data chronologically without leakage or shuffling.

    Args:
        df: Input DataFrame or Series with DatetimeIndex.
        train_ratio: Proportion of data for training (e.g. 0.70).
        val_ratio: Proportion of data for validation / conformal calibration (e.g. 0.15).
        test_ratio: Proportion of data for test evaluation (e.g. 0.15).

    Returns:
        ChronologicalSplit object containing train, val, and test slices.

    Raises:
        ValueError: If ratios do not sum to 1.0 or if dataset is too small.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.999 <= total_ratio <= 1.001):
        raise ValueError(f"Ratios must sum to 1.0 (got {total_ratio:.4f})")

    n = len(df)
    if n < 50:
        raise ValueError(f"Dataset too small ({n} rows) for meaningful chronological split.")

    train_end_idx = int(n * train_ratio)
    val_end_idx = int(n * (train_ratio + val_ratio))

    if train_end_idx <= 0 or val_end_idx <= train_end_idx or val_end_idx >= n:
        raise ValueError("Calculated split points result in empty partition.")

    train_data = df.iloc[:train_end_idx]
    val_data = df.iloc[train_end_idx:val_end_idx]
    test_data = df.iloc[val_end_idx:]

    # Assert strictly non-overlapping temporal bounds
    assert train_data.index.max() < val_data.index.min(), (
        f"Temporal leakage detected: train end {train_data.index.max()} >= val start {val_data.index.min()}"
    )
    assert val_data.index.max() < test_data.index.min(), (
        f"Temporal leakage detected: val end {val_data.index.max()} >= test start {test_data.index.min()}"
    )

    return ChronologicalSplit(train=train_data, val=val_data, test=test_data)
