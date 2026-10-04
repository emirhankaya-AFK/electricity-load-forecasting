"""Tests ensuring strict leakage-safety across feature engineering and temporal splitting."""

import numpy as np
import pandas as pd
import pytest

from electricity_load_forecasting.data import generate_synthetic_hourly_data
from electricity_load_forecasting.features import FeatureEngineer
from electricity_load_forecasting.split import chronological_split


def test_lag_features_strict_past(sample_hourly_df: pd.DataFrame):
    """Verify lag features only reference historical observations."""
    target_col = "Global_active_power"
    engineer = FeatureEngineer(lag_hours=[1, 2, 24, 168], rolling_windows=[])
    X, y = engineer.create_features(sample_hourly_df, target_col=target_col, drop_na=False)

    # For any timestamp t, lag_1h must match original series at t - 1h
    for lag in [1, 2, 24, 168]:
        col_name = f"lag_{lag}h"
        # Check against ground truth series shifted manually
        expected = sample_hourly_df[target_col].shift(lag)
        pd.testing.assert_series_equal(
            X[col_name].dropna(),
            expected.dropna(),
            check_names=False,
        )


def test_rolling_features_shift_one_safety():
    """Verify rolling statistics strictly exclude current time step y_t."""
    # Create simple sequential series: 1, 2, 3, 4, 5, ...
    dates = pd.date_range("2026-01-01", periods=50, freq="1h")
    values = np.arange(1.0, 51.0)
    df = pd.DataFrame({"load": values}, index=dates)

    engineer = FeatureEngineer(
        lag_hours=[],
        rolling_windows=[3],
        rolling_stats=["mean"],
        include_calendar=False,
        include_cyclical=False,
    )
    X, y = engineer.create_features(df, target_col="load", drop_na=False)

    # At row index 4 (t=4, value=5.0), the 3-hour rolling window should use values [t=1(2), t=2(3), t=3(4)]
    # Mean of (2, 3, 4) = 3.0. If y_t (5.0) leaked into rolling window, mean would be (3+4+5)/3 = 4.0
    row_4_mean = X.loc[dates[4], "rolling_mean_3h"]
    assert np.isclose(row_4_mean, 3.0), f"Expected 3.0, got {row_4_mean} (Target leakage detected!)"


def test_perturbation_at_t_does_not_affect_features_at_t():
    """Corrupting y_t must not change any features at time t."""
    df1 = generate_synthetic_hourly_data(n_hours=300, random_state=42)
    df2 = df1.copy()

    engineer = FeatureEngineer(
        lag_hours=[1, 24],
        rolling_windows=[24],
        rolling_stats=["mean", "std"],
    )

    X1, _ = engineer.create_features(df1, drop_na=True)

    # Alter the target value at index 250 in df2
    idx_target = df1.index[250]
    df2.loc[idx_target, "Global_active_power"] += 999.0

    X2, _ = engineer.create_features(df2, drop_na=True)

    # At idx_target, the engineered features in X1 and X2 MUST be identical
    # because features at t only depend on t-1, t-2, ...
    pd.testing.assert_series_equal(
        X1.loc[idx_target],
        X2.loc[idx_target],
        check_names=False,
    )


def test_chronological_split_strict_temporality(sample_hourly_df: pd.DataFrame):
    """Verify chronological split guarantees non-overlapping temporal partitions without shuffle."""
    splits = chronological_split(
        sample_hourly_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    assert splits.train_end < splits.val_start, "Train end must precede validation start"
    assert splits.val_end < splits.test_start, "Validation end must precede test start"

    # Verify no timestamp gaps between partitions
    assert splits.train_end + pd.Timedelta(hours=1) == splits.val_start
    assert splits.val_end + pd.Timedelta(hours=1) == splits.test_start

    # Verify complete partition conservation
    assert len(splits.train) + len(splits.val) + len(splits.test) == len(sample_hourly_df)


def test_chronological_split_invalid_ratios(sample_hourly_df: pd.DataFrame):
    """Test error handling for bad split ratios."""
    with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
        chronological_split(sample_hourly_df, 0.5, 0.5, 0.5)

    with pytest.raises(ValueError, match="Dataset too small"):
        small_df = sample_hourly_df.iloc[:20]
        chronological_split(small_df, 0.7, 0.15, 0.15)
