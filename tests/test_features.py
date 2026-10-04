"""Tests for feature engineering logic."""

import numpy as np
import pandas as pd
import pytest

from electricity_load_forecasting.features import FeatureEngineer


def test_feature_columns_and_shapes(sample_hourly_df: pd.DataFrame):
    """Verify engineered features have expected column names and no NaNs."""
    engineer = FeatureEngineer(
        lag_hours=[1, 2, 24],
        rolling_windows=[24],
        rolling_stats=["mean", "std"],
        include_calendar=True,
        include_cyclical=True,
    )

    X, y = engineer.create_features(sample_hourly_df, drop_na=True)

    expected_cols = [
        "lag_1h",
        "lag_2h",
        "lag_24h",
        "rolling_mean_24h",
        "rolling_std_24h",
        "hour",
        "dayofweek",
        "day",
        "month",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "month_sin",
        "month_cos",
    ]

    for col in expected_cols:
        assert col in X.columns, f"Missing feature column: {col}"

    assert not X.isna().any().any(), "Features should contain no NaNs after drop_na=True"
    assert len(X) == len(y)
    assert len(X) < len(sample_hourly_df)  # Warmup rows dropped


def test_cyclical_features_bounds(sample_hourly_df: pd.DataFrame):
    """Sine and cosine features must be bounded in [-1.0, 1.0]."""
    engineer = FeatureEngineer()
    X, _ = engineer.create_features(sample_hourly_df, drop_na=True)

    cyclical_cols = [col for col in X.columns if col.endswith("_sin") or col.endswith("_cos")]
    for col in cyclical_cols:
        assert X[col].min() >= -1.0 - 1e-6, f"{col} has values < -1"
        assert X[col].max() <= 1.0 + 1e-6, f"{col} has values > 1"


def test_extract_inference_features_consistency(sample_hourly_df: pd.DataFrame):
    """Inference feature extraction must match full dataset feature generation."""
    engineer = FeatureEngineer(lag_hours=[1, 2, 24], rolling_windows=[24], rolling_stats=["mean"])
    X_full, _ = engineer.create_features(sample_hourly_df, drop_na=True)

    # Pick a timestamp in X_full
    target_idx = X_full.index[100]
    # Slice historical series up to target_idx - 1h
    history_slice = sample_hourly_df.loc[
        : target_idx - pd.Timedelta(hours=1), "Global_active_power"
    ]

    inference_row = engineer.extract_inference_features(
        history_slice, forecast_timestamp=target_idx
    )

    # Ensure identical columns and values
    assert list(inference_row.columns) == list(X_full.columns)
    for col in inference_row.columns:
        val_inf = inference_row[col].iloc[0]
        val_full = X_full.loc[target_idx, col]
        assert np.isclose(val_inf, val_full), (
            f"Mismatch in {col}: inference={val_inf}, batch={val_full}"
        )


def test_extract_inference_insufficient_history():
    """Extracting features with too short history raises ValueError."""
    engineer = FeatureEngineer(lag_hours=[168])
    short_history = pd.Series(np.ones(50))
    with pytest.raises(ValueError, match="less than required"):
        engineer.extract_inference_features(short_history)
