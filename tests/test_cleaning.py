"""Tests for schema validation and data cleaning."""

import numpy as np
import pandas as pd
import pytest

from electricity_load_forecasting.cleaning import clean_hourly_data, validate_schema


def test_validate_schema_valid():
    """Valid hourly DataFrame passes schema validation without error."""
    idx = pd.date_range("2026-01-01", periods=10, freq="1h")
    df = pd.DataFrame({"Global_active_power": np.ones(10)}, index=idx)
    validate_schema(df, "Global_active_power")


def test_validate_schema_invalid_index():
    """Non-datetime index raises ValueError."""
    df = pd.DataFrame({"Global_active_power": [1.0, 2.0]})
    with pytest.raises(ValueError, match="pandas DatetimeIndex"):
        validate_schema(df)


def test_validate_schema_missing_target():
    """Missing target column raises ValueError."""
    idx = pd.date_range("2026-01-01", periods=5, freq="1h")
    df = pd.DataFrame({"wrong_col": np.ones(5)}, index=idx)
    with pytest.raises(ValueError, match="Required target column"):
        validate_schema(df, "Global_active_power")


def test_validate_schema_non_monotonic():
    """Non-monotonic index raises ValueError."""
    idx = pd.to_datetime(["2026-01-01 02:00", "2026-01-01 01:00"])
    df = pd.DataFrame({"Global_active_power": [1.0, 2.0]}, index=idx)
    with pytest.raises(ValueError, match="strictly monotonic"):
        validate_schema(df)


def test_clean_hourly_data_imputes_and_clips():
    """clean_hourly_data interpolates missing hourly entries and clips out-of-range values."""
    idx = pd.to_datetime(
        [
            "2026-01-01 00:00:00",
            "2026-01-01 01:00:00",
            "2026-01-01 03:00:00",  # 02:00 is missing
            "2026-01-01 04:00:00",
        ]
    )
    df = pd.DataFrame({"Global_active_power": [1.0, 2.0, -5.0, 30.0]}, index=idx)

    cleaned_df, report = clean_hourly_data(
        df, target_col="Global_active_power", min_value=0.0, max_value=20.0
    )

    # Missing 02:00 timestamp should be inserted and interpolated
    assert len(cleaned_df) == 5
    assert report.missing_imputed >= 1
    assert report.negative_clipped >= 1

    # Check clipping: negative value clipped to 0.0, 30.0 clipped to 20.0
    assert cleaned_df["Global_active_power"].min() >= 0.0
    assert cleaned_df["Global_active_power"].max() <= 20.0
    assert not cleaned_df["Global_active_power"].isna().any()
