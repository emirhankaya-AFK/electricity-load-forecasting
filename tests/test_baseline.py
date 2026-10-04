"""Tests for SeasonalNaiveBaseline model."""

import numpy as np
import pandas as pd

from electricity_load_forecasting.models import SeasonalNaiveBaseline


def test_seasonal_naive_baseline_prediction():
    """Seasonal naive predicts the lag_24h column."""
    df = pd.DataFrame(
        {
            "lag_24h": [2.5, 3.1, 4.2],
            "other_feat": [1.0, 1.0, 1.0],
        }
    )
    y = pd.Series([2.6, 3.0, 4.1])

    baseline = SeasonalNaiveBaseline(season_lag=24, lag_feature_col="lag_24h")
    baseline.fit(df, y)
    preds = baseline.predict(df)

    np.testing.assert_array_equal(preds, df["lag_24h"].to_numpy())


def test_seasonal_naive_baseline_fallback():
    """Seasonal naive falls back to mean when feature column missing."""
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    y = np.array([10.0, 20.0, 30.0])

    baseline = SeasonalNaiveBaseline(season_lag=24, lag_feature_col="non_existent")
    baseline.fit(df, y)
    preds = baseline.predict(df)

    # When 2D array or missing column, fallback handles appropriately
    assert len(preds) == len(df)
    assert np.all(preds == 20.0)
