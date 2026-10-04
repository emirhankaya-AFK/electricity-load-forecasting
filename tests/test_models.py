"""Tests for LoadForecaster model wrapper."""

import numpy as np
import pandas as pd
import pytest

from electricity_load_forecasting.models import LoadForecaster


def test_load_forecaster_fit_and_predict():
    """Verify HistGradientBoosting wrapper trains and outputs non-negative predictions."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "lag_1h": rng.uniform(0.5, 4.0, size=200),
            "lag_24h": rng.uniform(0.5, 4.0, size=200),
            "hour": rng.integers(0, 24, size=200),
        }
    )
    y = X["lag_1h"] * 0.7 + X["lag_24h"] * 0.3 + rng.normal(0, 0.1, size=200)

    model = LoadForecaster(max_iter=50, random_state=42)
    assert not model.is_fitted_

    model.fit(X, y)
    assert model.is_fitted_
    assert len(model.feature_names_) == 3

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert np.all(preds >= 0.0)  # Power load cannot be negative


def test_load_forecaster_feature_importance():
    """Verify permutation feature importance computes correctly."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "strong_signal": rng.uniform(0, 10, size=200),
            "random_noise": rng.uniform(0, 10, size=200),
        }
    )
    y = X["strong_signal"] * 2.0 + rng.normal(0, 0.01, size=200)

    model = LoadForecaster(max_iter=50, random_state=42).fit(X, y)
    importances = model.compute_feature_importance(X, y, n_repeats=3)

    assert "strong_signal" in importances
    assert "random_noise" in importances
    assert importances["strong_signal"] > importances["random_noise"]


def test_predict_before_fit_raises():
    """Predicting on unfitted forecaster raises RuntimeError."""
    model = LoadForecaster()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(np.array([[1.0, 2.0]]))
