"""Tests for forecasting and interval evaluation metrics."""

import numpy as np
import pytest

from electricity_load_forecasting.metrics import (
    calculate_forecast_metrics,
    calculate_interval_metrics,
)


def test_calculate_forecast_metrics():
    """Verify point forecast metrics on known vectors."""
    y_true = np.array([2.0, 4.0, 6.0])
    y_pred = np.array([2.5, 3.5, 6.0])
    # Errors: [0.5, -0.5, 0.0]

    metrics = calculate_forecast_metrics(y_true, y_pred)
    assert np.isclose(metrics.mae, 1.0 / 3.0)
    assert np.isclose(metrics.rmse, np.sqrt(0.5 / 3.0))
    assert metrics.r2 > 0.0

    d = metrics.to_dict()
    assert "mae" in d
    assert "rmse" in d
    assert "r2" in d


def test_calculate_interval_metrics():
    """Verify conformal interval metrics and Winkler score."""
    y_true = np.array([2.0, 4.0, 10.0])  # third point is outside [1, 5]
    lower = np.array([1.0, 1.0, 1.0])
    upper = np.array([5.0, 5.0, 5.0])
    alpha = 0.10

    metrics = calculate_interval_metrics(y_true, lower, upper, alpha=alpha)

    # 2 out of 3 inside interval -> coverage = 2/3
    assert np.isclose(metrics.empirical_coverage, 2.0 / 3.0)
    # Width is 4.0 for all
    assert np.isclose(metrics.mean_width, 4.0)
    # Winkler score includes penalty for out-of-bounds point
    assert metrics.winkler_score > metrics.mean_width


def test_metrics_length_mismatch():
    """Mismatched array lengths should raise ValueError."""
    with pytest.raises(ValueError):
        calculate_forecast_metrics(np.array([1.0]), np.array([1.0, 2.0]))
