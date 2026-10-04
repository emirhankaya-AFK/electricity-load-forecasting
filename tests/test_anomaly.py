"""Tests for residual anomaly detector and deterministic synthetic anomaly evaluation."""

import numpy as np
import pytest

from electricity_load_forecasting.anomaly import ResidualAnomalyDetector


def test_anomaly_detector_fit_and_detect():
    """Verify residual anomaly detector fit and detection."""
    y_val = np.array([2.0, 2.5, 3.0, 3.5, 4.0])
    y_val_pred = np.array([2.0, 2.4, 3.1, 3.4, 4.0])

    detector = ResidualAnomalyDetector(method="quantile", quantile=0.80)
    detector.fit(y_val, y_val_pred)

    assert detector.threshold_ is not None
    assert detector.threshold_ > 0.0

    # Test detection on normal vs anomalous samples
    y_test = np.array([2.0, 10.0])  # second sample has residual ~7.5
    y_test_pred = np.array([2.0, 2.5])

    res = detector.detect(y_test, y_test_pred)
    assert not res.is_anomaly[0]
    assert res.is_anomaly[1]
    assert res.anomaly_count == 1
    assert res.anomaly_rate == 0.5


def test_synthetic_anomaly_benchmark_labeling():
    """Verify synthetic benchmark is explicitly and unambiguously labeled as synthetic."""
    rng = np.random.default_rng(42)
    n = 200
    y_test = rng.uniform(1.0, 4.0, size=n)
    y_test_pred = y_test + rng.normal(0, 0.1, size=n)

    detector = ResidualAnomalyDetector(method="sigma", sigma_multiplier=2.5)
    detector.fit(y_test, y_test_pred)

    benchmark, corrupted_y = detector.evaluate_synthetic_anomalies(
        y_test=y_test,
        y_test_pred=y_test_pred,
        injection_rate=0.05,
        synthetic_magnitude=4.0,
        random_state=42,
    )

    # Strict synthetic labeling assertions
    assert benchmark.is_synthetic_benchmark is True
    assert benchmark.benchmark_type == "SYNTHETIC_INJECTED_EVALUATION"
    assert benchmark.injected_count == 10
    assert 0.0 <= benchmark.precision <= 1.0
    assert 0.0 <= benchmark.recall <= 1.0
    assert 0.0 <= benchmark.f1 <= 1.0
    # For large 4-sigma anomalies, recall should be very high
    assert benchmark.recall >= 0.80


def test_synthetic_anomaly_reproducibility():
    """Deterministic random seed must produce identical synthetic evaluation metrics."""
    y_test = np.linspace(1.0, 5.0, 100)
    y_test_pred = y_test + 0.05

    detector = ResidualAnomalyDetector(method="quantile", quantile=0.95).fit(y_test, y_test_pred)

    b1, _ = detector.evaluate_synthetic_anomalies(y_test, y_test_pred, random_state=123)
    b2, _ = detector.evaluate_synthetic_anomalies(y_test, y_test_pred, random_state=123)

    assert b1.injected_count == b2.injected_count
    assert b1.true_positives == b2.true_positives
    assert b1.f1 == b2.f1


def test_unfitted_anomaly_detector_raises():
    """Calling detect before fit must raise RuntimeError."""
    detector = ResidualAnomalyDetector()
    with pytest.raises(RuntimeError, match="not fitted"):
        detector.detect(np.array([1.0]), np.array([1.0]))
