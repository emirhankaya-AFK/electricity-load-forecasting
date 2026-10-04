"""Tests for validation-calibrated conformal prediction intervals."""

import numpy as np
import pytest

from electricity_load_forecasting.intervals import ConformalIntervalCalibrator


def test_conformal_calibration_basic():
    """Verify basic conformal quantile calibration."""
    calibrator = ConformalIntervalCalibrator(alpha=0.10)
    assert calibrator.confidence_level == 0.90

    y_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_val_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.0])
    # Residuals: [0.1, 0.1, 0.2, 0.2, 0.0]

    calibrator.calibrate(y_val, y_val_pred)
    assert calibrator.quantile_error_ is not None
    assert calibrator.quantile_error_ > 0.0
    assert calibrator.n_calibration_samples_ == 5


def test_conformal_coverage_on_calibration():
    """Conformal prediction on calibration set must cover at least 1 - alpha."""
    rng = np.random.default_rng(42)
    n = 200
    y_val = rng.uniform(1.0, 5.0, size=n)
    y_val_pred = y_val + rng.normal(0, 0.2, size=n)

    alpha = 0.10
    calibrator = ConformalIntervalCalibrator(alpha=alpha).calibrate(y_val, y_val_pred)

    pred_res = calibrator.predict_intervals(y_val_pred)
    coverage = calibrator.compute_coverage(y_val, pred_res.lower_bound, pred_res.upper_bound)

    # Empirical coverage on calibration set must be >= nominal (1 - alpha)
    assert coverage >= (1.0 - alpha)


def test_conformal_bounds_ordering():
    """Verify lower <= point_pred <= upper for all samples."""
    calibrator = ConformalIntervalCalibrator(alpha=0.10)
    calibrator.quantile_error_ = 0.5
    calibrator.n_calibration_samples_ = 100

    preds = np.array([1.5, 2.0, 0.3, 10.0])
    intervals = calibrator.predict_intervals(preds, clip_lower=0.0)

    assert np.all(intervals.lower_bound <= intervals.point_forecast)
    assert np.all(intervals.point_forecast <= intervals.upper_bound)
    assert np.all(intervals.lower_bound >= 0.0)


def test_conformal_serialization_roundtrip():
    """Verify serialization to/from dictionary."""
    calibrator = ConformalIntervalCalibrator(alpha=0.05)
    calibrator.quantile_error_ = 0.4285
    calibrator.n_calibration_samples_ = 500

    data = calibrator.to_dict()
    restored = ConformalIntervalCalibrator.from_dict(data)

    assert restored.alpha == 0.05
    assert restored.confidence_level == 0.95
    assert np.isclose(restored.quantile_error_, 0.4285)
    assert restored.n_calibration_samples_ == 500


def test_invalid_alpha_raises():
    """Alpha outside (0, 1) should raise ValueError."""
    with pytest.raises(ValueError):
        ConformalIntervalCalibrator(alpha=0.0)
    with pytest.raises(ValueError):
        ConformalIntervalCalibrator(alpha=1.5)


def test_uncalibrated_predict_raises():
    """Predicting before calibration should raise RuntimeError."""
    calibrator = ConformalIntervalCalibrator(alpha=0.10)
    with pytest.raises(RuntimeError, match="not calibrated"):
        calibrator.predict_intervals(np.array([1.0, 2.0]))
