"""Validation-calibrated split conformal prediction intervals."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalPrediction:
    point_forecast: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    confidence_level: float
    quantile_error: float


class ConformalIntervalCalibrator:
    """Split Conformal Prediction Interval Calibrator.

    Computes nonconformity residuals on an independent validation set
    to guarantee finite-sample marginal coverage (1 - alpha).
    """

    def __init__(self, alpha: float = 0.10):
        """Initialize calibrator.

        Args:
            alpha: Significance level (default 0.10 for 90% prediction intervals).
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.quantile_error_: float | None = None
        self.n_calibration_samples_: int = 0

    @property
    def confidence_level(self) -> float:
        return 1.0 - self.alpha

    def calibrate(
        self,
        y_val: np.ndarray,
        y_val_pred: np.ndarray,
    ) -> "ConformalIntervalCalibrator":
        """Calibrate nonconformity threshold using validation set residuals.

        Args:
            y_val: True validation target values.
            y_val_pred: Point forecasts on validation set.

        Returns:
            Fitted calibrator instance.
        """
        y_true = np.asarray(y_val).ravel()
        y_hat = np.asarray(y_val_pred).ravel()

        if len(y_true) != len(y_hat):
            raise ValueError("y_val and y_val_pred must have identical length.")

        n = len(y_true)
        if n == 0:
            raise ValueError("Validation set cannot be empty.")

        self.n_calibration_samples_ = n
        residuals = np.abs(y_true - y_hat)

        # Finite-sample conformal quantile: ceil((n + 1) * (1 - alpha)) / n
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(1.0, level)

        self.quantile_error_ = float(np.quantile(residuals, level, method="higher"))
        return self

    def predict_intervals(
        self,
        y_pred: np.ndarray,
        clip_lower: float = 0.0,
    ) -> ConformalPrediction:
        """Construct conformal prediction intervals around point forecasts.

        Args:
            y_pred: Array of point predictions.
            clip_lower: Lower bound clipping value (default 0.0 for electricity load).

        Returns:
            ConformalPrediction containing point forecast, lower bound, and upper bound.
        """
        if self.quantile_error_ is None:
            raise RuntimeError("Calibrator is not calibrated yet. Call calibrate() first.")

        preds = np.asarray(y_pred)
        lower = preds - self.quantile_error_
        if clip_lower is not None:
            lower = np.clip(lower, clip_lower, None)

        upper = preds + self.quantile_error_

        return ConformalPrediction(
            point_forecast=preds,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=self.confidence_level,
            quantile_error=self.quantile_error_,
        )

    @staticmethod
    def compute_coverage(
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> float:
        """Calculate empirical coverage percentage of true values within intervals."""
        y = np.asarray(y_true).ravel()
        low = np.asarray(lower).ravel()
        up = np.asarray(upper).ravel()
        inside = (y >= low) & (y <= up)
        return float(np.mean(inside))

    @staticmethod
    def compute_mean_width(
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> float:
        """Calculate average width of the prediction intervals."""
        low = np.asarray(lower).ravel()
        up = np.asarray(upper).ravel()
        return float(np.mean(up - low))

    def to_dict(self) -> dict[str, float | int]:
        """Serialize calibrator state."""
        return {
            "alpha": self.alpha,
            "confidence_level": self.confidence_level,
            "quantile_error": self.quantile_error_ if self.quantile_error_ is not None else 0.0,
            "n_calibration_samples": self.n_calibration_samples_,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float | int]) -> "ConformalIntervalCalibrator":
        """Reconstruct calibrator from dictionary."""
        instance = cls(alpha=float(data["alpha"]))
        instance.quantile_error_ = float(data["quantile_error"])
        instance.n_calibration_samples_ = int(data.get("n_calibration_samples", 0))
        return instance
