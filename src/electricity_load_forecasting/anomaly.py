"""Residual-based anomaly detection and deterministic synthetic anomaly evaluation."""

from dataclasses import dataclass

import numpy as np


@dataclass
class AnomalyDetectionResult:
    is_anomaly: np.ndarray
    residuals: np.ndarray
    threshold: float
    anomaly_count: int
    anomaly_rate: float


@dataclass
class SyntheticAnomalyBenchmark:
    # Explicitly labeled synthetic evaluation metadata
    benchmark_type: str = "SYNTHETIC_INJECTED_EVALUATION"
    is_synthetic_benchmark: bool = True
    injected_count: int = 0
    detected_count: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    injection_rate: float = 0.0
    threshold: float = 0.0


class ResidualAnomalyDetector:
    """Detects anomalies in electricity consumption based on model prediction residuals."""

    def __init__(
        self,
        method: str = "quantile",
        quantile: float = 0.99,
        sigma_multiplier: float = 3.0,
    ):
        self.method = method
        self.quantile = quantile
        self.sigma_multiplier = sigma_multiplier
        self.threshold_: float | None = None
        self.mean_residual_: float | None = None
        self.std_residual_: float | None = None

    def fit(self, y_val: np.ndarray, y_val_pred: np.ndarray) -> "ResidualAnomalyDetector":
        """Calibrate anomaly threshold using validation residuals.

        Args:
            y_val: Ground truth validation values.
            y_val_pred: Model predictions on validation set.

        Returns:
            Fitted detector.
        """
        y_true = np.asarray(y_val).ravel()
        y_hat = np.asarray(y_val_pred).ravel()
        residuals = np.abs(y_true - y_hat)

        self.mean_residual_ = float(np.mean(residuals))
        self.std_residual_ = float(np.std(residuals))

        if self.method in ("quantile", "validation_quantile"):
            self.threshold_ = float(np.quantile(residuals, self.quantile))
        elif self.method in ("sigma", "validation_sigma"):
            self.threshold_ = float(
                self.mean_residual_ + self.sigma_multiplier * self.std_residual_
            )
        else:
            raise ValueError(f"Unknown anomaly threshold method: {self.method}")

        return self

    def detect(self, y_true: np.ndarray, y_pred: np.ndarray) -> AnomalyDetectionResult:
        """Detect anomalies from observed and predicted time series.

        Args:
            y_true: Observed electricity load values.
            y_pred: Predicted electricity load values.

        Returns:
            AnomalyDetectionResult containing boolean mask, residuals, and stats.
        """
        if self.threshold_ is None:
            raise RuntimeError("Anomaly detector is not fitted. Call fit() first.")

        y_t = np.asarray(y_true).ravel()
        y_p = np.asarray(y_pred).ravel()
        residuals = np.abs(y_t - y_p)

        is_anomaly = residuals > self.threshold_
        count = int(np.sum(is_anomaly))
        rate = float(count / len(residuals)) if len(residuals) > 0 else 0.0

        return AnomalyDetectionResult(
            is_anomaly=is_anomaly,
            residuals=residuals,
            threshold=self.threshold_,
            anomaly_count=count,
            anomaly_rate=rate,
        )

    def evaluate_synthetic_anomalies(
        self,
        y_test: np.ndarray,
        y_test_pred: np.ndarray,
        injection_rate: float = 0.03,
        synthetic_magnitude: float = 3.5,
        random_state: int = 42,
    ) -> tuple[SyntheticAnomalyBenchmark, np.ndarray]:
        """Evaluate detector performance using deterministic synthetic anomaly injection.

        IMPORTANT: Clearly labeled as SYNTHETIC evaluation to avoid conflation
        with natural operational abnormalities.

        Args:
            y_test: Ground truth test values.
            y_test_pred: Model predictions on test set.
            injection_rate: Fraction of test samples to corrupt with synthetic anomalies.
            synthetic_magnitude: Multiplier on residual std to create synthetic spikes/drops.
            random_state: Deterministic random seed.

        Returns:
            Tuple of (SyntheticAnomalyBenchmark, corrupted_y_test).
        """
        if self.threshold_ is None or self.std_residual_ is None:
            raise RuntimeError("Detector must be fitted before running evaluation.")

        rng = np.random.default_rng(random_state)
        n = len(y_test)
        y_corrupted = np.array(y_test, copy=True, dtype=float).ravel()

        # Deterministically select injection indices
        n_injected = max(1, int(n * injection_rate))
        injected_indices = rng.choice(n, size=n_injected, replace=False)
        injected_mask = np.zeros(n, dtype=bool)
        injected_mask[injected_indices] = True

        # Inject synthetic perturbations (alternating spikes and sudden drops)
        perturbation = synthetic_magnitude * max(self.threshold_, self.std_residual_ * 2.0)
        signs = rng.choice([-1.0, 1.0], size=n_injected)

        for idx, sign in zip(injected_indices, signs, strict=False):
            if sign > 0:
                y_corrupted[idx] += perturbation
            else:
                y_corrupted[idx] = max(0.0, y_corrupted[idx] - perturbation)

        # Detect anomalies on corrupted series
        res = self.detect(y_corrupted, y_test_pred)
        detected_mask = res.is_anomaly

        # Compute confusion metrics against synthetic ground truth
        tp = int(np.sum(injected_mask & detected_mask))
        fp = int(np.sum((~injected_mask) & detected_mask))
        fn = int(np.sum(injected_mask & (~detected_mask)))
        tn = int(np.sum((~injected_mask) & (~detected_mask)))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (
            float(2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

        benchmark = SyntheticAnomalyBenchmark(
            benchmark_type="SYNTHETIC_INJECTED_EVALUATION",
            is_synthetic_benchmark=True,
            injected_count=n_injected,
            detected_count=int(np.sum(detected_mask)),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            precision=precision,
            recall=recall,
            f1=f1,
            injection_rate=injection_rate,
            threshold=self.threshold_,
        )

        return benchmark, y_corrupted
