"""Population Stability Index (PSI) drift monitoring for features and predictions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PSIMetric:
    feature_name: str
    psi: float
    status: str  # "stable", "moderate_drift", "significant_drift"
    reference_count: int
    target_count: int


def calculate_psi(
    reference: np.ndarray,
    target: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between reference and target distributions.

    PSI Formula:
        PSI = sum((target_pct - ref_pct) * ln(target_pct / ref_pct))

    Rule of thumb:
        PSI < 0.10: No significant distribution change (stable)
        0.10 <= PSI < 0.25: Moderate distribution shift
        PSI >= 0.25: Significant distribution shift

    Args:
        reference: Baseline/reference sample array (e.g. training data).
        target: Comparison sample array (e.g. test or production data).
        num_bins: Number of quantile bins to form from reference.
        epsilon: Small smoothing constant to avoid log(0) and division by zero.

    Returns:
        Scalar PSI value.
    """
    ref = np.asarray(reference, dtype=float).ravel()
    tar = np.asarray(target, dtype=float).ravel()

    ref = ref[~np.isnan(ref)]
    tar = tar[~np.isnan(tar)]

    if len(ref) == 0 or len(tar) == 0:
        return 0.0

    # Handle constant series
    if np.all(ref == ref[0]):
        if np.all(tar == ref[0]):
            return 0.0
        return 1.0

    # Adaptively bound bins for small sample sizes to avoid empty-bin artifacts
    min_samples = min(len(ref), len(tar))
    actual_bins = min(num_bins, max(2, min_samples // 20))

    # Determine quantile breakpoints from reference distribution
    percentiles = np.linspace(0, 100, actual_bins + 1)
    bin_edges = np.percentile(ref, percentiles)

    # Ensure strictly increasing bins
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    # Extend edge boundaries to include all possible values
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Bin counts
    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    tar_counts, _ = np.histogram(tar, bins=bin_edges)
    n_bins_effective = len(ref_counts)

    # Convert to proportions with Laplace smoothing
    ref_pct = (ref_counts + 1.0) / (len(ref) + n_bins_effective)
    tar_pct = (tar_counts + 1.0) / (len(tar) + n_bins_effective)

    # PSI calculation
    psi_val = np.sum((tar_pct - ref_pct) * np.log(tar_pct / ref_pct))
    return float(np.maximum(0.0, psi_val))


class DriftMonitor:
    """Monitors feature and prediction distribution stability between train and serving data."""

    def __init__(
        self,
        num_bins: int = 10,
        warning_threshold: float = 0.10,
        alert_threshold: float = 0.25,
    ):
        self.num_bins = num_bins
        self.warning_threshold = warning_threshold
        self.alert_threshold = alert_threshold

    def evaluate_drift(
        self,
        reference_data: pd.DataFrame | dict[str, np.ndarray],
        current_data: pd.DataFrame | dict[str, np.ndarray],
    ) -> list[PSIMetric]:
        """Compute PSI for all common columns between reference and current data.

        Args:
            reference_data: Reference DataFrame or dict of arrays (e.g. train features).
            current_data: Current DataFrame or dict of arrays (e.g. test features).

        Returns:
            List of PSIMetric objects.
        """
        ref_keys = (
            reference_data.columns
            if isinstance(reference_data, pd.DataFrame)
            else list(reference_data.keys())
        )
        curr_keys = (
            current_data.columns
            if isinstance(current_data, pd.DataFrame)
            else list(current_data.keys())
        )

        common_keys = [k for k in ref_keys if k in curr_keys]
        metrics: list[PSIMetric] = []

        for key in common_keys:
            ref_arr = (
                reference_data[key].to_numpy()
                if isinstance(reference_data, pd.DataFrame)
                else reference_data[key]
            )
            curr_arr = (
                current_data[key].to_numpy()
                if isinstance(current_data, pd.DataFrame)
                else current_data[key]
            )

            psi = calculate_psi(ref_arr, curr_arr, num_bins=self.num_bins)

            if psi < self.warning_threshold:
                status = "stable"
            elif psi < self.alert_threshold:
                status = "moderate_drift"
            else:
                status = "significant_drift"

            metrics.append(
                PSIMetric(
                    feature_name=key,
                    psi=psi,
                    status=status,
                    reference_count=len(ref_arr),
                    target_count=len(curr_arr),
                )
            )

        return metrics

    def summarize_report(self, metrics: list[PSIMetric]) -> dict[str, object]:
        """Summarize drift status across all evaluated features."""
        if not metrics:
            return {"total_features": 0, "status": "no_data"}

        psi_values = [m.psi for m in metrics]
        stable_count = sum(1 for m in metrics if m.status == "stable")
        moderate_count = sum(1 for m in metrics if m.status == "moderate_drift")
        alert_count = sum(1 for m in metrics if m.status == "significant_drift")

        return {
            "total_features": len(metrics),
            "mean_psi": float(np.mean(psi_values)),
            "max_psi": float(np.max(psi_values)),
            "stable_count": stable_count,
            "moderate_drift_count": moderate_count,
            "significant_drift_count": alert_count,
            "overall_status": (
                "significant_drift"
                if alert_count > 0
                else ("moderate_drift" if moderate_count > 0 else "stable")
            ),
            "features": [
                {
                    "feature": m.feature_name,
                    "psi": round(m.psi, 4),
                    "status": m.status,
                }
                for m in sorted(metrics, key=lambda x: x.psi, reverse=True)
            ],
        }
