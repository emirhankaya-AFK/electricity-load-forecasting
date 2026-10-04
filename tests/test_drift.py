"""Tests for Population Stability Index (PSI) drift monitoring."""

import numpy as np
import pandas as pd

from electricity_load_forecasting.drift import DriftMonitor, calculate_psi


def test_psi_identical_distributions():
    """Identical distributions should yield PSI close to 0."""
    rng = np.random.default_rng(42)
    ref = rng.normal(10.0, 2.0, size=1000)
    target = rng.normal(10.0, 2.0, size=1000)

    psi = calculate_psi(ref, target, num_bins=10)
    assert psi < 0.05, f"Expected near-zero PSI, got {psi}"


def test_psi_significant_shift():
    """Significantly shifted distribution should yield PSI >= 0.25."""
    rng = np.random.default_rng(42)
    ref = rng.normal(10.0, 2.0, size=1000)
    target = rng.normal(18.0, 2.0, size=1000)  # Huge shift of 4 std

    psi = calculate_psi(ref, target, num_bins=10)
    assert psi >= 0.25, f"Expected significant drift (>= 0.25), got {psi}"


def test_drift_monitor_feature_evaluation():
    """DriftMonitor evaluates multiple features and tags status properly."""
    rng = np.random.default_rng(42)
    n = 500

    df_ref = pd.DataFrame(
        {
            "stable_feat": rng.normal(0, 1, size=n),
            "shifted_feat": rng.normal(0, 1, size=n),
        }
    )

    df_curr = pd.DataFrame(
        {
            "stable_feat": rng.normal(0, 1, size=n),
            "shifted_feat": rng.normal(3.5, 1, size=n),
        }
    )

    monitor = DriftMonitor(num_bins=10, warning_threshold=0.10, alert_threshold=0.25)
    metrics = monitor.evaluate_drift(df_ref, df_curr)

    assert len(metrics) == 2
    metric_map = {m.feature_name: m for m in metrics}

    assert metric_map["stable_feat"].status == "stable"
    assert metric_map["shifted_feat"].status == "significant_drift"
    assert metric_map["shifted_feat"].psi > 0.25


def test_drift_summary_report():
    """Summarize report creates proper rollup dictionary."""
    monitor = DriftMonitor()
    summary = monitor.summarize_report([])
    assert summary["status"] == "no_data"

    rng = np.random.default_rng(42)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 100)})
    curr = pd.DataFrame({"a": rng.normal(0, 1, 100)})

    metrics = monitor.evaluate_drift(ref, curr)
    summary = monitor.summarize_report(metrics)

    assert summary["total_features"] == 1
    assert summary["overall_status"] == "stable"
    assert "mean_psi" in summary
