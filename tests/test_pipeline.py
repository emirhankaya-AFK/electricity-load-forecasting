"""End-to-end pipeline integration tests (offline & deterministic)."""

from electricity_load_forecasting.config import AppConfig
from electricity_load_forecasting.data import generate_synthetic_hourly_data
from electricity_load_forecasting.pipeline import PipelineRunner, run_pipeline


def test_pipeline_runner_offline():
    """Verify end-to-end offline pipeline executes successfully and generates all reports."""
    cfg = AppConfig()
    cfg.data.subset_hours = 600  # fast test run
    runner = PipelineRunner(config=cfg)

    results = runner.run(offline_mode=True, save_reports=False)

    # Check primary sections
    assert "dataset_info" in results
    assert "models" in results
    assert "conformal_intervals" in results
    assert "anomaly_detection" in results
    assert "drift_monitoring" in results
    assert "feature_importance_top5" in results

    # Verify models
    assert "seasonal_naive_baseline" in results["models"]
    assert "hist_gradient_boosting" in results["models"]
    hgb_metrics = results["models"]["hist_gradient_boosting"]
    assert hgb_metrics["mae"] > 0.0
    assert hgb_metrics["rmse"] > 0.0

    # Verify conformal intervals
    conf = results["conformal_intervals"]
    assert conf["nominal_coverage"] == 0.90
    assert 0.0 <= conf["empirical_coverage"] <= 1.0
    assert conf["mean_width"] > 0.0

    # Verify anomaly detection
    anom = results["anomaly_detection"]
    assert anom["threshold"] > 0.0
    synth = anom["synthetic_benchmark"]
    assert synth["is_synthetic_benchmark"] is True
    assert synth["benchmark_label"] == "SYNTHETIC_INJECTED_EVALUATION"
    assert synth["f1"] >= 0.0

    # Verify drift
    drift = results["drift_monitoring"]
    assert "total_features" in drift
    assert drift["total_features"] > 0


def test_top_level_run_pipeline():
    """Verify top-level convenience runner function."""
    results = run_pipeline(offline_mode=True, save_reports=False)
    assert isinstance(results, dict)
    assert results["models"]["hist_gradient_boosting"]["r2"] is not None


def test_pipeline_with_custom_dataframe():
    """Verify pipeline accepts custom DataFrame cleanly."""
    custom_df = generate_synthetic_hourly_data(n_hours=500, random_state=99)
    cfg = AppConfig()
    runner = PipelineRunner(config=cfg)
    results = runner.run(custom_df=custom_df, save_reports=False)

    assert results["dataset_info"]["total_hourly_samples"] == len(custom_df)
    assert results["conformal_intervals"]["empirical_coverage"] > 0.5
