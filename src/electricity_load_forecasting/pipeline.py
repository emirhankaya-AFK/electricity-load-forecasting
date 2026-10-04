"""End-to-end execution pipeline for electricity load forecasting."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from electricity_load_forecasting.anomaly import ResidualAnomalyDetector
from electricity_load_forecasting.cleaning import clean_hourly_data
from electricity_load_forecasting.config import AppConfig, load_config
from electricity_load_forecasting.data import (
    download_dataset,
    generate_synthetic_hourly_data,
    load_raw_data,
    resample_hourly,
)
from electricity_load_forecasting.drift import DriftMonitor
from electricity_load_forecasting.features import FeatureEngineer
from electricity_load_forecasting.intervals import ConformalIntervalCalibrator
from electricity_load_forecasting.metrics import (
    calculate_forecast_metrics,
    calculate_interval_metrics,
)
from electricity_load_forecasting.models import LoadForecaster, SeasonalNaiveBaseline
from electricity_load_forecasting.split import chronological_split

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrates end-to-end load forecasting, intervals, anomalies, and drift monitoring."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self.feature_engineer = FeatureEngineer(
            lag_hours=self.config.features.lag_hours,
            rolling_windows=self.config.features.rolling_windows,
            rolling_stats=self.config.features.rolling_stats,
            include_calendar=self.config.features.include_calendar,
            include_cyclical=self.config.features.include_cyclical,
        )
        self.baseline_model = SeasonalNaiveBaseline(season_lag=24)
        self.forecaster = LoadForecaster(
            max_iter=self.config.model.max_iter,
            max_depth=self.config.model.max_depth,
            learning_rate=self.config.model.learning_rate,
            min_samples_leaf=self.config.model.min_samples_leaf,
            l2_regularization=self.config.model.l2_regularization,
            random_state=self.config.model.random_state,
        )
        self.calibrator = ConformalIntervalCalibrator(alpha=self.config.conformal.alpha)
        self.anomaly_detector = ResidualAnomalyDetector(
            method=self.config.anomaly.threshold_method,
            quantile=self.config.anomaly.quantile,
            sigma_multiplier=self.config.anomaly.sigma_multiplier,
        )
        self.drift_monitor = DriftMonitor(
            num_bins=self.config.drift.num_bins,
            warning_threshold=self.config.drift.warning_threshold,
            alert_threshold=self.config.drift.alert_threshold,
        )
        self.latest_results: dict[str, Any] = {}

    def load_data(
        self, offline_mode: bool = False, custom_df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Acquire and hourly-resample data, with fallback to deterministic synthetic data."""
        if custom_df is not None:
            return custom_df

        if offline_mode:
            logger.info("Running in offline mode; generating deterministic synthetic load data.")
            return generate_synthetic_hourly_data(
                n_hours=self.config.data.subset_hours,
                target_col=self.config.data.target_col,
            )

        try:
            raw_file = download_dataset(
                url=self.config.data.dataset_url,
                dest_dir=self.config.data.raw_dir,
            )
            raw_df = load_raw_data(raw_file)
            hourly_df = resample_hourly(
                raw_df,
                target_col=self.config.data.target_col,
                subset_hours=self.config.data.subset_hours,
            )
            return hourly_df
        except Exception as e:
            logger.warning(
                "Live data acquisition failed (%s). Falling back to synthetic series.", e
            )
            return generate_synthetic_hourly_data(
                n_hours=self.config.data.subset_hours,
                target_col=self.config.data.target_col,
            )

    def run(
        self,
        offline_mode: bool = False,
        custom_df: pd.DataFrame | None = None,
        save_reports: bool = True,
    ) -> dict[str, Any]:
        """Execute the full machine learning and monitoring pipeline."""
        logger.info("Step 1: Ingesting dataset...")
        raw_hourly = self.load_data(offline_mode=offline_mode, custom_df=custom_df)

        logger.info("Step 2: Cleaning and validating time series...")
        clean_df, clean_report = clean_hourly_data(
            raw_hourly,
            target_col=self.config.data.target_col,
        )

        logger.info("Step 3: Engineering leakage-safe lag, rolling, and calendar features...")
        X, y = self.feature_engineer.create_features(
            clean_df, target_col=self.config.data.target_col
        )

        logger.info("Step 4: Splitting chronologically (train/val/test)...")
        combined = X.copy()
        combined["__target__"] = y
        split_data = chronological_split(
            combined,
            train_ratio=self.config.split.train_ratio,
            val_ratio=self.config.split.val_ratio,
            test_ratio=self.config.split.test_ratio,
        )

        X_train = split_data.train.drop(columns=["__target__"])
        y_train = split_data.train["__target__"]

        X_val = split_data.val.drop(columns=["__target__"])
        y_val = split_data.val["__target__"]

        X_test = split_data.test.drop(columns=["__target__"])
        y_test = split_data.test["__target__"]

        logger.info("Step 5: Training seasonal-naive baseline...")
        self.baseline_model.fit(X_train, y_train)
        baseline_test_pred = self.baseline_model.predict(X_test)
        baseline_metrics = calculate_forecast_metrics(y_test.to_numpy(), baseline_test_pred)

        logger.info("Step 6: Fitting HistGradientBoosting point forecaster...")
        self.forecaster.fit(X_train, y_train)
        y_val_pred = self.forecaster.predict(X_val)
        y_test_pred = self.forecaster.predict(X_test)
        hgb_test_metrics = calculate_forecast_metrics(y_test.to_numpy(), y_test_pred)

        logger.info("Step 7: Calibrating split conformal prediction intervals on validation set...")
        self.calibrator.calibrate(y_val.to_numpy(), y_val_pred)
        conformal_test = self.calibrator.predict_intervals(y_test_pred)
        interval_metrics = calculate_interval_metrics(
            y_true=y_test.to_numpy(),
            lower=conformal_test.lower_bound,
            upper=conformal_test.upper_bound,
            alpha=self.config.conformal.alpha,
        )

        logger.info(
            "Step 8: Calibrating residual anomaly detector and evaluating synthetic benchmark..."
        )
        self.anomaly_detector.fit(y_val.to_numpy(), y_val_pred)
        natural_anomalies = self.anomaly_detector.detect(y_test.to_numpy(), y_test_pred)
        synthetic_benchmark, _ = self.anomaly_detector.evaluate_synthetic_anomalies(
            y_test.to_numpy(),
            y_test_pred,
            injection_rate=self.config.anomaly.synthetic_injection_rate,
            synthetic_magnitude=self.config.anomaly.synthetic_magnitude,
            random_state=self.config.anomaly.random_state,
        )

        logger.info("Step 9: Computing Population Stability Index (PSI) drift monitoring...")
        drift_metrics = self.drift_monitor.evaluate_drift(X_train, X_test)
        drift_report = self.drift_monitor.summarize_report(drift_metrics)

        # Feature importances
        importance_dict = self.forecaster.compute_feature_importance(X_val, y_val, n_repeats=3)

        results: dict[str, Any] = {
            "dataset_info": {
                "total_hourly_samples": len(clean_df),
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "train_period": f"{split_data.train_start} to {split_data.train_end}",
                "val_period": f"{split_data.val_start} to {split_data.val_end}",
                "test_period": f"{split_data.test_start} to {split_data.test_end}",
                "missing_imputed": clean_report.missing_imputed,
            },
            "models": {
                "seasonal_naive_baseline": baseline_metrics.to_dict(),
                "hist_gradient_boosting": hgb_test_metrics.to_dict(),
                "improvement_vs_baseline_mae_pct": round(
                    ((baseline_metrics.mae - hgb_test_metrics.mae) / baseline_metrics.mae) * 100.0,
                    2,
                ),
            },
            "conformal_intervals": {
                **interval_metrics.to_dict(),
                "calibrated_quantile_error": round(self.calibrator.quantile_error_ or 0.0, 4),
            },
            "anomaly_detection": {
                "threshold": round(self.anomaly_detector.threshold_ or 0.0, 4),
                "natural_test_anomalies_detected": natural_anomalies.anomaly_count,
                "natural_test_anomaly_rate": round(natural_anomalies.anomaly_rate, 4),
                "synthetic_benchmark": {
                    "is_synthetic_benchmark": True,
                    "benchmark_label": "SYNTHETIC_INJECTED_EVALUATION",
                    "injected_count": synthetic_benchmark.injected_count,
                    "true_positives": synthetic_benchmark.true_positives,
                    "false_positives": synthetic_benchmark.false_positives,
                    "false_negatives": synthetic_benchmark.false_negatives,
                    "precision": round(synthetic_benchmark.precision, 4),
                    "recall": round(synthetic_benchmark.recall, 4),
                    "f1": round(synthetic_benchmark.f1, 4),
                },
            },
            "drift_monitoring": drift_report,
            "feature_importance_top5": dict(list(importance_dict.items())[:5]),
        }

        self.latest_results = results

        if save_reports:
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            report_file = reports_dir / "pipeline_results.json"
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            logger.info("Saved pipeline report to %s", report_file)

        return results


def run_pipeline(
    config_path: str | Path | None = None,
    offline_mode: bool = False,
    save_reports: bool = True,
) -> dict[str, Any]:
    """Top-level pipeline execution function."""
    cfg = load_config(config_path)
    runner = PipelineRunner(config=cfg)
    return runner.run(offline_mode=offline_mode, save_reports=save_reports)
