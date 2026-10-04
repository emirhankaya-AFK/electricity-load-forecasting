"""Shared pytest fixtures."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from electricity_load_forecasting.api import create_app
from electricity_load_forecasting.config import AppConfig
from electricity_load_forecasting.data import generate_synthetic_hourly_data
from electricity_load_forecasting.features import FeatureEngineer
from electricity_load_forecasting.pipeline import PipelineRunner
from electricity_load_forecasting.split import chronological_split


@pytest.fixture(scope="session")
def sample_hourly_df() -> pd.DataFrame:
    """Generate 1000 hours of deterministic synthetic load data."""
    return generate_synthetic_hourly_data(
        n_hours=1000,
        start_date="2026-01-01 00:00:00",
        random_state=42,
    )


@pytest.fixture(scope="session")
def sample_features_and_target(sample_hourly_df: pd.DataFrame):
    """Generate feature matrix and target from sample hourly data."""
    engineer = FeatureEngineer(
        lag_hours=[1, 2, 3, 24, 48, 168],
        rolling_windows=[24, 168],
        rolling_stats=["mean", "std"],
    )
    X, y = engineer.create_features(sample_hourly_df)
    return engineer, X, y


@pytest.fixture(scope="session")
def split_data(sample_features_and_target):
    """Create chronological split."""
    _, X, y = sample_features_and_target
    combined = X.copy()
    combined["target"] = y
    splits = chronological_split(combined, 0.70, 0.15, 0.15)
    return splits


@pytest.fixture(scope="session")
def fitted_pipeline_runner(sample_hourly_df: pd.DataFrame) -> PipelineRunner:
    """Create and run pipeline runner with synthetic data."""
    cfg = AppConfig()
    runner = PipelineRunner(config=cfg)
    runner.run(offline_mode=True, custom_df=sample_hourly_df, save_reports=False)
    return runner


@pytest.fixture(scope="session")
def api_test_client(fitted_pipeline_runner: PipelineRunner) -> TestClient:
    """FastAPI TestClient with pre-fitted runner."""
    app = create_app(runner=fitted_pipeline_runner)
    return TestClient(app)
