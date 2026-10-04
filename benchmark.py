"""Latency and throughput benchmarking for feature engineering, inference, intervals, and API."""

import time
from typing import Any

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from electricity_load_forecasting.api import create_app
from electricity_load_forecasting.config import load_config
from electricity_load_forecasting.pipeline import PipelineRunner


def compute_latencies_ms(timings_sec: list[float]) -> dict[str, float]:
    """Calculate mean, p50, p95, and p99 in milliseconds."""
    arr = np.asarray(timings_sec) * 1000.0
    return {
        "mean_ms": round(float(np.mean(arr)), 3),
        "p50_ms": round(float(np.median(arr)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
    }


def run_benchmarks(n_iterations: int = 500) -> dict[str, Any]:
    """Execute latency benchmark suite."""
    cfg = load_config()
    runner = PipelineRunner(config=cfg)
    # Train offline model for benchmarking
    runner.run(offline_mode=True, save_reports=False)

    # 1. Feature Engineering Latency (Single-step lookback)
    history_len = runner.feature_engineer.max_history_required + 24
    dummy_history = pd.Series(
        np.random.default_rng(42).uniform(0.5, 3.5, size=history_len),
        index=pd.date_range("2026-01-01", periods=history_len, freq="1h"),
    )
    target_ts = dummy_history.index[-1] + pd.Timedelta(hours=1)

    # Warmup
    for _ in range(20):
        _ = runner.feature_engineer.extract_inference_features(
            dummy_history, forecast_timestamp=target_ts
        )

    feat_times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = runner.feature_engineer.extract_inference_features(
            dummy_history, forecast_timestamp=target_ts
        )
        feat_times.append(time.perf_counter() - t0)

    features_row = runner.feature_engineer.extract_inference_features(
        dummy_history, forecast_timestamp=target_ts
    )

    # 2. Point Prediction Latency
    for _ in range(20):
        _ = runner.forecaster.predict(features_row)

    predict_times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = runner.forecaster.predict(features_row)
        predict_times.append(time.perf_counter() - t0)

    # 3. Conformal Prediction Interval Latency
    point_pred = runner.forecaster.predict(features_row)
    for _ in range(20):
        _ = runner.calibrator.predict_intervals(point_pred)

    conformal_times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = runner.calibrator.predict_intervals(point_pred)
        conformal_times.append(time.perf_counter() - t0)

    # 4. End-to-End Inference + Conformal Latency
    e2e_times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        pred = runner.forecaster.predict(features_row)
        _ = runner.calibrator.predict_intervals(pred)
        e2e_times.append(time.perf_counter() - t0)

    # 5. FastAPI Endpoint Latency (via TestClient)
    app = create_app(runner=runner)
    client = TestClient(app)

    payload = {"features": {col: float(features_row[col].iloc[0]) for col in features_row.columns}}

    # Warmup API
    for _ in range(20):
        _ = client.post("/forecast", json=payload)

    api_forecast_times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        resp = client.post("/forecast", json=payload)
        api_forecast_times.append(time.perf_counter() - t0)
        assert resp.status_code == 200

    # API Anomaly Endpoint Latency
    anomaly_payload = {
        "actual_load": 2.5,
        "predicted_load": 1.1,
    }
    for _ in range(20):
        _ = client.post("/anomaly", json=anomaly_payload)

    api_anomaly_times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        resp = client.post("/anomaly", json=anomaly_payload)
        api_anomaly_times.append(time.perf_counter() - t0)
        assert resp.status_code == 200

    results = {
        "benchmark_iterations": n_iterations,
        "feature_engineering": compute_latencies_ms(feat_times),
        "point_prediction": compute_latencies_ms(predict_times),
        "conformal_intervals": compute_latencies_ms(conformal_times),
        "e2e_predict_and_conformal": compute_latencies_ms(e2e_times),
        "fastapi_post_forecast": compute_latencies_ms(api_forecast_times),
        "fastapi_post_anomaly": compute_latencies_ms(api_anomaly_times),
    }

    return results


if __name__ == "__main__":
    benchmark_results = run_benchmarks(500)
    import json

    print(json.dumps(benchmark_results, indent=2))
