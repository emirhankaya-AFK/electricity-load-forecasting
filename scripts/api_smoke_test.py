"""API live smoke test script."""

import json
import logging

from fastapi.testclient import TestClient

from electricity_load_forecasting.api import create_app
from electricity_load_forecasting.config import load_config
from electricity_load_forecasting.pipeline import PipelineRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke_test")


def main():
    logger.info("Initializing API runner with latest report data...")
    cfg = load_config()
    runner = PipelineRunner(config=cfg)
    # Run pipeline on clean data
    runner.run(offline_mode=False, save_reports=False)

    app = create_app(runner=runner)
    client = TestClient(app)

    logger.info("Smoke test 1: GET /health")
    r_health = client.get("/health")
    assert r_health.status_code == 200, f"Expected 200, got {r_health.status_code}"
    health_data = r_health.json()
    logger.info("Health response: %s", health_data)
    assert health_data["status"] == "healthy"
    assert health_data["model_fitted"] is True

    logger.info("Smoke test 2: POST /forecast with recent history")
    history_vals = [1.2 + 0.3 * ((i % 24) / 24.0) for i in range(170)]
    r_forecast = client.post(
        "/forecast",
        json={
            "recent_history": history_vals,
            "forecast_timestamp": "2026-10-05T12:00:00",
        },
    )
    assert r_forecast.status_code == 200, (
        f"Expected 200, got {r_forecast.status_code}: {r_forecast.text}"
    )
    forecast_data = r_forecast.json()
    logger.info("Forecast response: %s", json.dumps(forecast_data, indent=2))
    assert (
        forecast_data["lower_bound"]
        <= forecast_data["point_forecast"]
        <= forecast_data["upper_bound"]
    )

    logger.info("Smoke test 3: POST /anomaly (normal vs anomalous)")
    r_normal = client.post("/anomaly", json={"actual_load": 1.25, "predicted_load": 1.20})
    assert r_normal.status_code == 200
    norm_res = r_normal.json()
    logger.info("Normal test: %s", norm_res)
    assert not norm_res["is_anomaly"]

    r_anom = client.post("/anomaly", json={"actual_load": 12.5, "predicted_load": 1.20})
    assert r_anom.status_code == 200
    anom_res = r_anom.json()
    logger.info("Anomaly test: %s", anom_res)
    assert anom_res["is_anomaly"]

    logger.info("Smoke test 4: GET /drift")
    r_drift = client.get("/drift")
    assert r_drift.status_code == 200
    drift_data = r_drift.json()
    logger.info("Drift summary: %s", drift_data.get("overall_status"))

    print("\nALL API SMOKE TESTS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    main()
