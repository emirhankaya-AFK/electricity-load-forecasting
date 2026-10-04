"""FastAPI endpoint smoke and contract tests."""

from fastapi.testclient import TestClient


def test_api_health_endpoint(api_test_client: TestClient):
    """GET /health returns healthy status and active components."""
    response = api_test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_fitted"] is True
    assert data["conformal_calibrated"] is True
    assert data["anomaly_detector_calibrated"] is True
    assert "version" in data


def test_api_forecast_with_features(api_test_client: TestClient):
    """POST /forecast with pre-computed features returns valid forecast and interval."""
    # Dummy feature payload
    features_payload = {
        "lag_1h": 1.5,
        "lag_2h": 1.4,
        "lag_3h": 1.3,
        "lag_24h": 1.6,
        "lag_48h": 1.5,
        "lag_168h": 1.4,
        "rolling_mean_24h": 1.45,
        "rolling_std_24h": 0.2,
        "rolling_mean_168h": 1.42,
        "rolling_std_168h": 0.25,
        "hour": 14,
        "dayofweek": 2,
        "day": 15,
        "month": 6,
        "is_weekend": 0,
        "hour_sin": 0.5,
        "hour_cos": -0.86,
        "dayofweek_sin": 0.97,
        "dayofweek_cos": -0.22,
        "month_sin": 0.5,
        "month_cos": 0.86,
    }

    response = api_test_client.post("/forecast", json={"features": features_payload})
    assert response.status_code == 200
    data = response.json()

    assert "point_forecast" in data
    assert "lower_bound" in data
    assert "upper_bound" in data
    assert data["confidence_level"] == 0.90
    assert data["unit"] == "kW"

    # Interval consistency
    assert data["lower_bound"] <= data["point_forecast"]
    assert data["point_forecast"] <= data["upper_bound"]
    assert data["lower_bound"] >= 0.0


def test_api_forecast_with_recent_history(api_test_client: TestClient):
    """POST /forecast with 168+ recent historical hourly values."""
    history = [1.5 + 0.3 * (i % 24) / 24.0 for i in range(170)]
    payload = {
        "recent_history": history,
        "forecast_timestamp": "2026-10-05T15:00:00",
    }
    response = api_test_client.post("/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["point_forecast"] > 0.0
    assert data["lower_bound"] <= data["upper_bound"]


def test_api_forecast_insufficient_history(api_test_client: TestClient):
    """POST /forecast with insufficient history returns 400."""
    response = api_test_client.post("/forecast", json={"recent_history": [1.0, 2.0, 3.0]})
    assert response.status_code == 400
    assert "requires at least" in response.json()["detail"]


def test_api_forecast_missing_payload(api_test_client: TestClient):
    """POST /forecast without features or history returns 400."""
    response = api_test_client.post("/forecast", json={})
    assert response.status_code == 400


def test_api_anomaly_endpoint(api_test_client: TestClient):
    """POST /anomaly detects abnormal consumption."""
    # Case 1: Normal load
    resp_norm = api_test_client.post("/anomaly", json={"actual_load": 1.2, "predicted_load": 1.15})
    assert resp_norm.status_code == 200
    norm_data = resp_norm.json()
    assert not norm_data["is_anomaly"]
    assert norm_data["residual"] < norm_data["threshold"]

    # Case 2: Extreme spike anomaly
    resp_spike = api_test_client.post("/anomaly", json={"actual_load": 15.0, "predicted_load": 1.0})
    assert resp_spike.status_code == 200
    spike_data = resp_spike.json()
    assert spike_data["is_anomaly"]
    assert spike_data["severity_ratio"] > 1.0


def test_api_drift_endpoint(api_test_client: TestClient):
    """GET /drift retrieves monitoring summary."""
    response = api_test_client.get("/drift")
    assert response.status_code == 200
    data = response.json()
    assert "overall_status" in data or "status" in data
