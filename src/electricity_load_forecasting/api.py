"""FastAPI service exposing health, forecasting, conformal intervals, and anomaly endpoints."""

from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from electricity_load_forecasting import __version__
from electricity_load_forecasting.config import load_config
from electricity_load_forecasting.pipeline import PipelineRunner

# Global state holder
_runner: PipelineRunner | None = None


def get_runner() -> PipelineRunner:
    global _runner
    if _runner is None:
        cfg = load_config()
        _runner = PipelineRunner(config=cfg)
        # Initialize with synthetic dataset if not trained yet
        _runner.run(offline_mode=True, save_reports=False)
    return _runner


def set_runner(runner: PipelineRunner) -> None:
    global _runner
    _runner = runner


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure pipeline is initialized
    runner = get_runner()
    if not runner.forecaster.is_fitted_:
        runner.run(offline_mode=True, save_reports=False)
    yield
    # Shutdown logic if any


def create_app(runner: PipelineRunner | None = None) -> FastAPI:
    """Create and configure FastAPI application."""
    if runner is not None:
        set_runner(runner)

    app = FastAPI(
        title="Electricity Load Forecasting API",
        version=__version__,
        description=(
            "CPU-friendly electricity load forecasting service with conformal prediction "
            "intervals, residual anomaly detection, and PSI drift monitoring."
        ),
        lifespan=lifespan,
    )

    class HealthResponse(BaseModel):
        status: str = "healthy"
        version: str
        model_fitted: bool
        conformal_calibrated: bool
        anomaly_detector_calibrated: bool

    class ForecastRequest(BaseModel):
        features: dict[str, float] | None = Field(
            default=None,
            description="Pre-computed feature mapping (e.g. lag_1h, rolling_mean_24h, hour, etc.)",
        )
        recent_history: list[float] | None = Field(
            default=None,
            description="Recent historical hourly load values (at least 168 observations)",
        )
        forecast_timestamp: str | None = Field(
            default=None,
            description="ISO timestamp for the target forecast hour (e.g. '2026-10-05T12:00:00')",
        )

    class ForecastResponse(BaseModel):
        point_forecast: float
        lower_bound: float
        upper_bound: float
        confidence_level: float
        quantile_error: float
        unit: str = "kW"

    class AnomalyRequest(BaseModel):
        actual_load: float = Field(..., description="Observed electricity load (kW)")
        predicted_load: float | None = Field(
            default=None,
            description="Predicted load (kW). If omitted, calculated from features or history.",
        )
        features: dict[str, float] | None = None

    class AnomalyResponse(BaseModel):
        is_anomaly: bool
        residual: float
        threshold: float
        severity_ratio: float
        message: str

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Service health check and model status."""
        active_runner = get_runner()
        return HealthResponse(
            status="healthy",
            version=__version__,
            model_fitted=active_runner.forecaster.is_fitted_,
            conformal_calibrated=active_runner.calibrator.quantile_error_ is not None,
            anomaly_detector_calibrated=active_runner.anomaly_detector.threshold_ is not None,
        )

    @app.post("/forecast", response_model=ForecastResponse)
    async def forecast(req: ForecastRequest) -> ForecastResponse:
        """Generate point forecast with conformal prediction intervals."""
        active_runner = get_runner()
        if not active_runner.forecaster.is_fitted_:
            raise HTTPException(status_code=503, detail="Forecasting model is not ready.")

        # Determine feature row
        if req.features is not None:
            # Build DataFrame from provided feature dict
            expected_feats = active_runner.forecaster.feature_names_
            feat_dict = {f: float(req.features.get(f, 0.0)) for f in expected_feats}
            df_features = pd.DataFrame([feat_dict])
        elif req.recent_history is not None:
            min_req = active_runner.feature_engineer.max_history_required
            if len(req.recent_history) < min_req:
                raise HTTPException(
                    status_code=400,
                    detail=f"recent_history requires at least {min_req} values, got {len(req.recent_history)}.",
                )
            ts = (
                pd.to_datetime(req.forecast_timestamp)
                if req.forecast_timestamp
                else pd.Timestamp.now()
            )
            history_idx = pd.date_range(
                end=ts - pd.Timedelta(hours=1), periods=len(req.recent_history), freq="1h"
            )
            s = pd.Series(req.recent_history, index=history_idx)
            df_features = active_runner.feature_engineer.extract_inference_features(
                s, forecast_timestamp=ts
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'features' dictionary or 'recent_history' array must be provided.",
            )

        # Generate point prediction
        preds = active_runner.forecaster.predict(df_features)
        point_pred = float(preds[0])

        # Generate conformal interval
        conformal_res = active_runner.calibrator.predict_intervals(preds)
        lower = float(conformal_res.lower_bound[0])
        upper = float(conformal_res.upper_bound[0])

        return ForecastResponse(
            point_forecast=round(point_pred, 4),
            lower_bound=round(lower, 4),
            upper_bound=round(upper, 4),
            confidence_level=active_runner.calibrator.confidence_level,
            quantile_error=round(conformal_res.quantile_error, 4),
            unit="kW",
        )

    @app.post("/anomaly", response_model=AnomalyResponse)
    async def anomaly(req: AnomalyRequest) -> AnomalyResponse:
        """Evaluate whether an observed electricity consumption is anomalous."""
        active_runner = get_runner()
        if active_runner.anomaly_detector.threshold_ is None:
            raise HTTPException(status_code=503, detail="Anomaly detector is not calibrated.")

        pred_val = req.predicted_load
        if pred_val is None:
            if req.features is not None:
                expected_feats = active_runner.forecaster.feature_names_
                feat_dict = {f: float(req.features.get(f, 0.0)) for f in expected_feats}
                df_features = pd.DataFrame([feat_dict])
                pred_val = float(active_runner.forecaster.predict(df_features)[0])
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Either predicted_load or features must be provided.",
                )

        detection = active_runner.anomaly_detector.detect(
            np.array([req.actual_load]),
            np.array([pred_val]),
        )

        res = float(detection.residuals[0])
        thresh = float(detection.threshold)
        is_anom = bool(detection.is_anomaly[0])
        severity = float(res / thresh) if thresh > 0 else 1.0

        msg = (
            f"Residual {res:.4f} kW exceeds threshold {thresh:.4f} kW."
            if is_anom
            else f"Residual {res:.4f} kW is within normal threshold {thresh:.4f} kW."
        )

        return AnomalyResponse(
            is_anomaly=is_anom,
            residual=round(res, 4),
            threshold=round(thresh, 4),
            severity_ratio=round(severity, 4),
            message=msg,
        )

    @app.get("/drift")
    async def drift() -> dict[str, Any]:
        """Retrieve recent PSI drift monitoring summary."""
        active_runner = get_runner()
        if not active_runner.latest_results:
            return {"status": "not_executed_yet", "message": "Pipeline has not run yet."}
        return active_runner.latest_results.get("drift_monitoring", {})

    return app


app = create_app()
