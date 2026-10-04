"""Comprehensive evaluation metrics for point forecasting, intervals, and anomaly detection."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ForecastMetrics:
    mae: float
    rmse: float
    mape: float
    smape: float
    r2: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "mape": round(self.mape, 4),
            "smape": round(self.smape, 4),
            "r2": round(self.r2, 4),
        }


@dataclass
class IntervalMetrics:
    nominal_coverage: float
    empirical_coverage: float
    mean_width: float
    winkler_score: float

    def to_dict(self) -> dict[str, float]:
        return {
            "nominal_coverage": round(self.nominal_coverage, 4),
            "empirical_coverage": round(self.empirical_coverage, 4),
            "mean_width": round(self.mean_width, 4),
            "winkler_score": round(self.winkler_score, 4),
        }


def calculate_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ForecastMetrics:
    """Calculate point forecasting metrics (MAE, RMSE, MAPE, sMAPE, R2)."""
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()

    if len(y_t) != len(y_p):
        raise ValueError("y_true and y_pred must have identical length.")
    if len(y_t) == 0:
        raise ValueError("Inputs cannot be empty.")

    mae = float(np.mean(np.abs(y_t - y_p)))
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))

    # MAPE with epsilon guarding against division by zero
    eps = 1e-4
    mape = float(np.mean(np.abs((y_t - y_p) / np.maximum(np.abs(y_t), eps)))) * 100.0

    # Symmetric MAPE
    denom = (np.abs(y_t) + np.abs(y_p)) / 2.0
    smape = float(np.mean(np.where(denom > eps, np.abs(y_p - y_t) / denom, 0.0))) * 100.0

    # R-squared
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return ForecastMetrics(mae=mae, rmse=rmse, mape=mape, smape=smape, r2=r2)


def calculate_interval_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.10,
) -> IntervalMetrics:
    """Calculate interval evaluation metrics (Coverage, Mean Width, Winkler Score).

    The Winkler Score rewards narrow intervals and penalizes points outside the interval:
    Score = (upper - lower) + (2/alpha) * (lower - y) [if y < lower] + (2/alpha) * (y - upper) [if y > upper]
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    low = np.asarray(lower, dtype=float).ravel()
    up = np.asarray(upper, dtype=float).ravel()

    inside = (y_t >= low) & (y_t <= up)
    empirical_coverage = float(np.mean(inside))
    mean_width = float(np.mean(up - low))

    # Winkler Score calculation
    width = up - low
    lower_penalty = np.where(y_t < low, (2.0 / alpha) * (low - y_t), 0.0)
    upper_penalty = np.where(y_t > up, (2.0 / alpha) * (y_t - up), 0.0)
    winkler = float(np.mean(width + lower_penalty + upper_penalty))

    return IntervalMetrics(
        nominal_coverage=1.0 - alpha,
        empirical_coverage=empirical_coverage,
        mean_width=mean_width,
        winkler_score=winkler,
    )
