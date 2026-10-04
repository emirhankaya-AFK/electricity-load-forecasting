"""Forecasting models: Seasonal-naive baseline and HistGradientBoosting forecaster."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


class SeasonalNaiveBaseline(BaseEstimator, RegressorMixin):
    """Seasonal-naive baseline forecasting model.

    Predicts the load from s periods ago (default s=24 hours for daily seasonality).
    """

    def __init__(self, season_lag: int = 24, lag_feature_col: str = "lag_24h"):
        self.season_lag = season_lag
        self.lag_feature_col = lag_feature_col
        self.last_observed_value: float | None = None

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None
    ) -> "SeasonalNaiveBaseline":
        """Fit baseline by recording fallback load value."""
        if y is not None:
            self.last_observed_value = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict using the seasonal lag column."""
        if isinstance(X, pd.DataFrame) and self.lag_feature_col in X.columns:
            preds = X[self.lag_feature_col].to_numpy()
            if self.last_observed_value is not None:
                preds = np.nan_to_num(preds, nan=self.last_observed_value)
            return preds

        # Fallback if column not present
        if isinstance(X, np.ndarray) and X.ndim == 2:
            return X[:, 0]

        if self.last_observed_value is not None:
            n_samples = len(X)
            return np.full(n_samples, self.last_observed_value)

        raise ValueError("Cannot predict with SeasonalNaiveBaseline: missing input data.")


class LoadForecaster(BaseEstimator, RegressorMixin):
    """Production wrapper around HistGradientBoostingRegressor."""

    def __init__(
        self,
        max_iter: int = 150,
        max_depth: int = 6,
        learning_rate: float = 0.08,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.1,
        random_state: int = 42,
    ):
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state

        self.model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
        )
        self.feature_names_: list[str] = []
        self.is_fitted_: bool = False

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "LoadForecaster":
        """Fit the HistGradientBoosting regressor.

        Args:
            X: Feature matrix.
            y: Target vector.

        Returns:
            Fitted forecaster instance.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)
            self.feature_names_ = [f"f_{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y).ravel()

        logger.info(
            "Fitting HistGradientBoostingRegressor with %d samples and %d features...",
            X_arr.shape[0],
            X_arr.shape[1],
        )
        self.model.fit(X_arr, y_arr)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict electricity load (kW).

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted load values, clipped to non-negative range.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted yet.")

        if isinstance(X, pd.DataFrame):
            if self.feature_names_:
                X = X[self.feature_names_]
            X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)

        preds = self.model.predict(X_arr)
        # Power load is non-negative
        return np.clip(preds, 0.0, None)

    def compute_feature_importance(
        self,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
        n_repeats: int = 5,
    ) -> dict[str, float]:
        """Compute validation permutation feature importances."""
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted yet.")

        if isinstance(X_val, pd.DataFrame):
            feat_names = list(X_val.columns)
            X_arr = X_val.to_numpy()
        else:
            feat_names = self.feature_names_
            X_arr = np.asarray(X_val)

        y_arr = np.asarray(y_val).ravel()

        res = permutation_importance(
            self.model,
            X_arr,
            y_arr,
            n_repeats=n_repeats,
            random_state=self.random_state,
            scoring="neg_mean_absolute_error",
        )

        importances = {
            name: float(imp) for name, imp in zip(feat_names, res.importances_mean, strict=False)
        }
        return dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get estimator parameters."""
        return {
            "max_iter": self.max_iter,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
            "random_state": self.random_state,
        }
