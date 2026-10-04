"""Leakage-safe feature engineering: lags, rolling statistics, calendar, and cyclical features."""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Builds lag, rolling, calendar, and cyclical features safely without target leakage."""

    def __init__(
        self,
        lag_hours: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        rolling_stats: list[str] | None = None,
        include_calendar: bool = True,
        include_cyclical: bool = True,
    ):
        self.lag_hours = sorted(lag_hours or [1, 2, 3, 24, 48, 168])
        self.rolling_windows = sorted(rolling_windows or [24, 168])
        self.rolling_stats = rolling_stats or ["mean", "std"]
        self.include_calendar = include_calendar
        self.include_cyclical = include_cyclical
        self.feature_names: list[str] = []

    @property
    def max_history_required(self) -> int:
        """Maximum lookback hours required to compute all lag and rolling features."""
        max_lag = max(self.lag_hours) if self.lag_hours else 0
        max_roll = max(self.rolling_windows) if self.rolling_windows else 0
        return max(max_lag, max_roll + 1)

    def create_features(
        self,
        df: pd.DataFrame,
        target_col: str = "Global_active_power",
        drop_na: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Generate full feature matrix and target series from time series dataframe.

        Strict leakage safety:
        - Lag features: shift(k) where k >= 1 strictly queries past values.
        - Rolling features: shift(1).rolling(w) ensures the current observation y_t is never in the window.

        Args:
            df: Cleaned DataFrame with DatetimeIndex and target column.
            target_col: Target column name.
            drop_na: If True, drop the initial warmup rows where lags/rolling stats are NaN.

        Returns:
            Tuple of (X feature DataFrame, y target Series).
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex.")

        series = df[target_col]
        features = pd.DataFrame(index=df.index)

        # 1. Lag features: y_{t-k}
        for lag in self.lag_hours:
            features[f"lag_{lag}h"] = series.shift(lag)

        # 2. Rolling features on shifted series (shift(1) avoids current step leakage)
        shifted_series = series.shift(1)
        for window in self.rolling_windows:
            rolling_obj = shifted_series.rolling(window=window, min_periods=window)
            if "mean" in self.rolling_stats:
                features[f"rolling_mean_{window}h"] = rolling_obj.mean()
            if "std" in self.rolling_stats:
                features[f"rolling_std_{window}h"] = rolling_obj.std()

        # 3. Calendar features
        if self.include_calendar:
            features["hour"] = df.index.hour
            features["dayofweek"] = df.index.dayofweek
            features["day"] = df.index.day
            features["month"] = df.index.month
            features["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

        # 4. Cyclical sine/cosine features
        if self.include_cyclical:
            hour = df.index.hour.to_numpy()
            dow = df.index.dayofweek.to_numpy()
            month = df.index.month.to_numpy()

            features["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
            features["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
            features["dayofweek_sin"] = np.sin(2 * np.pi * dow / 7.0)
            features["dayofweek_cos"] = np.cos(2 * np.pi * dow / 7.0)
            features["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
            features["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)

        # Target series
        target = series.copy()

        if drop_na:
            valid_mask = ~features.isna().any(axis=1) & ~target.isna()
            features = features.loc[valid_mask]
            target = target.loc[valid_mask]

        self.feature_names = list(features.columns)
        return features, target

    def extract_inference_features(
        self,
        historical_series: pd.Series,
        forecast_timestamp: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Generate a single-row feature vector for forecasting the next hour.

        Args:
            historical_series: Recent historical target values (length >= max_history_required).
            forecast_timestamp: Timestamp of the hour being forecast. If None, infers from history.

        Returns:
            Single-row DataFrame matching the model's feature structure.
        """
        min_required = self.max_history_required
        if len(historical_series) < min_required:
            raise ValueError(
                f"Historical series length ({len(historical_series)}) is less than required ({min_required})."
            )

        if forecast_timestamp is None:
            forecast_timestamp = historical_series.index[-1] + pd.Timedelta(hours=1)

        row_dict: dict[str, float | int] = {}

        # 1. Lag features
        for lag in self.lag_hours:
            row_dict[f"lag_{lag}h"] = float(historical_series.iloc[-lag])

        # 2. Rolling features (using history strictly up to t-1)
        for window in self.rolling_windows:
            window_slice = historical_series.iloc[-window:]
            if "mean" in self.rolling_stats:
                row_dict[f"rolling_mean_{window}h"] = float(window_slice.mean())
            if "std" in self.rolling_stats:
                row_dict[f"rolling_std_{window}h"] = float(window_slice.std())

        # 3. Calendar features
        if self.include_calendar:
            row_dict["hour"] = int(forecast_timestamp.hour)
            row_dict["dayofweek"] = int(forecast_timestamp.dayofweek)
            row_dict["day"] = int(forecast_timestamp.day)
            row_dict["month"] = int(forecast_timestamp.month)
            row_dict["is_weekend"] = int(forecast_timestamp.dayofweek >= 5)

        # 4. Cyclical features
        if self.include_cyclical:
            hour = forecast_timestamp.hour
            dow = forecast_timestamp.dayofweek
            month = forecast_timestamp.month
            row_dict["hour_sin"] = float(np.sin(2 * np.pi * hour / 24.0))
            row_dict["hour_cos"] = float(np.cos(2 * np.pi * hour / 24.0))
            row_dict["dayofweek_sin"] = float(np.sin(2 * np.pi * dow / 7.0))
            row_dict["dayofweek_cos"] = float(np.cos(2 * np.pi * dow / 7.0))
            row_dict["month_sin"] = float(np.sin(2 * np.pi * (month - 1) / 12.0))
            row_dict["month_cos"] = float(np.cos(2 * np.pi * (month - 1) / 12.0))

        feat_df = pd.DataFrame([row_dict], index=[forecast_timestamp])
        if self.feature_names:
            feat_df = feat_df[self.feature_names]
        return feat_df
