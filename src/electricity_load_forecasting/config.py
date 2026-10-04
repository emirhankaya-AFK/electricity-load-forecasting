"""Configuration management for electricity load forecasting."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dataset_url: str = (
        "https://archive.ics.uci.edu/static/public/235/"
        "individual+household+electric+power+consumption.zip"
    )
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    target_col: str = "Global_active_power"
    resample_rule: str = "1h"
    subset_hours: int = 17520  # 2 years hourly contiguous subset


@dataclass
class SplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class FeaturesConfig:
    lag_hours: list[int] = field(default_factory=lambda: [1, 2, 3, 24, 48, 168])
    rolling_windows: list[int] = field(default_factory=lambda: [24, 168])
    rolling_stats: list[str] = field(default_factory=lambda: ["mean", "std"])
    include_calendar: bool = True
    include_cyclical: bool = True


@dataclass
class ModelConfig:
    name: str = "HistGradientBoostingRegressor"
    max_iter: int = 150
    max_depth: int = 6
    learning_rate: float = 0.08
    min_samples_leaf: int = 20
    l2_regularization: float = 0.1
    random_state: int = 42


@dataclass
class ConformalConfig:
    alpha: float = 0.10  # 90% confidence interval


@dataclass
class AnomalyConfig:
    threshold_method: str = "validation_quantile"
    quantile: float = 0.99
    sigma_multiplier: float = 3.0
    synthetic_injection_rate: float = 0.03
    synthetic_magnitude: float = 3.5
    random_state: int = 42


@dataclass
class DriftConfig:
    num_bins: int = 10
    warning_threshold: float = 0.10
    alert_threshold: float = 0.25


@dataclass
class ApiConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    title: str = "Electricity Load Forecasting API"
    version: str = "0.1.0"


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    api: ApiConfig = field(default_factory=ApiConfig)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration from YAML file or return defaults."""
    if config_path is None:
        default_yaml = Path("configs/config.yaml")
        if default_yaml.exists():
            config_path = default_yaml
        else:
            return AppConfig()

    path = Path(config_path)
    if not path.exists():
        return AppConfig()

    with open(path, encoding="utf-8") as f:
        raw_cfg: dict[str, Any] = yaml.safe_load(f) or {}

    return AppConfig(
        data=DataConfig(**raw_cfg.get("data", {})),
        split=SplitConfig(**raw_cfg.get("split", {})),
        features=FeaturesConfig(**raw_cfg.get("features", {})),
        model=ModelConfig(**raw_cfg.get("model", {})),
        conformal=ConformalConfig(**raw_cfg.get("conformal", {})),
        anomaly=AnomalyConfig(**raw_cfg.get("anomaly", {})),
        drift=DriftConfig(**raw_cfg.get("drift", {})),
        api=ApiConfig(**raw_cfg.get("api", {})),
    )
