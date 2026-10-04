# Model Card: Electricity Load Forecasting (HistGradientBoosting + Conformal Intervals)

## Model Details
- **Model Name:** Electricity Load Forecaster (`LoadForecaster`)
- **Version:** 0.1.0
- **Model Type:** Histogram-based Gradient Boosting Decision Trees (`HistGradientBoostingRegressor`)
- **Framework:** scikit-learn >= 1.4.0, Python >= 3.11
- **Uncertainty Quantification:** Split Conformal Prediction (`ConformalIntervalCalibrator`)
- **Anomaly Detection:** Validation-calibrated Residual Anomaly Detector (`ResidualAnomalyDetector`)
- **Drift Monitoring:** Population Stability Index (PSI) Feature & Prediction Drift Monitor (`DriftMonitor`)
- **License:** MIT
- **Contact:** Antigravity Engineering Team

---

## Intended Use
- **Primary Use Case:** Hourly household electricity load forecasting (in kilowatts, kW) 1 to 24 steps ahead for energy management systems, peak shaving, battery storage scheduling, and smart meter anomaly detection.
- **Primary Users:** Energy grid operators, building management systems (BMS), home automation platforms, and energy analytics researchers.
- **Out-of-Scope Use Cases:**
  - Real-time protection relay tripping or sub-second power quality transient protection (frequency < 1 second).
  - Multi-year macroeconomic grid load forecasting without macro-covariates.
  - Multi-household aggregation without local transfer calibration.

---

## Dataset & Training Pipeline
- **Dataset:** UCI Machine Learning Repository - *Individual Household Electric Power Consumption*
  - **Source URL:** `https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip`
  - **Characteristics:** Electric power consumption in one household over ~4 years (December 2006 to November 2010), originally sampled at 1-minute intervals.
  - **Resampling:** Resampled to hourly mean consumption (`1h`) to produce a CPU-friendly, contiguous, noise-filtered time series.
  - **Subset:** Practical recent contiguous subset (2 years / 17,520 hours) balancing computational efficiency and seasonal representation (2 complete annual cycles).
- **Target Variable:** `Global_active_power` (household global minute-averaged active power resampled to hourly mean in kW).
- **Leakage Prevention:**
  - Strict chronological temporal train/validation/test split (70% train, 15% validation/calibration, 15% test).
  - No random shuffling.
  - All rolling statistics use explicit `shift(1)` lookback to ensure current target $y_t$ is never within the feature window.
  - All lag features $y_{t-k}$ enforce $k \ge 1$.

---

## Feature Engineering
1. **Lags:** $y_{t-1}, y_{t-2}, y_{t-3}, y_{t-24}, y_{t-48}, y_{t-168}$ capturing immediate autoregression, daily diurnal cycle (24h), and weekly cycle (168h).
2. **Rolling Statistics:** 24-hour and 168-hour rolling mean and standard deviation computed strictly on $y_{t-1}$ backwards.
3. **Calendar Covariates:** Hour of day (0-23), Day of week (0-6), Day of month (1-31), Month (1-12), Weekend indicator (binary).
4. **Cyclical Transforms:** Trigonometric sine and cosine encodings for hour, day of week, and month:
   $$\sin(2\pi \cdot \text{hour} / 24), \quad \cos(2\pi \cdot \text{hour} / 24)$$

---

## Conformal Prediction Intervals
- **Methodology:** Inductive / Split Conformal Prediction.
- **Calibration Set:** Non-overlapping validation partition.
- **Nonconformity Score:** Absolute prediction error $R_i = |y_i - \hat{y}_i|$.
- **Coverage Guarantee:** Given significance level $\alpha = 0.10$, the $(1 - \alpha) = 90\%$ marginal coverage guarantee satisfies:
  $$P(Y_{t} \in [\hat{y}_t - \hat{q}, \hat{y}_t + \hat{q}]) \ge 1 - \alpha$$
  with finite-sample correction $\hat{q} = \text{Quantile}\left(R, \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$.
- **Domain Constraint:** Interval lower bounds are clipped at $0.0\text{ kW}$ as active power is physically non-negative.

---

## Anomaly Detection & Synthetic Evaluation
- **Detector Mechanism:** Flags timestamps where model prediction residual $|y_t - \hat{y}_t| > \tau$, where $\tau$ is calibrated on validation residual quantiles (99th percentile).
- **Synthetic Evaluation Benchmark:**
  > [!NOTE]
  > Because raw historical data lacks verified ground-truth operational fault annotations, detection performance is benchmarked using **deterministic synthetic anomaly injection**. These benchmarks are explicitly labeled synthetic and must not be conflated with natural anomalies.
  - Synthetic injection rate: 3% of test samples corrupted by deterministic spikes and step-downs ($\pm 3.5 \cdot \sigma$).
  - Evaluated on precision, recall, and F1-score against synthetic ground truth.

---

## Distribution Drift Monitoring
- **Metric:** Population Stability Index (PSI) computed on quantile bins:
  $$\text{PSI} = \sum_{i=1}^B (q_i - p_i) \cdot \ln(q_i / p_i)$$
- **Thresholds:**
  - $\text{PSI} < 0.10$: Stable / No significant change.
  - $0.10 \le \text{PSI} < 0.25$: Moderate distribution shift (monitoring alert).
  - $\text{PSI} \ge 0.25$: Significant drift (retraining triggered).

---

## Quantitative Evaluation Results (Empirically Measured)

### 1. Point Forecasting Performance (Test Partition: 2,603 hours)
- **Seasonal-Naive Baseline (24h lag):** MAE = `0.5127 kW` | RMSE = `0.7846 kW` | MAPE = `64.53%` | $R^2 = -0.0949$
- **HistGradientBoosting Forecaster:** MAE = `0.3243 kW` | RMSE = `0.4715 kW` | MAPE = `41.86%` | $R^2 = 0.6045$
- **Performance Gain:** **36.76% reduction in MAE** over seasonal-naive baseline.

### 2. Conformal Prediction Interval Metrics
- **Nominal Coverage ($1 - \alpha$):** 90.00%
- **Empirical Coverage:** **89.17%**
- **Mean Interval Width:** **1.3618 kW**
- **Calibrated Nonconformity Quantile ($\hat{q}$):** **0.7361 kW**
- **Winkler Score:** **2.1176**

### 3. Residual Anomaly Detection
- **Decision Threshold ($\tau$):** `1.5095 kW`
- **Natural Test Set Detections:** 36 events (1.38% rate)
- **Deterministic Synthetic Injection Benchmark (`SYNTHETIC_INJECTED_EVALUATION`):**
  - Injected Samples: 78 (3% rate)
  - True Positives: 42 | False Positives: 34 | False Negatives: 36
  - Precision: **0.5526** | Recall: **0.5385** | F1-Score: **0.5455**

### 4. Drift Monitoring (PSI)
- **Overall Status:** `significant_drift` (Mean PSI = `0.7797`)
- **Stable Features (14):** `hour`, `dayofweek`, `is_weekend`, `lag_1h` through `lag_168h` (PSI < 0.08)
- **Seasonality Drift Detected:** `month_sin` (PSI 5.7065), `month` (PSI 4.3305), `month_cos` (PSI 3.5776) reflect chronological shift from annual train split to autumn test split.

---

## Limitations and Caveats
1. **Single Household Footprint:** The dataset reflects a single European household. Behavior may not generalize to commercial facilities, multi-family complexes, or different climate zones without fine-tuning.
2. **Exogenous Weather Absence:** The dataset does not include ambient outdoor temperature or solar irradiance. Extreme heat waves or cold snaps will manifest as unmodeled residual spikes.
3. **Conformal Marginal Guarantee:** Standard split conformal prediction provides *marginal* coverage across the test set, but conditional coverage may fluctuate during unusual seasonal transitions.
4. **Synthetic Anomaly Evaluation:** The reported anomaly benchmark metrics evaluate synthetic additive perturbations; real-world hardware degradations or cyber-physical anomalies may exhibit different spectral signatures.
