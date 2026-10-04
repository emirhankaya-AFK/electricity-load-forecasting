"""Data ingestion and preprocessing for UCI Household Electric Power Consumption dataset."""

import logging
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


def download_dataset(
    url: str = (
        "https://archive.ics.uci.edu/static/public/235/"
        "individual+household+electric+power+consumption.zip"
    ),
    dest_dir: str | Path = "data/raw",
    force: bool = False,
) -> Path:
    """Download and extract the UCI household power consumption dataset.

    Args:
        url: Dataset URL.
        dest_dir: Destination directory.
        force: If True, re-download even if files exist.

    Returns:
        Path to the extracted txt data file.
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    extracted_file = dest_path / "household_power_consumption.txt"
    zip_path = dest_path / "household_power_consumption.zip"

    if extracted_file.exists() and not force:
        logger.info("Found existing raw dataset at %s", extracted_file)
        return extracted_file

    logger.info("Downloading dataset from %s ...", url)
    download_success = False

    # Attempt download via curl if available
    curl_bin = shutil.which("curl") or shutil.which("curl.exe")
    if curl_bin:
        try:
            logger.info("Downloading with system curl (%s) ...", curl_bin)
            res = subprocess.run(
                [curl_bin, "-L", "--retry", "3", "-o", str(zip_path), url],
                capture_output=True,
                check=False,
                timeout=300,
            )
            if res.returncode == 0 and zip_path.exists() and zip_path.stat().st_size > 1000:
                download_success = True
        except Exception as curl_err:
            logger.warning("Curl download failed (%s); trying requests...", curl_err)

    if not download_success:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            with requests.get(url, headers=headers, stream=True, timeout=120) as response:
                response.raise_for_status()
                with open(zip_path, "wb") as out_file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            out_file.write(chunk)
            download_success = True
        except Exception as e:
            logger.error("Failed to download dataset: %s", e)
            if zip_path.exists():
                zip_path.unlink()
            raise RuntimeError(f"Could not download dataset from {url}: {e}") from e

    logger.info("Extracting %s to %s ...", zip_path, dest_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_path)

    # Clean up zip file to save disk space
    if zip_path.exists():
        zip_path.unlink()

    if not extracted_file.exists():
        # Check if extracted file has alternate naming
        txt_files = list(dest_path.glob("*.txt"))
        if txt_files:
            return txt_files[0]
        raise FileNotFoundError(f"household_power_consumption.txt not found in {dest_path}")

    return extracted_file


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Load raw semicolon-delimited UCI power consumption data.

    Args:
        filepath: Path to household_power_consumption.txt.

    Returns:
        DataFrame with unified datetime index and numeric columns.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info("Loading raw data from %s ...", path)
    # The UCI dataset has '?' for missing values and ';' delimiter
    df = pd.read_csv(
        path,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )

    # Create timestamp column from Date and Time
    # Expected format: Date dd/mm/yyyy, Time hh:mm:ss
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # Cast numeric columns
    numeric_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def resample_hourly(
    df: pd.DataFrame,
    target_col: str = "Global_active_power",
    subset_hours: int = 17520,
) -> pd.DataFrame:
    """Resample minute-level load data to hourly means and extract contiguous subset.

    Args:
        df: Minute-level DataFrame with DatetimeIndex.
        target_col: Target column name.
        subset_hours: Number of most recent contiguous hours to keep (e.g. 17520 for 2 years).

    Returns:
        Hourly resampled DataFrame with complete datetime index and interpolated short gaps.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # Resample to hourly mean
    hourly = df[[target_col]].resample("1h").mean()

    # Impute small gaps via linear interpolation, then backward/forward fill edges
    hourly[target_col] = hourly[target_col].interpolate(method="time", limit=6)
    hourly[target_col] = hourly[target_col].ffill().bfill()

    # Drop any remaining unfillable rows
    hourly = hourly.dropna(subset=[target_col])

    # Select the practical recent contiguous subset for CPU efficiency
    if subset_hours > 0 and len(hourly) > subset_hours:
        hourly = hourly.iloc[-subset_hours:]

    # Ensure regular hourly frequency
    hourly = hourly.asfreq("1h")
    # Re-fill in case asfreq introduced single missing step
    hourly[target_col] = hourly[target_col].interpolate(method="time").ffill().bfill()

    return hourly


def generate_synthetic_hourly_data(
    n_hours: int = 1000,
    start_date: str = "2023-01-01 00:00:00",
    target_col: str = "Global_active_power",
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate realistic deterministic synthetic hourly electricity load data.

    Useful for offline testing, CI pipelines, and benchmarks without external downloads.

    Args:
        n_hours: Number of hourly timestamps to simulate.
        start_date: Start timestamp.
        target_col: Name of target load column.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with hourly DatetimeIndex and target load values.
    """
    rng = np.random.default_rng(random_state)
    dates = pd.date_range(start=start_date, periods=n_hours, freq="1h")

    hour_of_day = dates.hour.to_numpy()
    day_of_week = dates.dayofweek.to_numpy()

    # Diurnal cycle: peak morning (~8am) and evening (~7pm)
    diurnal = (
        1.0
        + 0.6 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        + 0.3 * np.sin(2 * np.pi * (hour_of_day - 17) / 12)
    )
    # Weekend effect: lower overall load on weekends
    weekend_factor = np.where(day_of_week >= 5, 0.85, 1.0)
    # Seasonal / trend modulation
    time_trend = np.linspace(0, 0.1, n_hours)
    # Gaussian noise
    noise = rng.normal(0, 0.15, size=n_hours)

    load = diurnal * weekend_factor + time_trend + noise
    # Ensure active power is strictly non-negative (household power min ~0.1 kW)
    load = np.clip(load, 0.1, None)

    df = pd.DataFrame({target_col: load}, index=dates)
    df.index.name = "timestamp"
    df = df.asfreq("1h")
    return df
