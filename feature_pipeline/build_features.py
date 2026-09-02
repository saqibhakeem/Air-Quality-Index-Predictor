"""
build_features.py

Step 2 of the feature pipeline: turn the raw-data history into a
model-ready feature table with targets.

Assumes fetch_raw_data.py has been run enough times (or backfilled) to
produce roughly-hourly rows in data/raw_data.csv.

Targets: AQI 24h, 48h and 72h ahead (next 3 days), using aqi_us_epa as the
ground-truth AQI scale. Rows without a known future value (the most recent
~72h of data) are dropped, since we can't yet know their targets.

Usage:
    python build_features.py
Writes: data/features_targets.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_CSV_PATH = DATA_DIR / "raw_data.csv"
FEATURES_CSV_PATH = DATA_DIR / "features_targets.csv"

# Assumed spacing between raw rows (hourly collection cadence)
ROWS_PER_HOUR = 1
HORIZONS_HOURS = [24, 48, 72]  # next-day, +2 days, +3 days


def load_raw(path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp_utc"])
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    df = df.drop_duplicates(subset="timestamp_utc")
    return df


def reindex_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Reindexes to a strict hourly grid before any lag/rolling calculation.

    The lag and rolling features below (aqi_lag_24h, aqi_roll_mean_24h, etc.)
    are computed by counting ROWS, which is only correct if every row really
    is exactly 1 hour apart. A single missed hourly run (API hiccup, rate
    limit, a workflow that didn't fire) silently shifts every later "24h ago"
    lookup to actually mean "23h ago" or less - corrupting exactly the
    features the models rely on most, without raising any error.

    Reindexing to a complete hourly DatetimeIndex makes gaps explicit (as
    NaN rows) instead of invisible. Short gaps (<=3 hours) are linearly
    interpolated as a reasonable approximation; longer gaps are left as NaN
    and get dropped downstream by the required-feature check in
    push_to_feature_store.py.
    """
    df = df.set_index("timestamp_utc")
    # Defensively force a proper tz-aware DatetimeIndex regardless of what
    # load_raw()'s pd.read_csv(parse_dates=...) actually produced - this has
    # been observed to silently yield a plain (non-datetime) Index depending
    # on the pandas version and the exact mix of timestamp string formats
    # accumulated in raw_data.csv over time, which would otherwise crash on
    # the .tz access below with an unhelpful AttributeError.
    df.index = pd.to_datetime(df.index, utc=True, format="mixed")
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="1h", tz=df.index.tz)

    n_missing = len(full_index) - len(df)
    if n_missing > 0:
        log.warning(
            "Found %d missing hourly row(s) in raw_data.csv - reindexing to "
            "a full hourly grid so lag/rolling features stay time-accurate",
            n_missing,
        )

    df = df.reindex(full_index)
    df.index.name = "timestamp_utc"

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=3)

    for col in ["city", "lat", "lon", "weather_main"]:
        if col in df.columns:
            df[col] = df[col].ffill()

    return df.reset_index()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp_utc"]
    df["hour"] = ts.dt.hour
    df["day"] = ts.dt.day
    df["month"] = ts.dt.month
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    aqi = df["aqi_us_epa"]

    # Rate of change vs previous reading (per collection interval)
    df["aqi_change_rate"] = aqi.diff()

    # Lag features - what was AQI 1h / 3h / 24h ago
    df["aqi_lag_1h"] = aqi.shift(1 * ROWS_PER_HOUR)
    df["aqi_lag_3h"] = aqi.shift(3 * ROWS_PER_HOUR)
    df["aqi_lag_24h"] = aqi.shift(24 * ROWS_PER_HOUR)

    # Rolling stats - short and daily trend
    df["aqi_roll_mean_3h"] = aqi.rolling(3 * ROWS_PER_HOUR, min_periods=1).mean()
    df["aqi_roll_mean_24h"] = aqi.rolling(24 * ROWS_PER_HOUR, min_periods=1).mean()
    df["aqi_roll_std_24h"] = aqi.rolling(24 * ROWS_PER_HOUR, min_periods=1).std()

    # Pollutant ratio - a cheap signal for traffic/industrial vs dust-driven pollution
    df["pm_ratio"] = df["pm2_5"] / df["pm10"].replace(0, np.nan)

    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    for h in HORIZONS_HOURS:
        df[f"target_aqi_{h}h"] = df["aqi_us_epa"].shift(-h * ROWS_PER_HOUR)
    return df


def build_feature_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df = reindex_to_hourly(df)

    if df["aqi_us_epa"].isna().all():
        raise ValueError(
            "aqi_us_epa is empty for every row - this is computed from pm2_5/pm10, "
            "so check that fetch_raw_data.py (or backfill_historical.py) is populating "
            "those pollutant columns correctly."
        )

    df = add_time_features(df)
    df = add_derived_features(df)
    df = add_targets(df)

    target_cols = [f"target_aqi_{h}h" for h in HORIZONS_HOURS]
    before = len(df)
    df = df.dropna(subset=target_cols)
    log.info("Dropped %d rows with unknown future targets (most recent ~72h)", before - len(df))

    return df.reset_index(drop=True)


def main() -> None:
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_CSV_PATH} not found - run fetch_raw_data.py first (repeatedly, "
            "or via backfill) to build up raw history."
        )

    raw_df = load_raw()
    log.info("Loaded %d raw rows", len(raw_df))

    feature_df = build_feature_table(raw_df)
    log.info("Built %d feature rows with targets", len(feature_df))

    FEATURES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(FEATURES_CSV_PATH, index=False)
    log.info("Wrote %s", FEATURES_CSV_PATH)


if __name__ == "__main__":
    main()