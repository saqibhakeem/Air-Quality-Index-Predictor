"""
load_features.py

Step 1 of the training pipeline: fetch (features, targets) from the
Hopsworks Feature View and split them chronologically into train/test sets.

IMPORTANT: this is a time-series problem, so we deliberately do NOT use a
random train/test split (sklearn's default). A random split would leak
future information into training via the lag/rolling features and give an
unrealistically good test score. Instead the most recent `test_size`
fraction of rows (by timestamp) is held out as the test set.

Usage (as a library):
    from load_features import get_train_test_split
    X_train, X_test, y_train, y_test = get_train_test_split()
"""

import os
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import hopsworks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FEATURE_VIEW_NAME = "aqi_karachi_fv"
FEATURE_VIEW_VERSION = 3  # bumped alongside feature_pipeline/push_to_feature_store.py
TARGET_COLUMNS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
TIMESTAMP_COL = "timestamp_utc"

# Columns that are identifiers/metadata, not model inputs
NON_FEATURE_COLUMNS = [TIMESTAMP_COL, "city", "lat", "lon", "weather_main", "aqi_aqicn"]

HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.environ.get("HOPSWORKS_PROJECT")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOCAL_CACHE_PATH = DATA_DIR / "training_data_cache.csv"


def connect_feature_store():
    if not HOPSWORKS_API_KEY:
        raise RuntimeError("HOPSWORKS_API_KEY is not set")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
    return project.get_feature_store()


def fetch_training_dataframe() -> pd.DataFrame:
    """Pull the full feature+label table from the Hopsworks feature view."""
    fs = connect_feature_store()
    fv = fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)

    X, y = fv.training_data(primary_key=True, event_time=True)
    df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

    # De-dup columns in case primary_key/event_time overlap with X's own columns
    df = df.loc[:, ~df.columns.duplicated()]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOCAL_CACHE_PATH, index=False)
    log.info("Fetched %d rows from feature view, cached to %s", len(df), LOCAL_CACHE_PATH)
    return df


def chronological_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time: oldest (1 - test_size) rows -> train, most recent -> test."""
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    log.info(
        "Chronological split: train=%d rows (%s -> %s), test=%d rows (%s -> %s)",
        len(train_df), train_df[TIMESTAMP_COL].min(), train_df[TIMESTAMP_COL].max(),
        len(test_df), test_df[TIMESTAMP_COL].min(), test_df[TIMESTAMP_COL].max(),
    )

    feature_cols = [
        c for c in df.columns if c not in NON_FEATURE_COLUMNS + TARGET_COLUMNS
    ]

    X_train, y_train = train_df[feature_cols], train_df[TARGET_COLUMNS]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COLUMNS]
    return X_train, X_test, y_train, y_test


def get_train_test_split(test_size: float = 0.2, use_cache: bool = False, max_history_days: int = None):
    """Main entry point used by train_models.py.

    max_history_days: if set, restricts training data to the most recent N
    days before splitting. Useful when older history spans a different
    seasonal/weather regime than the current period - a model trained on
    the full history may generalize worse to "now" than one trained on a
    shorter, more homogeneous recent window. Try this if a model performs
    notably worse than a naive persistence baseline (see check_baseline.py).
    """
    if use_cache and LOCAL_CACHE_PATH.exists():
        log.info("Using cached training data at %s", LOCAL_CACHE_PATH)
        df = pd.read_csv(LOCAL_CACHE_PATH, parse_dates=[TIMESTAMP_COL])
    else:
        df = fetch_training_dataframe()

    # fv.training_data() can return timestamp_utc as a plain string rather
    # than a parsed datetime (unlike the cached-CSV path above, which parses
    # it explicitly). Force it here so downstream Timedelta arithmetic and
    # comparisons work regardless of which path the data came through.
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])

    if max_history_days is not None:
        cutoff = df[TIMESTAMP_COL].max() - pd.Timedelta(days=max_history_days)
        before = len(df)
        df = df[df[TIMESTAMP_COL] >= cutoff]
        log.info(
            "max_history_days=%d: restricted to %d rows (dropped %d older rows, cutoff=%s)",
            max_history_days, len(df), before - len(df), cutoff,
        )

    return chronological_split(df, test_size=test_size)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_split()
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Feature columns: {list(X_train.columns)}")