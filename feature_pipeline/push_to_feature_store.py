"""
push_to_feature_store.py

Step 3 of the feature pipeline: push the engineered features (+ targets)
into Hopsworks, and set up a Feature View that the training pipeline will
read from later.

Usage:
    export HOPSWORKS_API_KEY="..."       # from your Hopsworks account settings
    export HOPSWORKS_PROJECT="..."       # optional if you only have one project
    python build_features.py             # produces data/features_targets.csv
    python push_to_feature_store.py
"""

import os
import logging
from pathlib import Path

import pandas as pd
import hopsworks



logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEATURES_CSV_PATH = DATA_DIR / "features_targets.csv"

FEATURE_GROUP_NAME = "aqi_karachi_features"
FEATURE_GROUP_VERSION = 3  # v2 locked in bigint for some columns from an all-integer batch; float64 is now enforced on every push, so a fresh version is needed once more

FEATURE_VIEW_NAME = "aqi_karachi_fv"
FEATURE_VIEW_VERSION = 3

TARGET_COLUMNS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]

# Columns that are allowed to be null (optional/cross-check fields) and
# should NOT trigger row-dropping in load_features().
NULLABLE_COLUMNS = ["aqi_aqicn", "weather_main"]

HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.environ.get("HOPSWORKS_PROJECT")  # optional


def load_features(path: Path = FEATURES_CSV_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run build_features.py first")

    df = pd.read_csv(path, parse_dates=["timestamp_utc"])

    # Hopsworks feature names must be lowercase - ours already are, but this
    # keeps the script safe if that ever changes upstream.
    df.columns = [c.lower() for c in df.columns]

    # weather_main is None for every backfilled row (Open-Meteo doesn't provide
    # a weather description). If a push ever contains an all-null weather_main
    # column, Hopsworks' schema inference has nothing to go on and can register
    # the column as `double` instead of `string` - which then breaks every
    # future push once real string values show up from live fetches. Filling
    # with a placeholder guarantees the column is always seen as string.
    df["weather_main"] = df["weather_main"].fillna("unknown")

    # Drop rows where any REQUIRED feature/target is still NaN (the first
    # ~24h of history, before there's enough lookback) - Hopsworks feature
    # groups don't accept NaN in non-nullable numeric columns. Optional
    # fields like aqi_aqicn are excluded from this check since they're
    # expected to be null on most rows.
    required_cols = [c for c in df.columns if c not in NULLABLE_COLUMNS]
    before = len(df)
    df = df.dropna(subset=required_cols)
    log.info("Dropped %d rows with incomplete lag/rolling features", before - len(df))

    # Force every numeric column to float64, deterministically, on every push.
    # Without this, a column that happens to be all-whole-numbers in one batch
    # (e.g. humidity_pct, aqi_openweather, or aqi_us_epa) gets inferred as an
    # integer type by Hopsworks on first creation, then breaks on any later
    # push where the same column legitimately contains fractional values
    # (e.g. after gap-interpolation in build_features.py). Locking every
    # numeric feature to float64 up front removes this whole class of bug.
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].astype("float64")

    return df


def connect_feature_store():
    if not HOPSWORKS_API_KEY:
        raise RuntimeError("HOPSWORKS_API_KEY is not set")

    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
    log.info("Connected to Hopsworks project: %s", project.name)
    return project.get_feature_store()


def get_or_create_feature_group(fs):
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly Karachi AQI features + 24h/48h/72h-ahead targets",
        primary_key=["timestamp_utc"],
        event_time="timestamp_utc",
        online_enabled=True,  # so the Streamlit app can pull the latest row fast
    )
    return fg


def get_or_create_feature_view(fs, fg):
    query = fg.select_all()
    fv = fs.get_or_create_feature_view(
        name=FEATURE_VIEW_NAME,
        version=FEATURE_VIEW_VERSION,
        description="Karachi AQI features with 24h/48h/72h targets as labels",
        query=query,
        labels=TARGET_COLUMNS,
    )
    return fv


def main():
    df = load_features()
    log.info("Loaded %d feature rows to push", len(df))

    if df.empty:
        log.warning(
            "No feature rows ready yet (need ~72h of raw history before "
            "targets exist) - skipping this run, nothing to push."
        )
        return

    fs = connect_feature_store()
    fg = get_or_create_feature_group(fs)

    # storage="offline" only: nothing in this project reads from the online
    # store's low-latency API (both the training pipeline and the web app
    # read via fg.select_all().read(), an offline batch read), so there's no
    # reason to also write through the online Kafka path - which is the part
    # that was hitting a persistent TOPIC_AUTHORIZATION_FAILED. This can be
    # revisited later if true online serving is ever needed.
    job, validation_report = fg.insert(df, wait=True, storage="offline")
    log.info("Insert complete for feature group '%s' v%d", FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION)

    fv = get_or_create_feature_view(fs, fg)
    log.info("Feature view '%s' v%d ready", FEATURE_VIEW_NAME, FEATURE_VIEW_VERSION)

    # Read-back check: confirm what we just wrote is actually queryable
    readback = fg.select_all().read()
    log.info("Read-back check: feature group now has %d rows in Hopsworks", len(readback))
    print(readback.sort_values("timestamp_utc").tail(3).to_string())


if __name__ == "__main__":
    main()