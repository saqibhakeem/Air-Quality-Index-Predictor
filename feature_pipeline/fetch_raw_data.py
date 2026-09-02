"""
fetch_raw_data.py

Step 1 of the feature pipeline: fetch raw weather + pollutant data
for a target city and append it to a local raw-data CSV.

Data sources:
- OpenWeather "Current Weather" + "Air Pollution" APIs (primary, free tier)
- AQICN "feed" API (secondary AQI reading, used mainly for backfill later)

Usage:
    export OPENWEATHER_API_KEY="..."
    export AQICN_API_KEY="..."          # optional, AQICN token from aqicn.org/data-platform/token
    python fetch_raw_data.py

Each run appends ONE row (a snapshot for "now") to data/raw_data.csv.
Run this on a schedule (e.g. hourly via GitHub Actions) to build history.
"""

import os
import csv
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd

from epa_aqi import compute_overall_aqi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---- Config -----------------------------------------------------------

CITY_NAME = "Karachi"
LAT, LON = 24.8607, 67.0011  # Karachi coordinates

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_CSV_PATH = DATA_DIR / "raw_data.csv"

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
AQICN_API_KEY = os.environ.get("AQICN_API_KEY")  # optional

OPENWEATHER_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHER_POLLUTION_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
AQICN_URL = f"https://api.waqi.info/feed/{CITY_NAME.lower()}/"

RAW_FIELDS = [
    "timestamp_utc",
    "city",
    "lat",
    "lon",
    "temp_c",
    "humidity_pct",
    "wind_speed_ms",
    "pressure_hpa",
    "weather_main",
    "aqi_openweather",  # OpenWeather's own 1-5 categorical AQI (context only)
    "pm2_5",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
    "nh3",
    "aqi_us_epa",  # computed 0-500 EPA AQI from pm2_5/pm10 - THIS is the target scale
    "aqi_aqicn",  # optional cross-check reading from AQICN, live-only, may be null
]


def fetch_weather() -> dict:
    """Fetch current temperature, humidity, wind, pressure from OpenWeather."""
    if not OPENWEATHER_API_KEY:
        raise RuntimeError("OPENWEATHER_API_KEY is not set")

    params = {"lat": LAT, "lon": LON, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    resp = requests.get(OPENWEATHER_WEATHER_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    return {
        "temp_c": data["main"]["temp"],
        "humidity_pct": data["main"]["humidity"],
        "wind_speed_ms": data["wind"]["speed"],
        "pressure_hpa": data["main"]["pressure"],
        "weather_main": data["weather"][0]["main"] if data.get("weather") else None,
    }


def fetch_pollution() -> dict:
    """Fetch pollutant concentrations + OpenWeather's own AQI index."""
    if not OPENWEATHER_API_KEY:
        raise RuntimeError("OPENWEATHER_API_KEY is not set")

    params = {"lat": LAT, "lon": LON, "appid": OPENWEATHER_API_KEY}
    resp = requests.get(OPENWEATHER_POLLUTION_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    item = data["list"][0]
    comp = item["components"]

    return {
        "aqi_openweather": item["main"]["aqi"],
        "pm2_5": comp.get("pm2_5"),
        "pm10": comp.get("pm10"),
        "no2": comp.get("no2"),
        "so2": comp.get("so2"),
        "co": comp.get("co"),
        "o3": comp.get("o3"),
        "nh3": comp.get("nh3"),
    }


def fetch_aqicn() -> dict:
    """Fetch AQICN's US-EPA-style AQI (0-500 scale) as the training target source.

    Returns {"aqi_aqicn": None} if no token is set or the call fails, so the
    pipeline can still run using OpenWeather data alone during early testing.
    """
    if not AQICN_API_KEY:
        log.warning("AQICN_API_KEY not set - skipping AQICN reading")
        return {"aqi_aqicn": None}

    try:
        resp = requests.get(AQICN_URL, params={"token": AQICN_API_KEY}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "ok":
            log.warning("AQICN returned status=%s", data.get("status"))
            return {"aqi_aqicn": None}
        return {"aqi_aqicn": data["data"]["aqi"]}
    except (requests.RequestException, KeyError) as exc:
        log.warning("AQICN fetch failed: %s", exc)
        return {"aqi_aqicn": None}


def build_row() -> dict:
    """Combine all sources into a single raw-data row for right now."""
    row = {
        # Use pandas' own Timestamp string format (space-separated) rather
        # than Python's datetime.isoformat() (which uses "T") - raw_data.csv
        # also gets rows from backfill_historical.py written via pandas
        # to_csv(), so matching that format here avoids a mixed-format column
        # that pd.to_datetime() can only parse with format="mixed".
        "timestamp_utc": str(pd.Timestamp.now(tz="UTC")),
        "city": CITY_NAME,
        "lat": LAT,
        "lon": LON,
    }
    row.update(fetch_weather())
    row.update(fetch_pollution())
    row["aqi_us_epa"] = compute_overall_aqi(row.get("pm2_5"), row.get("pm10"))
    row.update(fetch_aqicn())
    return row


def append_to_csv(row: dict, path: Path = RAW_CSV_PATH) -> None:
    """Append one row to the raw-data CSV, writing the header if new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    log.info("Appended row for %s to %s", row["timestamp_utc"], path)


def main() -> None:
    try:
        row = build_row()
    except Exception:
        log.exception("Failed to fetch raw data")
        sys.exit(1)

    append_to_csv(row)


if __name__ == "__main__":
    main()